// The event queue: at-least-once delivery of reader events to POST /v1/events.
//
// Every event gets its id when it is created and keeps it through retries, reloads and
// keepalive sends, so the server deduplicates whatever arrives twice (ADR 0016). Events
// stay in the queue (and in storage) until the server has answered for them.
//
// Flushes: a click, like or hide → now (carrying pending impressions); 60 s after the
// oldest pending impression; 100 pending events; and the page or app going to the
// background → one keepalive batch, priority-ordered (see batch.ts).
import { backoffMs } from '@/api/queries';
import type { KV } from '@/storage/kv';

import { isStale, keepaliveBatch, makeEvent, toBatches, type QueuedEvent } from './batch';

export const IMPRESSION_FLUSH_MS = 60_000;
export const FLUSH_AT_EVENTS = 100;
/** Storage is a safety net, not an archive: beyond this the oldest impressions go. */
export const MAX_QUEUED = 1000;

export type SendOutcome =
  /** The server answered for these ids (accepted, duplicate or rejected). */
  | { kind: 'done'; ids: string[] }
  /** Try again later: rate limited, overloaded, offline, or a session problem. */
  | { kind: 'retry'; retryAfterS: number | null }
  /** The server refused the batch itself (422, 413): a client bug, never retried. */
  | { kind: 'drop'; status: number };

export interface Transport {
  /**
   * Send one batch. `immediate`: the page is closing, so the request must start
   * synchronously (no await before fetch) and survive the page (keepalive).
   */
  send(events: QueuedEvent[], immediate?: boolean): Promise<SendOutcome>;
}

export interface QueueDeps {
  transport: Transport;
  store: KV;
  storageKey: string;
  newId: () => string;
  now?: () => number;
  random?: () => number;
  isOnline?: () => boolean;
}

export type NewEvent = Parameters<typeof makeEvent>[0];

export class EventQueue {
  private events: QueuedEvent[] = [];
  private readonly inFlight = new Set<string>();
  private flushing: Promise<void> | null = null;
  private again = false;
  private impressionTimer: ReturnType<typeof setTimeout> | null = null;
  private retryTimer: ReturnType<typeof setTimeout> | null = null;
  private attempt = 0;
  private hidden = false;
  private collecting = false;
  private stopped = false;
  /** Events the server refused as a batch (422/413), and stale events never sent. */
  dropped = 0;

  private readonly now: () => number;
  private readonly random: () => number;
  private readonly isOnline: () => boolean;

  constructor(private readonly deps: QueueDeps) {
    this.now = deps.now ?? Date.now;
    this.random = deps.random ?? Math.random;
    this.isOnline = deps.isOnline ?? (() => true);
  }

  /** Events not yet answered for (a copy). */
  get pending(): QueuedEvent[] {
    return [...this.events];
  }

  /** Restore what a previous session left behind and send it. */
  async load(): Promise<void> {
    let saved: QueuedEvent[] = [];
    try {
      const raw = await this.deps.store.get(this.deps.storageKey);
      const parsed: unknown = raw ? JSON.parse(raw) : [];
      if (Array.isArray(parsed)) saved = parsed as QueuedEvent[];
    } catch {
      saved = []; // unreadable storage is not worth failing the app for
    }
    const known = new Set(this.events.map((e) => e.event_id));
    this.events = [...saved.filter((e) => e && !known.has(e.event_id)), ...this.events];
    this.dropStale();
    this.persist();
    if (this.events.length) void this.flush();
  }

  enqueue(fields: NewEvent): QueuedEvent {
    const event = makeEvent(fields, this.deps.newId, this.now());
    this.events.push(event);
    this.trim();
    this.persist();
    if (this.collecting) return event; // pageHidden() sends it in its one batch
    if (this.hidden) this.sendKeepalive();
    else if (event.type !== 'impression' || this.events.length >= FLUSH_AT_EVENTS) {
      void this.flush();
    } else this.armImpressionTimer();
    return event;
  }

  /** Send everything that isn't already on its way, in batches, one at a time. */
  flush(): Promise<void> {
    if (this.flushing) {
      this.again = true;
      return this.flushing;
    }
    this.flushing = this.run().finally(() => {
      this.flushing = null;
      if (this.again) {
        this.again = false;
        void this.flush();
      }
    });
    return this.flushing;
  }

  /**
   * The page or app is going to the background. `commit` runs first (it commits hides
   * still inside their undo window); their events go out in the same keepalive batch.
   */
  pageHidden(commit?: () => void): void {
    this.hidden = true;
    this.clearImpressionTimer();
    this.collecting = true;
    try {
      commit?.();
    } finally {
      this.collecting = false;
    }
    this.sendKeepalive();
  }

  pageVisible(): void {
    this.hidden = false;
    if (this.events.some((e) => !this.inFlight.has(e.event_id))) this.armImpressionTimer();
  }

  /** Forget every pending event (Delete my data: nothing left to re-create the data). */
  clear(): void {
    this.events = [];
    this.clearImpressionTimer();
    if (this.retryTimer) clearTimeout(this.retryTimer);
    this.retryTimer = null;
    this.attempt = 0;
    this.persist();
  }

  /** (Re)start after stop(): the provider's effect can run twice. */
  start(): void {
    this.stopped = false;
  }

  stop(): void {
    this.stopped = true;
    this.clearImpressionTimer();
    if (this.retryTimer) clearTimeout(this.retryTimer);
    this.retryTimer = null;
  }

  private async run(): Promise<void> {
    this.clearImpressionTimer();
    if (this.stopped || this.retryTimer || !this.isOnline()) return;
    this.dropStale();
    for (const batch of toBatches(this.unsent())) {
      const outcome = await this.sendBatch(batch, false);
      if (outcome.kind === 'retry') {
        this.scheduleRetry(outcome.retryAfterS);
        return;
      }
    }
    this.attempt = 0;
    if (this.events.some((e) => e.type === 'impression' && !this.inFlight.has(e.event_id))) {
      this.armImpressionTimer();
    }
  }

  private sendKeepalive(): void {
    if (this.stopped) return;
    this.dropStale();
    const batch = keepaliveBatch(this.unsent());
    if (batch.length) void this.sendBatch(batch, true);
  }

  private async sendBatch(batch: QueuedEvent[], immediate: boolean): Promise<SendOutcome> {
    const ids = batch.map((e) => e.event_id);
    ids.forEach((id) => this.inFlight.add(id));
    let outcome: SendOutcome;
    try {
      outcome = await this.deps.transport.send(batch, immediate);
    } catch {
      outcome = { kind: 'retry', retryAfterS: null };
    } finally {
      ids.forEach((id) => this.inFlight.delete(id));
    }
    if (outcome.kind === 'done') this.remove(new Set(outcome.ids));
    else if (outcome.kind === 'drop') {
      this.dropped += batch.length;
      console.warn(`events: dropped a batch of ${batch.length} (HTTP ${outcome.status})`);
      this.remove(new Set(ids));
    }
    return outcome;
  }

  private scheduleRetry(retryAfterS: number | null): void {
    if (this.stopped || this.retryTimer) return;
    const delay = backoffMs(this.attempt++, retryAfterS, this.random);
    this.retryTimer = setTimeout(() => {
      this.retryTimer = null;
      void this.flush();
    }, delay);
  }

  private armImpressionTimer(): void {
    if (this.impressionTimer || this.hidden || this.stopped) return;
    const oldest = this.events.find(
      (e) => e.type === 'impression' && !this.inFlight.has(e.event_id),
    );
    if (!oldest) return;
    const due = Date.parse(oldest.client_ts) + IMPRESSION_FLUSH_MS - this.now();
    this.impressionTimer = setTimeout(
      () => {
        this.impressionTimer = null;
        void this.flush();
      },
      Math.max(0, due),
    );
  }

  private clearImpressionTimer(): void {
    if (this.impressionTimer) clearTimeout(this.impressionTimer);
    this.impressionTimer = null;
  }

  private unsent(): QueuedEvent[] {
    return this.events.filter((e) => !this.inFlight.has(e.event_id));
  }

  private remove(ids: Set<string>): void {
    if (!ids.size) return;
    this.events = this.events.filter((e) => !ids.has(e.event_id));
    this.persist();
  }

  private dropStale(): void {
    const now = this.now();
    const fresh = this.events.filter((e) => this.inFlight.has(e.event_id) || !isStale(e, now));
    if (fresh.length !== this.events.length) {
      this.dropped += this.events.length - fresh.length;
      this.events = fresh;
      this.persist();
    }
  }

  private trim(): void {
    let excess = this.events.length - MAX_QUEUED;
    if (excess <= 0) return;
    this.events = this.events.filter((e) => {
      if (excess > 0 && e.type === 'impression' && !this.inFlight.has(e.event_id)) {
        excess--;
        return false;
      }
      return true;
    });
  }

  private persist(): void {
    // KV writes on web are synchronous inside (localStorage), so a closing page keeps
    // what it queued.
    void this.deps.store.set(this.deps.storageKey, JSON.stringify(this.events));
  }
}
