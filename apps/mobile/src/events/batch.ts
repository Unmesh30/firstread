// Pure helpers for the event queue: event creation, batching under the API's limits,
// stale-event drop and the priority order of a keepalive flush.
import type { EventIn } from '@/api/schema-types';

/** The API takes ≤ 100 events per batch (EventsBatch) in ≤ 64 KiB (bodylimit.py). */
export const MAX_BATCH_EVENTS = 100;
/** Under fetch keepalive's 64 KiB body quota (MDN), with room for headers. */
export const MAX_BATCH_BYTES = 60_000;
/** The API rejects a client_ts more than 24 h off; drop ours an hour before that. */
export const MAX_EVENT_AGE_MS = 23 * 60 * 60 * 1000;

export type QueuedEvent = EventIn & { event_id: string; client_ts: string };

const encoder = new TextEncoder();
export const byteLength = (s: string) => encoder.encode(s).length;

/** A new event: its id and timestamp are fixed now, so every retry is deduplicated. */
export function makeEvent(
  fields: Omit<EventIn, 'event_id' | 'client_ts'>,
  newId: () => string,
  now: number = Date.now(),
): QueuedEvent {
  return { ...fields, event_id: newId(), client_ts: new Date(now).toISOString() };
}

export function isStale(e: QueuedEvent, now: number = Date.now()): boolean {
  const at = Date.parse(e.client_ts);
  return Number.isNaN(at) || now - at > MAX_EVENT_AGE_MS;
}

/** The serialized size of `{"events":[...]}` holding these events. */
export function batchBytes(events: QueuedEvent[]): number {
  return byteLength(JSON.stringify({ events }));
}

/** Split into batches of ≤ maxEvents and ≤ maxBytes, keeping the order. */
export function toBatches(
  events: QueuedEvent[],
  maxEvents: number = MAX_BATCH_EVENTS,
  maxBytes: number = MAX_BATCH_BYTES,
): QueuedEvent[][] {
  const out: QueuedEvent[][] = [];
  const envelope = byteLength('{"events":[]}');
  let current: QueuedEvent[] = [];
  let size = envelope;
  for (const e of events) {
    const bytes = byteLength(JSON.stringify(e));
    const comma = current.length ? 1 : 0;
    if (current.length && (current.length >= maxEvents || size + comma + bytes > maxBytes)) {
      out.push(current);
      current = [];
      size = envelope;
    }
    size += (current.length ? 1 : 0) + bytes;
    current.push(e);
  }
  if (current.length) out.push(current);
  return out;
}

const RANK: Record<EventIn['type'], number> = {
  hide: 0,
  click: 1,
  like: 1,
  dwell: 2,
  impression: 3,
};

/**
 * The one batch a closing page can send (fetch keepalive): hides first, then clicks and
 * likes, then the newest impressions, up to the batch limits. The rest stays persisted.
 */
export function keepaliveBatch(
  events: QueuedEvent[],
  maxEvents: number = MAX_BATCH_EVENTS,
  maxBytes: number = MAX_BATCH_BYTES,
): QueuedEvent[] {
  const ordered = [...events].sort(
    (a, b) => RANK[a.type] - RANK[b.type] || b.client_ts.localeCompare(a.client_ts),
  );
  return toBatches(ordered, maxEvents, maxBytes)[0] ?? [];
}
