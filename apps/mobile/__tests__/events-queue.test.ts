import type { QueuedEvent } from '@/events/batch';
import {
  EventQueue,
  FLUSH_AT_EVENTS,
  IMPRESSION_FLUSH_MS,
  type SendOutcome,
  type Transport,
} from '@/events/queue';
import { memoryKV } from '@/storage/kv';

const A = '00000000-0000-4000-8000-0000000000aa';
const KEY = 'events:u1';

/** A server that remembers ids, like the API's idempotent insert (ADR 0016). */
function fakeServer() {
  const stored = new Set<string>();
  const calls: { events: QueuedEvent[]; immediate: boolean }[] = [];
  let script: (SendOutcome | 'lose-response')[] = [];
  const transport: Transport = {
    async send(events, immediate = false) {
      calls.push({ events, immediate });
      const next = script.shift();
      if (next && next !== 'lose-response') return next;
      const accepted: string[] = [];
      const duplicate: string[] = [];
      for (const e of events) (stored.has(e.event_id) ? duplicate : accepted).push(e.event_id);
      events.forEach((e) => stored.add(e.event_id));
      if (next === 'lose-response') throw new Error('network'); // stored, but no answer
      return { kind: 'done', ids: [...accepted, ...duplicate], accepted, duplicate } as SendOutcome;
    },
  };
  return {
    transport,
    calls,
    stored,
    then: (...outcomes: (SendOutcome | 'lose-response')[]) => {
      script = outcomes;
    },
  };
}

let ids = 0;
function setup(opts: { online?: () => boolean; now?: () => number } = {}) {
  const server = fakeServer();
  const store = memoryKV();
  const queue = new EventQueue({
    transport: server.transport,
    store,
    storageKey: KEY,
    newId: () => `00000000-0000-4000-8000-${String(++ids).padStart(12, '0')}`,
    random: () => 0.5,
    isOnline: opts.online,
    now: opts.now,
  });
  return { server, store, queue };
}

const flushPromises = async () => {
  for (let i = 0; i < 10; i++) await Promise.resolve();
};
const impression = (position = 0) =>
  ({ type: 'impression', article_id: A, position, request_id: 'r1' }) as const;

beforeEach(() => jest.useFakeTimers({ now: new Date('2026-10-24T12:00:00Z') }));
afterEach(() => jest.useRealTimers());

describe('flush triggers', () => {
  it('a click goes out at once, carrying the pending impressions', async () => {
    const { server, queue } = setup();
    queue.enqueue(impression(0));
    queue.enqueue(impression(1));
    await flushPromises();
    expect(server.calls).toHaveLength(0); // impressions alone wait
    queue.enqueue({ type: 'click', article_id: A, position: 1, request_id: 'r1' });
    await flushPromises();
    expect(server.calls).toHaveLength(1);
    expect(server.calls[0].events.map((e) => e.type)).toEqual([
      'impression',
      'impression',
      'click',
    ]);
    expect(server.calls[0].immediate).toBe(false);
    expect(queue.pending).toEqual([]);
  });

  it('impressions alone go 60 s after the oldest one', async () => {
    const { server, queue } = setup();
    queue.enqueue(impression(0));
    jest.advanceTimersByTime(30_000);
    queue.enqueue(impression(1));
    jest.advanceTimersByTime(IMPRESSION_FLUSH_MS - 30_000 - 1);
    await flushPromises();
    expect(server.calls).toHaveLength(0);
    jest.advanceTimersByTime(1);
    await flushPromises();
    expect(server.calls).toHaveLength(1);
    expect(server.calls[0].events).toHaveLength(2);
  });

  it('100 pending events go at once', async () => {
    const { server, queue } = setup();
    for (let i = 0; i < FLUSH_AT_EVENTS - 1; i++) queue.enqueue(impression(i));
    await flushPromises();
    expect(server.calls).toHaveLength(0);
    queue.enqueue(impression(99));
    await flushPromises();
    expect(server.calls).toHaveLength(1);
    expect(server.calls[0].events).toHaveLength(100);
  });

  it('likes and hides flush like clicks', async () => {
    const { server, queue } = setup();
    queue.enqueue({ type: 'like', article_id: A });
    await flushPromises();
    queue.enqueue({ type: 'hide', article_id: A, scope: 'source' });
    await flushPromises();
    expect(server.calls.map((c) => c.events[0].type)).toEqual(['like', 'hide']);
  });
});

describe('retries', () => {
  it('a retried batch keeps its ids, so the server reports duplicates only', async () => {
    const { server, queue } = setup();
    server.then('lose-response');
    queue.enqueue({ type: 'click', article_id: A });
    await flushPromises();
    expect(server.stored.size).toBe(1);
    expect(queue.pending).toHaveLength(1); // no answer: still queued
    jest.advanceTimersByTime(2000); // backoff (attempt 0: 0.5–1 s)
    await flushPromises();
    expect(server.calls).toHaveLength(2);
    expect(server.calls[1].events[0].event_id).toBe(server.calls[0].events[0].event_id);
    expect(queue.pending).toEqual([]);
  });

  it('honors Retry-After on 429/503', async () => {
    const { server, queue } = setup();
    server.then({ kind: 'retry', retryAfterS: 7 });
    queue.enqueue({ type: 'click', article_id: A });
    await flushPromises();
    jest.advanceTimersByTime(6999);
    await flushPromises();
    expect(server.calls).toHaveLength(1);
    // A new click during the backoff waits for it too.
    queue.enqueue({ type: 'like', article_id: A });
    await flushPromises();
    expect(server.calls).toHaveLength(1);
    jest.advanceTimersByTime(1);
    await flushPromises();
    expect(server.calls).toHaveLength(2);
    expect(server.calls[1].events).toHaveLength(2);
  });

  it('backs off exponentially without Retry-After', async () => {
    const { server, queue } = setup();
    const retry: SendOutcome = { kind: 'retry', retryAfterS: null };
    server.then(retry, retry, retry);
    queue.enqueue({ type: 'click', article_id: A });
    await flushPromises();
    // random() = 0.5 → 75% of 1 s, 2 s, 4 s
    for (const wait of [750, 1500, 3000]) {
      jest.advanceTimersByTime(wait - 1);
      await flushPromises();
      const before = server.calls.length;
      jest.advanceTimersByTime(1);
      await flushPromises();
      expect(server.calls.length).toBe(before + 1);
    }
    expect(queue.pending).toEqual([]);
  });

  it('drops and counts a batch the server refuses (422, 413)', async () => {
    const { server, queue } = setup();
    const warn = jest.spyOn(console, 'warn').mockImplementation(() => undefined);
    server.then({ kind: 'drop', status: 422 });
    queue.enqueue({ type: 'click', article_id: A });
    await flushPromises();
    expect(queue.pending).toEqual([]);
    expect(queue.dropped).toBe(1);
    jest.advanceTimersByTime(60_000);
    await flushPromises();
    expect(server.calls).toHaveLength(1); // never retried
    warn.mockRestore();
  });

  it('drops events older than 23 h before sending', async () => {
    let now = Date.parse('2026-10-24T12:00:00Z');
    const { server, queue } = setup({ now: () => now });
    queue.enqueue(impression(0));
    now += 24 * 3600 * 1000;
    queue.enqueue({ type: 'click', article_id: A });
    await flushPromises();
    expect(server.calls[0].events.map((e) => e.type)).toEqual(['click']);
    expect(queue.dropped).toBe(1);
  });

  it('waits while offline', async () => {
    let online = false;
    const { server, queue } = setup({ online: () => online });
    queue.enqueue({ type: 'click', article_id: A });
    await flushPromises();
    expect(server.calls).toHaveLength(0);
    online = true;
    await queue.flush();
    expect(server.calls).toHaveLength(1);
  });
});

describe('persistence', () => {
  it('stores pending events and re-sends them on the next launch', async () => {
    const first = setup({ online: () => false });
    first.queue.enqueue({ type: 'click', article_id: A });
    const saved = await first.store.get(KEY);
    expect(JSON.parse(saved!)).toHaveLength(1);

    const second = setup();
    await second.store.set(KEY, saved!);
    await second.queue.load();
    await flushPromises();
    expect(second.server.calls[0].events[0].event_id).toBe(JSON.parse(saved!)[0].event_id);
    expect(JSON.parse((await second.store.get(KEY))!)).toEqual([]);
  });

  it('survives unreadable storage', async () => {
    const { store, queue } = setup();
    await store.set(KEY, '{not json');
    await expect(queue.load()).resolves.toBeUndefined();
    expect(queue.pending).toEqual([]);
  });
});

describe('page hidden', () => {
  it('commits pending hides first and sends one keepalive batch', async () => {
    const { server, queue } = setup();
    queue.enqueue(impression(0));
    queue.pageHidden(() => queue.enqueue({ type: 'hide', article_id: A, scope: 'article' }));
    await flushPromises();
    expect(server.calls).toHaveLength(1);
    expect(server.calls[0].immediate).toBe(true);
    expect(server.calls[0].events.map((e) => e.type)).toEqual(['hide', 'impression']);
  });

  it('a repeated hidden signal (visibilitychange, then pagehide) sends nothing twice', async () => {
    const { server } = setup();
    let release: () => void = () => undefined;
    const slow: Transport = {
      send: (events, immediate) =>
        new Promise((resolve) => {
          release = () => resolve({ kind: 'done', ids: events.map((e) => e.event_id) });
          server.calls.push({ events, immediate: !!immediate });
        }),
    };
    const q = new EventQueue({
      transport: slow,
      store: memoryKV(),
      storageKey: KEY,
      newId: () => 'e1',
    });
    q.enqueue(impression(0));
    q.pageHidden();
    q.pageHidden();
    expect(server.calls).toHaveLength(1);
    release();
    await flushPromises();
    expect(q.pending).toEqual([]);
  });

  it('while hidden, a new event goes straight out with keepalive', async () => {
    const { server, queue } = setup();
    queue.pageHidden();
    queue.enqueue({ type: 'click', article_id: A });
    await flushPromises();
    expect(server.calls).toEqual([expect.objectContaining({ immediate: true })]);
  });
});
