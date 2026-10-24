import {
  batchBytes,
  isStale,
  keepaliveBatch,
  makeEvent,
  MAX_BATCH_BYTES,
  MAX_EVENT_AGE_MS,
  toBatches,
  type QueuedEvent,
} from '@/events/batch';

const A = '00000000-0000-4000-8000-000000000001';
let n = 0;
const id = () => `00000000-0000-4000-8000-${String(++n).padStart(12, '0')}`;
const ev = (type: QueuedEvent['type'], at: number, extra: Partial<QueuedEvent> = {}) =>
  ({ ...makeEvent({ type, article_id: A }, id, at), ...extra }) as QueuedEvent;

describe('makeEvent', () => {
  it('fixes the id and the timestamp when the event is created', () => {
    const e = makeEvent(
      { type: 'click', article_id: A, position: 3, request_id: 'r1' },
      () => 'x',
      0,
    );
    expect(e).toEqual({
      type: 'click',
      article_id: A,
      position: 3,
      request_id: 'r1',
      event_id: 'x',
      client_ts: '1970-01-01T00:00:00.000Z',
    });
  });
});

describe('isStale', () => {
  it('drops events older than 23 h (the API refuses 24 h)', () => {
    const e = ev('impression', 0);
    expect(isStale(e, MAX_EVENT_AGE_MS)).toBe(false);
    expect(isStale(e, MAX_EVENT_AGE_MS + 1)).toBe(true);
    expect(isStale({ ...e, client_ts: 'garbage' }, 0)).toBe(true);
  });
});

describe('toBatches', () => {
  it('splits at 100 events', () => {
    const events = Array.from({ length: 250 }, (_, i) => ev('impression', i));
    expect(toBatches(events).map((b) => b.length)).toEqual([100, 100, 50]);
  });

  it('splits at 60 KB serialized, measured in UTF-8 bytes', () => {
    // A request id of 64 multi-byte characters makes each event ~400 bytes.
    const big = 'é'.repeat(64);
    const events = Array.from({ length: 100 }, (_, i) =>
      ev('impression', i, { request_id: big, position: i }),
    );
    const batches = toBatches(events, 100, 10_000);
    expect(batches.length).toBeGreaterThan(1);
    for (const b of batches) expect(batchBytes(b)).toBeLessThanOrEqual(10_000);
    expect(batches.flat()).toEqual(events); // order kept, nothing lost
  });

  it('keeps the default batches under the keepalive quota', () => {
    const events = Array.from({ length: 100 }, (_, i) =>
      ev('impression', i, { request_id: 'r'.repeat(64) }),
    );
    for (const b of toBatches(events)) expect(batchBytes(b)).toBeLessThanOrEqual(MAX_BATCH_BYTES);
  });
});

describe('keepaliveBatch', () => {
  it('sends hides first, then clicks and likes, then the newest impressions', () => {
    const old = ev('impression', 1000);
    const newer = ev('impression', 3000);
    const click = ev('click', 500);
    const hide = ev('hide', 2000);
    const like = ev('like', 2500);
    const batch = keepaliveBatch([old, click, hide, newer, like]);
    expect(batch.map((e) => e.type)).toEqual(['hide', 'like', 'click', 'impression', 'impression']);
    expect(batch[3]).toBe(newer);
  });

  it('fits one batch and leaves the rest', () => {
    const events = [...Array.from({ length: 150 }, (_, i) => ev('impression', i)), ev('hide', 0)];
    const batch = keepaliveBatch(events);
    expect(batch).toHaveLength(100);
    expect(batch[0].type).toBe('hide');
  });
});
