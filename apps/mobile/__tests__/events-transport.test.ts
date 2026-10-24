import { httpTransport, outcomeOf } from '@/events/transport';

const json = (body: unknown, status = 200, headers: Record<string, string> = {}) =>
  new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json', ...headers },
  });
const E = {
  event_id: '00000000-0000-4000-8000-000000000001',
  type: 'click' as const,
  article_id: '00000000-0000-4000-8000-0000000000aa',
  client_ts: '2026-10-24T12:00:00.000Z',
};

describe('outcomeOf', () => {
  it('answers for accepted, duplicate and rejected ids', async () => {
    const o = await outcomeOf(
      json({
        request_id: 'r',
        model_version: 'm',
        accepted: ['a'],
        duplicate: ['b'],
        rejected: [{ event_id: 'c', code: 'unknown_article' }],
      }),
    );
    expect(o).toEqual({ kind: 'done', ids: ['a', 'b', 'c'] });
  });

  it('retries 429/503/5xx/401 with Retry-After, drops 422 and 413', async () => {
    const err = { error: { code: 'x', message: 'x' }, request_id: 'r' };
    expect(await outcomeOf(json(err, 503, { 'Retry-After': '9' }))).toEqual({
      kind: 'retry',
      retryAfterS: 9,
    });
    expect(await outcomeOf(json(err, 429))).toEqual({ kind: 'retry', retryAfterS: null });
    expect(await outcomeOf(json(err, 401))).toEqual({ kind: 'retry', retryAfterS: null });
    expect(await outcomeOf(new Response('oops', { status: 502 }))).toEqual({
      kind: 'retry',
      retryAfterS: null,
    });
    expect(await outcomeOf(json(err, 422))).toEqual({ kind: 'drop', status: 422 });
    expect(await outcomeOf(json(err, 413))).toEqual({ kind: 'drop', status: 413 });
  });
});

describe('httpTransport', () => {
  const ok = () =>
    json({
      request_id: 'r',
      model_version: 'm',
      accepted: [E.event_id],
      duplicate: [],
      rejected: [],
    });

  it('posts with keepalive and the auth headers, and remembers them for a closing page', async () => {
    const seen: Request[] = [];
    const t = httpTransport(
      'http://api.test',
      { headers: async () => ({ 'X-Dev-User': 'u1' }), refresh: async () => false },
      async (r) => {
        seen.push(r);
        return ok();
      },
    );
    expect(await t.send([E])).toEqual({ kind: 'done', ids: [E.event_id] });
    expect(seen[0].url).toBe('http://api.test/v1/events');
    expect(seen[0].method).toBe('POST');
    expect(seen[0].keepalive).toBe(true);
    expect(seen[0].headers.get('X-Dev-User')).toBe('u1');
    expect(JSON.parse(await seen[0].text())).toEqual({ events: [E] });

    // immediate: fetch is called synchronously, with the remembered headers.
    const p = t.send([E], true);
    expect(seen).toHaveLength(2);
    expect(seen[1].headers.get('X-Dev-User')).toBe('u1');
    await p;
  });

  it('refreshes once after a 401', async () => {
    let calls = 0;
    const refresh = jest.fn(async () => true);
    const t = httpTransport('http://api.test', { headers: async () => ({}), refresh }, async () =>
      ++calls === 1 ? json({}, 401) : ok(),
    );
    expect(await t.send([E])).toEqual({ kind: 'done', ids: [E.event_id] });
    expect(refresh).toHaveBeenCalledTimes(1);
  });

  it('treats a network failure as retryable', async () => {
    const t = httpTransport(
      'http://api.test',
      { headers: async () => ({}), refresh: async () => false },
      async () => {
        throw new TypeError('Failed to fetch');
      },
    );
    expect(await t.send([E])).toEqual({ kind: 'retry', retryAfterS: null });
  });
});
