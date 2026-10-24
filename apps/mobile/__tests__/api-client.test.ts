import { authFetch, call, createApi } from '@/api/client';
import { ApiError } from '@/api/errors';

const json = (status: number, body: unknown, headers: Record<string, string> = {}) =>
  new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json', ...headers },
  });

function fakeAuth(tokens: string[], refreshOk = true) {
  let i = 0;
  return {
    refreshes: 0,
    async headers() {
      return { Authorization: `Bearer ${tokens[i]}` };
    },
    async refresh() {
      this.refreshes += 1;
      if (refreshOk) i += 1;
      return refreshOk;
    },
  };
}

describe('authFetch', () => {
  it('adds the auth headers', async () => {
    const seen: string[] = [];
    const f = authFetch(fakeAuth(['a']), async (r) => {
      seen.push(r.headers.get('Authorization') ?? '');
      return json(200, {});
    });
    await f(new Request('http://api.test/v1/me'));
    expect(seen).toEqual(['Bearer a']);
  });

  it('refreshes once after a 401 and retries with the new token and the same body', async () => {
    const seen: [string, string][] = [];
    const auth = fakeAuth(['old', 'new']);
    const f = authFetch(auth, async (r) => {
      seen.push([r.headers.get('Authorization') ?? '', await r.text()]);
      return seen.length === 1 ? json(401, {}) : json(200, { ok: true });
    });
    const out = await f(
      new Request('http://api.test/v1/events', { method: 'POST', body: '{"events":[]}' }),
    );
    expect(out.status).toBe(200);
    expect(seen).toEqual([
      ['Bearer old', '{"events":[]}'],
      ['Bearer new', '{"events":[]}'],
    ]);
    expect(auth.refreshes).toBe(1);
  });

  it('returns the 401 when the refresh fails, and never loops', async () => {
    const auth = fakeAuth(['x'], false);
    let calls = 0;
    const f = authFetch(auth, async () => {
      calls += 1;
      return json(401, {});
    });
    expect((await f(new Request('http://api.test/v1/me'))).status).toBe(401);
    expect(calls).toBe(1);
  });
});

describe('call', () => {
  it('returns data, or throws an ApiError with the server details', async () => {
    const api = createApi('http://api.test', fakeAuth(['t']), async (r) =>
      new URL(r.url).pathname === '/v1/topics'
        ? json(200, { topics: [{ id: 'us', name: 'U.S.' }], request_id: 'r', model_version: 'm' })
        : json(400, { error: { code: 'cursor_expired', message: 'expired' }, request_id: 'r2' }),
    );
    expect((await call(api.GET('/v1/topics'))).topics[0].id).toBe('us');
    await expect(
      call(api.GET('/v1/feed', { params: { query: { cursor: 'c' } } })),
    ).rejects.toMatchObject({
      status: 400,
      code: 'cursor_expired',
      requestId: 'r2',
    });
  });

  it('turns a failed fetch into a network ApiError', async () => {
    const api = createApi('http://api.test', fakeAuth(['t']), async () => {
      throw new TypeError('Failed to fetch');
    });
    const e = await call(api.GET('/v1/topics')).catch((x: unknown) => x);
    expect(e).toBeInstanceOf(ApiError);
    expect((e as ApiError).code).toBe('network');
  });
});
