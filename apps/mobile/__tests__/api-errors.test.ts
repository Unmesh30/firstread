import { ApiError, apiErrorFrom, isApiError, networkError, parseRetryAfter } from '@/api/errors';

const res = (status: number, headers: Record<string, string> = {}) =>
  new Response(null, { status, headers });

describe('parseRetryAfter', () => {
  it('reads delta-seconds and HTTP dates', () => {
    expect(parseRetryAfter('5')).toBe(5);
    const now = Date.parse('2026-10-24T00:00:00Z');
    expect(parseRetryAfter('Sat, 24 Oct 2026 00:00:30 GMT', now)).toBe(30);
    expect(parseRetryAfter('Sat, 24 Oct 2026 00:00:00 GMT', now + 10_000)).toBe(0);
  });
  it('ignores missing or malformed values', () => {
    expect(parseRetryAfter(null)).toBeNull();
    expect(parseRetryAfter('')).toBeNull();
    expect(parseRetryAfter('soon')).toBeNull();
  });
});

describe('apiErrorFrom', () => {
  it('maps the shared error body', () => {
    const e = apiErrorFrom(res(409, { 'Retry-After': '3' }), {
      error: { code: 'history_exists', message: 'has history', details: [{ a: 1 }] },
      request_id: 'rid-1',
    });
    expect(e).toBeInstanceOf(ApiError);
    expect([e.status, e.code, e.message, e.requestId, e.retryAfterS]).toEqual([
      409,
      'history_exists',
      'has history',
      'rid-1',
      3,
    ]);
    expect(e.details).toEqual([{ a: 1 }]);
    expect(isApiError(e, 'history_exists')).toBe(true);
    expect(isApiError(e, 'other')).toBe(false);
  });
  it('falls back to the status and the X-Request-ID header', () => {
    const e = apiErrorFrom(res(503, { 'X-Request-ID': 'hdr', 'Retry-After': '5' }), 'not json');
    expect([e.code, e.requestId, e.retryAfterS, e.retryable]).toEqual([
      'unavailable',
      'hdr',
      5,
      true,
    ]);
    expect(apiErrorFrom(res(502), null).code).toBe('internal');
    expect(apiErrorFrom(res(400), {}).retryable).toBe(false);
  });
  it('treats a missing response as a retryable network error', () => {
    const e = networkError(new TypeError('Failed to fetch'));
    expect([e.status, e.code, e.retryable]).toEqual([0, 'network', true]);
  });
});
