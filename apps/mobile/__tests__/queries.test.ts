import { ApiError } from '@/api/errors';
import { MAX_RETRY_AFTER_S, retryDelay, shouldRetry } from '@/api/queries';

describe('retry policy', () => {
  const busy = new ApiError(503, 'unavailable', 'busy', null, 5);
  it('retries only rate limits, overload and network errors, at most 3 times', () => {
    expect(shouldRetry(0, busy)).toBe(true);
    expect(shouldRetry(2, new ApiError(429, 'rate_limited', ''))).toBe(true);
    expect(shouldRetry(0, new ApiError(0, 'network', ''))).toBe(true);
    expect(shouldRetry(3, busy)).toBe(false);
    expect(shouldRetry(0, new ApiError(400, 'cursor_expired', ''))).toBe(false);
    expect(shouldRetry(0, new ApiError(401, 'unauthorized', ''))).toBe(false);
    expect(shouldRetry(0, new Error('x'))).toBe(false);
  });
  it('waits Retry-After (capped), else backs off exponentially with jitter', () => {
    expect(retryDelay(0, busy)).toBe(5000);
    expect(retryDelay(0, new ApiError(503, 'unavailable', '', null, 600))).toBe(
      MAX_RETRY_AFTER_S * 1000,
    );
    expect(retryDelay(0, new ApiError(0, 'network', ''), () => 0)).toBe(500);
    expect(retryDelay(2, new ApiError(0, 'network', ''), () => 1)).toBe(4000);
    expect(retryDelay(10, new ApiError(0, 'network', ''), () => 1)).toBe(30_000);
  });
});
