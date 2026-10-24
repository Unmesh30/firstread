// One error type for every API failure: the API's shared error body
// ({"error": {"code", "message", "details?"}, "request_id"}, ADR 0015), HTTP status,
// the request id, and Retry-After in seconds when the server sent one.

export class ApiError extends Error {
  constructor(
    readonly status: number, // 0: the request never got a response (offline, CORS, DNS)
    readonly code: string,
    message: string,
    readonly requestId: string | null = null,
    readonly retryAfterS: number | null = null,
    readonly details: unknown = null,
  ) {
    super(message);
    this.name = 'ApiError';
  }

  /** Worth retrying the same request later: rate limits, overload, or no network. */
  get retryable(): boolean {
    return this.status === 0 || this.status === 429 || this.status === 503;
  }
}

/** Retry-After as seconds: delta-seconds or an HTTP date (RFC 9110 §10.2.3). */
export function parseRetryAfter(value: string | null, now: number = Date.now()): number | null {
  if (value == null || value.trim() === '') return null;
  const v = value.trim();
  if (/^\d+$/.test(v)) return Number(v);
  const at = Date.parse(v);
  if (Number.isNaN(at)) return null;
  return Math.max(0, Math.ceil((at - now) / 1000));
}

interface ErrorBody {
  error?: { code?: unknown; message?: unknown; details?: unknown };
  request_id?: unknown;
}

const FALLBACK: Record<number, string> = {
  400: 'bad_request',
  401: 'unauthorized',
  404: 'not_found',
  409: 'conflict',
  413: 'payload_too_large',
  422: 'invalid_request',
  429: 'rate_limited',
  503: 'unavailable',
};

/** Build an ApiError from a non-2xx response and its parsed body (if any). */
export function apiErrorFrom(response: Response, body: unknown): ApiError {
  const b = (body && typeof body === 'object' ? body : {}) as ErrorBody;
  const code =
    typeof b.error?.code === 'string'
      ? b.error.code
      : (FALLBACK[response.status] ?? (response.status >= 500 ? 'internal' : 'error'));
  const message =
    typeof b.error?.message === 'string' ? b.error.message : `HTTP ${response.status}`;
  const requestId =
    typeof b.request_id === 'string' ? b.request_id : response.headers.get('X-Request-ID');
  return new ApiError(
    response.status,
    code,
    message,
    requestId,
    parseRetryAfter(response.headers.get('Retry-After')),
    b.error?.details ?? null,
  );
}

export function networkError(cause: unknown): ApiError {
  const message = cause instanceof Error ? cause.message : 'network request failed';
  return new ApiError(0, 'network', message);
}

export function isApiError(e: unknown, code?: string): e is ApiError {
  return e instanceof ApiError && (code === undefined || e.code === code);
}
