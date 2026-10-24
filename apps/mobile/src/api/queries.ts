// Query keys, cache lifetimes and the shared retry policy (TanStack Query).
//
// The feed is never refetched automatically: every refetch would be a new first page,
// 7 Redis commands against a monthly budget (ADR 0011). Public data follows the API's
// own Cache-Control (60 s).
import { QueryClient } from '@tanstack/react-query';

import type { Api } from './client';
import { call } from './client';
import { ApiError } from './errors';

export const keys = {
  me: ['me'] as const,
  topics: ['topics'] as const,
  stats: ['stats'] as const,
  personas: ['personas'] as const,
  feed: ['feed'] as const,
  why: (id: string) => ['why', id] as const,
};

export const staleTimes = {
  topics: 24 * 60 * 60 * 1000,
  me: 30 * 1000,
  stats: 60 * 1000,
  personas: 60 * 1000,
} as const;

export const MAX_RETRIES = 3;
export const MAX_RETRY_AFTER_S = 30;

export function shouldRetry(failureCount: number, error: unknown): boolean {
  return error instanceof ApiError && error.retryable && failureCount < MAX_RETRIES;
}

/** Retry-After when the server sent one (capped), else exponential backoff with jitter. */
export function retryDelay(attempt: number, error: unknown, random: () => number = Math.random) {
  if (error instanceof ApiError && error.retryAfterS != null) {
    return Math.min(error.retryAfterS, MAX_RETRY_AFTER_S) * 1000;
  }
  const base = Math.min(1000 * 2 ** attempt, MAX_RETRY_AFTER_S * 1000);
  return base / 2 + random() * (base / 2);
}

export function createQueryClient(): QueryClient {
  return new QueryClient({
    defaultOptions: {
      queries: { retry: shouldRetry, retryDelay, refetchOnWindowFocus: false },
      mutations: { retry: false },
    },
  });
}

export const fetchers = {
  me: (api: Api) => call(api.GET('/v1/me')),
  topics: (api: Api) => call(api.GET('/v1/topics')),
  stats: (api: Api) => call(api.GET('/v1/stats')),
  personas: (api: Api) => call(api.GET('/v1/demo/personas')),
};
