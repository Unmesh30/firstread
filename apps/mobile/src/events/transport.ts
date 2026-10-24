// POST /v1/events over fetch. Every send uses keepalive (each batch is ≤ 60 KB, under
// the 64 KiB quota), so a request that is already on its way survives a closing page.
// A closing page's send starts synchronously with the last known auth headers: an
// await before fetch could be cut off by the unload.
import { authFetch, type AuthHeaders } from '@/api/client';
import { apiErrorFrom } from '@/api/errors';
import type { EventsResult } from '@/api/schema-types';

import type { QueuedEvent } from './batch';
import type { SendOutcome, Transport } from './queue';

type Fetch = (input: Request) => Promise<Response>;

export async function outcomeOf(response: Response): Promise<SendOutcome> {
  if (response.ok) {
    const body = (await response.json()) as EventsResult;
    return {
      kind: 'done',
      ids: [...body.accepted, ...body.duplicate, ...body.rejected.map((r) => r.event_id)],
    };
  }
  const s = response.status;
  // 401 survived a refresh (authFetch); keep the events for a later session state.
  if (s === 401 || s === 408 || s === 429 || s >= 500) {
    let body: unknown = null;
    try {
      body = await response.json();
    } catch {
      body = null;
    }
    return { kind: 'retry', retryAfterS: apiErrorFrom(response, body).retryAfterS };
  }
  return { kind: 'drop', status: s };
}

export function httpTransport(
  apiUrl: string,
  auth: AuthHeaders,
  base: Fetch = (r) => fetch(r),
): Transport {
  const url = `${apiUrl}/v1/events`;
  let lastHeaders: Record<string, string> = {};
  const remembering: AuthHeaders = {
    headers: async () => (lastHeaders = await auth.headers()),
    refresh: () => auth.refresh(),
  };
  const send = authFetch(remembering, base);
  const request = (events: QueuedEvent[], headers: Record<string, string> = {}) =>
    new Request(url, {
      method: 'POST',
      keepalive: true,
      headers: { 'Content-Type': 'application/json', ...headers },
      body: JSON.stringify({ events }),
    });

  return {
    async send(events, immediate = false) {
      try {
        // A closing page can't wait for headers (or a refresh): send with the last ones.
        const response = immediate
          ? await base(request(events, lastHeaders))
          : await send(request(events));
        return await outcomeOf(response);
      } catch {
        return { kind: 'retry', retryAfterS: null }; // offline, CORS, an aborted unload
      }
    },
  };
}
