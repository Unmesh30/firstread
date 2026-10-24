// The typed API client: openapi-fetch over types generated from the committed spec
// (services/api/openapi.json → src/api/schema.d.ts, `npm run gen:api`).
//
// Every call goes through `authFetch`: it adds the auth headers, and after a 401 it
// asks the auth provider to refresh once and retries the request once.
import createClient from 'openapi-fetch';

import { ApiError, apiErrorFrom, networkError } from './errors';
import type { paths } from './schema';

export interface AuthHeaders {
  headers(): Promise<Record<string, string>>;
  /** Try to get fresh credentials; true if a retry is worth it. */
  refresh(): Promise<boolean>;
}

type Fetch = (input: Request) => Promise<Response>;

export function authFetch(auth: AuthHeaders, base: Fetch = (r) => fetch(r)): Fetch {
  return async (request) => {
    const retry = request.clone(); // a body can be read only once
    const send = async (r: Request) => {
      for (const [k, v] of Object.entries(await auth.headers())) r.headers.set(k, v);
      return base(r);
    };
    const first = await send(request);
    if (first.status !== 401 || !(await auth.refresh())) return first;
    return send(retry);
  };
}

export function createApi(baseUrl: string, auth: AuthHeaders, base?: Fetch) {
  return createClient<paths>({ baseUrl, fetch: authFetch(auth, base) });
}
export type Api = ReturnType<typeof createApi>;

interface Result<T> {
  data?: T;
  error?: unknown;
  response: Response;
}

/** Await an openapi-fetch call: the data, or an ApiError (network failures included). */
export async function call<T>(request: Promise<Result<T>>): Promise<T> {
  let result: Result<T>;
  try {
    result = await request;
  } catch (e) {
    throw e instanceof ApiError ? e : networkError(e);
  }
  if (!result.response.ok) throw apiErrorFrom(result.response, result.error);
  return result.data as T;
}
