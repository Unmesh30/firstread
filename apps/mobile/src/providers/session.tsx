// Who the reader is, and the API client that speaks for them.
//
// The auth provider and client are created in an effect: static rendering pre-renders
// every route in Node, where there is no storage and no session. Until `ready`,
// screens show their skeletons.
import {
  createContext,
  useContext,
  useEffect,
  useMemo,
  useState,
  type PropsWithChildren,
} from 'react';

import { createApi, type Api } from '@/api/client';
import { createAuth, type AuthProvider, type AuthSession } from '@/auth';
import { configErrors, env } from '@/lib/env';

export type SessionState =
  | { status: 'loading' }
  | { status: 'error'; message: string }
  | { status: 'ready'; session: AuthSession; auth: AuthProvider; api: Api; apiUrl: string };

const SessionContext = createContext<SessionState>({ status: 'loading' });

export function SessionProvider({
  children,
  factory = () => createAuth(env),
  apiUrl = env.apiUrl,
}: PropsWithChildren<{ factory?: () => AuthProvider; apiUrl?: string }>) {
  const [state, setState] = useState<SessionState>({ status: 'loading' });
  const problems = configErrors(env).join(' ');
  useEffect(() => {
    if (problems) return;
    let alive = true;
    const auth = factory();
    const api = createApi(apiUrl, auth);
    const unsubscribe = auth.subscribe((session) => {
      if (alive && session) setState({ status: 'ready', session, auth, api, apiUrl });
    });
    auth
      .init()
      .then((session) => alive && setState({ status: 'ready', session, auth, api, apiUrl }))
      .catch(
        (e: unknown) =>
          alive &&
          setState({ status: 'error', message: e instanceof Error ? e.message : 'Sign-in failed' }),
      );
    return () => {
      alive = false;
      unsubscribe();
    };
    // The factory and the build's configuration are fixed for the app's lifetime.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  const value: SessionState = problems ? { status: 'error', message: problems } : state;
  return <SessionContext.Provider value={value}>{children}</SessionContext.Provider>;
}

export function useSession(): SessionState {
  return useContext(SessionContext);
}

/** The API client once signed in, else null (queries stay disabled until then). */
export function useApi(): Api | null {
  const s = useSession();
  return useMemo(() => (s.status === 'ready' ? s.api : null), [s]);
}
