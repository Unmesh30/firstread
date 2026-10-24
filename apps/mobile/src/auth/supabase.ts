// Production sign-in (P5): Supabase anonymous sign-in on first launch, then optional
// Google linking with linkIdentity, which keeps the same user id (so the history stays).
// The client is created lazily, never at import time (static rendering runs in Node).
// Only the public URL and publishable key reach the bundle; RLS protects the data.
import { AppState, Platform } from 'react-native';

import {
  AUTH_MESSAGES,
  AuthError,
  authErrorFromUrl,
  toAuthError,
  type AuthProvider,
  type AuthSession,
} from './types';

interface SbSession {
  access_token: string;
  user: { id: string; is_anonymous?: boolean };
}
type SbResult<T> = Promise<{ data: T; error: unknown }>;

/** The part of supabase-js' auth client this app uses (mocked in tests). */
export interface SupabaseAuthClient {
  getSession(): SbResult<{ session: SbSession | null }>;
  signInAnonymously(): SbResult<{ session: SbSession | null }>;
  refreshSession(): SbResult<{ session: SbSession | null }>;
  linkIdentity(c: { provider: 'google'; options?: { redirectTo?: string } }): SbResult<unknown>;
  signInWithOAuth(c: { provider: 'google'; options?: { redirectTo?: string } }): SbResult<unknown>;
  signOut(): Promise<{ error: unknown }>;
  onAuthStateChange(cb: (event: string, session: SbSession | null) => void): {
    data: { subscription: { unsubscribe(): void } };
  };
  startAutoRefresh(): Promise<void>;
  stopAutoRefresh(): Promise<void>;
}

export interface SupabaseAuthOptions {
  getClient: () => Promise<SupabaseAuthClient>;
  redirectTo: () => string;
  currentUrl?: () => string | null;
  cleanUrl?: () => void;
}

const toSession = (s: SbSession): AuthSession => ({
  userId: s.user.id,
  isAnonymous: s.user.is_anonymous ?? false,
});

export function createSupabaseAuth(opts: SupabaseAuthOptions): AuthProvider {
  let client: SupabaseAuthClient | null = null;
  let token: string | null = null;
  let redirectError: AuthError | null = null;
  const listeners = new Set<(s: AuthSession | null) => void>();
  const emit = (s: AuthSession | null) => listeners.forEach((l) => l(s));
  const get = async () => (client ??= await opts.getClient());

  function check<T>(r: { data: T; error: unknown }): T {
    if (r.error) throw toAuthError(r.error);
    return r.data;
  }

  return {
    mode: 'supabase',
    async init() {
      const href = opts.currentUrl?.();
      if (href) {
        redirectError = authErrorFromUrl(href);
        if (redirectError) opts.cleanUrl?.();
      }
      const c = await get();
      c.onAuthStateChange((_event, s) => {
        token = s?.access_token ?? null;
        emit(s ? toSession(s) : null);
      });
      if (Platform.OS !== 'web') {
        // As in Expo's Supabase guide: refresh tokens only while the app is in the foreground.
        AppState.addEventListener('change', (state) =>
          state === 'active' ? c.startAutoRefresh() : c.stopAutoRefresh(),
        );
      }
      let session = check(await c.getSession()).session;
      if (!session) session = check(await c.signInAnonymously()).session;
      if (!session) throw new AuthError('unknown', AUTH_MESSAGES.unknown);
      token = session.access_token;
      return toSession(session);
    },
    async headers(): Promise<Record<string, string>> {
      return token ? { Authorization: `Bearer ${token}` } : {};
    },
    async refresh() {
      try {
        const { session } = check(await (await get()).refreshSession());
        token = session?.access_token ?? null;
        return token != null;
      } catch {
        return false;
      }
    },
    async linkGoogle() {
      check(
        await (
          await get()
        ).linkIdentity({
          provider: 'google',
          options: { redirectTo: opts.redirectTo() },
        }),
      );
    },
    async signInWithGoogle() {
      const c = await get();
      check(
        await c.signInWithOAuth({ provider: 'google', options: { redirectTo: opts.redirectTo() } }),
      );
    },
    async reset() {
      const c = await get();
      await c.signOut();
      token = null;
      const { session } = check(await c.signInAnonymously());
      if (!session) throw new AuthError('unknown', AUTH_MESSAGES.unknown);
      token = session.access_token;
      return toSession(session);
    },
    takeRedirectError() {
      const e = redirectError;
      redirectError = null;
      return e;
    },
    subscribe(listener) {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
  };
}

/** The real client, loaded only in `supabase` mode (a separate chunk where Metro splits). */
export async function loadSupabaseClient(url: string, key: string): Promise<SupabaseAuthClient> {
  const { createClient } = await import('@supabase/supabase-js');
  const storage =
    Platform.OS === 'web'
      ? undefined // supabase-js uses localStorage in the browser
      : (await import('@react-native-async-storage/async-storage')).default;
  const sb = createClient(url, key, {
    auth: {
      storage,
      persistSession: true,
      autoRefreshToken: true,
      detectSessionInUrl: Platform.OS === 'web',
    },
  });
  return sb.auth as unknown as SupabaseAuthClient;
}
