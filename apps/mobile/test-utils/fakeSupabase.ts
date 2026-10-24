// A Supabase auth client double for the Supabase AuthProvider (no network, no SDK).
import type { SupabaseAuthClient } from '@/auth/supabase';

export const FAKE_UID = '11111111-2222-4333-8444-555555555555';

export function session(token: string, anonymous = true) {
  return { access_token: token, user: { id: FAKE_UID, is_anonymous: anonymous } };
}

export function fakeSupabase(overrides: Partial<SupabaseAuthClient> = {}) {
  const calls: string[] = [];
  const ok = <T>(data: T) => Promise.resolve({ data, error: null });
  const client: SupabaseAuthClient = {
    getSession: () => (calls.push('getSession'), ok({ session: null })),
    signInAnonymously: () => (calls.push('signInAnonymously'), ok({ session: session('anon-1') })),
    refreshSession: () => (calls.push('refreshSession'), ok({ session: session('anon-2') })),
    linkIdentity: (c) => (
      calls.push(`linkIdentity:${c.provider}:${c.options?.redirectTo}`),
      ok({})
    ),
    signInWithOAuth: (c) => (calls.push(`signInWithOAuth:${c.provider}`), ok({})),
    signOut: () => (calls.push('signOut'), Promise.resolve({ error: null })),
    onAuthStateChange: () => ({ data: { subscription: { unsubscribe() {} } } }),
    startAutoRefresh: async () => undefined,
    stopAutoRefresh: async () => undefined,
    ...overrides,
  };
  return { client, calls };
}
