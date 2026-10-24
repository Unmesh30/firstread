// Local sign-in: a UUID kept on this device and sent as X-Dev-User. The API accepts it
// only with AUTH_DEV_BYPASS on the local stack (ADR 0015); prod refuses to start with it.
import type { KV } from '@/storage/kv';

import { AUTH_MESSAGES, AuthError, type AuthProvider, type AuthSession } from './types';

export const DEV_USER_KEY = 'dev-user';
const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

export function createDevAuth(store: KV, newId: () => string): AuthProvider {
  let session: AuthSession | null = null;
  const listeners = new Set<(s: AuthSession | null) => void>();
  const emit = () => listeners.forEach((l) => l(session));

  return {
    mode: 'dev',
    async init() {
      let id = await store.get(DEV_USER_KEY);
      if (!id || !UUID.test(id)) {
        id = newId();
        await store.set(DEV_USER_KEY, id);
      }
      session = { userId: id, isAnonymous: true };
      emit();
      return session;
    },
    async headers(): Promise<Record<string, string>> {
      return session ? { 'X-Dev-User': session.userId } : {};
    },
    async refresh() {
      return false; // nothing to refresh: a 401 here is a configuration problem
    },
    async linkGoogle() {
      throw new AuthError('not_available', AUTH_MESSAGES.not_available);
    },
    async signInWithGoogle() {
      throw new AuthError('not_available', AUTH_MESSAGES.not_available);
    },
    async reset() {
      await store.remove(DEV_USER_KEY);
      return this.init();
    },
    takeRedirectError() {
      return null;
    },
    subscribe(listener) {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
  };
}
