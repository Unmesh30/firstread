import * as Crypto from 'expo-crypto';
import { Platform } from 'react-native';

import type { AppEnv } from '@/lib/env';
import { kv } from '@/storage/kv';

import { createDevAuth } from './dev';
import { createSupabaseAuth, loadSupabaseClient } from './supabase';
import type { AuthProvider } from './types';

export * from './types';

/** The provider for this build. Call it inside an effect, never at import time. */
export function createAuth(e: AppEnv): AuthProvider {
  if (e.authMode === 'dev') return createDevAuth(kv, () => Crypto.randomUUID());
  return createSupabaseAuth({
    getClient: () => loadSupabaseClient(e.supabaseUrl, e.supabaseKey),
    redirectTo: () => `${e.siteUrl}/account`,
    currentUrl: () => (Platform.OS === 'web' ? window.location.href : null),
    cleanUrl: () => {
      if (Platform.OS === 'web') window.history.replaceState(null, '', window.location.pathname);
    },
  });
}
