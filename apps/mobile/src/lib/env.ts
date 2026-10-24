// Build-time configuration. Expo inlines EXPO_PUBLIC_* values into the bundle, so they
// must be read with literal `process.env.EXPO_PUBLIC_...` expressions, and they are
// public by design: never put a secret in one (see .env.example at the repo root).

export type AuthMode = 'dev' | 'supabase';

export interface AppEnv {
  apiUrl: string;
  authMode: AuthMode;
  supabaseUrl: string;
  supabaseKey: string;
  siteUrl: string;
}

export const env: AppEnv = {
  apiUrl: (process.env.EXPO_PUBLIC_API_URL ?? 'http://localhost:8000').replace(/\/+$/, ''),
  authMode: process.env.EXPO_PUBLIC_AUTH_MODE === 'supabase' ? 'supabase' : 'dev',
  supabaseUrl: process.env.EXPO_PUBLIC_SUPABASE_URL ?? '',
  supabaseKey: process.env.EXPO_PUBLIC_SUPABASE_PUBLISHABLE_KEY ?? '',
  siteUrl: (process.env.EXPO_PUBLIC_SITE_URL ?? 'https://unmesh30.github.io/firstread').replace(
    /\/+$/,
    '',
  ),
};

export function isLocalUrl(url: string): boolean {
  try {
    const host = new URL(url).hostname;
    return host === 'localhost' || host === '127.0.0.1';
  } catch {
    return false;
  }
}

/** Configuration problems that must stop the app (shown on screen, never ignored). */
export function configErrors(e: AppEnv): string[] {
  const errors: string[] = [];
  if (e.authMode === 'dev' && !isLocalUrl(e.apiUrl)) {
    errors.push('Dev sign-in only works against a local API (localhost or 127.0.0.1).');
  }
  if (e.authMode === 'supabase' && (!e.supabaseUrl || !e.supabaseKey)) {
    errors.push('Supabase sign-in needs EXPO_PUBLIC_SUPABASE_URL and a publishable key.');
  }
  return errors;
}
