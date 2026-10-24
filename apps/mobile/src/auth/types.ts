// The app talks to one AuthProvider; which one is chosen by EXPO_PUBLIC_AUTH_MODE.
//   dev:      a local UUID sent as X-Dev-User (local stack only, ADR 0015).
//   supabase: anonymous sign-in, then optional Google linking (P5's project).

export interface AuthSession {
  userId: string;
  isAnonymous: boolean;
}

export type AuthErrorCode =
  | 'identity_already_exists'
  | 'manual_linking_disabled'
  | 'anonymous_provider_disabled'
  | 'over_request_rate_limit'
  | 'captcha_failed'
  | 'not_available'
  | 'unknown';

export class AuthError extends Error {
  constructor(
    readonly code: AuthErrorCode,
    message: string,
  ) {
    super(message);
    this.name = 'AuthError';
  }
}

/** What a reader sees for each failure (the account screen shows these). */
export const AUTH_MESSAGES: Record<AuthErrorCode, string> = {
  identity_already_exists:
    'That Google account is already linked to another FirstRead reader. You can sign in to it instead, but this device’s reading history won’t come with you.',
  manual_linking_disabled: 'Linking a Google account isn’t enabled yet. Please try again later.',
  anonymous_provider_disabled: 'Guest sign-in is turned off right now. Please try again later.',
  over_request_rate_limit:
    'Too many sign-in attempts from this network. Try again in a few minutes.',
  captcha_failed: 'We couldn’t verify that you’re human. Reload the page and try again.',
  not_available: 'Google sign-in is available on the live site, not on the local stack.',
  unknown: 'Something went wrong while signing in. Please try again.',
};

export interface AuthProvider {
  readonly mode: 'dev' | 'supabase';
  /** Restore or create the session. */
  init(): Promise<AuthSession>;
  headers(): Promise<Record<string, string>>;
  /** Fresh credentials after a 401; true if retrying is worth it. */
  refresh(): Promise<boolean>;
  /** Start linking Google to this reader (on web this leaves the page). */
  linkGoogle(): Promise<void>;
  /** Switch to the reader that owns this Google account (after identity_already_exists). */
  signInWithGoogle(): Promise<void>;
  /** Forget this reader on this device and start a new one. */
  reset(): Promise<AuthSession>;
  /** A failure carried back in the URL after an OAuth redirect, if any (consumed once). */
  takeRedirectError(): AuthError | null;
  subscribe(listener: (session: AuthSession | null) => void): () => void;
}

export function toAuthError(e: unknown): AuthError {
  if (e instanceof AuthError) return e;
  const code = (e as { code?: unknown } | null)?.code;
  const known =
    typeof code === 'string' && code in AUTH_MESSAGES ? (code as AuthErrorCode) : 'unknown';
  return new AuthError(known, AUTH_MESSAGES[known]);
}

/** Supabase reports OAuth failures in the redirect URL (?error_code=… or #error_code=…). */
export function authErrorFromUrl(href: string): AuthError | null {
  try {
    const url = new URL(href);
    const params = new URLSearchParams(url.hash.replace(/^#/, ''));
    url.searchParams.forEach((v, k) => params.set(k, v));
    const code = params.get('error_code') ?? (params.get('error') ? 'unknown' : null);
    return code ? toAuthError({ code }) : null;
  } catch {
    return null;
  }
}
