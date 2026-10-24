import { createDevAuth, DEV_USER_KEY } from '@/auth/dev';
import { createSupabaseAuth, type SupabaseAuthClient } from '@/auth/supabase';
import { AuthError, authErrorFromUrl, toAuthError } from '@/auth/types';
import { memoryKV } from '@/storage/kv';

import { fakeSupabase, session } from '../test-utils/fakeSupabase';

const ID = '11111111-2222-4333-8444-555555555555';

describe('dev auth', () => {
  it('creates a device id once, sends it as X-Dev-User and resets it on request', async () => {
    const store = memoryKV();
    let n = 0;
    const ids = [ID, '66666666-7777-4888-9999-000000000000'];
    const auth = createDevAuth(store, () => ids[n++]);
    expect(await auth.headers()).toEqual({});
    expect((await auth.init()).userId).toBe(ID);
    expect(await auth.headers()).toEqual({ 'X-Dev-User': ID });
    expect((await createDevAuth(store, () => 'unused').init()).userId).toBe(ID); // kept
    expect(await auth.refresh()).toBe(false);
    expect((await auth.reset()).userId).toBe(ids[1]);
    expect(store.data.get(DEV_USER_KEY)).toBe(ids[1]);
  });
  it('replaces a corrupt stored id and refuses Google', async () => {
    const auth = createDevAuth(memoryKV({ [DEV_USER_KEY]: 'nope' }), () => ID);
    expect((await auth.init()).userId).toBe(ID);
    await expect(auth.linkGoogle()).rejects.toMatchObject({ code: 'not_available' });
  });
});

const make = (client: SupabaseAuthClient, href: string | null = null) =>
  createSupabaseAuth({
    getClient: async () => client,
    redirectTo: () => 'https://site/firstread/account',
    currentUrl: () => href,
    cleanUrl: () => undefined,
  });

describe('supabase auth (mocked client)', () => {
  it('signs in anonymously on first launch and sends the bearer token', async () => {
    const { client, calls } = fakeSupabase();
    const auth = make(client);
    expect(await auth.init()).toEqual({ userId: ID, isAnonymous: true });
    expect(calls).toEqual(['getSession', 'signInAnonymously']);
    expect(await auth.headers()).toEqual({ Authorization: 'Bearer anon-1' });
  });
  it('restores an existing session without a new sign-in', async () => {
    const { client, calls } = fakeSupabase({
      getSession: () => Promise.resolve({ data: { session: session('kept', false) }, error: null }),
    });
    expect(await make(client).init()).toEqual({ userId: ID, isAnonymous: false });
    expect(calls).not.toContain('signInAnonymously');
  });
  it('refreshes the token after a 401', async () => {
    const { client } = fakeSupabase();
    const auth = make(client);
    await auth.init();
    expect(await auth.refresh()).toBe(true);
    expect(await auth.headers()).toEqual({ Authorization: 'Bearer anon-2' });
  });
  it('reports a failed refresh as "not worth retrying"', async () => {
    const { client } = fakeSupabase({
      refreshSession: () =>
        Promise.resolve({ data: { session: null }, error: { code: 'refresh_token_not_found' } }),
    });
    const auth = make(client);
    await auth.init();
    expect(await auth.refresh()).toBe(false);
  });
  it('links Google to the same reader, redirecting back to Account', async () => {
    const { client, calls } = fakeSupabase();
    const auth = make(client);
    await auth.init();
    await auth.linkGoogle();
    expect(calls).toContain('linkIdentity:google:https://site/firstread/account');
  });
  it('maps sign-in failures to reader-facing codes', async () => {
    const { client } = fakeSupabase({
      signInAnonymously: () =>
        Promise.resolve({ data: { session: null }, error: { code: 'over_request_rate_limit' } }),
    });
    await expect(make(client).init()).rejects.toMatchObject({ code: 'over_request_rate_limit' });
    const linking = fakeSupabase({
      linkIdentity: () =>
        Promise.resolve({ data: null, error: { code: 'manual_linking_disabled' } }),
    });
    const auth = make(linking.client);
    await auth.init();
    await expect(auth.linkGoogle()).rejects.toMatchObject({ code: 'manual_linking_disabled' });
  });
  it('picks up identity_already_exists from the OAuth redirect, once', async () => {
    const { client } = fakeSupabase();
    const auth = make(
      client,
      'https://site/firstread/account#error=server_error&error_code=identity_already_exists',
    );
    await auth.init();
    expect(auth.takeRedirectError()?.code).toBe('identity_already_exists');
    expect(auth.takeRedirectError()).toBeNull();
  });
  it('can switch to the Google account’s own reader', async () => {
    const { client, calls } = fakeSupabase();
    const auth = make(client);
    await auth.init();
    await auth.signInWithGoogle();
    expect(calls).toContain('signInWithOAuth:google');
  });
});

describe('auth error helpers', () => {
  it('maps codes and URLs', () => {
    expect(toAuthError({ code: 'captcha_failed' }).code).toBe('captcha_failed');
    expect(toAuthError({ code: 'something_new' }).code).toBe('unknown');
    expect(toAuthError(new AuthError('not_available', 'x')).code).toBe('not_available');
    expect(authErrorFromUrl('https://s/a?error_code=anonymous_provider_disabled')?.code).toBe(
      'anonymous_provider_disabled',
    );
    expect(authErrorFromUrl('https://s/a?error=access_denied')?.code).toBe('unknown');
    expect(authErrorFromUrl('https://s/a?id=1')).toBeNull();
  });
});
