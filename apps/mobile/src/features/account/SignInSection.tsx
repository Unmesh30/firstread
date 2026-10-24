// Who the reader is: a guest by default, optionally linked to Google so the profile
// follows them across devices (Supabase linkIdentity keeps the same user and history).
//
// On web, linking leaves the page for Google and comes back to /account; a failure
// comes back in the URL (takeRedirectError). identity_already_exists means that
// Google account belongs to another FirstRead reader: the reader can switch to it, but
// this device's history stays behind, and the app says so first.
import { useState } from 'react';
import { StyleSheet, View } from 'react-native';

import { toAuthError, type AuthError, type AuthProvider } from '@/auth';
import { Banner, Button, Dialog, Text } from '@/ui';
import { space } from '@/ui/tokens';

export const SWITCH_WARNING =
  'You’ll see that account’s topics and history instead. This device’s reading history won’t come with you.';

// takeRedirectError() consumes the error; keep it per provider so a re-run initializer
// (React may call it twice in development) still sees it.
const redirectErrors = new WeakMap<AuthProvider, AuthError | null>();
function redirectError(auth: AuthProvider): AuthError | null {
  if (!redirectErrors.has(auth)) redirectErrors.set(auth, auth.takeRedirectError());
  return redirectErrors.get(auth) ?? null;
}

export function SignInSection({ auth, isAnonymous }: { auth: AuthProvider; isAnonymous: boolean }) {
  // Back from Google with an error in the URL, if any.
  const [error, setError] = useState<AuthError | null>(() => redirectError(auth));
  const [busy, setBusy] = useState<'link' | 'switch' | null>(null);
  const [confirmSwitch, setConfirmSwitch] = useState(false);

  const run = async (kind: 'link' | 'switch', action: () => Promise<void>) => {
    setBusy(kind);
    setError(null);
    try {
      await action(); // on web a success leaves the page
    } catch (e) {
      setError(toAuthError(e));
    } finally {
      setBusy(null);
      setConfirmSwitch(false);
    }
  };

  if (!isAnonymous) {
    return (
      <View style={styles.base} testID="signed-in">
        <Text>Signed in with Google. Your topics and history follow you to other devices.</Text>
      </View>
    );
  }

  return (
    <View style={styles.base}>
      <Text testID="guest-status">
        You’re reading as a guest. Your topics and reading history are kept for this{' '}
        {auth.mode === 'dev' ? 'browser (a developer sign-in on the local stack)' : 'browser'}.
      </Text>
      <Text tone="muted" variant="small">
        Continue with Google to keep them on other devices too. Nothing you’ve read is lost.
      </Text>
      <View style={styles.actions}>
        <Button
          variant="secondary"
          label="Continue with Google"
          loading={busy === 'link'}
          disabled={busy != null}
          onPress={() => void run('link', () => auth.linkGoogle())}
          testID="link-google"
        />
      </View>
      {error ? (
        <View style={styles.base} testID="auth-error">
          <Banner
            tone={error.code === 'not_available' ? 'info' : 'danger'}
            message={error.message}
          />
          {error.code === 'identity_already_exists' ? (
            <View style={styles.actions}>
              <Button
                variant="secondary"
                label="Sign in to that Google account instead"
                onPress={() => setConfirmSwitch(true)}
                testID="switch-google"
              />
            </View>
          ) : null}
        </View>
      ) : null}
      <Dialog
        visible={confirmSwitch}
        title="Switch to that Google account?"
        body={SWITCH_WARNING}
        confirmLabel="Switch account"
        busy={busy === 'switch'}
        onConfirm={() => void run('switch', () => auth.signInWithGoogle())}
        onCancel={() => setConfirmSwitch(false)}
      />
    </View>
  );
}

const styles = StyleSheet.create({
  base: { gap: space.md },
  actions: { flexDirection: 'row', flexWrap: 'wrap', gap: space.md },
});
