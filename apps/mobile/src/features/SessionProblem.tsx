// A 401 that survived one refresh. With the dev bypass it's a configuration problem (the
// API isn't running with AUTH_DEV_BYPASS); with Supabase the session expired, and a new
// guest session starts only after the reader agrees (it leaves this history behind).
import { useQueryClient } from '@tanstack/react-query';
import { useState } from 'react';
import { StyleSheet, View } from 'react-native';

import { useSession } from '@/providers/session';
import { Banner, Button, Dialog, Text } from '@/ui';
import { space } from '@/ui/tokens';

export const DEV_401 =
  'The local API rejected the developer sign-in. Start it with AUTH_DEV_BYPASS=true and APP_ENV=local.';

export function SessionProblem() {
  const session = useSession();
  const qc = useQueryClient();
  const [confirm, setConfirm] = useState(false);
  const [busy, setBusy] = useState(false);
  if (session.status !== 'ready') return null;
  if (session.auth.mode === 'dev') {
    return <Banner tone="danger" message={DEV_401} testID="dev-401" />;
  }
  const restart = async () => {
    setBusy(true);
    try {
      await session.auth.reset();
      qc.clear();
    } finally {
      setBusy(false);
      setConfirm(false);
    }
  };
  return (
    <View role="alert" style={styles.base} testID="session-expired">
      <Text variant="title" serif>
        Your session expired
      </Text>
      <Text tone="muted">
        Start a new guest session to keep reading. Your current reading history stays with the old
        session.
      </Text>
      <Button label="Start a new session" onPress={() => setConfirm(true)} />
      <Dialog
        visible={confirm}
        title="Start a new session?"
        body="Your feed will start fresh: topics and reading history from the expired session won’t carry over."
        confirmLabel="Start fresh"
        busy={busy}
        onConfirm={() => void restart()}
        onCancel={() => setConfirm(false)}
      />
    </View>
  );
}

const styles = StyleSheet.create({ base: { gap: space.md, paddingVertical: space.xl } });
