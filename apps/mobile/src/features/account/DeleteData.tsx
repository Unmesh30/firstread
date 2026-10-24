// "Delete my data": DELETE /v1/me removes the profile, history, events, served slates
// and cached state. 204 means every copy is gone; 503 with Retry-After means Redis
// wasn't cleared yet (the Postgres rows already are), and retrying is safe, so the app
// retries on its own (ADR 0016). Then this device starts over as a new reader.
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { useState } from 'react';
import { StyleSheet, View } from 'react-native';

import { call } from '@/api/client';
import { retryDelay, shouldRetry } from '@/api/queries';
import { useEventQueue } from '@/events/EventsProvider';
import { useSession } from '@/providers/session';
import { Banner, Button, Dialog, Text, describeError } from '@/ui';
import { space } from '@/ui/tokens';

import { discardPending, startFresh } from './fresh';

export const DELETE_BODY =
  'This deletes your topics, reading history, likes, hides and the record of what you were shown. It can’t be undone. This device then starts over as a new reader.';
export const DELETED = 'Your data was deleted';

export function DeleteData() {
  const session = useSession();
  const qc = useQueryClient();
  const queue = useEventQueue();
  const [confirm, setConfirm] = useState(false);
  const ready = session.status === 'ready' ? session : null;

  const del = useMutation({
    mutationFn: async () => {
      discardPending(queue); // nothing queued may re-create the data
      await call(ready!.api.DELETE('/v1/me'));
    },
    retry: shouldRetry, // 503/429/offline: idempotent, so retried with Retry-After
    retryDelay,
    onSuccess: async () => {
      setConfirm(false);
      await startFresh(ready!.auth, qc, DELETED);
    },
  });

  const retrying = del.isPending && del.failureCount > 0;
  const failed = del.error
    ? `Your data isn’t fully deleted yet. ${describeError(del.error)}`
    : null;
  return (
    <View style={styles.base}>
      <Text tone="muted">
        Removes everything FirstRead keeps about you: topics, history, likes, hides and impressions.
      </Text>
      {failed && !confirm ? (
        <Banner
          tone="danger"
          message={failed}
          action={{ label: 'Try again', onPress: () => del.mutate() }}
          testID="delete-error"
        />
      ) : null}
      <View style={styles.actions}>
        <Button
          variant="danger"
          label="Delete my data"
          disabled={!ready}
          onPress={() => setConfirm(true)}
          testID="delete-data"
        />
      </View>
      <Dialog
        visible={confirm}
        title="Delete your data?"
        body={
          <View style={styles.base}>
            <Text tone="muted">{DELETE_BODY}</Text>
            {retrying ? (
              <Text role="status" variant="small" testID="delete-retrying">
                Still deleting. Retrying…
              </Text>
            ) : null}
            {failed && !del.isPending ? (
              <Text role="alert" variant="small" tone="danger" testID="delete-failed">
                {failed}
              </Text>
            ) : null}
          </View>
        }
        confirmLabel={failed ? 'Try again' : 'Delete'}
        destructive
        busy={del.isPending}
        onConfirm={() => del.mutate()}
        onCancel={() => setConfirm(false)}
      />
    </View>
  );
}

const styles = StyleSheet.create({
  base: { gap: space.md },
  actions: { flexDirection: 'row', flexWrap: 'wrap', gap: space.md },
});
