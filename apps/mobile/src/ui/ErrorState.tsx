import { StyleSheet, View } from 'react-native';

import { ApiError } from '@/api/errors';

import { Button } from './Button';
import { Text } from './Text';
import { space } from './tokens';

/** What a reader sees when a request failed: a plain message, Retry, and the request id. */
export function describeError(error: unknown): string {
  if (error instanceof ApiError) {
    if (error.status === 0) return 'Can’t reach FirstRead. Check your connection and try again.';
    if (error.status === 429 || error.status === 503) {
      const wait = error.retryAfterS ? ` Try again in ${error.retryAfterS} s.` : '';
      return `FirstRead is busy right now.${wait}`;
    }
    if (error.status === 401) return 'Your session isn’t valid any more.';
  }
  return 'Something went wrong on our side.';
}

export function ErrorState({
  error,
  onRetry,
  testID,
}: {
  error: unknown;
  onRetry?: () => void;
  testID?: string;
}) {
  const requestId = error instanceof ApiError ? error.requestId : null;
  return (
    <View role="alert" style={styles.base} testID={testID}>
      <Text variant="title" serif>
        {describeError(error)}
      </Text>
      {requestId ? (
        <Text variant="caption" tone="muted" selectable>
          Request id: {requestId}
        </Text>
      ) : null}
      {onRetry ? (
        <Button variant="secondary" icon="refresh" label="Try again" onPress={onRetry} />
      ) : null}
    </View>
  );
}

const styles = StyleSheet.create({
  base: { alignItems: 'flex-start', gap: space.md, paddingVertical: space.xl },
});
