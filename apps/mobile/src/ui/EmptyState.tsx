import { StyleSheet, View } from 'react-native';

import { Button } from './Button';
import { Text } from './Text';
import { space } from './tokens';

export interface EmptyStateProps {
  title: string;
  body?: string;
  action?: { label: string; onPress: () => void };
  testID?: string;
}

export function EmptyState({ title, body, action, testID }: EmptyStateProps) {
  return (
    <View style={styles.base} testID={testID}>
      <Text variant="title" serif role="heading" aria-level={2}>
        {title}
      </Text>
      {body ? (
        <Text tone="muted" style={styles.body}>
          {body}
        </Text>
      ) : null}
      {action ? <Button variant="secondary" label={action.label} onPress={action.onPress} /> : null}
    </View>
  );
}

const styles = StyleSheet.create({
  base: {
    alignItems: 'center',
    gap: space.md,
    paddingVertical: space.xxxl,
    paddingHorizontal: space.lg,
  },
  body: { textAlign: 'center', maxWidth: 420 },
});
