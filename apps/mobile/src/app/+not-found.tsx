import { Link, Stack } from 'expo-router';

import { EmptyState, Screen, Text } from '@/ui';

export default function NotFound() {
  return (
    <Screen title="Page not found">
      <Stack.Screen options={{ title: 'Page not found' }} />
      <EmptyState title="This page doesn’t exist" body="The link may be old, or mistyped." />
      <Link href="/" style={{ alignSelf: 'center' }}>
        <Text tone="accent" weight="600">
          Go to your feed
        </Text>
      </Link>
    </Screen>
  );
}
