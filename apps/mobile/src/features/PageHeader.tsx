// The top of a page outside the sections (personas, About, Privacy): a way back and the
// page's one h1. A deep link has nothing to go back to, so Back goes to the feed then.
import { router } from 'expo-router';
import { StyleSheet, View } from 'react-native';

import { Button, Text } from '@/ui';
import { space } from '@/ui/tokens';

export function goBack() {
  if (router.canGoBack()) router.back();
  else router.replace('/');
}

export function PageHeader({ title, lead }: { title: string; lead?: string }) {
  return (
    <View style={styles.base}>
      <View style={styles.back}>
        <Button variant="ghost" icon="arrowLeft" label="Back" onPress={goBack} testID="back" />
      </View>
      <Text variant="headline" serif role="heading" aria-level={1}>
        {title}
      </Text>
      {lead ? (
        <Text variant="lead" tone="muted">
          {lead}
        </Text>
      ) : null}
    </View>
  );
}

const styles = StyleSheet.create({
  base: { gap: space.sm },
  back: { alignSelf: 'flex-start', marginLeft: -space.lg },
});
