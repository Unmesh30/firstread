// About · Privacy · Source code, at the bottom of the section pages and onboarding.
import { Link } from 'expo-router';
import { StyleSheet, View } from 'react-native';

import { Divider, ExternalLink, Text } from '@/ui';
import { space } from '@/ui/tokens';

import { REPO_URL } from './credits';

export function FooterLinks() {
  return (
    <View role="contentinfo" style={styles.base}>
      <Divider />
      <View style={styles.row}>
        <Link href="/about">
          <Text variant="small" tone="accent" weight="600">
            About FirstRead
          </Text>
        </Link>
        <Link href="/privacy">
          <Text variant="small" tone="accent" weight="600">
            Privacy
          </Text>
        </Link>
        <ExternalLink href={REPO_URL} variant="small">
          Source code on GitHub
        </ExternalLink>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  base: { gap: space.lg, paddingTop: space.lg },
  row: { flexDirection: 'row', flexWrap: 'wrap', gap: space.lg },
});
