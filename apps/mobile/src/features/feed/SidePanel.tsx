// Beside the feed on wide screens: the reader's topics and a short "how it works".
import { Link } from 'expo-router';
import { StyleSheet, View } from 'react-native';

import { useMe, useTopics } from '@/features/me';
import { Divider, Text, useTheme } from '@/ui';
import { space } from '@/ui/tokens';

export const PANEL_WIDTH = 280;

export function SidePanel() {
  const { colors } = useTheme();
  const me = useMe();
  const topics = useTopics();
  const names = new Map(topics.data?.topics.map((t) => [t.id, t.name]));
  const followed = me.data?.topics ?? [];
  return (
    <View
      role="complementary"
      aria-label="About your feed"
      style={[styles.panel, { borderLeftColor: colors.border }]}
    >
      <Text variant="title" serif role="heading" aria-level={2}>
        Your topics
      </Text>
      {me.data?.active_persona ? <Text tone="muted">You’re reading as a demo reader.</Text> : null}
      {followed.length ? (
        <View role="list" style={styles.list}>
          {followed.map((t) => (
            <Text key={t} role="listitem">
              {names.get(t) ?? t}
            </Text>
          ))}
        </View>
      ) : (
        <Text tone="muted">No topics yet.</Text>
      )}
      <Divider />
      <Text variant="title" serif role="heading" aria-level={2}>
        How it works
      </Text>
      <Text tone="muted" variant="small">
        Every story is understood from its headline the moment it arrives, so it can reach the right
        readers before anyone has clicked. Each click, like or hide reshapes the rest of your feed.
      </Text>
      <Link href="/system" style={{ color: colors.accent }}>
        <Text variant="small" tone="accent" weight="600">
          See the live system
        </Text>
      </Link>
    </View>
  );
}

const styles = StyleSheet.create({
  panel: {
    width: PANEL_WIDTH,
    paddingTop: space.xl + space.sm,
    paddingHorizontal: space.xl,
    gap: space.md,
    borderLeftWidth: StyleSheet.hairlineWidth,
  },
  list: { gap: space.xs },
});
