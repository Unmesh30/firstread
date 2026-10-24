// One story in the feed. The headline block is a real link to the publisher (a new tab
// on web), labelled "headline, source, 12 minutes ago". Like, hide and why are separate
// buttons beside it (a link can't contain buttons), each named with the headline.
import { Image } from 'expo-image';
import { useState } from 'react';
import { Platform, StyleSheet, View } from 'react-native';

import { timeAgo } from '@/lib/time';

import { IconButton } from './IconButton';
import { Pressable } from './Pressable';
import { Text } from './Text';
import { useTheme } from './theme';
import { radius, space } from './tokens';

export const THUMB = 88;

export interface ArticleCardProps {
  id: string;
  title: string;
  sourceName: string;
  url: string;
  imageUrl: string | null;
  /** published_at, else first_seen_at. */
  at: string;
  reason?: string;
  liked: boolean;
  onOpen: () => void;
  onLike: () => void;
  onHide: () => void;
  onWhy: () => void;
  now?: number;
}

export function Thumbnail({ uri, label, id }: { uri: string | null; label: string; id: string }) {
  const { colors } = useTheme();
  const [failed, setFailed] = useState(false);
  if (!uri || failed) {
    return (
      <View
        aria-hidden
        style={[styles.thumb, styles.initial, { backgroundColor: colors.surfaceAlt }]}
      >
        <Text variant="title" serif tone="muted">
          {label.trim().charAt(0).toUpperCase() || '·'}
        </Text>
      </View>
    );
  }
  return (
    <Image
      source={{ uri }}
      style={styles.thumb}
      contentFit="cover"
      cachePolicy="memory-disk"
      recyclingKey={id}
      testID={`thumb-${id}`}
      accessible={false}
      onError={() => setFailed(true)}
      {...(Platform.OS === 'web' ? { loading: 'lazy' as const } : null)}
    />
  );
}

export function ArticleCard(p: ArticleCardProps) {
  const { colors } = useTheme();
  const ago = timeAgo(p.at, p.now);
  const link = Platform.select({
    web: { href: p.url, hrefAttrs: { target: '_blank', rel: 'noopener noreferrer' } },
    default: {},
  }) as object;
  return (
    <View testID={`card-${p.id}`} style={[styles.card, { borderBottomColor: colors.border }]}>
      <Pressable
        {...link}
        role="link"
        accessibilityLabel={`${p.title}, ${p.sourceName}${ago ? `, ${ago}` : ''}`}
        accessibilityHint={Platform.OS === 'web' ? 'Opens in a new tab' : undefined}
        onPress={p.onOpen}
        testID={`open-${p.id}`}
        style={({ hovered }) => [styles.link, hovered ? { opacity: 0.85 } : null]}
      >
        <View style={styles.body}>
          <Text variant="caption" tone="muted" weight="600" numberOfLines={1}>
            {p.sourceName.toUpperCase()}
            {ago ? ` · ${ago}` : ''}
          </Text>
          <Text variant="lead" serif weight="600" style={styles.title}>
            {p.title}
          </Text>
          {p.reason ? (
            <Text variant="small" tone="accent" numberOfLines={2}>
              {p.reason}
            </Text>
          ) : null}
        </View>
        <Thumbnail uri={p.imageUrl} label={p.sourceName} id={p.id} />
      </Pressable>
      <View style={styles.actions}>
        <IconButton
          icon="heart"
          label={p.liked ? `Liked: ${p.title}` : `Like: ${p.title}`}
          selected={p.liked}
          onPress={p.onLike}
          testID={`like-${p.id}`}
        />
        <IconButton
          icon="eyeOff"
          label={`Hide: ${p.title}`}
          onPress={p.onHide}
          testID={`hide-${p.id}`}
        />
        <IconButton
          icon="info"
          label={`Why am I seeing this: ${p.title}`}
          onPress={p.onWhy}
          testID={`why-${p.id}`}
        />
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  card: {
    paddingTop: space.lg,
    paddingBottom: space.xs,
    borderBottomWidth: StyleSheet.hairlineWidth,
    gap: space.xs,
  },
  link: { flexDirection: 'row', gap: space.lg, alignItems: 'flex-start', borderRadius: radius.sm },
  body: { flex: 1, gap: space.xs },
  title: { marginTop: 2 },
  thumb: { width: THUMB, height: THUMB, borderRadius: radius.sm },
  initial: { alignItems: 'center', justifyContent: 'center' },
  actions: { flexDirection: 'row', justifyContent: 'flex-end', marginRight: -space.sm },
});
