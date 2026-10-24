// Loading placeholders. They pulse unless the reader asked the OS to reduce motion, and
// are hidden from screen readers (the screen announces "Loading" once instead).
import { useEffect, useState } from 'react';
import { Animated, StyleSheet, View, type DimensionValue } from 'react-native';

import { useReducedMotion } from '@/hooks/useReducedMotion';

import { useTheme } from './theme';
import { radius, space } from './tokens';

export function Skeleton({
  width = '100%',
  height = 16,
  round = false,
}: {
  width?: DimensionValue;
  height?: number;
  round?: boolean;
}) {
  const { colors } = useTheme();
  const reduced = useReducedMotion();
  const [opacity] = useState(() => new Animated.Value(1));
  useEffect(() => {
    if (reduced) {
      opacity.setValue(1);
      return;
    }
    const loop = Animated.loop(
      Animated.sequence([
        Animated.timing(opacity, { toValue: 0.5, duration: 700, useNativeDriver: true }),
        Animated.timing(opacity, { toValue: 1, duration: 700, useNativeDriver: true }),
      ]),
    );
    loop.start();
    return () => loop.stop();
  }, [reduced, opacity]);
  return (
    <Animated.View
      aria-hidden
      importantForAccessibility="no-hide-descendants"
      style={{
        width,
        height,
        opacity,
        backgroundColor: colors.skeleton,
        borderRadius: round ? radius.pill : radius.sm,
      }}
    />
  );
}

/** The shape of an article card while the feed loads. */
export function SkeletonCard() {
  const { colors } = useTheme();
  return (
    <View aria-hidden style={[styles.card, { borderBottomColor: colors.border }]}>
      <View style={styles.text}>
        <Skeleton width="40%" height={12} />
        <Skeleton height={20} />
        <Skeleton width="75%" height={20} />
        <Skeleton width="55%" height={12} />
      </View>
      <Skeleton width={88} height={88} />
    </View>
  );
}

export function LoadingList({ count = 4, label = 'Loading' }: { count?: number; label?: string }) {
  return (
    <View role="progressbar" accessibilityLabel={label} aria-busy>
      {Array.from({ length: count }, (_, i) => (
        <SkeletonCard key={i} />
      ))}
    </View>
  );
}

const styles = StyleSheet.create({
  card: {
    flexDirection: 'row',
    gap: space.lg,
    paddingVertical: space.lg,
    borderBottomWidth: StyleSheet.hairlineWidth,
  },
  text: { flex: 1, gap: space.sm },
});
