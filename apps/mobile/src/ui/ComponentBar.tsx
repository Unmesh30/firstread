// The blend behind one story's score (ADR 0014): model + fresh + trend = final, as a
// stacked bar with each part's value written out (the bar alone isn't the information).
import { StyleSheet, View } from 'react-native';

import { Text } from './Text';
import { useTheme } from './theme';
import { radius, space, type Colors } from './tokens';

export interface ComponentValues {
  model: number;
  fresh: number;
  trend: number;
  final: number;
}

const PARTS: { key: 'model' | 'fresh' | 'trend'; label: string; color: keyof Colors }[] = [
  { key: 'model', label: 'Your interests', color: 'accent' },
  { key: 'fresh', label: 'Freshness', color: 'info' },
  { key: 'trend', label: 'Trending', color: 'success' },
];

const fmt = (v: number) => (Math.abs(v) < 0.005 ? '0.00' : v.toFixed(2));

/** Each part's share of the positive total (negative parts draw nothing). */
export function shares(v: ComponentValues): Record<'model' | 'fresh' | 'trend', number> {
  const pos = (x: number) => Math.max(0, x);
  const total = pos(v.model) + pos(v.fresh) + pos(v.trend);
  if (total <= 0) return { model: 0, fresh: 0, trend: 0 };
  return { model: pos(v.model) / total, fresh: pos(v.fresh) / total, trend: pos(v.trend) / total };
}

export function ComponentBar({ values }: { values: ComponentValues }) {
  const { colors } = useTheme();
  const s = shares(values);
  return (
    <View style={styles.wrap} testID="component-bar">
      <View aria-hidden style={[styles.bar, { backgroundColor: colors.surfaceAlt }]}>
        {PARTS.map((p) =>
          s[p.key] > 0 ? (
            <View key={p.key} style={{ flex: s[p.key], backgroundColor: colors[p.color] }} />
          ) : null,
        )}
      </View>
      <View role="list" style={styles.legend}>
        {PARTS.map((p) => (
          <View key={p.key} role="listitem" style={styles.item}>
            <View aria-hidden style={[styles.swatch, { backgroundColor: colors[p.color] }]} />
            <Text variant="small">
              {p.label} {fmt(values[p.key])}
            </Text>
          </View>
        ))}
        <View role="listitem" style={styles.item}>
          <Text variant="small" weight="600">
            Score {fmt(values.final)}
          </Text>
        </View>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  wrap: { gap: space.sm },
  bar: { flexDirection: 'row', height: 12, borderRadius: radius.pill, overflow: 'hidden' },
  legend: { flexDirection: 'row', flexWrap: 'wrap', columnGap: space.lg, rowGap: space.xs },
  item: { flexDirection: 'row', alignItems: 'center', gap: space.xs },
  swatch: { width: 10, height: 10, borderRadius: radius.pill },
});
