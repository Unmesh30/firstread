// One live number on the System page: a label, the value, and a line of context.
// While loading it keeps its size (no layout shift) with skeleton bars.
import { StyleSheet, View } from 'react-native';

import { Skeleton } from './Skeleton';
import { Text } from './Text';
import { useTheme } from './theme';
import { radius, space } from './tokens';

export interface StatTileProps {
  label: string;
  value?: string;
  detail?: string;
  testID?: string;
}

export function StatTile({ label, value, detail, testID }: StatTileProps) {
  const { colors } = useTheme();
  return (
    <View
      role="group"
      aria-label={label}
      testID={testID}
      style={[styles.tile, { backgroundColor: colors.surface, borderColor: colors.border }]}
    >
      <Text variant="caption" tone="muted" weight="600">
        {label.toUpperCase()}
      </Text>
      {value == null ? (
        <View style={styles.loading}>
          <Skeleton width="60%" height={28} />
          <Skeleton width="80%" height={12} />
        </View>
      ) : (
        <>
          <Text variant="headline" serif>
            {value}
          </Text>
          {detail ? (
            <Text variant="small" tone="muted">
              {detail}
            </Text>
          ) : null}
        </>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  tile: {
    flexGrow: 1,
    flexBasis: 200,
    gap: space.xs,
    padding: space.lg,
    borderRadius: radius.md,
    borderWidth: StyleSheet.hairlineWidth,
  },
  loading: { gap: space.sm, paddingTop: space.xs },
});
