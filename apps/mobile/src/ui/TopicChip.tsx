import { StyleSheet, View } from 'react-native';

import { Icon } from './icons';
import { Pressable } from './Pressable';
import { Text } from './Text';
import { useTheme } from './theme';
import { HIT, radius, space } from './tokens';

export interface TopicChipProps {
  label: string;
  selected: boolean;
  onToggle: () => void;
  testID?: string;
}

/** A toggle: announced as a checkbox with its checked state. */
export function TopicChip({ label, selected, onToggle, testID }: TopicChipProps) {
  const { colors } = useTheme();
  return (
    <Pressable
      role="checkbox"
      accessibilityLabel={label}
      accessibilityState={{ checked: selected }}
      aria-checked={selected}
      onPress={onToggle}
      testID={testID}
      style={({ hovered }) => [
        styles.base,
        {
          backgroundColor: selected ? colors.accentSoft : colors.surface,
          borderColor: selected ? colors.accent : colors.borderStrong,
        },
        hovered && !selected ? { backgroundColor: colors.surfaceAlt } : null,
      ]}
    >
      <View style={styles.row}>
        {selected ? <Icon name="check" size={16} color={colors.accent} /> : null}
        <Text variant="small" weight={selected ? '600' : '500'}>
          {label}
        </Text>
      </View>
    </Pressable>
  );
}

const styles = StyleSheet.create({
  base: {
    minHeight: HIT,
    paddingHorizontal: space.lg,
    borderRadius: radius.pill,
    borderWidth: 1,
    justifyContent: 'center',
  },
  row: { flexDirection: 'row', alignItems: 'center', gap: space.xs },
});
