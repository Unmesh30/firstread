// A short list of exclusive options (the theme): a radio group, each option a radio
// with its checked state, laid out as a segmented row.
import { StyleSheet, View } from 'react-native';

import { Pressable } from './Pressable';
import { Text } from './Text';
import { useTheme } from './theme';
import { HIT, radius, space } from './tokens';

export interface ChoiceGroupProps<T extends string> {
  label: string;
  options: { value: T; label: string }[];
  value: T;
  onChange: (value: T) => void;
  testID?: string;
}

export function ChoiceGroup<T extends string>({
  label,
  options,
  value,
  onChange,
  testID,
}: ChoiceGroupProps<T>) {
  const { colors } = useTheme();
  return (
    <View
      role="radiogroup"
      aria-label={label}
      testID={testID}
      style={[styles.group, { borderColor: colors.borderStrong }]}
    >
      {options.map((o, i) => {
        const checked = o.value === value;
        return (
          <Pressable
            key={o.value}
            role="radio"
            accessibilityLabel={o.label}
            accessibilityState={{ checked }}
            aria-checked={checked}
            onPress={() => onChange(o.value)}
            testID={testID ? `${testID}-${o.value}` : undefined}
            style={({ hovered }) => [
              styles.option,
              i > 0 ? { borderLeftWidth: 1, borderLeftColor: colors.borderStrong } : null,
              {
                backgroundColor: checked
                  ? colors.accentSoft
                  : hovered
                    ? colors.surfaceAlt
                    : colors.surface,
              },
            ]}
          >
            <Text variant="small" weight={checked ? '600' : '500'}>
              {o.label}
            </Text>
          </Pressable>
        );
      })}
    </View>
  );
}

const styles = StyleSheet.create({
  group: {
    flexDirection: 'row',
    alignSelf: 'flex-start',
    borderWidth: 1,
    borderRadius: radius.md,
    overflow: 'hidden',
  },
  option: {
    minHeight: HIT,
    minWidth: 88,
    paddingHorizontal: space.lg,
    alignItems: 'center',
    justifyContent: 'center',
  },
});
