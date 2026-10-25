import { StyleSheet } from 'react-native';

import { Icon, type IconName } from './icons';
import { Pressable } from './Pressable';
import { useTheme } from './theme';
import { HIT, radius } from './tokens';

export interface IconButtonProps {
  icon: IconName;
  /** Required: an icon has no visible text, so this is its accessible name. */
  label: string;
  onPress?: () => void;
  selected?: boolean;
  disabled?: boolean;
  testID?: string;
  /** The element's id on web (to move focus to it). */
  nativeID?: string;
}

export function IconButton({
  icon,
  label,
  onPress,
  selected,
  disabled,
  testID,
  nativeID,
}: IconButtonProps) {
  const { colors } = useTheme();
  return (
    <Pressable
      role="button"
      accessibilityLabel={label}
      accessibilityState={{ selected: selected ?? undefined, disabled: !!disabled }}
      aria-pressed={selected}
      disabled={disabled}
      onPress={onPress}
      testID={testID}
      nativeID={nativeID}
      hitSlop={4}
      style={({ hovered, pressed }) => [
        styles.base,
        (hovered || pressed) && !disabled ? { backgroundColor: colors.surfaceAlt } : null,
      ]}
    >
      <Icon name={icon} filled={selected} color={selected ? colors.accent : colors.textMuted} />
    </Pressable>
  );
}

const styles = StyleSheet.create({
  base: {
    width: HIT,
    height: HIT,
    borderRadius: radius.pill,
    alignItems: 'center',
    justifyContent: 'center',
  },
});
