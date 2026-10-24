import { ActivityIndicator, StyleSheet, View } from 'react-native';

import { Icon, type IconName } from './icons';
import { Pressable } from './Pressable';
import { Text } from './Text';
import { useTheme } from './theme';
import { HIT, radius, space, type Colors } from './tokens';

export type ButtonVariant = 'primary' | 'secondary' | 'ghost' | 'danger';

export interface ButtonProps {
  label: string;
  onPress?: () => void;
  variant?: ButtonVariant;
  icon?: IconName;
  disabled?: boolean;
  loading?: boolean;
  accessibilityLabel?: string;
  accessibilityHint?: string;
  testID?: string;
  onFocus?: () => void;
  onBlur?: () => void;
  /** The element's id on web (to move focus to it). */
  nativeID?: string;
}

function palette(c: Colors, v: ButtonVariant) {
  switch (v) {
    case 'primary':
      return { bg: c.accent, fg: c.onAccent, border: c.accent, tone: 'onAccent' as const };
    case 'danger':
      return { bg: c.surface, fg: c.danger, border: c.danger, tone: 'danger' as const };
    case 'secondary':
      return { bg: c.surface, fg: c.text, border: c.borderStrong, tone: 'default' as const };
    case 'ghost':
      return { bg: 'transparent', fg: c.accent, border: 'transparent', tone: 'accent' as const };
  }
}

export function Button({
  label,
  onPress,
  variant = 'primary',
  icon,
  disabled,
  loading,
  accessibilityLabel,
  accessibilityHint,
  testID,
  onFocus,
  onBlur,
  nativeID,
}: ButtonProps) {
  const { colors } = useTheme();
  const p = palette(colors, variant);
  const inactive = disabled || loading;
  return (
    <Pressable
      role="button"
      accessibilityLabel={accessibilityLabel ?? label}
      accessibilityHint={accessibilityHint}
      accessibilityState={{ disabled: !!inactive, busy: !!loading }}
      disabled={inactive}
      onPress={onPress}
      onFocus={onFocus}
      onBlur={onBlur}
      testID={testID}
      nativeID={nativeID}
      style={({ pressed, hovered }) => [
        styles.base,
        { backgroundColor: p.bg, borderColor: p.border },
        (pressed || hovered) && !inactive ? { opacity: 0.85 } : null,
        inactive ? { opacity: 0.5 } : null,
      ]}
    >
      <View style={styles.row}>
        {loading ? (
          <ActivityIndicator size="small" color={p.fg} />
        ) : icon ? (
          <Icon name={icon} color={p.fg} size={18} />
        ) : null}
        <Text variant="body" weight="600" tone={p.tone}>
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
    paddingVertical: space.sm,
    borderRadius: radius.md,
    borderWidth: 1,
    alignItems: 'center',
    justifyContent: 'center',
  },
  row: { flexDirection: 'row', alignItems: 'center', gap: space.sm },
});
