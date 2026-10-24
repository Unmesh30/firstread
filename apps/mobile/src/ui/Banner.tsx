import { StyleSheet, View } from 'react-native';

import { Button } from './Button';
import { Icon, type IconName } from './icons';
import { Text, type Tone } from './Text';
import { useTheme } from './theme';
import { radius, space, type Colors } from './tokens';

export type BannerTone = 'info' | 'warning' | 'danger' | 'offline';

const STYLE: Record<
  BannerTone,
  { bg: keyof Colors; fg: keyof Colors; tone: Tone; icon: IconName }
> = {
  info: { bg: 'infoSoft', fg: 'info', tone: 'info', icon: 'info' },
  warning: { bg: 'warningSoft', fg: 'warning', tone: 'warning', icon: 'info' },
  danger: { bg: 'dangerSoft', fg: 'danger', tone: 'danger', icon: 'info' },
  offline: { bg: 'surfaceAlt', fg: 'text', tone: 'default', icon: 'offline' },
};

export interface BannerProps {
  tone?: BannerTone;
  message: string;
  action?: { label: string; onPress: () => void };
  testID?: string;
}

/** An inline status message, announced politely (danger: assertively, as an alert). */
export function Banner({ tone = 'info', message, action, testID }: BannerProps) {
  const { colors } = useTheme();
  const s = STYLE[tone];
  return (
    <View
      role={tone === 'danger' ? 'alert' : 'status'}
      aria-live={tone === 'danger' ? 'assertive' : 'polite'}
      testID={testID}
      style={[styles.base, { backgroundColor: colors[s.bg] }]}
    >
      <Icon name={s.icon} color={colors[s.fg]} />
      <Text variant="small" tone={s.tone} style={styles.message}>
        {message}
      </Text>
      {action ? <Button variant="ghost" label={action.label} onPress={action.onPress} /> : null}
    </View>
  );
}

const styles = StyleSheet.create({
  base: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: space.md,
    paddingHorizontal: space.lg,
    paddingVertical: space.sm,
    borderRadius: radius.md,
  },
  message: { flex: 1 },
});
