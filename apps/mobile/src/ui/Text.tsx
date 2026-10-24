import { Text as RNText, type TextProps } from 'react-native';

import { useTheme } from './theme';
import { fonts, MAX_FONT_SCALE, type Colors, type TypeVariant, type as typeScale } from './tokens';

export type Tone =
  'default' | 'muted' | 'accent' | 'danger' | 'success' | 'warning' | 'info' | 'onAccent';

const TONES: Record<Tone, keyof Colors> = {
  default: 'text',
  muted: 'textMuted',
  accent: 'accent',
  danger: 'danger',
  success: 'success',
  warning: 'warning',
  info: 'info',
  onAccent: 'onAccent',
};

export interface AppTextProps extends TextProps {
  variant?: TypeVariant;
  tone?: Tone;
  /** The editorial serif (headlines and titles). */
  serif?: boolean;
  weight?: '400' | '500' | '600' | '700';
}

export function Text({
  variant = 'body',
  tone = 'default',
  serif,
  weight,
  style,
  ...rest
}: AppTextProps) {
  const { colors } = useTheme();
  return (
    <RNText
      maxFontSizeMultiplier={MAX_FONT_SCALE}
      {...rest}
      style={[
        typeScale[variant],
        { color: colors[TONES[tone]], fontFamily: serif ? fonts.headline : fonts.body },
        weight ? { fontWeight: weight } : null,
        style,
      ]}
    />
  );
}
