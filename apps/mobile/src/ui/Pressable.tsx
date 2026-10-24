// Pressable with the app's focus ring (keyboard focus only) and hover state.
import { useState, type ReactNode } from 'react';
import {
  Pressable as RNPressable,
  type PressableProps,
  type StyleProp,
  type ViewStyle,
} from 'react-native';

import { useKeyboardModality } from '@/lib/inputModality';

import { useTheme } from './theme';

export interface InteractionState {
  pressed: boolean;
  hovered: boolean;
  focused: boolean;
}

export interface AppPressableProps extends Omit<PressableProps, 'style' | 'children'> {
  style?: StyleProp<ViewStyle> | ((s: InteractionState) => StyleProp<ViewStyle>);
  children?: ReactNode | ((s: InteractionState) => ReactNode);
}

export function Pressable({
  style,
  children,
  onFocus,
  onBlur,
  onHoverIn,
  onHoverOut,
  ...rest
}: AppPressableProps) {
  const { colors } = useTheme();
  const keyboard = useKeyboardModality();
  const [focused, setFocused] = useState(false);
  const [hovered, setHovered] = useState(false);
  return (
    <RNPressable
      {...rest}
      onFocus={(e) => {
        setFocused(true);
        onFocus?.(e);
      }}
      onBlur={(e) => {
        setFocused(false);
        onBlur?.(e);
      }}
      onHoverIn={(e) => {
        setHovered(true);
        onHoverIn?.(e);
      }}
      onHoverOut={(e) => {
        setHovered(false);
        onHoverOut?.(e);
      }}
      style={({ pressed }) => {
        const s = { pressed, hovered, focused };
        return [
          typeof style === 'function' ? style(s) : style,
          focused && keyboard
            ? {
                outlineColor: colors.focus,
                outlineStyle: 'solid',
                outlineWidth: 2,
                outlineOffset: 2,
              }
            : null,
        ] as StyleProp<ViewStyle>;
      }}
    >
      {({ pressed }) => {
        const s = { pressed, hovered, focused };
        return typeof children === 'function' ? children(s) : children;
      }}
    </RNPressable>
  );
}
