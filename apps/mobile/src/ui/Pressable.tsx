// Pressable with the app's focus ring (keyboard focus only) and hover state. On web,
// Space also presses checkboxes, radios and switches: react-native-web maps Space to a
// press only for buttons, but Space is the key that toggles these controls.
import { useState, type ReactNode } from 'react';
import {
  Platform,
  Pressable as RNPressable,
  type GestureResponderEvent,
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

const TOGGLE_ROLES = new Set(['checkbox', 'radio', 'switch']);

interface KeyEvent {
  key: string;
  repeat?: boolean;
  preventDefault(): void;
}

export function pressOnSpace(e: KeyEvent, p: Pick<PressableProps, 'onPress' | 'disabled'>) {
  if (e.key !== ' ') return;
  e.preventDefault(); // not a page scroll
  if (!e.repeat && !p.disabled) p.onPress?.(e as unknown as GestureResponderEvent);
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
  const spaceToggles =
    Platform.OS === 'web' && typeof rest.role === 'string' && TOGGLE_ROLES.has(rest.role);
  return (
    <RNPressable
      {...rest}
      {...(spaceToggles ? { onKeyDown: (e: KeyEvent) => pressOnSpace(e, rest) } : null)}
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
