// "Skip to content" for keyboard users on web: off-screen until it has focus.
import { useState } from 'react';
import { Platform, StyleSheet } from 'react-native';

import { Text } from './Text';
import { useTheme } from './theme';
import { radius, space } from './tokens';

export function SkipLink() {
  const { colors } = useTheme();
  const [focused, setFocused] = useState(false);
  if (Platform.OS !== 'web') return null;
  // react-native-web renders Text with href as <a>, focusable like any link.
  const webProps = {
    href: '#main',
    onFocus: () => setFocused(true),
    onBlur: () => setFocused(false),
  } as object;
  return (
    <Text
      {...webProps}
      weight="600"
      style={[
        styles.base,
        { backgroundColor: colors.surface, color: colors.accent, borderColor: colors.focus },
        focused ? styles.shown : styles.hidden,
      ]}
    >
      Skip to content
    </Text>
  );
}

const styles = StyleSheet.create({
  base: {
    position: 'absolute',
    left: space.lg,
    zIndex: 10,
    padding: space.md,
    borderWidth: 2,
    borderRadius: radius.md,
  },
  hidden: { top: -200 },
  shown: { top: space.lg },
});
