import { StyleSheet, View } from 'react-native';

import { useTheme } from './theme';

export function Divider() {
  const { colors } = useTheme();
  return (
    <View
      aria-hidden
      style={{ height: StyleSheet.hairlineWidth, backgroundColor: colors.border }}
    />
  );
}
