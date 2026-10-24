import { Slot } from 'expo-router';
import { StyleSheet, View } from 'react-native';

import { NavBar, useNavPosition } from '@/features/NavBar';
import { useHydrated } from '@/hooks/useHydrated';
import { useTheme } from '@/ui';

export default function SectionsLayout() {
  const { colors } = useTheme();
  const position = useNavPosition();
  // The static HTML has both bars and a media query shows the right one before any
  // JavaScript runs (+html.tsx); once hydrated, only the bar for this width is kept.
  const both = !useHydrated();
  return (
    <View style={[styles.fill, { backgroundColor: colors.background }]}>
      {both || position === 'top' ? <NavBar position="top" /> : null}
      <View style={styles.fill}>
        <Slot />
      </View>
      {both || position === 'bottom' ? <NavBar position="bottom" /> : null}
    </View>
  );
}

const styles = StyleSheet.create({ fill: { flex: 1 } });
