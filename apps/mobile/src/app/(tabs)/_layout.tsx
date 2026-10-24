import { Slot } from 'expo-router';
import { StyleSheet, View } from 'react-native';

import { NavBar, useNavPosition } from '@/features/NavBar';
import { useTheme } from '@/ui';

export default function SectionsLayout() {
  const { colors } = useTheme();
  const position = useNavPosition();
  return (
    <View style={[styles.fill, { backgroundColor: colors.background }]}>
      {position === 'top' ? <NavBar position="top" /> : null}
      <View style={styles.fill}>
        <Slot />
      </View>
      {position === 'bottom' ? <NavBar position="bottom" /> : null}
    </View>
  );
}

const styles = StyleSheet.create({ fill: { flex: 1 } });
