import { StyleSheet, Text, useColorScheme, View } from 'react-native';

const palette = {
  light: { background: '#ffffff', text: '#11181c', muted: '#5f6b73' },
  dark: { background: '#0f1417', text: '#ecedee', muted: '#9ba1a6' },
} as const;

export function HomeScreen() {
  const colors = palette[useColorScheme() === 'dark' ? 'dark' : 'light'];

  return (
    <View style={[styles.container, { backgroundColor: colors.background }]}>
      <Text accessibilityRole="header" style={[styles.title, { color: colors.text }]}>
        FirstRead
      </Text>
      <Text style={[styles.tagline, { color: colors.muted }]}>
        Breaking news, ranked for you before anyone has clicked.
      </Text>
      <Text style={[styles.note, { color: colors.muted }]}>Live demo coming soon.</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, alignItems: 'center', justifyContent: 'center', padding: 24, gap: 12 },
  title: { fontSize: 40, fontWeight: '700' },
  tagline: { fontSize: 18, textAlign: 'center', maxWidth: 480 },
  note: { fontSize: 14 },
});
