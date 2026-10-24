// A page: the theme background, a centered content column (≤ 720 px) with 16 px
// gutters, and the skip link's target (`main`). `scroll={false}` with `column={false}`
// leaves layout to the children (the feed's own full-width list).
import Head from 'expo-router/head';
import type { PropsWithChildren } from 'react';
import { ScrollView, StyleSheet, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

import { useTheme } from './theme';
import { layout, space } from './tokens';

/** The document title on web: "<title> · FirstRead" (WCAG 2.4.2). */
export function pageTitle(title?: string): string {
  return title ? `${title} · FirstRead` : 'FirstRead';
}

export function Screen({
  children,
  scroll = true,
  column: useColumn = true,
  title,
  testID,
}: PropsWithChildren<{ scroll?: boolean; column?: boolean; title?: string; testID?: string }>) {
  const { colors } = useTheme();
  const column = useColumn ? (
    <View nativeID="main" role="main" style={styles.column}>
      {children}
    </View>
  ) : (
    children
  );
  return (
    <SafeAreaView
      edges={['top', 'left', 'right']}
      style={[styles.fill, { backgroundColor: colors.background }]}
      testID={testID}
    >
      <Head>
        <title>{pageTitle(title)}</title>
      </Head>
      {scroll ? (
        <ScrollView contentContainerStyle={styles.scroll} keyboardShouldPersistTaps="handled">
          {column}
        </ScrollView>
      ) : (
        column
      )}
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  fill: { flex: 1 },
  scroll: { flexGrow: 1 },
  column: {
    flexGrow: 1,
    width: '100%',
    maxWidth: layout.maxContent,
    alignSelf: 'center',
    paddingHorizontal: layout.gutter,
    paddingVertical: space.xl,
    gap: space.lg,
  },
});
