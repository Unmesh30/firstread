import { SourceSerif4_600SemiBold } from '@expo-google-fonts/source-serif-4/600SemiBold';
import { FontDisplay, useFonts } from 'expo-font';
import { Stack } from 'expo-router';
import { StatusBar } from 'expo-status-bar';

import { AppProviders } from '@/providers/AppProviders';
import { SkipLink, ToastHost, useTheme } from '@/ui';
import { HEADLINE_FONT } from '@/ui/tokens';

export const unstable_settings = {
  // A deep link to /why opens over the feed, not over nothing.
  anchor: '(tabs)',
};

function Navigator() {
  const { name, colors } = useTheme();
  return (
    <>
      <SkipLink />
      <Stack
        screenOptions={{ headerShown: false, contentStyle: { backgroundColor: colors.background } }}
      >
        {/* The route draws its own sheet and backdrop over the feed. */}
        <Stack.Screen
          name="why"
          options={{
            presentation: 'transparentModal',
            animation: 'none',
            contentStyle: { backgroundColor: 'transparent' },
          }}
        />
      </Stack>
      <ToastHost />
      <StatusBar style={name === 'dark' ? 'light' : 'dark'} />
    </>
  );
}

export default function RootLayout() {
  // One headline weight, never awaited: text renders in the fallback serif until it loads
  // (font-display: swap on web), so the font can't block the first render.
  useFonts({ [HEADLINE_FONT]: { uri: SourceSerif4_600SemiBold, display: FontDisplay.SWAP } });
  return (
    <AppProviders>
      <Navigator />
    </AppProviders>
  );
}
