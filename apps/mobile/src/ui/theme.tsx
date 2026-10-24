// Light/dark theme: the system setting unless the reader picks one in Account. The
// preference is loaded in an effect (never at import time) and stored on this device.
import { createContext, useContext, useEffect, useMemo, type PropsWithChildren } from 'react';
import { Platform } from 'react-native';
import { create } from 'zustand';

import { useHydrated, useSystemScheme } from '@/hooks/useHydrated';
import { kv } from '@/storage/kv';

import { colors, type Colors, type ThemeName } from './tokens';

export type ThemePreference = 'system' | ThemeName;
const KEY = 'theme';

interface PrefsState {
  theme: ThemePreference;
  setTheme(t: ThemePreference): void;
}

export const usePrefs = create<PrefsState>((set) => ({
  theme: 'system',
  setTheme: (theme) => {
    set({ theme });
    void kv.set(KEY, theme);
  },
}));

export interface Theme {
  name: ThemeName;
  colors: Colors;
}

const ThemeContext = createContext<Theme>({ name: 'light', colors: colors.light });

export function resolveTheme(pref: ThemePreference, system: string | null | undefined): ThemeName {
  if (pref !== 'system') return pref;
  return system === 'dark' ? 'dark' : 'light';
}

export function ThemeProvider({ children }: PropsWithChildren) {
  // Light while hydrating (as the static HTML was rendered), then the system's.
  const system = useSystemScheme();
  const hydrated = useHydrated();
  const pref = usePrefs((s) => s.theme);
  useEffect(() => {
    void kv.get(KEY).then((v) => {
      if (v === 'light' || v === 'dark' || v === 'system') usePrefs.setState({ theme: v });
    });
  }, []);
  const name = resolveTheme(pref, system);
  useEffect(() => {
    // Until hydrated, +html.tsx's media query already paints the system theme.
    if (Platform.OS !== 'web' || !hydrated) return;
    // +html.tsx paints the system theme before JS loads; follow the reader's choice after.
    document.documentElement.style.colorScheme = name;
    document.body.style.backgroundColor = colors[name].background;
  }, [name, hydrated]);
  const value = useMemo(() => ({ name, colors: colors[name] }), [name]);
  return <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>;
}

export function useTheme(): Theme {
  return useContext(ThemeContext);
}
