// Light/dark theme: the system setting unless the reader picks one in Account. The
// preference is loaded in an effect (never at import time) and stored on this device.
// While a static page hydrates (web), colors are CSS variables that +html.tsx resolves
// for the system scheme or the stored choice, so the first paint is already right.
import { createContext, useContext, useEffect, useMemo, type PropsWithChildren } from 'react';
import { Platform } from 'react-native';
import { create } from 'zustand';

import { useHydrated, useSystemScheme } from '@/hooks/useHydrated';
import { kv } from '@/storage/kv';

import { colors, colorVars, type Colors, type ThemeName } from './tokens';

export type ThemePreference = 'system' | ThemeName;
/** kv key: "fr:theme" in localStorage, read by +html.tsx's pre-paint script. */
export const THEME_KEY = 'theme';

interface PrefsState {
  theme: ThemePreference;
  setTheme(t: ThemePreference): void;
}

export const usePrefs = create<PrefsState>((set) => ({
  theme: 'system',
  setTheme: (theme) => {
    set({ theme });
    void kv.set(THEME_KEY, theme);
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
  // The static HTML's scheme while hydrating (null: the CSS decides), then the system's.
  const system = useSystemScheme();
  const hydrated = useHydrated();
  const pref = usePrefs((s) => s.theme);
  useEffect(() => {
    void kv.get(THEME_KEY).then((v) => {
      if (v === 'light' || v === 'dark' || v === 'system') usePrefs.setState({ theme: v });
    });
  }, []);
  const name = resolveTheme(pref, system);
  const vars = Platform.OS === 'web' && !hydrated;
  useEffect(() => {
    if (Platform.OS !== 'web' || !hydrated) return;
    // The variables and html/body now follow the resolved theme (the reader's choice
    // included), not just the media query.
    document.documentElement.setAttribute('data-theme', name);
  }, [name, hydrated]);
  const value = useMemo(() => ({ name, colors: vars ? colorVars : colors[name] }), [name, vars]);
  return <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>;
}

export function useTheme(): Theme {
  return useContext(ThemeContext);
}
