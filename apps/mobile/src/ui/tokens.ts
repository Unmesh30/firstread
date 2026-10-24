// Design tokens: the only place colors, spacing, radii and type sizes are defined.
// Every text/background pair used by the components is listed in `contrastPairs` and
// checked against WCAG 2.2 by __tests__/contrast.test.ts, in both themes.
import { Platform } from 'react-native';

export type ThemeName = 'light' | 'dark';

const light = {
  background: '#FBFAF7', // paper
  surface: '#FFFFFF',
  surfaceAlt: '#F2EFE8',
  text: '#1B1A17', // ink
  textMuted: '#5C5850',
  border: '#E2DED5', // hairlines (decorative)
  borderStrong: '#857F73', // control outlines (≥ 3:1)
  accent: '#A8221B',
  onAccent: '#FFFFFF',
  accentSoft: '#F8E7E5',
  focus: '#1A5FB4',
  info: '#1F4F8F',
  infoSoft: '#E8F0FA',
  success: '#1E6B3A',
  warning: '#7A4F00',
  warningSoft: '#FBF1DC',
  danger: '#A8221B',
  dangerSoft: '#F8E7E5',
  skeleton: '#E9E5DC',
  overlay: 'rgba(27, 26, 23, 0.45)',
};

const dark: typeof light = {
  background: '#121110',
  surface: '#1C1B19',
  surfaceAlt: '#262421',
  text: '#F2EFE8',
  textMuted: '#ABA597',
  border: '#33302C',
  borderStrong: '#7E786C',
  accent: '#F2867D',
  onAccent: '#121110',
  accentSoft: '#3A2220',
  focus: '#8AB4F8',
  info: '#9CC2F5',
  infoSoft: '#1B2636',
  success: '#7CCB95',
  warning: '#E8C06A',
  warningSoft: '#33290F',
  danger: '#F2867D',
  dangerSoft: '#3A2220',
  skeleton: '#2A2825',
  overlay: 'rgba(0, 0, 0, 0.6)',
};

export const colors: Record<ThemeName, typeof light> = { light, dark };
export type Colors = typeof light;

/** Text (≥ 4.5:1) and UI-component (≥ 3:1) pairs the components actually use. */
export const contrastPairs: { fg: keyof Colors; bg: keyof Colors; min: 4.5 | 3 }[] = [
  { fg: 'text', bg: 'background', min: 4.5 },
  { fg: 'text', bg: 'surface', min: 4.5 },
  { fg: 'text', bg: 'surfaceAlt', min: 4.5 },
  { fg: 'textMuted', bg: 'background', min: 4.5 },
  { fg: 'textMuted', bg: 'surface', min: 4.5 },
  { fg: 'textMuted', bg: 'surfaceAlt', min: 4.5 },
  { fg: 'accent', bg: 'background', min: 4.5 },
  { fg: 'accent', bg: 'surface', min: 4.5 },
  { fg: 'onAccent', bg: 'accent', min: 4.5 },
  { fg: 'text', bg: 'accentSoft', min: 4.5 },
  { fg: 'info', bg: 'infoSoft', min: 4.5 },
  { fg: 'text', bg: 'infoSoft', min: 4.5 },
  { fg: 'warning', bg: 'warningSoft', min: 4.5 },
  { fg: 'text', bg: 'warningSoft', min: 4.5 },
  { fg: 'danger', bg: 'dangerSoft', min: 4.5 },
  { fg: 'success', bg: 'surface', min: 4.5 },
  { fg: 'borderStrong', bg: 'background', min: 3 },
  { fg: 'borderStrong', bg: 'surface', min: 3 },
  { fg: 'focus', bg: 'background', min: 3 },
  { fg: 'focus', bg: 'surface', min: 3 },
];

/** 4-point spacing scale. */
export const space = { xs: 4, sm: 8, md: 12, lg: 16, xl: 24, xxl: 32, xxxl: 48 } as const;
export const radius = { sm: 4, md: 8, lg: 16, pill: 999 } as const;

/** The one bundled face (Source Serif 4 SemiBold, OFL), and its fallbacks until it loads. */
export const HEADLINE_FONT = 'SourceSerif4_600SemiBold';
export const fonts = {
  headline: Platform.select({
    web: `${HEADLINE_FONT}, Georgia, 'Times New Roman', serif`,
    // Native takes one family name; before the font loads the system draws its default.
    default: HEADLINE_FONT,
  }) as string,
  body: Platform.select({
    web: "system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif",
    default: undefined,
  }),
};

export const type = {
  caption: { fontSize: 12, lineHeight: 16 },
  small: { fontSize: 14, lineHeight: 20 },
  body: { fontSize: 16, lineHeight: 24 },
  lead: { fontSize: 18, lineHeight: 26 },
  title: { fontSize: 22, lineHeight: 28 },
  headline: { fontSize: 28, lineHeight: 34 },
  display: { fontSize: 36, lineHeight: 42 },
} as const;
export type TypeVariant = keyof typeof type;

/** Content column and breakpoints (px). */
export const layout = { maxContent: 720, gutter: 16, compact: 600, expanded: 1024, nav: 768 };

/** Minimum touch target (WCAG 2.5.8 asks 24; platform guidelines 44). */
export const HIT = 44;

/** Native text can scale up to 2x; layouts never fix text heights. */
export const MAX_FONT_SCALE = 2;
