// Width-dependent parts of the layout before JavaScript runs. Static HTML is rendered at
// no particular width, so a view that exists only at some widths (the top or bottom
// bar, the feed's side panel) is rendered in the static HTML and tagged; +html.tsx's
// media queries show it only at its widths. After hydration the app renders just the
// views for the real width (useBreakpoint), and the tags keep agreeing with it.
import { Platform } from 'react-native';

import { layout } from './tokens';

export type ShownAt = 'nav-top' | 'nav-bottom' | 'expanded';

/** Props that tag a view with the widths it belongs to (data-shown on web). */
export function shownAt(at: ShownAt): object {
  return Platform.OS === 'web' ? { dataSet: { shown: at } } : {};
}

const below = (px: number) => `(max-width: ${px - 0.02}px)`;

export const responsiveCss = `
@media ${below(layout.nav)} { [data-shown="nav-top"] { display: none !important; } }
@media (min-width: ${layout.nav}px) { [data-shown="nav-bottom"] { display: none !important; } }
@media ${below(layout.expanded)} { [data-shown="expanded"] { display: none !important; } }
`;
