import { layout } from '@/ui/tokens';

import { useWindowWidth } from './useHydrated';

export type Breakpoint = 'compact' | 'medium' | 'expanded';

export function breakpointFor(width: number): Breakpoint {
  if (width >= layout.expanded) return 'expanded';
  if (width >= layout.compact) return 'medium';
  return 'compact';
}

export function useBreakpoint(): { breakpoint: Breakpoint; width: number; wideNav: boolean } {
  // Hydration-safe: the static HTML's (narrowest) layout first, then the real one.
  const width = useWindowWidth();
  return { breakpoint: breakpointFor(width), width, wideNav: width >= layout.nav };
}
