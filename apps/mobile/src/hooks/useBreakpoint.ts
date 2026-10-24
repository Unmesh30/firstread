import { useWindowDimensions } from 'react-native';

import { layout } from '@/ui/tokens';

export type Breakpoint = 'compact' | 'medium' | 'expanded';

export function breakpointFor(width: number): Breakpoint {
  if (width >= layout.expanded) return 'expanded';
  if (width >= layout.compact) return 'medium';
  return 'compact';
}

export function useBreakpoint(): { breakpoint: Breakpoint; width: number; wideNav: boolean } {
  const { width } = useWindowDimensions();
  return { breakpoint: breakpointFor(width), width, wideNav: width >= layout.nav };
}
