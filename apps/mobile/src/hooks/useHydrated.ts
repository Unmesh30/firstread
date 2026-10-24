// Static rendering pre-renders every page in Node, where there is no window, no color
// scheme and no portal target. While React hydrates that HTML, the client's first
// render must produce the same markup, or React throws the page away (error #418) and
// keeps the server's attributes (a dark-mode reader would stay on light colors).
//
// useSyncExternalStore's server snapshot is exactly that: React uses it for the
// hydration render and re-renders at once with the real value. Native apps don't
// hydrate, so they get the real value on the first render.
import { useSyncExternalStore } from 'react';
import { Appearance, Dimensions } from 'react-native';

const never = () => () => {};

/** False while hydrating a statically rendered page, true from the next render on. */
export function useHydrated(): boolean {
  return useSyncExternalStore(
    never,
    () => true,
    () => false,
  );
}

/** The width static pages are rendered at (Node has no window): the narrowest layout. */
export const SERVER_WIDTH = 0;

function subscribeDimensions(listener: () => void) {
  const sub = Dimensions.addEventListener('change', listener);
  return () => sub.remove();
}

/** The window width; SERVER_WIDTH while hydrating. */
export function useWindowWidth(): number {
  return useSyncExternalStore(
    subscribeDimensions,
    () => Dimensions.get('window').width,
    () => SERVER_WIDTH,
  );
}

function subscribeScheme(listener: () => void) {
  const sub = Appearance.addChangeListener(listener);
  return () => sub.remove();
}

/** The system color scheme; null (light) while hydrating, like the static HTML. */
export function useSystemScheme(): 'light' | 'dark' | null {
  return useSyncExternalStore(
    subscribeScheme,
    () => {
      const s = Appearance.getColorScheme();
      return s === 'dark' || s === 'light' ? s : null;
    },
    () => null,
  );
}
