// Foreground/background for the whole app: Page Visibility and pagehide on web,
// AppState on native. Impressions pause, undo timers pause, and the event queue
// flushes on 'hidden' (a closing tab fires pagehide; a switched tab, visibilitychange).
import { useSyncExternalStore } from 'react';
import { AppState, Platform } from 'react-native';

export type Lifecycle = 'active' | 'hidden';

export function currentLifecycle(): Lifecycle {
  if (Platform.OS === 'web') {
    return typeof document !== 'undefined' && document.visibilityState === 'hidden'
      ? 'hidden'
      : 'active';
  }
  return AppState.currentState === 'active' || AppState.currentState == null ? 'active' : 'hidden';
}

/** Calls `listener` on every change; 'hidden' may repeat (visibilitychange, then pagehide). */
export function subscribeLifecycle(listener: (state: Lifecycle) => void): () => void {
  if (Platform.OS === 'web') {
    if (typeof document === 'undefined') return () => undefined;
    const onVisibility = () => listener(currentLifecycle());
    const onPageHide = () => listener('hidden');
    document.addEventListener('visibilitychange', onVisibility);
    window.addEventListener('pagehide', onPageHide);
    return () => {
      document.removeEventListener('visibilitychange', onVisibility);
      window.removeEventListener('pagehide', onPageHide);
    };
  }
  const sub = AppState.addEventListener('change', (s) =>
    listener(s === 'active' ? 'active' : 'hidden'),
  );
  return () => sub.remove();
}

export function useForeground(): boolean {
  return useSyncExternalStore(
    subscribeLifecycle,
    () => currentLifecycle() === 'active',
    () => true, // static rendering
  );
}
