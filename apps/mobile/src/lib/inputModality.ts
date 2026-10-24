// Focus rings for keyboard users only (like CSS :focus-visible): the last input was a
// key press, not a pointer. Web only; native shows no rings (it has its own focus UI).
import { useSyncExternalStore } from 'react';
import { Platform } from 'react-native';

let keyboard = false;
const listeners = new Set<() => void>();
let installed = false;

function install() {
  if (installed || Platform.OS !== 'web' || typeof window === 'undefined') return;
  installed = true;
  const set = (v: boolean) => () => {
    if (keyboard !== v) {
      keyboard = v;
      listeners.forEach((l) => l());
    }
  };
  window.addEventListener('keydown', set(true), true);
  window.addEventListener('pointerdown', set(false), true);
}

/** The last input was a key press (read once, e.g. when deciding where focus goes). */
export function isKeyboardModality(): boolean {
  install();
  return keyboard;
}

export function useKeyboardModality(): boolean {
  return useSyncExternalStore(
    (l) => {
      install();
      listeners.add(l);
      return () => listeners.delete(l);
    },
    () => keyboard,
    () => false,
  );
}
