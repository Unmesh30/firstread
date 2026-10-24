// react-native-web's Modal handles Esc (and traps focus, and becomes role="dialog")
// only once its open animation has ended, so an Esc pressed during the fade-in did
// nothing. This handles Esc from the moment the modal is visible until its onShow,
// in the capture phase, so a modal underneath (still the active one) doesn't close too.
import { useCallback, useEffect, useRef } from 'react';
import { Platform } from 'react-native';

/** Returns the Modal's onShow. */
export function useEscapeWhileOpening(visible: boolean, onClose: () => void): () => void {
  const shown = useRef(false);
  const close = useRef(onClose);
  useEffect(() => {
    close.current = onClose;
  });
  useEffect(() => {
    if (Platform.OS !== 'web' || !visible) return;
    shown.current = false;
    const onKey = (e: KeyboardEvent) => {
      if (shown.current || e.key !== 'Escape') return;
      e.stopPropagation();
      close.current();
    };
    window.addEventListener('keyup', onKey, true);
    return () => window.removeEventListener('keyup', onKey, true);
  }, [visible]);
  return useCallback(() => {
    shown.current = true;
  }, []);
}
