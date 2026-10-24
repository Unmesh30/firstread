// Renders the current toast at the bottom of the screen in a polite live region. The
// action button is a normal control, reachable with Tab. It takes focus only when asked
// (focusAction): a keyboard reader who hid a story from the menu, whose focus would
// otherwise fall to the top of the page, far from Undo.
import { useEffect } from 'react';
import { Platform, StyleSheet, View } from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';

import { subscribeLifecycle } from '@/lib/lifecycle';

import { Button } from './Button';
import { Text } from './Text';
import { useTheme } from './theme';
import { useToasts } from './toast';
import { layout, radius, space } from './tokens';

const ACTION_ID = 'toast-action';

export function ToastHost() {
  const { colors } = useTheme();
  const insets = useSafeAreaInsets();
  const toast = useToasts((s) => s.toasts[0]);
  const { act, pause, resume, dismiss } = useToasts.getState();

  useEffect(
    () =>
      subscribeLifecycle((state) =>
        state === 'hidden' ? useToasts.getState().hidden() : useToasts.getState().visible(),
      ),
    [],
  );

  const focusAction = toast?.focusAction === true;
  useEffect(() => {
    if (focusAction && Platform.OS === 'web') document.getElementById(ACTION_ID)?.focus();
  }, [focusAction]);

  return (
    <View pointerEvents="box-none" style={[styles.wrap, { bottom: insets.bottom + space.lg }]}>
      <View role="status" aria-live="polite" aria-atomic style={styles.region}>
        {toast ? (
          <View
            key={toast.id}
            testID="toast"
            onPointerEnter={() => pause(toast.id, 'hover')}
            onPointerLeave={() => resume(toast.id, 'hover')}
            style={[styles.toast, { backgroundColor: colors.text }]}
          >
            <Text variant="small" style={[styles.message, { color: colors.background }]}>
              {toast.message}
              {toast.actionLabel ? `. ${toast.actionLabel} available` : ''}
            </Text>
            {toast.actionLabel ? (
              <Button
                variant="secondary"
                label={toast.actionLabel}
                onPress={() => act(toast.id)}
                onFocus={() => pause(toast.id, 'focus')}
                onBlur={() => resume(toast.id, 'focus')}
                testID="toast-action"
                nativeID={ACTION_ID}
              />
            ) : (
              <Button variant="ghost" label="Dismiss" onPress={() => dismiss(toast.id)} />
            )}
          </View>
        ) : null}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  wrap: {
    position: 'absolute',
    left: 0,
    right: 0,
    alignItems: 'center',
    paddingHorizontal: layout.gutter,
  },
  region: { width: '100%', maxWidth: 560 },
  toast: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: space.md,
    paddingVertical: space.sm,
    paddingLeft: space.lg,
    paddingRight: space.sm,
    borderRadius: radius.md,
  },
  message: { flex: 1 },
});
