// A modal confirmation (never window.confirm: browser dialogs block the page). On web
// react-native-web's Modal traps focus and restores it on close; Esc closes it.
import type { ReactNode } from 'react';
import { Modal, StyleSheet, View } from 'react-native';

import { Button } from './Button';
import { modalLabel } from './Sheet';
import { Text } from './Text';
import { useTheme } from './theme';
import { radius, space } from './tokens';

export interface DialogProps {
  visible: boolean;
  title: string;
  body?: ReactNode;
  confirmLabel: string;
  cancelLabel?: string;
  destructive?: boolean;
  busy?: boolean;
  onConfirm: () => void;
  onCancel: () => void;
}

export function Dialog({
  visible,
  title,
  body,
  confirmLabel,
  cancelLabel = 'Cancel',
  destructive,
  busy,
  onConfirm,
  onCancel,
}: DialogProps) {
  const { colors } = useTheme();
  return (
    <Modal
      visible={visible}
      transparent
      animationType="fade"
      onRequestClose={onCancel}
      {...modalLabel(title)}
    >
      <View style={[styles.backdrop, { backgroundColor: colors.overlay }]}>
        <View style={[styles.card, { backgroundColor: colors.surface }]}>
          <Text variant="title" serif role="heading" aria-level={2}>
            {title}
          </Text>
          {typeof body === 'string' ? <Text tone="muted">{body}</Text> : body}
          <View style={styles.actions}>
            <Button variant="secondary" label={cancelLabel} onPress={onCancel} disabled={busy} />
            <Button
              variant={destructive ? 'danger' : 'primary'}
              label={confirmLabel}
              onPress={onConfirm}
              loading={busy}
            />
          </View>
        </View>
      </View>
    </Modal>
  );
}

const styles = StyleSheet.create({
  backdrop: { flex: 1, alignItems: 'center', justifyContent: 'center', padding: space.lg },
  card: { width: '100%', maxWidth: 480, borderRadius: radius.lg, padding: space.xl, gap: space.lg },
  actions: { flexDirection: 'row', justifyContent: 'flex-end', flexWrap: 'wrap', gap: space.md },
});
