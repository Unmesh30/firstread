// A sheet over the current screen: from the bottom on narrow screens, centered on wide
// ones. React Native's Modal (react-native-web on web) is the dialog: it moves focus to
// the first focusable control (Close), traps it and gives it back on close; Esc and the
// backdrop close it. No slide when the reader asked for reduced motion.
import type { ReactNode } from 'react';
import { Modal, ScrollView, StyleSheet, View } from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';

import { useBreakpoint } from '@/hooks/useBreakpoint';
import { useReducedMotion } from '@/hooks/useReducedMotion';

import { IconButton } from './IconButton';
import { Text } from './Text';
import { useTheme } from './theme';
import { layout, radius, space } from './tokens';

export interface SheetProps {
  visible: boolean;
  title: string;
  onClose: () => void;
  children: ReactNode;
  /** Heading level of the title: 1 when the sheet is the page (a deep link). */
  level?: 1 | 2;
  testID?: string;
}

export function Sheet({ visible, title, onClose, children, level = 2, testID }: SheetProps) {
  const { colors } = useTheme();
  const insets = useSafeAreaInsets();
  const reduced = useReducedMotion();
  const wide = useBreakpoint().breakpoint !== 'compact';
  return (
    <Modal
      visible={visible}
      transparent
      animationType={reduced ? 'none' : wide ? 'fade' : 'slide'}
      onRequestClose={onClose}
      {...modalLabel(title)}
    >
      <View style={[styles.backdrop, wide ? styles.center : styles.bottom]}>
        {/* Not a control (Esc and Close are): a responder, so it can't take focus. */}
        <View
          aria-hidden
          onStartShouldSetResponder={() => true}
          onResponderRelease={onClose}
          style={[StyleSheet.absoluteFill, { backgroundColor: colors.overlay }]}
        />
        <View
          testID={testID}
          style={[
            styles.sheet,
            wide ? styles.sheetWide : styles.sheetNarrow,
            { backgroundColor: colors.surface, paddingBottom: space.xl + insets.bottom },
          ]}
        >
          <View style={styles.header}>
            <Text variant="title" serif role="heading" aria-level={level} style={styles.title}>
              {title}
            </Text>
            <IconButton icon="close" label="Close" onPress={onClose} testID="sheet-close" />
          </View>
          <ScrollView contentContainerStyle={styles.content}>{children}</ScrollView>
        </View>
      </View>
    </Modal>
  );
}

/** react-native-web gives Modal's other props to its role="dialog" element. */
export function modalLabel(label: string): object {
  return { 'aria-label': label };
}

const styles = StyleSheet.create({
  backdrop: { flex: 1 },
  bottom: { justifyContent: 'flex-end' },
  center: { justifyContent: 'center', alignItems: 'center', padding: space.lg },
  sheet: { maxHeight: '90%', paddingTop: space.md, paddingHorizontal: space.xl },
  sheetNarrow: { borderTopLeftRadius: radius.lg, borderTopRightRadius: radius.lg },
  sheetWide: { width: '100%', maxWidth: layout.maxContent - 160, borderRadius: radius.lg },
  header: { flexDirection: 'row', alignItems: 'center', gap: space.sm },
  title: { flex: 1 },
  content: { gap: space.lg, paddingTop: space.sm },
});
