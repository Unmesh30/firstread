// The app's sections: a bottom bar on narrow screens, a top bar from 768 px.
// Real links (role link, aria-current on the active one), in visual order.
import { Link, usePathname, type Href } from 'expo-router';
import { StyleSheet, View } from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';

import { useBreakpoint } from '@/hooks/useBreakpoint';
import { Icon, Text, useTheme, type IconName } from '@/ui';
import { shownAt } from '@/ui/responsive';
import { HIT, layout, space } from '@/ui/tokens';

export const SECTIONS: { href: Href; path: string; label: string; icon: IconName }[] = [
  { href: '/feed', path: '/feed', label: 'For You', icon: 'news' },
  { href: '/system', path: '/system', label: 'System', icon: 'chart' },
  { href: '/account', path: '/account', label: 'Account', icon: 'user' },
];

export function NavBar({ position }: { position: 'top' | 'bottom' }) {
  const { colors } = useTheme();
  const pathname = usePathname();
  const insets = useSafeAreaInsets();
  const top = position === 'top';
  return (
    <View
      role="navigation"
      aria-label="Sections"
      {...shownAt(top ? 'nav-top' : 'nav-bottom')}
      style={[
        styles.bar,
        { backgroundColor: colors.surface, borderColor: colors.border },
        top
          ? { borderBottomWidth: StyleSheet.hairlineWidth, paddingTop: insets.top }
          : { borderTopWidth: StyleSheet.hairlineWidth, paddingBottom: insets.bottom },
      ]}
    >
      <View style={[styles.inner, top ? styles.innerTop : null]}>
        {top ? (
          <Text variant="title" serif style={styles.brand}>
            FirstRead
          </Text>
        ) : null}
        <View style={[styles.items, top ? null : styles.itemsBottom]}>
          {SECTIONS.map((s) => {
            const active = pathname === s.path;
            const color = active ? colors.accent : colors.textMuted;
            return (
              <Link
                key={s.path}
                href={s.href}
                aria-current={active ? 'page' : undefined}
                style={[styles.item, top ? styles.itemTop : null]}
              >
                <View style={[styles.itemInner, top ? styles.itemInnerTop : null]}>
                  <Icon name={s.icon} color={color} />
                  <Text
                    variant={top ? 'small' : 'caption'}
                    weight={active ? '600' : '500'}
                    style={{ color }}
                  >
                    {s.label}
                  </Text>
                </View>
              </Link>
            );
          })}
        </View>
      </View>
    </View>
  );
}

export function useNavPosition(): 'top' | 'bottom' {
  return useBreakpoint().wideNav ? 'top' : 'bottom';
}

const styles = StyleSheet.create({
  bar: { width: '100%' },
  inner: { width: '100%', alignSelf: 'center' },
  innerTop: {
    maxWidth: layout.expanded,
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: layout.gutter,
  },
  brand: { marginRight: 'auto' },
  items: { flexDirection: 'row' },
  itemsBottom: { justifyContent: 'space-around' },
  item: { minHeight: HIT + 8, justifyContent: 'center', flex: 1 },
  itemTop: { flex: 0, paddingHorizontal: space.md },
  itemInner: { alignItems: 'center', gap: 2, paddingVertical: space.xs },
  itemInnerTop: { flexDirection: 'row', gap: space.sm },
});
