// PR 4d: the theme before JavaScript (CSS variables and the stored choice), the share
// card, Space on toggles, middle-clicks, and focusing a toast's action. The browser
// behavior itself is checked end to end in e2e/polish.spec.ts.
import { render, screen } from '@testing-library/react-native';
import { renderToString } from 'react-dom/server';
import { Platform, Text } from 'react-native';

import { env } from '@/lib/env';
import { OG_IMAGE, SITE_DESCRIPTION, THEME_SCRIPT, THEME_STORAGE_KEY } from '@/lib/site';
import { ThemeProvider, useTheme } from '@/ui';
import { webLinkProps } from '@/ui/ArticleCard';
import { pressOnSpace } from '@/ui/Pressable';
import { THEME_KEY } from '@/ui/theme';
import { useToasts } from '@/ui/toast';
import { colors, colorVarDeclarations, colorVars } from '@/ui/tokens';

function onWeb() {
  jest.replaceProperty(Platform, 'OS', 'web');
}
afterEach(() => jest.restoreAllMocks());

describe('the theme before JavaScript', () => {
  it('has one CSS variable per color token, declared for both themes', () => {
    for (const key of Object.keys(colors.light) as (keyof typeof colors.light)[]) {
      expect(colorVars[key]).toBe(`var(--fr-${key})`);
      expect(colorVarDeclarations('light')).toContain(`--fr-${key}: ${colors.light[key]};`);
      expect(colorVarDeclarations('dark')).toContain(`--fr-${key}: ${colors.dark[key]};`);
    }
  });

  function Probe() {
    const { colors: c } = useTheme();
    return <Text>{c.background}</Text>;
  }

  it('a static web page is styled with the variables', () => {
    onWeb();
    expect(
      renderToString(
        <ThemeProvider>
          <Probe />
        </ThemeProvider>,
      ),
    ).toContain('var(--fr-background)');
  });

  it('native renders real colors from the first render', async () => {
    await render(
      <ThemeProvider>
        <Probe />
      </ThemeProvider>,
    );
    expect(screen.getByText(colors.light.background)).toBeTruthy();
  });

  it('the pre-paint script reads the key the theme provider writes', () => {
    expect(THEME_STORAGE_KEY).toBe(`fr:${THEME_KEY}`);
    const run = (stored: string | null, throws = false) => {
      const attrs: Record<string, string> = {};
      const localStorage = {
        getItem: (k: string) => {
          if (throws) throw new Error('blocked');
          return k === THEME_STORAGE_KEY ? stored : null;
        },
      };
      const document = {
        documentElement: { setAttribute: (k: string, v: string) => (attrs[k] = v) },
      };
      new Function('localStorage', 'document', THEME_SCRIPT)(localStorage, document);
      return attrs['data-theme'];
    };
    expect(run('dark')).toBe('dark');
    expect(run('light')).toBe('light');
    expect(run('system')).toBeUndefined(); // the media query decides
    expect(run('<script>')).toBeUndefined();
    expect(run(null, true)).toBeUndefined(); // blocked storage never breaks the page
  });
});

describe('the share card', () => {
  it('points at an absolute 1200×630 image under the site URL', () => {
    expect(OG_IMAGE).toMatchObject({ width: 1200, height: 630 });
    expect(OG_IMAGE.url).toBe(`${env.siteUrl}/og-image.png`);
    expect(OG_IMAGE.url).toMatch(/^https:\/\//);
    expect(OG_IMAGE.alt.length).toBeGreaterThan(20);
    expect(SITE_DESCRIPTION).toMatch(/before anyone has clicked/);
  });
});

describe('keyboard and pointer details on web', () => {
  const key = (k: string, repeat = false) => ({ key: k, repeat, preventDefault: jest.fn() });

  it('Space presses a toggle (react-native-web maps it only for buttons)', () => {
    const onPress = jest.fn();
    const space = key(' ');
    pressOnSpace(space, { onPress });
    expect(onPress).toHaveBeenCalledTimes(1);
    expect(space.preventDefault).toHaveBeenCalled(); // no page scroll
    pressOnSpace(key(' ', true), { onPress }); // a held key doesn't repeat
    pressOnSpace(key('a'), { onPress });
    pressOnSpace(key(' '), { onPress, disabled: true });
    expect(onPress).toHaveBeenCalledTimes(1);
  });

  it('a middle-click on a story records a click; other buttons do not', () => {
    const onOpen = jest.fn();
    const link = webLinkProps('https://daily.example.com/a', onOpen);
    expect(link).toMatchObject({
      href: 'https://daily.example.com/a',
      hrefAttrs: { target: '_blank', rel: 'noopener noreferrer' },
    });
    link.onAuxClick({ button: 2 }); // the context menu
    expect(onOpen).not.toHaveBeenCalled();
    link.onAuxClick({ button: 1 });
    expect(onOpen).toHaveBeenCalledTimes(1);
  });
});

describe('toasts', () => {
  it('focusAction flags only a toast that is still showing', () => {
    useToasts.setState({ toasts: [] });
    const id = useToasts.getState().show({ message: 'Story hidden', actionLabel: 'Undo' });
    useToasts.getState().focusAction(id + 1);
    expect(useToasts.getState().toasts[0].focusAction).toBeUndefined();
    useToasts.getState().focusAction(id);
    expect(useToasts.getState().toasts[0].focusAction).toBe(true);
    useToasts.getState().discardAll();
  });
});
