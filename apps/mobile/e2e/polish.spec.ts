/// <reference types="node" />
// PR 4d: the polish and accessibility pass. Share-card tags in the exported HTML; the
// theme before JavaScript; deep links that mustn't load the feed; Esc during a sheet's
// fade-in; middle-clicks; a full axe pass in light and dark; a keyboard-only walk;
// reflow at 320 px and 200% zoom; reduced motion; thumbnails; the feed list.
import { readdirSync, readFileSync } from 'node:fs';
import { join } from 'node:path';

import { expect, test, type Browser, type Page } from '@playwright/test';

import { API_URL, BASE } from './env';
import {
  cards,
  eventsOf,
  onboard,
  scrollUntil,
  seriousViolations,
  stubPublishers,
  waitForEvents,
  waitForFeed,
  type FeedPage,
} from './helpers';

test.beforeEach(async ({ context }) => stubPublishers(context));

const DIST = join(__dirname, '..', 'dist');
const SITE_URL = (
  process.env.EXPO_PUBLIC_SITE_URL ?? 'https://unmesh30.github.io/firstread'
).replace(/\/+$/, '');

// colors.dark in src/ui/tokens.ts, as computed styles report them.
const DARK = {
  background: 'rgb(18, 17, 16)',
  surface: 'rgb(28, 27, 25)',
  text: 'rgb(242, 239, 232)',
};
const LIGHT = { background: 'rgb(251, 250, 247)', surface: 'rgb(255, 255, 255)' };

function metaTags(html: string): Record<string, string> {
  const out: Record<string, string> = {};
  for (const m of html.matchAll(/<meta\s+(?:property|name)="([^"]+)"\s+content="([^"]*)"/g)) {
    out[m[1]] = m[2];
  }
  return out;
}

test('every exported page carries the Open Graph and Twitter card tags', async ({ request }) => {
  const pages = readdirSync(DIST).filter((f) => f.endsWith('.html'));
  expect(pages.length).toBeGreaterThan(8);
  const image = `${SITE_URL}/og-image.png`;
  for (const file of pages) {
    const tags = metaTags(readFileSync(join(DIST, file), 'utf8'));
    expect(tags, file).toMatchObject({
      'og:type': 'website',
      'og:site_name': 'FirstRead',
      'og:url': `${SITE_URL}/`,
      'og:image': image,
      'og:image:type': 'image/png',
      'og:image:width': '1200',
      'og:image:height': '630',
      'twitter:card': 'summary_large_image',
      'twitter:image': image,
    });
    for (const key of ['og:title', 'og:description', 'og:image:alt', 'twitter:title']) {
      expect(tags[key]?.length, `${file} ${key}`).toBeGreaterThan(10);
    }
  }
  // The image itself is published with the site: a 1200×630 PNG.
  const r = await request.get(`${BASE}/og-image.png`);
  expect(r.status()).toBe(200);
  expect(r.headers()['content-type']).toBe('image/png');
  const png = await r.body();
  expect(png.subarray(1, 4).toString()).toBe('PNG');
  expect([png.readUInt32BE(16), png.readUInt32BE(20)]).toEqual([1200, 630]);
});

/** The page, the section bar (a surface; only in the tab screens) and the h1's ink. */
async function surfaces(page: Page) {
  const bar = page.getByRole('navigation', { name: 'Sections' });
  return {
    background: await page.evaluate(() => getComputedStyle(document.body).backgroundColor),
    surface: (await bar.count())
      ? await bar.first().evaluate((el) => getComputedStyle(el).backgroundColor)
      : null,
    heading: await page
      .getByRole('heading', { level: 1 })
      .first()
      .evaluate((el) => getComputedStyle(el).color),
  };
}

test('a dark-mode reader sees dark surfaces before any JavaScript runs', async ({ browser }) => {
  const context = await browser.newContext({ colorScheme: 'dark', javaScriptEnabled: false });
  const page = await context.newPage();
  for (const [path, bar] of [
    ['/system', true],
    ['/account', true],
    ['/about', false],
    ['/onboarding', false],
  ] as const) {
    await page.goto(`${BASE}${path}`);
    expect(await surfaces(page), path).toEqual({
      background: DARK.background,
      surface: bar ? DARK.surface : null,
      heading: DARK.text,
    });
  }
  await context.close();
});

async function themedContext(browser: Browser, system: 'light' | 'dark', stored: string) {
  const context = await browser.newContext({ colorScheme: system });
  await context.addInitScript((t) => localStorage.setItem('fr:theme', t), stored);
  return context;
}

test('a stored Light/Dark choice is applied before JavaScript, and kept after', async ({
  browser,
}) => {
  for (const [system, stored, want] of [
    ['light', 'dark', DARK],
    ['dark', 'light', LIGHT],
  ] as const) {
    const context = await themedContext(browser, system, stored);
    const page = await context.newPage();
    // Only the inline <head> script runs: the app's bundle never loads.
    await page.route('**/_expo/static/js/**', (route) => route.abort());
    await page.goto(`${BASE}/system`);
    const before = await surfaces(page);
    expect([before.background, before.surface], `${stored} before JS`).toEqual([
      want.background,
      want.surface,
    ]);
    await page.unroute('**/_expo/static/js/**');
    await page.reload();
    await expect(page.getByTestId('stat-last-run')).toBeVisible();
    await expect
      .poll(() => page.evaluate(() => document.documentElement.getAttribute('data-theme')))
      .toBe(stored);
    const after = await surfaces(page);
    expect([after.background, after.surface], `${stored} after hydration`).toEqual([
      want.background,
      want.surface,
    ]);
    await context.close();
  }
});

test('deep links to /about, /privacy and /personas never fetch the feed', async ({ page }) => {
  await onboard(page);
  let feeds = 0;
  page.on('request', (r) => {
    if (r.url().startsWith(`${API_URL}/v1/feed`)) feeds++;
  });
  for (const [path, ready] of [
    ['/about', 'font-credit'],
    ['/privacy', 'retention'],
    ['/personas', 'persona-markets-tech'],
  ] as const) {
    await page.goto(`${BASE}${path}`);
    await expect(page.getByTestId(ready)).toBeVisible();
    await page.waitForLoadState('networkidle');
    await page.waitForTimeout(500);
    expect(feeds, path).toBe(0);
  }
  // Back from there (the feed is underneath) loads its first page.
  const first = waitForFeed(page, false);
  await page.getByTestId('back').click();
  expect((await first).ok()).toBe(true);
  await expect(page).toHaveURL(`${BASE}/feed`);
  await expect(cards(page).first()).toBeVisible();
});

test('Esc closes a sheet during its fade-in', async ({ page }) => {
  await page.setViewportSize({ width: 1280, height: 800 }); // wide: the sheets fade
  const first = await onboard(page);
  const story = first.items[0];

  // The hide menu: Esc as soon as it's in the page, before it's a finished dialog.
  await page.getByTestId(`hide-${story.id}`).click();
  await page.getByTestId('hide-menu').waitFor({ state: 'attached' });
  expect(await page.getByRole('dialog').count()).toBe(0); // still fading in
  await page.keyboard.press('Escape');
  await expect(page.getByTestId('hide-menu')).toHaveCount(0);
  await page.waitForTimeout(400); // past the fade: nothing reopened, nothing hidden
  await expect(page.getByRole('dialog')).toHaveCount(0);
  await expect(page.getByTestId(`card-${story.id}`)).toBeVisible();

  // The Why sheet (a route): Esc during its fade goes back to the feed.
  await page.getByTestId(`why-${story.id}`).click();
  await expect(page).toHaveURL(`${BASE}/why?id=${story.id}`);
  await page.keyboard.press('Escape');
  await expect(page).toHaveURL(`${BASE}/feed`);
});

test('a middle-click on a story is recorded as a click', async ({ page }) => {
  const first = await onboard(page);
  const story = first.items[1];
  const sent = waitForEvents(page, (e) => e.some((x) => x.type === 'click'));
  await page.getByTestId(`open-${story.id}`).click({ button: 'middle' });
  const clicks = eventsOf(await sent).filter((e) => e.type === 'click');
  expect(clicks).toEqual([
    expect.objectContaining({ article_id: story.id, request_id: first.request_id, position: 1 }),
  ]);
});

test('full axe pass: every screen and overlay, light and dark', async ({ browser }) => {
  test.setTimeout(180_000);
  const found: Record<string, string[]> = {};
  const scan = async (page: Page, label: string) => {
    const v = await seriousViolations(page);
    if (v.length) found[label] = v;
  };
  for (const colorScheme of ['light', 'dark'] as const) {
    // A new reader: the gate, onboarding and the 404.
    const fresh = await browser.newContext({ colorScheme });
    const p0 = await fresh.newPage();
    await p0.goto(`${BASE}/`);
    await expect(p0.getByRole('checkbox')).toHaveCount(13);
    await scan(p0, `${colorScheme} /onboarding`);
    await p0.goto(`${BASE}/no-such-page`);
    await expect(p0.getByText('This page doesn’t exist')).toBeVisible();
    await scan(p0, `${colorScheme} 404`);
    await fresh.close();

    // An onboarded reader: every screen, then the overlays.
    const context = await browser.newContext({ colorScheme });
    await stubPublishers(context);
    const page = await context.newPage();
    const first = await onboard(page);
    await scan(page, `${colorScheme} /feed`);
    const story = first.items[0];
    await page.getByTestId(`hide-${story.id}`).click();
    await expect(page.getByRole('dialog', { name: 'Hide' })).toBeVisible();
    await scan(page, `${colorScheme} hide menu`);
    await page.getByTestId('hide-article').click();
    await expect(page.getByTestId('toast')).toBeVisible();
    await scan(page, `${colorScheme} undo toast`);
    await page.getByRole('button', { name: 'Undo' }).click();
    await page.goto(`${BASE}/why?id=${first.items[1].id}`);
    // Scanned once the fade-in is over (mid-fade, opacity skews every contrast ratio).
    await expect(page.getByRole('dialog', { name: 'Why am I seeing this?' })).toBeVisible();
    await expect(page.getByText('How the score adds up')).toBeVisible();
    await scan(page, `${colorScheme} why`);
    for (const [path, ready] of [
      ['/personas', 'persona-markets-tech'],
      ['/account', 'followed-topics'],
      ['/system', 'stat-last-run'],
      ['/privacy', 'retention'],
      ['/about', 'font-credit'],
    ] as const) {
      await page.goto(`${BASE}${path}`);
      await expect(page.getByTestId(ready)).toBeVisible();
      await scan(page, `${colorScheme} ${path}`);
    }
    await page.goto(`${BASE}/personas`);
    await page.getByTestId('activate-markets-tech').click();
    await expect(page.getByRole('dialog', { name: 'Replace your reading history?' })).toBeVisible();
    await scan(page, `${colorScheme} persona dialog`);
    await context.close();
  }
  expect(found).toEqual({});
});

/** Press Tab until `done` holds for the focused element; fails after `max` presses. */
async function tabUntil(page: Page, done: string, max = 60) {
  for (let i = 0; i < max; i++) {
    await page.keyboard.press('Tab');
    if (await page.evaluate((sel) => document.activeElement?.matches(sel) ?? false, done)) return;
  }
  throw new Error(`Tab never reached ${done}`);
}

const focusRing = (page: Page) =>
  page.evaluate(() => {
    const s = getComputedStyle(document.activeElement!);
    return `${s.outlineStyle} ${s.outlineWidth}`;
  });

test('keyboard only: skip link, onboarding, a card’s controls, hide, Undo, back on the card', async ({
  page,
}) => {
  await page.goto(`${BASE}/`);
  await expect(page.getByRole('checkbox')).toHaveCount(13);

  // The first Tab is the skip link; following it lands in the main region.
  await page.keyboard.press('Tab');
  await expect(page.getByText('Skip to content')).toBeFocused();
  await page.keyboard.press('Enter');
  await page.keyboard.press('Tab');
  expect(await page.evaluate(() => !!document.activeElement?.closest('#main'))).toBe(true);

  // Three topics with Space, then Continue with Enter.
  for (const t of ['business', 'tech_science', 'sports']) {
    await tabUntil(page, `[data-testid="topic-${t}"]`);
    expect(await focusRing(page)).toBe('solid 2px');
    await page.keyboard.press('Space');
    await expect(page.getByTestId(`topic-${t}`)).toBeChecked();
  }
  const firstPage = waitForFeed(page, false);
  await tabUntil(page, '[data-testid="onboarding-continue"]');
  await page.keyboard.press('Enter');
  const first = (await (await firstPage).json()) as FeedPage;
  await expect(page).toHaveURL(`${BASE}/feed`);
  const story = first.items[0];

  // A card in visual order: the story link, then like, hide, why.
  await tabUntil(page, `[data-testid="open-${story.id}"]`);
  expect(await focusRing(page)).toBe('solid 2px');
  for (const id of ['like', 'hide', 'why']) {
    await page.keyboard.press('Tab');
    await expect(page.getByTestId(`${id}-${story.id}`)).toBeFocused();
  }

  // Hide by keyboard: the menu takes focus, then Undo is reachable with Tab.
  await page.keyboard.press('Shift+Tab');
  await page.keyboard.press('Enter');
  const menu = page.getByRole('dialog', { name: 'Hide' });
  await expect(menu).toBeVisible();
  expect(await page.evaluate(() => !!document.activeElement?.closest('[role="dialog"]'))).toBe(
    true,
  );
  await tabUntil(page, '[data-testid="hide-article"]', 5);
  await page.keyboard.press('Enter');
  await expect(page.getByTestId(`card-${story.id}`)).toHaveCount(0);
  // The hidden card took the focus target with it: focus goes to Undo, whose timer
  // waits while it has focus (5 s would be gone otherwise).
  await expect(page.getByTestId('toast-action')).toBeFocused();
  await page.waitForTimeout(5500);
  await expect(page.getByTestId('toast')).toBeVisible();
  await page.keyboard.press('Enter');
  await expect(page.getByTestId(`card-${story.id}`)).toBeVisible();
  // The round trip ends where it started: on the restored card's hide button (not on
  // <body> with the unmounted toast), with a visible ring, and Tab goes on to Why.
  await expect(page.getByTestId(`hide-${story.id}`)).toBeFocused();
  expect(await focusRing(page)).toBe('solid 2px');
  await page.keyboard.press('Tab');
  await expect(page.getByTestId(`why-${story.id}`)).toBeFocused();
});

/** Visible elements that stick out past the right edge (horizontal scrolling). */
const overflowing = (page: Page) =>
  page.evaluate(() => {
    const w = document.documentElement.clientWidth;
    return [...document.querySelectorAll('body *')]
      .filter((el) => {
        const r = el.getBoundingClientRect();
        return r.width > 0 && r.height > 0 && r.bottom > 0 && r.right > w + 1;
      })
      .slice(0, 5)
      .map(
        (el) =>
          `${el.tagName} ${el.getAttribute('data-testid') ?? ''} ${Math.round(el.getBoundingClientRect().right)}`,
      );
  });

test('reflow: 320 px wide and 200% zoom, no horizontal scrolling', async ({ page }) => {
  test.setTimeout(90_000);
  const first = await onboard(page);
  // 200% zoom on a 1280×800 window is a 640×400 CSS-pixel viewport.
  for (const viewport of [
    { width: 320, height: 640 },
    { width: 640, height: 400 },
  ]) {
    await page.setViewportSize(viewport);
    for (const path of [
      '/feed',
      `/why?id=${first.items[0].id}`,
      '/personas',
      '/account',
      '/system',
      '/privacy',
      '/about',
    ]) {
      await page.goto(`${BASE}${path}`);
      await expect(page.getByRole('heading', { level: 1 }).first()).toBeVisible();
      await page.waitForLoadState('networkidle');
      expect(await overflowing(page), `${viewport.width} px ${path}`).toEqual([]);
      expect(
        await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth),
        `${viewport.width} px ${path}`,
      ).toBe(true);
    }
    // The headline text wraps instead of being cut off.
    await page.goto(`${BASE}/feed`);
    await expect(page.getByText(first.items[0].title)).toBeVisible();
  }
});

/** CSS animations that start while the hide menu opens and the undo toast shows. */
async function animationsWhileHiding(browser: Browser, reducedMotion: 'reduce' | 'no-preference') {
  const context = await browser.newContext({
    reducedMotion,
    viewport: { width: 1280, height: 800 },
  });
  await stubPublishers(context);
  const page = await context.newPage();
  const first = await onboard(page);
  await page.evaluate(() => {
    const w = window as unknown as { started: string[] };
    w.started = [];
    document.addEventListener('animationstart', (e) => w.started.push(e.animationName), true);
  });
  await page.getByTestId(`hide-${first.items[0].id}`).click();
  await expect(page.getByRole('dialog', { name: 'Hide' })).toBeVisible();
  await page.getByTestId('hide-article').click();
  await expect(page.getByTestId('toast')).toBeVisible();
  await page.waitForTimeout(300);
  const started = await page.evaluate(() => (window as unknown as { started: string[] }).started);
  await context.close();
  return started;
}

test('reduced motion: the hide menu and the toast appear without animating', async ({
  browser,
}) => {
  // The probe works: the same steps animate without the preference.
  expect((await animationsWhileHiding(browser, 'no-preference')).length).toBeGreaterThan(0);
  expect(await animationsWhileHiding(browser, 'reduce')).toEqual([]);
});

test('thumbnails: lazy, fixed-size, decorative, and a tile when one fails', async ({ page }) => {
  // Seed articles have no images: give the first two stories one that loads and one
  // that's missing (registered after stubPublishers, so it wins for img.example.com).
  const png = readFileSync(join(__dirname, '..', 'public', 'og-image.png'));
  await page.route('https://img.example.com/**', (route) =>
    route.request().url().endsWith('/ok.png')
      ? route.fulfill({ contentType: 'image/png', body: png })
      : route.fulfill({ status: 404, body: '' }),
  );
  let ids: string[] = [];
  await page.route(`${API_URL}/v1/feed**`, async (route) => {
    const response = await route.fetch();
    const body = (await response.json()) as FeedPage & { items: { image_url: string | null }[] };
    body.items[0].image_url = 'https://img.example.com/ok.png';
    body.items[1].image_url = 'https://img.example.com/missing.png';
    ids = body.items.map((i) => i.id);
    await route.fulfill({ response, json: body });
  });
  await onboard(page);

  const ok = page.getByTestId(`card-${ids[0]}`).locator('img');
  await expect(ok).toHaveAttribute('loading', 'lazy');
  await expect(ok).toHaveAttribute('src', 'https://img.example.com/ok.png');
  const box = (await page.getByTestId(`thumb-${ids[0]}`).boundingBox())!;
  expect([Math.round(box.width), Math.round(box.height)]).toEqual([88, 88]);
  // Decorative: the story link's name already says everything.
  const name = await page.getByTestId(`open-${ids[0]}`).getAttribute('aria-label');
  expect(name).not.toContain('image');
  expect(await ok.evaluate((el) => el.getAttribute('alt') ?? '')).toBe('');

  // The missing one falls back to the source's initial, at the same size.
  const broken = page.getByTestId(`card-${ids[1]}`);
  await expect(broken.locator('img')).toHaveCount(0);
  await expect(broken.getByText(/^[A-Z]$/)).toBeVisible();
});

test('the feed list: unique rows in visual order, each one a named link', async ({ page }) => {
  await onboard(page);
  await scrollUntil(page, waitForFeed(page, true));
  await expect(cards(page).nth(25)).toBeAttached();
  const rows = await page.evaluate(() =>
    [...document.querySelectorAll('[data-testid^="card-"]')].map((el) => ({
      id: el.getAttribute('data-testid')!,
      top: el.getBoundingClientRect().top,
      name: el.querySelector('[data-testid^="open-"]')?.getAttribute('aria-label') ?? '',
    })),
  );
  expect(new Set(rows.map((r) => r.id)).size).toBe(rows.length);
  const tops = rows.map((r) => r.top);
  expect(tops).toEqual([...tops].sort((a, b) => a - b));
  // "headline, source, 12 minutes ago"
  for (const r of rows) expect(r.name, r.id).toMatch(/^.+, .+, (just now|\d+ \w+ ago)$/);
});
