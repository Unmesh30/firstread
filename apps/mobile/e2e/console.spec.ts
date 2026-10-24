/// <reference types="node" />
// No page may log an error on load: every exported route, for a new reader and for an
// onboarded one, at a phone and a desktop width, in light and dark. This is the guard
// for React's hydration error #418 (static HTML that the first client render doesn't
// reproduce): a new screen that renders differently in Node and in the browser fails
// here. Routes come from the export itself, so a new screen is checked without a change.
import { readdirSync } from 'node:fs';
import { join } from 'node:path';

import { expect, test, type Page } from '@playwright/test';

import { BASE } from './env';
import { onboard, stubPublishers } from './helpers';

const NOT_ROUTES = new Set(['_sitemap', '+not-found', '404']);
const routes = readdirSync(join(__dirname, '..', 'dist'))
  .filter((f) => f.endsWith('.html'))
  .map((f) => f.slice(0, -'.html'.length))
  .filter((name) => !NOT_ROUTES.has(name))
  .map((name) => (name === 'index' ? '/' : `/${name}`))
  .sort();

test('the export has every screen', () => {
  for (const r of ['/', '/about', '/account', '/feed', '/onboarding', '/personas', '/privacy']) {
    expect(routes).toContain(r);
  }
  expect(routes).toEqual(expect.arrayContaining(['/system', '/why']));
});

function collectErrors(page: Page): string[] {
  const errors: string[] = [];
  page.on('console', (m) => {
    if (m.type() === 'error') errors.push(`console: ${m.text()}`);
  });
  page.on('pageerror', (e) => errors.push(`uncaught: ${e.message}`));
  return errors;
}

async function loadEach(page: Page, errors: string[]) {
  const found: Record<string, string[]> = {};
  for (const route of routes) {
    errors.length = 0;
    await page.goto(`${BASE}${route}`);
    await page.waitForLoadState('networkidle');
    await page.waitForTimeout(300); // effects after the last response
    if (errors.length) found[route] = [...errors];
  }
  return found;
}

const shapes = [
  { name: 'desktop, light', viewport: { width: 1280, height: 800 }, colorScheme: 'light' },
  { name: 'phone, dark', viewport: { width: 375, height: 740 }, colorScheme: 'dark' },
] as const;

for (const shape of shapes) {
  test(`no console errors on any route: new reader, ${shape.name}`, async ({ browser }) => {
    test.setTimeout(60_000);
    const context = await browser.newContext(shape);
    const page = await context.newPage();
    const errors = collectErrors(page);
    expect(await loadEach(page, errors)).toEqual({});
    await context.close();
  });
}

test('no console errors on any route: onboarded reader, tablet, dark', async ({ browser }) => {
  test.setTimeout(90_000);
  const context = await browser.newContext({
    viewport: { width: 900, height: 800 },
    colorScheme: 'dark',
  });
  await stubPublishers(context);
  const page = await context.newPage();
  await onboard(page);
  const errors = collectErrors(page);
  expect(await loadEach(page, errors)).toEqual({});
  await context.close();
});

test('the system theme survives hydration (dark pages stay dark)', async ({ browser }) => {
  const context = await browser.newContext({ colorScheme: 'dark' });
  const page = await context.newPage();
  await page.goto(`${BASE}/about`);
  const heading = page.getByRole('heading', { level: 1 }).first();
  await expect(heading).toBeVisible();
  await expect
    .poll(() => heading.evaluate((el) => getComputedStyle(el).color))
    .toBe('rgb(242, 239, 232)'); // colors.dark.text
  await context.close();
});

test('the static HTML shows one nav bar per width, before any JavaScript', async ({ browser }) => {
  for (const [width, at] of [
    [375, 'bottom'],
    [800, 'top'],
  ] as const) {
    const context = await browser.newContext({
      viewport: { width, height: 800 },
      javaScriptEnabled: false,
    });
    const page = await context.newPage();
    await page.goto(`${BASE}/system`);
    const bars = page.getByRole('navigation', { name: 'Sections' });
    await expect(bars).toHaveCount(1); // display:none bars aren't in the a11y tree
    const box = (await bars.boundingBox())!;
    expect(at === 'top' ? box.y < 100 : box.y > 600).toBe(true);
    await context.close();
  }
});
