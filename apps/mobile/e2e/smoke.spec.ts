// The 4a smoke test: the static build under /firstread, the gate, the dev sign-in
// through CORS, deep links, the 404 page, the font rules and a first accessibility scan.
import { expect, test } from '@playwright/test';

import { API_URL, BASE } from './env';
import { seriousViolations } from './helpers';

test('a new reader lands on onboarding, signed in with the dev header', async ({ page }) => {
  const me = page.waitForResponse(
    (r) => r.url() === `${API_URL}/v1/me` && r.request().method() === 'GET',
  );
  await page.goto(`${BASE}/`);
  const response = await me;
  expect(response.status()).toBe(200);
  expect(response.request().headers()['x-dev-user']).toMatch(/^[0-9a-f-]{36}$/);
  await expect(page).toHaveURL(`${BASE}/onboarding`);
  await expect(page.getByRole('heading', { name: 'Pick at least 3 topics' })).toBeVisible();
  // The 13 topics come from the API.
  await expect(page.getByRole('checkbox')).toHaveCount(13);
  await expect(page.getByTestId('onboarding-continue')).toBeDisabled();
});

test('the reader id survives a reload', async ({ page }) => {
  const ids: string[] = [];
  page.on('request', (r) => {
    if (r.url() === `${API_URL}/v1/me`) ids.push(r.headers()['x-dev-user']);
  });
  await page.goto(`${BASE}/`);
  await expect(page).toHaveURL(`${BASE}/onboarding`);
  await page.reload();
  await expect(page.getByRole('checkbox').first()).toBeVisible();
  expect(new Set(ids.filter(Boolean)).size).toBe(1);
});

for (const [path, heading] of [
  ['/system', 'System'],
  ['/account', 'Account'],
  ['/personas', 'Demo readers'],
  ['/why?id=00000000-0000-4000-8000-000000000000', 'Why am I seeing this?'],
] as const) {
  test(`deep link ${path} loads on a fresh request and a reload`, async ({ page }) => {
    const r = await page.goto(`${BASE}${path}`);
    expect(r?.status()).toBe(200);
    await expect(page.getByRole('heading', { name: heading, level: 1 })).toBeVisible();
    await page.reload();
    await expect(page.getByRole('heading', { name: heading, level: 1 })).toBeVisible();
  });
}

test('a deep link to /feed sends a new reader to onboarding, also on reload', async ({ page }) => {
  // An onboarded reader's /feed deep link and reload are covered in feed.spec.ts.
  const r = await page.goto(`${BASE}/feed`);
  expect(r?.status()).toBe(200);
  await expect(page).toHaveURL(`${BASE}/onboarding`);
  await page.goto(`${BASE}/feed`);
  await expect(page.getByRole('heading', { name: 'Pick at least 3 topics' })).toBeVisible();
});

test('an unknown path is a real 404 with a way back', async ({ page }) => {
  const r = await page.goto(`${BASE}/no-such-page`);
  expect(r?.status()).toBe(404);
  await expect(page.getByText('This page doesn’t exist')).toBeVisible();
});

test('one headline font, preloaded, swapped in, never blocking', async ({ page }) => {
  await page.goto(`${BASE}/onboarding`);
  const html = await page.content();
  expect(html.match(/rel="preload"[^>]+as="font"/g) ?? []).toHaveLength(1);
  expect(html).toContain('font-display:swap');
  const family = await page
    .getByRole('heading', { name: 'FirstRead', level: 1 })
    .evaluate((el) => getComputedStyle(el).fontFamily);
  expect(family).toContain('Georgia'); // the fallback stack is there until the font loads
});

test('onboarding has no serious accessibility violations (light and dark)', async ({ page }) => {
  for (const colorScheme of ['light', 'dark'] as const) {
    await page.emulateMedia({ colorScheme });
    await page.goto(`${BASE}/onboarding`);
    await expect(page.getByRole('checkbox')).toHaveCount(13);
    expect(await seriousViolations(page)).toEqual([]);
  }
});
