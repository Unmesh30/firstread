// PR 4c: demo readers, Account (un-hide, delete my data), accessibility of the new
// screens, and the GDELT credit wherever GDELT headlines appear.
import AxeBuilder from '@axe-core/playwright';
import { expect, test, type Page } from '@playwright/test';

import { API_URL, BASE } from './env';
import { cards, devUser, onboard, stubPublishers, waitForEvents, eventsOf } from './helpers';

test.beforeEach(async ({ context }) => stubPublishers(context));

const GDELT = /^News discovery by The GDELT Project/;

async function meOf(page: Page, id?: string) {
  const r = await page.request.get(`${API_URL}/v1/me`, {
    headers: { 'X-Dev-User': id ?? (await devUser(page)) },
  });
  return (await r.json()) as {
    topics: string[];
    hidden_sources: string[];
    active_persona: string | null;
    history_count: number;
  };
}

async function seriousViolations(page: Page): Promise<string[]> {
  const results = await new AxeBuilder({ page })
    .withTags(['wcag2a', 'wcag2aa', 'wcag21aa', 'wcag22aa'])
    .analyze();
  return results.violations
    .filter((v) => v.impact === 'serious' || v.impact === 'critical')
    .map((v) => `${v.id}: ${v.nodes.map((n) => n.target.join(' ')).join(', ')}`);
}

test('a reader with topics is asked before a persona replaces them (409), then replaced', async ({
  page,
}) => {
  await onboard(page);
  await page.goto(`${BASE}/personas`);
  const card = page.getByTestId('persona-markets-tech');
  await expect(card.getByRole('heading', { name: 'Markets & tech' })).toBeVisible();

  const first = page.waitForResponse((r) => r.url().includes('/activate?replace=false'));
  await card.getByRole('button', { name: 'Read as Markets & tech' }).click();
  expect((await first).status()).toBe(409);
  const dialog = page.getByRole('dialog', { name: 'Replace your reading history?' });
  await expect(dialog).toBeVisible();
  await expect(dialog).toContainText(
    'Replace your reading history and topics with Markets & tech? Hidden stories and sources are kept.',
  );
  expect((await meOf(page)).active_persona).toBeNull(); // nothing changed yet

  const second = page.waitForResponse((r) => r.url().includes('/activate?replace=true'));
  const feed = page.waitForResponse(
    (r) => r.url().startsWith(`${API_URL}/v1/feed`) && r.request().method() === 'GET',
  );
  await dialog.getByRole('button', { name: 'Replace' }).click();
  expect((await second).status()).toBe(200);
  expect((await feed).ok()).toBe(true);
  await expect(page).toHaveURL(`${BASE}/feed`);
  await expect(cards(page).first()).toBeVisible();
  const me = await meOf(page);
  expect(me.active_persona).toBe('markets-tech');
  expect(me.topics.sort()).toEqual(['business', 'tech_science']);
});

test('"Keep mine" leaves the reader as they were', async ({ page }) => {
  await onboard(page, ['sports', 'health', 'weather']);
  await page.goto(`${BASE}/personas`);
  await page.getByTestId('activate-us-news').click();
  const dialog = page.getByRole('dialog', { name: 'Replace your reading history?' });
  await dialog.getByRole('button', { name: 'Keep mine' }).click();
  await expect(dialog).toHaveCount(0);
  const me = await meOf(page);
  expect(me.active_persona).toBeNull();
  expect(me.topics.sort()).toEqual(['health', 'sports', 'weather']);
});

test('a hidden source is listed in Account and shown again', async ({ page }) => {
  test.setTimeout(30_000);
  const first = await onboard(page);
  const story = first.items[0];
  await page.getByTestId(`hide-${story.id}`).click();
  const sent = waitForEvents(page, (e) => e.some((x) => x.type === 'hide'));
  await page.getByTestId('hide-source').click();
  expect(eventsOf(await sent)).toContainEqual(expect.objectContaining({ scope: 'source' }));
  await expect.poll(async () => (await meOf(page)).hidden_sources).toContain(story.source_domain);

  await page
    .getByRole('navigation', { name: 'Sections' })
    .getByRole('link', { name: 'Account' })
    .click();
  const row = page.getByTestId(`hidden-${story.source_domain}`);
  await expect(row).toBeVisible();
  const unhide = page.waitForResponse(
    (r) => r.url().includes('/v1/me/hidden-sources/') && r.request().method() === 'DELETE',
  );
  await row.getByRole('button', { name: `Show ${story.source_domain} again` }).click();
  expect((await unhide).ok()).toBe(true);
  await expect(page.getByTestId('no-hidden')).toBeVisible();
  await expect(page.getByText(`Stories from ${story.source_domain} will show again`)).toBeVisible();
  expect((await meOf(page)).hidden_sources).not.toContain(story.source_domain);
});

test('delete my data retries a 503, then starts over as a new reader', async ({ page }) => {
  test.setTimeout(30_000);
  const first = await onboard(page);
  const old = await devUser(page);
  // A click first, so there's history to delete.
  const acked = page.waitForResponse((r) => r.url() === `${API_URL}/v1/events`);
  const tab = page.waitForEvent('popup');
  await page.getByTestId(`open-${first.items[0].id}`).click();
  await (await tab).close();
  await acked;
  await expect.poll(async () => (await meOf(page, old)).history_count).toBe(1);

  let deletes = 0;
  await page.route(`${API_URL}/v1/me`, (route) =>
    route.request().method() === 'DELETE' && ++deletes === 1
      ? route.fulfill({
          status: 503,
          headers: { 'Retry-After': '1', 'Access-Control-Allow-Origin': '*' },
          contentType: 'application/json',
          body: JSON.stringify({
            error: { code: 'unavailable', message: 'retry' },
            request_id: 'x',
          }),
        })
      : route.continue(),
  );
  await page.goto(`${BASE}/account`);
  await page.getByTestId('delete-data').click();
  const dialog = page.getByRole('dialog', { name: 'Delete your data?' });
  const done = page.waitForResponse(
    (r) =>
      r.url() === `${API_URL}/v1/me` && r.request().method() === 'DELETE' && r.status() === 204,
  );
  await dialog.getByRole('button', { name: 'Delete' }).click();
  await expect(page.getByTestId('delete-retrying')).toBeVisible();
  await done;
  expect(deletes).toBe(2);

  await expect(page).toHaveURL(`${BASE}/onboarding`);
  await expect(page.getByText('Your data was deleted')).toBeVisible();
  const now = await devUser(page);
  expect(now).not.toBe(old);
  await page.unroute(`${API_URL}/v1/me`);
  // The old reader is gone: asking as them finds a brand-new, empty profile.
  const gone = await meOf(page, old);
  expect(gone).toMatchObject({ topics: [], history_count: 0, hidden_sources: [] });
});

test('the 4c screens have no serious accessibility violations (light and dark)', async ({
  page,
}) => {
  test.setTimeout(90_000);
  await onboard(page);
  const screens: [string, () => Promise<void>][] = [
    ['/personas', () => expect(page.getByTestId('persona-markets-tech')).toBeVisible()],
    ['/account', () => expect(page.getByTestId('followed-topics')).toBeVisible()],
    ['/system', () => expect(page.getByTestId('stat-last-run')).toBeVisible()],
    ['/privacy', () => expect(page.getByTestId('retention')).toBeVisible()],
    ['/about', () => expect(page.getByTestId('font-credit')).toBeVisible()],
  ];
  const found: Record<string, string[]> = {};
  for (const colorScheme of ['light', 'dark'] as const) {
    await page.emulateMedia({ colorScheme });
    for (const [path, ready] of screens) {
      await page.goto(`${BASE}${path}`);
      await ready();
      const v = await seriousViolations(page);
      if (v.length) found[`${colorScheme} ${path}`] = v;
    }
    // The persona question, open.
    await page.goto(`${BASE}/personas`);
    await page.getByTestId('activate-markets-tech').click();
    await expect(page.getByRole('dialog', { name: 'Replace your reading history?' })).toBeVisible();
    const v = await seriousViolations(page);
    if (v.length) found[`${colorScheme} persona dialog`] = v;
  }
  expect(found).toEqual({});
});

test('the GDELT credit links to gdeltproject.org wherever GDELT headlines appear', async ({
  page,
}) => {
  const first = await onboard(page);
  const credit = (scope: Page | ReturnType<Page['getByRole']>) =>
    scope.getByRole('link', { name: GDELT });
  const check = async (where: string, scope: Page | ReturnType<Page['getByRole']> = page) => {
    const link = credit(scope).first();
    await expect(link, where).toBeVisible();
    await expect(link, where).toHaveAttribute('href', 'https://www.gdeltproject.org/');
    await expect(link, where).toHaveAttribute('target', '_blank');
  };
  await check('feed', page.getByRole('main'));
  await page.goto(`${BASE}/why?id=${first.items[0].id}`);
  await check('why', page.getByRole('dialog', { name: 'Why am I seeing this?' }));
  for (const path of ['/personas', '/system', '/about']) {
    await page.goto(`${BASE}${path}`);
    await check(path);
  }
});
