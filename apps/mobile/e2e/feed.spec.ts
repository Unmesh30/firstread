// PR 4b: the feed against the local stack. Each test is a new reader (a new browser
// context, so a new dev id).
import { expect, test } from '@playwright/test';

import { API_URL, BASE } from './env';
import {
  cards,
  devUser,
  eventsOf,
  hidePage,
  metric,
  onboard,
  redisByCommand,
  scrollUntil,
  stubPublishers,
  visibleCardIds,
  waitForEvents,
  waitForFeed,
  type FeedPage,
} from './helpers';

test.beforeEach(async ({ context }) => stubPublishers(context));

test('the first page is fetched only after the topics are saved', async ({ page }) => {
  const order: string[] = [];
  page.on('request', (r) => {
    const u = new URL(r.url());
    if (r.method() !== 'OPTIONS' && u.origin === API_URL) order.push(`${r.method()} ${u.pathname}`);
  });
  const first = await onboard(page);
  expect(order.indexOf('PUT /v1/me/topics')).toBeLessThan(order.indexOf('GET /v1/feed'));
  expect(order.filter((o) => o === 'GET /v1/feed')).toHaveLength(1);
  expect(first.items).toHaveLength(20);
  await expect(page.getByRole('heading', { name: 'For You', level: 1 })).toBeVisible();
});

test('"New stories" loads one new first page', async ({ page }) => {
  const first = await onboard(page);
  const fresh = waitForFeed(page, false);
  await page.getByRole('button', { name: 'New stories' }).click();
  const next = (await (await fresh).json()) as FeedPage;
  expect(next.request_id).not.toBe(first.request_id);
  await expect(cards(page).first()).toBeVisible();
});

test('impressions follow the 50% / 1 s rule and go out with the first click', async ({ page }) => {
  const first = await onboard(page);
  await page.waitForTimeout(1500);
  const visible = await visibleCardIds(page);
  expect(visible.length).toBeGreaterThan(0);
  expect(visible.length).toBeLessThan(first.items.length);

  const target = first.items[1];
  const sent = waitForEvents(page, (e) => e.some((x) => x.type === 'click'));
  const popup = page.waitForEvent('popup');
  await page.getByTestId(`open-${target.id}`).click();
  expect((await popup).url()).toContain(target.id); // the publisher's page, in a new tab
  const events = eventsOf(await sent);

  const click = events.find((e) => e.type === 'click');
  expect(click).toMatchObject({
    article_id: target.id,
    position: 1,
    request_id: first.request_id,
  });
  const impressions = events.filter((e) => e.type === 'impression');
  expect(impressions.length).toBeGreaterThan(0);
  for (const i of impressions) {
    expect(i.request_id).toBe(first.request_id);
    expect(visible).toContain(i.article_id); // nothing below the fold
    expect(i.position).toBe(first.items.findIndex((x) => x.id === i.article_id));
  }
  expect(new Set(impressions.map((i) => i.article_id)).size).toBe(impressions.length);
});

test('after a click, page 2 has no repeats and is re-ranked; page 3 comes from the slate', async ({
  page,
}) => {
  const first = await onboard(page);
  const acked = page.waitForResponse((r) => r.url() === `${API_URL}/v1/events`);
  const popup = page.waitForEvent('popup');
  await page.getByTestId(`open-${first.items[0].id}`).click();
  await (await popup).close();
  expect((await acked).ok()).toBe(true);

  const ranks0 = await metric(page, 'feed_stage_seconds_count', 'stage="rank"');
  const second = (await (await scrollUntil(page, waitForFeed(page, true))).json()) as FeedPage;
  const ids1 = new Set(first.items.map((i) => i.id));
  expect(second.items.length).toBeGreaterThan(0);
  expect(second.items.filter((i) => ids1.has(i.id))).toEqual([]);
  expect(await metric(page, 'feed_stage_seconds_count', 'stage="rank"')).toBe(ranks0 + 1);

  const third = (await (await scrollUntil(page, waitForFeed(page, true))).json()) as FeedPage;
  const seen = new Set([...ids1, ...second.items.map((i) => i.id)]);
  expect(third.items.filter((i) => seen.has(i.id))).toEqual([]);
  expect(await metric(page, 'feed_stage_seconds_count', 'stage="rank"')).toBe(ranks0 + 1);
});

test('hide, then Undo: the story comes back and no hide is sent', async ({ page }) => {
  test.setTimeout(30_000);
  const first = await onboard(page);
  const batches: string[][] = [];
  page.on('request', (r) => {
    if (r.url() === `${API_URL}/v1/events`) batches.push(eventsOf(r).map((e) => e.type));
  });
  const story = first.items[0];
  await page.getByTestId(`hide-${story.id}`).click();
  await page.getByTestId('hide-article').click();
  await expect(page.getByTestId(`card-${story.id}`)).toHaveCount(0);
  await expect(page.getByRole('status').getByText('Story hidden. Undo available')).toBeVisible();
  await page.getByTestId('toast-action').click();
  await expect(page.getByTestId(`card-${story.id}`)).toBeVisible();
  await page.waitForTimeout(6000);
  expect(batches.flat()).not.toContain('hide');
});

test('a hide survives closing the page inside its undo window', async ({ context, page }) => {
  const first = await onboard(page);
  const story = first.items[0];
  const hides = async () => {
    const text = await (await context.request.get(`${API_URL}/metrics`)).text();
    const m = /^events_total\{result="accepted",type="hide"\} (\S+)$/m.exec(text);
    return m ? Number(m[1]) : 0;
  };
  const before = await hides();
  await page.getByTestId(`hide-${story.id}`).click();
  await page.getByTestId('hide-article').click();
  await expect(page.getByTestId('toast')).toBeVisible();
  await page.close({ runBeforeUnload: true }); // well inside the 5 s

  // No page is open: only the closing page's keepalive can have delivered it.
  await expect.poll(hides, { timeout: 5000 }).toBe(before + 1);
  const again = await context.newPage();
  const p1 = waitForFeed(again, false);
  await again.goto(`${BASE}/feed`); // also the onboarded reader's /feed deep link
  const fresh = (await (await p1).json()) as FeedPage;
  expect(fresh.items.map((i) => i.id)).not.toContain(story.id);
  await expect(cards(again).first()).toBeVisible();
  await expect(again.getByTestId(`card-${story.id}`)).toHaveCount(0);
});

test('hiding a source removes its stories and is saved on the profile', async ({ page }) => {
  test.setTimeout(30_000);
  const first = await onboard(page);
  const story = first.items[0];
  await page.getByTestId(`hide-${story.id}`).click();
  await page.getByTestId('hide-source').click();
  for (const i of first.items.filter((x) => x.source_domain === story.source_domain)) {
    await expect(page.getByTestId(`card-${i.id}`)).toHaveCount(0);
  }
  await expect(page.getByText(`Stories from ${story.source_name} hidden`)).toBeVisible();
  const sent = await waitForEvents(page, (e) => e.some((x) => x.type === 'hide'));
  expect(eventsOf(sent)).toContainEqual(expect.objectContaining({ type: 'hide', scope: 'source' }));
  await expect
    .poll(async () => {
      const me = await page.request.get(`${API_URL}/v1/me`, {
        headers: { 'X-Dev-User': await devUser(page) },
      });
      return ((await me.json()) as { hidden_sources: string[] }).hidden_sources;
    })
    .toContain(story.source_domain);
});

test('like: on after one press, one event', async ({ page }) => {
  const first = await onboard(page);
  const story = first.items[0];
  const sent = waitForEvents(page, (e) => e.some((x) => x.type === 'like'));
  await page.getByRole('button', { name: `Like: ${story.title}` }).click();
  expect(eventsOf(await sent)).toContainEqual(
    expect.objectContaining({ type: 'like', article_id: story.id, request_id: first.request_id }),
  );
  await expect(page.getByRole('button', { name: `Liked: ${story.title}` })).toHaveAttribute(
    'aria-pressed',
    'true',
  );
});

test('Why opens from a card, closes with Esc and gives focus back', async ({ page }) => {
  const first = await onboard(page);
  const story = first.items[0];
  const why = page.waitForResponse((r) => r.url().endsWith(`/v1/articles/${story.id}/why`));
  await page.getByTestId(`why-${story.id}`).focus();
  await page.keyboard.press('Enter');
  expect((await why).ok()).toBe(true);
  await expect(page).toHaveURL(`${BASE}/why?id=${story.id}`);
  const sheet = page.getByRole('dialog', { name: 'Why am I seeing this?' });
  await expect(sheet.getByText('How the score adds up')).toBeVisible();
  await expect(sheet.getByText(/^Score /)).toBeVisible();
  await page.keyboard.press('Escape');
  await expect(page).toHaveURL(`${BASE}/feed`);
  await expect(page.getByTestId(`why-${story.id}`)).toBeFocused();
});

test('Why by deep link, with the feed behind it', async ({ page }) => {
  const first = await onboard(page);
  const story = first.items[2];
  await page.goto(`${BASE}/why?id=${story.id}`);
  const sheet = page.getByRole('dialog', { name: 'Why am I seeing this?' });
  await expect(sheet.getByText(story.title)).toBeVisible();
  await page.reload();
  await expect(sheet.getByText('How the score adds up')).toBeVisible();
  await sheet.getByRole('button', { name: 'Close' }).click();
  await expect(page).toHaveURL(`${BASE}/feed`);
  await expect(cards(page).first()).toBeVisible();
});

test('a 503 with Retry-After shows "Busy, retrying", then the feed', async ({ page }) => {
  let calls = 0;
  await page.route(`${API_URL}/v1/feed**`, (route) =>
    route.request().method() === 'GET' && ++calls === 1
      ? route.fulfill({
          status: 503,
          headers: { 'Retry-After': '2', 'Access-Control-Allow-Origin': '*' },
          contentType: 'application/json',
          body: JSON.stringify({
            error: { code: 'unavailable', message: 'busy' },
            request_id: 'x',
          }),
        })
      : route.continue(),
  );
  await page.goto(`${BASE}/`);
  for (const t of ['business', 'tech_science', 'sports'])
    await page.getByTestId(`topic-${t}`).click();
  await page.getByTestId('onboarding-continue').click();
  await expect(page.getByTestId('busy-banner')).toContainText(/Busy, retrying in [12] s/);
  await expect(cards(page).first()).toBeVisible({ timeout: 10_000 });
  await expect(page.getByTestId('busy-banner')).toHaveCount(0);
});

test('an expired cursor starts the feed over at page 1, with a toast', async ({ page }) => {
  await onboard(page);
  await page.route(`${API_URL}/v1/feed?*cursor=*`, (route) =>
    route.fulfill({
      status: 400,
      headers: { 'Access-Control-Allow-Origin': '*' },
      contentType: 'application/json',
      body: JSON.stringify({
        error: { code: 'cursor_expired', message: 'expired' },
        request_id: 'req-expired',
      }),
    }),
  );
  await scrollUntil(page, waitForFeed(page, false));
  await expect(page.getByText('Your feed was refreshed')).toBeVisible();
  await expect(cards(page).first()).toBeVisible();
});

test('offline: a banner, the loaded feed stays, events wait for the network', async ({
  context,
  page,
}) => {
  const first = await onboard(page);
  await context.setOffline(true);
  await expect(page.getByTestId('offline-banner')).toBeVisible();
  const sent: string[] = [];
  page.on('request', (r) => {
    if (r.url() === `${API_URL}/v1/events`) sent.push(...eventsOf(r).map((e) => e.type));
  });
  await page.getByRole('button', { name: `Like: ${first.items[0].title}` }).click();
  await page.waitForTimeout(1000);
  expect(sent).toEqual([]);
  await expect(cards(page).first()).toBeVisible();
  const flushed = waitForEvents(page, (e) => e.some((x) => x.type === 'like'));
  await context.setOffline(false);
  await flushed;
  await expect(page.getByTestId('offline-banner')).toHaveCount(0);
});

test('a retried batch is reported as duplicates only', async ({ page }) => {
  const first = await onboard(page);
  let lost = false;
  await page.route(`${API_URL}/v1/events`, async (route) => {
    if (route.request().method() !== 'POST' || lost) return route.continue();
    lost = true;
    await route.fetch(); // the server stores the batch…
    await route.abort('failed'); // …and the answer never arrives
  });
  const retried = page.waitForResponse(
    (r) => r.url() === `${API_URL}/v1/events` && r.request().method() === 'POST',
  );
  await page.getByRole('button', { name: `Like: ${first.items[0].title}` }).click();
  const body = (await (await retried).json()) as { accepted: string[]; duplicate: string[] };
  expect(body.accepted).toEqual([]);
  expect(body.duplicate.length).toBeGreaterThan(0);
});

test('a reading session costs at most 37 Redis commands', async ({ page }) => {
  test.setTimeout(60_000);
  const before = await redisByCommand(page);
  const first = await onboard(page);
  await page.waitForTimeout(1500); // impressions on page 1

  const click1 = page.waitForResponse((r) => r.url() === `${API_URL}/v1/events`);
  const tab1 = page.waitForEvent('popup');
  await page.getByTestId(`open-${first.items[1].id}`).click();
  await (await tab1).close();
  await click1;

  await scrollUntil(page, waitForFeed(page, true));
  await page.waitForTimeout(1200);
  await scrollUntil(page, waitForFeed(page, true));
  await page.waitForTimeout(1200);

  const [onScreen] = await visibleCardIds(page);
  const click2 = page.waitForResponse((r) => r.url() === `${API_URL}/v1/events`);
  const tab2 = page.waitForEvent('popup');
  await page.getByTestId(`open-${onScreen}`).click();
  await (await tab2).close();
  await click2;

  const why = page.waitForResponse((r) => r.url().includes('/why'));
  await page.getByTestId(`why-${onScreen}`).click();
  await why;
  // The Modal turns into the dialog (and listens for Esc) when its fade-in ends.
  await expect(page.getByRole('dialog', { name: 'Why am I seeing this?' })).toBeVisible();
  await page.keyboard.press('Escape');
  await expect(page).toHaveURL(`${BASE}/feed`);
  const list = (await page.getByTestId('feed-list').boundingBox())!;
  await page.mouse.move(list.x + list.width / 2, list.y + list.height / 2);
  await page.mouse.wheel(0, 900); // a few more stories, then the reader switches tabs
  await page.waitForTimeout(1500);

  const last = page.waitForResponse((r) => r.url() === `${API_URL}/v1/events`);
  await hidePage(page); // the background flush of the last impressions
  await last;

  const after = await redisByCommand(page);
  const used = Object.fromEntries(
    Object.keys(after)
      .map((k) => [k, after[k] - (before[k] ?? 0)] as const)
      .filter(([, n]) => n > 0),
  );
  const total = Object.values(used).reduce((a, b) => a + b, 0);
  console.log(`Redis commands this session: ${total}`, used);
  expect(total).toBeLessThanOrEqual(37);
});
