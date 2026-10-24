// Shared steps for the feed E2E: onboarding, reading the API's /metrics, the events a
// page posts, and scrolling the feed's list.
import AxeBuilder from '@axe-core/playwright';
import { expect, type BrowserContext, type Page, type Request } from '@playwright/test';

import { API_URL, BASE } from './env';

export interface FeedPage {
  request_id: string;
  items: { id: string; source_domain: string; source_name: string; title: string }[];
  next_cursor: string | null;
}
export interface SentEvent {
  event_id: string;
  type: string;
  article_id?: string;
  position?: number;
  request_id?: string;
  scope?: string;
}

/** Publisher links open in a new tab; answer them locally (no real network in tests). */
export async function stubPublishers(context: BrowserContext) {
  await context.route(/^https:\/\/[a-z]+\.example\.com\//, (route) =>
    route.fulfill({ contentType: 'text/html', body: '<title>Publisher</title>publisher page' }),
  );
}

const isFeed = (url: string) => url.startsWith(`${API_URL}/v1/feed`);
const isEvents = (r: Request) => r.url() === `${API_URL}/v1/events` && r.method() === 'POST';

export function eventsOf(r: Request): SentEvent[] {
  return (JSON.parse(r.postData() ?? '{"events":[]}') as { events: SentEvent[] }).events;
}

export function waitForEvents(page: Page, has: (e: SentEvent[]) => boolean = () => true) {
  return page.waitForRequest((r) => isEvents(r) && has(eventsOf(r)));
}

export function waitForFeed(page: Page, cursor: boolean) {
  return page.waitForResponse(
    (r) =>
      isFeed(r.url()) && r.request().method() === 'GET' && r.url().includes('cursor=') === cursor,
  );
}

export const cards = (page: Page) => page.locator('[data-testid^="card-"]');

/** A new reader picks three topics and lands on the feed; returns its first page. */
export async function onboard(page: Page, topics = ['business', 'tech_science', 'sports']) {
  await page.goto(`${BASE}/`);
  await expect(page).toHaveURL(`${BASE}/onboarding`);
  for (const t of topics) await page.getByTestId(`topic-${t}`).click();
  const first = waitForFeed(page, false);
  await page.getByTestId('onboarding-continue').click();
  const response = await first;
  await expect(page).toHaveURL(`${BASE}/feed`);
  await expect(cards(page).first()).toBeVisible();
  return (await response.json()) as FeedPage;
}

/**
 * Scroll the feed with the mouse wheel, as a reader would, until `done` settles (the
 * next page loads near the bottom). Programmatic scrollTo doesn't reach RN-web's list.
 */
export async function scrollUntil<T>(page: Page, done: Promise<T>): Promise<T> {
  let settled = false;
  const result = done.finally(() => {
    settled = true;
  });
  const box = (await page.getByTestId('feed-list').boundingBox())!;
  await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
  for (let i = 0; i < 40 && !settled; i++) {
    await page.mouse.wheel(0, 2000);
    await page.waitForTimeout(250);
  }
  return result;
}

/** The page and app going to the background (Page Visibility), as on a tab switch. */
export async function hidePage(page: Page) {
  await page.evaluate(() => {
    Object.defineProperty(document, 'visibilityState', { configurable: true, get: () => 'hidden' });
    document.dispatchEvent(new Event('visibilitychange'));
  });
}

/** Sum a Prometheus counter over its labels, filtered by `match` on the label text. */
export async function metric(page: Page, name: string, match = ''): Promise<number> {
  const text = await (await page.request.get(`${API_URL}/metrics`)).text();
  let total = 0;
  for (const line of text.split('\n')) {
    if (!line.startsWith(`${name}{`) || !line.includes(match)) continue;
    total += Number(line.slice(line.lastIndexOf(' ') + 1));
  }
  return total;
}

export async function redisByCommand(page: Page): Promise<Record<string, number>> {
  const text = await (await page.request.get(`${API_URL}/metrics`)).text();
  const out: Record<string, number> = {};
  for (const m of text.matchAll(/^redis_commands_total\{command="([^"]+)"\} (\S+)$/gm)) {
    out[m[1]] = Number(m[2]);
  }
  return out;
}

/** The dev reader id this page signs in with (to call the API as them). */
export async function devUser(page: Page): Promise<string> {
  return page.evaluate(() => localStorage.getItem('fr:dev-user') ?? '');
}

/** Rows ≥ 50% inside the list's viewport right now. */
export async function visibleCardIds(page: Page): Promise<string[]> {
  return page.evaluate(() => {
    const list = document.querySelector('[data-testid="feed-list"]')!.getBoundingClientRect();
    return [...document.querySelectorAll('[data-testid^="card-"]')]
      .filter((el) => {
        const r = el.getBoundingClientRect();
        const shown = Math.min(r.bottom, list.bottom) - Math.max(r.top, list.top);
        return shown >= r.height / 2;
      })
      .map((el) => el.getAttribute('data-testid')!.slice('card-'.length));
  });
}

/** axe's serious and critical WCAG 2.2 A/AA violations on the page as it is now. */
export async function seriousViolations(page: Page): Promise<string[]> {
  const results = await new AxeBuilder({ page })
    .withTags(['wcag2a', 'wcag2aa', 'wcag21aa', 'wcag22aa'])
    .analyze();
  return results.violations
    .filter((v) => v.impact === 'serious' || v.impact === 'critical')
    .map((v) => `${v.id}: ${v.nodes.map((n) => n.target.join(' ')).join(', ')}`);
}
