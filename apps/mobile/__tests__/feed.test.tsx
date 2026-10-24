import { onlineManager, QueryClientProvider } from '@tanstack/react-query';
import { act, fireEvent, screen, waitFor, within } from '@testing-library/react-native';
import { Slot } from 'expo-router';
import { renderRouter } from 'expo-router/testing-library';
import type { PropsWithChildren } from 'react';
import { Text } from 'react-native';

import { createQueryClient } from '@/api/queries';
import type { FeedItem } from '@/api/schema-types';
import Feed from '@/app/(tabs)/feed';
import Why from '@/app/why';
import type { AuthProvider } from '@/auth';
import { createDevAuth } from '@/auth/dev';
import type { QueuedEvent } from '@/events/batch';
import { EventsProvider } from '@/events/EventsProvider';
import type { Transport } from '@/events/queue';
import { EMPTY_TITLE, END_TITLE } from '@/features/feed/FeedScreen';
import { DEGRADED_NOTE } from '@/features/feed/notices';
import { useFeedLocal } from '@/features/feed/store';
import { DEV_401 } from '@/features/SessionProblem';
import { SessionProvider } from '@/providers/session';
import { memoryKV } from '@/storage/kv';
import { ThemeProvider, ToastHost, useToasts } from '@/ui';

const UID = '11111111-2222-4333-8444-555555555555';
const uuid = (n: number) => `00000000-0000-4000-8000-${String(n).padStart(12, '0')}`;
const json = (body: unknown, status = 200, headers: Record<string, string> = {}) =>
  new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json', ...headers },
  });
const apiError = (status: number, code: string, headers: Record<string, string> = {}) =>
  json({ error: { code, message: code }, request_id: `req-${code}` }, status, headers);

function item(n: number, over: Partial<FeedItem> = {}): FeedItem {
  return {
    id: uuid(n),
    title: `Headline ${n}`,
    source_name: n % 2 ? 'Odd Times' : 'Even Post',
    source_domain: n % 2 ? 'odd.example.com' : 'even.example.com',
    url: `https://news.example.com/${n}`,
    image_url: null,
    category: 'world',
    published_at: null,
    first_seen_at: new Date(Date.now() - 12 * 60_000).toISOString(),
    score: 1,
    reasons: [{ code: 'topic', text: 'Matches topics you picked: World' }],
    ...over,
  };
}
const page = (items: FeedItem[], next: string | null, rid = 'req-1', degraded: string[] = []) => ({
  request_id: rid,
  model_version: 'm',
  items,
  next_cursor: next,
  degraded,
});

const ME = {
  request_id: 'r',
  model_version: 'm',
  user_id: UID,
  is_anonymous: true,
  topics: ['world', 'us', 'sports'],
  hidden_sources: [],
  active_persona: null,
  history_count: 0,
  recent_history: [],
};
const TOPICS = { request_id: 'r', model_version: 'm', topics: [{ id: 'world', name: 'World' }] };

type Handler = (url: URL, r: Request) => Response | Promise<Response>;
let routes: Record<string, Handler>;
let requests: string[];
let sent: QueuedEvent[][];

const transport: Transport = {
  async send(events) {
    sent.push(events);
    return { kind: 'done', ids: events.map((e) => e.event_id) };
  },
};

beforeEach(() => {
  makeAuth = () => createDevAuth(memoryKV(), () => UID);
  requests = [];
  sent = [];
  useToasts.setState({ toasts: [] });
  useFeedLocal.getState().reset();
  onlineManager.setOnline(true);
  routes = {
    '/v1/me': () => json(ME),
    '/v1/topics': () => json(TOPICS),
    '/v1/feed': (url) =>
      url.searchParams.get('cursor') === 'c2'
        ? json(page([item(4), item(5)], null, 'req-2'))
        : json(page([item(1), item(2), item(3)], 'c2')),
  };
  jest.spyOn(globalThis, 'fetch').mockImplementation(async (input: RequestInfo | URL) => {
    const r = input as Request;
    const url = new URL(r.url);
    requests.push(`${r.method} ${url.pathname}${url.search}`);
    const path = url.pathname.startsWith('/v1/articles/') ? '/v1/articles/why' : url.pathname;
    const h = routes[path];
    return h ? h(url, r) : apiError(404, 'not_found');
  });
});
afterEach(() => jest.restoreAllMocks());

let makeAuth: () => AuthProvider;

function Providers({ children }: PropsWithChildren) {
  const client = createQueryClient();
  return (
    <ThemeProvider>
      <QueryClientProvider client={client}>
        <SessionProvider factory={makeAuth} apiUrl="http://api.test">
          <EventsProvider store={memoryKV()} transport={transport}>
            {children}
            <ToastHost />
          </EventsProvider>
        </SessionProvider>
      </QueryClientProvider>
    </ThemeProvider>
  );
}

async function open(initialUrl = '/feed') {
  await renderRouter(
    {
      _layout: () => (
        <Providers>
          <Slot />
        </Providers>
      ),
      feed: Feed,
      why: Why,
      onboarding: () => <Text>onboarding screen</Text>,
      system: () => <Text>system screen</Text>,
    },
    { initialUrl },
  );
}

const allSent = () => sent.flat();
const flushEvents = async () => {
  await act(async () => {
    await Promise.resolve();
  });
};

describe('the feed', () => {
  it('shows the first page as links named "headline, source, age"', async () => {
    await open();
    expect(await screen.findByRole('link', { name: /^Headline 1, Odd Times, 12 minutes ago$/ }));
    expect(screen.getByRole('heading', { name: 'For You' })).toBeTruthy();
    expect(screen.getAllByText('Matches topics you picked: World')).toHaveLength(3);
    expect(requests.filter((r) => r.startsWith('GET /v1/feed'))).toEqual(['GET /v1/feed?limit=20']);
  });

  it('waits for an onboarded reader: no feed request before topics, then onboarding', async () => {
    routes['/v1/me'] = () => json({ ...ME, topics: [] });
    await open();
    await screen.findByText('onboarding screen');
    expect(requests.some((r) => r.includes('/v1/feed'))).toBe(false);
  });

  it('loads the next page at the end, then shows "all caught up"', async () => {
    await open();
    await screen.findByText('Headline 3');
    await act(async () => {
      fireEvent(screen.getByTestId('feed-list'), 'onEndReached');
    });
    expect(await screen.findByText('Headline 5')).toBeTruthy();
    expect(requests).toContain('GET /v1/feed?limit=20&cursor=c2');
    expect(screen.getByText(END_TITLE)).toBeTruthy();
  });

  it('an expired cursor starts the feed over from page 1, with a toast', async () => {
    const info = jest.spyOn(console, 'info').mockImplementation(() => undefined);
    routes['/v1/feed'] = (url) =>
      url.searchParams.get('cursor')
        ? apiError(400, 'cursor_expired')
        : json(page([item(1)], 'c2'));
    await open();
    await screen.findByText('Headline 1');
    await act(async () => {
      fireEvent(screen.getByTestId('feed-list'), 'onEndReached');
    });
    expect(await screen.findByText('Your feed was refreshed')).toBeTruthy();
    expect(requests.filter((r) => r === 'GET /v1/feed?limit=20')).toHaveLength(2);
    expect(info).toHaveBeenCalledWith(expect.stringContaining('req-cursor_expired'));
  });

  it('an empty first page explains when stories arrive', async () => {
    routes['/v1/feed'] = () => json(page([], null));
    await open();
    expect(await screen.findByText(EMPTY_TITLE)).toBeTruthy();
    expect(screen.getByText('New articles arrive every 15 minutes.')).toBeTruthy();
    await fireEvent.press(screen.getByRole('button', { name: 'See the System page' }));
    expect(await screen.findByText('system screen')).toBeTruthy();
  });

  it('a degraded page renders normally with a quiet note', async () => {
    routes['/v1/feed'] = () => json(page([item(1)], null, 'r', ['ann_timeout']));
    await open();
    await screen.findByText('Headline 1');
    expect(screen.getByText(DEGRADED_NOTE)).toBeTruthy();
  });

  it('a server error shows Retry and the request id', async () => {
    let fail = true;
    routes['/v1/feed'] = () => (fail ? apiError(500, 'internal') : json(page([item(1)], null)));
    await open();
    expect(await screen.findByText('Request id: req-internal')).toBeTruthy();
    fail = false;
    await fireEvent.press(screen.getByRole('button', { name: 'Try again' }));
    expect(await screen.findByText('Headline 1')).toBeTruthy();
  });

  it('a 401 with the dev bypass is a configuration banner', async () => {
    routes['/v1/me'] = () => apiError(401, 'unauthorized');
    await open();
    expect(await screen.findByText(DEV_401)).toBeTruthy();
  });

  it('a 401 with Supabase asks before starting a new session', async () => {
    const reset = jest.fn(async () => ({ userId: uuid(77), isAnonymous: true }));
    makeAuth = () => ({
      ...createDevAuth(memoryKV(), () => UID),
      mode: 'supabase',
      reset,
    });
    routes['/v1/me'] = () => apiError(401, 'unauthorized');
    await open();
    expect(await screen.findByText('Your session expired')).toBeTruthy();
    await fireEvent.press(screen.getByRole('button', { name: 'Start a new session' }));
    expect(reset).not.toHaveBeenCalled(); // never silently: the reader confirms first
    await fireEvent.press(screen.getByRole('button', { name: 'Start fresh' }));
    await waitFor(() => expect(reset).toHaveBeenCalledTimes(1));
  });

  it('pull-to-refresh loads one new first page (on web: the "New stories" button, E2E)', async () => {
    await open();
    await screen.findByText('Headline 1');
    expect(screen.queryByRole('button', { name: 'New stories' })).toBeNull(); // native
    await act(async () => screen.getByTestId('feed-list').props.refreshControl.props.onRefresh());
    await waitFor(() =>
      expect(requests.filter((r) => r === 'GET /v1/feed?limit=20')).toHaveLength(2),
    );
    expect(await screen.findByText('Headline 1')).toBeTruthy();
    expect(requests.some((r) => r.includes('cursor='))).toBe(false);
  });

  it('503 + Retry-After shows "Busy, retrying in N s", then the page', async () => {
    let calls = 0;
    routes['/v1/feed'] = () =>
      ++calls === 1
        ? apiError(503, 'unavailable', { 'Retry-After': '1' })
        : json(page([item(1)], null));
    await open();
    expect(await screen.findByText(/Busy, retrying/)).toBeTruthy();
    expect(await screen.findByText('Headline 1', {}, { timeout: 3000 })).toBeTruthy();
    expect(screen.queryByTestId('busy-banner')).toBeNull();
  });

  it('offline: a banner, and loaded stories stay', async () => {
    await open();
    await screen.findByText('Headline 1');
    await act(async () => onlineManager.setOnline(false));
    expect(await screen.findByTestId('offline-banner')).toBeTruthy();
    expect(screen.getByText('Headline 1')).toBeTruthy();
    // Back online while this test's client is still mounted (it resumes paused work).
    await act(async () => onlineManager.setOnline(true));
    await waitFor(() => expect(screen.queryByTestId('offline-banner')).toBeNull());
  });
});

describe('story actions', () => {
  it('opening a story records a click with its page position and request id', async () => {
    await open();
    await fireEvent.press(await screen.findByTestId(`open-${uuid(2)}`));
    await flushEvents();
    expect(allSent()).toEqual([
      expect.objectContaining({
        type: 'click',
        article_id: uuid(2),
        position: 1,
        request_id: 'req-1',
      }),
    ]);
  });

  it('like: one event, shown as on; a second press only explains', async () => {
    await open();
    const like = await screen.findByRole('button', { name: 'Like: Headline 1' });
    await fireEvent.press(like);
    await flushEvents();
    const on = screen.getByRole('button', { name: 'Liked: Headline 1' });
    expect(on).toBeSelected();
    await fireEvent.press(on);
    await flushEvents();
    expect(allSent().filter((e) => e.type === 'like')).toHaveLength(1);
    expect(screen.getByText(/Already liked/)).toBeTruthy();
  });

  it('hide, then Undo: the story comes back and nothing is sent', async () => {
    await open();
    await fireEvent.press(await screen.findByRole('button', { name: 'Hide: Headline 1' }));
    await fireEvent.press(within(screen.getByTestId('hide-menu')).getByText('Hide this story'));
    expect(screen.queryByText('Headline 1')).toBeNull();
    expect(screen.getByText('Story hidden. Undo available')).toBeTruthy();
    await fireEvent.press(screen.getByRole('button', { name: 'Undo' }));
    expect(screen.getByText('Headline 1')).toBeTruthy();
    await flushEvents();
    expect(allSent()).toEqual([]);
  });

  it('a hide is sent when its undo toast closes', async () => {
    await open();
    await fireEvent.press(await screen.findByRole('button', { name: 'Hide: Headline 2' }));
    await fireEvent.press(screen.getByTestId('hide-article'));
    await act(async () => useToasts.getState().dismiss(useToasts.getState().toasts[0].id));
    await flushEvents();
    expect(allSent()).toEqual([
      expect.objectContaining({ type: 'hide', scope: 'article', article_id: uuid(2), position: 1 }),
    ]);
  });

  it('hiding a source removes all its stories', async () => {
    await open();
    await fireEvent.press(await screen.findByRole('button', { name: 'Hide: Headline 1' }));
    await fireEvent.press(screen.getByRole('button', { name: 'Hide Odd Times' }));
    expect(screen.queryByText('Headline 1')).toBeNull();
    expect(screen.queryByText('Headline 3')).toBeNull();
    expect(screen.getByText('Headline 2')).toBeTruthy();
    expect(screen.getByText('Stories from Odd Times hidden. Undo available')).toBeTruthy();
    await act(async () => useToasts.getState().commitPending());
    await flushEvents();
    expect(allSent()).toEqual([expect.objectContaining({ type: 'hide', scope: 'source' })]);
  });

  it('a second hide commits the first one (one undo at a time)', async () => {
    await open();
    await fireEvent.press(await screen.findByRole('button', { name: 'Hide: Headline 1' }));
    await fireEvent.press(screen.getByTestId('hide-article'));
    await fireEvent.press(screen.getByRole('button', { name: 'Hide: Headline 2' }));
    await fireEvent.press(screen.getByTestId('hide-article'));
    await flushEvents();
    expect(allSent().map((e) => e.article_id)).toEqual([uuid(1)]);
    expect(useToasts.getState().toasts).toHaveLength(1);
  });
});

const WHY = {
  request_id: 'rw',
  model_version: 'm',
  article_id: uuid(1),
  title: 'Headline 1',
  category: 'world',
  components: { model: 0.36, fresh: 0.05, trend: 0.1, final: 0.51 },
  age_hours: 0.25,
  outlets: 3,
  followed_topic: true,
  topic_weight: 1,
  similar_reads: [
    { article_id: uuid(9), title: 'An earlier read', similarity: 0.82 },
    { article_id: uuid(8), title: 'Another read', similarity: 0.7 },
    { article_id: uuid(7), title: 'A third read', similarity: 0.6 },
  ],
  reasons: [{ code: 'topic', text: 'Matches topics you picked: World' }],
  degraded: [],
};

describe('Why am I seeing this?', () => {
  it('shows the reasons, the blend, the facts and the two closest reads', async () => {
    routes['/v1/articles/why'] = () => json(WHY);
    await open(`/why?id=${uuid(1)}`);
    expect(await screen.findByRole('heading', { name: 'Why am I seeing this?' })).toBeTruthy();
    expect(await screen.findByText('Matches topics you picked: World')).toBeTruthy();
    expect(screen.getByText('Your interests 0.36')).toBeTruthy();
    expect(screen.getByText('Score 0.51')).toBeTruthy();
    expect(screen.getByText('Published about 15 minutes ago')).toBeTruthy();
    expect(screen.getByText('Covered by 3 outlets')).toBeTruthy();
    expect(screen.getByText('In World, a topic you follow')).toBeTruthy();
    expect(screen.getByText('82% similar')).toBeTruthy();
    expect(screen.queryByText('A third read')).toBeNull();
    expect(screen.queryByText(/diversity/i)).toBeNull();
    expect(requests).toContain(`GET /v1/articles/${uuid(1)}/why`);
  });

  it('a degraded explanation says so', async () => {
    routes['/v1/articles/why'] = () => json({ ...WHY, degraded: ['history'] });
    await open(`/why?id=${uuid(1)}`);
    expect(await screen.findByText(DEGRADED_NOTE)).toBeTruthy();
  });

  it('an expired story says it is gone; a bad id never reaches the API', async () => {
    await open(`/why?id=${uuid(1)}`);
    expect(await screen.findByText('This story is no longer available')).toBeTruthy();
    requests = [];
    await open('/why?id=not-a-uuid');
    expect(await screen.findByText('No story given')).toBeTruthy();
    expect(requests.some((r) => r.includes('/why'))).toBe(false);
  });

  it('opens from a card', async () => {
    routes['/v1/articles/why'] = () => json(WHY);
    await open();
    await fireEvent.press(
      await screen.findByRole('button', { name: 'Why am I seeing this: Headline 1' }),
    );
    expect(await screen.findByText('Score 0.51')).toBeTruthy();
  });
});
