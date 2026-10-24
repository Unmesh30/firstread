/// <reference types="node" />
// PR 4c screens: personas (the 409 confirmation), Account (Google linking through the
// mocked Supabase provider, hidden sources, theme, reset, delete with the 503 retry),
// System, Privacy and About (the GDELT and font credits).
import { readFileSync } from 'node:fs';
import { join } from 'node:path';

import { QueryClientProvider } from '@tanstack/react-query';
import { act, fireEvent, screen, waitFor, within } from '@testing-library/react-native';
import { Slot } from 'expo-router';
import { renderRouter } from 'expo-router/testing-library';
import type { PropsWithChildren } from 'react';
import { Text } from 'react-native';

import { createQueryClient } from '@/api/queries';
import type { MeResponse, PersonaOut, StatsResponse } from '@/api/schema-types';
import About from '@/app/about';
import Account from '@/app/(tabs)/account';
import System from '@/app/(tabs)/system';
import Personas from '@/app/personas';
import Privacy from '@/app/privacy';
import { AUTH_MESSAGES, type AuthProvider } from '@/auth';
import { createDevAuth } from '@/auth/dev';
import { createSupabaseAuth } from '@/auth/supabase';
import { EventsProvider, useEventQueue } from '@/events/EventsProvider';
import type { Transport } from '@/events/queue';
import { DELETED } from '@/features/account/DeleteData';
import { SWITCH_WARNING } from '@/features/account/SignInSection';
import { GDELT, RSS_PUBLISHERS } from '@/features/credits';
import { replaceQuestion, WARMING_UP } from '@/features/personas/PersonasScreen';
import { DIAGRAM_DESCRIPTION } from '@/features/system/ArchitectureDiagram';
import { statTiles } from '@/features/system/SystemScreen';
import { duration } from '@/lib/time';
import { SessionProvider } from '@/providers/session';
import { memoryKV } from '@/storage/kv';
import { NEW_TAB, ThemeProvider, ToastHost, usePrefs, useToasts } from '@/ui';

import { fakeSupabase, FAKE_UID as UID } from '../test-utils/fakeSupabase';

const NEW_UID = '99999999-8888-4777-8666-555555555555';
const json = (body: unknown, status = 200, headers: Record<string, string> = {}) =>
  new Response(status === 204 ? null : JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json', ...headers },
  });
const apiError = (status: number, code: string, headers: Record<string, string> = {}) =>
  json({ error: { code, message: code }, request_id: `req-${code}` }, status, headers);

const me = (over: Partial<MeResponse> = {}): MeResponse => ({
  request_id: 'r',
  model_version: 'm',
  user_id: UID,
  is_anonymous: true,
  topics: ['world', 'sports', 'health'],
  hidden_sources: [],
  active_persona: null,
  history_count: 3,
  recent_history: [],
  ...over,
});
const persona = (id: string, name: string): PersonaOut => ({
  id,
  name,
  description: `A demo reader: ${name}`,
  topics: ['business', 'tech_science'],
  articles: 20,
  refreshed_at: new Date(Date.now() - 2 * 3600_000).toISOString(),
  samples: [
    { title: `${name} headline 1`, source_name: 'Example News', category: 'business' },
    { title: `${name} headline 2`, source_name: 'Example Post', category: 'business' },
  ],
});
const STATS: StatsResponse = {
  request_id: 'r',
  model_version: 'v0.2.0+rank-3',
  ingestion: {
    last_run_finished: new Date(Date.now() - 12 * 60_000).toISOString(),
    last_run_status: 'ok',
    runs_24h: 96,
    runs_ok_24h: 95,
    success_rate_24h: 95 / 96,
    snapshot_built_at: new Date(Date.now() - 5 * 60_000).toISOString(),
  },
  articles_24h: 2412,
  sources_24h: 310,
  median_freshness_lag_s: 840,
  trending_stories: 7,
  personas: 5,
  instance_uptime_s: 3 * 3600 + 300,
  credit: GDELT,
};
const TOPICS = {
  request_id: 'r',
  model_version: 'm',
  topics: [
    { id: 'business', name: 'Business' },
    { id: 'tech_science', name: 'Tech & science' },
  ],
};

type Handler = (url: URL, r: Request) => Response | Promise<Response>;
let routes: Record<string, Handler>;
let requests: string[];
let makeAuth: () => AuthProvider;

const transport: Transport = {
  async send(events) {
    return { kind: 'done', ids: events.map((e) => e.event_id) };
  },
};

beforeEach(() => {
  let n = 0;
  makeAuth = () => createDevAuth(memoryKV(), () => [UID, NEW_UID][Math.min(n++, 1)]);
  requests = [];
  useToasts.setState({ toasts: [] });
  usePrefs.setState({ theme: 'system' });
  routes = {
    '/v1/me': () => json(me()),
    '/v1/topics': () => json(TOPICS),
    '/v1/stats': () => json(STATS),
    '/v1/demo/personas': () =>
      json({
        request_id: 'r',
        model_version: 'm',
        personas: [persona('markets-tech', 'Markets & tech'), persona('us-news', 'U.S. news')],
      }),
  };
  jest.spyOn(globalThis, 'fetch').mockImplementation(async (input: RequestInfo | URL) => {
    const r = input as Request;
    const url = new URL(r.url);
    requests.push(`${r.method} ${url.pathname}${url.search}`);
    const key = `${r.method} ${url.pathname}`;
    const h = routes[key] ?? routes[url.pathname];
    return h ? h(url, r) : apiError(404, 'not_found');
  });
});
afterEach(() => jest.restoreAllMocks());

let queueRef: ReturnType<typeof useEventQueue> = null;
function QueueSpy() {
  queueRef = useEventQueue();
  return null;
}

function Providers({ children }: PropsWithChildren) {
  const client = createQueryClient();
  return (
    <ThemeProvider>
      <QueryClientProvider client={client}>
        <SessionProvider factory={makeAuth} apiUrl="http://api.test">
          <EventsProvider store={memoryKV()} transport={transport}>
            <QueueSpy />
            {children}
            <ToastHost />
          </EventsProvider>
        </SessionProvider>
      </QueryClientProvider>
    </ThemeProvider>
  );
}

async function open(initialUrl: string) {
  await renderRouter(
    {
      _layout: () => (
        <Providers>
          <Slot />
        </Providers>
      ),
      index: () => <Text>gate screen</Text>,
      feed: () => <Text>feed screen</Text>,
      onboarding: () => <Text>onboarding screen</Text>,
      personas: Personas,
      account: Account,
      system: System,
      about: About,
      privacy: Privacy,
    },
    { initialUrl },
  );
}

const linkNamed = (text: string) => screen.getByRole('link', { name: `${text} ${NEW_TAB}` });

describe('demo readers', () => {
  it('lists each persona with topics, sample headlines and the GDELT credit', async () => {
    await open('/personas');
    const card = await screen.findByTestId('persona-markets-tech');
    expect(within(card).getByRole('heading', { name: 'Markets & tech' })).toBeTruthy();
    expect(within(card).getByText('Tech & science')).toBeTruthy();
    expect(within(card).getByText('Markets & tech headline 1')).toBeTruthy();
    expect(within(card).getByText(/20 stories · picked 2 hours ago/)).toBeTruthy();
    expect(linkNamed(GDELT.text)).toBeTruthy();
  });

  it('activates directly when there is nothing to replace', async () => {
    routes['POST /v1/demo/personas/markets-tech/activate'] = () =>
      json(me({ active_persona: 'markets-tech' }));
    await open('/personas');
    await fireEvent.press(await screen.findByTestId('activate-markets-tech'));
    expect(await screen.findByText('feed screen')).toBeTruthy();
    expect(requests).toContain('POST /v1/demo/personas/markets-tech/activate?replace=false');
    expect(screen.getByText('You’re reading as Markets & tech')).toBeTruthy();
  });

  it('asks before replacing a reader’s own history (409), then sends replace=true', async () => {
    routes['POST /v1/demo/personas/markets-tech/activate'] = (url) =>
      url.searchParams.get('replace') === 'true'
        ? json(me({ active_persona: 'markets-tech' }))
        : apiError(409, 'history_exists');
    await open('/personas');
    await fireEvent.press(await screen.findByTestId('activate-markets-tech'));
    expect(await screen.findByText(replaceQuestion('Markets & tech'))).toBeTruthy();
    expect(screen.queryByText('feed screen')).toBeNull();
    await fireEvent.press(screen.getByRole('button', { name: 'Replace' }));
    expect(await screen.findByText('feed screen')).toBeTruthy();
    expect(requests.filter((r) => r.includes('/activate'))).toEqual([
      'POST /v1/demo/personas/markets-tech/activate?replace=false',
      'POST /v1/demo/personas/markets-tech/activate?replace=true',
    ]);
  });

  it('"Keep mine" closes the question and changes nothing', async () => {
    routes['POST /v1/demo/personas/markets-tech/activate'] = () => apiError(409, 'history_exists');
    await open('/personas');
    await fireEvent.press(await screen.findByTestId('activate-markets-tech'));
    await fireEvent.press(await screen.findByRole('button', { name: 'Keep mine' }));
    await waitFor(() => expect(screen.queryByText(replaceQuestion('Markets & tech'))).toBeNull());
    expect(requests.filter((r) => r.includes('replace=true'))).toEqual([]);
  });

  it('a 503 (no snapshot yet) says the demo readers are warming up', async () => {
    routes['/v1/demo/personas'] = () => apiError(503, 'unavailable', { 'Retry-After': '0' });
    await open('/personas');
    expect(await screen.findByText(WARMING_UP)).toBeTruthy();
  });
});

describe('account', () => {
  it('shows a guest, their topics and history, and offers Google', async () => {
    await open('/account');
    expect(await screen.findByTestId('guest-status')).toHaveTextContent(/reading as a guest/);
    expect(await screen.findByTestId('followed-topics')).toHaveTextContent(
      'Topics: world, sports, health.',
    );
    expect(screen.getByText(/3 stories in your reading history/)).toBeTruthy();
    expect(screen.getByRole('button', { name: 'Continue with Google' })).toBeTruthy();
  });

  it('on the local stack, Google explains it’s for the live site', async () => {
    await open('/account');
    await fireEvent.press(await screen.findByTestId('link-google'));
    expect(await screen.findByText(AUTH_MESSAGES.not_available)).toBeTruthy();
  });

  it('links Google through the Supabase provider (mocked client)', async () => {
    const sb = fakeSupabase();
    makeAuth = () =>
      createSupabaseAuth({
        getClient: async () => sb.client,
        redirectTo: () => 'https://site/firstread/account',
      });
    await open('/account');
    await fireEvent.press(await screen.findByTestId('link-google'));
    await waitFor(() =>
      expect(sb.calls).toContain('linkIdentity:google:https://site/firstread/account'),
    );
    expect(screen.queryByTestId('auth-error')).toBeNull();
    expect(screen.queryByTestId('reset-identity')).toBeNull(); // dev only
  });

  it('identity_already_exists: explains, then offers to switch after a warning', async () => {
    const sb = fakeSupabase({
      linkIdentity: () =>
        Promise.resolve({ data: null, error: { code: 'identity_already_exists' } }),
    });
    makeAuth = () =>
      createSupabaseAuth({ getClient: async () => sb.client, redirectTo: () => 'https://s/a' });
    await open('/account');
    await fireEvent.press(await screen.findByTestId('link-google'));
    expect(await screen.findByText(AUTH_MESSAGES.identity_already_exists)).toBeTruthy();
    await fireEvent.press(screen.getByTestId('switch-google'));
    expect(await screen.findByText(SWITCH_WARNING)).toBeTruthy();
    await fireEvent.press(screen.getByRole('button', { name: 'Switch account' }));
    await waitFor(() => expect(sb.calls).toContain('signInWithOAuth:google'));
  });

  it('shows an error that came back in the OAuth redirect', async () => {
    const sb = fakeSupabase();
    makeAuth = () =>
      createSupabaseAuth({
        getClient: async () => sb.client,
        redirectTo: () => 'https://s/a',
        currentUrl: () => 'https://s/firstread/account?error_code=manual_linking_disabled',
        cleanUrl: () => undefined,
      });
    await open('/account');
    expect(await screen.findByText(AUTH_MESSAGES.manual_linking_disabled)).toBeTruthy();
  });

  it('a Google-linked reader sees that, with no link button', async () => {
    const sb = fakeSupabase({
      getSession: () =>
        Promise.resolve({
          data: { session: { access_token: 't', user: { id: UID, is_anonymous: false } } },
          error: null,
        }),
    });
    makeAuth = () =>
      createSupabaseAuth({ getClient: async () => sb.client, redirectTo: () => 'https://s/a' });
    await open('/account');
    expect(await screen.findByTestId('signed-in')).toBeTruthy();
    expect(screen.queryByTestId('link-google')).toBeNull();
  });

  it('lists hidden sources and shows one again', async () => {
    routes['/v1/me'] = () => json(me({ hidden_sources: ['odd.example.com'] }));
    routes['DELETE /v1/me/hidden-sources/odd.example.com'] = () => json(me());
    await open('/account');
    await fireEvent.press(
      await screen.findByRole('button', { name: 'Show odd.example.com again' }),
    );
    expect(await screen.findByTestId('no-hidden')).toBeTruthy();
    expect(screen.getByText('Stories from odd.example.com will show again')).toBeTruthy();
    expect(requests).toContain('DELETE /v1/me/hidden-sources/odd.example.com');
  });

  it('the theme choice is a radio group and is applied', async () => {
    await open('/account');
    const dark = await screen.findByRole('radio', { name: 'Dark' });
    expect(screen.getByRole('radio', { name: 'System' })).toBeChecked();
    await fireEvent.press(dark);
    expect(usePrefs.getState().theme).toBe('dark');
    expect(screen.getByRole('radio', { name: 'Dark' })).toBeChecked();
  });

  it('a demo reader sees which one, by name', async () => {
    routes['/v1/me'] = () => json(me({ active_persona: 'us-news' }));
    await open('/account');
    expect(await screen.findByTestId('persona-status')).toHaveTextContent(
      'You’re reading as the demo reader U.S. news.',
    );
  });

  it('delete my data retries a 503, drops queued events, then starts a new reader', async () => {
    let calls = 0;
    routes['DELETE /v1/me'] = () =>
      ++calls === 1 ? apiError(503, 'unavailable', { 'Retry-After': '0' }) : json(null, 204);
    await open('/account');
    await screen.findByTestId('followed-topics');
    await waitFor(() => expect(queueRef).not.toBeNull());
    await act(async () => {
      queueRef!.enqueue({ type: 'impression', article_id: UID, position: 0, request_id: 'r' });
    });
    await fireEvent.press(screen.getByTestId('delete-data'));
    await fireEvent.press(await screen.findByRole('button', { name: 'Delete' }));
    expect(await screen.findByText('gate screen')).toBeTruthy();
    expect(calls).toBe(2);
    expect(screen.getByText(DELETED)).toBeTruthy();
    // The new reader's requests carry the new id; the old queue has nothing left.
    expect(queueRef?.pending ?? []).toEqual([]);
  });

  it('delete my data that keeps failing says so and offers to try again', async () => {
    routes['DELETE /v1/me'] = () => apiError(500, 'internal');
    await open('/account');
    await fireEvent.press(await screen.findByTestId('delete-data'));
    await fireEvent.press(await screen.findByRole('button', { name: 'Delete' }));
    expect(await screen.findByTestId('delete-failed')).toHaveTextContent(/isn’t fully deleted yet/);
    expect(screen.getByRole('button', { name: 'Try again' })).toBeTruthy();
  });

  it('reset demo identity asks first, then starts a new reader', async () => {
    await open('/account');
    await screen.findByTestId('followed-topics');
    await fireEvent.press(await screen.findByTestId('reset-identity'));
    await fireEvent.press(await screen.findByRole('button', { name: 'Start over' }));
    expect(await screen.findByText('gate screen')).toBeTruthy();
    expect(requests).not.toContain('DELETE /v1/me');
  });
});

describe('system', () => {
  it('shows the live stats, the GDELT credit from the API, the diagram and links', async () => {
    await open('/system');
    expect(await screen.findByTestId('stat-last-run')).toHaveTextContent(/OK.*12 minutes ago/);
    expect(screen.getByTestId('stat-runs')).toHaveTextContent(
      /95 \/ 96.*99% completed, partial included/,
    );
    expect(screen.getByTestId('stat-articles')).toHaveTextContent(/2,412.*From 310 sources/);
    expect(screen.getByTestId('stat-freshness')).toHaveTextContent(/14 min/);
    expect(screen.getByTestId('stat-model')).toHaveTextContent(/v0\.2\.0\+rank-3/);
    expect(linkNamed(GDELT.text)).toBeTruthy();
    expect(screen.getByRole('img', { name: DIAGRAM_DESCRIPTION })).toBeTruthy();
    expect(linkNamed('Model card')).toBeTruthy();
    expect(linkNamed('Source code on GitHub')).toBeTruthy();
  });

  it('a failed stats request shows the error and a retry', async () => {
    routes['/v1/stats'] = () => apiError(500, 'internal');
    await open('/system');
    expect(await screen.findByTestId('stats-error')).toBeTruthy();
  });

  it('tiles handle a store with no runs yet', () => {
    const tiles = statTiles({
      ...STATS,
      ingestion: {
        ...STATS.ingestion,
        last_run_finished: null,
        last_run_status: null,
        runs_24h: 0,
        runs_ok_24h: 0,
        success_rate_24h: null,
      },
      median_freshness_lag_s: null,
    });
    expect(tiles.find((t) => t.key === 'last-run')).toMatchObject({ value: 'None yet' });
    expect(tiles.find((t) => t.key === 'runs')).toMatchObject({ detail: 'No runs yet' });
    expect(tiles.find((t) => t.key === 'freshness')).toMatchObject({ value: 'No data' });
  });

  it('durations read naturally', () => {
    expect(duration(45)).toBe('45 s');
    expect(duration(840)).toBe('14 min');
    expect(duration(3 * 3600 + 300)).toBe('3 h 5 min');
    expect(duration(2 * 86400)).toBe('2 d');
    expect(duration(2 * 86400 + 4 * 3600)).toBe('2 d 4 h');
  });
});

describe('privacy and about', () => {
  it('privacy lists what is stored and for how long, and how to delete it', async () => {
    await open('/privacy');
    expect(await screen.findByRole('heading', { name: 'Privacy' })).toBeTruthy();
    expect(screen.getByText('7 days')).toBeTruthy();
    expect(screen.getByText(/No ads, no third-party trackers/)).toBeTruthy();
    expect(screen.getByText(/the publisher can see your IP/)).toBeTruthy();
  });

  it('about credits GDELT, the RSS publishers, MIND and the font’s OFL', async () => {
    await open('/about');
    expect(await screen.findByTestId('about-gdelt')).toBeTruthy();
    expect(linkNamed(GDELT.text)).toBeTruthy();
    for (const p of RSS_PUBLISHERS) expect(screen.getByText(new RegExp(p))).toBeTruthy();
    expect(screen.getByText(/MIND, the Microsoft News Dataset/)).toBeTruthy();
    expect(screen.getByTestId('font-credit')).toHaveTextContent(/SIL\s+Open Font License 1\.1/);
    expect(linkNamed('The font’s license (OFL 1.1)')).toBeTruthy();
  });

  it('the publisher list matches the feeds enabled in ingestion', () => {
    const yaml = readFileSync(
      join(__dirname, '../../../services/ingest/src/firstread_ingest/config/feeds.yaml'),
      'utf8',
    );
    const enabled = new Set<string>();
    for (const block of yaml.split(/\n\s*-\s+(?=\w+:)/)) {
      const name = /source_name:\s*"([^"]+)"/.exec(block)?.[1];
      if (name && /\n\s*enabled:\s*true\b/.test(block)) enabled.add(name);
    }
    expect([...enabled].sort()).toEqual([...RSS_PUBLISHERS].sort());
  });
});
