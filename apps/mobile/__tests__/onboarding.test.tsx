import { QueryClientProvider } from '@tanstack/react-query';
import { fireEvent, screen, waitFor } from '@testing-library/react-native';
import { Slot } from 'expo-router';
import { renderRouter } from 'expo-router/testing-library';
import type { PropsWithChildren } from 'react';
import { Text } from 'react-native';

import { createQueryClient } from '@/api/queries';
import Onboarding from '@/app/onboarding';
import { createDevAuth } from '@/auth/dev';
import { needsOnboarding } from '@/features/me';
import { SessionProvider } from '@/providers/session';
import { memoryKV } from '@/storage/kv';
import { ThemeProvider } from '@/ui';

const TOPICS = [
  { id: 'us', name: 'U.S.' },
  { id: 'world', name: 'World' },
  { id: 'sports', name: 'Sports' },
  { id: 'health', name: 'Health' },
];
const UID = '11111111-2222-4333-8444-555555555555';
const json = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } });

let requests: { method: string; path: string; body: string; devUser: string | null }[];

beforeEach(() => {
  requests = [];
  jest.spyOn(globalThis, 'fetch').mockImplementation(async (input: RequestInfo | URL) => {
    const r = input as Request;
    const path = new URL(r.url).pathname;
    requests.push({
      method: r.method,
      path,
      body: await r.clone().text(),
      devUser: r.headers.get('X-Dev-User'),
    });
    if (path === '/v1/topics') return json({ topics: TOPICS, request_id: 'r', model_version: 'm' });
    if (path === '/v1/me/topics') {
      const topics = JSON.parse(await r.text()).topics;
      return json({
        request_id: 'r',
        model_version: 'm',
        user_id: UID,
        is_anonymous: true,
        topics,
        hidden_sources: [],
        active_persona: null,
        history_count: 0,
        recent_history: [],
      });
    }
    return json({ error: { code: 'not_found', message: 'nope' }, request_id: 'r' }, 404);
  });
});
afterEach(() => jest.restoreAllMocks());

function Providers({ children }: PropsWithChildren) {
  const client = createQueryClient();
  return (
    <ThemeProvider>
      <QueryClientProvider client={client}>
        <SessionProvider
          factory={() => createDevAuth(memoryKV(), () => UID)}
          apiUrl="http://api.test"
        >
          {children}
        </SessionProvider>
      </QueryClientProvider>
    </ThemeProvider>
  );
}

async function open() {
  await renderRouter(
    {
      _layout: () => (
        <Providers>
          <Slot />
        </Providers>
      ),
      onboarding: Onboarding,
      feed: () => <Text>the feed</Text>,
      personas: () => <Text>demo readers</Text>,
    },
    { initialUrl: '/onboarding' },
  );
}

describe('onboarding', () => {
  it('needs at least 3 topics, then saves them and opens the feed', async () => {
    await open();
    const cont = screen.getByTestId('onboarding-continue');
    await screen.findByRole('checkbox', { name: 'Sports' });
    expect(cont).toBeDisabled();
    for (const name of ['Sports', 'Health'])
      await fireEvent.press(screen.getByRole('checkbox', { name }));
    expect(screen.getByText('2 picked, 1 more to go')).toBeTruthy();
    expect(screen.getByRole('checkbox', { name: 'Sports' })).toBeChecked();
    expect(cont).toBeDisabled();
    await fireEvent.press(screen.getByRole('checkbox', { name: 'World' }));
    expect(screen.getByTestId('onboarding-continue')).toBeEnabled();
    await fireEvent.press(screen.getByTestId('onboarding-continue'));
    await screen.findByText('the feed');
    const put = requests.find((r) => r.method === 'PUT');
    expect(put?.path).toBe('/v1/me/topics');
    expect(JSON.parse(put!.body)).toEqual({ topics: ['sports', 'health', 'world'] });
    expect(put?.devUser).toBe(UID);
  });

  it('unselects a topic on a second press', async () => {
    await open();
    const sports = await screen.findByRole('checkbox', { name: 'Sports' });
    await fireEvent.press(sports);
    await fireEvent.press(screen.getByRole('checkbox', { name: 'Sports' }));
    expect(screen.getByRole('checkbox', { name: 'Sports' })).not.toBeChecked();
  });

  it('offers a demo reader instead', async () => {
    await open();
    await fireEvent.press(screen.getByTestId('onboarding-demo'));
    await waitFor(() => expect(screen.getByText('demo readers')).toBeTruthy());
  });
});

describe('the entry gate', () => {
  it('onboards readers with neither topics nor a persona', () => {
    expect(needsOnboarding({ topics: [], active_persona: null })).toBe(true);
    expect(needsOnboarding({ topics: ['us', 'world', 'sports'], active_persona: null })).toBe(
      false,
    );
    expect(needsOnboarding({ topics: [], active_persona: 'markets-tech' })).toBe(false);
  });
});
