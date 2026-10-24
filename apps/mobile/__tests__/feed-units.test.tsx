import { fireEvent, render, screen } from '@testing-library/react-native';

import type { FeedItem, FeedResponse } from '@/api/schema-types';
import { flattenPages, isCursorError } from '@/features/feed/useFeed';
import { ApiError } from '@/api/errors';
import { ageInWords, timeAgo } from '@/lib/time';
import { ArticleCard, ThemeProvider } from '@/ui';
import { shares } from '@/ui/ComponentBar';

const NOW = Date.parse('2026-10-24T12:00:00Z');

describe('timeAgo', () => {
  it.each([
    [30, 'just now'],
    [60, '1 minute ago'],
    [12 * 60, '12 minutes ago'],
    [3600, '1 hour ago'],
    [5 * 3600 + 59, '5 hours ago'],
    [86_400 * 3, '3 days ago'],
  ])('%s s → %s', (s, text) => {
    expect(timeAgo(new Date(NOW - s * 1000).toISOString(), NOW)).toBe(text);
  });
  it('is empty for a bad date and never negative', () => {
    expect(timeAgo('nope', NOW)).toBe('');
    expect(timeAgo(new Date(NOW + 60_000).toISOString(), NOW)).toBe('just now');
  });
});

describe('ageInWords', () => {
  it.each([
    [0.25, '15 minutes'],
    [0.001, '1 minute'],
    [1, '1 hour'],
    [30, '30 hours'],
    [72, '3 days'],
  ])('%s h → %s', (h, text) => expect(ageInWords(h)).toBe(text));
});

describe('component bar shares', () => {
  it('splits the positive total; negative parts draw nothing', () => {
    expect(shares({ model: 3, fresh: 1, trend: 0, final: 4 })).toEqual({
      model: 0.75,
      fresh: 0.25,
      trend: 0,
    });
    expect(shares({ model: -0.2, fresh: 0.2, trend: 0, final: 0 })).toEqual({
      model: 0,
      fresh: 1,
      trend: 0,
    });
    expect(shares({ model: 0, fresh: 0, trend: 0, final: 0 })).toEqual({
      model: 0,
      fresh: 0,
      trend: 0,
    });
  });
});

const item = (id: string, domain = 'a.example.com') => ({ id, source_domain: domain }) as FeedItem;
const page = (rid: string, items: FeedItem[]) =>
  ({ request_id: rid, items, next_cursor: null, degraded: [], model_version: 'm' }) as FeedResponse;

describe('flattenPages', () => {
  it('keeps each page’s request id and positions, skips repeats and local hides', () => {
    const rows = flattenPages(
      [
        page('r1', [item('a'), item('b', 'hidden.example.com'), item('c')]),
        page('r2', [item('c'), item('d'), item('e')]),
      ],
      new Set(['e']),
      new Set(['hidden.example.com']),
    );
    expect(rows.map((r) => [r.item.id, r.requestId, r.position])).toEqual([
      ['a', 'r1', 0],
      ['c', 'r1', 2],
      ['d', 'r2', 1],
    ]);
  });

  it('knows which errors mean "start over"', () => {
    expect(isCursorError(new ApiError(400, 'cursor_expired', 'x'))).toBe(true);
    expect(isCursorError(new ApiError(400, 'invalid_cursor', 'x'))).toBe(true);
    expect(isCursorError(new ApiError(400, 'bad_request', 'x'))).toBe(false);
  });
});

describe('ArticleCard', () => {
  const props = {
    id: 'a1',
    title: 'Ships return to port',
    sourceName: 'Harbor Daily',
    url: 'https://harbor.example.com/1',
    at: new Date(NOW - 5 * 60_000).toISOString(),
    reason: 'New in the last hour',
    now: NOW,
    onOpen: jest.fn(),
    onLike: jest.fn(),
    onHide: jest.fn(),
    onWhy: jest.fn(),
  };

  it('is a link with the headline, source and age, plus three named buttons', async () => {
    await render(
      <ThemeProvider>
        <ArticleCard {...props} imageUrl={null} liked={false} />
      </ThemeProvider>,
    );
    await fireEvent.press(
      screen.getByRole('link', { name: 'Ships return to port, Harbor Daily, 5 minutes ago' }),
    );
    expect(props.onOpen).toHaveBeenCalled();
    for (const name of [
      'Like: Ships return to port',
      'Hide: Ships return to port',
      'Why am I seeing this: Ships return to port',
    ]) {
      expect(screen.getByRole('button', { name })).toBeTruthy();
    }
    expect(screen.getByText('H', { includeHiddenElements: true })).toBeTruthy(); // the source-initial tile, no image
  });

  it('falls back to the initial tile when the thumbnail fails', async () => {
    await render(
      <ThemeProvider>
        <ArticleCard {...props} imageUrl="https://img.example.com/x.jpg" liked />
      </ThemeProvider>,
    );
    expect(screen.queryByText('H', { includeHiddenElements: true })).toBeNull();
    expect(screen.getByRole('button', { name: 'Liked: Ships return to port' })).toBeSelected();
    const img = screen.getByTestId('thumb-a1', { includeHiddenElements: true });
    await fireEvent(img, 'error', { nativeEvent: { error: '404' } });
    expect(screen.getByText('H', { includeHiddenElements: true })).toBeTruthy();
  });
});
