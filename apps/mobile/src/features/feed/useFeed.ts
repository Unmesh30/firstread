// The ranked feed: cursor pages of 20 from GET /v1/feed.
//
// Never refetched on its own (mount, focus, reconnect): every refetch would be a new
// first page, 7 Redis commands (ADR 0011). Refreshing resets the query to one new first
// page. An expired or unreadable cursor starts the feed over, with a toast. The query
// waits for /v1/me to show an onboarded reader.
import { useInfiniteQuery, useQueryClient } from '@tanstack/react-query';
import { useCallback, useMemo } from 'react';

import { call } from '@/api/client';
import { isApiError } from '@/api/errors';
import { keys } from '@/api/queries';
import type { FeedItem, FeedResponse } from '@/api/schema-types';
import { needsOnboarding, useMe } from '@/features/me';
import { useApi } from '@/providers/session';
import { useToasts } from '@/ui/toast';

import { useFeedLocal } from './store';

export const PAGE_SIZE = 20;
/** The API keeps a slate 30 min (ADR 0017); a feed kept longer would only expire. */
const FEED_GC_MS = 30 * 60 * 1000;

export interface FeedRow {
  item: FeedItem;
  /** The feed response that served it: impressions join served_slates on it. */
  requestId: string;
  /** Its place within that page. */
  position: number;
}

export function flattenPages(
  pages: FeedResponse[],
  hiddenArticles: ReadonlySet<string>,
  hiddenSources: ReadonlySet<string>,
): FeedRow[] {
  const seen = new Set<string>();
  const rows: FeedRow[] = [];
  for (const page of pages) {
    page.items.forEach((item, position) => {
      if (seen.has(item.id)) return;
      seen.add(item.id);
      if (hiddenArticles.has(item.id) || hiddenSources.has(item.source_domain)) return;
      rows.push({ item, requestId: page.request_id, position });
    });
  }
  return rows;
}

export function isCursorError(e: unknown): boolean {
  return isApiError(e, 'cursor_expired') || isApiError(e, 'invalid_cursor');
}

export function useFeed() {
  const api = useApi();
  const qc = useQueryClient();
  // Only for an onboarded reader: a feed fetched before the topics are saved would be
  // the wrong one, and cost a first page.
  const me = useMe();
  const onboarded = me.data != null && !needsOnboarding(me.data);
  const hiddenArticles = useFeedLocal((s) => s.hiddenArticles);
  const hiddenSources = useFeedLocal((s) => s.hiddenSources);
  const query = useInfiniteQuery({
    queryKey: keys.feed,
    queryFn: ({ pageParam }) =>
      call(
        api!.GET('/v1/feed', {
          params: { query: { limit: PAGE_SIZE, cursor: pageParam ?? undefined } },
        }),
      ),
    initialPageParam: null as string | null,
    getNextPageParam: (last) => last.next_cursor,
    enabled: api != null && onboarded,
    staleTime: Infinity,
    gcTime: FEED_GC_MS,
    refetchOnMount: false,
    refetchOnWindowFocus: false,
    refetchOnReconnect: false,
  });

  const pages = query.data?.pages;
  const rows = useMemo(
    () => flattenPages(pages ?? [], hiddenArticles, hiddenSources),
    [pages, hiddenArticles, hiddenSources],
  );

  /** One new first page (pull-to-refresh, "New stories", end of feed). */
  const refresh = useCallback(() => qc.resetQueries({ queryKey: keys.feed }), [qc]);

  const { hasNextPage, isFetchingNextPage, isFetching, fetchNextPage } = query;
  const loadMore = useCallback(async () => {
    if (!hasNextPage || isFetchingNextPage || isFetching) return;
    const result = await fetchNextPage();
    if (isCursorError(result.error)) {
      const rid = isApiError(result.error) ? result.error.requestId : null;
      console.info(`feed: ${(result.error as Error).message} (request ${rid}); starting over`);
      await qc.resetQueries({ queryKey: keys.feed });
      useToasts.getState().show({ message: 'Your feed was refreshed' });
    }
  }, [hasNextPage, isFetchingNextPage, isFetching, fetchNextPage, qc]);

  const last = pages?.[pages.length - 1];
  const empty = pages != null && pages.length === 1 && pages[0].items.length === 0;
  return {
    query,
    me,
    /** Known to need onboarding (the screen sends them there). */
    needsOnboarding: me.data != null && !onboarded,
    rows,
    refresh,
    loadMore,
    /** The newest page's degraded parts (worked around; shown as a quiet note). */
    degraded: last?.degraded ?? [],
    empty,
    end: last != null && last.next_cursor == null && !empty,
  };
}
