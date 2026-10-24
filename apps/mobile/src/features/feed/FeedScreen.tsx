// For You: the ranked feed as an endless list of cursor pages.
//
// Impressions: FlatList reports rows ≥ 50% on screen, the ViewabilityTracker counts one
// after a continuous second in the foreground, once per (request, article), and the
// event queue sends them with the next click or within a minute.
import { Redirect, router, useIsFocused } from 'expo-router';
import { useEffect, useState } from 'react';
import { FlatList, Platform, RefreshControl, StyleSheet, View, type ViewToken } from 'react-native';

import { ApiError } from '@/api/errors';
import { useEventQueue } from '@/events/EventsProvider';
import type { EventQueue } from '@/events/queue';
import { viewabilityConfig, ViewabilityTracker } from '@/events/viewability';
import { useBreakpoint } from '@/hooks/useBreakpoint';
import { useHydrated } from '@/hooks/useHydrated';
import { useForeground } from '@/lib/lifecycle';
import { useOnline } from '@/providers/AppProviders';
import { useSession } from '@/providers/session';
import {
  ArticleCard,
  Banner,
  Button,
  EmptyState,
  ErrorState,
  LoadingList,
  Screen,
  Sheet,
  SkeletonCard,
  Text,
  useTheme,
} from '@/ui';
import { layout, space } from '@/ui/tokens';

import { SessionProblem } from '../SessionProblem';
import { useArticleActions, type HideScope } from './actions';
import { BusyBanner, DegradedNote, OfflineBanner, useRetryCountdown } from './notices';
import { SidePanel } from './SidePanel';
import { useFeedLocal } from './store';
import { useFeed, type FeedRow } from './useFeed';

export const EMPTY_TITLE = 'No fresh stories yet';
export const EMPTY_BODY = 'New articles arrive every 15 minutes.';
export const END_TITLE = 'You’re all caught up';

type Viewable = { viewableItems: ViewToken<FeedRow>[] };

function useImpressions(queue: EventQueue | null, focused: boolean) {
  const foreground = useForeground() && focused;
  const [state] = useState(() => {
    const tracker = new ViewabilityTracker<FeedRow>();
    // FlatList requires the same function for its lifetime.
    const onViewableItemsChanged = ({ viewableItems }: Viewable) =>
      tracker.update(
        viewableItems
          .filter((v) => v.isViewable && v.item)
          .map((v) => [`${v.item.requestId}:${v.item.item.id}`, v.item] as [string, FeedRow]),
      );
    return { tracker, onViewableItemsChanged };
  });
  useEffect(
    () =>
      state.tracker.setSink((_, row) =>
        queue?.enqueue({
          type: 'impression',
          article_id: row.item.id,
          position: row.position,
          request_id: row.requestId,
        }),
      ),
    [state, queue],
  );
  useEffect(() => state.tracker.setForeground(foreground), [state, foreground]);
  useEffect(() => () => state.tracker.dispose(), [state]);
  return state.onViewableItemsChanged;
}

export function FeedScreen() {
  const { colors } = useTheme();
  const session = useSession();
  const online = useOnline();
  const wide = useBreakpoint().breakpoint === 'expanded';
  // In the static HTML at every width; CSS hides it below `expanded` (ui/responsive).
  const hydrated = useHydrated();
  const panel = wide || !hydrated;
  const feed = useFeed();
  const { query, rows } = feed;
  const actions = useArticleActions();
  const queue = useEventQueue();
  // Under another screen (Why, or a copy left in the stack) it counts and loads nothing.
  const focused = useIsFocused();
  const onViewableItemsChanged = useImpressions(queue, focused);
  const liked = useFeedLocal((s) => s.liked);
  const [menuFor, setMenuFor] = useState<FeedRow | null>(null);
  const countdown = useRetryCountdown(
    query.isFetching ? query.failureReason : null,
    query.failureCount,
  );
  const refreshing = query.isFetching && !query.isFetchingNextPage && query.data != null;

  const hide = (scope: HideScope) => {
    if (menuFor) actions.hide(menuFor, scope);
    setMenuFor(null);
  };

  const header = (
    <View style={styles.header}>
      <View style={styles.titleRow}>
        <Text variant="headline" serif role="heading" aria-level={1} style={styles.flex}>
          For You
        </Text>
        {Platform.OS === 'web' && query.data ? (
          <Button
            variant="secondary"
            icon="refresh"
            label="New stories"
            accessibilityHint="Loads a fresh first page"
            loading={refreshing}
            onPress={() => void feed.refresh()}
            testID="feed-refresh"
          />
        ) : null}
      </View>
      {session.status === 'error' ? <Banner tone="danger" message={session.message} /> : null}
      {!online ? <OfflineBanner /> : null}
      {countdown != null ? <BusyBanner seconds={countdown} /> : null}
      {feed.degraded.length ? <DegradedNote /> : null}
    </View>
  );

  const nextPageError =
    query.isFetchNextPageError && query.error instanceof ApiError ? query.error : null;
  const footer = query.isFetchingNextPage ? (
    <View aria-busy accessibilityLabel="Loading more stories">
      <SkeletonCard />
      <SkeletonCard />
    </View>
  ) : nextPageError ? (
    <ErrorState error={nextPageError} onRetry={() => void feed.loadMore()} testID="next-error" />
  ) : feed.end ? (
    <EmptyState
      title={END_TITLE}
      body="That’s every story we have for you right now."
      action={{ label: 'Refresh', onPress: () => void feed.refresh() }}
      testID="feed-end"
    />
  ) : null;

  // The stack can keep an unfocused copy of this screen (the root anchor); only the
  // one the reader sees may redirect.
  if (feed.needsOnboarding && focused) return <Redirect href="/onboarding" />;

  const failed = query.data == null ? (query.error ?? feed.me.error) : null;
  const retry = () => void (query.error ? query.refetch() : feed.me.refetch());
  let body;
  if (failed) {
    body =
      failed instanceof ApiError && failed.status === 401 ? (
        <SessionProblem />
      ) : (
        <ErrorState error={failed} onRetry={retry} testID="feed-error" />
      );
  } else if (feed.empty) {
    body = (
      <EmptyState
        title={EMPTY_TITLE}
        body={EMPTY_BODY}
        action={{ label: 'See the System page', onPress: () => router.push('/system') }}
        testID="feed-empty"
      />
    );
  } else if (query.data == null) {
    body = <LoadingList label="Loading your feed" />;
  }

  return (
    <Screen title="For You" scroll={false} column={false} testID="screen-feed">
      <View style={styles.row}>
        <View nativeID="main" role="main" style={styles.flex}>
          <FlatList
            data={body ? [] : rows}
            keyExtractor={(r) => r.item.id}
            renderItem={({ item: row }) => (
              <ArticleCard
                id={row.item.id}
                title={row.item.title}
                sourceName={row.item.source_name}
                url={row.item.url}
                imageUrl={row.item.image_url}
                at={row.item.published_at ?? row.item.first_seen_at}
                reason={row.item.reasons[0]?.text}
                liked={liked.has(row.item.id)}
                onOpen={() => actions.open(row)}
                onLike={() => actions.like(row)}
                onHide={() => setMenuFor(row)}
                onWhy={() => router.push({ pathname: '/why', params: { id: row.item.id } })}
              />
            )}
            ListHeaderComponent={header}
            ListFooterComponent={body ?? footer}
            onEndReached={() => (focused ? void feed.loadMore() : undefined)}
            onEndReachedThreshold={1.5}
            viewabilityConfig={viewabilityConfig}
            onViewableItemsChanged={onViewableItemsChanged}
            windowSize={7}
            removeClippedSubviews={Platform.OS !== 'web'}
            refreshControl={
              Platform.OS !== 'web' ? (
                <RefreshControl
                  refreshing={refreshing}
                  onRefresh={() => void feed.refresh()}
                  tintColor={colors.accent}
                />
              ) : undefined
            }
            style={styles.flex}
            contentContainerStyle={styles.column}
            testID="feed-list"
          />
        </View>
        {panel ? <SidePanel /> : null}
      </View>
      <Sheet
        visible={menuFor != null}
        title="Hide"
        onClose={() => setMenuFor(null)}
        testID="hide-menu"
      >
        <Text tone="muted">You can undo this for a few seconds.</Text>
        <Button
          variant="secondary"
          label="Hide this story"
          onPress={() => hide('article')}
          testID="hide-article"
        />
        <Button
          variant="secondary"
          label={`Hide ${menuFor?.item.source_name ?? 'this source'}`}
          accessibilityHint="Stops showing stories from this source. You can show it again in Account."
          onPress={() => hide('source')}
          testID="hide-source"
        />
      </Sheet>
    </Screen>
  );
}

const styles = StyleSheet.create({
  flex: { flex: 1 },
  row: { flex: 1, flexDirection: 'row', justifyContent: 'center' },
  column: {
    width: '100%',
    maxWidth: layout.maxContent,
    alignSelf: 'center',
    paddingHorizontal: layout.gutter,
    paddingTop: space.xl,
    paddingBottom: space.xxxl * 2, // room for the toast over the last card
  },
  header: { gap: space.md, paddingBottom: space.sm },
  titleRow: { flexDirection: 'row', alignItems: 'center', gap: space.md },
});
