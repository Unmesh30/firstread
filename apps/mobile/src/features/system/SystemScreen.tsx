// System: the live numbers behind FirstRead, from GET /v1/stats (served from the pool
// snapshot, never a database read; cached 60 s like its Cache-Control), plus how it
// fits together, the model card and the source code.
import { useQuery, useQueryClient, type InfiniteData } from '@tanstack/react-query';
import { StyleSheet, View } from 'react-native';

import { fetchers, keys, staleTimes } from '@/api/queries';
import type { FeedResponse, StatsResponse } from '@/api/schema-types';
import { duration, timeAgo } from '@/lib/time';
import { useApi } from '@/providers/session';
import { ErrorState, ExternalLink, Screen, StatTile, Text } from '@/ui';
import { space } from '@/ui/tokens';

import { GdeltCredit, MODEL_CARD_URL, MODELS_URL } from '../credits';
import { FooterLinks } from '../FooterLinks';
import { ArchitectureDiagram } from './ArchitectureDiagram';

export const SYSTEM_TITLE = 'System';

const STATUS: Record<string, string> = { ok: 'OK', partial: 'Partial', failed: 'Failed' };

export interface Tile {
  key: string;
  label: string;
  value: string;
  detail?: string;
}

/** The stat tiles, in reading order. */
export function statTiles(s: StatsResponse, now: number = Date.now()): Tile[] {
  const ing = s.ingestion;
  const last = ing.last_run_finished ? timeAgo(ing.last_run_finished, now) : '';
  const rate = ing.success_rate_24h;
  return [
    {
      key: 'last-run',
      label: 'Last ingestion run',
      value: ing.last_run_status
        ? (STATUS[ing.last_run_status] ?? ing.last_run_status)
        : 'None yet',
      detail: last ? `Finished ${last}` : 'No run has finished yet',
    },
    {
      key: 'runs',
      label: 'Runs OK, last 24 h',
      value: `${ing.runs_ok_24h} / ${ing.runs_24h}`,
      // The API counts partial runs (some sources failed, the rest stored) as completed.
      detail:
        rate == null
          ? 'No runs yet'
          : `${Math.round(rate * 1000) / 10}% completed, partial included`,
    },
    {
      key: 'articles',
      label: 'New articles, 24 h',
      value: s.articles_24h.toLocaleString('en-US'),
      detail: `From ${s.sources_24h.toLocaleString('en-US')} sources`,
    },
    {
      key: 'freshness',
      label: 'Freshness',
      value: s.median_freshness_lag_s == null ? 'No data' : duration(s.median_freshness_lag_s),
      detail: 'Median time from GDELT seeing a story to it being recommendable',
    },
    {
      key: 'trending',
      label: 'Trending stories',
      value: String(s.trending_stories),
      detail: 'Stories several outlets are covering',
    },
    {
      key: 'model',
      label: 'Model and ranking',
      value: s.model_version,
      detail: `${s.personas} demo readers · API instance up ${duration(s.instance_uptime_s)}`,
    },
  ];
}

/** The parts the newest feed page worked around (its `degraded` codes), if any. */
function useLastDegraded(): string[] {
  const qc = useQueryClient();
  const feed = qc.getQueryData<InfiniteData<FeedResponse>>(keys.feed);
  return feed?.pages[feed.pages.length - 1]?.degraded ?? [];
}

export function SystemScreen() {
  const api = useApi();
  const stats = useQuery({
    queryKey: keys.stats,
    queryFn: () => fetchers.stats(api!),
    enabled: api != null,
    staleTime: staleTimes.stats,
  });
  const degraded = useLastDegraded();
  const s = stats.data;
  const tiles = s ? statTiles(s) : null;

  return (
    <Screen title={SYSTEM_TITLE} testID="screen-system">
      <View style={styles.section}>
        <Text variant="headline" serif role="heading" aria-level={1}>
          {SYSTEM_TITLE}
        </Text>
        <Text variant="lead" tone="muted">
          Live numbers from the news pipeline and the recommender behind your feed.
        </Text>
      </View>

      {stats.error ? (
        <ErrorState error={stats.error} onRetry={() => void stats.refetch()} testID="stats-error" />
      ) : (
        <View style={styles.section}>
          <View style={styles.tiles} testID="stats">
            {tiles
              ? tiles.map((t) => (
                  <StatTile
                    key={t.key}
                    label={t.label}
                    value={t.value}
                    detail={t.detail}
                    testID={`stat-${t.key}`}
                  />
                ))
              : ['a', 'b', 'c', 'd', 'e', 'f'].map((k) => <StatTile key={k} label="Loading" />)}
          </View>
          {s ? (
            <Text variant="caption" tone="muted">
              Candidate pools rebuilt {timeAgo(s.ingestion.snapshot_built_at) || 'recently'}.
            </Text>
          ) : null}
          <GdeltCredit credit={s?.credit} prefix="Live headlines:" />
        </View>
      )}

      {degraded.length ? (
        <Text variant="small" tone="muted" testID="degraded-codes">
          Your last feed page worked around: {degraded.join(', ')}.
        </Text>
      ) : null}

      <View style={styles.section}>
        <Text variant="title" serif role="heading" aria-level={2}>
          How it fits together
        </Text>
        <ArchitectureDiagram />
        <Text tone="muted" variant="small">
          New stories are understood from their headline the moment they arrive, so they can be
          ranked for the right readers before anyone has clicked.
        </Text>
      </View>

      <View style={styles.section}>
        <Text variant="title" serif role="heading" aria-level={2}>
          The model
        </Text>
        <Text>
          A small attention model turns the stories you read into a reader profile, in the same
          space as each story’s text embedding. It was trained and evaluated offline on the MIND
          research dataset; no MIND articles are shown here.
        </Text>
        <View style={styles.links}>
          <ExternalLink href={MODEL_CARD_URL}>Model card</ExternalLink>
          <ExternalLink href={MODELS_URL}>Model files on Hugging Face</ExternalLink>
        </View>
      </View>
      <FooterLinks />
    </Screen>
  );
}

const styles = StyleSheet.create({
  section: { gap: space.md },
  tiles: { flexDirection: 'row', flexWrap: 'wrap', gap: space.md },
  links: { gap: space.sm, alignItems: 'flex-start' },
});
