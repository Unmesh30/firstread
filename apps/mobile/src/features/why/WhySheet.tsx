// "Why am I seeing this?": the reasons, the blend's parts (model + fresh + trend), how
// old the story is, how many outlets carry it, whether it's in a followed topic, and
// the reader's two closest reads. `diversity` is never shown (ADR 0018).
import { useQuery } from '@tanstack/react-query';
import { StyleSheet, View } from 'react-native';

import { call } from '@/api/client';
import { isApiError } from '@/api/errors';
import { keys } from '@/api/queries';
import type { WhyResponse } from '@/api/schema-types';
import { useTopics } from '@/features/me';
import { ageInWords } from '@/lib/time';
import { useApi } from '@/providers/session';
import {
  Banner,
  ComponentBar,
  Divider,
  EmptyState,
  ErrorState,
  LoadingList,
  Sheet,
  Text,
} from '@/ui';
import { space } from '@/ui/tokens';

import { GdeltCredit } from '../credits';
import { DEGRADED_NOTE } from '../feed/notices';

export const WHY_TITLE = 'Why am I seeing this?';
const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
const WHY_STALE_MS = 60 * 1000;

export function useWhy(id: string | undefined) {
  const api = useApi();
  const valid = !!id && UUID.test(id);
  return useQuery({
    queryKey: keys.why(id ?? ''),
    queryFn: () =>
      call(api!.GET('/v1/articles/{article_id}/why', { params: { path: { article_id: id! } } })),
    enabled: api != null && valid,
    staleTime: WHY_STALE_MS,
  });
}

export function WhyDetails({ why, topicName }: { why: WhyResponse; topicName: string }) {
  const outlets = why.outlets === 1 ? '1 outlet' : `${why.outlets} outlets`;
  return (
    <View style={styles.details}>
      <Text variant="lead" serif weight="600">
        {why.title}
      </Text>
      {why.degraded.length ? <Banner tone="info" message={DEGRADED_NOTE} /> : null}
      {why.reasons.length ? (
        <View role="list" style={styles.list}>
          {why.reasons.map((r) => (
            <View key={r.code} role="listitem" style={styles.bullet}>
              <Text aria-hidden tone="accent">
                •
              </Text>
              <Text style={styles.flex}>{r.text}</Text>
            </View>
          ))}
        </View>
      ) : null}
      <Divider />
      <Text variant="small" weight="600" role="heading" aria-level={3}>
        How the score adds up
      </Text>
      <ComponentBar values={why.components} />
      <Divider />
      <View role="list" style={styles.list} testID="why-facts">
        <Text role="listitem">Published about {ageInWords(why.age_hours)} ago</Text>
        <Text role="listitem">Covered by {outlets}</Text>
        <Text role="listitem">
          {why.followed_topic ? `In ${topicName}, a topic you follow` : `Topic: ${topicName}`}
        </Text>
      </View>
      {why.similar_reads.length ? (
        <>
          <Divider />
          <Text variant="small" weight="600" role="heading" aria-level={3}>
            Close to stories you read
          </Text>
          <View role="list" style={styles.list} testID="why-similar">
            {why.similar_reads.slice(0, 2).map((s) => (
              <View key={s.article_id} role="listitem" style={styles.similar}>
                <Text style={styles.flex}>{s.title}</Text>
                <Text variant="small" tone="muted">
                  {Math.round(s.similarity * 100)}% similar
                </Text>
              </View>
            ))}
          </View>
        </>
      ) : null}
      <GdeltCredit prefix="Headline:" />
    </View>
  );
}

export function WhySheet({
  id,
  onClose,
  level = 2,
}: {
  id: string | undefined;
  onClose: () => void;
  level?: 1 | 2;
}) {
  const why = useWhy(id);
  const topics = useTopics();
  const valid = !!id && UUID.test(id);
  const topicName = (c: string) => topics.data?.topics.find((t) => t.id === c)?.name ?? c;

  let content;
  if (!valid) content = <EmptyState title="No story given" body="This link is missing a story." />;
  else if (isApiError(why.error, 'not_found') || isApiError(why.error, 'unknown_article')) {
    content = (
      <EmptyState
        title="This story is no longer available"
        body="Stories leave FirstRead after two weeks."
        testID="why-gone"
      />
    );
  } else if (why.error) {
    content = <ErrorState error={why.error} onRetry={() => void why.refetch()} />;
  } else if (why.data) {
    content = <WhyDetails why={why.data} topicName={topicName(why.data.category)} />;
  } else content = <LoadingList count={1} label="Loading the explanation" />;

  return (
    <Sheet visible title={WHY_TITLE} onClose={onClose} level={level} testID="why-sheet">
      {content}
    </Sheet>
  );
}

const styles = StyleSheet.create({
  flex: { flex: 1 },
  details: { gap: space.md },
  list: { gap: space.sm },
  bullet: { flexDirection: 'row', gap: space.sm },
  similar: { flexDirection: 'row', gap: space.md, alignItems: 'baseline' },
});
