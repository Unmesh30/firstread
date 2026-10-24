// Onboarding: follow at least 3 of the 13 topics, or try a demo reader.
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { Link, router } from 'expo-router';
import { useState } from 'react';
import { StyleSheet, View } from 'react-native';

import { call } from '@/api/client';
import { keys } from '@/api/queries';
import { FooterLinks } from '@/features/FooterLinks';
import { useMe, useTopics } from '@/features/me';
import { useApi } from '@/providers/session';
import { Banner, Button, ErrorState, Screen, Skeleton, Text, TopicChip } from '@/ui';
import { space } from '@/ui/tokens';

export const MIN_TOPICS = 3;

export default function Onboarding() {
  const api = useApi();
  const qc = useQueryClient();
  const topics = useTopics();
  const me = useMe();
  // A reader changing topics (from Account) starts from the ones they follow.
  const [choice, setChoice] = useState<string[] | null>(null);
  const picked = choice ?? me.data?.topics ?? [];
  const save = useMutation({
    mutationFn: (ids: string[]) => call(api!.PUT('/v1/me/topics', { body: { topics: ids } })),
    onSuccess: (me) => {
      qc.setQueryData(keys.me, me);
      qc.removeQueries({ queryKey: keys.feed }); // any earlier feed was for other topics
      // Back to the feed under this screen (the root anchor), not a second copy of it.
      router.dismissTo('/feed');
    },
  });
  const toggle = (id: string) =>
    setChoice(picked.includes(id) ? picked.filter((x) => x !== id) : [...picked, id]);
  const enough = picked.length >= MIN_TOPICS;

  return (
    <Screen title="Pick your topics" testID="onboarding">
      <View style={styles.header}>
        <Text variant="display" serif role="heading" aria-level={1}>
          FirstRead
        </Text>
        <Text variant="lead" tone="muted">
          Breaking news, ranked for you before anyone has clicked.
        </Text>
      </View>
      <View style={styles.section}>
        <Text variant="title" serif role="heading" aria-level={2}>
          Pick at least {MIN_TOPICS} topics
        </Text>
        <Text tone="muted" aria-live="polite">
          {picked.length} picked{enough ? '' : `, ${MIN_TOPICS - picked.length} more to go`}
        </Text>
        {topics.error ? (
          <ErrorState error={topics.error} onRetry={() => void topics.refetch()} />
        ) : (
          <View style={styles.chips} role="group" aria-label="Topics">
            {topics.data
              ? topics.data.topics.map((t) => (
                  <TopicChip
                    key={t.id}
                    label={t.name}
                    selected={picked.includes(t.id)}
                    onToggle={() => toggle(t.id)}
                    testID={`topic-${t.id}`}
                  />
                ))
              : Array.from({ length: 13 }, (_, i) => (
                  <Skeleton key={i} width={96} height={44} round />
                ))}
          </View>
        )}
        {save.error ? (
          <Banner tone="danger" message="Couldn’t save your topics. Please try again." />
        ) : null}
        <Button
          label="Show my feed"
          disabled={!enough || api == null}
          loading={save.isPending}
          onPress={() => save.mutate(picked)}
          testID="onboarding-continue"
        />
      </View>
      <View style={styles.section}>
        <Text variant="title" serif role="heading" aria-level={2}>
          Just looking?
        </Text>
        <Text tone="muted">See the feed through the eyes of a demo reader, with live news.</Text>
        <Link href="/personas" asChild>
          <Button variant="secondary" label="Try a demo reader" testID="onboarding-demo" />
        </Link>
      </View>
      <FooterLinks />
    </Screen>
  );
}

const styles = StyleSheet.create({
  header: { gap: space.sm, paddingTop: space.xl },
  section: { gap: space.md },
  chips: { flexDirection: 'row', flexWrap: 'wrap', gap: space.sm },
});
