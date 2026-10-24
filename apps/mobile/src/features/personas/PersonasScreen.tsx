// "Try a demo reader": five synthetic readers built from live articles (ADR 0018).
//
// Activating asks the API with replace=false. A reader who already has their own
// history or topics gets 409 history_exists, and the app asks before replacing them
// (an in-app dialog, never window.confirm). Hidden stories and sources are kept.
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { router } from 'expo-router';
import { useState } from 'react';
import { StyleSheet, View } from 'react-native';

import { call } from '@/api/client';
import { isApiError } from '@/api/errors';
import { fetchers, keys, staleTimes } from '@/api/queries';
import type { MeResponse, PersonaOut } from '@/api/schema-types';
import { useMe, useTopics } from '@/features/me';
import { timeAgo } from '@/lib/time';
import { useApi } from '@/providers/session';
import {
  Banner,
  Button,
  Dialog,
  EmptyState,
  ErrorState,
  Screen,
  Skeleton,
  Text,
  describeError,
  useTheme,
  useToasts,
} from '@/ui';
import { radius, space } from '@/ui/tokens';

import { GdeltCredit } from '../credits';
import { PageHeader } from '../PageHeader';

export const PERSONAS_TITLE = 'Demo readers';
export const WARMING_UP = 'Demo readers are warming up';

export function replaceQuestion(name: string): string {
  return `Replace your reading history and topics with ${name}? Hidden stories and sources are kept.`;
}

function usePersonas() {
  const api = useApi();
  return useQuery({
    queryKey: keys.personas,
    queryFn: () => fetchers.personas(api!),
    enabled: api != null,
    staleTime: staleTimes.personas,
  });
}

function useActivate(onDone: (me: MeResponse, persona: PersonaOut) => void) {
  const api = useApi();
  return useMutation({
    mutationFn: ({ persona, replace }: { persona: PersonaOut; replace: boolean }) =>
      call(
        api!.POST('/v1/demo/personas/{persona_id}/activate', {
          params: { path: { persona_id: persona.id }, query: { replace } },
        }),
      ),
    onSuccess: (me, { persona }) => onDone(me, persona),
  });
}

export function PersonasScreen() {
  const qc = useQueryClient();
  const personas = usePersonas();
  const me = useMe();
  const topics = useTopics();
  const names = new Map(topics.data?.topics.map((t) => [t.id, t.name]));
  const [confirm, setConfirm] = useState<PersonaOut | null>(null);

  const activate = useActivate((profile, persona) => {
    setConfirm(null);
    qc.setQueryData(keys.me, profile);
    qc.removeQueries({ queryKey: keys.feed }); // the next feed is the persona's
    useToasts.getState().show({ message: `You’re reading as ${persona.name}` });
    router.dismissTo('/feed');
  });
  const choose = (persona: PersonaOut, replace = false) =>
    activate.mutate(
      { persona, replace },
      {
        onError: (e) => {
          if (isApiError(e, 'history_exists')) setConfirm(persona);
        },
      },
    );
  const failed = activate.error && !isApiError(activate.error, 'history_exists');

  // 503: no pool snapshot yet. Shown from the first 503, while the query retries too.
  const warming =
    !personas.data && isApiError(personas.error ?? personas.failureReason, 'unavailable');

  let body;
  if (warming) {
    body = (
      <EmptyState
        title={WARMING_UP}
        body="They’re picked from the latest stories once a day. Please try again in a minute."
        action={
          personas.error
            ? { label: 'Try again', onPress: () => void personas.refetch() }
            : undefined
        }
        testID="personas-warming"
      />
    );
  } else if (personas.error) {
    body = <ErrorState error={personas.error} onRetry={() => void personas.refetch()} />;
  } else if (!personas.data) {
    body = (
      <View
        role="progressbar"
        accessibilityLabel="Loading demo readers"
        aria-busy
        style={styles.list}
      >
        {[0, 1, 2].map((i) => (
          <Skeleton key={i} height={180} />
        ))}
      </View>
    );
  } else if (personas.data.personas.length === 0) {
    body = (
      <EmptyState
        title="No demo readers yet"
        body="They’re built from the latest stories once a day."
        action={{ label: 'Try again', onPress: () => void personas.refetch() }}
        testID="personas-empty"
      />
    );
  } else {
    body = (
      <View role="list" style={styles.list}>
        {personas.data.personas.map((p) => (
          <PersonaCard
            key={p.id}
            persona={p}
            topicName={(t) => names.get(t) ?? t}
            current={me.data?.active_persona === p.id}
            busy={activate.isPending && activate.variables?.persona.id === p.id}
            disabled={activate.isPending}
            onChoose={() => choose(p)}
          />
        ))}
      </View>
    );
  }

  return (
    <Screen title={PERSONAS_TITLE} testID="screen-personas">
      <PageHeader
        title={PERSONAS_TITLE}
        lead="See FirstRead through the eyes of a reader with a history. Each one has read 20 of today’s stories; their feed is ranked live, like yours."
      />
      {failed ? <Banner tone="danger" message={describeError(activate.error)} /> : null}
      {body}
      <GdeltCredit prefix="Sample headlines:" />
      <Dialog
        visible={confirm != null}
        title="Replace your reading history?"
        body={confirm ? replaceQuestion(confirm.name) : ''}
        confirmLabel="Replace"
        cancelLabel="Keep mine"
        busy={activate.isPending}
        onConfirm={() => confirm && choose(confirm, true)}
        onCancel={() => setConfirm(null)}
      />
    </Screen>
  );
}

function PersonaCard({
  persona,
  topicName,
  current,
  busy,
  disabled,
  onChoose,
}: {
  persona: PersonaOut;
  topicName: (id: string) => string;
  current: boolean;
  busy: boolean;
  disabled: boolean;
  onChoose: () => void;
}) {
  const { colors } = useTheme();
  const refreshed = timeAgo(persona.refreshed_at);
  return (
    <View
      role="listitem"
      testID={`persona-${persona.id}`}
      style={[
        styles.card,
        { backgroundColor: colors.surface, borderColor: current ? colors.accent : colors.border },
      ]}
    >
      <View style={styles.cardHead}>
        <Text variant="title" serif role="heading" aria-level={2} style={styles.flex}>
          {persona.name}
        </Text>
        {current ? (
          <Text variant="caption" tone="accent" weight="600">
            CURRENT
          </Text>
        ) : null}
      </View>
      <Text tone="muted">{persona.description}</Text>
      <View style={styles.topics} aria-label="Topics">
        {persona.topics.map((t) => (
          <View key={t} style={[styles.topic, { backgroundColor: colors.surfaceAlt }]}>
            <Text variant="caption" weight="600">
              {topicName(t)}
            </Text>
          </View>
        ))}
      </View>
      {persona.samples.length ? (
        <View role="list" aria-label={`Some of ${persona.name}’s reads`} style={styles.samples}>
          {persona.samples.slice(0, 3).map((s, i) => (
            <Text key={i} role="listitem" variant="small">
              <Text variant="small" serif weight="600">
                {s.title}
              </Text>
              <Text variant="small" tone="muted">
                {` · ${s.source_name}`}
              </Text>
            </Text>
          ))}
        </View>
      ) : null}
      <View style={styles.cardFoot}>
        <Text variant="caption" tone="muted" style={styles.flex}>
          {persona.articles} stories{refreshed ? ` · picked ${refreshed}` : ''}
        </Text>
        <Button
          label={current ? 'Read again' : `Read as ${persona.name}`}
          variant={current ? 'secondary' : 'primary'}
          loading={busy}
          disabled={disabled}
          onPress={onChoose}
          testID={`activate-${persona.id}`}
        />
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  flex: { flex: 1 },
  list: { gap: space.lg },
  card: { gap: space.md, padding: space.xl, borderRadius: radius.lg, borderWidth: 1 },
  cardHead: { flexDirection: 'row', alignItems: 'center', gap: space.md },
  topics: { flexDirection: 'row', flexWrap: 'wrap', gap: space.sm },
  topic: { paddingHorizontal: space.md, paddingVertical: space.xs, borderRadius: radius.pill },
  samples: { gap: space.sm },
  cardFoot: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    alignItems: 'center',
    gap: space.md,
    paddingTop: space.sm,
  },
});
