// Account: who you are (guest or Google), your topics or demo reader, the sources
// you hid (with "Show again"), the theme, "Reset demo identity" on the local stack,
// and Delete my data.
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { router } from 'expo-router';
import { useState, type ReactNode } from 'react';
import { StyleSheet, View } from 'react-native';

import { call } from '@/api/client';
import { fetchers, keys, staleTimes } from '@/api/queries';
import { useFeedLocal } from '@/features/feed/store';
import { useMe, useTopics } from '@/features/me';
import { useApi, useSession } from '@/providers/session';
import {
  Banner,
  Button,
  ChoiceGroup,
  Dialog,
  ErrorState,
  Screen,
  Skeleton,
  Text,
  describeError,
  usePrefs,
  useTheme,
  useToasts,
} from '@/ui';
import type { ThemePreference } from '@/ui/theme';
import { radius, space } from '@/ui/tokens';

import { FooterLinks } from '../FooterLinks';
import { DeleteData } from './DeleteData';
import { startFresh } from './fresh';
import { SignInSection } from './SignInSection';

export const ACCOUNT_TITLE = 'Account';
export const NO_HIDDEN = 'You haven’t hidden any sources.';

const THEMES: { value: ThemePreference; label: string }[] = [
  { value: 'system', label: 'System' },
  { value: 'light', label: 'Light' },
  { value: 'dark', label: 'Dark' },
];

function Section({
  title,
  children,
  testID,
}: {
  title: string;
  children: ReactNode;
  testID?: string;
}) {
  return (
    <View style={styles.section} testID={testID}>
      <Text variant="title" serif role="heading" aria-level={2}>
        {title}
      </Text>
      {children}
    </View>
  );
}

export function AccountScreen() {
  const session = useSession();
  const me = useMe();
  return (
    <Screen title={ACCOUNT_TITLE} testID="screen-account">
      <Text variant="headline" serif role="heading" aria-level={1}>
        {ACCOUNT_TITLE}
      </Text>
      <Section title="Sign-in">
        {session.status === 'ready' ? (
          <SignInSection auth={session.auth} isAnonymous={session.session.isAnonymous} />
        ) : (
          <Skeleton height={48} />
        )}
      </Section>
      {me.error ? <ErrorState error={me.error} onRetry={() => void me.refetch()} /> : null}
      <Section title="Your feed">
        <FeedSummary />
      </Section>
      <Section title="Hidden sources" testID="hidden-sources">
        <HiddenSources />
      </Section>
      <Section title="Appearance">
        <ThemeChoice />
      </Section>
      {session.status === 'ready' && session.auth.mode === 'dev' ? (
        <Section title="Demo identity">
          <ResetIdentity />
        </Section>
      ) : null}
      <Section title="Your data">
        <DeleteData />
      </Section>
      <FooterLinks />
    </Screen>
  );
}

function FeedSummary() {
  const api = useApi();
  const me = useMe();
  const topics = useTopics();
  const persona = me.data?.active_persona ?? null;
  // Only for a demo reader: their name (the list comes from the pool snapshot).
  const personas = useQuery({
    queryKey: keys.personas,
    queryFn: () => fetchers.personas(api!),
    enabled: api != null && persona != null,
    staleTime: staleTimes.personas,
  });
  if (!me.data) return <Skeleton height={48} />;
  const names = new Map(topics.data?.topics.map((t) => [t.id, t.name]));
  const personaName = personas.data?.personas.find((p) => p.id === persona)?.name ?? persona;
  const followed = me.data.topics.map((t) => names.get(t) ?? t);
  const reads = me.data.history_count;
  return (
    <View style={styles.section}>
      {persona ? (
        <Text testID="persona-status">
          You’re reading as the demo reader <Text weight="600">{personaName}</Text>.
        </Text>
      ) : null}
      <Text testID="followed-topics">
        {followed.length ? `Topics: ${followed.join(', ')}.` : 'No topics yet.'}
      </Text>
      <Text tone="muted" variant="small">
        {reads === 1 ? '1 story' : `${reads} stories`} in your reading history. Your most recent 50
        shape your feed.
      </Text>
      <View style={styles.actions}>
        <Button
          variant="secondary"
          label="Change topics"
          onPress={() => router.push('/onboarding')}
        />
        <Button
          variant="secondary"
          label={persona ? 'Switch demo reader' : 'Try a demo reader'}
          onPress={() => router.push('/personas')}
        />
      </View>
    </View>
  );
}

function HiddenSources() {
  const api = useApi();
  const qc = useQueryClient();
  const me = useMe();
  const { colors } = useTheme();
  const unhide = useMutation({
    mutationFn: (domain: string) =>
      call(api!.DELETE('/v1/me/hidden-sources/{domain}', { params: { path: { domain } } })),
    onSuccess: (profile, domain) => {
      qc.setQueryData(keys.me, profile);
      // Stories from it that are already loaded come back now; the next page ranks it again.
      useFeedLocal.getState().unhide('source', domain);
      useToasts.getState().show({ message: `Stories from ${domain} will show again` });
    },
  });
  if (!me.data) return <Skeleton height={48} />;
  const hidden = me.data.hidden_sources;
  return (
    <View style={styles.section}>
      {unhide.error ? (
        <Banner tone="danger" message={`Couldn’t show it again. ${describeError(unhide.error)}`} />
      ) : null}
      {hidden.length === 0 ? (
        <Text tone="muted" testID="no-hidden">
          {NO_HIDDEN}
        </Text>
      ) : (
        <View role="list" style={[styles.rows, { borderColor: colors.border }]}>
          {hidden.map((domain) => (
            <View
              key={domain}
              role="listitem"
              style={[styles.row, { borderBottomColor: colors.border }]}
              testID={`hidden-${domain}`}
            >
              <Text style={styles.flex}>{domain}</Text>
              <Button
                variant="secondary"
                label="Show again"
                accessibilityLabel={`Show ${domain} again`}
                loading={unhide.isPending && unhide.variables === domain}
                disabled={unhide.isPending}
                onPress={() => unhide.mutate(domain)}
                testID={`unhide-${domain}`}
              />
            </View>
          ))}
        </View>
      )}
    </View>
  );
}

function ThemeChoice() {
  const theme = usePrefs((s) => s.theme);
  const setTheme = usePrefs((s) => s.setTheme);
  return (
    <View style={styles.section}>
      <ChoiceGroup
        label="Theme"
        options={THEMES}
        value={theme}
        onChange={setTheme}
        testID="theme"
      />
      <Text tone="muted" variant="small">
        System follows your device’s light or dark setting. Saved on this device.
      </Text>
    </View>
  );
}

function ResetIdentity() {
  const session = useSession();
  const qc = useQueryClient();
  const [confirm, setConfirm] = useState(false);
  const [busy, setBusy] = useState(false);
  if (session.status !== 'ready') return null;
  const reset = async () => {
    setBusy(true);
    try {
      // The old reader's data stays on the server (Delete my data removes it).
      await startFresh(session.auth, qc, 'You’re a new reader');
    } finally {
      setBusy(false);
      setConfirm(false);
    }
  };
  return (
    <View style={styles.section}>
      <Text tone="muted">
        This local build signs you in with a reader id kept in this browser. Reset it to try
        onboarding again as a brand-new reader.
      </Text>
      <View style={styles.actions}>
        <Button
          variant="secondary"
          label="Reset demo identity"
          onPress={() => setConfirm(true)}
          testID="reset-identity"
        />
      </View>
      <Dialog
        visible={confirm}
        title="Start as a new reader?"
        body="This browser forgets its reader id and starts over. The old reader’s data stays on the local server until it expires."
        confirmLabel="Start over"
        busy={busy}
        onConfirm={() => void reset()}
        onCancel={() => setConfirm(false)}
      />
    </View>
  );
}

const styles = StyleSheet.create({
  flex: { flex: 1 },
  section: { gap: space.md },
  actions: { flexDirection: 'row', flexWrap: 'wrap', gap: space.md },
  rows: { borderTopWidth: StyleSheet.hairlineWidth, borderRadius: radius.sm },
  row: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: space.md,
    paddingVertical: space.sm,
    borderBottomWidth: StyleSheet.hairlineWidth,
  },
});
