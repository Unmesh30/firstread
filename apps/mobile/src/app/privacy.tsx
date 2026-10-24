// What FirstRead stores, for how long, and how to delete it. Retention follows ADR 0016
// (the ingestion job deletes on schedule; a size guard can only shorten these).
import { Link } from 'expo-router';
import type { ReactNode } from 'react';
import { StyleSheet, View } from 'react-native';

import { FooterLinks } from '@/features/FooterLinks';
import { PageHeader } from '@/features/PageHeader';
import { Screen, Text } from '@/ui';
import { space } from '@/ui/tokens';

export const RETENTION: [what: string, kept: string][] = [
  ['Which stories you were shown (impressions)', '7 days'],
  ['Clicks, likes and hides, and the feed pages you were served', '30 days'],
  ['Your topics, hidden sources and your 50 most recent reads', 'Until you delete them'],
  ['Everything above, for a guest who hasn’t visited in 30 days', 'Deleted after 30 days'],
  ['Cached copies used to serve your feed quickly', '3 to 7 days'],
];

function Section({ title, children }: { title: string; children: ReactNode }) {
  return (
    <View style={styles.section}>
      <Text variant="title" serif role="heading" aria-level={2}>
        {title}
      </Text>
      {children}
    </View>
  );
}

function Bullets({ items }: { items: string[] }) {
  return (
    <View role="list" style={styles.list}>
      {items.map((t) => (
        <View key={t} role="listitem" style={styles.bullet}>
          <Text aria-hidden tone="accent">
            •
          </Text>
          <Text style={styles.flex}>{t}</Text>
        </View>
      ))}
    </View>
  );
}

export default function Privacy() {
  return (
    <Screen title="Privacy" testID="screen-privacy">
      <PageHeader
        title="Privacy"
        lead="FirstRead keeps only what it needs to rank your feed, for as long as it needs it."
      />
      <Section title="What we store">
        <Bullets
          items={[
            'The topics you follow, and the sources you hide.',
            'The stories you open, like or hide, and which stories were shown to you.',
            'A random reader id. As a guest, that’s all that identifies you.',
            'Your email address, only if you choose to continue with Google.',
          ]}
        />
      </Section>
      <Section title="How long we keep it">
        <View role="list" style={styles.list} testID="retention">
          {RETENTION.map(([what, kept]) => (
            <View key={what} role="listitem" style={styles.row}>
              <Text style={styles.flex}>{what}</Text>
              <Text weight="600">{kept}</Text>
            </View>
          ))}
        </View>
      </Section>
      <Section title="What we don’t do">
        <Bullets
          items={[
            'No ads, no third-party trackers or analytics, and nothing is sold or shared.',
            'No full articles: we show a headline, its source, the time and a link.',
          ]}
        />
      </Section>
      <Section title="Publishers">
        <Text>
          Thumbnails load straight from the publisher’s servers, so the publisher can see your IP
          address, as with any image on the web. Opening a story takes you to the publisher’s site,
          where their privacy policy applies.
        </Text>
      </Section>
      <Section title="On this device">
        <Text>
          Your browser keeps your sign-in, your theme choice and any events not yet sent. Clearing
          this site’s data signs you out as a guest.
        </Text>
      </Section>
      <Section title="Delete your data">
        <Text>
          Delete my data, in{' '}
          <Link href="/account">
            <Text tone="accent" weight="600">
              Account
            </Text>
          </Link>
          , removes your topics, history, likes, hides and impressions from FirstRead at once,
          including cached copies.
        </Text>
      </Section>
      <FooterLinks />
    </Screen>
  );
}

const styles = StyleSheet.create({
  flex: { flex: 1 },
  section: { gap: space.md },
  list: { gap: space.sm },
  bullet: { flexDirection: 'row', gap: space.sm },
  row: { flexDirection: 'row', flexWrap: 'wrap', columnGap: space.lg, rowGap: space.xs },
});
