// How FirstRead works, where its news comes from, and the credits it owes: GDELT
// (wherever its data appears), the RSS publishers that allow display with attribution,
// MIND (research use; none of its articles are shown) and the headline font's OFL.
import type { ReactNode } from 'react';
import { StyleSheet, View } from 'react-native';

import { GdeltCredit, MODEL_CARD_URL, OFL_URL, REPO_URL, RSS_PUBLISHERS } from '@/features/credits';
import { FooterLinks } from '@/features/FooterLinks';
import { PageHeader } from '@/features/PageHeader';
import { ExternalLink, Screen, Text } from '@/ui';
import { space } from '@/ui/tokens';

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

export default function About() {
  return (
    <Screen title="About FirstRead" testID="screen-about">
      <PageHeader
        title="About FirstRead"
        lead="A news feed that ranks breaking stories for you within minutes of publication, before anyone has clicked on them."
      />
      <Section title="How it works">
        <Text>
          Most recommenders need clicks before they know who a story is for. FirstRead reads each
          new headline the moment it arrives and turns it into a point in a “meaning space”, so a
          story about chip export rules lands near other stories about trade and technology.
        </Text>
        <Text>
          Your reader profile lives in the same space: it starts from the topics you pick and moves
          with every story you open. Your feed is the stories nearest to you, mixed with what’s
          fresh and what many outlets are covering, and spread across different stories. Every
          click, like or hide reshapes the rest of your feed.
        </Text>
      </Section>
      <Section title="Where the news comes from">
        <GdeltCredit testID="about-gdelt" />
        <Text>
          The GDELT Project monitors news sites worldwide and publishes the links it finds every 15
          minutes. FirstRead also reads the RSS feeds of publishers whose terms allow showing their
          headlines with attribution: {RSS_PUBLISHERS.join(', ')}.
        </Text>
        <Text>
          Every story shows its publisher and links to the original. FirstRead shows only the
          headline, source, time, link and the publisher’s own thumbnail, never the article.
        </Text>
      </Section>
      <Section title="The model">
        <Text>
          The recommender was trained and evaluated offline on MIND, the Microsoft News Dataset,
          which is licensed for research. No MIND articles are shown in this app, and the model is
          used for this non-commercial demo only.
        </Text>
        <ExternalLink href={MODEL_CARD_URL}>Read the model card</ExternalLink>
      </Section>
      <Section title="Credits">
        <Text testID="font-credit">
          Headlines are set in Source Serif 4, © 2014 The Source Serif 4 Project Authors, used under
          the SIL Open Font License 1.1.
        </Text>
        <ExternalLink href={OFL_URL} label="The font’s license (OFL 1.1)">
          The font’s license (OFL 1.1)
        </ExternalLink>
        <Text>FirstRead’s code is open source under the MIT license.</Text>
        <ExternalLink href={REPO_URL}>Source code on GitHub</ExternalLink>
      </Section>
      <FooterLinks />
    </Screen>
  );
}

const styles = StyleSheet.create({
  section: { gap: space.md, alignItems: 'flex-start' },
});
