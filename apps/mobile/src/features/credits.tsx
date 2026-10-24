// Credits the app owes wherever it shows the data or assets they cover.
//   GDELT: headlines come from GDELT's files; credit and link wherever they appear
//          (SPEC §4, CLAUDE.md). /v1/stats sends the same text and link.
//   RSS:   only publishers whose terms allow showing headlines with attribution
//          (services/ingest/src/firstread_ingest/config/feeds.yaml, ADR 0007).
//   Font:  Source Serif 4, SIL Open Font License 1.1 (its text ships with the app).
import { Platform, StyleSheet, View } from 'react-native';

import { env } from '@/lib/env';
import { ExternalLink, Text } from '@/ui';
import { space } from '@/ui/tokens';

export const GDELT = {
  text: 'News discovery by The GDELT Project',
  url: 'https://www.gdeltproject.org/',
};

/** The enabled feeds in feeds.yaml (a Jest test keeps this list in step with it). */
export const RSS_PUBLISHERS = ['BBC News', 'BBC Sport', 'MercoPress', 'ScienceDaily', 'TechCrunch'];

/** public/fonts/OFL-SourceSerif4.txt, served under the site's base path (app.json baseUrl). */
export const OFL_URL = Platform.select({
  web: '/firstread/fonts/OFL-SourceSerif4.txt',
  default: `${env.siteUrl}/fonts/OFL-SourceSerif4.txt`,
});
export const REPO_URL = 'https://github.com/Unmesh30/firstread';
export const MODELS_URL = 'https://huggingface.co/Unmesh30/firstread-models';
export const MODEL_CARD_URL = `${REPO_URL}/blob/main/docs/model-card.md`;

/** "News discovery by The GDELT Project", linked. */
export function GdeltCredit({
  credit = GDELT,
  prefix,
  testID = 'gdelt-credit',
}: {
  credit?: { text: string; url: string };
  prefix?: string;
  testID?: string;
}) {
  return (
    <View style={styles.row} testID={testID}>
      {prefix ? (
        <Text variant="caption" tone="muted">
          {prefix}
        </Text>
      ) : null}
      <ExternalLink href={credit.url} variant="caption" tone="muted">
        {credit.text}
      </ExternalLink>
    </View>
  );
}

const styles = StyleSheet.create({
  row: { flexDirection: 'row', flexWrap: 'wrap', alignItems: 'baseline', columnGap: space.xs },
});
