// How FirstRead fits together (docs/architecture.md's system overview), drawn as a
// top-to-bottom flow so it reads at phone width too. One image for screen readers,
// described in full by its label; the drawing itself is hidden from them.
import { StyleSheet, View } from 'react-native';
import Svg, { Path, Rect, Text as SvgText } from 'react-native-svg';

import { useTheme } from '@/ui';
import { fonts } from '@/ui/tokens';

export const DIAGRAM_DESCRIPTION =
  'Architecture: GDELT files and publisher RSS feeds flow into an ingest job that runs every 15 minutes on GitHub Actions. It writes articles and their embeddings to Postgres with pgvector, and a snapshot of candidate pools to Redis. The recommender API on Google Cloud Run reads both, scores stories with a pinned model from Hugging Face Hub, and serves this app on GitHub Pages.';

const W = 360;
const H = 430;

interface Box {
  x: number;
  y: number;
  w: number;
  h: number;
  title: string;
  sub: string;
  strong?: boolean;
}

const BOXES: Box[] = [
  { x: 8, y: 8, w: 164, h: 48, title: 'GDELT files', sub: 'every 15 minutes' },
  { x: 188, y: 8, w: 164, h: 48, title: 'Publisher RSS', sub: 'with permission' },
  {
    x: 8,
    y: 92,
    w: 344,
    h: 60,
    title: 'Ingest job',
    sub: 'dedupe · classify · embed · GitHub Actions',
    strong: true,
  },
  { x: 8, y: 188, w: 164, h: 48, title: 'Postgres + pgvector', sub: 'articles, readers' },
  { x: 188, y: 188, w: 164, h: 48, title: 'Redis', sub: 'pools, live state' },
  {
    x: 8,
    y: 272,
    w: 344,
    h: 60,
    title: 'Recommender API',
    sub: 'encode · retrieve · rank · Cloud Run',
    strong: true,
  },
  { x: 8, y: 368, w: 344, h: 52, title: 'This app', sub: 'Expo web on GitHub Pages' },
];

/** Downward arrows: [x, fromY, toY]. */
const ARROWS: [number, number, number][] = [
  [90, 56, 92],
  [270, 56, 92],
  [90, 152, 188],
  [270, 152, 188],
  [90, 236, 272],
  [270, 236, 272],
  [300, 332, 368],
];

export function ArchitectureDiagram() {
  const { colors } = useTheme();
  const family = fonts.body ?? undefined;
  return (
    <View
      accessible
      role="img"
      aria-label={DIAGRAM_DESCRIPTION}
      style={styles.frame}
      testID="architecture"
    >
      <Svg width="100%" height="100%" viewBox={`0 0 ${W} ${H}`} aria-hidden>
        {ARROWS.map(([x, y1, y2]) => (
          <Path
            key={`${x}-${y1}`}
            d={`M${x} ${y1 + 2}V${y2 - 3}M${x - 5} ${y2 - 9}L${x} ${y2 - 3}L${x + 5} ${y2 - 9}`}
            stroke={colors.accent}
            strokeWidth={2}
            fill="none"
          />
        ))}
        {BOXES.map((b) => (
          <Rect
            key={b.title}
            x={b.x}
            y={b.y}
            width={b.w}
            height={b.h}
            rx={8}
            fill={b.strong ? colors.accentSoft : colors.surface}
            stroke={b.strong ? colors.accent : colors.borderStrong}
            strokeWidth={1}
          />
        ))}
        {BOXES.map((b) => (
          <SvgText
            key={`${b.title}-t`}
            x={b.x + b.w / 2}
            y={b.y + b.h / 2 - 3}
            textAnchor="middle"
            fontSize={15}
            fontWeight="600"
            fontFamily={family}
            fill={colors.text}
          >
            {b.title}
          </SvgText>
        ))}
        {BOXES.map((b) => (
          <SvgText
            key={`${b.title}-s`}
            x={b.x + b.w / 2}
            y={b.y + b.h / 2 + 14}
            textAnchor="middle"
            fontSize={12}
            fontFamily={family}
            fill={colors.textMuted}
          >
            {b.sub}
          </SvgText>
        ))}
        <SvgText x={16} y={354} fontSize={11} fontFamily={family} fill={colors.textMuted}>
          model version pinned from Hugging Face Hub
        </SvgText>
      </Svg>
    </View>
  );
}

const styles = StyleSheet.create({
  frame: { width: '100%', maxWidth: 520, aspectRatio: W / H, alignSelf: 'center' },
});
