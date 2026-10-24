// Checks on the exported web build (CI runs this after `npm run export:web`):
//   1. No secrets in the bundle: only EXPO_PUBLIC_* values belong there.
//   2. The size budget: the gzip size of what the first page loads (its scripts and the
//      preloaded font) and of lazy chunks may grow at most `maxGrowth` over the
//      baseline in bundle-budget.json. Prints a table either way. 4d set the baseline
//      to 469.8 KB (from 501.2 KB) and the growth to 5%, so the limit (~493 KB) stays
//      below the old size: raising it is a reviewed change to bundle-budget.json.
// Usage: node scripts/check-dist.mjs [dist] [--update-baseline]
import { readdirSync, readFileSync, statSync, writeFileSync } from 'node:fs';
import { join, relative } from 'node:path';
import { gzipSync } from 'node:zlib';

const dist = process.argv.slice(2).find((a) => !a.startsWith('--')) ?? 'dist';
const update = process.argv.includes('--update-baseline');
const BUDGET = 'bundle-budget.json';

// Never in a client bundle: Supabase secret/service keys, our server-side secrets.
const FORBIDDEN = [
  /sb_secret_[A-Za-z0-9_-]{8,}/,
  /service_role/,
  /METRICS_TOKEN/,
  /LOG_UID_SALT/,
  /HF_TOKEN/,
];

function walk(dir) {
  return readdirSync(dir).flatMap((name) => {
    const p = join(dir, name);
    return statSync(p).isDirectory() ? walk(p) : [p];
  });
}

const files = walk(dist);
const failures = [];
for (const f of files.filter((f) => /\.(js|html|json|map)$/.test(f))) {
  const text = readFileSync(f, 'utf8');
  for (const re of FORBIDDEN)
    if (re.test(text)) failures.push(`${relative(dist, f)} matches ${re}`);
}

const index = readFileSync(join(dist, 'index.html'), 'utf8');
const toFile = (href) => join(dist, href.replace(/^\/firstread\//, ''));
const initialScripts = [...index.matchAll(/<script[^>]+src="([^"]+)"/g)].map((m) => toFile(m[1]));
const fonts = [...index.matchAll(/<link[^>]+rel="preload"[^>]+href="([^"]+)"[^>]*as="font"/g)].map(
  (m) => toFile(m[1]),
);
const js = files.filter((f) => f.includes('_expo/static/js/web/') && f.endsWith('.js'));
const lazy = js.filter((f) => !initialScripts.includes(f));
const gz = (f) => gzipSync(readFileSync(f), { level: 9 }).length;

const rows = [
  ...initialScripts.map((f) => ['initial script', f]),
  ...fonts.map((f) => ['initial font', f]),
  ...lazy.map((f) => ['lazy chunk', f]),
].map(([kind, f]) => ({ kind, file: relative(dist, f), raw: statSync(f).size, gzip: gz(f) }));
const sum = (kinds) =>
  rows.filter((r) => kinds.some((k) => r.kind.startsWith(k))).reduce((n, r) => n + r.gzip, 0);
const totals = { initialGzipBytes: sum(['initial']), lazyGzipBytes: sum(['lazy']) };

console.log('kind            gzip KB   raw KB  file');
for (const r of rows) {
  console.log(
    `${r.kind.padEnd(15)} ${(r.gzip / 1024).toFixed(1).padStart(7)} ${(r.raw / 1024).toFixed(1).padStart(8)}  ${r.file}`,
  );
}
console.log(
  `initial (scripts + font): ${(totals.initialGzipBytes / 1024).toFixed(1)} KB gzip; lazy: ${(totals.lazyGzipBytes / 1024).toFixed(1)} KB gzip`,
);

if (fonts.length !== 1) failures.push(`expected exactly one preloaded font, found ${fonts.length}`);

if (update) {
  // The allowed growth is a deliberate choice: kept from the file (see apps/mobile/README.md).
  const { maxGrowth } = JSON.parse(readFileSync(BUDGET, 'utf8'));
  writeFileSync(BUDGET, JSON.stringify({ ...totals, maxGrowth }, null, 2) + '\n');
  console.log(`wrote ${BUDGET}`);
} else {
  const budget = JSON.parse(readFileSync(BUDGET, 'utf8'));
  for (const key of ['initialGzipBytes', 'lazyGzipBytes']) {
    const limit = Math.round(budget[key] * (1 + budget.maxGrowth));
    console.log(`${key}: ${totals[key]} of ${limit} allowed (baseline ${budget[key]})`);
    if (totals[key] > limit)
      failures.push(
        `${key} ${totals[key]} > ${limit}: shrink it, or raise the baseline on purpose`,
      );
  }
}

if (failures.length) {
  console.error(failures.map((f) => `FAIL: ${f}`).join('\n'));
  process.exit(1);
}
console.log('check-dist: ok');
