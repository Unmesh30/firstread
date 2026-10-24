// Renders the share-card image, public/og-image.png (1200×630, the size Open Graph and
// X's summary_large_image cards display), from the HTML below with Playwright's
// Chromium and the bundled headline font. Run it again after a brand change and commit
// the PNG: node scripts/make-og-image.mjs
// The colors are the light theme's tokens (src/ui/tokens.ts). The "cards" are abstract
// bars, not headlines: the image must never look like a real outlet's story.
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { chromium } from 'playwright-core';

const root = join(import.meta.dirname, '..');
const out = join(root, 'public', 'og-image.png');
const font = readFileSync(join(root, 'assets', 'fonts', 'SourceSerif4-SemiBold.woff2'));

const c = {
  paper: '#FBFAF7',
  surface: '#FFFFFF',
  ink: '#1B1A17',
  muted: '#5C5850',
  border: '#E2DED5',
  accent: '#A8221B',
  accentSoft: '#F8E7E5',
  skeleton: '#E9E5DC',
};

const card = (w, reason) => `
  <div class="card">
    <div class="bar small" style="width: 30%"></div>
    <div class="bar title" style="width: ${w}%"></div>
    <div class="bar title" style="width: ${w - 22}%"></div>
    <div class="reason">${reason}</div>
  </div>`;

const html = `<!doctype html><html><head><meta charset="utf-8"><style>
@font-face { font-family: Serif; src: url(data:font/woff2;base64,${font.toString('base64')}) format('woff2'); }
* { box-sizing: border-box; margin: 0; }
body { width: 1200px; height: 630px; background: ${c.paper}; color: ${c.ink};
  font-family: -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
  display: flex; padding: 72px 80px; gap: 64px; }
.left { flex: 1.15; display: flex; flex-direction: column; justify-content: center; }
.kicker { color: ${c.accent}; font-weight: 700; letter-spacing: 0.14em; font-size: 22px; }
h1 { font-family: Serif; font-size: 112px; line-height: 1; margin: 18px 0 28px; letter-spacing: -0.01em; }
p { font-size: 34px; line-height: 1.3; color: ${c.muted}; max-width: 560px; }
.rule { height: 6px; width: 96px; background: ${c.accent}; margin-top: 40px; border-radius: 3px; }
.right { flex: 1; display: flex; flex-direction: column; justify-content: center; gap: 22px; }
.card { background: ${c.surface}; border: 2px solid ${c.border}; border-radius: 16px; padding: 24px 28px;
  display: flex; flex-direction: column; gap: 12px; box-shadow: 0 8px 24px rgba(27, 26, 23, 0.06); }
.bar { height: 14px; border-radius: 7px; background: ${c.skeleton}; }
.bar.small { height: 10px; }
.bar.title { height: 20px; background: #CFC9BD; }
.reason { align-self: flex-start; font-size: 18px; font-weight: 600; color: ${c.accent};
  background: ${c.accentSoft}; padding: 6px 14px; border-radius: 999px; margin-top: 4px; }
</style></head><body>
  <div class="left">
    <div class="kicker">FOR YOU · LIVE</div>
    <h1>FirstRead</h1>
    <p>Breaking news, ranked for you before anyone has clicked.</p>
    <div class="rule"></div>
  </div>
  <div class="right">
    ${card(92, 'New in the last 10 minutes')}
    ${card(84, 'Because you read about science')}
    ${card(88, 'Trending across 14 outlets')}
  </div>
</body></html>`;

const browser = await chromium.launch();
try {
  const page = await browser.newPage({ viewport: { width: 1200, height: 630 } });
  await page.setContent(html);
  await page.evaluate(() => document.fonts.ready);
  await page.screenshot({ path: out, type: 'png' });
  console.log(`wrote ${out}`);
} finally {
  await browser.close();
}
