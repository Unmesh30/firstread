# FirstRead app (Expo)

Expo Router + TypeScript app. Runs on web now (static export on GitHub Pages) and iOS/Android later. How it's built and why: [docs/learn/04-app.md](../../docs/learn/04-app.md), [ADR 0019](../../docs/adr/0019-app-architecture.md) (architecture) and [ADR 0020](../../docs/adr/0020-impressions-and-event-delivery.md) (impressions and events).

```bash
npm ci
npx expo start          # dev server (press w for web); talks to the API on :8000 (`make up`)
npm run lint            # ESLint (expo config + prettier), 0 warnings
npm run typecheck       # tsc --noEmit (strict)
npm test                # Jest (jest-expo preset, React Native Testing Library)
npm run gen:api         # regenerate src/api/schema.d.ts from services/api/openapi.json
npm run export:web      # static web build into dist/ (served under /firstread), plus 404.html
npm run check:dist      # bundle budget (bundle-budget.json) and no secrets in dist/
npm run serve:web       # serve dist/ like GitHub Pages at http://localhost:19006/firstread/
npm run e2e             # Playwright against the local db + redis (seeds firstread_e2e first)
```

Layout:

- `src/app/`: routes only (every file is a screen; `_layout.tsx` files are navigators;
  `+html.tsx` is the HTML shell: theme variables, share-card tags).
- `src/ui/`: design tokens and core components. `src/api/`: the typed client, errors and
  query policy. `src/auth/`: the dev and Supabase sign-in providers. `src/events/`: the
  event queue and viewability. `src/features/`: screens' logic. `src/providers/`,
  `src/lib/`, `src/hooks/`, `src/storage/`: everything else.
- `e2e/`: Playwright specs; `scripts/`: the Pages-like server, build checks and the
  share-image renderer.

Configuration is `EXPO_PUBLIC_*` (public by design; see `.env.example` at the repo root).

## Rules for new screens

- No `window`, `localStorage` or AsyncStorage at import time (static rendering runs every
  route in Node; `npm run export:web` fails otherwise).
- Browser-only values (width, color scheme) come from `src/hooks/useHydrated.ts`, and
  colors only from `useTheme()`, so the static HTML and the first client render match.
  `e2e/console.spec.ts` fails on any console error on any exported route.
- No dynamic path segments (Pages has no fallback): use query strings, like `/why?id=`.

## The bundle budget

`check:dist` adds up the gzip size of what the first page loads (its scripts and the
preloaded font) and fails above `initialGzipBytes × (1 + maxGrowth)`. The baseline is
469.8 KB with 5% growth (since 4d). Raising it is a deliberate, reviewed change: export,
then `node scripts/check-dist.mjs --update-baseline` (it keeps `maxGrowth`), and say why
in the PR.

## Rebuilding assets

- **Web headline font** (Source Serif 4 SemiBold as WOFF2; native uses the package's TTF).
  A container conversion only, no subsetting; allowed because this Google Fonts copy
  declares no Reserved Font Name. Keep shipping `public/fonts/OFL-SourceSerif4.txt`.

  ```bash
  uvx --from 'fonttools[woff]==4.60.1' fonttools ttLib.woff2 compress \
    -o assets/fonts/SourceSerif4-SemiBold.woff2 \
    node_modules/@expo-google-fonts/source-serif-4/600SemiBold/SourceSerif4_600SemiBold.ttf
  ```

- **Share-card image** (`public/og-image.png`, 1200×630, used by the Open Graph and
  Twitter tags in `+html.tsx` with `EXPO_PUBLIC_SITE_URL`):

  ```bash
  npx playwright install chromium   # once
  node scripts/make-og-image.mjs
  ```
