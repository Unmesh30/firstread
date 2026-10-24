# FirstRead app (Expo)

Expo Router + TypeScript app. Runs on web now (GitHub Pages) and iOS/Android later.

```bash
npm ci
npx expo start          # dev server (press w for web); talks to the API on :8000 (`make up`)
npm run lint            # ESLint (expo config + prettier)
npm run typecheck       # tsc --noEmit (strict)
npm test                # Jest (jest-expo preset, React Native Testing Library)
npm run gen:api         # regenerate src/api/schema.d.ts from services/api/openapi.json
npm run export:web      # static web build into dist/ (served under /firstread), plus 404.html
npm run check:dist      # bundle budget (bundle-budget.json) and no secrets in dist/
npm run serve:web       # serve dist/ like GitHub Pages at http://localhost:19006/firstread/
npm run e2e             # Playwright against the local db + redis (seeds firstread_e2e first)
```

Layout:

- `src/app/`: routes only (every file is a screen; `_layout.tsx` files are navigators).
- `src/ui/`: design tokens and core components. `src/api/`: the typed client, errors and
  query policy. `src/auth/`: the dev and Supabase sign-in providers. `src/providers/`,
  `src/features/`, `src/lib/`, `src/hooks/`, `src/storage/`: everything else.
- `e2e/`: Playwright specs; `scripts/`: the Pages-like server and build checks.

Configuration is `EXPO_PUBLIC_*` (public by design; see `.env.example` at the repo root).
See `docs/learn/00-bootstrap.md` (scaffold) and `docs/learn/04-app.md` (P4) for walkthroughs.
