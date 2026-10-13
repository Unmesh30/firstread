# FirstRead app (Expo)

Expo Router + TypeScript app. Runs on web now (GitHub Pages) and iOS/Android later.

```bash
npm ci
npx expo start          # dev server (press w for web)
npm run lint            # ESLint (expo config + prettier)
npm run typecheck       # tsc --noEmit (strict)
npm test                # Jest (jest-expo preset)
npx expo export -p web  # static web build into dist/ (served under /firstread)
```

Routes live in `src/app/`; everything else (components, hooks) lives outside it.
See `docs/learn/00-bootstrap.md` at the repo root for a full walkthrough.
