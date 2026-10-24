# ADR 0019: App architecture (Expo web first, static routes, server state vs client state)

- Status: Accepted
- Date: 2026-10-24

## Context

P4 builds the reader app SPEC §2 describes: onboarding, a live "For You" feed that adapts in real time, like/hide, "Why this?", demo readers, Account, System, Privacy and About. It ships to the web first (GitHub Pages, P5) and must be able to become an iOS/Android app later without a rewrite. The constraints:

1. **$0 hosting.** GitHub Pages serves static files only: no server, no SPA fallback, and a 404 page for any unknown path.
2. **Every feed refetch costs money.** A new first page is 7 Upstash commands (ADR 0011), so the client must never refetch the feed by itself.
3. **Auth must work locally without Supabase** (the dev bypass, ADR 0015) and in production with Supabase anonymous sign-in plus Google linking, behind one interface.
4. **Recruiters judge it in a browser:** it must load fast, look finished, and pass an accessibility audit in light and dark.
5. **No secrets in the bundle.** Anything an Expo app reads at build time (`EXPO_PUBLIC_*`) is public.

## Decision

### Platform and routing
- **Expo SDK 57 + React Native + react-native-web + Expo Router**, one codebase for web and native. Routes are files in `src/app/`.
- **Static rendering** (`web.output: "static"`, `experiments.baseUrl: "/firstread"`). Every route is pre-rendered to HTML in Node at build time, so the first paint needs no JavaScript and every deep link has a real file.
- **No dynamic path segments.** Static hosting has no fallback for `/why/<id>`, so "Why" is `/why?id=<uuid>` and GitHub Pages serves `why.html`.
- `scripts/postexport.mjs` copies `+not-found.html` to `404.html`; `scripts/serve-pages.mjs` (no dependencies) mimics Pages locally and in E2E: the prefix, extensionless → `.html`, a real 404 status.
- **The root stack anchors `(tabs)`**, so a deep link to the Why sheet has the feed behind it. The cost: any deep link mounts the feed underneath. The feed's query therefore runs only while the feed is on screen (focused, or behind the Why sheet), never for a deep link to `/about`, `/privacy` or `/personas` (4d).
- **Never touch `window`, `localStorage` or AsyncStorage at import time.** `expo export` runs every route in Node, so CI's export step is the test for it.

### Hydration-safe rendering
- Values that exist only in a browser (window width, system color scheme, the Modal portal) come from `useSyncExternalStore` with a **server snapshot** (`src/hooks/useHydrated.ts`). React uses the snapshot while hydrating, so the first client render matches the HTML (no error #418), then re-renders with the real value. Native apps don't hydrate and get the real value at once.
- **Width-dependent views** (top vs bottom nav, the feed's side panel) are all in the static HTML, tagged `data-shown`; a media query in `+html.tsx` shows the right one before JavaScript.
- **Theme before JavaScript** (4d): while hydrating, the theme hands out CSS variables (`var(--fr-surface)` …) instead of colors. `+html.tsx` defines them per `prefers-color-scheme`, and an inline head script applies a stored Light/Dark choice (`data-theme`) before the first paint. After hydration the provider sets `data-theme` and switches to real colors.

### State
- **Server state in TanStack Query** (`src/api/queries.ts`): caching, retries and cursor pages (`useInfiniteQuery` over `next_cursor`).
  - The feed is `staleTime: Infinity` with refetch on mount, focus and reconnect all **off**; only an explicit refresh resets it to one new first page.
  - Topics 24 h; `/v1/me` 30 s; stats and personas 60 s (the API's `Cache-Control`).
  - Retries only for 429, 503 and network errors, at most 3, honoring `Retry-After` up to 30 s.
- **Client state in zustand**: the event queue, toasts, the theme preference, and what this session did that loaded pages don't know yet (likes, hidden stories and sources).
- **No persisted query cache.** Only the event queue, the dev reader id and the theme preference are stored (`src/storage/kv.ts`: localStorage on web, AsyncStorage on native, all access guarded).

### API client
- **`openapi-fetch` over types generated from the committed `openapi.json`** (`npm run gen:api`; CI fails when they're stale). One `ApiError` type carries status, code, request id and `Retry-After`.
- Every call goes through `authFetch`: auth headers, and after a 401 one refresh and one retry.

### Auth abstraction
- One `AuthProvider` interface (`ready`, `userId`, `isAnonymous`, `getAuthHeaders()`, `refresh()`, `linkGoogle()`, `reset()`), chosen by `EXPO_PUBLIC_AUTH_MODE`:
  - **`dev`:** a UUID per device sent as `X-Dev-User`. The app refuses this mode unless the API URL is localhost.
  - **`supabase`:** anonymous sign-in, then `linkIdentity` for Google (the same user, so history is kept); `identity_already_exists` and the other error codes are handled. supabase-js is loaded with a dynamic `import()`, so it's a separate 65.5 KB chunk that dev builds never fetch.
- Only five public variables reach the bundle; `check-dist.mjs` fails the build if `dist/` contains secret patterns.

### Lists, images and fonts
- **FlatList, not FlashList.** A session holds a few hundred variable-height rows; FlatList virtualizes on react-native-web with no dependency. `windowSize` 7, `removeClippedSubviews` on native. FlashList stays the option if a profile shows dropped frames (P6).
- **expo-image** thumbnails at a fixed 88×88, lazy on web, decorative (the story link names the story), and a source-initial tile when the image fails.
- **One headline face** (Source Serif 4 SemiBold, OFL) that never blocks first render: a system serif fallback, `font-display: swap`, preloaded. Since 4d the web gets a **WOFF2** of the same file (≈ 27 KB smaller), allowed because the Google Fonts copy declares no Reserved Font Name; native keeps the TTF (`src/ui/headlineFont.ts` / `.web.ts`).
- **Platform splits** keep native-only libraries out of the web bundle: `expo-web-browser` and `expo-crypto` live behind `src/lib/platformKit.ts` (the web file uses `crypto.randomUUID()` and links that open their own tabs).

### Design system and accessibility
- Tokens are the only place colors, spacing, radii and type sizes are defined; a Jest test checks every text/background pair against WCAG 2.2 in both themes (≥ 4.5:1 text, ≥ 3:1 UI and focus).
- Sheets and dialogs are react-native-web's `Modal` (focus moved in, trapped and returned; Esc closes, including during the fade-in since 4d).
- Space presses checkboxes, radios and switches on web (react-native-web maps Space only for buttons).
- Bundle budget in CI: the first page's scripts plus the preloaded font. 4d set the baseline to 469.8 KB gzip with 5% growth allowed, so the limit (493 KB) sits below the 4c size (501.2 KB).

## Alternatives considered

- **A plain React (Vite) web app.** Smaller bundle (no react-native-web, ~75 KB gzip), but the native apps in SPEC §17 would be a second codebase.
- **SPA output** (`web.output: "single"`). One `index.html`; deep links on Pages would need the 404-redirect hack, and the first paint would wait for the whole bundle.
- **Dynamic routes** (`/why/[id]`). Not servable from Pages without a fallback.
- **Redux Toolkit / RTK Query.** Works, but TanStack Query's infinite queries, `onlineManager` and retry policy fit with less code; zustand covers the small client state.
- **A persisted query cache.** Would show yesterday's feed on launch and tempt a background refetch (7 commands); not worth it for a news feed.
- **FlashList.** Faster for very long lists; not needed at this size, and it's another native dependency.
- **An icon font or icon library.** About ten hand-drawn SVG paths are smaller.
- **Dropping the unused bottom-tabs code (6.5 KB gzip).** Expo Router's CommonJS index requires `layouts/Tabs` eagerly and its error boundary imports the bottom-tabs barrel; Metro can't tree-shake CommonJS. Only a resolver override of private build paths could remove it, which would break on upgrades. Not done; recorded in learn chapter 04.
- **Subsetting the font.** Smaller still, but a modified font under the OFL is the owner's call each time and the gain past WOFF2 is small for one weight.

## Consequences

- The app works offline-first for what's loaded, loads a skeleton with no JavaScript, and every route is a deep link.
- First load is 469.8 KB gzip (JS 391.5 + font 78.3); react-native-web, react-dom and Expo Router are ~60% of the JS.
- The same components run on iOS/Android; native-only behavior (in-app browser, AppState, AsyncStorage) is behind platform files.
- Static rendering adds a rule every new screen must follow (no browser APIs at import time, hydration-safe values); CI's export step and the console-error E2E guard enforce it.
