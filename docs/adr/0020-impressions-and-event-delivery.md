# ADR 0020: Impressions and event delivery from the app

- Status: Accepted
- Date: 2026-10-24

## Context

The feed learns from what readers do: a click re-ranks the rest of the slate (ADR 0017), and every event is stored for a future re-ranker (ADR 0016). The app has to report **impressions** (what was actually seen), **clicks**, **likes** and **hides** to `POST /v1/events`, which takes ≤ 100 events in ≤ 64 KiB, deduplicates by client `event_id`, and rejects a whole batch with 422 if any `client_ts` is more than 24 h off.

What makes this hard:

1. **"Seen" needs a definition.** Counting every row React renders would log stories nobody looked at; the feed's training data would be wrong.
2. **Browsers kill pages.** A tab closed or switched away can cut off a normal `fetch`, and the reader's last clicks and impressions are exactly the ones we want.
3. **Networks fail.** Retries must not double-count, and one bad event must not block the rest forever.
4. **Budgets.** Each accepted batch costs Upstash commands (ADR 0011: ~2 for impressions only, 3 with a click), so sending every event alone would burn the monthly allowance.
5. **Undo.** A hide can be undone for 5 s; it must not reach the server if it's undone, but it must not be lost if the page closes inside that window.

## Decision

### The viewability rule (`src/events/viewability.ts`)
- **An impression counts when ≥ 50% of the card is visible for ≥ 1 continuous second while the page or app is in the foreground**, once per (feed `request_id`, article). This is the MRC/IAB display viewability guideline.
- FlatList reports 50% crossings at once (`itemVisiblePercentThreshold: 50`, no `minimumViewTime`); a small `ViewabilityTracker` owns the clock:
  - leaving before the second is up cancels it (FlatList's own timer only re-checks at the end, so leave-and-return counted);
  - the background (Page Visibility on web, `AppState` on native) pauses it (FlatList's kept running in a hidden tab);
  - an unfocused feed (under another screen) counts nothing.
- Each impression carries `position` and `request_id`, so it joins `served_slates` on the server.

### The queue (`src/events/queue.ts`, `batch.ts`, `transport.ts`)
- **Ids at creation.** `event_id` (a v4 UUID) and `client_ts` are fixed when the event is created and persisted per reader. A retry, a reload or a duplicate keepalive sends the same id; the server answers `duplicate`.
- **Batches:** ≤ 100 events **and** ≤ 60 KB of UTF-8 JSON (under keepalive's 64 KiB quota with room for headers). Bigger queues split.
- **Flush triggers:**
  - a click, like or hide → now, carrying pending impressions (one batch, one round of Redis commands);
  - 60 s after the oldest pending impression;
  - 100 pending events;
  - **the page or app going to the background** (`visibilitychange → hidden`, `pagehide`, `AppState → background`) → one keepalive batch.
- **No flush before the next feed page.** The server already excludes served ids, so impressions don't need to arrive first.
- **Keepalive:** every send uses `fetch(…, { keepalive: true })`, so a request already on its way survives the page closing. A closing page's send starts **synchronously** with the last known auth headers: an `await` before `fetch` could be cut off by the unload.
- **Priority within a closing batch:** hides, then clicks and likes, then the newest impressions; whatever doesn't fit stays persisted for the next launch.
- **Undo window:** a hide is queued only when its toast closes. When the page hides, pending hides are committed first, so they go out in the keepalive batch.

### Retries and failures
| Server answer | What the queue does |
|---|---|
| accepted, `duplicate`, rejected `unknown_article` | removed |
| 429, 503, network error | exponential backoff with jitter, honoring `Retry-After` |
| 401 | one token refresh, then one retry |
| 422, 413 | the batch is dropped and counted (a client bug; never retried forever) |
| older than 23 h | dropped before sending (the API refuses > 24 h, and one stale event would fail the batch) |

- The persisted queue (≤ 1,000 events; the oldest impressions go first) is re-sent on the next launch: the safety net if a keepalive is lost.
- Offline: TanStack's `onlineManager` (`expo-network`) pauses sends; back online flushes.
- **Dwell isn't sent in v1:** time spent in a publisher's tab isn't reading time we can measure.
- **Middle-clicks count** (4d): a middle-click opens the publisher without a press, so the card's link records a click on `auxclick` (button 1). The same 1 s debounce as a press keeps one event per open.

### The budget, measured
The scripted E2E session (onboarding, page 1, a click with its impressions, re-ranked page 2, page 3 from the slate, a second click, `/why`, a background flush) costs **33 Redis commands** (34 in another run), against ADR 0011's measured 30–37 and the plan's estimate of 32. The E2E asserts ≤ 37 on every run.

## Alternatives considered

- **Count a render as an impression.** Cheap, but logs what scrolled past at speed, and virtualized lists render rows off screen.
- **FlatList's `minimumViewTime` alone.** Not continuous and not foreground-aware (above).
- **An IntersectionObserver tracker on web.** Equivalent here; a second code path when react-native-web's FlatList already reports crossings on both platforms.
- **`navigator.sendBeacon`.** Survives unload too, but can't send an `Authorization` header (only CORS-safelisted content types and no custom headers); keepalive `fetch` can.
- **Send each event immediately.** Simplest, but 2–3 Redis commands per event instead of per batch, and many more requests.
- **Server-generated ids.** A retry after a lost response would double-count; client ids make retries safe.
- **Retry 422s.** A permanently invalid batch would block every later event.

## Consequences

- Delivery is **at least once** and the server makes it effectively once (idempotent ids, ADR 0016).
- Impressions arrive in batches, up to a minute late unless a click flushes them; nothing on the server needs them sooner.
- A lost keepalive delays events until the next launch rather than losing them (unless the reader never returns, or the events age past 23 h).
- The rule and the budget are checked end to end: impressions follow 50%/1 s, a hide survives closing the page within its undo window, a retried batch is reported as duplicates only, and a session stays ≤ 37 Redis commands.
