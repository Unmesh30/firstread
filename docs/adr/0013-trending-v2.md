# ADR 0013: Trending v2: merged split stories, syndication networks, representatives by quality

- Status: Accepted
- Date: 2026-10-18
- Amends: [ADR 0010](0010-redis-pools-and-trending.md) (trending formula, representative) and [ADR 0009](0009-dedupe-and-story-clustering.md) (adds the feed-time grouping threshold).

## Context

The Phase 2 replay left three trending problems for P3 (learn chapter 02, §8):

1. **Split stories.** With GDELT↔GDELT τ_story raised to 0.925 (ADR 0009, 2b), big evolving stories split into several clusters. The Pope's surgery appeared as three trending entries on 2026-10-13.
2. **Syndication inflates breadth.** One wire story republished across dozens of UK local sites (one owner) counts as dozens of independent outlets, so a regional network decides what "trending" means.
3. **The representative was simply the earliest stored article,** often a headline with a publisher suffix and no image.

Phase 3's feed has the same problem as split stories: SPEC §7.8 allows at most 2 items per story, so it needs a way to treat near-identical stories as one. Grouping must not chain, though (A~B and B~C merging A and C), for the reason ADR 0009 rejected union-find.

## Decision

### A feed-time grouping threshold, τ_feed (`firstread_core/stories.py`)

- **Rule, fixed before computing it:** on the pooled 2a + 2b labelled pairs (`docs/learn/data/02-thresholds{,-2b}.json`), take the lowest bin edge from which every bin with ≥ 3 pairs is ≥ 90% same story or duplicate.
- **Why 90%, not ingestion's 95%:** a feed-time false merge only hides one item from one slate, while an ingestion false merge corrupts stored data.
- **Result** (a test recomputes it from the committed files):

  | Pair type | τ_feed | τ_story (ingestion) |
  |---|---:|---:|
  | GDELT↔GDELT (title↔title) | 0.875 | 0.925 |
  | GDELT↔RSS | 0.75 | 0.80 |
  | RSS↔RSS | 0.875 | 0.875 |

- **Caveats:**
  - The labelled pairs were *nearest neighbours*, so their precision doesn't carry over to arbitrary pairs in a candidate set.
  - 2b found different-story templated headlines at 0.85–0.92, inside the G↔G value.
- **Pre-registered fallback:** if any reviewed merge (this ADR's replay sample, and the P3e persona report) is two different stories, that pair type's τ_feed rises to its τ_story.
- **"Near-identical to something already read"** (P3d) uses the stricter, validated τ_story, not τ_feed.

### Grouping without chaining (`stories.group`)

- Items are visited in score order, so the first member of a group is its leader. An item joins:
  1. the group that already holds its `story_cluster_id`; else
  2. the first group in which its cosine to **every** member is ≥ τ_feed for that pair type (complete linkage); else
  3. a new group.
- Groups are therefore bounded in diameter. Used for trending (here) and for the feed's 2-per-story cap (P3d).

### Trending v2 (`firstread_ingest/pools.trending_entries`)

1. **Candidates:** one SQL aggregate over `story_coverage` (24 h, every outlet counts, dropped duplicates included). It returns the top 400 clusters by `Σ_outlets 2^(−age/6 h)` with ≥ 2 outlets, where an *outlet* is a syndication network when the domain is in an applied network, otherwise the domain.
2. **Representative,** chosen locally from the window snapshot (ADR 0012), best first:
   1. RSS origin (a permitted feed, usually with an image and description);
   2. has an image;
   3. not a syndicated copy (domain not in a network);
   4. a confident model topic (`category_source = model`, confidence ≥ 0.50);
   5. earliest first seen;
   6. id.

   A story with no stored article in the 72 h feed window is dropped: there is nothing to show.
3. **Merge:** representatives are grouped with complete linkage at τ_feed. A merged group's score is recomputed exactly by a second aggregate: each outlet counts once, at its earliest sighting across the merged clusters (≤ 200 rows).
4. **"Covered by N outlets"** = the distinct outlets of the merged story, stored per entry in the pool snapshot. The API never computes it.

### Syndication networks (`config/outlet_networks.yaml`)

- **Found by** `scripts/outlet_networks.py` (pure logic in `firstread_ingest/networks.py`) on the replay's last 48 h of coverage (56,030 rows, 9,224 stories):
  - domains with ≥ 10 stories;
  - pairs linked at Jaccard ≥ 0.5 of their story sets;
  - connected components of ≥ 3 domains.
- **Result: 47 networks, 869 domains.** The largest are 142 UK local titles, 106 US public-radio stations, 68 Australian regional papers, 64 and 41 US radio station groups, and 58 US local TV stations.
- **Components can chain,** so the list is a draft for the owner's review, not applied blindly.
- **Owner review, 2026-10-22:** the 38 sampled merges were approved (all the same story) and the 47 networks were approved, with one change:
  - `netscape.com` was removed from network-29 and added to the aggregator blocklist (`gdelt_domains.yaml`, the same rule as `aol.com`, ADR 0007);
  - why: five of its stored URLs (`isp.netscape.com/.../story/0001/<date>/<hash>`) were all word-for-word AP wire copy, three of them the same stories the Graham Media stations' AP pages carried;
  - `screencrush.com` stays in network-27 (Townsquare Media, like the rest of that network).
- **Materiality rule (pre-registered):** apply networks if counting each as one outlet changes ≥ 3 of the daily top 10 on ≥ 2 of 3 replay days, **or** networks supply > 30% of the summed top-10 score.

## Evidence (3-day replay, 2026-10-12 02:30 → 10-15 02:15; aggregates in `docs/learn/data/03-replay.json`)

**Networks are material, by a wide margin** (top 10 at 23:45 each day, merged variant vs merged + networks):

| Day | Top-10 entries changed | Network share of the top-10 score |
|---|---:|---:|
| 10-12 | 6 | 80% |
| 10-13 | 4 | 70% |
| 10-14 | 9 | 91% |
| 10-15 (partial, to 02:15) | 10 | 97% |

- Without networks, 10-14's top 10 is entirely UK local syndication: a Commonwealth event, a UK university strike, Omagh, a lost fishing boat.
- With networks, stories covered nationally by independent outlets lead: the US Social Security cost-of-living rise, Netflix's layoffs, the Ilhan Omar probe.
- **`apply: true`.**

**Merging:**
- **Every sampled merge is the same story** (38 sampled, up to 10 a day, reviewed with headlines in the PR), for example:
  - two RJ Mitte obituaries;
  - three write-ups of Selma Blair's arrest;
  - three of the Tupac sentencing;
  - two of Kroger's AI layoffs;
  - two Pope cancer explainers.
- **No different-story merge was found,** so the τ_feed fallback is not triggered.
- **What merging fixes:** on 10-13, Phase 2's two separate cancer-diagnosis entries become one entry of 261 outlets. That day's top 10 still holds three Pope entries (the diagnosis, "surgery completed", "surgery in coming days"), now one per development rather than two copies of the same news.

**Limitations seen:**
- **Complete linkage under-merges on purpose.** Selma Blair still appears as two entries on 10-12. The Pope's "surgery in coming days", "surgery completed" and "diagnosis" stay separate entries; they're arguably separate developments.
- **The representative order** can still pick a title with a publisher suffix ("… | Wilts and Gloucester") when it's the one with an image. That's cosmetic; P4 displays it as-is.

## Alternatives considered

- **Union-find (connected components) over candidate pairs.** Rejected for chaining, as in ADR 0009.
- **Re-merging clusters in the database** (updating `story_cluster_id`). It would break the "rows never change after insert" property the window snapshot's delta relies on (ADR 0012), and would make ingestion's clustering depend on later arrivals.
- **A curated ownership list** (Newsquest, Sinclair, NPR member stations, …). Hand-maintained and incomplete. Co-coverage finds the networks that actually behave as one outlet in our data; the owner reviews the list instead of writing it.
- **Down-weighting networks** (e.g. counting a network as log2(size) outlets). More knobs and no data to tune them. Counting a network once is the simplest rule that removes the inflation.
- **A `story_coverage.outlet` column** (the plan's migration 0007). Passing the network map as query parameters keeps the table unchanged, so a network edit takes effect on the next run without a backfill.

## Consequences

- **Trending reflects independent breadth.** The 10-14 example shows what changes for readers.
- **Re-run the network finder after sources change** (e.g. after P5's scheduled runs) and review the diff. A network that splits or grows only changes trending scores, never stored data.
- **Costs** (included in the 84 KB-per-run egress, [ADR 0011](0011-free-tier-budgets.md)): two aggregates per run (≤ 400 and ≤ 200 rows). The representative and the merging use the runner-side window.
- The P3d feed reuses `stories.group` for its 2-per-story cap and `stories.near_any` (τ_story) for already-read stories. P3e reports per-persona group sizes and cap drops, with examples, under the same fallback rule.
