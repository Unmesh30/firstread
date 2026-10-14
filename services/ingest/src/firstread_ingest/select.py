"""Which GDELT candidates get stored this run (ADR 0007).

GKG delivers far more articles than the database budget allows, so each run keeps
the best ``max_per_run`` GDELT inserts, not the first ones:

1. **Breadth first:** a story carried by many outlets in this run ranks higher
   (distinct domains among this run's items in the same story cluster, counting
   duplicates and near-duplicates too). Ties go to the most recent.
2. **At most ``max_per_domain`` per outlet,** so one prolific site can't fill the run.
3. **Topic floor:** each topic first gets up to ``topic_min`` of its best items, so
   thin topics aren't starved; the remaining slots go by global rank.

RSS inserts are always kept (the curated feeds are few).
"""

from __future__ import annotations

import uuid
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

from firstread_ingest.dedupe import Decision
from firstread_ingest.items import Candidate


@dataclass(frozen=True)
class Choice:
    index: int  # into the candidates / decisions lists
    topic: str


def breadth(cands: Sequence[Candidate], decisions: Sequence[Decision]) -> dict[uuid.UUID, int]:
    """Distinct source domains per story cluster among this run's items."""
    domains: dict[uuid.UUID, set[str]] = defaultdict(set)
    for c, d in zip(cands, decisions, strict=True):
        domains[d.cluster_id].add(c.raw.source_domain)
    return {k: len(v) for k, v in domains.items()}


def select_gdelt(
    cands: Sequence[Candidate],
    decisions: Sequence[Decision],
    choices: Sequence[Choice],
    max_per_run: int,
    max_per_domain: int,
    topic_min: int,
) -> tuple[set[int], dict[str, int]]:
    """Indices of the GDELT inserts to keep, plus counts of why others were dropped."""
    width = breadth(cands, decisions)

    def rank(ch: Choice) -> tuple[int, float]:
        c = cands[ch.index]
        return (-width[decisions[ch.index].cluster_id], -c.raw.seen_at.timestamp())

    ranked = sorted(choices, key=rank)
    per_domain: Counter[str] = Counter()
    kept: set[int] = set()
    dropped: Counter[str] = Counter()

    def take(ch: Choice) -> bool:
        domain = cands[ch.index].raw.source_domain
        if ch.index in kept or len(kept) >= max_per_run or per_domain[domain] >= max_per_domain:
            return False
        kept.add(ch.index)
        per_domain[domain] += 1
        return True

    per_topic: Counter[str] = Counter()
    for ch in ranked:  # pass 1: the topic floor
        if per_topic[ch.topic] < topic_min and take(ch):
            per_topic[ch.topic] += 1
    for ch in ranked:  # pass 2: fill by global rank
        take(ch)
    for ch in ranked:
        if ch.index in kept:
            continue
        domain = cands[ch.index].raw.source_domain
        dropped["domain_cap" if per_domain[domain] >= max_per_domain else "run_cap"] += 1
    return kept, dict(dropped)
