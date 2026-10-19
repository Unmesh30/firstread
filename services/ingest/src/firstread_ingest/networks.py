"""Find syndication networks: domains that carry the same stories (ADR 0013).

Pure: given (story, domain) coverage pairs, link domains whose story sets have a
Jaccard similarity of at least ``jaccard`` (among domains with ≥ ``min_stories``),
and return connected components of at least ``min_size`` domains. Components can
chain, so the output is a draft for human review, never applied automatically.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations


def networks(
    coverage: list[tuple[str, str]], min_stories: int, jaccard: float, min_size: int
) -> tuple[list[list[str]], dict[str, set[str]]]:
    by_domain: dict[str, set[str]] = defaultdict(set)
    for story, domain in coverage:
        by_domain[domain].add(story)
    keep = {d for d, s in by_domain.items() if len(s) >= min_stories}
    by_story: dict[str, list[str]] = defaultdict(list)
    for story, domain in coverage:
        if domain in keep:
            by_story[story].append(domain)
    shared: Counter[tuple[str, str]] = Counter()
    for domains in by_story.values():
        for a, b in combinations(sorted(set(domains)), 2):
            shared[(a, b)] += 1
    parent = {d: d for d in keep}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for (a, b), n in shared.items():
        union = len(by_domain[a]) + len(by_domain[b]) - n
        if n / union >= jaccard:
            parent[find(a)] = find(b)
    comps: dict[str, list[str]] = defaultdict(list)
    for d in keep:
        comps[find(d)].append(d)
    found = sorted((sorted(c) for c in comps.values() if len(c) >= min_size), key=len, reverse=True)
    return found, by_domain
