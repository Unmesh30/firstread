import uuid
from datetime import UTC, datetime, timedelta

from firstread_ingest.dedupe import Decision
from firstread_ingest.items import Candidate, RawItem
from firstread_ingest.select import Choice, breadth, select_gdelt

NOW = datetime(2026, 10, 14, 18, 45, tzinfo=UTC)


def cand(domain: str, minutes_ago: int = 0) -> Candidate:
    raw = RawItem(
        origin="gdelt", source_id="gdelt-gkg", url=f"https://{domain}/{uuid.uuid4()}",
        title="t", source_name=domain, source_domain=domain, geo="us",
        seen_at=NOW - timedelta(minutes=minutes_ago),
    )  # fmt: skip
    return Candidate(raw, raw.url, "t", "fp", "t", "title")


def insert(cluster: uuid.UUID) -> Decision:
    return Decision("insert", uuid.uuid4(), cluster, None, "new_story")


def test_breadth_counts_distinct_domains_per_cluster_including_duplicates() -> None:
    a, b = uuid.uuid4(), uuid.uuid4()
    cands = [cand("x.com"), cand("y.com"), cand("y.com"), cand("z.com")]
    decs = [insert(a), Decision("duplicate", None, a, None, "url"), insert(a), insert(b)]
    assert breadth(cands, decs) == {a: 2, b: 1}


def test_broad_stories_win_then_recency() -> None:
    wide, narrow_new, narrow_old = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    cands = [cand("a.com", 30), cand("b.com"), cand("c.com"), cand("d.com", 0), cand("e.com", 60)]
    decs = [insert(wide), insert(wide), insert(wide), insert(narrow_new), insert(narrow_old)]
    choices = [Choice(i, "us") for i in range(5)]
    kept, dropped = select_gdelt(cands, decs, choices, max_per_run=4, max_per_domain=3, topic_min=0)
    assert kept == {0, 1, 2, 3}  # the 3-domain story first, then the newer single
    assert dropped == {"run_cap": 1}


def test_per_domain_cap() -> None:
    cands = [cand("spam.com", i) for i in range(5)] + [cand("ok.com")]
    decs = [insert(uuid.uuid4()) for _ in cands]
    choices = [Choice(i, "us") for i in range(6)]
    kept, dropped = select_gdelt(
        cands, decs, choices, max_per_run=10, max_per_domain=2, topic_min=0
    )
    assert sum(1 for i in kept if cands[i].raw.source_domain == "spam.com") == 2
    assert 5 in kept
    assert dropped == {"domain_cap": 3}


def test_topic_floor_keeps_thin_topics() -> None:
    wide = uuid.uuid4()
    cands = [cand(f"s{i}.com") for i in range(6)] + [cand("w.com", 90)]
    decs = [insert(wide) for _ in range(6)] + [insert(uuid.uuid4())]
    choices = [Choice(i, "sports") for i in range(6)] + [Choice(6, "weather")]
    kept, _ = select_gdelt(cands, decs, choices, max_per_run=3, max_per_domain=3, topic_min=1)
    assert 6 in kept  # weather got its floor slot despite ranking last
    assert len(kept) == 3


def test_per_story_cap_counts_already_stored_articles() -> None:
    from collections import Counter

    big, other = uuid.uuid4(), uuid.uuid4()
    cands = [cand(f"s{i}.com") for i in range(5)] + [cand("o.com", 30)]
    decs = [insert(big) for _ in range(5)] + [insert(other)]
    choices = [Choice(i, "us") for i in range(6)]
    kept, dropped = select_gdelt(cands, decs, choices, max_per_run=10, max_per_domain=3,
                                 topic_min=0, max_per_story=3,
                                 stored_per_story=Counter({big: 2}))  # fmt: skip
    assert sum(1 for i in kept if decs[i].cluster_id == big) == 1  # 2 stored + 1 new = 3
    assert 5 in kept  # the smaller story gets in
    assert dropped == {"story_cap": 4}
