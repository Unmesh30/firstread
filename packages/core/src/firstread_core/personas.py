"""Demo personas as they travel in the pool snapshot (ADR 0012, ADR 0018).

Ingestion selects each persona's articles once a day and puts a short summary of
every persona into the snapshot blob; the API serves ``GET /v1/demo/personas`` from
that copy, with no Postgres read. The articles themselves stay in Postgres
(``persona_articles``) and are copied server-side on activation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from firstread_core.topics import TOPIC_IDS

MAX_SAMPLES = 3  # sample headlines per persona in the snapshot


@dataclass(frozen=True)
class PersonaSample:
    title: str
    source_name: str
    category: str


@dataclass(frozen=True)
class PersonaSummary:
    id: str
    name: str
    description: str
    topics: tuple[str, ...]
    articles: int  # how many articles activation copies into the reader's history
    refreshed_at: int  # epoch seconds
    samples: tuple[PersonaSample, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["topics"] = list(self.topics)
        d["samples"] = [asdict(s) for s in self.samples]
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PersonaSummary:
        """Raises ``ValueError``/``KeyError``/``TypeError`` on anything malformed."""
        topics = tuple(str(t) for t in d["topics"])
        if not topics or not set(topics) <= set(TOPIC_IDS):
            raise ValueError(f"bad persona topics {topics!r}")
        samples = tuple(
            PersonaSample(str(s["title"]), str(s["source_name"]), str(s["category"]))
            for s in d["samples"][:MAX_SAMPLES]
        )
        return cls(
            str(d["id"]), str(d["name"]), str(d["description"]), topics, int(d["articles"]),
            int(d["refreshed_at"]), samples,
        )  # fmt: skip


def parse_all(raw: list[dict[str, Any]]) -> list[PersonaSummary]:
    """Every well-formed summary in a snapshot's ``personas`` list (others skipped)."""
    out = []
    for d in raw:
        try:
            out.append(PersonaSummary.from_dict(d))
        except (KeyError, TypeError, ValueError):
            continue
    return out
