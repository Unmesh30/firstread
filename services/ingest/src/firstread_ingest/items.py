"""What flows through the pipeline: raw items from a source, then normalized candidates."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

from firstread_core.topics import Geo

Origin = Literal["gdelt", "rss"]


@dataclass(frozen=True)
class RawItem:
    """One article as a source reports it, before any cleaning.

    ``description`` is used only to build the embedding text; it is never stored
    (SPEC section 4).
    """

    origin: Origin
    source_id: str  # feed id, or "gdelt:<query>"
    url: str
    title: str
    source_name: str
    source_domain: str
    geo: Geo
    seen_at: datetime  # when the source first saw it (GDELT seendate, or fetch time)
    published_at: datetime | None = None
    description: str | None = None
    image_url: str | None = None
    language: str = "en"
    source_country: str | None = None  # ISO 3166-1 alpha-2
    topic_hint: str | None = None


@dataclass
class Candidate:
    """A cleaned item that passed the filters; dedupe and scoring fill in the rest."""

    raw: RawItem
    url_canonical: str
    title: str
    title_fingerprint: str
    embed_text: str
    text_source: Literal["title", "title_summary"]
    extra: dict[str, object] = field(default_factory=dict)
