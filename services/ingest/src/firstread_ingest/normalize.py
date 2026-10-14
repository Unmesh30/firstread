"""RawItem -> Candidate: clean, filter and prepare the embedding text.

Every drop has a reason, counted in the run summary: ``no_url``, ``non_english``,
``short_title``, ``junk_title``, ``stale``.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from collections.abc import Iterable
from dataclasses import replace
from datetime import datetime, timedelta

from firstread_core.text import article_text, clean_text, strip_source_suffix, title_fingerprint
from firstread_core.urls import InvalidURLError, canonicalize_url
from firstread_ingest.config import JunkTitles, load_junk_titles
from firstread_ingest.items import Candidate, RawItem

_WORDS = re.compile(r"[0-9a-z]+")
MAX_AGE = timedelta(hours=72)
MAX_FUTURE = timedelta(hours=6)


def _squash(text: str) -> str:
    return " ".join(_WORDS.findall(unicodedata.normalize("NFKC", text).casefold()))


def is_junk(title: str, junk: JunkTitles) -> bool:
    """Whole-title matches and media-only prefixes; "Live updates: ..." is kept."""
    if _squash(title) in junk.whole:
        return True
    folded = title.casefold().lstrip()
    return folded.startswith(junk.media_prefixes)


def normalize(
    item: RawItem,
    now: datetime,
    min_words: int = 3,
    junk: JunkTitles | None = None,
    max_age: timedelta = MAX_AGE,
) -> Candidate | str:
    """A Candidate, or the reason the item was dropped."""
    junk = junk or load_junk_titles()
    try:
        url_canonical = canonicalize_url(item.url)
    except InvalidURLError:
        return "no_url"
    if not item.language.lower().startswith("en"):
        return "non_english"
    title = strip_source_suffix(clean_text(item.title), item.source_name, item.source_domain)
    if len(title.split()) < min_words:
        return "short_title"
    if is_junk(title, junk):
        return "junk_title"
    published = item.published_at
    if published is not None and published > now + MAX_FUTURE:
        published = None  # a bogus future date says nothing about age
    if published is not None and now - published > max_age:
        return "stale"
    description = clean_text(item.description or "")
    if _squash(description) == _squash(title):
        description = ""  # some feeds repeat the headline as the description
    if published != item.published_at:
        item = replace(item, published_at=published)
    return Candidate(
        raw=item,
        url_canonical=url_canonical,
        title=title,
        title_fingerprint=title_fingerprint(title),
        embed_text=article_text(title, description or None),
        text_source="title_summary" if description else "title",
    )


def normalize_all(
    items: Iterable[RawItem], now: datetime, min_words: int = 3
) -> tuple[list[Candidate], Counter[str]]:
    junk = load_junk_titles()
    kept, dropped = [], Counter[str]()
    for it in items:
        out = normalize(it, now, min_words, junk)
        if isinstance(out, str):
            dropped[out] += 1
        else:
            kept.append(out)
    return kept, dropped
