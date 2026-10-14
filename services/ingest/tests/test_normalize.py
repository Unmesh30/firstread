from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from firstread_ingest.items import Candidate, RawItem
from firstread_ingest.normalize import normalize, normalize_all

NOW = datetime(2026, 10, 14, 20, 0, tzinfo=UTC)
BASE = RawItem(
    origin="rss", source_id="f", url="https://www.example.com/a?utm_source=rss",
    title="Earthquake hits northern Japan - Example News", source_name="Example News",
    source_domain="example.com", geo="intl", seen_at=NOW, published_at=NOW - timedelta(hours=1),
    description="<p>A strong quake struck on Wednesday.</p>",
)  # fmt: skip


def ok(item: RawItem) -> Candidate:
    out = normalize(item, NOW)
    assert isinstance(out, Candidate), out
    return out


def test_cleans_title_url_and_builds_embedding_text() -> None:
    c = ok(BASE)
    assert c.title == "Earthquake hits northern Japan"
    assert c.url_canonical == "https://example.com/a"
    assert c.embed_text == "Earthquake hits northern Japan. A strong quake struck on Wednesday."
    assert c.text_source == "title_summary"


def test_three_word_breaking_headline_is_kept() -> None:
    assert ok(replace(BASE, title="Earthquake hits Japan")).title == "Earthquake hits Japan"


@pytest.mark.parametrize(
    ("title", "reason"),
    [
        ("Breaking news", "short_title"),
        ("Live updates", "short_title"),
        ("Page not found!", "junk_title"),
        ("Just a moment...", "junk_title"),
        ("Photos", "short_title"),
        ("Watch: Storm surge floods the harbour", "junk_title"),
        ("VIDEO: Storm surge floods the harbour", "junk_title"),
        ("Podcast: The week in markets", "junk_title"),
        ("Today's headlines", "short_title"),
        ("Today's headlines for you", None),
    ],
)
def test_short_and_junk_titles(title: str, reason: str | None) -> None:
    out = normalize(replace(BASE, title=title), NOW)
    assert (out if isinstance(out, str) else None) == reason


@pytest.mark.parametrize(
    "title", ["Live updates: Earthquake hits Japan", "Opinion: The Senate got this wrong"]
)
def test_live_blogs_and_opinion_are_kept(title: str) -> None:
    assert ok(replace(BASE, title=title)).title == title


def test_other_drop_reasons() -> None:
    assert normalize(replace(BASE, url="not a url"), NOW) == "no_url"
    assert normalize(replace(BASE, language="fr"), NOW) == "non_english"
    assert normalize(replace(BASE, published_at=NOW - timedelta(days=5)), NOW) == "stale"


def test_future_dates_are_cleared_not_trusted() -> None:
    c = ok(replace(BASE, published_at=NOW + timedelta(days=30)))
    assert c.raw.published_at is None


def test_headline_repeated_as_description_counts_as_title_only() -> None:
    c = ok(replace(BASE, description="Earthquake hits northern Japan"))
    assert (c.text_source, c.embed_text) == ("title", "Earthquake hits northern Japan")
    assert ok(replace(BASE, description=None)).text_source == "title"


def test_normalize_all_counts_drops_by_reason() -> None:
    items = [
        BASE,
        replace(BASE, title="Hi"),
        replace(BASE, language="de"),
        replace(BASE, title="Hi"),
    ]
    kept, dropped = normalize_all(items, NOW)
    assert len(kept) == 1
    assert dropped == {"short_title": 2, "non_english": 1}


@pytest.mark.parametrize(
    ("title", "english"),
    [
        ("Hay que repensar cómo hablamos del cambio climático", False),
        ("Les incendies se propagent dans le sud de la France", False),
        ("Die Regierung will das Gesetz nicht ändern", False),
        ("Earthquake hits northern Japan", True),
        ("Real Madrid stun Roma after dramatic late comeback", True),
        ("Pope Leo XIV has lung cancer and underwent successful surgery", True),
        ("Cinco de Mayo parade draws crowds in the city", True),
    ],
)
def test_items_in_other_languages_are_dropped(title: str, english: bool) -> None:
    out = normalize(replace(BASE, title=title, description=None), NOW)
    assert (out != "non_english") is english
