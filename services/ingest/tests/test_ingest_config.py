from collections import Counter
from typing import Any

import pytest

from firstread_core.topics import TOPIC_IDS
from firstread_ingest.config import Settings, load_feeds, load_junk_titles


def test_curated_feeds_load_with_unique_ids_and_known_hints() -> None:
    feeds = load_feeds(enabled_only=False)
    assert len(feeds) >= 50
    assert len({f.id for f in feeds}) == len(feeds)
    assert all(f.url.startswith("https://") for f in feeds)


def test_every_topic_has_a_feed_and_both_geos_exist() -> None:
    feeds = load_feeds(enabled_only=False)
    hints = Counter(f.topic_hint for f in feeds)
    assert set(TOPIC_IDS) - {"politics"} <= set(hints)
    assert {f.geo for f in feeds} == {"us", "intl"}
    guardian_us = next(f for f in feeds if f.id == "guardian-us")
    assert guardian_us.geo == "us"


def test_junk_list_keeps_live_updates_and_opinion_prefixes() -> None:
    junk = load_junk_titles()
    assert "live updates" in junk.whole
    assert not any(p.startswith(("live", "opinion")) for p in junk.media_prefixes)


def test_default_settings_pin_the_topic_release() -> None:
    s = Settings(_env_file=None)  # type: ignore[call-arg]
    assert s.ingest_model_tag == "v0.2.0"
    assert s.ingest_min_title_words == 3


def test_empty_model_dir_means_download_from_the_hub(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INGEST_MODEL_DIR", "")
    assert Settings(_env_file=None).ingest_model_dir is None  # type: ignore[call-arg]


def test_feeds_whose_terms_forbid_website_use_are_disabled() -> None:
    for f in load_feeds(enabled_only=False):
        if f.terms == "forbidden":
            assert not f.enabled, f.id
        if f.enabled:
            assert f.terms == "website_with_attribution"
            assert f.terms_url
        assert f.terms_checked, f.id


def test_a_feed_without_explicit_permission_cannot_be_enabled() -> None:
    from pydantic import ValidationError

    from firstread_ingest.config import Feed

    base: dict[str, Any] = {
        "id": "x",
        "url": "https://x.example/f",
        "source_name": "X",
        "domain": "x.example",
        "region": "US",
        "geo": "us",
    }
    with pytest.raises(ValidationError, match="explicit website permission"):
        Feed(**base, enabled=True, terms="personal_noncommercial")
    assert not Feed(**base, enabled=False, terms="personal_noncommercial").enabled
    assert Feed(**base, enabled=True, terms="website_with_attribution").enabled


def test_text_only_permission_drops_images() -> None:
    ycc = next(f for f in load_feeds() if f.id == "yale-climate-connections")
    assert not ycc.allow_images
