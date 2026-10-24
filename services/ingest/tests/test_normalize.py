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


@pytest.mark.parametrize(
    ("title", "rule"),
    [
        ("Dexterra Announces Date of Q3 2026 Results and Conference Call", "results_period_year"),
        ("Dexterra Group Inc.: Dexterra Announces Date of Q3 2026 Results", "results_period_year"),
        ("ROK Resources, Inc.: ROK Resources Announces Return of Capital", "company_prefix"),
        (
            "Hathway Cable and Datacom Limited Reports Q2 & H1 FY27 Financial Results",
            "results_period_year",
        ),
        ("Acme Corp Reports Third Quarter 2026 Financial Results", "results_period_year"),
        ("Acme to Host Third Quarter 2026 Earnings Conference Call", "call_announced"),
        ("Acme to Host Earnings Conference Call on November 5", "call_announced"),
        ("Acme earnings conference call and webcast details", "call_and_webcast"),
        ("Acme Declares Quarterly Cash Dividend of $0.25 per Share", "dividend_declared"),
        ("Acme announces the date of its fiscal results release", "results_date"),
        ("Acme files Form 8-K with the SEC", "filing"),
    ],
)
def test_corporate_announcements_are_dropped(title: str, rule: str) -> None:
    assert normalize(replace(BASE, title=title, description=None), NOW) == f"press_release:{rule}"


@pytest.mark.parametrize(
    "title",
    [
        "Meta shares jump after earnings call",
        "Nvidia to report results as AI boom faces test",
        "Netflix reports third-quarter results as subscriber growth slows",
        "Netflix is laying off 6% of its workforce",
        "Stocks fall after earnings miss at Nvidia",
        "Fed signals rate hike",
        "Bank of England keeps rates on hold: what it means for mortgages",
        "Election results 2026: who won in Ohio",
    ],
)
def test_news_about_results_and_calls_is_kept(title: str) -> None:
    out = normalize(replace(BASE, title=title, description=None), NOW)
    assert isinstance(out, Candidate), out


@pytest.mark.parametrize(
    ("title", "source", "domain", "section_page"),
    [
        ("Health - Carroll Broadcasting Inc.", "kcim.com", "kcim.com", True),
        ("Business - Carroll Broadcasting Inc.", "kcim.com", "kcim.com", True),
        ("Sports | The Daily Herald", "dailyherald.com", "dailyherald.com", True),
        ("Local News - WISN", "wisn.com", "wisn.com", True),
        ("Health - New study links sleep to memory", "kcim.com", "kcim.com", False),
        ("Business - Netflix cuts 1,000 jobs", "x.com", "x.com", False),
        ("Politics: Senate passes the farm bill", "x.com", "x.com", False),
    ],
)
def test_section_pages_are_dropped(
    title: str, source: str, domain: str, section_page: bool
) -> None:
    out = normalize(
        replace(BASE, title=title, description=None, source_name=source, source_domain=domain),
        NOW,
    )
    assert (out == "section_page") is section_page, out


@pytest.mark.parametrize(
    ("title", "rule"),
    [
        # AP's rolling digests (titles from the 3-day replay)
        ("AP News Summary at 2:50 p.m. EDT", "ap_digest"),
        ("AP News Summary at 12:44 a.m. EDT", "ap_digest"),
        ("AP News in Brief at 9:04 p.m. EDT", "ap_digest"),
        ("AP News in Brief at 12:04 a.m. EDT \u2013 Winnipeg Free Press", "ap_digest"),
        ("AP News Digest 6 a.m.", "ap_digest"),
        ("AP News Digest 12 p.m. - Daily Sitka Sentinel", "ap_digest"),
        ("ap news summary at 7:08 pm est", "ap_digest"),
        # weather forecasts named only by a date
        ("Weather forecast for October 14th", "dated_weather_forecast"),
        ("Weather Forecast For Wednesday October 14", "dated_weather_forecast"),
        ("Weather forecast, October 14", "dated_weather_forecast"),
        ("Weather Forecast for Monday, October 12, 2026", "dated_weather_forecast"),
        ("Weather & Atmospheric Forecast October 12th", "dated_weather_forecast"),
        ("Weather forecast for Oct. 14", "dated_weather_forecast"),
    ],
)
def test_templated_non_articles_are_dropped(title: str, rule: str) -> None:
    out = normalize(replace(BASE, title=title, description=None), NOW)
    assert out == f"junk_title:{rule}"


@pytest.mark.parametrize(
    "title",
    [
        # near-misses of ap_digest: a headline, another outlet's digest, no time
        "AP News: Senate passes the budget after a late vote",
        "AP News Summary: Senate passes the budget",
        "AP news summary shows strikes spreading at 3 plants",
        "Iran News in Brief \u2013 October 12, 2026",
        "World News in Brief: Gaza braces for winter amid attacks",
        "Pacific News In Brief For 13 October",
        "The AP news summary at 2:50 p.m. EDT is late today",
        # near-misses of dated_weather_forecast: a place, a storm, a warning, content
        "Omaha morning weather forecast for Tuesday, October 13",
        "Orangeville forecast for Wednesday, October 14th",
        "FOX29 First Alert Weather Forecast: Oct. 12, 2026",
        "Weather forecast: High pressure building this week",
        "Weather forecast for October 14: Hurricane Milton nears Florida",
        "Weather forecast, October 14 - Storm warning for the coast",
        "Bulgaria issues weather warnings over heavy rain forecast for October 13",
        "Weather forecast for Dublin on October 14",
        "IMF raises its forecast for world growth this year",
    ],
)
def test_near_misses_of_templated_non_articles_are_kept(title: str) -> None:
    out = normalize(replace(BASE, title=title, description=None), NOW)
    assert isinstance(out, Candidate), out


def test_normalize_all_counts_templated_non_articles_by_pattern() -> None:
    items = [
        BASE,
        replace(BASE, title="AP News Summary at 2:50 p.m. EDT"),
        replace(BASE, title="AP News in Brief at 9:04 p.m. EDT"),
        replace(BASE, title="Weather forecast for October 14th"),
    ]
    kept, dropped = normalize_all(items, NOW)
    assert [c.title for c in kept] == ["Earthquake hits northern Japan"]
    assert dropped == {"junk_title:ap_digest": 2, "junk_title:dated_weather_forecast": 1}
