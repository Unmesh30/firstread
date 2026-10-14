import pytest

from firstread_core import TEXT_PREP_VERSION, __version__, normalize_whitespace
from firstread_core.text import (
    MAX_CHARS,
    article_text,
    clean_text,
    strip_source_suffix,
    title_fingerprint,
    truncate,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("  Breaking   news\n\ttoday ", "Breaking news today"),
        ("", ""),
        ("already clean", "already clean"),
    ],
)
def test_normalize_whitespace(raw: str, expected: str) -> None:
    assert normalize_whitespace(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Tom &amp; Jerry", "Tom & Jerry"),
        ("<p>Hello <b>world</b></p>", "Hello world"),
        ("\ufb01nance\u00a0news", "finance news"),  # NFKC: ligature and no-break space
        ("  a \n b  ", "a b"),
    ],
)
def test_clean_text(raw: str, expected: str) -> None:
    assert clean_text(raw) == expected


def test_truncate_on_word_boundary() -> None:
    assert truncate("one two three", 9) == "one two"
    assert truncate("short", 9) == "short"
    assert truncate("abcdefghijk", 5) == "abcde"


# Pinned outputs. If you change clean_text/article_text so any of these change,
# bump TEXT_PREP_VERSION (it invalidates embedding caches) and update this table.
PINNED = [
    (("Markets rally", "Stocks rose on Friday."), "Markets rally. Stocks rose on Friday."),
    (("Who won?", "The final score."), "Who won? The final score."),
    (("Title only", ""), "Title only"),
    (("Title only", None), "Title only"),
    (("", "Summary only."), "Summary only."),
    (("<i>Rates</i> &amp; jobs", "  Fed\nspeaks "), "Rates & jobs. Fed speaks"),
]


@pytest.mark.parametrize(("args", "expected"), PINNED)
def test_article_text_pinned_outputs(args: tuple[str, str | None], expected: str) -> None:
    assert article_text(*args) == expected


def test_text_prep_version_matches_pinned_table() -> None:
    # Changing the pinned table above means text prep changed: bump the version too.
    assert TEXT_PREP_VERSION == 1


def test_article_text_is_capped() -> None:
    assert len(article_text("t", "word " * 1000)) <= MAX_CHARS


def test_version_is_set() -> None:
    assert __version__


@pytest.mark.parametrize(
    ("title", "name", "domain", "want"),
    [
        ("Storm hits coast - CNN", "CNN", "cnn.com", "Storm hits coast"),
        ("Storm hits coast | Reuters", "", "reuters.com", "Storm hits coast"),
        ("Storm hits coast — The Guardian", "The Guardian", "theguardian.com", "Storm hits coast"),
        (
            "Storm hits coast - The New York Times",
            "",
            "nytimes.com",
            "Storm hits coast - The New York Times",
        ),
        ("Storm hits coast - The New York Times", "The New York Times", "", "Storm hits coast"),
        ("Biden - Harris debate recap", "CNN", "cnn.com", "Biden - Harris debate recap"),
        ("No suffix here", "BBC", "bbc.co.uk", "No suffix here"),
        ("Markets rally - Yahoo Finance", "", "finance.yahoo.com", "Markets rally"),
        ("  Spaces   - NPR ", "NPR", "npr.org", "Spaces"),
    ],
)
def test_strip_source_suffix(title: str, name: str, domain: str, want: str) -> None:
    assert strip_source_suffix(title, name, domain) == want


def test_title_fingerprint_ignores_case_punctuation_and_spacing() -> None:
    a = title_fingerprint("Diddy hospitalized after prison fight")
    assert a == title_fingerprint("  DIDDY hospitalized — after prison-fight! ")
    assert a == title_fingerprint("Diddy hospitalized after <b>prison</b> fight")
    assert a != title_fingerprint("Diddy hospitalized after prison fights")
    assert len(a) == 40
