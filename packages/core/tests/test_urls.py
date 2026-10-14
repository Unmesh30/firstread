import pytest

from firstread_core.urls import InvalidURLError, canonical_host, canonicalize_url, source_domain

CASES = [
    # scheme, host case, www, default port, fragment, trailing slash
    ("http://WWW.Example.com:80/News/Story/#top", "https://example.com/News/Story"),
    ("https://example.com:443/a/", "https://example.com/a"),
    ("https://example.com:8443/a", "https://example.com:8443/a"),
    ("https://example.com/", "https://example.com"),
    # mobile and amp hosts
    ("https://m.bbc.co.uk/news/world-1", "https://bbc.co.uk/news/world-1"),
    ("https://mobile.reuters.com/x", "https://reuters.com/x"),
    (
        "https://amp.theguardian.com/world/2026/oct/14/a",
        "https://theguardian.com/world/2026/oct/14/a",
    ),
    # a two-label host keeps its prefix (m.com is a domain, not a mobile host)
    ("https://m.com/a", "https://m.com/a"),
    # AMP paths
    ("https://example.com/story/amp", "https://example.com/story"),
    ("https://example.com/story/amp/", "https://example.com/story"),
    ("https://example.com/amp/story/amp-guide", "https://example.com/story/amp-guide"),
    ("https://example.com/s?outputType=amp", "https://example.com/s"),
    # tracking params dropped, the rest sorted and kept
    ("https://example.com/a?utm_source=x&utm_medium=rss&id=7", "https://example.com/a?id=7"),
    ("https://example.com/a?b=2&a=1&fbclid=Z&gclid=Q", "https://example.com/a?a=1&b=2"),
    ("https://example.com/a?smid=tw&ocid=msn&cmpid=rss&mc_cid=1&mc_eid=2", "https://example.com/a"),
    ("https://example.com/a?cid=123", "https://example.com/a?cid=123"),  # may be a content id
    ("https://example.com/a?page=2&utm_campaign=c", "https://example.com/a?page=2"),
    # percent-encoding normalized
    ("https://example.com/caf%C3%A9", "https://example.com/caf%C3%A9"),
    ("https://example.com/a%7Eb", "https://example.com/a~b"),
    ("https://example.com/a%2Fb/c", "https://example.com/a%2Fb/c"),
    ("https://example.com/a b", "https://example.com/a%20b"),
    # BBC: RSS tracking params, and bbc.com / bbc.co.uk serve the same article ids
    (
        "https://www.bbc.co.uk/news/articles/cq80zk5k2ep4o?at_campaign=rss&at_medium=RSS",
        "https://bbc.co.uk/news/articles/cq80zk5k2ep4o",
    ),
    (
        "https://bbc.com/news/articles/cq80zk5k2ep4o",
        "https://bbc.co.uk/news/articles/cq80zk5k2ep4o",
    ),
    # IDNA host
    ("https://bücher.example/a", "https://xn--bcher-kva.example/a"),
    ("  https://example.com/a  ", "https://example.com/a"),
]


@pytest.mark.parametrize(("raw", "want"), CASES)
def test_canonicalize_url(raw: str, want: str) -> None:
    assert canonicalize_url(raw) == want


@pytest.mark.parametrize(("raw", "_"), CASES)
def test_canonicalize_url_is_idempotent(raw: str, _: str) -> None:
    once = canonicalize_url(raw)
    assert canonicalize_url(once) == once


@pytest.mark.parametrize(
    "bad", ["", "example.com/a", "ftp://example.com/a", "https:///a", "mailto:x@y"]
)
def test_invalid_urls_raise(bad: str) -> None:
    with pytest.raises(InvalidURLError):
        canonicalize_url(bad)


def test_source_domain() -> None:
    assert source_domain("https://www.nytimes.com/2026/10/14/us/x.html?smid=tw") == "nytimes.com"
    assert source_domain("http://feeds.bbci.co.uk/news/x") == "feeds.bbci.co.uk"


def test_canonical_host_matches_canonical_urls() -> None:
    assert canonical_host("www.BBC.com") == "bbc.co.uk"
    assert canonical_host("m.nytimes.com") == "nytimes.com"
    assert canonical_host("wisn.com") == "wisn.com"
