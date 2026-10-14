"""URL canonicalization: one string per article, whatever link variant a source gives us.

Ingestion deduplicates on ``canonicalize_url`` (a unique column in Postgres), so
the same story from GDELT and an RSS feed, or with and without tracking
parameters, collapses to one row. The rules are deliberately conservative: they
only drop parts of a URL that never select a different article.
"""

from urllib.parse import parse_qsl, quote, unquote, urlencode, urlsplit, urlunsplit

# Host prefixes that serve the same article as the bare host.
_HOST_PREFIXES = ("www.", "m.", "mobile.", "amp.")
# Query parameters that only track the click, never select content. Generic names
# that some sites use as content ids (cid, src, share, partner, via) are kept.
_TRACKING_EXACT = frozenset(
    {
        "fbclid", "gclid", "dclid", "msclkid", "yclid", "mc_cid", "mc_eid", "ocid", "cmpid",
        "smid", "ref", "ref_src", "referrer", "rss", "taid", "icid", "ito", "sr_share",
        "s_cid", "_ga", "igshid", "mbid", "outputtype", "amp", "ftag", "cmp", "spm",
        "guccounter", "guce_referrer", "guce_referrer_sig",
    }
)  # fmt: skip
_TRACKING_PREFIXES = ("utm_", "pk_", "hsa_", "__")
_DEFAULT_PORTS = {"http": 80, "https": 443}


class InvalidURLError(ValueError):
    """Not an absolute http(s) URL with a host."""


def _is_tracking(key: str) -> bool:
    k = key.lower()
    return k in _TRACKING_EXACT or k.startswith(_TRACKING_PREFIXES)


def _host(netloc_host: str) -> str:
    host = netloc_host.lower().rstrip(".")
    for prefix in _HOST_PREFIXES:
        if host.startswith(prefix) and host.count(".") >= 2:
            host = host[len(prefix) :]
            break
    try:
        return host.encode("idna").decode("ascii")
    except UnicodeError:
        return host


def _path(path: str) -> str:
    # Normalize percent-encoding (decode, then re-encode only what must be encoded),
    # but keep encoded slashes: decoding %2F would change the path's segments.
    segments = [quote(unquote(seg), safe=":@!$&'()*+,;=-._~") for seg in path.split("/")]
    p = "/".join(segments).replace("%2f", "%2F")
    if p.endswith("/amp"):
        p = p[: -len("/amp")]
    elif p.endswith("/amp/"):
        p = p[: -len("/amp/")]
    if p.startswith("/amp/"):
        p = p[len("/amp") :]
    p = p.rstrip("/")
    return p or ""


def canonicalize_url(url: str) -> str:
    """Return the canonical form of an article URL.

    https, lowercase IDNA host without ``www.``/``m.``/``mobile.``/``amp.``, no
    default port, no fragment, no tracking parameters, remaining parameters
    sorted, no trailing slash and no AMP suffix.
    """
    parts = urlsplit(url.strip())
    if parts.scheme.lower() not in ("http", "https") or not parts.hostname:
        raise InvalidURLError(f"not an absolute http(s) URL: {url!r}")
    host = _host(parts.hostname)
    port = parts.port
    netloc = host if port in (None, *_DEFAULT_PORTS.values()) else f"{host}:{port}"
    query = sorted(
        (k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True) if not _is_tracking(k)
    )
    return urlunsplit(("https", netloc, _path(parts.path), urlencode(query), ""))


def source_domain(url: str) -> str:
    """The publisher's host from a URL, canonicalized the same way (``www.`` dropped)."""
    return urlsplit(canonicalize_url(url)).hostname or ""
