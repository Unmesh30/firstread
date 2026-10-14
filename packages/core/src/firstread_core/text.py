"""Text helpers shared by training, ingestion and serving.

Anything that changes what text reaches the embedding model must bump
``TEXT_PREP_VERSION``: it is part of the embedding cache key, so a change here
forces cached vectors to be rebuilt instead of silently reused.
"""

import hashlib
import html
import re
import unicodedata

# Bump whenever clean_text/article_text change their output (see module docstring).
TEXT_PREP_VERSION = 1

# bge-small truncates at 512 tokens (~2,000 characters of English). Titles plus
# abstracts are far shorter; the cap only guards against runaway summaries.
MAX_CHARS = 1000

_WHITESPACE = re.compile(r"\s+")
_TAG = re.compile(r"<[^>]*>")
_TERMINAL_PUNCT = (".", "!", "?", ":", ";")


def normalize_whitespace(text: str) -> str:
    """Collapse runs of whitespace into single spaces and trim the ends."""
    return _WHITESPACE.sub(" ", text).strip()


def clean_text(text: str) -> str:
    """Unescape HTML entities, drop tags, apply NFKC and collapse whitespace."""
    text = html.unescape(text)
    text = _TAG.sub(" ", text)
    text = unicodedata.normalize("NFKC", text)
    return normalize_whitespace(text)


def truncate(text: str, max_chars: int = MAX_CHARS) -> str:
    """Cut ``text`` to at most ``max_chars``, on a word boundary when possible."""
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    space = cut.rfind(" ")
    return (cut[:space] if space > 0 else cut).rstrip()


def article_text(title: str, summary: str | None = None, max_chars: int = MAX_CHARS) -> str:
    """Build the text we embed for an article: ``title + ". " + summary``.

    The separator is a single space when the title already ends in punctuation,
    so "Who won?" does not become "Who won?. ...". An empty summary yields the
    title alone.
    """
    title = clean_text(title)
    summary = clean_text(summary or "")
    if not summary:
        text = title
    elif not title:
        text = summary
    elif title.endswith(_TERMINAL_PUNCT):
        text = f"{title} {summary}"
    else:
        text = f"{title}. {summary}"
    return truncate(text, max_chars)


# Separators publishers put between a headline and their own name: "Title - CNN",
# "Title | Reuters", "Title — The Guardian", "Title :: NPR".
_SUFFIX_SEPARATORS = (" - ", " | ", " \u2014 ", " \u2013 ", " :: ", " \u00b7 ")
_NON_ALNUM = re.compile(r"[^0-9a-z]+")


def _squash(text: str) -> str:
    return _NON_ALNUM.sub("", unicodedata.normalize("NFKC", text).casefold())


def strip_source_suffix(title: str, source_name: str = "", domain: str = "") -> str:
    """Drop a trailing " - Publisher" from a headline when it names the source.

    Only the last separator is considered, and only removed when what follows it
    matches the source name or the domain's first label (ignoring case, spaces and
    punctuation), so "Biden - Harris debate" is left alone.
    """
    title = normalize_whitespace(title)
    names = {_squash(source_name)} if source_name else set()
    if domain:
        host = domain.lower().removeprefix("www.")
        names |= {_squash(host), _squash(host.split(".")[0])}
    names.discard("")
    best = -1
    for sep in _SUFFIX_SEPARATORS:
        best = max(best, title.rfind(sep))
    if best <= 0:
        return title
    sep_len = next(len(sep) for sep in _SUFFIX_SEPARATORS if title.startswith(sep, best))
    tail = _squash(title[best + sep_len :])
    if tail and any(tail == n or (len(n) >= 4 and n in tail) for n in names):
        return title[:best].rstrip()
    return title


def title_fingerprint(title: str) -> str:
    """A hash of the headline's words, for spotting the same headline on many sites.

    NFKC + casefold, keep alphanumeric tokens, join with single spaces, SHA1.
    Punctuation, case and spacing differences don't change it.
    """
    tokens = re.findall(r"[0-9a-z]+", unicodedata.normalize("NFKC", clean_text(title)).casefold())
    return hashlib.sha1(" ".join(tokens).encode()).hexdigest()  # noqa: S324 (not security)
