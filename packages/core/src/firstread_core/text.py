"""Text helpers shared by training, ingestion and serving.

Anything that changes what text reaches the embedding model must bump
``TEXT_PREP_VERSION``: it is part of the embedding cache key, so a change here
forces cached vectors to be rebuilt instead of silently reused.
"""

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
