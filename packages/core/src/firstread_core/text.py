"""Text helpers shared by training, ingestion and serving."""

import re

_WHITESPACE = re.compile(r"\s+")


def normalize_whitespace(text: str) -> str:
    """Collapse runs of whitespace into single spaces and trim the ends."""
    return _WHITESPACE.sub(" ", text).strip()
