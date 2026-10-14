"""Shared FirstRead code.

Text cleaning, URL canonicalization, the embedding function, feature code and
pydantic schemas live here, and only here, so training and serving can never
drift apart (train/serve skew).
"""

from firstread_core.text import (
    MAX_CHARS,
    TEXT_PREP_VERSION,
    article_text,
    clean_text,
    normalize_whitespace,
)

__all__ = [
    "MAX_CHARS",
    "TEXT_PREP_VERSION",
    "__version__",
    "article_text",
    "clean_text",
    "normalize_whitespace",
]

__version__ = "0.1.0"
