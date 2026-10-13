"""Shared FirstRead code.

Text cleaning, URL canonicalization, the embedding function, feature code and
pydantic schemas live here, and only here, so training and serving can never
drift apart (train/serve skew). Phase 0 only ships the package skeleton.
"""

from firstread_core.text import normalize_whitespace

__all__ = ["__version__", "normalize_whitespace"]

__version__ = "0.0.0"
