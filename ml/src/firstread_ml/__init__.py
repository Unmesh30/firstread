"""Offline training and evaluation on MIND. Implemented in Phase 1."""

__version__ = "0.0.0"

# The MIND-standard ranking metrics we will report (SPEC section 7.6).
METRICS = ("auc", "mrr", "ndcg@5", "ndcg@10")
