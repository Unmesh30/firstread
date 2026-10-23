"""Ranking: the one implementation used offline (ml) and online (API) (ADR 0014).

``final = normalize(model) + w_fresh·freshness(age) + w_trend·trend_feature(trend)``,
then MMR with at most ``story_cap`` items per story group, then reasons. Every
constant comes from one versioned config (``rank-<n>.yaml``).
"""

from firstread_core.ranking.cold_start import cold_start_vector, topic_weight
from firstread_core.ranking.config import RankingConfig, SegmentWeights, load_config, model_version
from firstread_core.ranking.features import blend, freshness, normalize, trend_feature
from firstread_core.ranking.mmr import MMRResult, mmr
from firstread_core.ranking.reasons import Reason, ReasonInputs, reasons

__all__ = [
    "MMRResult",
    "RankingConfig",
    "Reason",
    "ReasonInputs",
    "SegmentWeights",
    "blend",
    "cold_start_vector",
    "freshness",
    "load_config",
    "mmr",
    "model_version",
    "normalize",
    "reasons",
    "topic_weight",
    "trend_feature",
]
