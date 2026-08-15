"""Reusable metric primitives for semantic alignment and attention analysis."""

from .attention import collapse_metrics
from .semantic_alignment import (
    aggregate_proofs,
    average_precision,
    baselines,
    critical_ids,
    entropy,
    score_alignment,
)

__all__ = [
    "aggregate_proofs",
    "average_precision",
    "baselines",
    "collapse_metrics",
    "critical_ids",
    "entropy",
    "score_alignment",
]
