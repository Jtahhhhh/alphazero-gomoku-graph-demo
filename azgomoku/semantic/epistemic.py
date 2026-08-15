"""Fact-level epistemic classes for Semantic KG v1."""

from enum import Enum


class EpistemicClass(str, Enum):
    """Mutually exclusive source-of-knowledge classes.

    EXACT is restricted to full-minimax, exact-complete results. CERTIFIED is
    restricted to replayed tactical/VCF proof claims. DERIVED is deterministic
    structure or serialization. HEURISTIC is reserved and may not be emitted by
    Phase 1-3. LEARNED is model or search evidence and is never tactical truth.
    """

    EXACT = "EXACT"
    CERTIFIED = "CERTIFIED"
    DERIVED = "DERIVED"
    HEURISTIC = "HEURISTIC"
    LEARNED = "LEARNED"


def epistemic_value(value: EpistemicClass | str) -> str:
    return value.value if isinstance(value, EpistemicClass) else str(value)
