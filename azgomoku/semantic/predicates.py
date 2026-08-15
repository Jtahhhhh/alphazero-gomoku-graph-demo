"""Closed predicate vocabulary for Semantic KG v1."""

from enum import Enum


class Predicate(str, Enum):
    CONTAINS = "CONTAINS"
    PLAYED_AT = "PLAYED_AT"

    CREATES = "CREATES"
    BLOCKS = "BLOCKS"
    USES_CELL = "USES_CELL"
    HAS_COMPLETION = "HAS_COMPLETION"
    HAS_DIRECTION = "HAS_DIRECTION"
    FORCES = "FORCES"

    SUPPORTS = "SUPPORTS"
    HAS_WINDOW = "HAS_WINDOW"
    REQUIRES = "REQUIRES"

    OPTIMAL_IN = "OPTIMAL_IN"
    HAS_ACTION_VALUE = "HAS_ACTION_VALUE"

    CONNECTS = "CONNECTS"
    HAS_WEIGHT = "HAS_WEIGHT"
    OVERLAPS = "OVERLAPS"


# These names are intentionally not Predicate members and therefore cannot pass
# validation or be emitted accidentally.
UNAVAILABLE_PREDICATES = frozenset({"EXTENDS", "HAS_OPEN_END"})

VALUE_PREDICATES = frozenset(
    {
        Predicate.HAS_DIRECTION.value,
        Predicate.HAS_ACTION_VALUE.value,
        Predicate.HAS_WEIGHT.value,
    }
)

TACTICAL_TRUTH_PREDICATES = frozenset(
    {
        Predicate.CREATES.value,
        Predicate.BLOCKS.value,
        Predicate.HAS_COMPLETION.value,
        Predicate.HAS_DIRECTION.value,
        Predicate.FORCES.value,
        Predicate.SUPPORTS.value,
        Predicate.REQUIRES.value,
        Predicate.OPTIMAL_IN.value,
    }
)


def predicate_value(value: Predicate | str) -> str:
    return value.value if isinstance(value, Predicate) else str(value)
