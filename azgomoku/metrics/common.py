"""Small shared helpers for metric aggregation."""

from __future__ import annotations

import numpy as np


def mean_or_none(values) -> float | None:
    return None if not values else float(np.mean(values))
