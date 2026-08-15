"""Fail-closed access to the frozen E-3b H1 benchmark contract."""

from __future__ import annotations

import json
from pathlib import Path

from azgomoku.h1_schema import state_from_record, validate_record


def phase_of(record: dict) -> str:
    """Return the frozen benchmark phase for a validated record."""
    ply = int(record["provenance"]["ply"])
    if 5 <= ply <= 9:
        return "mid"
    if ply >= 10:
        return "late"
    raise ValueError(f"gold state outside E-3b phases: ply={ply}")


def load_gold_fail_closed(path: Path) -> list[dict]:
    """Load only exact-complete, eligible 6x6/k=4 frozen-gold records."""
    records = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        validation = validate_record(record)
        if not validation.accepted:
            raise ValueError(f"record {line_number} rejected: {validation.errors}")
        normalized = validation.record
        normalized.pop("_validation", None)
        solver = normalized["solver"]
        if (
            solver["status"] != "exact_complete"
            or solver["method"] != "full_minimax"
            or not solver["optimal_actions_complete"]
            or not validation.eligible
        ):
            raise ValueError(f"record {line_number} is not exact-complete gold")
        state = state_from_record(normalized)
        if state.size != 6 or state.win_length != 4:
            raise ValueError(f"record {line_number} is not 6x6/k=4")
        phase_of(normalized)
        records.append(normalized)
    if not records:
        raise ValueError("gold benchmark is empty")
    if len({record["state_id"] for record in records}) != len(records):
        raise ValueError("duplicate state_id in gold")
    return records
