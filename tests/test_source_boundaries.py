"""Architecture and compatibility gates for the source-cleanup refactor."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest


def test_core_package_does_not_import_investigation() -> None:
    violations = []
    root = Path(__file__).resolve().parents[1] / "azgomoku"
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] == "investigation":
                violations.append(f"{path.relative_to(root.parent)}:{node.lineno}")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] == "investigation":
                        violations.append(f"{path.relative_to(root.parent)}:{node.lineno}")
    assert violations == []


def test_average_precision_preserves_stable_tie_order() -> None:
    from azgomoku.metrics.semantic_alignment import average_precision

    assert average_precision([True, False, True], [1.0, 1.0, 1.0]) == pytest.approx((1.0 + 2.0 / 3.0) / 2.0)


def test_compatibility_imports_are_identity_aliases() -> None:
    from azgomoku.metrics.attention import collapse_metrics
    from azgomoku.metrics.semantic_alignment import critical_ids
    from azgomoku.proof_replay import replay_flat_proof, replay_record_proofs
    from azgomoku.metrics.semantic_xai import aggregate_multi_proof, proof_semantic_targets
    from investigation.e3b_common import replay_flat_proof as legacy_flat
    from investigation.e3b_common import replay_record_proofs as legacy_record
    from investigation.e3b_pipeline import _collapse_metrics
    from investigation.evaluate_h1 import critical_ids as legacy_critical_ids
    from investigation.semantic_xai import aggregate_multi_proof as legacy_multi_proof
    from investigation.semantic_xai import proof_semantic_targets as legacy_semantic_targets

    assert legacy_flat is replay_flat_proof
    assert legacy_record is replay_record_proofs
    assert legacy_critical_ids is critical_ids
    assert _collapse_metrics is collapse_metrics
    assert legacy_multi_proof is aggregate_multi_proof
    assert legacy_semantic_targets is proof_semantic_targets
