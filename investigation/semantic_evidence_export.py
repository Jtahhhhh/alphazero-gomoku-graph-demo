"""Compatibility CLI for :mod:`azgomoku.semantic.evidence_export`."""

from azgomoku.semantic.evidence_export import (
    BASE_FILENAMES,
    ITERATIONS,
    MCTS_ITERATIONS,
    MODELS,
    OVERLAY_FILENAMES,
    OverlayWriter,
    _common_provenance,
    _overlay_hashes,
    _write_json,
    assert_base_kg_unchanged,
    export_evidence_overlay,
    extract_checkpoint_state_overlay,
    freeze_base_kg,
    load_base_entities,
    load_checkpoint_index,
    main,
    run_full_d4_release_gate,
    verify_evidence_release,
)

__all__ = [
    "BASE_FILENAMES",
    "ITERATIONS",
    "MCTS_ITERATIONS",
    "MODELS",
    "OVERLAY_FILENAMES",
    "OverlayWriter",
    "assert_base_kg_unchanged",
    "export_evidence_overlay",
    "extract_checkpoint_state_overlay",
    "freeze_base_kg",
    "load_base_entities",
    "load_checkpoint_index",
    "main",
    "run_full_d4_release_gate",
    "verify_evidence_release",
]


if __name__ == "__main__":
    main()
