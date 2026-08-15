import json
from pathlib import Path

from azgomoku.h3_checkpoint import model_from_bundle
from azgomoku.semantic.evidence_schema import (
    EvidencePredicate,
    validate_evidence_overlay,
)
from investigation.semantic_evidence_export import (
    MODELS,
    extract_checkpoint_state_overlay,
    load_base_entities,
)
from investigation.e3b_common import sha256_file


BENCHMARK = Path("diagnostic/h1_benchmark_v1/h1_benchmark_v1.jsonl")
CHECKPOINT = Path("results/h3_pilot_v2/rgat/seed_7/checkpoints/iter_000.pt")


def test_real_rgat_checkpoint_emits_distinct_evidence_predicates_with_exact_joins():
    record = json.loads(BENCHMARK.read_text(encoding="utf-8").splitlines()[0])
    model, bundle = model_from_bundle(CHECKPOINT, MODELS)
    base_entities = load_base_entities(Path("semantic_kg"))
    overlay = extract_checkpoint_state_overlay(
        record,
        model,
        bundle,
        CHECKPOINT,
        sha256_file(CHECKPOINT),
        "b" * 64,
        include_mcts=False,
    )
    report = validate_evidence_overlay(overlay, base_entities, raise_on_error=True)
    predicates = {fact.predicate for fact in overlay.facts.values()}
    assert report.valid
    assert EvidencePredicate.HAS_POLICY_PROB in predicates
    assert EvidencePredicate.HAS_STATE_VALUE in predicates
    assert EvidencePredicate.OBSERVES in predicates
    assert EvidencePredicate.HAS_ATTENTION_WEIGHT in predicates
    assert "HAS_ACTION_VALUE" not in predicates
    assert not {"CREATES", "BLOCKS", "SUPPORTS"} & predicates

