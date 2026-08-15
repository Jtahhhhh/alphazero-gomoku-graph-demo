from pathlib import Path

from azgomoku.semantic.export_kg import load_records, semantic_d4_gate


BENCHMARK = Path("diagnostic/h1_benchmark_v1/h1_benchmark_v1.jsonl")


def test_semantic_d4_gate_checks_roundtrip_references_epistemic_and_counts():
    record = load_records(BENCHMARK)[0]
    gate = semantic_d4_gate([record], verify_roundtrip=True)
    assert gate["passed"]
    assert gate["record_transform_checks"] == 8
    assert gate["roundtrip_checks"] == 8
    assert gate["fact_count_invariance_checks"] == 8
    assert gate["referential_integrity_checks"] == gate["fact_transform_checks"]
    assert gate["epistemic_invariance_checks"] == gate["fact_transform_checks"]

