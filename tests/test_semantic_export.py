import json
from pathlib import Path

from azgomoku.semantic.export_kg import (
    export_records,
    load_records,
    select_pilot_records,
    sha256_file,
)


BENCHMARK = Path("diagnostic/h1_benchmark_v1/h1_benchmark_v1.jsonl")


def test_three_state_pilot_exports_valid_jsonl_without_mutating_source(tmp_path):
    before = sha256_file(BENCHMARK)
    records = load_records(BENCHMARK)
    pilot = select_pilot_records(records)
    manifest = export_records(
        pilot,
        tmp_path,
        source_benchmark=str(BENCHMARK),
        benchmark_sha256=before,
        d4_validation={"passed": True, "test_fixture": True},
    )
    assert sha256_file(BENCHMARK) == before
    assert manifest["state_count"] == 3
    assert manifest["proof_bearing_state_count"] == 2
    assert manifest["no_proof_state_count"] == 1
    assert manifest["replay_backed_proof_count"] == 3
    assert manifest["epistemic_counts"]["HEURISTIC"] == 0
    assert manifest["epistemic_counts"]["EXACT"] > 0
    assert manifest["epistemic_counts"]["CERTIFIED"] == 3
    for name in ("entities.jsonl", "facts.jsonl", "provenance.jsonl", "manifest.json"):
        assert (tmp_path / name).is_file()
    facts = [json.loads(line) for line in (tmp_path / "facts.jsonl").read_text(encoding="utf-8").splitlines()]
    assert all(item["predicate"] not in {"EXTENDS", "HAS_OPEN_END"} for item in facts)
    assert not any(item["epistemic_class"] == "HEURISTIC" for item in facts)

    repeated = tmp_path / "repeated"
    export_records(
        pilot,
        repeated,
        source_benchmark=str(BENCHMARK),
        benchmark_sha256=before,
        d4_validation={"passed": True, "test_fixture": True},
    )
    for name in ("entities.jsonl", "facts.jsonl", "provenance.jsonl", "manifest.json"):
        assert (tmp_path / name).read_bytes() == (repeated / name).read_bytes()
