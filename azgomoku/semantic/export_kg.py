"""Validated JSONL export for the source-grounded Semantic KG v1."""

from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
from typing import Iterable

from azgomoku.artifacts import sha256_file, write_json as _write_json
from azgomoku.explanation.explanation_schema import state_identifier
from azgomoku.h1_schema import state_from_record, validate_record
from azgomoku.proof_replay import replay_record_proofs
from azgomoku.symmetry import (
    inverse_symmetry,
    transform_action,
    transform_flat_proof,
    transform_relation,
    transform_state,
)
from .extract_proofs import extract_record_proofs
from .extract_state import extract_state
from .extract_tactics import extract_tactics
from .identity import canonical_json, normalize_flat_proof, transform_proof_node_dict
from .predicates import Predicate, predicate_value
from .schema import SemanticArtifact, entity_type_value
from .validation import validate_artifact


SEMANTIC_KG_VERSION = "semantic_kg_v1"
PILOT_STATE_IDS = (
    "4dca2566ec2be9b6",  # tactical replay proofs
    "b3a6c7628630359d",  # VCF certificate
    "74c55e1c7c911cc9",  # exact-complete, no tactical proof
)


def load_records(path: Path, *, require_exact_complete: bool = True) -> list[dict]:
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
        if require_exact_complete and (
            normalized["solver"].get("status") != "exact_complete"
            or normalized["solver"].get("method") != "full_minimax"
            or not normalized["solver"].get("optimal_actions_complete")
        ):
            raise ValueError(f"record {line_number} is not exact-complete full-minimax gold")
        passed, total = replay_record_proofs(normalized)
        if passed != total:
            raise ValueError(f"record {line_number} proof replay failed: {passed}/{total}")
        records.append(normalized)
    if not records:
        raise ValueError("semantic export source is empty")
    if len({record["state_id"] for record in records}) != len(records):
        raise ValueError("semantic export source has duplicate state_id")
    return records


def extract_record_semantics(record: dict, *, artifact_ref: str) -> SemanticArtifact:
    state = state_from_record(record)
    artifact = extract_state(state)
    extract_tactics(state, artifact)
    extract_record_proofs(record, artifact, artifact_ref=artifact_ref)
    validate_artifact(artifact, raise_on_error=True)
    return artifact


def _transform_record(record: dict, symmetry: int) -> dict:
    transformed_record = copy.deepcopy(record)
    state = state_from_record(record)
    transformed = transform_state(state, symmetry)
    new_state_id = state_identifier(transformed)
    transformed_record["state_id"] = new_state_id
    transformed_record["state"] = {
        "board_size": transformed.size,
        "win_length": transformed.win_length,
        "current_player": int(transformed.to_play),
        "last_move": int(transformed.last_move),
        "board": transformed.board.astype(int).tolist(),
        "legal_actions": sorted(map(int, transformed.legal_actions())),
    }
    provenance = transformed_record.get("provenance", {})
    if "history" in provenance:
        provenance["history"] = [
            transform_action(int(action), state.size, symmetry)
            for action in provenance["history"]
        ]

    solver = transformed_record["solver"]
    if solver.get("optimal_actions") is not None:
        solver["optimal_actions"] = sorted(
            transform_action(int(action), state.size, symmetry)
            for action in solver["optimal_actions"]
        )
    if solver.get("action_values") is not None:
        solver["action_values"] = {
            str(transform_action(int(action), state.size, symmetry)): value
            for action, value in solver["action_values"].items()
        }
    if solver.get("proof") is not None:
        solver["proof"] = transform_proof_node_dict(solver["proof"], state.size, symmetry)
    solver["valid_proofs"] = [
        transform_flat_proof(proof, state.size, symmetry)
        for proof in solver.get("valid_proofs", [])
    ]

    id_map = {}
    certificates = []
    for certificate in transformed_record.get("proof_certificates", []):
        mapped_action = transform_action(int(certificate["action"]), state.size, symmetry)
        new_id = f"d4:{new_state_id}:{mapped_action}:{hashlib.sha256(certificate['certificate_id'].encode()).hexdigest()[:12]}"
        id_map[certificate["certificate_id"]] = new_id
        certificates.append(
            {
                **certificate,
                "certificate_id": new_id,
                "action": mapped_action,
                "tree": transform_proof_node_dict(certificate["tree"], state.size, symmetry),
            }
        )
    transformed_record["proof_certificates"] = certificates
    proofs = []
    for proof in transformed_record.get("valid_proofs", []):
        mapped = transform_flat_proof(proof, state.size, symmetry)
        if proof.get("certificate_id") in id_map:
            mapped["certificate_id"] = id_map[proof["certificate_id"]]
        proofs.append(mapped)
    transformed_record["valid_proofs"] = proofs
    return transformed_record


def _fact_signature(artifact: SemanticArtifact, fact, *, symmetry: int | None = None):
    subject = artifact.entities[fact.subject_id]
    predicate = predicate_value(fact.predicate)
    if fact.object_id is not None:
        target = ("entity", artifact.entities[fact.object_id].canonical_key)
    else:
        value = fact.value
        if symmetry is not None and predicate == Predicate.HAS_DIRECTION.value:
            value = transform_relation(str(value), symmetry)
        target = ("value", value)
    return (
        subject.canonical_key,
        predicate,
        target,
        fact.epistemic_class.value if hasattr(fact.epistemic_class, "value") else str(fact.epistemic_class),
    )


def _record_roundtrip_signature(record: dict) -> str:
    solver = record["solver"]
    certificates = [
        {
            "action": int(item["action"]),
            "tree": item["tree"],
        }
        for item in record.get("proof_certificates", [])
    ]
    certificates.sort(key=canonical_json)
    return canonical_json(
        {
            "state_id": record["state_id"],
            "state": record["state"],
            "history": record.get("provenance", {}).get("history"),
            "solver": {
                "value": solver.get("value"),
                "optimal_actions": sorted(map(int, solver.get("optimal_actions", []))),
                "action_values": {
                    str(key): value
                    for key, value in sorted(
                        solver.get("action_values", {}).items(),
                        key=lambda item: int(item[0]),
                    )
                },
                "proof": solver.get("proof"),
                "valid_proofs": sorted(
                    (normalize_flat_proof(item) for item in solver.get("valid_proofs", [])),
                    key=canonical_json,
                ),
            },
            "valid_proofs": sorted(
                (normalize_flat_proof(item) for item in record.get("valid_proofs", [])),
                key=canonical_json,
            ),
            "proof_certificates": certificates,
        }
    )


def semantic_d4_gate(
    records: Iterable[dict],
    *,
    verify_roundtrip: bool = False,
    progress: bool = False,
) -> dict:
    records = list(records)
    record_checks = 0
    fact_checks = 0
    entity_checks = 0
    reference_checks = 0
    epistemic_checks = 0
    fact_count_checks = 0
    proof_lineage_checks = 0
    roundtrip_checks = 0
    raw_id_checks = 0
    for record in records:
        original = extract_record_semantics(record, artifact_ref=f"d4-gate:{record['state_id']}")
        original_entity_keys = {
            (entity_type_value(item.entity_type), item.canonical_key)
            for item in original.entities.values()
        }
        for symmetry in range(8):
            mapped_record = _transform_record(record, symmetry)
            mapped = extract_record_semantics(
                mapped_record,
                artifact_ref=f"d4-gate:{mapped_record['state_id']}",
            )
            wrongly_scoped = [
                entity.entity_id
                for entity in mapped.entities.values()
                if entity.state_id != mapped_record["state_id"]
            ]
            if wrongly_scoped:
                raise RuntimeError(
                    f"D4 raw identity state scope failed state={record['state_id']} "
                    f"symmetry={symmetry} entities={wrongly_scoped[:3]}"
                )
            mapped_entity_keys = {
                (entity_type_value(item.entity_type), item.canonical_key)
                for item in mapped.entities.values()
            }
            if original_entity_keys != mapped_entity_keys:
                missing = sorted(original_entity_keys - mapped_entity_keys)[:3]
                extra = sorted(mapped_entity_keys - original_entity_keys)[:3]
                raise RuntimeError(
                    f"D4 entity equivalence failed state={record['state_id']} symmetry={symmetry} "
                    f"missing={missing} extra={extra}"
                )
            original_facts = {
                _fact_signature(original, fact, symmetry=symmetry)
                for fact in original.facts.values()
            }
            mapped_facts = {
                _fact_signature(mapped, fact)
                for fact in mapped.facts.values()
            }
            if original_facts != mapped_facts:
                missing = list(original_facts - mapped_facts)[:3]
                extra = list(mapped_facts - original_facts)[:3]
                raise RuntimeError(
                    f"D4 fact equivalence failed state={record['state_id']} symmetry={symmetry} "
                    f"missing={missing} extra={extra}"
                )
            if len(original.facts) != len(mapped.facts):
                raise RuntimeError(
                    f"D4 fact count changed state={record['state_id']} symmetry={symmetry}"
                )
            original_certified = sum(
                fact.epistemic_class.value == "CERTIFIED"
                for fact in original.facts.values()
            )
            mapped_certified = sum(
                fact.epistemic_class.value == "CERTIFIED"
                for fact in mapped.facts.values()
            )
            if original_certified != mapped_certified:
                raise RuntimeError(
                    f"D4 proof lineage changed state={record['state_id']} symmetry={symmetry}"
                )
            raw_id_checks += len(mapped.entities)
            reference_checks += len(mapped.facts)
            epistemic_checks += len(mapped.facts)
            fact_count_checks += 1
            proof_lineage_checks += mapped_certified
            if verify_roundtrip:
                inverse = inverse_symmetry(symmetry, state_from_record(mapped_record).size)
                restored = _transform_record(mapped_record, inverse)
                if _record_roundtrip_signature(restored) != _record_roundtrip_signature(record):
                    raise RuntimeError(
                        f"D4 semantic round-trip failed state={record['state_id']} symmetry={symmetry}"
                    )
                roundtrip_checks += 1
            record_checks += 1
            entity_checks += len(original_entity_keys)
            fact_checks += len(original_facts)
        if progress:
            print(
                json.dumps(
                    {
                        "stage": "semantic_d4_release_gate",
                        "record": record["state_id"],
                        "completed": record_checks // 8,
                        "total": len(records),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    return {
        "passed": True,
        "records": len(records),
        "symmetries_per_record": 8,
        "record_transform_checks": record_checks,
        "entity_canonical_checks": entity_checks,
        "fact_transform_checks": fact_checks,
        "referential_integrity_checks": reference_checks,
        "epistemic_invariance_checks": epistemic_checks,
        "fact_count_invariance_checks": fact_count_checks,
        "proof_lineage_checks": proof_lineage_checks,
        "raw_id_checks": raw_id_checks,
        "roundtrip_checks": roundtrip_checks,
    }


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False))
            handle.write("\n")
    temporary.replace(path)


def export_records(
    records: Iterable[dict],
    output_dir: Path,
    *,
    source_benchmark: str,
    benchmark_sha256: str,
    d4_validation: dict | None = None,
) -> dict:
    records = list(records)
    artifact = SemanticArtifact()
    for record in records:
        artifact.merge(
            extract_record_semantics(
                record,
                artifact_ref=f"{source_benchmark}#state_id={record['state_id']}",
            )
        )
    validation = validate_artifact(artifact, raise_on_error=True)
    output_dir = Path(output_dir)
    entities = sorted(artifact.entities.values(), key=lambda item: item.entity_id)
    facts = sorted(artifact.facts.values(), key=lambda item: item.fact_id)
    provenance = sorted(artifact.provenance.values(), key=lambda item: item.provenance_id)
    _write_jsonl(output_dir / "entities.jsonl", (item.dict() for item in entities))
    _write_jsonl(output_dir / "facts.jsonl", (item.dict() for item in facts))
    _write_jsonl(output_dir / "provenance.jsonl", (item.dict() for item in provenance))

    epistemic_counts = Counter(
        fact.epistemic_class.value if hasattr(fact.epistemic_class, "value") else str(fact.epistemic_class)
        for fact in facts
    )
    predicate_counts = Counter(predicate_value(fact.predicate) for fact in facts)
    entity_type_counts = Counter(entity_type_value(item.entity_type) for item in entities)
    proof_total = sum(len(record.get("valid_proofs", [])) for record in records)
    proof_records = sum(bool(record.get("valid_proofs")) for record in records)
    manifest = {
        "schema_version": 1,
        "semantic_kg_version": SEMANTIC_KG_VERSION,
        "source_benchmark": source_benchmark,
        "benchmark_sha256": benchmark_sha256,
        "state_count": len(records),
        "proof_bearing_state_count": proof_records,
        "no_proof_state_count": len(records) - proof_records,
        "replay_backed_proof_count": proof_total,
        "entity_count": len(entities),
        "fact_count": len(facts),
        "provenance_count": len(provenance),
        "epistemic_counts": {
            name: int(epistemic_counts.get(name, 0))
            for name in ("EXACT", "CERTIFIED", "DERIVED", "HEURISTIC", "LEARNED")
        },
        "predicate_counts": dict(sorted(predicate_counts.items())),
        "entity_type_counts": dict(sorted(entity_type_counts.items())),
        "d4_validation": d4_validation,
        "validation_status": {
            "valid": validation.valid,
            "errors": list(validation.errors),
            "referential_integrity": True,
            "epistemic_integrity": True,
            "heuristic_fact_count": int(epistemic_counts.get("HEURISTIC", 0)),
        },
    }
    _write_json(output_dir / "manifest.json", manifest)
    return manifest


def select_pilot_records(records: Iterable[dict]) -> list[dict]:
    by_id = {record["state_id"]: record for record in records}
    missing = [state_id for state_id in PILOT_STATE_IDS if state_id not in by_id]
    if missing:
        raise ValueError(f"pilot states missing from source: {missing}")
    return [by_id[state_id] for state_id in PILOT_STATE_IDS]


def export_frozen_benchmark(
    benchmark: Path,
    output_dir: Path,
    *,
    pilot_output: Path | None = None,
) -> dict:
    benchmark = Path(benchmark)
    before_hash = sha256_file(benchmark)
    source_manifest = benchmark.with_name("manifest.json")
    source_manifest_before_hash = (
        sha256_file(source_manifest) if source_manifest.is_file() else None
    )
    records = load_records(benchmark)
    pilot = select_pilot_records(records)
    d4_gate = semantic_d4_gate(pilot)
    pilot_manifest = None
    if pilot_output is not None:
        pilot_manifest = export_records(
            pilot,
            pilot_output,
            source_benchmark=str(benchmark).replace("\\", "/"),
            benchmark_sha256=before_hash,
            d4_validation=d4_gate,
        )
    full_manifest = export_records(
        records,
        output_dir,
        source_benchmark=str(benchmark).replace("\\", "/"),
        benchmark_sha256=before_hash,
        d4_validation=d4_gate,
    )
    after_hash = sha256_file(benchmark)
    if after_hash != before_hash:
        raise RuntimeError("frozen benchmark changed during semantic export")
    source_manifest_after_hash = (
        sha256_file(source_manifest) if source_manifest.is_file() else None
    )
    if source_manifest_after_hash != source_manifest_before_hash:
        raise RuntimeError("frozen benchmark manifest changed during semantic export")
    full_manifest["freeze_integrity"] = {
        "passed": True,
        "sha256_before": before_hash,
        "sha256_after": after_hash,
        "source_manifest": str(source_manifest).replace("\\", "/"),
        "source_manifest_sha256_before": source_manifest_before_hash,
        "source_manifest_sha256_after": source_manifest_after_hash,
    }
    _write_json(Path(output_dir) / "manifest.json", full_manifest)
    return {"pilot": pilot_manifest, "frozen": full_manifest}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=Path("diagnostic/h1_benchmark_v1/h1_benchmark_v1.jsonl"),
    )
    parser.add_argument("--output", type=Path, default=Path("semantic_kg"))
    parser.add_argument("--pilot-output", type=Path, default=Path("semantic_kg/pilot"))
    args = parser.parse_args()
    result = export_frozen_benchmark(
        args.benchmark,
        args.output,
        pilot_output=args.pilot_output,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
