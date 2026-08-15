"""Fail-closed benchmark preparation primitives for Phase E-3b."""

from __future__ import annotations

import copy
import json
from pathlib import Path

from azgomoku.artifacts import sha256_file
from azgomoku.benchmark import load_gold_fail_closed, phase_of
from azgomoku.h1_schema import state_from_record
from azgomoku.proof_replay import replay_flat_proof, replay_record_proofs
from azgomoku.symmetry import transform_flat_proof
from azgomoku.tactics import extract_tactical_proofs
from azgomoku.vcf import solve_vcf


N_MIN = 30
ANNOTATION_VERSION = "e3b_replay_proofs_v1"


def annotate_gold(records: list[dict], vcf_time_cap_ms: int = 2_000) -> tuple[list[dict], dict]:
    """Add replayed tactical evidence while leaving exact labels untouched."""
    annotated = []
    stats = {
        "records": len(records),
        "tactical_proofs": 0,
        "vcf_attempts": 0,
        "vcf_proofs": 0,
        "proof_replay_passed": 0,
        "proofs_total": 0,
    }
    for original in records:
        record = copy.deepcopy(original)
        state = state_from_record(record)
        optimal = set(map(int, record["solver"]["optimal_actions"]))
        proofs = []
        for proof in extract_tactical_proofs(state):
            proof = copy.deepcopy(proof)
            proof.update({"proof_method": "tactical_replay", "proof_status": "exact"})
            if int(proof["action"]) in optimal and replay_flat_proof(state, proof):
                proofs.append(proof)
        stats["tactical_proofs"] += len(proofs)

        certificates = []
        if record["solver"]["value"] == 1 and not proofs:
            stats["vcf_attempts"] += 1
            result = solve_vcf(state, node_cap=1_000_000, time_cap_ms=vcf_time_cap_ms)
            if result.status == "exact_partial" and result.proof is not None:
                flat = copy.deepcopy(result.valid_proofs[0])
                action = int(flat["action"])
                certificate_id = f"vcf:{record['state_id']}:{action}"
                flat["certificate_id"] = certificate_id
                certificate = {
                    "certificate_id": certificate_id,
                    "action": action,
                    "tree": result.proof.dict(),
                }
                if action in optimal and replay_flat_proof(state, flat, certificate):
                    proofs.append(flat)
                    certificates.append(certificate)
                    stats["vcf_proofs"] += 1

        # Canonical list ordering is part of the coordinate contract. Even the
        # identity transform must be byte/structure preserving.
        proofs = [transform_flat_proof(proof, state.size, 0) for proof in proofs]
        unique = {}
        for proof in proofs:
            key = json.dumps(proof, sort_keys=True, separators=(",", ":"))
            unique[key] = proof
        record["valid_proofs"] = list(unique.values())
        record["proof_certificates"] = certificates
        record["proof_annotation"] = {
            "version": ANNOTATION_VERSION,
            "label_source_unchanged": True,
            "alignment_eligible": bool(record["valid_proofs"]),
        }
        passed, total = replay_record_proofs(record)
        if passed != total:
            raise RuntimeError(f"proof replay failed for {record['state_id']}: {passed}/{total}")
        stats["proof_replay_passed"] += passed
        stats["proofs_total"] += total
        annotated.append(record)

    stats["proof_records"] = sum(bool(record["valid_proofs"]) for record in annotated)
    stats["proof_records_by_phase"] = {
        phase: sum(phase_of(record) == phase and bool(record["valid_proofs"]) for record in annotated)
        for phase in ("mid", "late")
    }
    return annotated, stats


def write_jsonl(path: Path, records: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "".join(
        json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
        for record in records
    )
    path.write_text(content, encoding="utf-8")
