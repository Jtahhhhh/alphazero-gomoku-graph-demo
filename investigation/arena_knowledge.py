"""Post-process arena moves with solver-backed knowledge.svg artifacts."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np

from azgomoku.explanation.explanation_export import load_model
from azgomoku.explanation.explanation_schema import state_identifier
from azgomoku.explanation.model_evidence import collect_model_evidence
from azgomoku.explanation.rendering import render_knowledge_notice_svg, render_knowledge_svg
from azgomoku.game import GomokuState
from azgomoku.ground_truth import GroundTruthBudget, route_ground_truth
from azgomoku.symmetry import transform_flat_proof
from azgomoku.tactics import extract_tactical_proofs
from azgomoku.vcf import solve_vcf
from investigation.e3b_common import replay_flat_proof
from investigation.e3b_graph import coordinate_gate, structural_edges
from azgomoku.metrics.attention import collapse_metrics as _collapse_metrics
from azgomoku.metrics.semantic_alignment import aggregate_proofs


def state_from_explanation(document):
    item = document["state"]
    last = item.get("last_move")
    last_action = -1 if last is None else int(last[0]) * int(item["board_size"]) + int(last[1])
    return GomokuState(
        np.asarray(item["board"], dtype=np.int8),
        int(item["current_player"]),
        last_action,
        int(item["win_length"]),
    )


def _state_dict(state):
    return {
        "board_size": state.size,
        "win_length": state.win_length,
        "current_player": int(state.to_play),
        "last_move": int(state.last_move),
        "board": state.board.astype(int).tolist(),
        "legal_actions": list(map(int, state.legal_actions())),
    }


def _deduplicate_proofs(proofs, state):
    unique = {}
    for proof in proofs:
        canonical = transform_flat_proof(proof, state.size, 0)
        unique[json.dumps(canonical, sort_keys=True, separators=(",", ":"))] = canonical
    return list(unique.values())


def solve_arena_record(state, budget):
    """Route labels, then add only replayed flat proofs for those labels."""
    result = route_ground_truth(state, budget)
    solver = result.dict()
    proofs = []
    certificates = []
    if result.status == "exact_complete":
        optimal = set(map(int, result.optimal_actions or ()))
        for item in extract_tactical_proofs(state):
            proof = copy.deepcopy(item)
            proof.update({"proof_method": "tactical_replay", "proof_status": "exact"})
            if int(proof["action"]) in optimal and replay_flat_proof(state, proof):
                proofs.append(proof)
        if result.value == 1 and not proofs:
            vcf = solve_vcf(state, node_cap=budget.node_cap, time_cap_ms=budget.time_cap_ms)
            if vcf.status == "exact_partial" and vcf.proof is not None:
                for item in vcf.valid_proofs:
                    proof = copy.deepcopy(item)
                    action = int(proof["action"])
                    certificate_id = f"arena-vcf:{state_identifier(state)}:{action}"
                    certificate = {"certificate_id": certificate_id, "action": action, "tree": vcf.proof.dict()}
                    proof["certificate_id"] = certificate_id
                    if action in optimal and replay_flat_proof(state, proof, certificate):
                        proofs.append(proof)
                        certificates.append(certificate)
    elif result.status == "exact_partial" and result.proof is not None:
        for item in result.valid_proofs:
            proof = copy.deepcopy(item)
            action = int(proof["action"])
            certificate_id = f"arena-vcf:{state_identifier(state)}:{action}"
            certificate = {"certificate_id": certificate_id, "action": action, "tree": result.proof.dict()}
            proof["certificate_id"] = certificate_id
            if replay_flat_proof(state, proof, certificate):
                proofs.append(proof)
                certificates.append(certificate)
    proofs = _deduplicate_proofs(proofs, state)
    return {
        "schema_version": 1,
        "artifact_type": "arena_solver_knowledge",
        "state_id": state_identifier(state),
        "state": _state_dict(state),
        "solver": solver,
        "valid_proofs": proofs,
        "proof_certificates": certificates,
    }


def _load_or_solve(move_dir, state, budget):
    cache_path = move_dir / "knowledge.json"
    if cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if cached.get("state_id") != state_identifier(state):
            raise RuntimeError(f"stale arena knowledge cache: {cache_path}")
        return cached, True
    record = solve_arena_record(state, budget)
    cache_path.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    return record, False


def render_arena_knowledge(arena_dir, rgat_checkpoint, budget=GroundTruthBudget()):
    arena_dir = Path(arena_dir)
    checkpoint = Path(rgat_checkpoint)
    move_dirs = sorted(path.parent for path in arena_dir.glob("game_*/move_*/explanation.json"))
    if not move_dirs:
        raise ValueError("arena contains no completed explanation moves")
    first = json.loads((move_dirs[0] / "explanation.json").read_text(encoding="utf-8"))
    board_size = int(first["state"]["board_size"])
    rgat_model = load_model("rgat", checkpoint, board_size)
    manifest = {
        "artifact_type": "arena_solver_knowledge_manifest",
        "arena": str(arena_dir),
        "rgat_checkpoint": str(checkpoint),
        "budget": {"node_cap": budget.node_cap, "time_cap_ms": budget.time_cap_ms},
        "moves": [],
    }
    for index, move_dir in enumerate(move_dirs, 1):
        document = json.loads((move_dir / "explanation.json").read_text(encoding="utf-8"))
        state = state_from_explanation(document)
        record, cached = _load_or_solve(move_dir, state, budget)
        proofs = record.get("valid_proofs", [])
        if proofs:
            gate = coordinate_gate([record])
            rgat_edges = collect_model_evidence(state, rgat_model, int(proofs[0]["action"]))["graph_evidence"]["edges"]
            metrics = _collapse_metrics(rgat_edges)
            alignment = aggregate_proofs(rgat_edges, [float(edge["attention"]) for edge in rgat_edges], proofs)["mean"]
            metrics["graph_critical_mass"] = alignment["mass"]
            payload = {
                "record": record,
                "rgat_edges": rgat_edges,
                "structural_edges": structural_edges(state.size),
                "metrics": metrics,
                "graph_gate": gate,
            }
            svg = render_knowledge_svg(payload)
            render_status = "contrast_rendered"
        else:
            reason = record["solver"].get("unknown_reason") or record["solver"].get("coverage_note") or "no replayed tactical/VCF proof"
            svg = render_knowledge_notice_svg(record, reason)
            render_status = "notice_no_proof"
        (move_dir / "knowledge.svg").write_text(svg, encoding="utf-8")
        manifest["moves"].append({
            "move_dir": str(move_dir.relative_to(arena_dir)).replace("\\", "/"),
            "state_id": record["state_id"],
            "solver_status": record["solver"]["status"],
            "optimal_actions_complete": bool(record["solver"].get("optimal_actions_complete", False)),
            "proof_count": len(proofs),
            "render_status": render_status,
            "cache_reused": cached,
        })
        print(json.dumps({"move": index, "total": len(move_dirs), "status": record["solver"]["status"], "proofs": len(proofs)}), flush=True)
    manifest["counts"] = {
        key: sum(item["render_status"] == key for item in manifest["moves"])
        for key in ("contrast_rendered", "notice_no_proof")
    }
    manifest["solver_status_counts"] = {
        key: sum(item["solver_status"] == key for item in manifest["moves"])
        for key in ("exact_complete", "exact_partial", "unknown")
    }
    (arena_dir / "knowledge_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arena", type=Path, default=Path("results/h3_pilot/arena/rgcn_vs_rgat_data"))
    parser.add_argument("--rgat-checkpoint", type=Path, default=Path("results/h3_pilot_v2/rgat/seed_7/model.pt"))
    parser.add_argument("--node-cap", type=int, default=1_000_000)
    parser.add_argument("--time-cap-ms", type=int, default=2_000)
    args = parser.parse_args()
    manifest = render_arena_knowledge(
        args.arena,
        args.rgat_checkpoint,
        GroundTruthBudget(node_cap=args.node_cap, time_cap_ms=args.time_cap_ms),
    )
    print(json.dumps({"counts": manifest["counts"], "solver_status_counts": manifest["solver_status_counts"]}, indent=2))


if __name__ == "__main__":
    main()
