"""Post-process arena moves with solver-backed knowledge.svg artifacts."""

from __future__ import annotations

import argparse
import copy
import hashlib
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


ARTIFACT_VERSION = 2
ATTENTION_TOP_K = 20


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


def _json_bytes(document):
    return (json.dumps(document, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _write_json(path, document):
    content = _json_bytes(document)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return hashlib.sha256(content).hexdigest()


def _checkpoint_key(value):
    if value in (None, ""):
        return None
    return str(Path(value).expanduser().resolve())


def _load_game_moves(arena_dir):
    moves = {}
    for game_path in sorted(arena_dir.glob("game_*/game.json")):
        game = json.loads(game_path.read_text(encoding="utf-8"))
        for item in game.get("moves", []):
            artifact_dir = item.get("artifact_dir")
            if not artifact_dir:
                continue
            move_dir = (game_path.parent / artifact_dir).resolve()
            if move_dir in moves:
                raise RuntimeError(f"duplicate arena move artifact: {move_dir}")
            moves[move_dir] = item
    return moves


def _decision_context(document, game_move, attention_checkpoint):
    selected = {key: int(document["selected_move"][key]) for key in ("action", "row", "col")}
    size = int(document["state"]["board_size"])
    expected_row, expected_col = divmod(selected["action"], size)
    if (selected["row"], selected["col"]) != (expected_row, expected_col):
        raise RuntimeError("selected_move action/row/col mismatch in explanation.json")
    if int(game_move["action"]) != selected["action"]:
        raise RuntimeError("game.json action does not match explanation selected_move")
    if game_move.get("state_id") != document.get("state_id"):
        raise RuntimeError("game.json state_id does not match explanation state_id")

    actor = dict(game_move["actor"])
    document_model = document["model"]
    if actor.get("type") != document_model.get("type"):
        raise RuntimeError("game.json actor type does not match explanation model type")
    if _checkpoint_key(actor.get("checkpoint")) != _checkpoint_key(document_model.get("checkpoint")):
        raise RuntimeError("game.json actor checkpoint does not match explanation model checkpoint")

    attention_model = {"type": "rgat", "checkpoint": str(attention_checkpoint)}
    same_actor = actor.get("type") == "rgat" and _checkpoint_key(actor.get("checkpoint")) == _checkpoint_key(attention_checkpoint)
    return {
        "selected_move": selected,
        "actor": actor,
        "attention_source": {
            "model": attention_model,
            "relationship_to_actor": "actor" if same_actor else "counterfactual",
        },
    }


def _load_or_solve(move_dir, output_move_dir, state, budget):
    source_cache = move_dir / "knowledge.json"
    output_cache = output_move_dir / "knowledge.json"
    cache_candidates = [output_cache]
    if source_cache.resolve() != output_cache.resolve():
        cache_candidates.append(source_cache)
    for cache_path in cache_candidates:
        if not cache_path.exists():
            continue
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if cached.get("state_id") != state_identifier(state):
            raise RuntimeError(f"stale arena knowledge cache: {cache_path}")
        cached_budget = cached.get("solver", {}).get("budget", {})
        if (
            int(cached_budget.get("node_cap", -1)) != int(budget.node_cap)
            or int(cached_budget.get("time_cap_ms", -1)) != int(budget.time_cap_ms)
        ):
            continue
        if cache_path.resolve() != output_cache.resolve():
            _write_json(output_cache, cached)
        return cached, True
    record = solve_arena_record(state, budget)
    _write_json(output_cache, record)
    return record, False


def render_arena_knowledge(arena_dir, rgat_checkpoint, budget=GroundTruthBudget(), output_dir=None):
    arena_dir = Path(arena_dir)
    output_dir = arena_dir if output_dir is None else Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(rgat_checkpoint)
    move_dirs = sorted(path.parent for path in arena_dir.glob("game_*/move_*/explanation.json"))
    if not move_dirs:
        raise ValueError("arena contains no completed explanation moves")
    game_moves = _load_game_moves(arena_dir)
    missing_game_moves = [move_dir for move_dir in move_dirs if move_dir.resolve() not in game_moves]
    if missing_game_moves:
        raise RuntimeError(f"game.json lineage missing for {missing_game_moves[0]}")
    first = json.loads((move_dirs[0] / "explanation.json").read_text(encoding="utf-8"))
    board_size = int(first["state"]["board_size"])
    rgat_model = load_model("rgat", checkpoint, board_size)
    manifest = {
        "schema_version": 2,
        "artifact_version": ARTIFACT_VERSION,
        "artifact_type": "arena_solver_knowledge_manifest",
        "arena": str(arena_dir),
        "output": str(output_dir),
        "rgat_checkpoint": str(checkpoint),
        "attention_top_k": ATTENTION_TOP_K,
        "path_bases": {
            "game_path": "arena",
            "explanation_path": "arena",
            "solver_record_path": "output",
            "knowledge_svg_path": "output",
            "attention_evidence_path": "output",
        },
        "budget": {"node_cap": budget.node_cap, "time_cap_ms": budget.time_cap_ms},
        "moves": [],
    }
    for index, move_dir in enumerate(move_dirs, 1):
        relative_move_dir = move_dir.relative_to(arena_dir)
        output_move_dir = output_dir / relative_move_dir
        output_move_dir.mkdir(parents=True, exist_ok=True)
        document = json.loads((move_dir / "explanation.json").read_text(encoding="utf-8"))
        decision = _decision_context(document, game_moves[move_dir.resolve()], checkpoint)
        state = state_from_explanation(document)
        record, cached = _load_or_solve(move_dir, output_move_dir, state, budget)
        proofs = record.get("valid_proofs", [])
        evidence_path = None
        evidence_sha256 = None
        if proofs:
            gate = coordinate_gate([record])
            selected_action = int(decision["selected_move"]["action"])
            evidence = collect_model_evidence(state, rgat_model, selected_action)
            rgat_edges = evidence["graph_evidence"]["edges"]
            metrics = _collapse_metrics(rgat_edges)
            alignment = aggregate_proofs(rgat_edges, [float(edge["attention"]) for edge in rgat_edges], proofs)["mean"]
            metrics["graph_critical_mass"] = alignment["mass"]
            evidence_document = {
                "schema_version": 1,
                "artifact_version": ARTIFACT_VERSION,
                "artifact_type": "arena_knowledge_attention_evidence",
                "state_id": record["state_id"],
                "selected_move": decision["selected_move"],
                "actor": decision["actor"],
                "attention_source": decision["attention_source"],
                "conditioning": {
                    "state": "immutable_pre_move_root",
                    "attention_conditioned_on_selected_move": False,
                    "selected_move_usage": "raw_policy_prior_index_only",
                },
                "network": evidence["network"],
                "graph_evidence": evidence["graph_evidence"],
                "limitations": evidence["limitations"],
            }
            evidence_destination = output_move_dir / "knowledge_evidence.json"
            evidence_sha256 = _write_json(evidence_destination, evidence_document)
            evidence_path = str(evidence_destination.relative_to(output_dir)).replace("\\", "/")
            payload = {
                "record": record,
                "rgat_edges": rgat_edges,
                "structural_edges": structural_edges(state.size),
                "metrics": metrics,
                "graph_gate": gate,
                "attention_top_k": ATTENTION_TOP_K,
                "artifact_version": ARTIFACT_VERSION,
                "decision": decision,
            }
            svg = render_knowledge_svg(payload)
            render_status = "contrast_rendered"
        else:
            reason = record["solver"].get("unknown_reason") or record["solver"].get("coverage_note") or "no replayed tactical/VCF proof"
            svg = render_knowledge_notice_svg(record, reason, decision=decision, artifact_version=ARTIFACT_VERSION)
            render_status = "notice_no_proof"
        knowledge_svg_path = output_move_dir / "knowledge.svg"
        knowledge_svg_path.write_text(svg, encoding="utf-8")
        selected = decision["selected_move"]
        actor = decision["actor"]
        attention_source = decision["attention_source"]
        manifest["moves"].append({
            "move_dir": str(relative_move_dir).replace("\\", "/"),
            "state_id": record["state_id"],
            "selected_action": selected["action"],
            "selected_row": selected["row"],
            "selected_col": selected["col"],
            "actor_model": actor.get("type"),
            "actor_checkpoint": actor.get("checkpoint"),
            "attention_model": attention_source["model"]["type"],
            "attention_checkpoint": attention_source["model"]["checkpoint"],
            "attention_relationship_to_actor": attention_source["relationship_to_actor"],
            "game_path": str((relative_move_dir.parent / "game.json")).replace("\\", "/"),
            "explanation_path": str((relative_move_dir / "explanation.json")).replace("\\", "/"),
            "solver_record_path": str((relative_move_dir / "knowledge.json")).replace("\\", "/"),
            "knowledge_svg_path": str((relative_move_dir / "knowledge.svg")).replace("\\", "/"),
            "attention_evidence_path": evidence_path,
            "attention_evidence_sha256": evidence_sha256,
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
    _write_json(output_dir / "knowledge_manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arena", type=Path, default=Path("results/h3_pilot/arena/rgcn_vs_rgat_data"))
    parser.add_argument("--rgat-checkpoint", type=Path, default=Path("results/h3_pilot_v2/rgat/seed_7/model.pt"))
    parser.add_argument("--output", type=Path, default=None, help="write versioned knowledge artifacts outside the source arena")
    parser.add_argument("--node-cap", type=int, default=1_000_000)
    parser.add_argument("--time-cap-ms", type=int, default=2_000)
    args = parser.parse_args()
    manifest = render_arena_knowledge(
        args.arena,
        args.rgat_checkpoint,
        GroundTruthBudget(node_cap=args.node_cap, time_cap_ms=args.time_cap_ms),
        output_dir=args.output,
    )
    print(json.dumps({"counts": manifest["counts"], "solver_status_counts": manifest["solver_status_counts"], "output": manifest["output"]}, indent=2))


if __name__ == "__main__":
    main()
