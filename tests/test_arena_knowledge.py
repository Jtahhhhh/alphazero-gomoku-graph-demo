import hashlib
import json

import numpy as np

from azgomoku.explanation.explanation_export import explain_decision
from azgomoku.explanation.rendering import render_knowledge_notice_svg
from azgomoku.game import GomokuState
from azgomoku.ground_truth import GroundTruthBudget
from investigation.arena_knowledge import _decision_context, _load_or_solve, render_arena_knowledge, solve_arena_record
from investigation.e3b_graph import structural_edges
from models.rgcn import RGCN


def test_arena_solver_record_has_replayed_flat_proof_for_immediate_win():
    board = np.zeros((3, 3), dtype=np.int8)
    board[0, :2] = 1
    board[1, :2] = -1
    state = GomokuState(board, to_play=1, last_move=4, win_length=3)
    record = solve_arena_record(state, GroundTruthBudget(node_cap=10_000, time_cap_ms=1_000))
    assert record["solver"]["status"] == "exact_complete"
    assert record["solver"]["optimal_actions_complete"]
    assert record["valid_proofs"]
    assert all(proof["proof_status"] == "exact" for proof in record["valid_proofs"])


def test_notice_svg_does_not_invent_solver_overlay():
    record = {
        "state_id": "unknown-arena",
        "state": {"board_size": 3, "board": [[0, 0, 0], [0, 1, 0], [0, 0, -1]]},
        "solver": {"status": "unknown", "optimal_actions_complete": False},
        "valid_proofs": [],
    }
    svg = render_knowledge_notice_svg(record, "budget")
    assert "Contrast intentionally omitted" in svg
    assert "UNKNOWN · NO GROUND-TRUTH PROOF" in svg
    assert 'data-layer="solver-proof"' not in svg


def test_decision_context_distinguishes_actor_and_counterfactual_attention(tmp_path):
    selected = {"action": 5, "row": 1, "col": 2}
    base_document = {
        "state_id": "state",
        "state": {"board_size": 3},
        "selected_move": selected,
        "model": {"type": "rgat", "checkpoint": str(tmp_path / "rgat.pt")},
    }
    game_move = {
        "state_id": "state",
        "action": 5,
        "actor": dict(base_document["model"]),
    }
    actor_context = _decision_context(base_document, game_move, tmp_path / "rgat.pt")
    assert actor_context["attention_source"]["relationship_to_actor"] == "actor"

    rgcn_document = dict(base_document)
    rgcn_document["model"] = {"type": "rgcn", "checkpoint": str(tmp_path / "rgcn.pt")}
    game_move["actor"] = dict(rgcn_document["model"])
    counterfactual = _decision_context(rgcn_document, game_move, tmp_path / "rgat.pt")
    assert counterfactual["attention_source"]["relationship_to_actor"] == "counterfactual"


def test_solver_cache_is_not_reused_across_different_budgets(tmp_path):
    board = np.zeros((3, 3), dtype=np.int8)
    board[0, :2] = 1
    board[1, :2] = -1
    state = GomokuState(board, to_play=1, last_move=4, win_length=3)
    source_move = tmp_path / "source" / "move_001"
    output_move = tmp_path / "output" / "move_001"
    source_move.mkdir(parents=True)
    first_budget = GroundTruthBudget(node_cap=10_000, time_cap_ms=1_000)
    second_budget = GroundTruthBudget(node_cap=20_000, time_cap_ms=2_000)
    first_record = solve_arena_record(state, first_budget)
    (source_move / "knowledge.json").write_text(json.dumps(first_record), encoding="utf-8")

    second_record, cache_reused = _load_or_solve(source_move, output_move, state, second_budget)
    assert not cache_reused
    assert second_record["solver"]["budget"] == {"node_cap": 20_000, "time_cap_ms": 2_000}


def test_versioned_arena_output_persists_lineage_evidence_and_distinct_markers(tmp_path, monkeypatch):
    arena = tmp_path / "arena"
    move_dir = arena / "game_01" / "move_001"
    move_dir.mkdir(parents=True)
    board = np.zeros((3, 3), dtype=np.int8)
    board[0, :2] = 1
    board[1, :2] = -1
    state = GomokuState(board, to_play=1, last_move=4, win_length=3)
    budget = GroundTruthBudget(node_cap=10_000, time_cap_ms=1_000)
    record = solve_arena_record(state, budget)
    assert record["valid_proofs"][0]["action"] == 2
    (move_dir / "knowledge.json").write_text(json.dumps(record), encoding="utf-8")

    document = explain_decision(state, RGCN(board_size=3, hidden_dim=8), 5, checkpoint="actor-rgcn.pt")
    (move_dir / "explanation.json").write_text(json.dumps(document), encoding="utf-8")
    game = {
        "moves": [{
            "ply": 1,
            "state_id": document["state_id"],
            "action": 5,
            "actor": {"type": "rgcn", "checkpoint": "actor-rgcn.pt"},
            "artifact_dir": "move_001",
        }],
    }
    (arena / "game_01" / "game.json").write_text(json.dumps(game), encoding="utf-8")

    edges = structural_edges(3)
    fake_evidence = {
        "network": {"value": 0.0, "raw_policy_prior": 0.1, "raw_policy_priors": [1 / 9] * 9},
        "graph_evidence": {"attention_available": True, "evidence_kind": "learned_attention", "edges": edges},
        "limitations": ["test evidence"],
    }
    monkeypatch.setattr("investigation.arena_knowledge.load_model", lambda *args, **kwargs: object())
    monkeypatch.setattr("investigation.arena_knowledge.collect_model_evidence", lambda *args, **kwargs: fake_evidence)

    output = tmp_path / "knowledge_v2"
    manifest = render_arena_knowledge(arena, "diagnostic-rgat.pt", budget, output_dir=output)
    item = manifest["moves"][0]
    assert manifest["schema_version"] == 2 and manifest["artifact_version"] == 2
    assert item["selected_action"] == 5 and item["actor_model"] == "rgcn"
    assert item["attention_relationship_to_actor"] == "counterfactual"
    assert not (move_dir / "knowledge.svg").exists()

    evidence_path = output / item["attention_evidence_path"]
    evidence_bytes = evidence_path.read_bytes()
    assert hashlib.sha256(evidence_bytes).hexdigest() == item["attention_evidence_sha256"]
    evidence = json.loads(evidence_bytes)
    assert not evidence["conditioning"]["attention_conditioned_on_selected_move"]
    assert evidence["selected_move"]["action"] == 5

    svg = (output / item["knowledge_svg_path"]).read_text(encoding="utf-8")
    assert 'data-role="proof-action-marker" data-action="2"' in svg
    assert 'data-layer="mcts-selected" data-board="tactic" data-action="5"' in svg
    assert "PROOF #1" in svg and "MCTS SELECTED action=5" in svg
    assert "COUNTERFACTUAL R-GAT ATTENTION" in svg
