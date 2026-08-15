import numpy as np
import pytest

from azgomoku.game import GomokuState
from azgomoku.h1_schema import make_record
from azgomoku.tactics import extract_tactical_proofs
from azgomoku.vcf import solve_vcf
from investigation.e3b_common import load_gold_fail_closed, replay_flat_proof
from investigation.e3b_graph import coordinate_gate, structural_edges
from investigation.e3b_pipeline import _collapse_metrics
from azgomoku.explanation.rendering import render_knowledge_svg


def immediate_win_state():
    board = np.zeros((6, 6), dtype=np.int8)
    board[0, :3] = 1
    board[1, :3] = -1
    return GomokuState(board, to_play=1, last_move=8, win_length=4)


def test_tactical_annotation_replays_and_joins_graph_coordinates():
    state = immediate_win_state()
    proof = next(item for item in extract_tactical_proofs(state) if item["action"] == 3)
    proof.update({"proof_method": "tactical_replay", "proof_status": "exact"})
    assert replay_flat_proof(state, proof)
    record = {
        "state_id": "test",
        "state": {
            "board_size": 6,
            "win_length": 4,
            "current_player": 1,
            "last_move": 8,
            "board": state.board.tolist(),
        },
        "solver": {"optimal_actions": [3]},
        "valid_proofs": [proof],
    }
    gate = coordinate_gate([record])
    assert gate["passed"] and gate["proofs"] == 1
    assert gate["d4_proof_roundtrips"] == 8


def test_uniform_attention_is_detected_separately_from_alignment():
    edges = []
    for source in (0, 1):
        edges.append({
            "relation": "horizontal",
            "target": {"action": 2},
            "source": {"action": source},
            "attention": 0.5,
            "head_attention": [0.5, 0.5],
        })
    metrics = _collapse_metrics(edges)
    assert metrics["attention_normalized_entropy"] == 1.0
    assert metrics["attention_structural_mae"] == 0.0
    assert metrics["attention_collapse_flag"] == 1


def test_graph_gate_rejects_noncanonical_proof_order():
    state = immediate_win_state()
    proof = next(item for item in extract_tactical_proofs(state) if item["action"] == 3)
    proof.update({"proof_method": "tactical_replay", "proof_status": "exact"})
    proof["windows"] = [[6, 7, 8, 9], [0, 1, 2, 3]]
    record = {
        "state_id": "bad-order",
        "state": {"board_size": 6, "win_length": 4, "current_player": 1, "last_move": 8, "board": state.board.tolist()},
        "solver": {"optimal_actions": [3]},
        "valid_proofs": [proof],
    }
    with pytest.raises(RuntimeError, match="D4 round-trip failed"):
        coordinate_gate([record])


def test_fail_closed_gold_reader_rejects_valid_partial(tmp_path):
    state = immediate_win_state()
    result = solve_vcf(state, node_cap=10_000, time_cap_ms=1_000)
    assert result.status == "exact_partial"
    record = make_record(
        state,
        history=[],
        result=result,
        seed=7,
        generator_version="test",
        ply=5,
        dedup_mode="d4",
    )
    record["solver"]["perspective"] = {
        "convention_version": 2,
        "value": "player_to_move_at_state",
    }
    path = tmp_path / "partial.jsonl"
    import json
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not exact-complete gold"):
        load_gold_fail_closed(path)


def test_knowledge_svg_renders_every_flat_proof_and_honest_concept_resolution():
    state = immediate_win_state()
    first = dict(next(item for item in extract_tactical_proofs(state) if item["action"] == 3))
    first["concepts"] = ["mandatory_block"]
    second = dict(first)
    second["concepts"] = ["immediate_win", "winning_line"]
    edges = structural_edges(state.size)
    record = {
        "state_id": "all-proofs",
        "state": {
            "board_size": state.size,
            "win_length": state.win_length,
            "current_player": int(state.to_play),
            "last_move": int(state.last_move),
            "board": state.board.tolist(),
        },
        "solver": {"status": "exact_complete", "optimal_actions_complete": True},
        "valid_proofs": [first, second],
    }
    svg = render_knowledge_svg({
        "record": record,
        "rgat_edges": edges,
        "structural_edges": edges,
        "metrics": {
            "attention_collapse_flag": 0,
            "attention_normalized_entropy": .97,
            "attention_head_diversity": .03,
            "attention_topology_correlation": .97,
            "graph_critical_mass": .04,
        },
        "graph_gate": {"passed": True, "d4_proof_roundtrips": 16},
    })
    assert svg.count('data-role="proof-legend"') == 2
    assert 'data-proof-index="1"' in svg and 'data-proof-index="2"' in svg
    assert 'data-concept="mandatory_block"' in svg
    assert 'proof-level: immediate_win + winning_line' in svg
    assert "NO COLLAPSE" in svg and "GOLD · COMPLETE KNOWLEDGE" in svg


def test_knowledge_svg_fails_closed_without_proof_or_green_gate():
    state = immediate_win_state()
    record = {
        "state_id": "blocked",
        "state": {"board_size": 6, "board": state.board.tolist()},
        "solver": {"status": "exact_complete", "optimal_actions_complete": True},
        "valid_proofs": [],
    }
    with pytest.raises(ValueError, match="proof-bearing"):
        render_knowledge_svg({"record": record, "graph_gate": {"passed": True}})
    proof = next(item for item in extract_tactical_proofs(state) if item["action"] == 3)
    record["valid_proofs"] = [proof]
    with pytest.raises(RuntimeError, match="green D4"):
        render_knowledge_svg({"record": record, "graph_gate": {"passed": False}})
