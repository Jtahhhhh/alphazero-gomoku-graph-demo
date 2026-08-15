import pytest

from investigation.developmental_evaluate import analyze_tail, baseline_gate


def row(iteration, policy, critical, collapse=0.0, topology=0.97):
    return {
        "model_type": "rgat",
        "phase": "late",
        "iteration": iteration,
        "policy_optimal_mass": policy,
        "attention_collapse_flag": collapse,
        "attention_topology_correlation": topology,
        "graph_critical_mass": critical,
        "structural_critical_mass": 0.033,
        "random_critical_mass": 0.035,
    }


def test_null_is_rejected_only_on_competent_collapse_free_flat_tail():
    result = analyze_tail([
        row(0, 0.10, 0.020, collapse=0.8),
        row(50, 0.49, 0.0320),
        row(55, 0.50, 0.0322),
        row(60, 0.51, 0.0323),
    ])
    assert result["tail_iterations"] == [50, 55, 60]
    assert result["H_null_verdict"] == "rejected"
    assert result["all_tail_alignment_at_or_below_baseline"]


def test_meaningful_tail_alignment_rise_prevents_rejection():
    result = analyze_tail([
        row(50, 0.49, 0.0300),
        row(55, 0.50, 0.0320),
        row(60, 0.51, 0.0340),
    ])
    assert result["meaningful_tail_alignment_rise"]
    assert result["H_null_verdict"] == "not_rejected"


def test_baseline_gate_catches_checkpoint_dependent_control():
    rows = []
    for phase in ("late", "mid"):
        for iteration in (0, 5):
            rows.append({"phase": phase, "structural_critical_mass": 0.03, "random_critical_mass": 0.04})
    assert baseline_gate(rows)["passed"]
    rows[-1]["random_critical_mass"] = 0.041
    with pytest.raises(RuntimeError, match="baseline drift"):
        baseline_gate(rows)

