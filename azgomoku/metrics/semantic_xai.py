"""Pure provenance-aware Semantic XAI metric transformations."""

from __future__ import annotations

from collections import defaultdict
import hashlib
from typing import Iterable

import numpy as np

from azgomoku.graph import structural_edges

SEMANTIC_TYPES = (
    "WinningThreat",
    "TacticalLineWindow",
    "CompletionCell",
    "ForcedResponse",
    "BlockingMove",
    "ProofSupportedGeometry",
)

BOOTSTRAP_REPLICATES = 1000
BOOTSTRAP_SEED = 20260814

def _edge_order(state_id: str, base: BaseKGIndex) -> list[dict]:
    entities = {
        item["attributes"]["legacy_edge_id"]: item
        for item in (
            base.entities[entity_id]
            for entity_id in base.by_state_type[(state_id, "StructuralEdge")]
        )
    }
    ordered = []
    size = int(round(len(base.by_state_type[(state_id, "Cell")]) ** 0.5))
    for edge in structural_edges(size):
        item = entities[edge["edge_id"]]
        ordered.append(
            {
                "edge_id": edge["edge_id"],
                "entity_id": item["entity_id"],
                "relation": edge["relation"],
                "source": int(edge["source"]["action"]),
                "target": int(edge["target"]["action"]),
            }
        )
    return ordered


def _window_edges(edges: list[dict], relation: str, cells: Iterable[int]) -> set[str]:
    cells = set(map(int, cells))
    return {
        edge["edge_id"]
        for edge in edges
        if edge["relation"] == relation and edge["source"] in cells and edge["target"] in cells
    }


def proof_semantic_targets(base: BaseKGIndex, state_id: str, proof_id: str) -> dict:
    proof = base.entities[proof_id]
    attrs = proof["attributes"]
    edges = _edge_order(state_id, base)
    proof_windows = base.objects(proof_id, "HAS_WINDOW")
    proof_geometry: set[str] = set()
    for line_id in proof_windows:
        line = base.entities[line_id]["attributes"]
        proof_geometry |= _window_edges(edges, line["relation"], line["cells"])

    supported_moves = base.objects(proof_id, "SUPPORTS")
    selected_threats: set[str] = set()
    for move_id in supported_moves:
        selected_threats.update(base.objects(move_id, "CREATES"))
        selected_threats.update(base.objects(move_id, "BLOCKS"))
    proof_window_keys = {
        (base.entities[line_id]["attributes"]["relation"], tuple(base.entities[line_id]["attributes"]["cells"]))
        for line_id in proof_windows
    }
    for threat_id in base.by_state_type[(state_id, "WinningThreat")]:
        threat = base.entities[threat_id]["attributes"]
        key = (threat["relation"], tuple(threat["window"]))
        reverse_key = (threat["relation"], tuple(reversed(threat["window"])))
        if key in proof_window_keys or reverse_key in proof_window_keys:
            selected_threats.add(threat_id)

    threat_edges: set[str] = set()
    completion_cells: set[int] = set()
    forced_cells: set[int] = set()
    blocking_cells: set[int] = set()
    lineage = set(base.lineage_ids(proof_id, ("SUPPORTS", "HAS_WINDOW", "USES_CELL", "REQUIRES")))
    for threat_id in selected_threats:
        threat = base.entities[threat_id]["attributes"]
        threat_edges |= _window_edges(edges, threat["relation"], threat["window"])
        for cell_id in base.objects(threat_id, "HAS_COMPLETION"):
            completion_cells.add(int(base.entities[cell_id]["attributes"]["action"]))
        for response_id in base.objects(threat_id, "FORCES"):
            forced_cells.add(int(base.entities[response_id]["attributes"]["action"]))
        for move_id in base.subjects(threat_id, "BLOCKS"):
            blocking_cells.add(int(base.entities[move_id]["attributes"]["action"]))
        lineage.update(base.lineage_ids(threat_id, ("USES_CELL", "HAS_COMPLETION", "HAS_DIRECTION", "FORCES")))
    for response_id in base.objects(proof_id, "REQUIRES"):
        forced_cells.add(int(base.entities[response_id]["attributes"]["action"]))
    for move_id in supported_moves:
        lineage.update(base.lineage_ids(move_id, ("CREATES", "BLOCKS")))

    targets = {
        "WinningThreat": threat_edges,
        "TacticalLineWindow": proof_geometry,
        "CompletionCell": completion_cells,
        "ForcedResponse": forced_cells,
        "BlockingMove": blocking_cells,
        "ProofSupportedGeometry": proof_geometry,
    }
    return {
        "targets": targets,
        "lineage_fact_ids": sorted(lineage),
        "proof_action": int(attrs["action"]),
        "proof_concepts": list(attrs.get("concepts", [])),
    }


def _average_precision(labels: list[bool], scores: list[float], tie_ids: list[str]) -> float:
    positives = sum(labels)
    if not positives:
        return 0.0
    order = sorted(range(len(scores)), key=lambda index: (-scores[index], tie_ids[index]))
    hits = 0
    total = 0.0
    for rank, index in enumerate(order, 1):
        if labels[index]:
            hits += 1
            total += hits / rank
    return total / positives


def _alignment(ids: list[str], scores: list[float], target: set[str]) -> dict:
    labels = [item in target for item in ids]
    total = sum(max(0.0, value) for value in scores)
    mass = sum(max(0.0, value) for value, label in zip(scores, labels) if label) / (total or 1.0)
    k = max(1, len(target))
    order = sorted(range(len(scores)), key=lambda index: (-scores[index], ids[index]))
    hits = sum(labels[index] for index in order[:k])
    ranks = {index: rank for rank, index in enumerate(order, 1)}
    target_ranks = [ranks[index] for index, label in enumerate(labels) if label]
    percentile = (
        float(np.mean([(len(ids) - rank) / max(1, len(ids) - 1) for rank in target_ranks]))
        if target_ranks
        else None
    )
    return {
        "mass": float(mass),
        "top_k_recall": hits / (len(target) or 1),
        "auprc": _average_precision(labels, scores, ids),
        "rank_percentile": percentile,
        "target_count": len(target),
    }


def _structural_scores(edges: list[dict]) -> list[float]:
    indegree: dict[tuple[str, int], int] = defaultdict(int)
    for edge in edges:
        indegree[(edge["relation"], edge["target"])] += 1
    return [1.0 / indegree[(edge["relation"], edge["target"])] for edge in edges]


def _incoming_node_scores(edges: list[dict], scores: list[float], size: int) -> tuple[list[str], list[float]]:
    values = [0.0] * (size * size)
    for edge, score in zip(edges, scores):
        values[edge["target"]] += float(score)
    return [f"{action:04d}" for action in range(size * size)], values


def _matched_random_mass(
    ids: list[str], scores: list[float], target: set[str], seed_key: str, draws: int = 32
) -> float:
    if not target:
        return 0.0
    available = np.asarray(ids, dtype=object)
    count = min(len(target), len(ids))
    seed = int(hashlib.sha256(seed_key.encode()).hexdigest()[:16], 16)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(draws):
        sampled = set(rng.choice(available, size=count, replace=False).tolist())
        values.append(_alignment(ids, scores, sampled)["mass"])
    return float(np.mean(values))



def aggregate_multi_proof(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["state_id"], row["phase"], row["iteration"], row["semantic_type"])].append(row)
    result = []
    for (state_id, phase, iteration, semantic_type), items in sorted(grouped.items()):
        result.append(
            {
                "state_id": state_id,
                "phase": phase,
                "iteration": iteration,
                "semantic_type": semantic_type,
                "applicable_proof_count": len(items),
                "existential_max_alignment": max(float(item["attention_mass"]) for item in items),
                "coverage_mean_alignment": float(np.mean([float(item["attention_mass"]) for item in items])),
                "coverage_mean_excess_structural": float(
                    np.mean([float(item["excess_over_structural"]) for item in items])
                ),
                "coverage_mean_excess_random": float(
                    np.mean([float(item["excess_over_random"]) for item in items])
                ),
                "coverage_mean_auprc": float(np.mean([float(item["auprc"]) for item in items])),
                "coverage_mean_rank_percentile": float(
                    np.mean([float(item["rank_percentile"]) for item in items])
                ),
                "coverage_mean_top_k_recall": float(
                    np.mean([float(item["top_k_recall"]) for item in items])
                ),
                "proof_ids": "|".join(sorted(item["proof_id"] for item in items)),
                "checkpoint_sha256": items[0]["checkpoint_sha256"],
                "base_kg_manifest_sha256": items[0]["base_kg_manifest_sha256"],
                "evidence_manifest_sha256": items[0]["evidence_manifest_sha256"],
            }
        )
    return result



def _bootstrap_ci(rows: list[dict], metric: str, seed_offset: int = 0) -> dict:
    by_state: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row.get(metric) is not None:
            by_state[row["state_id"]].append(float(row[metric]))
    state_values = {
        state_id: float(np.mean(values)) for state_id, values in by_state.items() if values
    }
    states = sorted(state_values)
    if not states:
        return {"mean": None, "ci95": [None, None], "n_states": 0, "replicates": 0}
    values = np.asarray([state_values[state_id] for state_id in states], dtype=float)
    rng = np.random.default_rng(BOOTSTRAP_SEED + seed_offset)
    samples = [float(np.mean(rng.choice(values, size=len(values), replace=True))) for _ in range(BOOTSTRAP_REPLICATES)]
    return {
        "mean": float(np.mean(values)),
        "ci95": [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))],
        "n_states": len(states),
        "replicates": BOOTSTRAP_REPLICATES,
        "unit": "state",
        "seed": BOOTSTRAP_SEED + seed_offset,
    }


def bootstrap_results(
    endpoint_state_rows: list[dict], contrast_rows: list[dict], search_rows: list[dict]
) -> dict:
    result = {
        "method": "state-level nonparametric bootstrap; no game-level independence assumed",
        "endpoint_semantic": {},
        "certified_vs_structural": {},
        "semantic_search_lift": {},
    }
    for index, semantic_type in enumerate(SEMANTIC_TYPES):
        selected = [row for row in endpoint_state_rows if row["semantic_type"] == semantic_type]
        result["endpoint_semantic"][semantic_type] = {
            metric: _bootstrap_ci(selected, metric, index * 10 + offset)
            for offset, metric in enumerate(
                (
                    "existential_max_alignment",
                    "coverage_mean_alignment",
                    "coverage_mean_excess_structural",
                    "coverage_mean_excess_random",
                )
            )
        }
    endpoint_contrast = [row for row in contrast_rows if int(row["iteration"]) == 60]
    result["certified_vs_structural"] = {
        metric: _bootstrap_ci(endpoint_contrast, metric, 100 + index)
        for index, metric in enumerate(("A_minus_B", "A_minus_C"))
    }
    endpoint_search = [row for row in search_rows if int(row["iteration"]) == 60]
    for model_index, model_type in enumerate(("rgat", "rgcn")):
        result["semantic_search_lift"][model_type] = {}
        model_rows = [row for row in endpoint_search if row["model_type"] == model_type]
        for index, category in enumerate(
            sorted({row["semantic_move_category"] for row in model_rows})
        ):
            selected = [
                row for row in model_rows if row["semantic_move_category"] == category
            ]
            result["semantic_search_lift"][model_type][category] = _bootstrap_ci(
                selected, "semantic_search_lift", 200 + model_index * 20 + index
            )
    return result


def _summary_by_type(endpoint_rows: list[dict]) -> list[dict]:
    result = []
    for semantic_type in SEMANTIC_TYPES:
        rows = [row for row in endpoint_rows if row["semantic_type"] == semantic_type]
        if not rows:
            continue
        result.append(
            {
                "semantic_type": semantic_type,
                "n_states": len({row["state_id"] for row in rows}),
                "n_applicable_proofs": sum(int(row["applicable_proof_count"]) for row in rows),
                "mean_existential_alignment": float(
                    np.mean([float(row["existential_max_alignment"]) for row in rows])
                ),
                "mean_coverage_alignment": float(
                    np.mean([float(row["coverage_mean_alignment"]) for row in rows])
                ),
                "mean_excess_structural": float(
                    np.mean([float(row["coverage_mean_excess_structural"]) for row in rows])
                ),
                "mean_excess_random": float(
                    np.mean([float(row["coverage_mean_excess_random"]) for row in rows])
                ),
                "mean_auprc": float(np.mean([float(row["coverage_mean_auprc"]) for row in rows])),
                "mean_rank_percentile": float(
                    np.mean([float(row["coverage_mean_rank_percentile"]) for row in rows])
                ),
                "mean_top_k_recall": float(
                    np.mean([float(row["coverage_mean_top_k_recall"]) for row in rows])
                ),
            }
        )
    return result



