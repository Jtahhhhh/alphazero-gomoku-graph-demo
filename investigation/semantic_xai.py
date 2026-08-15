"""Phase 5 provenance-aware semantic interpretability experiments."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from azgomoku.graph import structural_edges
from azgomoku.semantic.evidence_schema import EvidencePredicate
from azgomoku.semantic.export_kg import load_records, sha256_file
from azgomoku.semantic.identity import stable_digest
from investigation.e3b_common import phase_of
from azgomoku.metrics.semantic_alignment import aggregate_proofs
from azgomoku.metrics.semantic_xai import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    SEMANTIC_TYPES,
    _alignment,
    _average_precision,
    _bootstrap_ci,
    _edge_order,
    _incoming_node_scores,
    _matched_random_mass,
    _structural_scores,
    _summary_by_type,
    _window_edges,
    aggregate_multi_proof,
    bootstrap_results,
    proof_semantic_targets,
)


EDGE_TYPES = frozenset(
    {"WinningThreat", "TacticalLineWindow", "ProofSupportedGeometry"}
)
NODE_TYPES = frozenset({"CompletionCell", "ForcedResponse", "BlockingMove"})
NODE_ATTENTION_CONVENTION = "incoming_final_layer_mean_head_attention_mass"


def _read_jsonl(path: Path) -> Iterable[dict]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"refusing to write empty metric table: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


class BaseKGIndex:
    def __init__(self, base_dir: Path) -> None:
        self.base_dir = Path(base_dir)
        self.manifest = json.loads((self.base_dir / "manifest.json").read_text(encoding="utf-8"))
        self.manifest_sha256 = sha256_file(self.base_dir / "manifest.json")
        self.entities = {item["entity_id"]: item for item in _read_jsonl(self.base_dir / "entities.jsonl")}
        self.by_state_type: dict[tuple[str, str], list[str]] = defaultdict(list)
        for entity_id, item in self.entities.items():
            if item.get("state_id"):
                self.by_state_type[(item["state_id"], item["entity_type"])].append(entity_id)
        self.outgoing: dict[tuple[str, str], list[dict]] = defaultdict(list)
        self.incoming: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for fact in _read_jsonl(self.base_dir / "facts.jsonl"):
            self.outgoing[(fact["subject_id"], fact["predicate"])].append(fact)
            if fact.get("object_id"):
                self.incoming[(fact["object_id"], fact["predicate"])].append(fact)

    def objects(self, subject_id: str, predicate: str) -> list[str]:
        return [
            item["object_id"]
            for item in self.outgoing.get((subject_id, predicate), [])
            if item.get("object_id")
        ]

    def subjects(self, object_id: str, predicate: str) -> list[str]:
        return [item["subject_id"] for item in self.incoming.get((object_id, predicate), [])]

    def lineage_ids(self, subject_id: str, predicates: Iterable[str]) -> list[str]:
        return sorted(
            fact["fact_id"]
            for predicate in predicates
            for fact in self.outgoing.get((subject_id, predicate), [])
        )


class EvidenceIndex:
    def __init__(self, evidence_dir: Path, base: BaseKGIndex) -> None:
        self.evidence_dir = Path(evidence_dir)
        self.manifest = json.loads((self.evidence_dir / "manifest.json").read_text(encoding="utf-8"))
        self.manifest_sha256 = sha256_file(self.evidence_dir / "manifest.json")
        if not json.loads((self.evidence_dir / "evidence_release_gate.json").read_text(encoding="utf-8"))["passed"]:
            raise RuntimeError("learned evidence release gate is not passing")
        frozen_manifest = self.manifest["base_semantic_kg"]["files"]["manifest.json"]["sha256"]
        if frozen_manifest != base.manifest_sha256:
            raise RuntimeError("evidence overlay references a different base Semantic KG")

        provenance = {}
        checkpoint_sha: dict[tuple[str, int], str] = {}
        for item in _read_jsonl(self.evidence_dir / "provenance.jsonl"):
            if item["source_kind"] in {"network", "mcts"}:
                provenance[item["provenance_id"]] = item
            checkpoint_sha[(item["model_type"], int(item["checkpoint_iteration"]))] = item[
                "checkpoint_sha256"
            ]
        self.checkpoint_sha = checkpoint_sha

        overlay_entities = {
            item["entity_id"]: item for item in _read_jsonl(self.evidence_dir / "entities.jsonl")
        }
        observes: dict[str, str] = {}
        attention_weights: dict[str, float] = {}
        candidate_values: dict[str, dict[str, Any]] = defaultdict(dict)
        self.policy: dict[tuple[str, int, str], dict[int, float]] = defaultdict(dict)
        self.state_value: dict[tuple[str, int, str], float] = {}
        self.attention: dict[tuple[int, str], dict[str, float]] = defaultdict(dict)
        self.mcts: dict[tuple[str, int, str], dict[int, dict[str, Any]]] = defaultdict(
            lambda: defaultdict(dict)
        )

        for fact in _read_jsonl(self.evidence_dir / "facts.jsonl"):
            predicate = fact["predicate"]
            if predicate == EvidencePredicate.OBSERVES.value:
                observes[fact["subject_id"]] = fact["object_id"]
                continue
            if predicate == EvidencePredicate.HAS_ATTENTION_WEIGHT.value:
                attention_weights[fact["subject_id"]] = float(fact["value"])
                continue
            source = provenance.get(fact["provenance_id"])
            if predicate in {
                EvidencePredicate.HAS_POLICY_PROB.value,
                EvidencePredicate.HAS_STATE_VALUE.value,
            }:
                if source is None:
                    raise RuntimeError(f"network fact lacks indexed provenance: {fact['fact_id']}")
                key = (source["model_type"], int(source["checkpoint_iteration"]), source["state_id"])
                if predicate == EvidencePredicate.HAS_POLICY_PROB.value:
                    action = int(base.entities[fact["subject_id"]]["attributes"]["action"])
                    self.policy[key][action] = float(fact["value"])
                else:
                    self.state_value[key] = float(fact["value"])
                continue
            if predicate in {
                EvidencePredicate.HAS_MCTS_PRIOR.value,
                EvidencePredicate.HAS_VISITS.value,
                EvidencePredicate.HAS_Q.value,
                EvidencePredicate.HAS_SEARCH_PROB.value,
                EvidencePredicate.IS_SELECTED.value,
                EvidencePredicate.REFERS_TO_MOVE.value,
            }:
                candidate_values[fact["subject_id"]][predicate] = (
                    fact["object_id"] if fact.get("object_id") is not None else fact["value"]
                )

        for observation_id, weight in attention_weights.items():
            entity = overlay_entities[observation_id]
            edge_id = observes.get(observation_id)
            if edge_id not in base.entities:
                raise RuntimeError(f"attention observation does not join base edge: {observation_id}")
            attrs = entity["attributes"]
            if base.entities[edge_id]["state_id"] != entity["state_id"]:
                raise RuntimeError(f"attention/base state mismatch: {observation_id}")
            self.attention[(int(attrs["checkpoint_iteration"]), entity["state_id"])][
                base.entities[edge_id]["attributes"]["legacy_edge_id"]
            ] = weight

        for candidate_id, values in candidate_values.items():
            entity = overlay_entities[candidate_id]
            attrs = entity["attributes"]
            key = (
                attrs["model_type"],
                int(attrs["checkpoint_iteration"]),
                entity["state_id"],
            )
            action = int(attrs["action"])
            move_id = values.get(EvidencePredicate.REFERS_TO_MOVE.value)
            if move_id not in base.entities or int(base.entities[move_id]["attributes"]["action"]) != action:
                raise RuntimeError(f"MCTS candidate does not join exact base move: {candidate_id}")
            self.mcts[key][action] = {
                "prior": float(values[EvidencePredicate.HAS_MCTS_PRIOR.value]),
                "visits": int(values[EvidencePredicate.HAS_VISITS.value]),
                "q": float(values[EvidencePredicate.HAS_Q.value]),
                "search_prob": float(values[EvidencePredicate.HAS_SEARCH_PROB.value]),
                "selected": bool(values[EvidencePredicate.IS_SELECTED.value]),
            }


def evaluate_semantic_checkpoint(
    records: list[dict],
    base: BaseKGIndex,
    evidence: EvidenceIndex,
    iteration: int,
) -> tuple[list[dict], list[dict]]:
    proof_rows: list[dict] = []
    contrast_rows: list[dict] = []
    checkpoint_sha = evidence.checkpoint_sha[("rgat", iteration)]
    for record in records:
        state_id = record["state_id"]
        proof_ids = sorted(base.by_state_type[(state_id, "Proof")])
        if not proof_ids:
            continue
        edges = _edge_order(state_id, base)
        attention_map = evidence.attention[(iteration, state_id)]
        edge_ids = [edge["edge_id"] for edge in edges]
        if set(attention_map) != set(edge_ids):
            raise RuntimeError(f"incomplete R-GAT attention for {state_id} at iter {iteration}")
        attention_scores = [float(attention_map[edge_id]) for edge_id in edge_ids]
        structural_scores = _structural_scores(edges)
        size = int(record["state"]["board_size"])
        node_ids, node_attention = _incoming_node_scores(edges, attention_scores, size)
        _, node_structural = _incoming_node_scores(edges, structural_scores, size)
        union_proof_edges: set[str] = set()
        for proof_id in proof_ids:
            semantic = proof_semantic_targets(base, state_id, proof_id)
            union_proof_edges |= semantic["targets"]["ProofSupportedGeometry"]
            lineage_hash = stable_digest(semantic["lineage_fact_ids"], length=64)
            for semantic_type in SEMANTIC_TYPES:
                target = semantic["targets"][semantic_type]
                if not target:
                    continue
                if semantic_type in EDGE_TYPES:
                    ids = edge_ids
                    learned_scores = attention_scores
                    baseline_scores = structural_scores
                    target_ids = set(map(str, target))
                else:
                    ids = node_ids
                    learned_scores = node_attention
                    baseline_scores = node_structural
                    target_ids = {f"{int(item):04d}" for item in target}
                learned = _alignment(ids, learned_scores, target_ids)
                structural = _alignment(ids, baseline_scores, target_ids)
                random_mass = _matched_random_mass(
                    ids,
                    learned_scores,
                    target_ids,
                    f"{state_id}:{proof_id}:{semantic_type}:{iteration}",
                )
                proof_rows.append(
                    {
                        "state_id": state_id,
                        "phase": phase_of(record),
                        "proof_id": proof_id,
                        "proof_action": semantic["proof_action"],
                        "proof_concepts": "|".join(semantic["proof_concepts"]),
                        "semantic_type": semantic_type,
                        "target_domain": "edge" if semantic_type in EDGE_TYPES else "node",
                        "iteration": iteration,
                        "checkpoint_sha256": checkpoint_sha,
                        "attention_mass": learned["mass"],
                        "structural_mass": structural["mass"],
                        "matched_random_mass": random_mass,
                        "excess_over_structural": learned["mass"] - structural["mass"],
                        "excess_over_random": learned["mass"] - random_mass,
                        "auprc": learned["auprc"],
                        "rank_percentile": learned["rank_percentile"],
                        "top_k_recall": learned["top_k_recall"],
                        "target_count": learned["target_count"],
                        "node_attention_convention": NODE_ATTENTION_CONVENTION,
                        "target_lineage_sha256": lineage_hash,
                        "base_kg_manifest_sha256": base.manifest_sha256,
                        "evidence_manifest_sha256": evidence.manifest_sha256,
                    }
                )
        if union_proof_edges:
            target_count = len(union_proof_edges)
            outside = [edge_id for edge_id in edge_ids if edge_id not in union_proof_edges]
            structural_lookup = dict(zip(edge_ids, structural_scores))
            learned_lookup = dict(zip(edge_ids, attention_scores))
            strongest = set(
                sorted(outside, key=lambda item: (-structural_lookup[item], item))[:target_count]
            )
            seed = int(hashlib.sha256(f"contrast:{state_id}:{iteration}".encode()).hexdigest()[:16], 16)
            rng = np.random.default_rng(seed)
            random_region = set(
                rng.choice(np.asarray(outside, dtype=object), size=min(target_count, len(outside)), replace=False).tolist()
            )
            contrast_rows.append(
                {
                    "state_id": state_id,
                    "phase": phase_of(record),
                    "iteration": iteration,
                    "checkpoint_sha256": checkpoint_sha,
                    "certified_supporting_mass_A": _alignment(edge_ids, attention_scores, union_proof_edges)["mass"],
                    "derived_structural_nonproof_mass_B": _alignment(edge_ids, attention_scores, strongest)["mass"],
                    "matched_random_nonproof_mass_C": _alignment(edge_ids, attention_scores, random_region)["mass"],
                    "A_minus_B": _alignment(edge_ids, attention_scores, union_proof_edges)["mass"]
                    - _alignment(edge_ids, attention_scores, strongest)["mass"],
                    "A_minus_C": _alignment(edge_ids, attention_scores, union_proof_edges)["mass"]
                    - _alignment(edge_ids, attention_scores, random_region)["mass"],
                    "region_size": target_count,
                    "base_kg_manifest_sha256": base.manifest_sha256,
                    "evidence_manifest_sha256": evidence.manifest_sha256,
                }
            )
    return proof_rows, contrast_rows


def legacy_reproduction_gate(
    records: list[dict],
    base: BaseKGIndex,
    evidence: EvidenceIndex,
    legacy_progress_path: Path,
) -> dict:
    legacy = json.loads(Path(legacy_progress_path).read_text(encoding="utf-8"))
    comparisons = 0
    maximum_delta = 0.0
    field_deltas: dict[str, float] = defaultdict(float)
    alignment_fields = {
        "mass": "graph_critical_mass",
        "precision_at_k": "graph_precision_at_k",
        "recall_at_k": "graph_recall_at_k",
        "auprc": "graph_auprc",
    }
    for record in records:
        state_id = record["state_id"]
        expected = legacy[f"rgat:{state_id}"]
        key = ("rgat", 60, state_id)
        policy = evidence.policy[key]
        optimal = set(map(int, record["solver"]["optimal_actions"]))
        checks = {
            "policy_optimal_mass": sum(policy[action] for action in optimal),
            "value_prediction": evidence.state_value[key],
        }
        if key in evidence.mcts:
            checks["mcts_optimal_mass"] = sum(
                evidence.mcts[key][action]["search_prob"] for action in optimal
            )
        if record.get("valid_proofs"):
            legacy_edges = structural_edges(int(record["state"]["board_size"]))
            attention = evidence.attention[(60, state_id)]
            scores = [attention[edge["edge_id"]] for edge in legacy_edges]
            result = aggregate_proofs(legacy_edges, scores, record["valid_proofs"])["mean"]
            checks.update({legacy_name: result[name] for name, legacy_name in alignment_fields.items()})
        for field, actual in checks.items():
            delta = abs(float(actual) - float(expected[field]))
            tolerance = 1e-12 if field.startswith("graph_") else 1e-6
            if delta > tolerance:
                raise RuntimeError(
                    f"legacy reproduction failed {state_id}/{field}: delta={delta} tolerance={tolerance}"
                )
            field_deltas[field] = max(field_deltas[field], delta)
            maximum_delta = max(maximum_delta, delta)
            comparisons += 1
    return {
        "passed": True,
        "comparisons": comparisons,
        "states": len(records),
        "proof_bearing_states": sum(bool(record.get("valid_proofs")) for record in records),
        "no_proof_states_excluded_from_alignment": sum(not record.get("valid_proofs") for record in records),
        "maximum_absolute_delta": maximum_delta,
        "maximum_absolute_delta_by_field": dict(sorted(field_deltas.items())),
        "alignment_tolerance": 1e-12,
        "network_search_tolerance": 1e-6,
    }


def semantic_search_lift(
    records: list[dict], base: BaseKGIndex, evidence: EvidenceIndex
) -> list[dict]:
    rows = []
    for model_type in ("rgat", "rgcn"):
        for iteration in (0, 20, 40, 60):
            for record in records:
                state_id = record["state_id"]
                key = (model_type, iteration, state_id)
                policy = evidence.policy[key]
                search = evidence.mcts[key]
                categories: dict[str, set[int]] = {
                    "exact_optimal": set(map(int, record["solver"]["optimal_actions"])),
                    "proof_supported": set(),
                    "threat_creating": set(),
                    "blocking": set(),
                    "forced_response": set(),
                }
                for proof_id in base.by_state_type[(state_id, "Proof")]:
                    for move_id in base.objects(proof_id, "SUPPORTS"):
                        categories["proof_supported"].add(int(base.entities[move_id]["attributes"]["action"]))
                for move_id in base.by_state_type[(state_id, "Move")]:
                    action = int(base.entities[move_id]["attributes"]["action"])
                    if base.objects(move_id, "CREATES"):
                        categories["threat_creating"].add(action)
                    if base.objects(move_id, "BLOCKS"):
                        categories["blocking"].add(action)
                for response_id in base.by_state_type[(state_id, "ForcedResponse")]:
                    categories["forced_response"].add(
                        int(base.entities[response_id]["attributes"]["action"])
                    )
                for category, actions in categories.items():
                    actions &= set(policy) & set(search)
                    if not actions:
                        continue
                    policy_mass = sum(policy[action] for action in actions)
                    search_mass = sum(search[action]["search_prob"] for action in actions)
                    rows.append(
                        {
                            "model_type": model_type,
                            "iteration": iteration,
                            "state_id": state_id,
                            "phase": phase_of(record),
                            "semantic_move_category": category,
                            "move_count": len(actions),
                            "policy_probability_mass": policy_mass,
                            "mcts_search_probability_mass": search_mass,
                            "semantic_search_lift": search_mass - policy_mass,
                            "playouts": 50,
                            "interpretation": "observational_fixed_budget",
                            "checkpoint_sha256": evidence.checkpoint_sha[(model_type, iteration)],
                            "base_kg_manifest_sha256": base.manifest_sha256,
                            "evidence_manifest_sha256": evidence.manifest_sha256,
                        }
                    )
    return rows


def _developmental_analysis(rows: list[dict], legacy_developmental: Path) -> dict:
    legacy = list(csv.DictReader(Path(legacy_developmental).open(encoding="utf-8")))
    competence = {
        int(row["iteration"]): {
            "policy_optimal_mass": float(row["policy_optimal_mass"]),
            "attention_topology_correlation": float(row["attention_topology_correlation"]),
            "attention_collapse_flag": float(row["attention_collapse_flag"]),
        }
        for row in legacy
        if row["model_type"] == "rgat" and row["phase"] == "late"
    }
    trajectories = {}
    for semantic_type in SEMANTIC_TYPES:
        trajectory = []
        for iteration in range(0, 61, 5):
            selected = [
                row
                for row in rows
                if row["semantic_type"] == semantic_type
                and int(row["iteration"]) == iteration
                and row["phase"] == "late"
            ]
            if not selected:
                continue
            trajectory.append(
                {
                    "iteration": iteration,
                    "n_states": len(selected),
                    "coverage_alignment": float(
                        np.mean([float(row["coverage_mean_alignment"]) for row in selected])
                    ),
                    "existential_alignment": float(
                        np.mean([float(row["existential_max_alignment"]) for row in selected])
                    ),
                    "excess_structural": float(
                        np.mean([float(row["coverage_mean_excess_structural"]) for row in selected])
                    ),
                    **competence[iteration],
                }
            )
        trajectories[semantic_type] = trajectory
    return {
        "scope": "R-GAT late-phase main denominator; mid phase retained in CSV as suggestive",
        "node_attention_convention": NODE_ATTENTION_CONVENTION,
        "trajectories": trajectories,
    }


def _bar_svg(title: str, labels: list[str], series: list[tuple[str, list[float]]], output: Path) -> None:
    width, height = 1040, 520
    left, right, top, bottom = 105, 30, 70, 110
    values = [value for _, items in series for value in items]
    low = min(0.0, min(values, default=0.0))
    high = max(0.0, max(values, default=1.0))
    padding = (high - low) * 0.12 or 0.1
    low -= padding
    high += padding
    chart_w = width - left - right
    chart_h = height - top - bottom
    y = lambda value: top + (high - value) / (high - low) * chart_h
    group_w = chart_w / max(1, len(labels))
    bar_w = group_w * 0.75 / max(1, len(series))
    colors = ("#2563eb", "#dc2626", "#16a34a", "#9333ea")
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>text{font-family:Arial;fill:#0f172a}.title{font-size:20px;font-weight:700}.label{font-size:11px}.axis{stroke:#64748b}.grid{stroke:#e2e8f0}</style>',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width/2}" y="32" text-anchor="middle" class="title">{title}</text>',
    ]
    for fraction in (0, .25, .5, .75, 1):
        value = low + fraction * (high - low)
        py = y(value)
        parts.append(f'<line x1="{left}" y1="{py}" x2="{width-right}" y2="{py}" class="grid"/><text x="{left-8}" y="{py+4}" text-anchor="end" class="label">{value:.3f}</text>')
    parts.append(f'<line x1="{left}" y1="{y(0)}" x2="{width-right}" y2="{y(0)}" class="axis"/>')
    for label_index, label in enumerate(labels):
        center = left + (label_index + .5) * group_w
        parts.append(f'<text x="{center}" y="{height-72}" text-anchor="middle" class="label">{label}</text>')
        for series_index, (name, items) in enumerate(series):
            value = items[label_index]
            x = center - len(series) * bar_w / 2 + series_index * bar_w
            y0, y1 = y(0), y(value)
            parts.append(f'<rect x="{x}" y="{min(y0,y1)}" width="{bar_w-3}" height="{abs(y0-y1)}" fill="{colors[series_index % len(colors)]}"/>')
    legend_x = left
    for index, (name, _) in enumerate(series):
        parts.append(f'<rect x="{legend_x}" y="{height-38}" width="16" height="12" fill="{colors[index % len(colors)]}"/><text x="{legend_x+22}" y="{height-28}" class="label">{name}</text>')
        legend_x += 210
    parts.append("</svg>")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(parts), encoding="utf-8")


def render_figures(
    type_summary: list[dict],
    contrasts: list[dict],
    developmental: dict,
    search_rows: list[dict],
    output_dir: Path,
) -> None:
    labels = [row["semantic_type"] for row in type_summary]
    _bar_svg(
        "Endpoint semantic alignment by component",
        labels,
        [
            ("Coverage mean", [row["mean_coverage_alignment"] for row in type_summary]),
            ("Existential max", [row["mean_existential_alignment"] for row in type_summary]),
            ("Excess structural", [row["mean_excess_structural"] for row in type_summary]),
        ],
        output_dir / "semantic_alignment_by_type.svg",
    )
    endpoint_contrasts = [row for row in contrasts if int(row["iteration"]) == 60]
    _bar_svg(
        "Certified-supporting versus non-proof regions",
        ["Endpoint R-GAT"],
        [
            ("Certified A", [float(np.mean([row["certified_supporting_mass_A"] for row in endpoint_contrasts]))]),
            ("Structural B", [float(np.mean([row["derived_structural_nonproof_mass_B"] for row in endpoint_contrasts]))]),
            ("Random C", [float(np.mean([row["matched_random_nonproof_mass_C"] for row in endpoint_contrasts]))]),
        ],
        output_dir / "certified_vs_structural.svg",
    )
    trajectory_labels = [str(item["iteration"]) for item in developmental["trajectories"]["ProofSupportedGeometry"]]
    trajectory_series = []
    for semantic_type in SEMANTIC_TYPES:
        trajectory_series.append(
            (
                semantic_type,
                [item["coverage_alignment"] for item in developmental["trajectories"][semantic_type]],
            )
        )
    _bar_svg(
        "Developmental semantic alignment (R-GAT late states)",
        trajectory_labels,
        trajectory_series[:4],
        output_dir / "semantic_developmental.svg",
    )
    endpoint_search = [row for row in search_rows if int(row["iteration"]) == 60 and row["model_type"] == "rgat"]
    categories = sorted({row["semantic_move_category"] for row in endpoint_search})
    _bar_svg(
        "Semantic search lift at 50 playouts (observational)",
        categories,
        [
            (
                "MCTS - policy",
                [
                    float(np.mean([row["semantic_search_lift"] for row in endpoint_search if row["semantic_move_category"] == category]))
                    for category in categories
                ],
            )
        ],
        output_dir / "semantic_search_lift.svg",
    )


def run_phase5(
    benchmark: Path,
    base_dir: Path,
    evidence_dir: Path,
    output_dir: Path,
    legacy_progress: Path,
    legacy_developmental: Path,
) -> dict:
    records = load_records(benchmark)
    if len(records) != 94:
        raise ValueError("Phase 5 requires the frozen 94-state benchmark")
    base = BaseKGIndex(base_dir)
    evidence = EvidenceIndex(evidence_dir, base)
    reproduction = legacy_reproduction_gate(records, base, evidence, legacy_progress)

    all_proof_rows: list[dict] = []
    all_contrasts: list[dict] = []
    all_state_rows: list[dict] = []
    for iteration in range(0, 61, 5):
        proof_rows, contrasts = evaluate_semantic_checkpoint(records, base, evidence, iteration)
        state_rows = aggregate_multi_proof(proof_rows)
        all_proof_rows.extend(proof_rows)
        all_contrasts.extend(contrasts)
        all_state_rows.extend(state_rows)
        print(
            json.dumps(
                {
                    "stage": "semantic_xai",
                    "iteration": iteration,
                    "proof_rows": len(proof_rows),
                    "state_rows": len(state_rows),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    endpoint_proof_rows = [row for row in all_proof_rows if int(row["iteration"]) == 60]
    endpoint_state_rows = [row for row in all_state_rows if int(row["iteration"]) == 60]
    type_summary = _summary_by_type(endpoint_state_rows)
    search_rows = semantic_search_lift(records, base, evidence)
    bootstrap = bootstrap_results(endpoint_state_rows, all_contrasts, search_rows)
    developmental = _developmental_analysis(all_state_rows, legacy_developmental)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "endpoint_semantic_metrics.csv", endpoint_proof_rows)
    _write_csv(output_dir / "semantic_by_type.csv", type_summary)
    _write_csv(output_dir / "certified_vs_structural.csv", all_contrasts)
    _write_csv(output_dir / "developmental_semantic_metrics.csv", all_state_rows)
    _write_csv(output_dir / "semantic_search_lift.csv", search_rows)
    _write_json(output_dir / "bootstrap_ci.json", bootstrap)
    _write_json(output_dir / "developmental_semantic_analysis.json", developmental)
    endpoint_summary = {
        "legacy_reproduction_gate": reproduction,
        "denominators": {
            "exact_policy_value_states": 94,
            "proof_alignment_states": 83,
            "no_proof_states_excluded_not_zero": 11,
            "proofs": 243,
        },
        "node_attention_convention": NODE_ATTENTION_CONVENTION,
        "multi_proof": {
            "existential": "maximum alignment across applicable valid proofs",
            "coverage": "mean alignment across applicable valid proofs",
        },
        "semantic_by_type": type_summary,
        "base_kg_manifest_sha256": base.manifest_sha256,
        "evidence_manifest_sha256": evidence.manifest_sha256,
        "causal_claims": False,
    }
    _write_json(output_dir / "endpoint_semantic_summary.json", endpoint_summary)
    render_figures(type_summary, all_contrasts, developmental, search_rows, output_dir / "figures")
    result = {
        "passed": True,
        "legacy_reproduction_gate": reproduction,
        "endpoint_proof_metric_rows": len(endpoint_proof_rows),
        "endpoint_state_semantic_rows": len(endpoint_state_rows),
        "developmental_state_semantic_rows": len(all_state_rows),
        "contrast_rows": len(all_contrasts),
        "search_lift_rows": len(search_rows),
        "denominators": endpoint_summary["denominators"],
    }
    _write_json(output_dir / "phase5_release_gate.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark", type=Path, default=Path("diagnostic/h1_benchmark_v1/h1_benchmark_v1.jsonl")
    )
    parser.add_argument("--base-kg", type=Path, default=Path("semantic_kg"))
    parser.add_argument("--evidence", type=Path, default=Path("semantic_evidence_v1"))
    parser.add_argument("--output", type=Path, default=Path("results/semantic_xai"))
    parser.add_argument(
        "--legacy-progress",
        type=Path,
        default=Path("results/h1_integration/e3b/evaluation_progress.json"),
    )
    parser.add_argument(
        "--legacy-developmental",
        type=Path,
        default=Path("results/h1_integration/developmental/developmental_metrics.csv"),
    )
    args = parser.parse_args()
    result = run_phase5(
        args.benchmark,
        args.base_kg,
        args.evidence,
        args.output,
        args.legacy_progress,
        args.legacy_developmental,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

