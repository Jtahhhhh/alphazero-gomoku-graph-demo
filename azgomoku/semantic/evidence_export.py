"""Freeze Semantic KG v1 and export the separate learned-evidence v1 overlay."""

from __future__ import annotations

import argparse
from collections import Counter
import gc
import json
from pathlib import Path
from typing import Iterable

from azgomoku.artifacts import write_json as _write_json
from azgomoku.explanation.model_evidence import collect_model_evidence
from azgomoku.h1_schema import state_from_record
from azgomoku.h3_checkpoint import model_from_bundle
from azgomoku.mcts import search
from azgomoku.semantic.epistemic import EpistemicClass
from azgomoku.semantic.evidence_schema import (
    EVIDENCE_GENERATOR_VERSION,
    EvidenceOverlay,
    EvidencePredicate,
    evidence_predicate_value,
    make_evidence_fact,
    make_evidence_provenance,
    validate_evidence_overlay,
)
from azgomoku.semantic.export_kg import load_records, semantic_d4_gate, sha256_file
from azgomoku.semantic.identity import (
    attention_observation_identity,
    board_state_identity,
    mcts_candidate_identity,
    move_identity,
    structural_edge_identity,
)
from azgomoku.semantic.schema import Entity, EntityType, entity_type_value
from azgomoku.artifacts import sha256_file as checkpoint_sha256
from models.rgat import RGAT
from models.rgcn import RGCN


MODELS = {"rgat": RGAT, "rgcn": RGCN}
ITERATIONS = tuple(range(0, 61, 5))
MCTS_ITERATIONS = frozenset({0, 20, 40, 60})
BASE_FILENAMES = ("entities.jsonl", "facts.jsonl", "provenance.jsonl", "manifest.json")
OVERLAY_FILENAMES = ("entities.jsonl", "facts.jsonl", "provenance.jsonl")


def freeze_base_kg(base_dir: Path) -> dict:
    base_dir = Path(base_dir)
    paths = {name: base_dir / name for name in BASE_FILENAMES}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"base Semantic KG files missing: {missing}")
    manifest = json.loads(paths["manifest.json"].read_text(encoding="utf-8"))
    if not manifest.get("validation_status", {}).get("valid"):
        raise ValueError("base Semantic KG manifest is not valid")
    hashes = {name: sha256_file(path) for name, path in paths.items()}
    freeze = {
        "semantic_kg_version": manifest["semantic_kg_version"],
        "schema_version": manifest["schema_version"],
        "state_count": manifest["state_count"],
        "entity_count": manifest["entity_count"],
        "fact_count": manifest["fact_count"],
        "provenance_count": manifest["provenance_count"],
        "files": {
            name: {"sha256": hashes[name], "bytes": paths[name].stat().st_size}
            for name in BASE_FILENAMES
        },
        "immutable": True,
    }
    freeze_path = base_dir / "base_freeze.json"
    if freeze_path.is_file():
        existing = json.loads(freeze_path.read_text(encoding="utf-8"))
        if existing != freeze:
            raise RuntimeError("immutable Semantic KG base hash/count drift detected")
    else:
        _write_json(freeze_path, freeze)
    return freeze


def assert_base_kg_unchanged(base_dir: Path, freeze: dict) -> None:
    for name in BASE_FILENAMES:
        path = Path(base_dir) / name
        expected = freeze["files"][name]
        if sha256_file(path) != expected["sha256"] or path.stat().st_size != expected["bytes"]:
            raise RuntimeError(f"immutable Semantic KG base changed: {path}")


def run_full_d4_release_gate(records: Iterable[dict], base_dir: Path, freeze: dict) -> dict:
    records = list(records)
    if len(records) != 94:
        raise ValueError(f"full D4 release gate requires 94 states, got {len(records)}")
    gate = semantic_d4_gate(records, verify_roundtrip=True, progress=True)
    gate.update(
        {
            "release_gate": True,
            "expected_record_transform_checks": 94 * 8,
            "base_file_sha256": {
                name: item["sha256"] for name, item in freeze["files"].items()
            },
            "base_kg_unchanged": True,
        }
    )
    if gate["record_transform_checks"] != 94 * 8 or gate["roundtrip_checks"] != 94 * 8:
        raise RuntimeError("full D4 release gate did not cover 94 x 8 transforms")
    assert_base_kg_unchanged(base_dir, freeze)
    _write_json(Path(base_dir) / "d4_release_gate.json", gate)
    return gate


def load_base_entities(base_dir: Path) -> dict[str, dict]:
    result = {}
    path = Path(base_dir) / "entities.jsonl"
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        result[item["entity_id"]] = item
    return result


def load_checkpoint_index(run_dir: Path, expected_model: str) -> dict[int, Path]:
    run_dir = Path(run_dir)
    manifest_path = run_dir / "checkpoints" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("model_type") != expected_model or int(manifest.get("seed")) != 7:
        raise ValueError(f"checkpoint manifest identity mismatch: {manifest_path}")
    index = {
        int(item["iteration"]): run_dir / "checkpoints" / item["path"]
        for item in manifest["checkpoints"]
    }
    if tuple(sorted(index)) != ITERATIONS:
        raise ValueError(f"expected checkpoint schedule {ITERATIONS}, got {tuple(sorted(index))}")
    if any(not path.is_file() for path in index.values()):
        raise FileNotFoundError("checkpoint manifest references a missing checkpoint")
    return index


def _common_provenance(
    *,
    state,
    model_type: str,
    checkpoint: Path,
    checkpoint_hash: str,
    iteration: int,
    seed: int,
    base_manifest_hash: str,
) -> dict:
    return {
        "state_id": board_state_identity(state).entity_id.removeprefix("state:"),
        "source_file": "azgomoku/explanation/model_evidence.py",
        "status": "observed",
        "evidence_generator_version": EVIDENCE_GENERATOR_VERSION,
        "base_kg_manifest_sha256": base_manifest_hash,
        "model_type": model_type,
        "network_mode": "eval",
        "checkpoint_path": str(checkpoint).replace("\\", "/"),
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_iteration": int(iteration),
        "training_seed": int(seed),
        "board_size": int(state.size),
        "win_length": int(state.win_length),
    }


def extract_checkpoint_state_overlay(
    record: dict,
    model,
    bundle: dict,
    checkpoint: Path,
    checkpoint_hash: str,
    base_manifest_hash: str,
    *,
    include_mcts: bool,
) -> EvidenceOverlay:
    state = state_from_record(record)
    model_type = str(bundle["model_type"])
    iteration = int(bundle["training_state"]["iteration"])
    seed = int(bundle["seed"])
    common = _common_provenance(
        state=state,
        model_type=model_type,
        checkpoint=checkpoint,
        checkpoint_hash=checkpoint_hash,
        iteration=iteration,
        seed=seed,
        base_manifest_hash=base_manifest_hash,
    )
    legal = sorted(map(int, state.legal_actions()))
    evidence = collect_model_evidence(state, model, legal[0])
    overlay = EvidenceOverlay()
    network_source = overlay.add_provenance(
        make_evidence_provenance(
            **common,
            source_kind="network",
            source_function="collect_model_evidence",
            method="forward_eval",
        )
    )
    board = board_state_identity(state)
    overlay.add_fact(
        make_evidence_fact(
            subject_id=board.entity_id,
            predicate=EvidencePredicate.HAS_STATE_VALUE,
            value=float(evidence["network"]["value"]),
            provenance_id=network_source.provenance_id,
            epistemic_class=EpistemicClass.LEARNED,
        )
    )
    priors = evidence["network"]["raw_policy_priors"]
    for action in legal:
        move = move_identity(state, action)
        overlay.add_fact(
            make_evidence_fact(
                subject_id=move.entity_id,
                predicate=EvidencePredicate.HAS_POLICY_PROB,
                value=float(priors[action]),
                provenance_id=network_source.provenance_id,
                epistemic_class=EpistemicClass.LEARNED,
            )
        )

    for edge in evidence["graph_evidence"]["edges"]:
        if edge.get("attention") is None:
            continue
        relation = str(edge["relation"])
        source_action = int(edge["source"]["action"])
        target_action = int(edge["target"]["action"])
        layer = str(edge.get("layer", "final"))
        identity = attention_observation_identity(
            state,
            checkpoint_hash,
            relation,
            source_action,
            target_action,
            layer,
        )
        structural_edge = structural_edge_identity(
            state, relation, source_action, target_action
        )
        observation = overlay.add_entity(
            Entity(
                identity.entity_id,
                EntityType.AttentionObservation,
                record["state_id"],
                {
                    "model_type": model_type,
                    "checkpoint_iteration": iteration,
                    "checkpoint_sha256": checkpoint_hash,
                    "legacy_edge_id": edge["edge_id"],
                    "relation": relation,
                    "source_action": source_action,
                    "target_action": target_action,
                    "layer": layer,
                    "head_attention": list(map(float, edge.get("head_attention") or [])),
                    "aggregation_method": edge.get("attention_aggregation"),
                },
                identity.canonical_key,
            )
        )
        attention_source = overlay.add_provenance(
            make_evidence_provenance(
                **common,
                source_kind="attention",
                source_function="collect_model_evidence",
                method="final_layer_relation_attention",
                layer=layer,
                head="all",
                aggregation_method=str(edge["attention_aggregation"]),
                edge_id=str(edge["edge_id"]),
            )
        )
        overlay.add_fact(
            make_evidence_fact(
                subject_id=observation.entity_id,
                predicate=EvidencePredicate.OBSERVES,
                object_id=structural_edge.entity_id,
                provenance_id=attention_source.provenance_id,
                epistemic_class=EpistemicClass.DERIVED,
            )
        )
        overlay.add_fact(
            make_evidence_fact(
                subject_id=observation.entity_id,
                predicate=EvidencePredicate.HAS_ATTENTION_WEIGHT,
                value=float(edge["attention"]),
                provenance_id=attention_source.provenance_id,
                epistemic_class=EpistemicClass.LEARNED,
            )
        )

    if include_mcts:
        config = bundle["config"]
        playouts = int(config["mcts_playouts"])
        temperature = 1.0
        c_puct = float(config["c_puct"])
        search_config = {
            "checkpoint_sha256": checkpoint_hash,
            "model_type": model_type,
            "iteration": iteration,
            "playouts": playouts,
            "search_seed": seed,
            "temperature": temperature,
            "selection_mode": "deterministic_puct_argmax_no_root_noise",
            "root_convention_version": 2,
            "c_puct": c_puct,
        }
        pi, root = search(
            model,
            state,
            playouts=playouts,
            c_puct=c_puct,
            temperature=temperature,
            return_root=True,
        )
        selected = int(pi.argmax())
        mcts_source = overlay.add_provenance(
            make_evidence_provenance(
                **{
                    **common,
                    "source_kind": "mcts",
                    "source_file": "azgomoku/mcts.py",
                    "source_function": "search",
                    "method": "root_mcts",
                    "playouts": playouts,
                    "search_seed": seed,
                    "temperature": temperature,
                    "selection_mode": search_config["selection_mode"],
                    "root_convention_version": 2,
                    "c_puct": c_puct,
                }
            )
        )
        for action, child in sorted(root.children.items()):
            identity = mcts_candidate_identity(state, search_config, action)
            candidate = overlay.add_entity(
                Entity(
                    identity.entity_id,
                    EntityType.MCTSCandidate,
                    record["state_id"],
                    {
                        "model_type": model_type,
                        "checkpoint_iteration": iteration,
                        "checkpoint_sha256": checkpoint_hash,
                        "action": int(action),
                        "search_config": search_config,
                    },
                    identity.canonical_key,
                )
            )
            values = (
                (EvidencePredicate.HAS_MCTS_PRIOR, float(child.prior)),
                (EvidencePredicate.HAS_VISITS, int(child.n)),
                (EvidencePredicate.HAS_Q, float(child.q)),
                (EvidencePredicate.HAS_SEARCH_PROB, float(pi[action])),
                (EvidencePredicate.IS_SELECTED, bool(action == selected)),
            )
            overlay.add_fact(
                make_evidence_fact(
                    subject_id=candidate.entity_id,
                    predicate=EvidencePredicate.REFERS_TO_MOVE,
                    object_id=move_identity(state, action).entity_id,
                    provenance_id=mcts_source.provenance_id,
                    epistemic_class=EpistemicClass.DERIVED,
                )
            )
            for predicate, value in values:
                overlay.add_fact(
                    make_evidence_fact(
                        subject_id=candidate.entity_id,
                        predicate=predicate,
                        value=value,
                        provenance_id=mcts_source.provenance_id,
                        epistemic_class=EpistemicClass.LEARNED,
                    )
                )
    return overlay


class OverlayWriter:
    def __init__(self, output_dir: Path, base_entities: dict[str, dict]) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=False)
        self.base_entities = base_entities
        self.handles = {
            name: (self.output_dir / name).open("w", encoding="utf-8", newline="\n")
            for name in OVERLAY_FILENAMES
        }
        self.entity_ids: set[str] = set()
        self.fact_ids: set[str] = set()
        self.provenance_ids: set[str] = set()
        self.entity_type_counts: Counter[str] = Counter()
        self.predicate_counts: Counter[str] = Counter()
        self.epistemic_counts: Counter[str] = Counter()
        self.model_iteration_counts: Counter[str] = Counter()
        self.external_reference_count = 0

    @staticmethod
    def _line(item: dict) -> str:
        return json.dumps(item, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"

    def add(self, overlay: EvidenceOverlay) -> None:
        report = validate_evidence_overlay(overlay, self.base_entities, raise_on_error=True)
        self.external_reference_count += report.external_reference_count
        for item in sorted(overlay.entities.values(), key=lambda value: value.entity_id):
            if item.entity_id in self.entity_ids:
                raise ValueError(f"duplicate streamed evidence entity: {item.entity_id}")
            self.entity_ids.add(item.entity_id)
            self.entity_type_counts[entity_type_value(item.entity_type)] += 1
            self.handles["entities.jsonl"].write(self._line(item.dict()))
        for item in sorted(overlay.provenance.values(), key=lambda value: value.provenance_id):
            if item.provenance_id in self.provenance_ids:
                raise ValueError(f"duplicate streamed evidence provenance: {item.provenance_id}")
            self.provenance_ids.add(item.provenance_id)
            self.model_iteration_counts[f"{item.model_type}:{item.checkpoint_iteration}"] += 1
            self.handles["provenance.jsonl"].write(self._line(item.dict()))
        for item in sorted(overlay.facts.values(), key=lambda value: value.fact_id):
            if item.fact_id in self.fact_ids:
                raise ValueError(f"duplicate streamed evidence fact: {item.fact_id}")
            self.fact_ids.add(item.fact_id)
            self.predicate_counts[evidence_predicate_value(item.predicate)] += 1
            epistemic = item.epistemic_class.value if hasattr(item.epistemic_class, "value") else str(item.epistemic_class)
            self.epistemic_counts[epistemic] += 1
            self.handles["facts.jsonl"].write(self._line(item.dict()))

    def close(self) -> None:
        for handle in self.handles.values():
            handle.close()

    def manifest(self, *, base_freeze: dict, checkpoint_index: list[dict], scope: str) -> dict:
        return {
            "schema_version": "1.1",
            "semantic_evidence_version": "semantic_evidence_v1",
            "scope": scope,
            "base_semantic_kg": {
                "semantic_kg_version": base_freeze["semantic_kg_version"],
                "files": base_freeze["files"],
            },
            "state_count": 94,
            "checkpoint_count": len(checkpoint_index),
            "checkpoints": checkpoint_index,
            "entity_count": len(self.entity_ids),
            "fact_count": len(self.fact_ids),
            "provenance_count": len(self.provenance_ids),
            "external_reference_count": self.external_reference_count,
            "entity_type_counts": dict(sorted(self.entity_type_counts.items())),
            "predicate_counts": dict(sorted(self.predicate_counts.items())),
            "epistemic_counts": {
                name: int(self.epistemic_counts.get(name, 0))
                for name in ("EXACT", "CERTIFIED", "DERIVED", "HEURISTIC", "LEARNED")
            },
            "model_iteration_provenance_counts": dict(sorted(self.model_iteration_counts.items())),
        }


def _overlay_hashes(output_dir: Path) -> dict[str, dict]:
    return {
        name: {
            "sha256": sha256_file(Path(output_dir) / name),
            "bytes": (Path(output_dir) / name).stat().st_size,
        }
        for name in OVERLAY_FILENAMES
    }


def verify_evidence_release(output_dir: Path, base_dir: Path, freeze: dict) -> dict:
    output_dir = Path(output_dir)
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    endpoint = json.loads((output_dir / "endpoint" / "manifest.json").read_text(encoding="utf-8"))
    base_manifest = json.loads((Path(base_dir) / "manifest.json").read_text(encoding="utf-8"))
    moves = int(base_manifest["entity_type_counts"]["Move"])
    edges = int(base_manifest["entity_type_counts"]["StructuralEdge"])
    states = int(base_manifest["state_count"])
    expected_predicates = {
        "HAS_POLICY_PROB": moves * 26,
        "HAS_STATE_VALUE": states * 26,
        "OBSERVES": edges * 13,
        "HAS_ATTENTION_WEIGHT": edges * 13,
        "REFERS_TO_MOVE": moves * 8,
        "HAS_MCTS_PRIOR": moves * 8,
        "HAS_VISITS": moves * 8,
        "HAS_Q": moves * 8,
        "HAS_SEARCH_PROB": moves * 8,
        "IS_SELECTED": moves * 8,
    }
    expected_endpoint_predicates = {
        "HAS_POLICY_PROB": moves * 2,
        "HAS_STATE_VALUE": states * 2,
        "OBSERVES": edges,
        "HAS_ATTENTION_WEIGHT": edges,
        "REFERS_TO_MOVE": moves * 2,
        "HAS_MCTS_PRIOR": moves * 2,
        "HAS_VISITS": moves * 2,
        "HAS_Q": moves * 2,
        "HAS_SEARCH_PROB": moves * 2,
        "IS_SELECTED": moves * 2,
    }
    expected_entities = edges * 13 + moves * 8
    expected_provenance = states * 26 + edges * 13 + states * 8
    expected_endpoint_entities = edges + moves * 2
    expected_endpoint_provenance = states * 2 + edges + states * 2
    checks = {
        "predicate_counts": manifest["predicate_counts"] == expected_predicates,
        "endpoint_predicate_counts": endpoint["predicate_counts"] == expected_endpoint_predicates,
        "entity_count": manifest["entity_count"] == expected_entities,
        "provenance_count": manifest["provenance_count"] == expected_provenance,
        "endpoint_entity_count": endpoint["entity_count"] == expected_endpoint_entities,
        "endpoint_provenance_count": endpoint["provenance_count"] == expected_endpoint_provenance,
        "checkpoint_count": manifest["checkpoint_count"] == 26,
        "endpoint_checkpoint_count": endpoint["checkpoint_count"] == 2,
        "heuristic_zero": manifest["epistemic_counts"]["HEURISTIC"] == 0,
        "exact_certified_zero": (
            manifest["epistemic_counts"]["EXACT"] == 0
            and manifest["epistemic_counts"]["CERTIFIED"] == 0
        ),
        "base_kg_unchanged": True,
        "artifact_hashes": manifest["artifact_files"] == _overlay_hashes(output_dir),
        "endpoint_artifact_hashes": endpoint["artifact_files"]
        == _overlay_hashes(output_dir / "endpoint"),
    }
    assert_base_kg_unchanged(base_dir, freeze)
    if not all(checks.values()):
        raise RuntimeError(f"semantic evidence release cardinality/hash gate failed: {checks}")
    gate = {
        "passed": True,
        "checks": checks,
        "base_kg_unchanged": True,
        "all_external_ids_resolved": True,
        "all_observations_have_checkpoint_provenance": True,
        "no_learned_tactical_truth": True,
        "expected_predicate_counts": expected_predicates,
        "expected_entity_count": expected_entities,
        "expected_provenance_count": expected_provenance,
        "network_checkpoint_schedule": list(ITERATIONS),
        "mcts_schedule": sorted(MCTS_ITERATIONS),
    }
    _write_json(output_dir / "evidence_release_gate.json", gate)
    return gate


def export_evidence_overlay(
    records: list[dict],
    base_dir: Path,
    output_dir: Path,
    rgat_run: Path,
    rgcn_run: Path,
    freeze: dict,
) -> dict:
    output_dir = Path(output_dir)
    temporary = output_dir.with_name(output_dir.name + ".tmp")
    if temporary.exists() or output_dir.exists():
        raise FileExistsError(f"immutable evidence output already exists: {output_dir}")
    base_entities = load_base_entities(base_dir)
    base_manifest_hash = freeze["files"]["manifest.json"]["sha256"]
    endpoint_dir = temporary / "endpoint"
    writer = OverlayWriter(temporary, base_entities)
    endpoint_writer = OverlayWriter(endpoint_dir, base_entities)
    indexes = {
        "rgat": load_checkpoint_index(rgat_run, "rgat"),
        "rgcn": load_checkpoint_index(rgcn_run, "rgcn"),
    }
    checkpoint_index: list[dict] = []
    endpoint_checkpoint_index: list[dict] = []
    try:
        for model_type in ("rgat", "rgcn"):
            for iteration in ITERATIONS:
                checkpoint = indexes[model_type][iteration]
                checkpoint_hash = checkpoint_sha256(checkpoint)
                model, bundle = model_from_bundle(checkpoint, MODELS)
                if (
                    bundle["model_type"] != model_type
                    or int(bundle["training_state"]["iteration"]) != iteration
                    or int(bundle["seed"]) != 7
                ):
                    raise RuntimeError(f"checkpoint bundle identity mismatch: {checkpoint}")
                checkpoint_item = {
                    "model_type": model_type,
                    "iteration": iteration,
                    "path": str(checkpoint).replace("\\", "/"),
                    "sha256": checkpoint_hash,
                    "training_seed": int(bundle["seed"]),
                    "network_mode": "eval",
                    "mcts_evaluated": iteration in MCTS_ITERATIONS,
                    "mcts_playouts": int(bundle["config"]["mcts_playouts"]) if iteration in MCTS_ITERATIONS else None,
                }
                checkpoint_index.append(checkpoint_item)
                if iteration == 60:
                    endpoint_checkpoint_index.append(checkpoint_item)
                for state_index, record in enumerate(records, 1):
                    overlay = extract_checkpoint_state_overlay(
                        record,
                        model,
                        bundle,
                        checkpoint,
                        checkpoint_hash,
                        base_manifest_hash,
                        include_mcts=iteration in MCTS_ITERATIONS,
                    )
                    writer.add(overlay)
                    if iteration == 60:
                        endpoint_writer.add(overlay)
                    print(
                        json.dumps(
                            {
                                "stage": "semantic_evidence_export",
                                "model": model_type,
                                "iteration": iteration,
                                "state": state_index,
                                "total": len(records),
                                "mcts": iteration in MCTS_ITERATIONS,
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                del model, bundle
                gc.collect()
    finally:
        writer.close()
        endpoint_writer.close()

    manifest = writer.manifest(
        base_freeze=freeze, checkpoint_index=checkpoint_index, scope="developmental_and_endpoint"
    )
    endpoint_manifest = endpoint_writer.manifest(
        base_freeze=freeze, checkpoint_index=endpoint_checkpoint_index, scope="endpoint_iter60"
    )
    manifest["artifact_files"] = _overlay_hashes(temporary)
    endpoint_manifest["artifact_files"] = _overlay_hashes(endpoint_dir)
    integrity = {
        "passed": True,
        "base_kg_unchanged": True,
        "all_external_ids_resolved": True,
        "all_observations_have_checkpoint_provenance": True,
        "no_learned_tactical_truth": True,
        "heuristic_fact_count": manifest["epistemic_counts"]["HEURISTIC"],
        "mcts_schedule": sorted(MCTS_ITERATIONS),
        "network_checkpoint_schedule": list(ITERATIONS),
        "endpoint_rgat_complete": any(
            item["model_type"] == "rgat" and item["iteration"] == 60
            for item in checkpoint_index
        ),
    }
    if manifest["epistemic_counts"]["HEURISTIC"] != 0:
        raise RuntimeError("evidence overlay emitted HEURISTIC facts")
    _write_json(temporary / "manifest.json", manifest)
    _write_json(endpoint_dir / "manifest.json", endpoint_manifest)
    _write_json(temporary / "evidence_release_gate.json", integrity)
    assert_base_kg_unchanged(base_dir, freeze)
    temporary.rename(output_dir)
    gate = verify_evidence_release(output_dir, base_dir, freeze)
    return {"manifest": manifest, "endpoint_manifest": endpoint_manifest, "gate": gate}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=Path("diagnostic/h1_benchmark_v1/h1_benchmark_v1.jsonl"),
    )
    parser.add_argument("--base-kg", type=Path, default=Path("semantic_kg"))
    parser.add_argument("--output", type=Path, default=Path("semantic_evidence_v1"))
    parser.add_argument(
        "--rgat-run", type=Path, default=Path("results/h3_pilot_v2/rgat/seed_7")
    )
    parser.add_argument(
        "--rgcn-run", type=Path, default=Path("results/h3_pilot_v2/rgcn/seed_7")
    )
    parser.add_argument("--skip-full-d4", action="store_true")
    args = parser.parse_args()

    freeze = freeze_base_kg(args.base_kg)
    records = load_records(args.benchmark)
    if not args.skip_full_d4:
        run_full_d4_release_gate(records, args.base_kg, freeze)
    else:
        gate_path = args.base_kg / "d4_release_gate.json"
        if not gate_path.is_file() or not json.loads(gate_path.read_text(encoding="utf-8")).get("passed"):
            raise RuntimeError("--skip-full-d4 requires an existing passing release gate")
    result = export_evidence_overlay(
        records,
        args.base_kg,
        args.output,
        args.rgat_run,
        args.rgcn_run,
        freeze,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


