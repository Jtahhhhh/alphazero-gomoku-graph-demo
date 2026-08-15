"""Render E-3b.5 proof-flat knowledge.svg artifacts without rerunning evaluation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from azgomoku.explanation.model_evidence import collect_model_evidence
from azgomoku.explanation.rendering import render_knowledge_svg
from azgomoku.h1_schema import state_from_record
from azgomoku.h3_checkpoint import model_from_bundle
from investigation.e3b_common import load_gold_fail_closed
from investigation.e3b_graph import structural_edges
from models.rgat import RGAT


def _read_metrics(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {
        row["state_id"]: row
        for row in rows
        if row.get("model_type") == "rgat"
    }


def render_knowledge_set(benchmark, graph_gate_path, metrics_path, checkpoint, output_dir):
    graph_gate = json.loads(Path(graph_gate_path).read_text(encoding="utf-8"))
    if not graph_gate.get("passed"):
        raise RuntimeError("knowledge.svg generation blocked: D4 graph gate is red")
    records = load_gold_fail_closed(Path(benchmark))
    proof_count = sum(len(record.get("valid_proofs", [])) for record in records)
    expected_roundtrips = proof_count * 8
    if int(graph_gate.get("d4_proof_roundtrips", -1)) != expected_roundtrips:
        raise RuntimeError("knowledge.svg generation blocked: stale D4 graph gate")

    metrics = _read_metrics(metrics_path)
    model, bundle = model_from_bundle(Path(checkpoint), {"rgat": RGAT})
    if bundle["model_type"] != "rgat" or bundle["training_state"]["iteration"] != 60:
        raise ValueError("knowledge.svg requires the frozen R-GAT iteration-60 checkpoint")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "artifact_type": "e3b5_proof_flat_knowledge_diagrams",
        "graph_gate": {
            "passed": True,
            "d4_proof_roundtrips": expected_roundtrips,
        },
        "records": [],
    }
    for record in records:
        state_id = record["state_id"]
        proofs = record.get("valid_proofs", [])
        if not proofs:
            manifest["records"].append({
                "state_id": state_id,
                "status": "out_of_scope_no_proof",
                "label": "complete label, no replayed proof; no contrast diagram",
                "knowledge_svg": None,
            })
            continue
        if state_id not in metrics:
            raise KeyError(f"missing frozen R-GAT metrics for {state_id}")
        action = int(proofs[0]["action"])
        rgat_edges = collect_model_evidence(state_from_record(record), model, action)["graph_evidence"]["edges"]
        payload = {
            "record": record,
            "rgat_edges": rgat_edges,
            "structural_edges": structural_edges(int(record["state"]["board_size"])),
            "metrics": metrics[state_id],
            "graph_gate": graph_gate,
        }
        destination = output_dir / state_id / "knowledge.svg"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(render_knowledge_svg(payload), encoding="utf-8")
        manifest["records"].append({
            "state_id": state_id,
            "status": "rendered_proof_contrast",
            "proof_count": len(proofs),
            "knowledge_svg": str(destination.relative_to(output_dir)).replace("\\", "/"),
        })

    manifest["rendered"] = sum(item["status"] == "rendered_proof_contrast" for item in manifest["records"])
    manifest["out_of_scope_no_proof"] = sum(item["status"] == "out_of_scope_no_proof" for item in manifest["records"])
    if manifest["rendered"] != 83 or manifest["out_of_scope_no_proof"] != 11:
        raise RuntimeError(
            f"unexpected E-3b.5 scope: rendered={manifest['rendered']} no-proof={manifest['out_of_scope_no_proof']}"
        )
    (output_dir / "knowledge_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=Path("diagnostic/h1_benchmark_v1/h1_benchmark_v1.jsonl"))
    parser.add_argument("--graph-gate", type=Path, default=Path("results/h1_integration/e3b/graph_gate.json"))
    parser.add_argument("--metrics", type=Path, default=Path("results/h1_integration/e3b/endpoint_metrics.csv"))
    parser.add_argument("--checkpoint", type=Path, default=Path("results/h3_pilot_v2/rgat/seed_7/checkpoints/iter_060.pt"))
    parser.add_argument("--output", type=Path, default=Path("results/h1_integration/e3b/knowledge"))
    args = parser.parse_args()
    manifest = render_knowledge_set(args.benchmark, args.graph_gate, args.metrics, args.checkpoint, args.output)
    print(json.dumps({"rendered": manifest["rendered"], "out_of_scope_no_proof": manifest["out_of_scope_no_proof"], "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
