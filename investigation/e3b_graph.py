"""Coordinate gates and comparison SVGs for Phase E-3b graph evidence."""

from __future__ import annotations

from pathlib import Path

from azgomoku.explanation.model_evidence import collect_model_evidence
from azgomoku.graph import structural_edges
from azgomoku.h1_schema import state_from_record
from azgomoku.mcts import search
from azgomoku.symmetry import (
    d4_roundtrip_self_check,
    inverse_symmetry,
    transform_action,
    transform_flat_proof,
)


COLORS = {
    "horizontal": "#2563eb",
    "vertical": "#16a34a",
    "diagonal_down": "#dc2626",
    "diagonal_up": "#9333ea",
}


def coordinate_gate(records: list[dict]) -> dict:
    if not d4_roundtrip_self_check():
        raise RuntimeError("global D4 round-trip self-check failed")
    proofs = 0
    action_checks = 0
    for record in records:
        state = state_from_record(record)
        graph_nodes = set(range(state.size * state.size))
        legal = set(map(int, state.legal_actions()))
        optimal = set(map(int, record["solver"]["optimal_actions"]))
        for proof in record.get("valid_proofs", []):
            proofs += 1
            action = int(proof["action"])
            if action not in legal or action not in optimal or action not in graph_nodes:
                raise RuntimeError(f"proof action join failed: {record['state_id']}:{action}")
            row, col = divmod(action, state.size)
            if row * state.size + col != action:
                raise RuntimeError("policy/graph coordinate identity failed")
            for cell in map(int, proof.get("critical_cells", [])):
                if cell not in graph_nodes:
                    raise RuntimeError(f"critical cell outside graph: {cell}")
            for symmetry in range(8):
                inverse = inverse_symmetry(symmetry, state.size)
                mapped = transform_flat_proof(proof, state.size, symmetry)
                restored = transform_flat_proof(mapped, state.size, inverse)
                if restored != proof:
                    raise RuntimeError(
                        f"proof D4 round-trip failed: {record['state_id']} symmetry={symmetry}"
                    )
                mapped_action = transform_action(action, state.size, symmetry)
                if transform_action(mapped_action, state.size, inverse) != action:
                    raise RuntimeError("action D4 round-trip failed")
                action_checks += 1
    return {
        "passed": True,
        "records": len(records),
        "proofs": proofs,
        "d4_proof_roundtrips": proofs * 8,
        "action_alignment_checks": action_checks,
    }


def runtime_mcts_action_gate(record: dict, model) -> dict:
    state = state_from_record(record)
    _, root = search(model, state, playouts=1, temperature=1.0, return_root=True)
    children = set(root.children)
    legal = set(map(int, state.legal_actions()))
    if children != legal:
        raise RuntimeError(f"MCTS root children mismatch for {record['state_id']}")
    proof_actions = {int(proof["action"]) for proof in record.get("valid_proofs", [])}
    if not proof_actions <= children:
        raise RuntimeError(f"proof action missing from MCTS children: {record['state_id']}")
    return {
        "state_id": record["state_id"],
        "legal_actions": len(legal),
        "mcts_children": len(children),
        "proof_actions_checked": len(proof_actions),
    }


def _point(action: int, size: int, ox: float, oy: float, step: float) -> tuple[float, float]:
    row, col = divmod(int(action), size)
    return ox + (col + 0.5) * step, oy + (row + 0.5) * step


def _panel(record, edges, proof, title, ox, oy, step):
    state = record["state"]
    size = int(state["board_size"])
    critical = set(map(int, proof["critical_cells"]))
    proof_action = int(proof["action"])
    parts = [f'<text x="{ox}" y="{oy-28}" class="panel">{title}</text>']
    for row in range(size):
        for col in range(size):
            action = row * size + col
            x, y = ox + col * step, oy + row * step
            fill = "#fee2e2" if action in critical else "#ffffff"
            stroke = "#e11d48" if action == proof_action else "#94a3b8"
            width = 4 if action == proof_action else 1
            parts.append(
                f'<rect x="{x}" y="{y}" width="{step}" height="{step}" '
                f'fill="{fill}" stroke="{stroke}" stroke-width="{width}"/>'
            )
            stone = int(state["board"][row][col])
            if stone:
                stone_fill = "#0f172a" if stone == 1 else "#ffffff"
                parts.append(
                    f'<circle cx="{x+step/2}" cy="{y+step/2}" r="{step*0.28}" '
                    f'fill="{stone_fill}" stroke="#0f172a" stroke-width="2"/>'
                )
            if action in critical:
                parts.append(
                    f'<text x="{x+5}" y="{y+13}" class="cell">C</text>'
                )

    ranked = sorted(edges, key=lambda edge: (-float(edge["attention"]), edge["edge_id"]))[:28]
    max_score = max((float(edge["attention"]) for edge in ranked), default=1.0) or 1.0
    for edge in ranked:
        source = int(edge["source"]["action"])
        target = int(edge["target"]["action"])
        x1, y1 = _point(source, size, ox, oy, step)
        x2, y2 = _point(target, size, ox, oy, step)
        strength = float(edge["attention"]) / max_score
        parts.append(
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
            f'stroke="{COLORS[edge["relation"]]}" stroke-width="{1+5*strength:.2f}" '
            f'opacity="{0.18+0.62*strength:.2f}"/>'
        )

    for window in proof.get("windows", []):
        points = " ".join(
            f"{x},{y}" for x, y in (_point(cell, size, ox, oy, step) for cell in window)
        )
        parts.append(
            f'<polyline points="{points}" fill="none" stroke="#f59e0b" '
            'stroke-width="5" stroke-dasharray="8 5" opacity="0.9"/>'
        )
    return parts


def render_comparison_svg(record: dict, rgat_model, output: Path) -> Path:
    if not record.get("valid_proofs"):
        raise ValueError("comparison SVG requires a replayed proof")
    state = state_from_record(record)
    proof = record["valid_proofs"][0]
    action = int(proof["action"])
    rgat = collect_model_evidence(state, rgat_model, action)["graph_evidence"]["edges"]
    rgcn = structural_edges(state.size)
    step, oy = 62, 105
    panel_width = state.size * step
    width, height = panel_width * 2 + 150, oy + panel_width + 105
    left, right = 45, panel_width + 105
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>text{font-family:Arial;fill:#0f172a}.title{font-size:20px;font-weight:700}.panel{font-size:16px;font-weight:700}.note{font-size:12px}.cell{font-size:11px;font-weight:700;fill:#b91c1c}</style>',
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
        f'<text x="{width/2}" y="30" text-anchor="middle" class="title">Proof vs graph evidence — {record["state_id"]}</text>',
        f'<text x="{width/2}" y="53" text-anchor="middle" class="note">critical cells/windows in red/amber; same immutable pre-move state</text>',
    ]
    parts.extend(_panel(record, rgat, proof, "R-GAT learned attention", left, oy, step))
    parts.extend(_panel(record, rgcn, proof, "R-GCN structural baseline (by design)", right, oy, step))
    parts.append(
        f'<text x="{width/2}" y="{height-28}" text-anchor="middle" class="note">Proof action={action}; relation={", ".join(proof["critical_relations"])}; R-GCN is a reference, not an alignment finding.</text>'
    )
    parts.append("</svg>")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(parts), encoding="utf-8")
    return output
