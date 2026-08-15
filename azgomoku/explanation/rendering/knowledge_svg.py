"""Proof-flat contrast knowledge diagram on immutable board coordinates."""

from __future__ import annotations

from html import escape
import json
import math


ATTENTION_COLOR = "#075985"
PROOF_COLOR = "#f59e0b"


def _point(action, size, ox, oy, step):
    row, col = divmod(int(action), size)
    return ox + (col + 0.5) * step, oy + (row + 0.5) * step


def _critical_edge_ids(proof, edges):
    relations = set(proof.get("critical_relations", []))
    windows = [set(map(int, window)) for window in proof.get("windows", [])]
    return {
        edge["edge_id"]
        for edge in edges
        if edge["relation"] in relations
        and any(
            int(edge["source"]["action"]) in window
            and int(edge["target"]["action"]) in window
            for window in windows
        )
    }


def _metric(metrics, key, default=0.0):
    value = metrics.get(key, default)
    return default if value in (None, "") else float(value)


def _decision_fields(payload, size):
    decision = payload.get("decision") or {}
    selected = decision.get("selected_move")
    if selected is None:
        return decision, None
    action = int(selected["action"])
    if action < 0 or action >= size * size:
        raise ValueError("knowledge decision selected_move is outside the board")
    row, col = divmod(action, size)
    if int(selected.get("row", row)) != row or int(selected.get("col", col)) != col:
        raise ValueError("knowledge decision selected_move action/row/col mismatch")
    return decision, action


def _selected_marker(action, size, board_x, board_y, step, board_name):
    row, col = divmod(action, size)
    x, y = board_x + col * step, board_y + row * step
    return (
        f'<g data-layer="mcts-selected" data-board="{board_name}" data-action="{action}" data-role="actual-selected-move">'
        f'<rect x="{x+4}" y="{y+4}" width="{step-8}" height="{step-8}" rx="4" fill="none" stroke="#e11d48" stroke-width="4"/>'
        f'<text x="{x+7}" y="{y+step-8}" class="tiny" fill="#be123c">MCTS</text></g>'
    )


def render_knowledge_svg(payload):
    """Render solver tactic and top-k R-GAT attention on separate boards."""
    record = payload["record"]
    proofs = list(record.get("valid_proofs", []))
    if not proofs:
        raise ValueError("knowledge.svg requires a proof-bearing record")
    gate = payload.get("graph_gate", {})
    if not gate.get("passed"):
        raise RuntimeError("knowledge.svg requires a green D4 graph gate")

    state = record["state"]
    size = int(state["board_size"])
    board = state["board"]
    decision, selected_action = _decision_fields(payload, size)
    actor = decision.get("actor") or {}
    attention_source = decision.get("attention_source") or {}
    attention_relationship = attention_source.get("relationship_to_actor")
    rgat_edges = list(payload["rgat_edges"])
    structural_edges = list(payload["structural_edges"])
    metrics = payload.get("metrics", {})
    if {edge["edge_id"] for edge in structural_edges} != {edge["edge_id"] for edge in rgat_edges}:
        raise ValueError("R-GAT/R-GCN edge identity mismatch")

    requested_top_k = int(payload.get("attention_top_k", 20))
    top_k = max(1, min(requested_top_k, len(rgat_edges)))
    ranked_attention = sorted(rgat_edges, key=lambda edge: (-float(edge["attention"]), edge["edge_id"]))
    selected_attention = ranked_attention[:top_k]
    attention_values = [float(edge["attention"]) for edge in rgat_edges]
    attention_min, attention_max = min(attention_values), max(attention_values)
    attention_span = attention_max - attention_min
    cutoff = float(selected_attention[-1]["attention"])

    critical_cells = set()
    proof_actions = {}
    proof_edge_ids = set()
    for proof_index, proof in enumerate(proofs):
        critical_cells.update(map(int, proof.get("critical_cells", [])))
        proof_actions.setdefault(int(proof["action"]), []).append(proof_index + 1)
        proof_edge_ids.update(_critical_edge_ids(proof, rgat_edges))
    selected_edge_ids = {edge["edge_id"] for edge in selected_attention}
    top_k_overlap = len(selected_edge_ids & proof_edge_ids)

    step, board_y = 68, 142
    board_width = size * step
    left_x, right_x = 58, 712
    width = right_x + board_width + 58
    proof_rows = math.ceil(len(proofs) / 2)
    metrics_y = board_y + board_width + 82
    proof_y = metrics_y + 150
    height = max(790, proof_y + proof_rows * 38 + 72)
    status = record["solver"]["status"]
    complete = bool(record["solver"].get("optimal_actions_complete", False))
    gold_label = "GOLD · COMPLETE KNOWLEDGE" if status == "exact_complete" and complete else "PARTIAL · PARTIAL KNOWLEDGE"
    gold_fill = "#dcfce7" if complete else "#fef3c7"
    gold_stroke = "#15803d" if complete else "#b45309"
    collapse = int(_metric(metrics, "attention_collapse_flag"))
    entropy = _metric(metrics, "attention_normalized_entropy")
    diversity = _metric(metrics, "attention_head_diversity")
    topology = _metric(metrics, "attention_topology_correlation")
    alignment = _metric(metrics, "graph_critical_mass")
    artifact_version = int(payload.get("artifact_version", 1))
    if attention_relationship == "counterfactual":
        attention_panel_title = "COUNTERFACTUAL R-GAT ATTENTION"
        evidence_note = "Right panel: separate R-GAT diagnostic; not actor attention."
    elif attention_relationship == "actor":
        attention_panel_title = "ACTOR R-GAT ATTENTION"
        evidence_note = "Right panel is attention from the acting R-GAT checkpoint."
    else:
        attention_panel_title = "R-GAT ATTENTION"
        evidence_note = "R-GCN is structural by design and has no learned attention coefficients."
    metadata = {
        "artifact_version": artifact_version,
        "attention_top_k": top_k,
        "attention_total_edges": len(rgat_edges),
        "selection": "value_desc_then_edge_id",
        "normalization": "per_state_min_max",
        "selected_action": selected_action,
        "actor_model": actor.get("type"),
        "attention_relationship_to_actor": attention_relationship,
    }
    context_line = None
    if selected_action is not None:
        selected_row, selected_col = divmod(selected_action, size)
        actor_label = str(actor.get("type", "unknown")).upper()
        source_label = "same actor" if attention_relationship == "actor" else "separate diagnostic model"
        context_line = (
            f'Actual actor: {escape(actor_label)} | MCTS SELECTED action={selected_action} '
            f'(row={selected_row}, col={selected_col}) | attention source: {source_label}'
        )

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
        f'data-artifact-version="{artifact_version}" data-attention-top-k="{top_k}" data-attention-total-edges="{len(rgat_edges)}" '
        f'data-attention-selection="value-desc-edge-id" data-selected-action="{selected_action if selected_action is not None else ""}" '
        f'data-attention-relationship="{escape(str(attention_relationship or "unspecified"))}">',
        '<title>Tactic location versus attention location</title>',
        f'<metadata>{escape(json.dumps(metadata, sort_keys=True), quote=False)}</metadata>',
        '<defs><marker id="proof-arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="#b45309"/></marker></defs>',
        '<style>text{font-family:Arial,sans-serif;fill:#0f172a}.title{font-size:24px;font-weight:700}.panel{font-size:16px;font-weight:700}.label{font-size:12px;font-weight:700}.note{font-size:11px}.tiny{font-size:9px}.metric{font-size:14px;font-weight:700}.bigmetric{font-size:27px;font-weight:700;fill:#075985}</style>',
        f'<rect width="{width}" height="{height}" fill="#f8fafc"/>',
        f'<text x="32" y="34" class="title">Tactic ở đâu vs Attention ở đâu — {escape(record["state_id"])}</text>',
        '<text x="32" y="58" class="note">Cùng một thế cờ. Trái: solver proof. Phải: các cạnh R-GAT chú ý mạnh nhất. Không overlay hai lớp.</text>',
        f'<rect x="{width-285}" y="20" width="250" height="32" rx="16" fill="{gold_fill}" stroke="{gold_stroke}" stroke-width="2"/>',
        f'<text x="{width-160}" y="41" text-anchor="middle" class="label">{gold_label}</text>',
        f'<text x="{left_x}" y="108" class="panel" fill="#b45309">SOLVER TACTIC · {len(critical_cells)} critical cells</text>',
        f'<text x="{right_x}" y="104" class="panel" fill="{ATTENTION_COLOR}">{attention_panel_title}</text>',
        f'<text x="{right_x}" y="126" class="note" fill="{ATTENTION_COLOR}">top {top_k}/{len(rgat_edges)} edges</text>',
    ]
    if context_line is not None:
        parts.append(f'<text x="32" y="80" class="note" data-role="decision-lineage">{context_line}</text>')

    # Left board: solver tactic only.
    for row in range(size):
        for col in range(size):
            action = row * size + col
            x, y = left_x + col * step, board_y + row * step
            fill = "#fed7aa" if action in critical_cells else "#ffffff"
            parts.append(f'<g data-board="tactic" data-action="{action}" data-role="{"critical-cell" if action in critical_cells else "cell"}"><rect x="{x}" y="{y}" width="{step}" height="{step}" rx="3" fill="{fill}" stroke="#dbe3ec"/>')
            stone = int(board[row][col])
            if stone:
                stone_fill = "#0f172a" if stone == 1 else "#ffffff"
                parts.append(f'<circle cx="{x+step/2}" cy="{y+step/2}" r="19" fill="{stone_fill}" stroke="#0f172a" stroke-width="2"/>')
            if action in proof_actions:
                labels = ",".join(str(i) for i in proof_actions[action])
                badge_width = min(step - 6, 43 + 5 * max(0, len(labels) - 1))
                parts.append(
                    f'<g data-role="proof-action-marker" data-action="{action}" data-proof-indices="{labels}">'
                    f'<rect x="{x+step-badge_width-3}" y="{y+3}" width="{badge_width}" height="17" rx="8" fill="#b45309"/>'
                    f'<text x="{x+step-7}" y="{y+15}" text-anchor="end" class="tiny" fill="#ffffff">PROOF #{labels}</text></g>'
                )
            parts.append("</g>")

    for proof_index, proof in enumerate(proofs):
        concepts = list(proof.get("concepts", []))
        relations = list(proof.get("critical_relations", []))
        concept_text = concepts[0] if len(concepts) == 1 else " + ".join(concepts)
        for window_index, window in enumerate(proof.get("windows", [])):
            if len(window) < 2:
                continue
            x1, y1 = _point(window[0], size, left_x, board_y, step)
            x2, y2 = _point(window[-1], size, left_x, board_y, step)
            relation = relations[min(window_index, len(relations)-1)] if relations else "relation"
            label_x, label_y = (x1+x2)/2, (y1+y2)/2 - 12 - proof_index*3
            parts.append(
                f'<line data-layer="solver-proof" data-proof-index="{proof_index+1}" data-window-index="{window_index+1}" '
                f'data-concept="{escape(concept_text)}" x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
                f'stroke="{PROOF_COLOR}" stroke-width="7" opacity="0.94" marker-end="url(#proof-arrow)"/>'
            )
            parts.append(f'<text data-role="proof-concept" x="{label_x}" y="{label_y}" text-anchor="middle" class="tiny" fill="#92400e">PROOF #{proof_index+1} | {escape(concept_text)} | {escape(relation)}</text>')

    # Right board: top-k learned attention only. No structural/topology edges.
    for row in range(size):
        for col in range(size):
            action = row * size + col
            x, y = right_x + col * step, board_y + row * step
            parts.append(f'<g data-board="attention" data-action="{action}" data-role="cell"><rect x="{x}" y="{y}" width="{step}" height="{step}" fill="#ffffff" stroke="#dbe3ec"/>')
            stone = int(board[row][col])
            if stone:
                stone_fill = "#0f172a" if stone == 1 else "#ffffff"
                parts.append(f'<circle cx="{x+step/2}" cy="{y+step/2}" r="19" fill="{stone_fill}" stroke="#0f172a" stroke-width="2"/>')
            parts.append("</g>")

    for edge in reversed(selected_attention):
        source, target = int(edge["source"]["action"]), int(edge["target"]["action"])
        x1, y1 = _point(source, size, right_x, board_y, step)
        x2, y2 = _point(target, size, right_x, board_y, step)
        normalized = 0.0 if attention_span <= 1e-12 else (float(edge["attention"]) - attention_min) / attention_span
        opacity = 0.42 + 0.50 * normalized
        width_px = 2.4 + 4.6 * normalized
        parts.append(
            f'<line data-layer="rgat-attention" data-rank="{ranked_attention.index(edge)+1}" data-edge-id="{escape(edge["edge_id"])}" '
            f'data-attention="{float(edge["attention"]):.9f}" data-attention-norm="{normalized:.9f}" '
            f'x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{ATTENTION_COLOR}" stroke-linecap="round" '
            f'stroke-width="{width_px:.3f}" opacity="{opacity:.3f}"/>'
        )

    for cell in sorted(critical_cells):
        row, col = divmod(cell, size)
        x, y = right_x + col * step + 5, board_y + row * step + 5
        parts.append(f'<rect data-layer="critical-reference" data-action="{cell}" x="{x}" y="{y}" width="{step-10}" height="{step-10}" rx="4" fill="none" stroke="#f59e0b" stroke-width="1.5" stroke-dasharray="5 4" opacity="0.62"/>')

    if selected_action is not None:
        parts.append(_selected_marker(selected_action, size, left_x, board_y, step, "tactic"))
        parts.append(_selected_marker(selected_action, size, right_x, board_y, step, "attention"))

    board_bottom = board_y + board_width
    parts.extend([
        f'<text x="{left_x}" y="{board_bottom+25}" class="note">Cam = hợp critical cells của toàn bộ {len(proofs)} flat proofs.</text>',
        f'<text x="{right_x}" y="{board_bottom+25}" class="note">Xanh = top-{top_k} theo attention thật; cam nét đứt = tactic reference.</text>',
        f'<text x="{right_x}" y="{board_bottom+58}" class="bigmetric">topology_corr = {topology:.3f}</text>',
        f'<text x="{right_x+315}" y="{board_bottom+55}" class="note">top-k ∩ proof edges: {top_k_overlap}/{top_k}</text>',
        f'<rect x="{left_x}" y="{metrics_y}" width="{width-2*left_x}" height="102" rx="10" fill="{"#fff7ed" if collapse else "#ecfeff"}" stroke="{"#c2410c" if collapse else "#0e7490"}" stroke-width="2"/>',
        f'<text x="{left_x+16}" y="{metrics_y+27}" class="metric">{"ATTENTION COLLAPSE" if collapse else "NO COLLAPSE — mismatch is not uniform attention"}</text>',
        f'<text x="{left_x+16}" y="{metrics_y+52}" class="note">collapse_flag={collapse} · entropy={entropy:.3f} · head diversity={diversity:.3f}</text>',
        f'<text x="{left_x+16}" y="{metrics_y+75}" class="note">topology_corr={topology:.3f} · proof critical mass={alignment:.3f} · attention cutoff={cutoff:.6f}</text>',
        f'<text x="{right_x}" y="{metrics_y+27}" class="note">{escape(evidence_note)}</text>',
        f'<text x="{left_x}" y="{proof_y-18}" class="panel">All flat proofs (none hidden)</text>',
    ])

    for proof_index, proof in enumerate(proofs):
        concepts = list(proof.get("concepts", []))
        concept_text = concepts[0] if len(concepts) == 1 else "proof-level: " + " + ".join(concepts)
        relations = ", ".join(proof.get("critical_relations", []))
        legend_x = left_x + (proof_index % 2) * 610
        item_y = proof_y + (proof_index // 2) * 38
        parts.append(
            f'<g data-role="proof-legend" data-proof-index="{proof_index+1}">'
            f'<circle cx="{legend_x+7}" cy="{item_y-4}" r="6" fill="{PROOF_COLOR}"/>'
            f'<text x="{legend_x+21}" y="{item_y}" class="note">PROOF #{proof_index+1} action={int(proof["action"])} | {escape(concept_text)}</text>'
            f'<text x="{legend_x+21}" y="{item_y+15}" class="tiny">relations: {escape(relations)} · windows: {len(proof.get("windows", []))}</text></g>'
        )

    parts.append(f'<text x="32" y="{height-25}" class="note">D4 gate: PASS · {int(gate.get("d4_proof_roundtrips", 0))}/{int(gate.get("d4_proof_roundtrips", 0))} proof round-trips · model evidence is not causal proof.</text>')
    parts.append("</svg>")
    return "".join(parts)


def render_knowledge_notice_svg(record, reason, decision=None, artifact_version=1):
    """Render an honest fourth artifact when no replayed proof exists."""
    state = record["state"]
    size = int(state["board_size"])
    board = state["board"]
    decision, selected_action = _decision_fields({"decision": decision or {}}, size)
    step, ox, oy = 64, 42, 112
    board_width = size * step
    width, height = board_width + 500, max(oy + board_width + 70, 600)
    solver = record.get("solver", {})
    status = solver.get("status", "unknown")
    complete = bool(solver.get("optimal_actions_complete", False))
    if status == "exact_complete" and complete:
        badge = "COMPLETE LABEL · NO REPLAYED PROOF"
        fill, stroke = "#e0f2fe", "#0369a1"
    elif status == "exact_partial":
        badge = "PARTIAL KNOWLEDGE · NO FLAT PROOF"
        fill, stroke = "#fef3c7", "#b45309"
    else:
        badge = "UNKNOWN · NO GROUND-TRUTH PROOF"
        fill, stroke = "#f1f5f9", "#475569"
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
        f'data-artifact-version="{int(artifact_version)}" data-selected-action="{selected_action if selected_action is not None else ""}">',
        '<title>Knowledge contrast unavailable</title>',
        '<style>text{font-family:Arial,sans-serif;fill:#0f172a}.title{font-size:22px;font-weight:700}.panel{font-size:15px;font-weight:700}.note{font-size:12px}.tiny{font-size:9px;font-weight:700}.badge{font-size:12px;font-weight:700}</style>',
        f'<rect width="{width}" height="{height}" fill="#f8fafc"/>',
        f'<text x="28" y="34" class="title">Knowledge diagram — {escape(str(record.get("state_id", "unknown")))}</text>',
        '<text x="28" y="58" class="note">Arena ground-truth routing result; no solver layer is invented.</text>',
        f'<rect x="{width-330}" y="20" width="300" height="32" rx="16" fill="{fill}" stroke="{stroke}" stroke-width="2"/>',
        f'<text x="{width-180}" y="41" text-anchor="middle" class="badge">{badge}</text>',
    ]
    for row in range(size):
        for col in range(size):
            action = row * size + col
            x, y = ox + col * step, oy + row * step
            parts.append(f'<g data-action="{action}" data-role="cell"><rect x="{x}" y="{y}" width="{step}" height="{step}" fill="#ffffff" stroke="#cbd5e1"/>')
            stone = int(board[row][col])
            if stone:
                stone_fill = "#0f172a" if stone == 1 else "#ffffff"
                parts.append(f'<circle cx="{x+step/2}" cy="{y+step/2}" r="18" fill="{stone_fill}" stroke="#0f172a" stroke-width="2"/>')
            parts.append("</g>")
    if selected_action is not None:
        parts.append(_selected_marker(selected_action, size, ox, oy, step, "notice"))
        actor = decision.get("actor") or {}
        row, col = divmod(selected_action, size)
        parts.append(
            f'<text x="28" y="82" class="note" data-role="decision-lineage">Actual actor: '
            f'{escape(str(actor.get("type", "unknown")).upper())} | MCTS SELECTED action={selected_action} '
            f'(row={row}, col={col})</text>'
        )
    side_x = ox + board_width + 38
    parts.extend([
        f'<text x="{side_x}" y="145" class="panel">Contrast intentionally omitted</text>',
        f'<text x="{side_x}" y="176" class="note">solver status: {escape(str(status))}</text>',
        f'<text x="{side_x}" y="198" class="note">optimal actions complete: {str(complete).lower()}</text>',
        f'<text x="{side_x}" y="230" class="note">reason: {escape(str(reason))}</text>',
        f'<text x="{side_x}" y="276" class="note">R-GAT/R-GCN overlays are withheld because</text>',
        f'<text x="{side_x}" y="296" class="note">there is no replayed proof to contrast against.</text>',
    ])
    parts.append("</svg>")
    return "".join(parts)
