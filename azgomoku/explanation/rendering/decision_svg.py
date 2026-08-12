"""Primary three-panel human-facing decision SVG."""

from .graph_svg import COLORS, select_render_edges


def render_decision_svg(document):
    state=document["state"]; size=state["board_size"]; board=state["board"]; selected=document["selected_move"]; step=46; ox,oy=28,92; board_right=ox+size*step
    width,height=1180,560; search_x=board_right+40; graph_x=760
    parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}"><title>AlphaZero state-specific decision evidence</title><style>text{{font-family:Arial;fill:#0f172a}}.t{{font-size:22px;font-weight:700}}.h{{font-size:15px;font-weight:700}}.n{{font-size:11px}}.b{{font-size:12px}}</style><rect width="1180" height="560" fill="#f8fafc"/><text x="24" y="32" class="t">AlphaZero decision evidence - {document["model"]["type"].upper()}</text><text x="24" y="53" class="n">Pre-move state | network evidence | MCTS root trace | model attention is not causal proof</text><text x="28" y="78" class="h">Board</text><text x="{search_x}" y="78" class="h">Decision / search</text><text x="{graph_x}" y="78" class="h">Relational evidence</text>']
    candidates={x["action"]:x for x in document["mcts"].get("top_candidates",[])}
    for row in range(size):
        for col in range(size):
            action=row*size+col; x=ox+col*step; y=oy+row*step; role="selected_move" if action==selected["action"] else ("mcts_candidate" if action in candidates else "cell")
            parts.append(f'<g data-node-row="{row}" data-node-col="{col}" data-action="{action}" data-role="{role}"><rect x="{x}" y="{y}" width="{step}" height="{step}" fill="{"#fef3c7" if action==selected["action"] else "#ffffff"}" stroke="#94a3b8"/>')
            stone=int(board[row][col])
            if stone: parts.append(f'<circle cx="{x+step/2}" cy="{y+step/2}" r="14" fill="{"#0f172a" if stone==state["current_player"] else "#ffffff"}" stroke="#0f172a" stroke-width="2"/>')
            if state["last_move"]==[row,col]: parts.append(f'<circle data-role="last_move" cx="{x+step/2}" cy="{y+step/2}" r="4" fill="#f97316"/>')
            if action==selected["action"]: parts.append(f'<rect x="{x+3}" y="{y+3}" width="{step-6}" height="{step-6}" fill="none" stroke="#e11d48" stroke-width="4"/>')
            parts.append('</g>')
    net=document["network"]; parts.append(f'<text x="{search_x}" y="110" class="b">Value V(s): {net["value"]:+.6f}</text><text x="{search_x}" y="132" class="b">Selected raw prior: {net["raw_policy_prior"]:.6f}</text><text x="{search_x}" y="166" class="n">move       raw P   tree P    N      Q       pi</text>')
    for index,item in enumerate(document["mcts"].get("top_candidates",[])):
        y=190+index*25; raw="null" if item["raw_policy_prior"] is None else f'{item["raw_policy_prior"]:.3f}'; marker="*" if item["selected"] else " "
        parts.append(f'<g data-action="{item["action"]}" data-visits="{item["visits"]}" data-q="{item["q"]:.9f}"><text x="{search_x}" y="{y}" class="b">{marker}({item["row"]},{item["col"]})   {raw}   {item["search_prior"]:.3f}   {item["visits"]:3d}   {item["q"]:+.3f}   {item["pi"]:.3f}</text></g>')
    gx0,gy0=graph_x+120,235
    for edge in select_render_edges(document):
        source=edge["source"]; target=edge["target"]; attention=edge.get("attention"); strength=.35 if attention is None else max(0,min(1,attention)); x1=gx0+(source["col"]-selected["col"])*36; y1=gy0+(source["row"]-selected["row"])*36; x2=gx0+(target["col"]-selected["col"])*36; y2=gy0+(target["row"]-selected["row"])*36
        parts.append(f'<path data-relation="{edge["relation"]}" data-attention="{"" if attention is None else f"{attention:.9f}"}" d="M {x1} {y1} L {x2} {y2}" stroke="{COLORS[edge["relation"]]}" stroke-width="{1+7*strength:.3f}" opacity="{.25+.7*strength:.3f}" fill="none"/>')
    parts.append(f'<circle cx="{gx0}" cy="{gy0}" r="18" fill="#fef3c7" stroke="#e11d48" stroke-width="4"/><text x="{gx0}" y="{gy0+4}" class="n" text-anchor="middle">SELECT</text>')
    if document["model"]["type"]=="rgcn": parts.append(f'<text x="{graph_x}" y="330" class="n">R-GCN exposes structural relations but no</text><text x="{graph_x}" y="347" class="n">learned attention coefficients.</text>')
    semantic=document.get("semantic_attention",{})
    if semantic:
        parts.append(f'<g data-role="semantic_attention"><text x="{graph_x}" y="350" class="h">HAN semantic attention</text>')
        for index,name in enumerate(("horizontal","vertical","diagonal_down","diagonal_up")):
            value=semantic[name]
            y=378+index*30; parts.append(f'<rect x="{graph_x}" y="{y-11}" width="{value*150:.2f}" height="13" fill="{COLORS[name]}" opacity=".75"/><text x="{graph_x+158}" y="{y}" class="n">{name}: {value:.6f}</text>')
        parts.append('</g>')
    parts.append('</svg>'); return "".join(parts)
