"""Filtered relational evidence on fixed board coordinates."""

COLORS={"horizontal":"#2563eb","vertical":"#16a34a","diagonal_down":"#dc2626","diagonal_up":"#9333ea"}


def select_render_edges(document):
    edges=document["graph_evidence"]["edges"]; selected=document["selected_move"]["action"]
    candidates={x["action"] for x in document.get("mcts",{}).get("top_candidates",[])}; limit=document["rendering"]["top_k_edges"]
    def rank(edge):
        source=edge["source"]["action"]; target=edge["target"]["action"]
        group=0 if selected in (source,target) else (1 if source in candidates or target in candidates else 2)
        attention=edge.get("attention"); return (group,-(attention if attention is not None else -1),edge["relation"],source,target)
    return sorted(edges,key=rank)[:limit]


def render_graph_svg(document):
    state=document["state"]; size=state["board_size"]; board=state["board"]; selected=document["selected_move"]["action"]
    step,ox,oy=58,42,76; right=ox+size*step; width=right+330; height=max(oy+size*step+34,500)
    parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}"><title>State-specific relational evidence</title><style>text{{font-family:Arial;fill:#0f172a}}.t{{font-size:20px;font-weight:700}}.h{{font-size:14px;font-weight:700}}.n{{font-size:11px}}</style><rect width="{width}" height="{height}" fill="#f8fafc"/><text x="24" y="30" class="t">State-specific graph evidence</text><text x="24" y="51" class="n">Filter: {document["rendering"]["edge_filter"]}; top {document["rendering"]["top_k_edges"]} edges</text>']
    for row in range(size):
        for col in range(size):
            action=row*size+col; x=ox+col*step; y=oy+row*step; stone=int(board[row][col]); fill="#fef3c7" if action==selected else "#ffffff"
            parts.append(f'<g data-node-row="{row}" data-node-col="{col}" data-action="{action}" data-role="{"selected_move" if action==selected else "cell"}"><rect x="{x}" y="{y}" width="{step}" height="{step}" fill="{fill}" stroke="#cbd5e1"/>')
            if stone: parts.append(f'<circle cx="{x+step/2}" cy="{y+step/2}" r="17" fill="{"#0f172a" if stone==state["current_player"] else "#ffffff"}" stroke="#0f172a" stroke-width="2"/>')
            parts.append('</g>')
    for edge in select_render_edges(document):
        source=edge["source"]; target=edge["target"]; attention=edge.get("attention"); strength=.35 if attention is None else max(0,min(1,attention))
        x1=ox+(source["col"]+.5)*step; y1=oy+(source["row"]+.5)*step; x2=ox+(target["col"]+.5)*step; y2=oy+(target["row"]+.5)*step
        att="" if attention is None else f'{attention:.9f}'
        parts.append(f'<path data-relation="{edge["relation"]}" data-attention="{att}" data-source-action="{source["action"]}" data-target-action="{target["action"]}" d="M {x1} {y1} L {x2} {y2}" stroke="{COLORS[edge["relation"]]}" stroke-width="{1+7*strength:.3f}" opacity="{.25+.7*strength:.3f}" fill="none"/>')
    sx=ox+(document["selected_move"]["col"]+.5)*step; sy=oy+(document["selected_move"]["row"]+.5)*step; parts.append(f'<circle cx="{sx}" cy="{sy}" r="23" fill="none" stroke="#e11d48" stroke-width="5"/>')
    px=right+28; parts.append(f'<text x="{px}" y="100" class="h">Relational evidence</text>')
    if document["model"]["type"]=="rgcn": parts.append(f'<text x="{px}" y="126" class="n">R-GCN exposes structural relations</text><text x="{px}" y="143" class="n">but no learned attention coefficients.</text>')
    else: parts.append(f'<text x="{px}" y="126" class="n">Larger attention = stronger rendered edge.</text>')
    parts.append('</svg>'); return "".join(parts)
