"""Physical-board SVG with semantic metadata."""


def render_board_svg(document):
    state=document["state"]; size=state["board_size"]; board=state["board"]; selected=document["selected_move"]["action"]
    candidates={item["action"]:item for item in document.get("mcts",{}).get("top_candidates",[])}
    step,ox,oy=58,42,72; width=ox*2+size*step; height=oy+size*step+34
    parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}"><title>Pre-move Gomoku root state</title><style>text{{font-family:Arial;fill:#0f172a}}.t{{font-size:20px;font-weight:700}}.n{{font-size:11px}}</style><rect width="{width}" height="{height}" fill="#f8fafc"/><text x="{width/2}" y="30" class="t" text-anchor="middle">Pre-move root state</text>']
    last=state["last_move"]
    for row in range(size):
        for col in range(size):
            action=row*size+col; x=ox+col*step; y=oy+row*step
            role="selected_move" if action==selected else ("mcts_candidate" if action in candidates else "cell")
            fill="#fef3c7" if action==selected else ("#eff6ff" if action in candidates else "#ffffff")
            parts.append(f'<g data-node-row="{row}" data-node-col="{col}" data-action="{action}" data-role="{role}"><rect x="{x}" y="{y}" width="{step}" height="{step}" fill="{fill}" stroke="#94a3b8"/>')
            stone=int(board[row][col])
            if stone:
                stone_role="current_player_stone" if stone==state["current_player"] else "opponent_stone"; stone_fill="#0f172a" if stone==state["current_player"] else "#ffffff"
                parts.append(f'<circle data-role="{stone_role}" cx="{x+step/2}" cy="{y+step/2}" r="18" fill="{stone_fill}" stroke="#0f172a" stroke-width="2"/>')
            if last==[row,col]: parts.append(f'<circle data-role="last_move" cx="{x+step/2}" cy="{y+step/2}" r="5" fill="#f97316"/>')
            if action in candidates:
                item=candidates[action]; parts.append(f'<text x="{x+step-5}" y="{y+14}" class="n" text-anchor="end">N={item["visits"]}</text>')
            if action==selected: parts.append(f'<rect x="{x+3}" y="{y+3}" width="{step-6}" height="{step-6}" fill="none" stroke="#e11d48" stroke-width="5"/>')
            parts.append('</g>')
    parts.append('</svg>'); return "".join(parts)
