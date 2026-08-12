"""Render clearly labeled static topology documentation SVGs."""

import argparse
from pathlib import Path

from .graph import RELATIONS, cell_graph, metapath_edges


COLORS = ("#2563eb", "#16a34a", "#dc2626", "#9333ea")


def _svg(size, model):
    groups = metapath_edges(size) if model == "han" else cell_graph(size)
    title = f"{model.upper()} static topology visualization"
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="760" height="470" viewBox="0 0 760 470"><title>{title}</title><style>text{{font-family:Arial;fill:#0f172a}}.t{{font-size:22px;font-weight:700}}.n{{font-size:12px}}</style><rect width="760" height="470" fill="#f8fafc"/><text x="380" y="32" class="t" text-anchor="middle">{title}</text><text x="380" y="55" class="n" text-anchor="middle">Architecture documentation only - not a state-specific decision explanation</text>']
    step, ox, oy = 54, 42, 92
    for relation, edges, color in zip(RELATIONS, groups, COLORS):
        for source, target in edges.t().tolist():
            if source >= target: continue
            sr, sc = divmod(source, size); tr, tc = divmod(target, size)
            parts.append(f'<line x1="{ox+(sc+.5)*step}" y1="{oy+(sr+.5)*step}" x2="{ox+(tc+.5)*step}" y2="{oy+(tr+.5)*step}" stroke="{color}" opacity=".22"/>')
    for i in range(size*size):
        r, c = divmod(i, size); parts.append(f'<circle cx="{ox+(c+.5)*step}" cy="{oy+(r+.5)*step}" r="5" fill="#fff" stroke="#334155"/>')
    x = 420
    for index, (relation, color, edges) in enumerate(zip(RELATIONS, COLORS, groups)):
        y = 132 + index*50; parts.append(f'<line x1="{x}" y1="{y}" x2="{x+40}" y2="{y}" stroke="{color}" stroke-width="4"/><text x="{x+52}" y="{y+5}" class="n">{relation}: {edges.shape[1]} directed edges</text>')
    note = "Same Cell graph as R-GCN; learned attention changes weights, not topology." if model == "rgat" else ("Cell-Line-Cell meta-path topology." if model == "han" else "Typed Cell adjacency with relation-specific transforms.")
    parts.append(f'<text x="420" y="365" class="n">{note}</text></svg>')
    return "".join(parts)


def render_comparison_svg(size=6):
    counts = [sum(e.shape[1] for e in cell_graph(size)), sum(e.shape[1] for e in cell_graph(size)), sum(e.shape[1] for e in metapath_edges(size))]
    names = ("R-GCN", "R-GAT", "HAN")
    notes = ("typed mean aggregation", "same topology + edge attention", "Cell-Line-Cell meta-path attention")
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="900" height="330" viewBox="0 0 900 330"><title>Static topology comparison</title><style>text{font-family:Arial;fill:#0f172a}.t{font-size:22px;font-weight:700}.h{font-size:16px;font-weight:700}.n{font-size:12px}</style><rect width="900" height="330" fill="#f8fafc"/><text x="450" y="34" class="t" text-anchor="middle">Static graph topology comparison</text><text x="450" y="57" class="n" text-anchor="middle">Architecture documentation only - not a decision explanation</text>']
    for i, (name, note, count) in enumerate(zip(names, notes, counts)):
        x = 25+i*292; parts.append(f'<rect x="{x}" y="82" width="270" height="205" rx="12" fill="#fff" stroke="{COLORS[i]}" stroke-width="2"/><text x="{x+135}" y="116" class="h" text-anchor="middle">{name}</text><text x="{x+135}" y="156" class="n" text-anchor="middle">{note}</text><text x="{x+135}" y="205" class="h" text-anchor="middle">{count}</text><text x="{x+135}" y="228" class="n" text-anchor="middle">directed edges across 4 views</text>')
    parts.append('</svg>'); return "".join(parts)


def export_model_visualizations(size=6, output_dir=Path("results/graphs")):
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True); outputs = {}
    for model in ("rgcn", "rgat", "han"):
        outputs[model] = output_dir / f"{model}_graph.svg"; outputs[model].write_text(_svg(size, model), encoding="utf-8")
    outputs["comparison"] = output_dir / "graph_comparison.svg"; outputs["comparison"].write_text(render_comparison_svg(size), encoding="utf-8")
    return outputs


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--board-size",type=int,default=6); parser.add_argument("--output-dir",type=Path,default=Path("results/graphs")); args=parser.parse_args()
    for path in export_model_visualizations(args.board_size,args.output_dir).values(): print(path.resolve())


if __name__ == "__main__": main()
