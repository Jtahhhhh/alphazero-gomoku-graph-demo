"""Export static graph topology JSON for architecture inspection."""

import argparse
import json
from pathlib import Path

from .graph import RELATIONS, cell_graph, line_memberships, metapath_edges


def _edges(groups):
    return {name: [{"source": a, "target": b} for a, b in edges.t().tolist()] for name, edges in zip(RELATIONS, groups)}


def export_graphs(size=6, output_dir=Path("results/graphs")):
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    nodes = [{"id": i, "row": i // size, "col": i % size} for i in range(size * size)]
    cell = {"artifact_type": "static_topology_visualization", "board_size": size, "nodes": nodes, "edges": _edges(cell_graph(size))}
    han = {
        "artifact_type": "static_topology_visualization", "board_size": size, "nodes": nodes,
        "line_memberships": {name: [list(line) for line in lines] for name, lines in zip(RELATIONS, line_memberships(size))},
        "metapath_edges": _edges(metapath_edges(size)),
    }
    outputs = {"cell": output_dir / "cell_graph.json", "han": output_dir / "han_metapaths.json"}
    outputs["cell"].write_text(json.dumps(cell, indent=2), encoding="utf-8")
    outputs["han"].write_text(json.dumps(han, indent=2), encoding="utf-8")
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--board-size", type=int, default=6); parser.add_argument("--output-dir", type=Path, default=Path("results/graphs")); args = parser.parse_args()
    for path in export_graphs(args.board_size, args.output_dir).values(): print(path.resolve())


if __name__ == "__main__": main()
