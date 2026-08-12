# AlphaZero Gomoku Graph Demo

A small PyTorch feasibility study comparing CNN, R-GCN, R-GAT, and HAN state encoders inside the same AlphaZero-style Gomoku pipeline.

The default game is 6×6 Gomoku with four stones in a row required to win. Every model maps the same state features to a policy over board Cells and a scalar value estimate. Graph processing is confined to the neural-network encoder; trajectories, replay data, and MCTS use the same contract for every model.

## Architectures

| Model | State encoder | Graph evidence |
|---|---|---|
| CNN | Two 3×3 convolutions | No graph |
| R-GCN | Relation-specific graph convolutions | Structural relations; no learned attention |
| R-GAT | Relational multi-head graph attention | Per-edge, per-head learned attention |
| HAN | Cell–Line–Cell meta-path views | Node-level and semantic attention |

The four graph relations are horizontal, vertical, diagonal-down, and diagonal-up. Policy action indexing is row-major: `action = row * board_size + col`.

Attention values are state-specific model evidence, not causal explanations.

## Setup

Create and activate a virtual environment, then install the dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Run the tests:

```powershell
python -m pytest -q
```

## Train or regenerate checkpoints

Run a small smoke profile:

```powershell
python -m experiments.run_cnn --profile smoke
python -m experiments.run_rgcn --profile smoke
python -m experiments.run_rgat --profile smoke
python -m experiments.run_han --profile smoke
```

Available profiles:

| Profile | MCTS playouts | Self-play games | Purpose |
|---|---:|---:|---|
| `smoke` | 10 | 2 | Verify execution and artifact collection |
| `default` | 50 | 30 | Small demonstration run |
| `pilot` | 100 | 100 | Larger experimental run |

Each experiment writes its checkpoint and metrics to `results/<model>/`.

## Static architecture/topology visualization

Static artifacts answer: “What graph structure does this architecture use?” They do not explain why a move was selected in a particular state.

Generate the topology JSON files:

```powershell
python -m azgomoku.graph_export
```

Generate the R-GCN, R-GAT, HAN, and comparison SVGs:

```powershell
python -m azgomoku.graph_visualization
```

Outputs are written to `results/graphs/`:

- `cell_graph.json`
- `han_metapaths.json`
- `rgcn_graph.svg`
- `rgat_graph.svg`
- `han_graph.svg`
- `graph_comparison.svg`

## State-specific decision SVG pipeline

The image subsystem is integrated into an explicit post-search explanation pipeline:

```text
pre-move root state
    → policy/value network
    → MCTS search with evidence disabled
    → selected action and root P/Q/N/π trace
    → one evidence-enabled forward on the root state
    → explanation.json
    → board.svg, graph.svg, decision.svg
```

It is intentionally not enabled during ordinary training, self-play, or individual MCTS playouts. This keeps the normal AlphaZero path free of JSON and SVG overhead.

Run an explanation for HAN:

```powershell
python -m azgomoku.explanation.explanation_export `
  --model han `
  --checkpoint results\han\model.pt `
  --state examples\tactical_positions\mixed_relations.json `
  --mcts-playouts 100 `
  --top-k-candidates 5 `
  --top-k-edges 12 `
  --output results\explanations\mixed_relations_100\han
```

Change `--model` and `--checkpoint` to run the other graph encoders:

```powershell
# R-GCN
python -m azgomoku.explanation.explanation_export `
  --model rgcn `
  --checkpoint results\rgcn\model.pt `
  --state examples\tactical_positions\mixed_relations.json `
  --mcts-playouts 100 `
  --output results\explanations\mixed_relations_100\rgcn

# R-GAT
python -m azgomoku.explanation.explanation_export `
  --model rgat `
  --checkpoint results\rgat\model.pt `
  --state examples\tactical_positions\mixed_relations.json `
  --mcts-playouts 100 `
  --output results\explanations\mixed_relations_100\rgat
```

Each explanation directory contains:

- `explanation.json`: versioned network, graph-evidence, MCTS, filtering, and runtime data.
- `board.svg`: diagnostic rendering of the exact pre-move root state.
- `graph.svg`: filtered state-specific relational evidence on fixed board coordinates.
- `decision.svg`: primary three-panel visualization combining board, search, and relational evidence.

R-GCN displays structural edges without assigning learned importance. R-GAT renders final-layer attention using the documented mean across heads. HAN keeps node-level meta-path attention separate from semantic attention.

### Render from a saved trace

SVGs can be reproduced without loading a model or running MCTS:

```powershell
python -m azgomoku.explanation.explanation_export `
  --trace results\explanations\mixed_relations_100\han\explanation.json `
  --output results\explanations\mixed_relations_100\han_rerendered
```

Rendering uses fixed board coordinates, stable sorting, and no random layout, so the same JSON produces identical SVG output.

## Example states

Held-out visualization fixtures are stored in `examples/tactical_positions/`:

- `horizontal_threat.json`
- `vertical_threat.json`
- `diagonal_configuration.json`
- `mixed_relations.json`

These files are inspection inputs only. Their labels and intended patterns are not used as training features or supervised targets.

## Interpretation limits

This repository demonstrates that the encoders and evidence pipeline execute under a shared AlphaZero contract. The smoke checkpoints are not evidence that one architecture is stronger than another. Meaningful comparison requires larger training budgets, repeated seeds, controlled parameter counts, evaluation games, and uncertainty estimates.

Likewise, learned attention indicates where a model assigned more attention; it does not prove that an edge or relation caused the selected action.
