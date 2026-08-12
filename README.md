# AlphaZero Gomoku Graph Demo

A deliberately small PyTorch feasibility study of graph state encoders inside an AlphaZero policy-value network. The default game is 6×6 Gomoku with four in a row. The training contract is unchanged: each replay item is `(state, MCTS policy, final outcome)`, and every model implements `state -> (policy, value)`. Graph processing happens only inside the network, never in trajectories or replay aggregation.

## Architectures

| Model | State encoder | Final smoke loss | Batch-1 CPU latency (ms) | Status |
|---|---|---:|---:|---|
| CNN | Two 3×3 convolutions | 4.507 | 0.570 | passed |
| R-GCN | Two relational convolution layers, four direction relations | 4.454 | 2.272 | passed |
| R-GAT | Learned relational attention on the identical R-GCN Cell graph | 4.388 | 35.600 | passed |
| HAN | Four Cell–Line–Cell meta-path views; node then global semantic attention | 4.577 | 15.522 | passed |

Every policy is one row-major logit per Cell. The measurements above are one tiny smoke run, intended to verify execution and artifact collection rather than rank models.

All use six Cell features: current-player stone, opponent stone, last move, side to move, row coordinate, and column coordinate. The shared loss is `MSE(z, v) - sum(pi * log(p))`; AdamW-style L2 regularization is supplied through optimizer weight decay.

## Run

```powershell
python -m experiments.run_cnn --profile smoke
python -m experiments.run_rgcn --profile smoke
python -m experiments.run_rgat --profile smoke
python -m experiments.run_han --profile smoke
```

Profiles are `smoke` (10 playouts, 2 games), `default` (50, 30), and optional `pilot` (100, 100). Other requested defaults are encoded in `azgomoku/config.py`.

Each run writes configuration, checkpoint, CSV metrics, runtime metadata, tactical inspection JSON, and a Markdown summary to its independent `results/<model>` directory. Tests cover game rules, winner/legal actions, exact Cell/action mapping, legal masking, cached symmetric topology, fixed-data overfit, output shape, and batch-1 latency is recorded by every run.

The tactical positions are held out for qualitative inspection only; their labels are not features or training inputs. Attention weights are descriptive diagnostics, not causal explanations.

## Interpretation

This tiny demo establishes that the four encoders can participate in the same AlphaZero loop. It is a feasibility study, not evidence that any architecture is superior. Meaningful comparison requires repeated seeds, larger training budgets, controlled parameter counts, and uncertainty estimates.
