# AlphaZero Gomoku Graph Pipeline

Pipeline nghiên cứu gọn cho Gomoku 6×6, `win_length=4`, so sánh hai graph encoder:

- **R-GCN:** message passing theo bốn quan hệ hình học.
- **R-GAT:** relational attention theo từng cạnh và attention head.

Pipeline chính:

```text
GomokuState
  → R-GCN hoặc R-GAT policy/value
  → MCTS self-play
  → replay buffer + symmetry augmentation
  → checkpoint H3
  → H1 tactical evaluation
  → arena + JSON/SVG graph evidence
```

Ground truth dùng exact solver trên bàn nhỏ và VCF certificate có abstention. CNN, HAN và các benchmark tối ưu hóa trung gian không còn thuộc source chính.

## Cài đặt trong WSL

```bash
cd /mnt/d/ThucNghiem/alphazero-gomoku-graph-demo
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Nếu dùng dependency đã có trong `.h3deps`:

```bash
export PYTHONPATH="$PWD/.h3deps${PYTHONPATH:+:$PYTHONPATH}"
```

## Chạy test

```bash
python -m pytest -q -p no:cacheprovider
```

## Train H3

Config chính:

- `configs/h3_pilot_rgat.yaml`
- `configs/h3_pilot_rgcn.yaml`

Hai config đã bật các cơ chế chống policy collapse:

- tám phép symmetry augmentation;
- root Dirichlet noise trong self-play;
- sampling temperature ở opening, greedy ở late game;
- diagnostic corner/edge mass, opening entropy và opening diversity.

Train mới từ đầu, không resume run cũ đã bị opening bias:

```bash
python -m experiments.run_h3_pilot \
  --config configs/h3_pilot_rgat.yaml \
  --output results/h3_pilot_v2/rgat/seed_7
```

```bash
python -m experiments.run_h3_pilot \
  --config configs/h3_pilot_rgcn.yaml \
  --output results/h3_pilot_v2/rgcn/seed_7
```

Resume từ checkpoint:

```bash
python -m experiments.run_h3_pilot \
  --config configs/h3_pilot_rgat.yaml \
  --output results/h3_pilot_v2/rgat/seed_7 \
  --resume results/h3_pilot_v2/rgat/seed_7/checkpoints/iter_020.pt
```

Artifact mỗi run:

```text
config.json
model.pt
training_log.csv
checkpoints/manifest.json
checkpoints/iter_XXX.pt
```

## H1 ground truth

Benchmark cố định:

```text
diagnostic/h1_tactical.jsonl
```

Sinh lại benchmark:

```bash
python -m investigation.generate_h1_benchmark \
  --target 24 \
  --seed 7 \
  --output diagnostic/h1_tactical.jsonl
```

Kiểm tra exact solver bằng exhaustive oracle độc lập:

```bash
python -m investigation.validate_solver
```

Chạy H1 trên toàn bộ checkpoint H3:

```bash
python -m investigation.h3_evaluate \
  --run-dir results/h3_pilot_v2/rgat/seed_7 \
  --output results/h3_pilot_v2/rgat/seed_7/developmental_metrics.csv
```

```bash
python -m investigation.h3_evaluate \
  --run-dir results/h3_pilot_v2/rgcn/seed_7 \
  --output results/h3_pilot_v2/rgcn/seed_7/developmental_metrics.csv
```

H1 báo riêng policy, value, MCTS và graph alignment. MCTS accuracy cao không đồng nghĩa graph attention đã align với tactical proof.

## Exact solver và VCF

- `azgomoku/solver.py`: bounded exact negamax/alpha-beta cho oracle bàn nhỏ.
- `azgomoku/tactics.py`: five/four/three primitives và tactical proofs.
- `azgomoku/vcf.py`: one-sided VCF, budget → `unknown`, certificate replay bắt buộc.
- `azgomoku/oracle_agreement.py`: chặn false-positive bằng exact oracle 6×6.

Chạy correctness gate:

```bash
python -m pytest \
  tests/test_solver.py \
  tests/test_tactics.py \
  tests/test_threat_primitives.py \
  tests/test_oracle_agreement.py \
  tests/test_vcf.py -q -p no:cacheprovider
```

Threat solver chỉ phát `exact_partial`, `value=+1`, `optimal_actions_complete=false` khi proof replay thành công. Không chứng minh được thì trả `unknown`.

## Cho R-GAT và R-GCN đấu nhau, đồng thời export graph

R-GAT đi trước:

```bash
python -m azgomoku.explanation.game_export \
  --model rgat \
  --checkpoint results/h3_pilot_v2/rgat/seed_7/model.pt \
  --opponent model \
  --opponent-model rgcn \
  --opponent-checkpoint results/h3_pilot_v2/rgcn/seed_7/model.pt \
  --model-player 1 \
  --mcts-playouts 50 \
  --temperature 0 \
  --output results/h3_pilot_v2/arena/rgat_vs_rgcn/game_01
```

Đổi bên đi trước bằng cách hoán đổi `--model` và `--opponent-model`.

Mỗi nước đi sinh:

```text
move_XXX/board.svg
move_XXX/graph.svg
move_XXX/decision.svg
move_XXX/explanation.json
```

`game.json` lưu toàn bộ nước đi, model mỗi bên và winner.

## Cấu trúc source giữ lại

```text
azgomoku/       game, graph, MCTS, training, solver, VCF, explanation
models/         R-GCN và R-GAT
experiments/    H3 pilot trainer
investigation/  H1 generator/evaluator và solver validation
configs/        config train R-GCN/R-GAT
diagnostic/     frozen H1 benchmark
tests/          correctness và regression suite
results/h1/     H1 artifacts hiện có
results/h3_pilot*/ checkpoint, metrics và arena artifacts
```

## Diễn giải kết quả

- Chuẩn vàng chính vẫn là exact 6×6/k=4.
- Kết quả threat solver trên bàn lớn chỉ áp dụng cho tập `tactically-decided` và phải báo coverage.
- R-GAT attention là model evidence, không phải causal proof.
- Arena cần nhiều seed và đổi bên đi trước; một hoặc hai game không đủ kết luận model mạnh hơn.
