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

## Arena + dashboard chung cho 3 model

Chạy trong WSL sau khi đã activate `.venv`; block dưới sẽ đánh giá đồng thời CNN baseline, R-GCN và R-GAT với cùng bộ agent heuristic, rồi sinh dashboard chung cho cả 3 model.

```bash
cd /mnt/d/ThucNghiem/alphazero-gomoku-graph-demo
source .venv/bin/activate

python - <<'PY'
import json
from pathlib import Path

import torch

from models.cnn_baseline import CNNBaseline
from models.rgcn import RGCN
from models.rgat import RGAT
from investigation.eval_harness import run_evaluation

MODEL_SPECS = {
    "cnn_baseline": {
        "checkpoint": "results/arena15_baseline/run1/checkpoints/iter_020.pt",
        "model_cls": CNNBaseline,
        "model_kwargs": {"board_size": 15, "hidden_dim": 64},
    },
    "rgcn": {
        "checkpoint": "results/arena15_rgcn/run1/checkpoints/iter_020.pt",
        "model_cls": RGCN,
        "model_kwargs": {"board_size": 15, "hidden_dim": 128, "attention_heads": 4},
    },
    "rgat": {
        "checkpoint": "results/arena15_rgat/run1/checkpoints/iter_020.pt",
        "model_cls": RGAT,
        "model_kwargs": {"board_size": 15, "hidden_dim": 128, "attention_heads": 4},
    },
}

for model_name, spec in MODEL_SPECS.items():
    checkpoint = spec["checkpoint"]
    model = spec["model_cls"](**spec["model_kwargs"])
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model.eval()

    results = run_evaluation(
        model=model,
        model_name=model_name,
        iteration=20,
        opponents=[("egreedy", 0.10), ("alphabeta", 2)],
        n_games_per_opponent=20,
        mcts_playouts_eval=100,
        board_size=15,
        win_length=5,
        log_dir=Path(f"results/eval_logs/{model_name}"),
    )

    print(f"\n=== {model_name.upper()} ===")
    for item in results:
        print(json.dumps(item.to_dict(), ensure_ascii=False))
PY

python investigation/plot_dashboard.py \
  --results-dir results \
  --eval-dir results/eval_logs \
  --output-dir results/figures
```

Block trên là cách chạy chung cho cả 3 model trong một lần. Nếu checkpoint khác tên hoặc ở iteration khác, chỉ cần sửa `checkpoint` bên trong `MODEL_SPECS` cho phù hợp.

### Arena strength và Elo (1.000 game/matchup)

Pipeline Arena mới cân bằng 500 game model đi trước và 500 game model đi sau,
sau đó xuất raw W/D/L, Win Rate, Arena Score, ΔElo từng matchup và global Elo:

```bash
python -m experiments.run_arena \
  --cnn-checkpoint results/arena15_baseline/run1/checkpoints/iter_100.pt \
  --rgcn-checkpoint results/arena15_rgcn/run1/checkpoints/iter_100.pt \
  --rgat-checkpoint results/arena15_rgat/run1/checkpoints/iter_100.pt \
  --output results/arena \
  --games 1000 \
  --mcts-playouts 400 \
  --checkpoint-iteration 100 \
  --device cpu
```

Artifacts:

```text
results/arena/arena_games.csv    # raw từng game
results/arena/arena_summary.csv  # W/D/L, Win Rate, Arena Score, ΔElo
results/arena/arena_elo.json     # global Elo, anchor e_greedy = 0
```

`--games` phải là số chẵn để giữ cân bằng lượt đi. Draw được tính `0.5` trong
Arena Score, còn Win Rate chỉ tính số trận thắng.

### TensorBoard

H3 tự ghi event files vào `results/<run>/tensorboard`; Arena ghi thêm W/D/L,
Arena Score và Elo vào thư mục TensorBoard của run. Cài dependency rồi chạy:

```bash
pip install -r requirements.txt
tensorboard --logdir results
```

Có thể chỉ định thư mục riêng cho H3 hoặc Arena bằng `--tensorboard-logdir` để
overlay nhiều seed/checkpoint. TensorBoard là
lớp quan sát bổ sung; `training_log.csv`, raw Arena CSV/JSON và checkpoint vẫn
được giữ làm nguồn reproducibility.

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

### Train ma trận 6×6, 10×10 và 15×15 trong WSL

Ma trận mở rộng train cả R-GCN và R-GAT với cùng hyperparameter H3 pilot hiện
tại, chỉ thay `board_size`, `win_length`, `seed` và `run_id`:

| Bàn cờ | `win_length` (`k`) | Seed | Số run |
|---|---:|---|---:|
| 6×6 | 4 | 17, 29 | 4 |
| 10×10 | 5 | 7, 17, 29 | 6 |
| 15×15 | 5 | 7, 17, 29 | 6 |

Tổng cộng có 16 run. Runner hiện không có CLI override cho board/seed và chưa có
tham số `--device`; model cùng tensor train vẫn ở CPU. Vì vậy nên chạy tuần tự,
đặc biệt với 15×15.

#### 1. Vào môi trường WSL

```bash
cd /mnt/d/ThucNghiem/alphazero-gomoku-graph-demo
source .venv/bin/activate

export PYTHONPATH="$PWD/.h3deps${PYTHONPATH:+:$PYTHONPATH}"
```

#### 2. Sinh config 10×10/k=5 và 15×15/k=5

Các file config có đuôi `.yaml` hiện tại thực chất chứa JSON và được
`run_h3_pilot` đọc bằng `json.loads`. Đoạn dưới tạo 12 config mới trong
`configs/multiboard/`. Lệnh có thể chạy lại: config trùng nội dung sẽ được giữ,
còn file cùng tên nhưng khác nội dung sẽ làm lệnh dừng.

```bash
python - <<'PY'
import json
from pathlib import Path

config_dir = Path("configs/multiboard")
config_dir.mkdir(parents=True, exist_ok=True)

for model in ("rgcn", "rgat"):
    base_path = Path(f"configs/h3_pilot_{model}.yaml")
    base = json.loads(base_path.read_text(encoding="utf-8"))

    for board_size in (10, 15):
        win_length = 5
        for seed in (7, 17, 29):
            config = dict(base)
            config.update({
                "run_id": (
                    f"h3_{model}_{board_size}x{board_size}_"
                    f"k{win_length}_seed{seed}"
                ),
                "model_type": model,
                "board_size": board_size,
                "win_length": win_length,
                "seed": seed,
            })
            destination = config_dir / (
                f"h3_{model}_{board_size}x{board_size}_"
                f"k{win_length}_seed{seed}.json"
            )

            if destination.exists():
                existing = json.loads(destination.read_text(encoding="utf-8"))
                if existing != config:
                    raise RuntimeError(
                        f"Config đã tồn tại nhưng khác nội dung: {destination}"
                    )
                print(f"Giữ config có sẵn: {destination}")
                continue

            destination.write_text(
                json.dumps(config, indent=2) + "\n",
                encoding="utf-8",
            )
            print(f"Đã tạo: {destination}")
PY
```

#### 3. Khai báo helper train an toàn

Helper từ chối chạy nếu output đã tồn tại, tránh việc vô tình append log hoặc
trộn checkpoint của hai lần train. Mỗi run ghi thêm console vào `console.log`.

```bash
set -euo pipefail

run_h3_config() {
  local config="$1"
  local output="$2"

  if [[ ! -f "$config" ]]; then
    echo "Không tìm thấy config: $config" >&2
    return 1
  fi

  if [[ -e "$output" ]]; then
    echo "Từ chối ghi đè output đã tồn tại: $output" >&2
    return 1
  fi

  mkdir -p "$output"
  python -m experiments.run_h3_pilot \
    --config "$config" \
    --output "$output" \
    2>&1 | tee "$output/console.log"
}
```

Helper chỉ tồn tại trong terminal hiện tại. Nếu mở terminal WSL mới, chạy lại
block trên trước khi gọi các lệnh bên dưới.

#### 4. CLI riêng cho 6×6/k=4

Seed 7 đã có trong H3 pilot v2, nên ma trận bổ sung chỉ chạy seed 17 và 29.

R-GCN seed 17:

```bash
run_h3_config \
  configs/phase6_rgcn_seed17.yaml \
  results/h3_multiboard/6x6_k4/rgcn/seed_17
```

R-GCN seed 29:

```bash
run_h3_config \
  configs/phase6_rgcn_seed29.yaml \
  results/h3_multiboard/6x6_k4/rgcn/seed_29
```

R-GAT seed 17:

```bash
run_h3_config \
  configs/phase6_rgat_seed17.yaml \
  results/h3_multiboard/6x6_k4/rgat/seed_17
```

R-GAT seed 29:

```bash
run_h3_config \
  configs/phase6_rgat_seed29.yaml \
  results/h3_multiboard/6x6_k4/rgat/seed_29
```

#### 5. CLI riêng cho 10×10/k=5

R-GCN seed 7:

```bash
run_h3_config \
  configs/multiboard/h3_rgcn_10x10_k5_seed7.json \
  results/h3_multiboard/10x10_k5/rgcn/seed_7
```

R-GCN seed 17:

```bash
run_h3_config \
  configs/multiboard/h3_rgcn_10x10_k5_seed17.json \
  results/h3_multiboard/10x10_k5/rgcn/seed_17
```

R-GCN seed 29:

```bash
run_h3_config \
  configs/multiboard/h3_rgcn_10x10_k5_seed29.json \
  results/h3_multiboard/10x10_k5/rgcn/seed_29
```

R-GAT seed 7:

```bash
run_h3_config \
  configs/multiboard/h3_rgat_10x10_k5_seed7.json \
  results/h3_multiboard/10x10_k5/rgat/seed_7
```

R-GAT seed 17:

```bash
run_h3_config \
  configs/multiboard/h3_rgat_10x10_k5_seed17.json \
  results/h3_multiboard/10x10_k5/rgat/seed_17
```

R-GAT seed 29:

```bash
run_h3_config \
  configs/multiboard/h3_rgat_10x10_k5_seed29.json \
  results/h3_multiboard/10x10_k5/rgat/seed_29
```

#### 6. CLI riêng cho 15×15/k=5

R-GCN seed 7:

```bash
run_h3_config \
  configs/multiboard/h3_rgcn_15x15_k5_seed7.json \
  results/h3_multiboard/15x15_k5/rgcn/seed_7
```

R-GCN seed 17:

```bash
run_h3_config \
  configs/multiboard/h3_rgcn_15x15_k5_seed17.json \
  results/h3_multiboard/15x15_k5/rgcn/seed_17
```

R-GCN seed 29:

```bash
run_h3_config \
  configs/multiboard/h3_rgcn_15x15_k5_seed29.json \
  results/h3_multiboard/15x15_k5/rgcn/seed_29
```

R-GAT seed 7:

```bash
run_h3_config \
  configs/multiboard/h3_rgat_15x15_k5_seed7.json \
  results/h3_multiboard/15x15_k5/rgat/seed_7
```

R-GAT seed 17:

```bash
run_h3_config \
  configs/multiboard/h3_rgat_15x15_k5_seed17.json \
  results/h3_multiboard/15x15_k5/rgat/seed_17
```

R-GAT seed 29:

```bash
run_h3_config \
  configs/multiboard/h3_rgat_15x15_k5_seed29.json \
  results/h3_multiboard/15x15_k5/rgat/seed_29
```
CNN baseline:

```bash
python -m experiments.run_h3_pilot \
  --config configs/arena15_baseline.json \
  --output results/arena15_baseline/run1
```
Rgcn baseline
``` bash
python -m experiments.run_h3_pilot \
  --config configs/arena15_rgcn.json \
  --output results/arena15_rgcn/run1
```
Rgat baseline:
```bash
python -m experiments.run_h3_pilot \
  --config configs/arena15_rgat.json \
  --output results/arena15_rgat/run1
```

#### 7. Chạy toàn bộ ma trận tuần tự

Sau khi đã sinh config và khai báo `run_h3_config`, có thể thay các lệnh riêng
bằng hai vòng lặp dưới đây. Lệnh sẽ chạy hết bốn run 6×6 trước, rồi đến 12 run
10×10/15×15.

```bash
for model in rgcn rgat; do
  for seed in 17 29; do
    run_h3_config \
      "configs/phase6_${model}_seed${seed}.yaml" \
      "results/h3_multiboard/6x6_k4/${model}/seed_${seed}"
  done
done

for board_size in 10 15; do
  for model in rgcn rgat; do
    for seed in 7 17 29; do
      run_h3_config \
        "configs/multiboard/h3_${model}_${board_size}x${board_size}_k5_seed${seed}.json" \
        "results/h3_multiboard/${board_size}x${board_size}_k5/${model}/seed_${seed}"
    done
  done
done
```

#### 8. Resume một run

Phải dùng đúng config, đúng output và checkpoint mới nhất của chính run đó. Ví
dụ resume R-GAT 10×10/k=5 seed 17 từ iteration 20:

```bash
python -m experiments.run_h3_pilot \
  --config configs/multiboard/h3_rgat_10x10_k5_seed17.json \
  --output results/h3_multiboard/10x10_k5/rgat/seed_17 \
  --resume results/h3_multiboard/10x10_k5/rgat/seed_17/checkpoints/iter_020.pt \
  2>&1 | tee -a results/h3_multiboard/10x10_k5/rgat/seed_17/console.log
```

Không dùng checkpoint của seed, model hoặc kích thước bàn khác. Checkpoint là
immutable; resume từ checkpoint cũ hơn checkpoint mới nhất trong cùng output có
thể đụng tên file đã tồn tại.

Artifact mỗi run:

```text
config.json
model.pt
training_log.csv
checkpoints/manifest.json
checkpoints/iter_XXX.pt
```

## H1 ground truth

### Sinh H1 candidates schema v2 (Phase E-3a)

Generator mới lấy state mid/late từ self-play `mode=data`, chạy router theo thứ tự
exact solver gốc → VCF replay-verified → unknown, rồi ghi schema v2. Đây là tập
candidate bàn giao cho E-3b, chưa phải benchmark đóng băng:

```bash
python -m investigation.generate_h1_benchmark \
  --checkpoint results/h3_pilot_v2/rgat/seed_7/model.pt \
  --model-type rgat \
  --target 24 \
  --seed 7 \
  --output results/h1_integration/h1_candidates_v2.jsonl
```

Summary cạnh file JSONL ghi tỷ lệ `exact_complete / exact_partial / unknown` theo
board và ply bucket, cùng dedup mode. Unknown có thể được lưu để đo coverage nhưng
validator luôn loại khỏi denominator ground truth.

Budget calibration và production distribution trước E-3b nằm trong
`docs/reports/phase_e3a1.md`; measurement có cache/resume tại
`results/h1_integration/e3a1/` và chưa phải benchmark đóng băng.

Mở rộng gold-only 6×6 mid-game (E-3a.2), có cache/resume và tùy chọn thiết bị
cho riêng self-play inference:

```bash
python -m investigation.e3a2_expand_gold \
  --checkpoint results/h3_pilot_v2/rgat/seed_7/model.pt \
  --existing results/h1_integration/e3a1/production_candidates.jsonl \
  --output-dir results/h1_integration/e3a2 \
  --device cpu \
  --enhanced-proposer
```

`expanded_gold.jsonl` chỉ nhận `exact_complete` đã được solver gốc xác nhận;
enhanced solver không bao giờ tự cấp nhãn. Kết quả và nhánh quyết định cho E-3b
nằm trong `docs/reports/phase_e3a2.md`.

### E-3b: freeze và đánh giá endpoint iter 60

Pipeline bắt buộc chạy theo thứ tự evaluator → graph gates/SVG → immutable freeze →
endpoint evaluation:

```bash
python -m investigation.e3b_pipeline \
  --source results/h1_integration/e3a2/expanded_gold.jsonl \
  --output-dir results/h1_integration/e3b \
  --freeze-dir diagnostic/h1_benchmark_v1 \
  --rgat-checkpoint results/h3_pilot_v2/rgat/seed_7/checkpoints/iter_060.pt \
  --rgcn-checkpoint results/h3_pilot_v2/rgcn/seed_7/checkpoints/iter_060.pt \
  --mcts-playouts 50
```

Benchmark v1 đã freeze có hash trong `diagnostic/h1_benchmark_v1/manifest.json`;
không ghi đè file này. Báo cáo endpoint nằm trong `docs/reports/phase_e3b.md`.

### Developmental evaluation trên benchmark frozen

Chấm 13 checkpoint hiện có của mỗi model; MCTS chỉ chạy tại iter 0/20/40/60:

```bash
python -m investigation.developmental_evaluate \
  --benchmark diagnostic/h1_benchmark_v1/h1_benchmark_v1.jsonl \
  --manifest diagnostic/h1_benchmark_v1/manifest.json \
  --rgcn-run results/h3_pilot_v2/rgcn/seed_7 \
  --rgat-run results/h3_pilot_v2/rgat/seed_7 \
  --e3b-summary results/h1_integration/e3b/endpoint_summary.json \
  --output-dir results/h1_integration/developmental \
  --mcts-playouts 50
```

Runner chỉ inference checkpoint sẵn có, không train và không sinh state. Kết quả nằm
trong `docs/reports/developmental_evaluation.md`.

### Hai contract H1 không được trộn lẫn

Release khoa học hiện tại dùng benchmark v1 đã đóng băng gồm 94 exact states:

```text
diagnostic/h1_benchmark_v1/h1_benchmark_v1.jsonl
```

File sau là H1 legacy, chỉ giữ cho compatibility/reproduction lịch sử; nó không
phải benchmark v1 và không được dùng thay denominator 94-state:

```text
diagnostic/h1_tactical.jsonl
```

Generator hiện tại sinh candidate schema v2 và bắt buộc có checkpoint. Luôn ghi
ra working results, không ghi đè benchmark frozen:

```bash
python -m investigation.generate_h1_benchmark \
  --checkpoint results/h3_pilot_v2/rgat/seed_7/model.pt \
  --model-type rgat \
  --target 24 \
  --seed 7 \
  --output results/h1_integration/h1_candidates_v2.jsonl
```

Luồng H1 → Semantic KG → Evidence Overlay → Semantic XAI và toàn bộ gate tái lập
được mô tả tại [`docs/guides/semantic_xai_reproduction.md`](docs/guides/semantic_xai_reproduction.md).
Danh mục tài liệu đầy đủ nằm tại [`docs/README.md`](docs/README.md).

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
  --board-size 6 \
  --win-length 4 \
  --mode eval \
  --mcts-playouts 50 \
  --base-seed 7 \
  --game-index 0 \
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

### Arena cùng seed và cùng loại bàn cờ

Ma trận arena chỉ ghép R-GCN và R-GAT khi cả hai checkpoint có cùng training
seed, cùng `board_size` và cùng `win_length`. Mỗi cặp dùng năm arena game seed;
mỗi game seed được chạy hai lần để đổi bên đi trước, tổng cộng 10 ván:

- nhánh `rgcn_first`: R-GCN là player `1`, được đi trước;
- nhánh `rgat_first`: R-GCN là player `-1`, R-GAT được đi trước;
- mỗi ván dùng 100 MCTS playout;
- `mode=data`, temperature `0.5` trong 10 ply đầu và `0` sau đó để tạo nhiều
  trajectory; đây là arena stochastic phục vụ so sánh/giải thích, không phải
  strict deterministic strength evaluation;
- `base_seed` bằng training seed và `game_index` khác nhau cho từng ván.

CLI `game_export` nhận `--board-size` và `--win-length`. Nếu dùng `--state`, hai
tham số này phải khớp state; lệnh sẽ fail thay vì âm thầm đổi luật bàn cờ.

#### 1. Khai báo helper arena

Chạy trong WSL sau khi activate `.venv`:

```bash
set -euo pipefail

run_seed_matched_arena() {
  local board_size="$1"
  local win_length="$2"
  local seed="$3"
  local board_tag="${board_size}x${board_size}_k${win_length}"
  local rgcn_checkpoint="results/h3_multiboard/${board_tag}/rgcn/seed_${seed}/model.pt"
  local rgat_checkpoint="results/h3_multiboard/${board_tag}/rgat/seed_${seed}/model.pt"
  local arena_root="results/h3_multiboard/arena/${board_tag}/seed_${seed}/rgcn_vs_rgat_data_100p_10games"

  if [[ ! -f "$rgcn_checkpoint" ]]; then
    echo "Không tìm thấy R-GCN checkpoint: $rgcn_checkpoint" >&2
    return 1
  fi
  if [[ ! -f "$rgat_checkpoint" ]]; then
    echo "Không tìm thấy R-GAT checkpoint: $rgat_checkpoint" >&2
    return 1
  fi

  mkdir -p "$arena_root"

  for game_index in $(seq 1 5); do
    for model_player in 1 -1; do
      local side_tag
      local game_name
      local game_output

      if (( model_player == 1 )); then
        side_tag="rgcn_first"
      else
        side_tag="rgat_first"
      fi

      game_name=$(printf "game_%02d_%s" "$game_index" "$side_tag")
      game_output="$arena_root/$game_name"

      if [[ -e "$game_output" ]]; then
        echo "Từ chối ghi đè game đã tồn tại: $game_output" >&2
        return 1
      fi

      python -m azgomoku.explanation.game_export \
        --model rgcn \
        --checkpoint "$rgcn_checkpoint" \
        --opponent model \
        --opponent-model rgat \
        --opponent-checkpoint "$rgat_checkpoint" \
        --model-player "$model_player" \
        --board-size "$board_size" \
        --win-length "$win_length" \
        --mode data \
        --mcts-playouts 100 \
        --temperature 0.5 \
        --opening-temperature-moves 10 \
        --late-temperature 0 \
        --base-seed "$seed" \
        --game-index "$game_index" \
        --output "$game_output" \
        2>&1 | tee "$arena_root/${game_name}.log"
    done
  done
}
```

Helper kiểm tra đủ hai checkpoint và từ chối ghi đè từng game. Nếu một pairing
đã chạy dở, hãy kiểm tra game/output hiện có trước khi quyết định tiếp tục; không
xóa hoặc ghi đè tự động.

Mỗi move do `game_export` sinh tự động ba SVG cùng structured evidence, không cần
thêm flag export:

```text
game_XX_<side>/move_XXX/board.svg
game_XX_<side>/move_XXX/graph.svg
game_XX_<side>/move_XXX/decision.svg
game_XX_<side>/move_XXX/explanation.json
```

- `board.svg`: pre-move board và nước MCTS thực sự chọn;
- `graph.svg`: relational evidence tại đúng pre-move state;
- `decision.svg`: policy/value, prior, MCTS visits/Q và selected action.

#### 2. Arena 6×6/k=4

R-GCN seed 17 đấu R-GAT seed 17:

```bash
run_seed_matched_arena 6 4 17
```

R-GCN seed 29 đấu R-GAT seed 29:

```bash
run_seed_matched_arena 6 4 29
```

#### 3. Arena 10×10/k=5

R-GCN seed 7 đấu R-GAT seed 7:

```bash
run_seed_matched_arena 10 5 7
```

R-GCN seed 17 đấu R-GAT seed 17:

```bash
run_seed_matched_arena 10 5 17
```

R-GCN seed 29 đấu R-GAT seed 29:

```bash
run_seed_matched_arena 10 5 29
```

#### 4. Arena 15×15/k=5

R-GCN seed 7 đấu R-GAT seed 7:

```bash
run_seed_matched_arena 15 5 7
```

R-GCN seed 17 đấu R-GAT seed 17:

```bash
run_seed_matched_arena 15 5 17
```

R-GCN seed 29 đấu R-GAT seed 29:

```bash
run_seed_matched_arena 15 5 29
```

#### 5. Chạy toàn bộ tám pairing tuần tự

Tám pairing tạo tổng cộng 80 ván và export evidence cho mọi move, nên có thể cần
nhiều thời gian và dung lượng, đặc biệt trên 15×15.

```bash
run_seed_matched_arena 6 4 17
run_seed_matched_arena 6 4 29

for seed in 7 17 29; do
  run_seed_matched_arena 10 5 "$seed"
done

for seed in 7 17 29; do
  run_seed_matched_arena 15 5 "$seed"
done
```

#### 6. Export `knowledge.svg` v2 cho từng pairing

`knowledge.svg` không được tạo trong cùng model forward với ba SVG trên vì nó
cần solver/VCF hậu xử lý. Helper sau đọc arena đã hoàn tất và ghi một sidecar v2,
không sửa arena gốc. Mọi move đều có `knowledge.svg`: vị trí có replayed proof
nhận biểu đồ tactic-vs-attention, còn vị trí chưa có proof nhận notice SVG trung
thực thay vì overlay được suy đoán.

```bash
export_seed_matched_knowledge() {
  local board_size="$1"
  local win_length="$2"
  local seed="$3"
  local node_cap="${4:-1000000}"
  local time_cap_ms="${5:-2000}"
  local board_tag="${board_size}x${board_size}_k${win_length}"
  local arena_root="results/h3_multiboard/arena/${board_tag}/seed_${seed}/rgcn_vs_rgat_data_100p_10games"
  local knowledge_root="${arena_root}_knowledge_v2"
  local rgat_checkpoint="results/h3_multiboard/${board_tag}/rgat/seed_${seed}/model.pt"

  if [[ ! -f "$rgat_checkpoint" ]]; then
    echo "Không tìm thấy R-GAT checkpoint: $rgat_checkpoint" >&2
    return 1
  fi
  if [[ -e "$knowledge_root" ]]; then
    echo "Từ chối ghi đè knowledge sidecar: $knowledge_root" >&2
    return 1
  fi

  for game_index in $(seq 1 5); do
    for side_tag in rgcn_first rgat_first; do
      local game_name
      game_name=$(printf "game_%02d_%s" "$game_index" "$side_tag")
      if [[ ! -f "$arena_root/$game_name/game.json" ]]; then
        echo "Arena thiếu game hoàn tất: $arena_root/$game_name/game.json" >&2
        return 1
      fi
    done
  done

  python -m investigation.arena_knowledge \
    --arena "$arena_root" \
    --output "$knowledge_root" \
    --rgat-checkpoint "$rgat_checkpoint" \
    --node-cap "$node_cap" \
    --time-cap-ms "$time_cap_ms"
}
```

Export từng pairing:

```bash
export_seed_matched_knowledge 6 4 17
export_seed_matched_knowledge 6 4 29

export_seed_matched_knowledge 10 5 7
export_seed_matched_knowledge 10 5 17
export_seed_matched_knowledge 10 5 29

export_seed_matched_knowledge 15 5 7
export_seed_matched_knowledge 15 5 17
export_seed_matched_knowledge 15 5 29
```

Sau bước này, bộ bốn SVG cho một move được nối bằng cùng relative path:

```text
# Arena gốc
results/h3_multiboard/arena/<board_tag>/seed_<seed>/rgcn_vs_rgat_data_100p_10games/
  game_XX_<side>/move_XXX/{board.svg,graph.svg,decision.svg}

# Knowledge sidecar v2
results/h3_multiboard/arena/<board_tag>/seed_<seed>/rgcn_vs_rgat_data_100p_10games_knowledge_v2/
  game_XX_<side>/move_XXX/{knowledge.json,knowledge.svg,knowledge_evidence.json?}
```

`knowledge_evidence.json` chỉ có ở move có proof/rendered contrast; dấu `?` biểu
thị file tùy chọn. Budget mặc định của helper là 1.000.000 node và 2.000 ms mỗi
state. Có thể truyền budget khác ở đối số 4–5, ví dụ:

```bash
export_seed_matched_knowledge 6 4 17 100000000 600000
```

Không nên dùng budget rất lớn cho toàn bộ arena 10×10/15×15 trước khi đo thử một
pairing: time cap áp dụng cho từng state, không phải cho toàn câu lệnh.

### Export arena knowledge v2 không ghi đè arena gốc

Dùng `--output` để tạo sidecar tree version mới. Solver cache chỉ được reuse khi
`node_cap` và `time_cap_ms` trùng budget yêu cầu; đổi budget sẽ buộc solve lại.

```bash
arena_dir=results/h3_pilot/arena/rgcn_vs_rgat_data_100p_5games
knowledge_v2=${arena_dir}_knowledge_v2

python -m investigation.arena_knowledge \
  --arena "$arena_dir" \
  --output "$knowledge_v2" \
  --rgat-checkpoint results/h3_pilot_v2/rgat/seed_7/model.pt \
  --node-cap 100000000 \
  --time-cap-ms 600000
```

Mỗi move trong sidecar có `knowledge.json` và `knowledge.svg`; move có proof còn
có `knowledge_evidence.json` chứa raw per-edge R-GAT evidence. Manifest v2 lưu
selected action, actor/checkpoint, attention source/checkpoint, quan hệ
`actor|counterfactual` và SHA-256 của evidence. Trong SVG, `PROOF #n` là action
của certificate, còn viền đỏ `MCTS` là nước agent thực sự chọn. Lượt R-GCN dùng
R-GAT đối chứng được ghi rõ là `COUNTERFACTUAL R-GAT ATTENTION`.

## Cấu trúc source giữ lại

```text
azgomoku/       core game/graph/MCTS/solver cùng reusable semantic + metrics APIs
models/         R-GCN và R-GAT
experiments/    H3 pilot trainer
investigation/  CLI/orchestration và legacy reproduction; chỉ consume azgomoku
configs/        config train R-GCN/R-GAT
diagnostic/     legacy H1 và frozen H1 benchmark v1 (hai contract riêng)
semantic_kg/    frozen exact/certified semantic graph
semantic_evidence_v1/ frozen learned/search overlay (JSONL qua Git LFS)
tests/          correctness và regression suite
results/        ignored mặc định; chỉ release allowlist/checkpoint lineage được track
```

## Diễn giải kết quả

- Chuẩn vàng chính vẫn là exact 6×6/k=4.
- Kết quả threat solver trên bàn lớn chỉ áp dụng cho tập `tactically-decided` và phải báo coverage.
- R-GAT attention là model evidence, không phải causal proof.
- Arena cần nhiều seed và đổi bên đi trước; một hoặc hai game không đủ kết luận model mạnh hơn.
