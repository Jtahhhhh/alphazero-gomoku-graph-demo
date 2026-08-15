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
