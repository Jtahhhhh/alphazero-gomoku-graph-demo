# Phase E-3b — H1 endpoint integration

Ngày chạy: 2026-08-14  
Endpoint: iteration 60, seed 7, R-GCN và R-GAT  
Benchmark SHA-256: `9abd52ef4991489586682e881e495fcb4c2ffe00fb55dc9dee1d9008aca4ff02`

## Kết luận chính

Trên 6×6 late-game, R-GAT **không vượt đồng thời** structural và random baseline về graph alignment. Critical mass của R-GAT là 0.03294, structural 0.03309 và random 0.03512; AUPRC của R-GAT là 0.04374, structural 0.04342 và random 0.05683. Vì vậy kết luận quan sát ở E-3b là:

> Learned R-GAT attention không cho thấy alignment mạnh hơn các baseline trên proof-bearing 6×6 late-game (n=63/71).

Đây là tương quan, không phải kết luận nhân quả. R-GCN alignment đúng bằng structural baseline theo thiết kế và **không phải một phát hiện**.

Mid-game có n=23 gold, dưới `N_min=30`; mọi số mid chỉ là **gợi ý, không kết luận**. Pattern mid cũng không cho thấy R-GAT vượt cả hai baseline.

## Bước 4 — Evaluator fail-closed

- Gold policy/value/MCTS: 94 `exact_complete` 6×6, gồm late n=71 và mid n=23.
- Partial/unknown trong denominator chính: 0.
- Proof annotation replay-verified: 243/243 PASS trên 83/94 gold.
- Alignment denominator riêng: late n=63/71, mid n=20/23. State thiếu proof bị loại khỏi alignment, không bị ghi thành 0.
- Partial 6×6 n=96 chỉ tham khảo, không vào metric chính.
- Bàn lớn chỉ kiểm chứng pipeline: 10×10 partial 19/19 replay PASS; 15×15 partial 20/20 replay PASS; không xuất bảng metric.

## Policy, value và MCTS

Các số là mean theo model × phase; MCTS dùng 50 playouts.

| Phase | Model | Policy top-1 | Optimal mass | Value error | MCTS top-1 | MCTS optimal mass | Search gain |
|---|---|---:|---:|---:|---:|---:|---:|
| Late, n=71 | R-GCN | 46.48% | 0.4972 | 0.9221 | 88.73% | 0.8113 | +0.3141 |
| Late, n=71 | R-GAT | 47.89% | 0.5148 | 0.8793 | 97.18% | 0.8808 | +0.3660 |
| Mid, n=23, gợi ý | R-GCN | 60.87% | 0.5329 | 1.0207 | 69.57% | 0.6930 | +0.1601 |
| Mid, n=23, gợi ý | R-GAT | 65.22% | 0.5620 | 1.0505 | 78.26% | 0.7157 | +0.1537 |

Mô tả endpoint: R-GAT cao hơn R-GCN ở policy và MCTS late; value error late thấp hơn. Đây là so sánh mô tả kiến trúc tại một endpoint, chưa có CI/ablation nên không diễn giải nhân quả.

## Graph alignment — chỉ R-GAT là đối tượng nghiên cứu

| Phase | Metric | R-GAT | Structural | Random |
|---|---|---:|---:|---:|
| Late, proof n=63 | Critical mass | 0.03294 | 0.03309 | 0.03512 |
| Late, proof n=63 | Precision@K / Recall@K | 0.03571 | 0.03571 | 0.03449 |
| Late, proof n=63 | AUPRC | 0.04374 | 0.04342 | 0.05683 |
| Mid, proof n=20, gợi ý | Critical mass | 0.02433 | 0.02455 | 0.02682 |
| Mid, proof n=20, gợi ý | Precision@K / Recall@K | 0.00417 | 0.00417 | 0.02695 |
| Mid, proof n=20, gợi ý | AUPRC | 0.04825 | 0.05958 | 0.04752 |

Một vài metric dao động rất nhỏ quanh structural/random, nhưng R-GAT không thắng nhất quán. Không có cơ sở nói attention đã học tactical proof alignment ở endpoint này.

## Collapse và no-alignment

Operational hard-collapse: normalized attention entropy ≥0.98, structural MAE ≤0.02 và head diversity ≤0.02.

| Phase | Hard-collapse rate | Norm. entropy | Structural MAE | Head diversity | Topology correlation |
|---|---:|---:|---:|---:|---:|
| Late, n=71 | **0%** | 0.9828 | 0.0307 | 0.0259 | 0.9687 |
| Mid, n=23, gợi ý | **13.04%** | 0.9897 | 0.0231 | 0.0197 | 0.9808 |
| Tổng, n=94 | **3.19%** | 0.9845 | 0.0289 | 0.0244 | — |

Late không bị hard-collapse theo ngưỡng đã khóa, nên flat alignment không thể quy hoàn toàn cho uniform collapse. Tuy vậy entropy và correlation với topology đều rất cao: attention vẫn gần structural/topological pattern. Kết luận thận trọng là **hard-collapse hiếm, topology dominance cao, learned attention không cho thấy proof alignment**. Causal edge/relation/attention ablation mới có thể phân biệt nguyên nhân; ablation không thuộc E-3b.

## Bước 5 — Graph/SVG gates

- 243 proof × 8 D4 transforms = **1,944/1,944 round-trip PASS**.
- Action/policy/graph joins: **1,944/1,944 PASS**.
- MCTS root children = legal actions trên state đại diện mid và late.
- SVG chỉ được sinh sau khi các gate trên xanh.

State đại diện:

- Mid: `42be4c9c478fcf87`.
- Late: `4dca2566ec2be9b6`.

Mỗi SVG đặt cùng board/proof overlay cạnh R-GAT learned attention và R-GCN structural baseline.

## Bước 6 — Benchmark freeze

- Frozen file: `diagnostic/h1_benchmark_v1/h1_benchmark_v1.jsonl`.
- Manifest: `diagnostic/h1_benchmark_v1/manifest.json`.
- Nội dung: 94 exact-complete 6×6, late 71 / mid 23.
- Exact solver agreement trước freeze: **94/94, mismatch 0** với gate 30 giây/5M node.
- Proof replay: **243/243 PASS**.
- D4 dedup; schema v2; BUDGET*=2 giây; file và manifest được đánh dấu read-only.
- Mọi thay đổi tiếp theo phải tạo `h1_benchmark_v2`, không ghi đè v1.

## Bước 7 — Endpoint evaluation

- R-GCN iter 60 SHA-256: `a9583a8bad6eb80f99398bc359428cd145849254dbfb7ea1a8e04e413e9407cc`.
- R-GAT iter 60 SHA-256: `897e41795fa2ff26e0355378cf8a5167b375d14ef03f185ae2be39fb7c1c6286`.
- 188/188 hàng metric có benchmark hash đúng.
- Full regression suite: **80 passed**.
- Developmental 13 checkpoint, MCTS ablation, causal ablation và arena nhiều ván chưa chạy vì nằm ngoài E-3b.

## Artifacts

- `diagnostic/h1_benchmark_v1/h1_benchmark_v1.jsonl`: benchmark immutable.
- `diagnostic/h1_benchmark_v1/manifest.json`: hash và provenance freeze.
- `results/h1_integration/e3b/endpoint_metrics.csv`: per-state metrics.
- `results/h1_integration/e3b/endpoint_summary.json`: model × phase summary.
- `results/h1_integration/e3b/evaluator_gate.json`: denominator/proof coverage gate.
- `results/h1_integration/e3b/graph_gate.json`: D4/action/MCTS coordinate gates.
- `results/h1_integration/e3b/freeze_exact_gate.json`: exact agreement gate.
- `results/h1_integration/e3b/figures/`: SVG mid/late comparison.
