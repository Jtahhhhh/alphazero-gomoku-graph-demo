# Semantic KG Phase 5 — Semantic Interpretability Experiments

Ngày chạy: 2026-08-14

## Kết luận

Phase 5 hoàn tất toàn bộ gate phương pháp và sinh đủ artifact bắt buộc. Semantic
evaluator tái lập evaluator P7 cũ trước khi chạy decomposition mới.

Trạng thái: **PASS về phương pháp**. Kết quả chính tiếp tục là negative/mixed,
không bị coi là failure.

## Reproduction gate

So sánh endpoint R-GAT trên cùng 94 frozen states và iter60 checkpoint:

| Metric | Maximum absolute delta |
|---|---:|
| graph critical mass | 0 |
| graph precision@k | 0 |
| graph recall@k | 0 |
| graph AUPRC | 0 |
| policy optimal mass | 3,14e-7 |
| value prediction | 1,19e-7 |
| MCTS optimal mass | 4,21e-7 |

Alignment tolerance là `1e-12`; network/search tolerance là `1e-6`. Tất cả 614
comparisons pass. Semantic evaluator vì vậy là decomposition của ground truth P7,
không phải một target mới thay thế P7.

## Denominator discipline

- exact policy/value/search: 94 states
- certified semantic alignment: 83 proof-bearing states, 243 proofs
- 11 no-proof states: excluded/not applicable, không score bằng 0
- ForcedResponse: applicable ở 55 states / 72 proofs
- BlockingMove: applicable ở 54 states / 71 proofs

Mỗi proof được score riêng. Report giữ hai đại lượng độc lập:

- existential alignment: max qua các valid proof applicable
- coverage alignment: mean qua các valid proof applicable

## Metric contract

Edge targets (`WinningThreat`, `TacticalLineWindow`, proof geometry) dùng attention
mass, AUPRC, rank và top-k recall trên directed StructuralEdge.

Node targets (`CompletionCell`, `ForcedResponse`, `BlockingMove`) dùng convention
đã khai báo trước: **incoming final-layer mean-head attention mass**. Không thử
nhiều aggregation rồi chọn hậu nghiệm.

Structural baseline giữ P7 indegree convention. Matched-random regions giữ cùng
cardinality với semantic target. Bootstrap là state-level nonparametric bootstrap,
1.000 replicates; không giả định game-level independence không có trong provenance.

## E5.1–E5.3 — Endpoint semantic alignment

| Semantic type | n states | n proofs | Existential | Coverage | Coverage − structural | Coverage − random |
|---|---:|---:|---:|---:|---:|---:|
| WinningThreat | 83 | 243 | 0,04638 | 0,03904 | -0,000199 | -0,002560 |
| TacticalLineWindow | 83 | 243 | 0,03527 | 0,03086 | -0,000166 | -0,002476 |
| CompletionCell | 83 | 243 | 0,05207 | 0,04501 | ~0 | +0,001104 |
| ForcedResponse | 55 | 72 | 0,02883 | 0,02883 | ~0 | +0,000509 |
| BlockingMove | 54 | 71 | 0,02831 | 0,02831 | ~0 | +0,000537 |
| ProofSupportedGeometry | 83 | 243 | 0,03527 | 0,03086 | -0,000166 | -0,002439 |

95% state-bootstrap CI cho excess-over-structural:

- WinningThreat: `[-0,000272; -0,000127]`
- proof geometry: `[-0,000240; -0,000096]`
- CompletionCell: `[-2,37e-11; 4,83e-11]`
- ForcedResponse: `[-5,37e-11; 2,83e-11]`
- BlockingMove: `[-5,66e-11; 2,74e-11]`

Node semantic targets không cho enrichment vượt structural baseline; edge tactical
targets thấp hơn structural baseline một lượng nhỏ nhưng CI không cắt 0.

`TacticalLineWindow` và `ProofSupportedGeometry` có cùng giá trị trong v1 vì các
source-supported proof windows hiện collapse đúng về cùng legacy critical-edge
region. Chúng được giữ thành hai lineage/view riêng để minh bạch, nhưng không được
diễn giải như hai kết quả thống kê độc lập.

### Certified-vs-structural contrast

Các region A/B/C được match cardinality trên 83 proof-bearing states:

- A, certified-supporting proof region: mean mass `0,04981`
- B, strongest derived structural non-proof region: `0,08417`
- C, matched random non-proof region: `0,05413`
- A − B: `-0,03435`, 95% CI `[-0,03841; -0,03033]`
- A − C: `-0,00432`, 95% CI `[-0,00587; -0,00290]`

Đây là evidence rằng endpoint R-GAT attention phân bổ theo structural region mạnh
hơn certified-supporting geometry trong protocol quan sát hiện tại.

## E5.4 — Developmental semantic trajectory

Late-phase R-GAT trajectory cho proof geometry:

- policy optimal mass: `0,49713` tại iter0 → `0,51482` tại iter60
- topology correlation: `0,99999` → `0,96872`
- hard-collapse rate: `1,0` → `0,0`
- proof-geometry coverage: `0,033084` → `0,032939`
- excess over structural: `-0,0000023` → `-0,0001475`

WinningThreat coverage tương tự: `0,040129` → `0,039944`. Completion/response/
blocking node mass gần như bất biến và tiếp tục trùng structural baseline.

Kết quả đo được không cho thấy semantic tactical alignment tăng cùng competence.
Attention thoát hard-collapse nhưng vẫn topology-dominant và không chuyển thành
proof-semantic enrichment ở endpoint.

## E5.5 — Semantic search lift

`MCTS probability mass − network policy probability mass`, 50 playouts, endpoint:

| Category | R-GAT mean [95% CI] | R-GCN mean [95% CI] |
|---|---:|---:|
| exact-optimal | +0,3141 `[0,2404; 0,3908]` | +0,2764 `[0,2022; 0,3547]` |
| proof-supported | +0,4070 `[0,3301; 0,4878]` | +0,3546 `[0,2783; 0,4343]` |
| blocking | +0,1727 `[0,0950; 0,2547]` | +0,1213 `[0,0530; 0,1894]` |
| forced-response | +0,1695 `[0,0933; 0,2583]` | +0,1189 `[0,0522; 0,1928]` |
| threat-creating | -0,3537 `[-0,4313; -0,2788]` | -0,3012 `[-0,3836; -0,2236]` |

MCTS tập trung probability mass vào exact-optimal/proof-supported/forced defensive
moves ở budget cố định, đồng thời lấy mass khỏi tập threat-creating rộng. Các
category có thể chồng lấp; kết quả chỉ mô tả redistribution quan sát được, không
chứng minh search “hiểu” semantic category hoặc có quan hệ nhân quả.

## MEASURED

- P7 legacy reproduction trên cùng endpoint checkpoint và frozen states.
- Component-wise attention alignment cho sáu source-supported views.
- Per-proof, existential và coverage metrics.
- Structural/matched-random baselines và certified-vs-structural contrast.
- 13-checkpoint R-GAT semantic trajectory.
- 50-playout semantic search lift tại iter 0/20/40/60 cho R-GAT và R-GCN.
- State-level 95% bootstrap CI với denominators explicit.

## SUPPORTED WITH CONDITIONS

- Endpoint attention ưu tiên structural non-proof regions hơn certified-supporting
  proof geometry trong đúng graph/attention protocol hiện tại.
- Không có evidence về sự xuất hiện dần của proof-semantic attention alignment dù
  competence tăng nhẹ và hard collapse biến mất.
- Fixed-budget MCTS tăng mass trên exact-optimal/proof-supported/defensive categories.
- Các phát biểu trên là observational, checkpoint/budget-specific và không causal.

## NOT MEASURED

- MCTS budget ablation hoặc causal intervention.
- Generalization sang board size, win length, seed, architecture hoặc benchmark khác.
- Game-clustered uncertainty; provenance hiện không có independent game grouping đủ
  bảo vệ cho giả định đó.
- Generic Pattern/OpenEnd ontology.
- Causal faithfulness của attention hoặc proof generation ngoài frozen H1.
- Statistical independence giữa semantic categories đang chồng lấp.

## Artifacts

`results/semantic_xai/` chứa:

- `endpoint_semantic_metrics.csv`
- `endpoint_semantic_summary.json`
- `semantic_by_type.csv`
- `certified_vs_structural.csv`
- `developmental_semantic_metrics.csv`
- `developmental_semantic_analysis.json`
- `semantic_search_lift.csv`
- `bootstrap_ci.json`
- `phase5_release_gate.json`
- bốn paper-ready SVG trong `figures/`

Mọi row metric chứa state/proof/checkpoint, base KG manifest SHA, evidence manifest
SHA và semantic target lineage hash khi applicable.

## Xác minh phát hành

- Legacy reproduction gate: PASS ở tolerance `1e-6`.
- Phase 5 release gate: PASS; đủ `94` exact states, `83` proof-bearing states và
  `243` proofs, với `11` states không có proof được khai báo rõ.
- Bốn SVG đã được raster-render và kiểm tra trực quan: không có clipping, overlap
  hoặc lỗi parse.
- Semantic/evidence tests: `26 passed`; toàn bộ repository: `114 passed` với
  UTF-8 mode.
