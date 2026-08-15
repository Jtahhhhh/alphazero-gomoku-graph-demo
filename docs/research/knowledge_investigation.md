# KNOWLEDGE INVESTIGATION — arena `unknown`, ground-truth position và SVG coordinates

Ngày audit: 2026-08-14
Phạm vi: **audit-only**. Không sửa renderer, router, schema, benchmark, evaluator; không chạy lại train/eval. Phép tính mới duy nhất là gọi trực tiếp original exact solver trên hai state `unknown`, read-only, với trần `30,000 ms / 5,000,000 nodes`.

## Phán quyết ngắn

1. Hai nhãn `unknown` trong arena **không phải lỗi dữ liệu**. Cả hai state 6×6 có 30/28 ô trống, vượt calibrated exact bound `15`, nên router không gọi exact solver mà chuyển thẳng sang VCF. VCF đi hết không gian forcing của chính nó sau 5 nodes và trả `exhausted`; từ này **không có nghĩa full minimax đã được vét hết**.
2. Khi gọi original exact solver trực tiếp với trần rộng hơn, `a22cb411f8a95341` được giải chính xác là thua (`value=-1`) trong `3,883.495 ms`, 4,484 nodes; `00d4f0790640cc2a` vẫn timeout sau `30,000.656 ms`, 71,855 nodes. Vì vậy:
   - `a22…`: unknown do routing/budget/capability envelope của arena; nhãn có thể được giải nếu chủ động nới exact.
   - `00d…`: unknown đúng dưới cả budget arena và phép thử 30 giây; audit không đủ bằng chứng gọi nó là intrinsically unsolvable.
3. Với `5cc82addcbf6e2c1`, tập ô cam trong SVG **trùng khít** `valid_proofs[0].critical_cells`; không có bug `index → row,col → pixel` ở tầng cell. KN-B bị loại ở tầng này.
4. Nghi vấn “agent đánh vào ô đỏ/critical” đang trộn ba đại lượng:
   - chấm `P1` trong SVG là **proof-action marker**, không phải selected-action marker;
   - raw policy argmax là action `7`, trùng proof action/critical;
   - nước MCTS thực sự chọn là action `18`, **không** trùng critical.
5. Chữ T không do renderer bịa hướng. Record có **một VCF proof**, nhưng proof đó chứa **hai terminal threat windows thật**, ngang `[0,1,2,3]` và dọc `[2,8,14,20]`, giao tại action 2. Renderer vẽ đúng hai window của cùng một certificate; đây là double-threat certificate, không phải hai proof độc lập.
6. `e5478eda019150b4` không phải gold benchmark. Nó là arena `exact_partial`, `optimal_actions_complete=false`; badge PARTIAL là đúng.

## Q1 — Hai state 6×6 mang nhãn `unknown`

### 1.1 Arena record và đường routing thực tế

| State | Arena path | Empty | Arena budget | Router path | VCF result |
|---|---|---:|---:|---|---|
| `00d4f0790640cc2a` | `results/h3_pilot/arena/rgcn_vs_rgat_data/game_02/move_007/knowledge.json` | 30 | 2,000 ms / 1,000,000 nodes | skip exact (`30 > 15`) → VCF | `unknown`, `exhausted`, 5 nodes, 33.470 ms |
| `a22cb411f8a95341` | `results/h3_pilot/arena/rgcn_vs_rgat_data/game_02/move_009/knowledge.json` | 28 | 2,000 ms / 1,000,000 nodes | skip exact (`28 > 15`) → VCF | `unknown`, `exhausted`, 5 nodes, 27.904 ms |

Nguồn budget toàn arena: `results/h3_pilot/arena/rgcn_vs_rgat_data/knowledge_manifest.json`, trường `budget={node_cap:1000000,time_cap_ms:2000}`. Manifest có 62 move: 20 `exact_partial`, 42 `unknown`, 0 `exact_complete`.

Code path xác nhận:

- `azgomoku/ground_truth.py:59-76`: `DEFAULT_EMPTY_BOUNDS={6:15}`; `skip_exact = empty_count > 15`; original `solve_actions` chỉ được gọi khi không skip.
- `azgomoku/ground_truth.py:86-96`: sau exact/skip, router gọi VCF và phát `unknown` nếu không có replayed certificate.
- `azgomoku/vcf.py:335-338`: `exhausted` được đặt khi VCF không đụng time/node limit; đó là trạng thái của VCF search, không phải chứng nhận full-game draw/loss.
- `investigation/arena_knowledge.py:180-190`: CLI arena mặc định đúng `1,000,000 / 2,000 ms`.

### 1.2 Đo original exact solver trực tiếp

Lệnh logic đã chạy cho mỗi state:

```python
solve_actions(state, deadline_ms=30_000, node_budget=5_000_000)
```

| State | Status | Value | Optimal actions | Root action values | Nodes | Elapsed |
|---|---|---:|---|---|---:|---:|
| `00d4f0790640cc2a` | `timeout` | null | `[]` | `{}` | 71,855 | 30,000.656 ms |
| `a22cb411f8a95341` | `exact` | -1 | toàn bộ 28 legal actions | cả 28 action đều -1 | 4,484 | 3,883.495 ms |

Diễn giải:

- `a22…` là bằng chứng trực tiếp cho trường hợp **(a) router/budget/capability envelope**: arena cố ý không thử exact vì 28 empties; VCF không tìm thấy forcing win; exact rộng hơn sau đó chứng minh full loss. Với budget arena 2 giây, kết quả exact 3.88 giây cũng không được bảo đảm ngay cả nếu bỏ skip.
- `00d…` không cho phép kết luận “solver không thể giải”; chỉ được nói phép thử đã timeout ở 30 giây. Nó phù hợp trường hợp **(d) genuinely unknown under declared budget**, và vẫn unknown dưới phép đo rộng đã chọn.
- Không có bằng chứng trường hợp **(c) mismatch arena ↔ benchmark**, vì cả hai state không tồn tại trong frozen benchmark.

## Q2′ — State `5cc82addcbf6e2c1`: GT-position, policy và attention

Nguồn:

- solver/proof: `results/h3_pilot/arena/rgcn_vs_rgat_data/game_03/move_007/knowledge.json`
- policy/MCTS: `results/h3_pilot/arena/rgcn_vs_rgat_data/game_03/move_007/explanation.json`
- pixels: `results/h3_pilot/arena/rgcn_vs_rgat_data/game_03/move_007/knowledge.svg`

### 2.1 Critical indices → row/col → SVG cell

Convention được code xác nhận là `row,col = divmod(action, board_size)`:

- `azgomoku/explanation/rendering/knowledge_svg.py:13-15`
- `investigation/e3b_graph.py:60-62`
- `azgomoku/symmetry.py:28`

SVG arena đang audit là bản một-board cũ, board origin `(48,112)`, cell step `76`; rect top-left phải là `(48 + 76*col, 112 + 76*row)`.

| Index | `(row,col)` | Pixel kỳ vọng `(x,y)` | Rect SVG `(x,y)` | Match |
|---:|---|---|---|---|
| 1 | (0,1) | (124,112) | (124,112) | PASS |
| 2 | (0,2) | (200,112) | (200,112) | PASS |
| 7 | (1,1) | (124,188) | (124,188) | PASS |
| 8 | (1,2) | (200,188) | (200,188) | PASS |
| 14 | (2,2) | (200,264) | (200,264) | PASS |
| 15 | (2,3) | (276,264) | (276,264) | PASS |

Tập record `{1,2,7,8,14,15}` bằng đúng tập sáu `<g data-role="critical-cell">` trong SVG. **KN-B renderer-cell-coordinate bị bác bỏ.** Kết luận đúng, hẹp là “ô cam vẽ đúng theo proof record”; không nên mở rộng thành mọi loại ground truth đều đã được chứng minh tuyệt đối.

### 2.2 Ba đại lượng độc lập

| Đại lượng | Nguồn | Action / vị trí | Trùng critical? |
|---|---|---|---|
| Critical GT-position | `valid_proofs[0].critical_cells` | `{1:(0,1), 2:(0,2), 7:(1,1), 8:(1,2), 14:(2,2), 15:(2,3)}` | mốc |
| Proof action P1 | `valid_proofs[0].action` | `7` / `(1,1)` | Có |
| Network raw-policy argmax | `network.raw_policy_priors` | `7` / `(1,1)`, prior `0.260926068` | Có |
| Agent/MCTS selected action | `selected_move`, `mcts.selected` | `18` / `(3,0)`, 90/100 visits | **Không** |
| Rendered R-GAT attention | `knowledge.svg` generated from separate RGAT checkpoint | 36 diagonal edges; 0/12 exact proof-edge IDs; 8/36 chỉ chạm ít nhất một critical endpoint | Không align theo exact proof-edge criterion |

Chấm đỏ có chữ `P1` trong SVG nằm ở action 7 vì nó đánh dấu **proof số 1**. SVG không có `selected_move` marker. Vì vậy câu “agent đánh đúng ô CRITICAL” chỉ đúng nếu “agent” được dùng để chỉ **network policy-prior argmax**; nó sai nếu chỉ nước được arena/MCTS thực sự chơi.

### 2.3 Attention: điều gì xác minh được, điều gì không

Các scalar được SVG lưu cho state này là:

- `collapse_flag=0`
- normalized entropy `0.990`
- head diversity `0.021`
- `topology_corr=0.983`
- proof critical mass `0.050`

Các số này được tính từ raw RGAT evidence ngay trước render (`investigation/arena_knowledge.py:138-143`). Chúng ủng hộ kết luận attention có critical mass thấp và correlation cao với topology, trong khi collapse flag tắt.

Tuy nhiên raw per-edge RGAT evidence không được lưu vào `knowledge.json`; `explanation.json` ở move này thuộc active R-GCN (`attention_available=false`). SVG cũ chỉ còn 36 rendered attention lines và tất cả đều mang `data-attention="1.000000000"`. Do đó **không thể tái dựng một thứ hạng top-k thật, độc lập, chỉ từ các artifact đã lưu**. Có thể audit vị trí 36 nét và exact proof-edge overlap, nhưng không thể chứng minh lại thứ tự top-k mà không rerun inference. Audit này không rerun eval/inference theo scope.

Phán quyết KN:

> **KN-B bị loại. KN-A đúng ở nghĩa hẹp “GT-position đúng + raw policy argmax trúng tactic + attention metric lệch”, nhưng không đúng nếu diễn đạt “nước agent đã chơi trúng tactic”: MCTS đã chơi action 18, không phải action 7.**

### 2.4 Chữ T: một proof, hai window

Record có `proof_count=1`, action 7, concepts `winning_line + forced_sequence + vcf`, relations `horizontal + vertical`, và đúng hai windows:

1. horizontal `[0,1,2,3]`
2. vertical `[2,8,14,20]`

Hai window giao tại action 2. Certificate OR/AND kết thúc ở `terminal="unstoppable_double_threat"`; chuỗi nước phía attacker trong certificate là 7, 8, 14, 15, 1, 2 — đúng tập critical cells. Vì vậy hình chữ T là hai terminal threat windows của **một forced-sequence proof**, không phải renderer nhân đôi một winning line. Nó hợp lệ nhưng nhãn gộp dễ khiến người xem tưởng có hai proof; two-board/caption rõ proof-count sẽ giảm nhầm lẫn.

## Q2 — Audit graph edge → pixel trên `e5478eda019150b4`

Lưu ý: state này là arena partial, không phải frozen gold. Vẫn dùng nó để kiểm tra đúng image được nêu; ngoài ra D4 gate gold được đánh giá riêng ở mục sau.

Board origin cũ `(48,112)`, center formula `(48+(col+0.5)*76, 112+(row+0.5)*76)`.

| Edge ID | Source `(action,row,col)` | Target `(action,row,col)` | Pixel kỳ vọng | Pixel trong SVG | Match |
|---|---|---|---|---|---|
| `diagonal_down:15:22` | 15,(2,3) | 22,(3,4) | (314,302)→(390,378) | (314,302)→(390,378) | PASS |
| `diagonal_down:15:8` | 15,(2,3) | 8,(1,2) | (314,302)→(238,226) | (314,302)→(238,226) | PASS |
| `diagonal_down:1:8` | 1,(0,1) | 8,(1,2) | (162,150)→(238,226) | (162,150)→(238,226) | PASS |
| `diagonal_down:22:15` | 22,(3,4) | 15,(2,3) | (390,378)→(314,302) | (390,378)→(314,302) | PASS |
| `diagonal_down:8:1` | 8,(1,2) | 1,(0,1) | (238,226)→(162,150) | (238,226)→(162,150) | PASS |

Critical action 22 map thành `(3,4)` và rect top-left `(352,340)`, đúng SVG. Window `[1,8,15,22]` là một đường `diagonal_down` liên tục; cả sáu directed proof edges được vẽ đúng endpoints.

### Khoảng trống của D4 gate

`results/h1_integration/e3b/graph_gate.json` vẫn PASS `1,944/1,944`, gồm 243 proofs × 8 D4 round-trips. Nhưng `investigation/e3b_graph.py:45-83` chỉ kiểm tra:

- proof action thuộc legal/optimal/graph node;
- `action ↔ row,col` round-trip;
- critical cell nằm trong graph;
- proof/action round-trip qua 8 symmetry.

Gate không parse SVG và không so `edge_id/source/target` với `x1,y1,x2,y2`; vì vậy nó **không phủ graph↔SVG-pixel**. Các test hiện tại kiểm proof count/concept/badge/export presence (`tests/test_e3b_pipeline.py:98-134`, `tests/test_explanation.py:57-75`), chưa có assertion pixel-coordinate cụ thể. Audit thủ công trên 5 cell + 5 edges đều PASS, nhưng nên coi test coord-render là coverage còn thiếu, không phải bug đã thấy.

## Q3 — Badge của `e5478eda019150b4`

Frozen `diagnostic/h1_benchmark_v1/h1_benchmark_v1.jsonl` có 94 records và **không chứa** `e5478eda019150b4` (cũng không chứa ba state còn lại trong audit). Arena record của `e547…` ghi:

- `solver.status = exact_partial`
- `solver.method = vcf`
- `optimal_actions = [22]`
- `optimal_actions_complete = false`
- coverage: replay-verified existential VCF win; proven action set incomplete
- one proof, action 22, critical cell 22, window `[1,8,15,22]`

Vì thế badge `PARTIAL · PARTIAL KNOWLEDGE` là đúng. Nó không được phép mang GOLD chỉ vì có một exact VCF certificate: certificate chứng minh tồn tại một nước thắng, không chứng minh toàn bộ optimal-action set.

Ở state này, MCTS selected action là 31 `(5,1)` và raw-policy argmax là 2 `(0,2)`; cả hai đều khác proof action 22 `(3,4)`. Đây cũng là ví dụ cho việc không đồng nhất proof marker với nước agent.

## Ảnh hưởng tới benchmark và phát hiện H1/H3

- Frozen benchmark SHA-256 đo lại: `9abd52ef4991489586682e881e495fcb4c2ffe00fb55dc9dee1d9008aca4ff02`, khớp `diagnostic/h1_benchmark_v1/manifest.json`.
- Benchmark vẫn 94/94 `exact_complete`, 83 proof-bearing, 243/243 replayed proofs. Bốn state audit đều không thuộc benchmark.
- E-3b late R-GAT mean `attention_topology_correlation = 0.9687206584` (thường làm tròn 0.969/0.97), nguồn `results/h1_integration/e3b/e3b_summary.json`; phép audit renderer/arena này không đi qua evaluator và không thay số đó.
- Không có căn cứ vứt ground truth hay topology result. Kết luận cần sửa về ngôn ngữ là: **GT-position được vẽ đúng ở các sample đã kiểm; policy-prior, MCTS-selected action, proof action và attention là bốn đối tượng khác nhau.**

## Hành động đề xuất sau audit (không thực hiện trong task này)

1. Giữ `unknown` cho arena dưới budget công bố; nếu cần coverage cao hơn, tạo một experiment/version mới thử exact trước ở 6×6 early/mid với budget được công bố, không overwrite cache/benchmark hiện tại.
2. Trong SVG/caption, đổi marker `P1` thành wording rõ `PROOF P1`; nếu muốn hiện nước agent thì thêm marker riêng `MCTS selected`, không tái dùng màu/biểu tượng proof.
3. Persist raw RGAT per-edge evidence hoặc hash/reference của evidence trong artifact manifest để top-k có thể audit mà không rerun model.
4. Thêm coord-render test: parse SVG và assert `edge_id → divmod → pixel` cùng `critical_cells → rect pixel`. D4 gate hiện tại không thay thế test này.
5. Với two-board v2, rerender arena thành artifact version mới và ghi `attention_top_k` trong manifest; không dùng file một-board cũ làm bằng chứng top-k.

## Tính toàn vẹn của audit

Không file benchmark/result/code nào được sửa bởi audit. Chỉ báo cáo này được tạo. Phép direct solve không ghi cache và không được dùng để relabel arena.
