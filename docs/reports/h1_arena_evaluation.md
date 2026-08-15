# Báo cáo tổng hợp Arena và H1 — R-GCN so với R-GAT

**Ngày đánh giá:** 13/08/2026  
**Checkpoint:** `results/h3_pilot_v2/{rgcn,rgat}/seed_7`  
**Phạm vi:** kiểm tra nguồn sinh ván, độ đa dạng arena, mức đóng góp của policy/value network, MCTS và graph evidence.

## 1. Câu hỏi nghiên cứu

1. Sau khi sửa thủ tục chọn nước, các ván arena có còn lặp tất định không?
2. Policy của model đã định hướng search hay MCTS vẫn phải gánh toàn bộ quyết định?
3. Graph evidence, đặc biệt attention của R-GAT, đã align với tactical proof chưa?
4. Kết quả đối đầu arena có nhất quán với benchmark tactical exact H1 không?

## 2. Source và vai trò

| Source | Vai trò |
|---|---|
| `azgomoku/explanation/game_export.py` | Sinh ván model-vs-model và lưu evidence từng nước. `eval` chọn greedy từ visit count; `data` sample từ phân phối visit riêng. |
| `azgomoku/mcts.py` | MCTS gốc. Không bị sửa trong thay đổi arena. |
| `azgomoku/explanation/explanation_export.py` | Ghi network output, MCTS trace và graph evidence của một state trước khi đi nước. |
| `azgomoku/explanation/mcts_trace.py` | Chuẩn hóa `P`, `N`, `Q`, `pi`; giữ `mcts_value_convention_version = 2`. |
| `diagnostic/h1_tactical.jsonl` | Benchmark H1 cố định gồm 24 state có ground truth exact. |
| `investigation/h3_evaluate.py` | Đánh giá policy, value, MCTS và graph alignment trên toàn bộ checkpoint. |
| `docs/contracts/h1_correctness.md` | Quy ước `V*(s)` theo player-to-move, tập đầy đủ optimal actions và yêu cầu solver exact. |
| `tests/test_game_export.py` | Kiểm tra eval greedy, data sampling, seed theo ván và metadata. |
| `tests/test_explanation.py` | Kiểm tra MCTS convention, Q perspective và evidence export. |

### 2.1. Thay đổi arena

Lỗi ban đầu nằm sau search: dù đã tính policy theo temperature, code vẫn chọn `argmax(pi)`. Vì phép biến đổi temperature không đổi phần tử lớn nhất, nhiều lần chạy cùng checkpoint tạo cùng đường ván.

Source hiện tại tách hai mục đích:

- `mode=eval`: chọn `argmax(N)`, phục vụ đánh giá sức mạnh tất định.
- `mode=data`: sample theo `N^(1/tau)`, phục vụ phân bố state cho H1/D2c.
- Seed mỗi ván được dẫn xuất từ `(base_seed, game_index)`: cùng cặp tham số tái tạo cùng ván, khác `game_index` tạo luồng RNG độc lập.
- Search và evidence vẫn đọc cùng MCTS root; không thêm Dirichlet noise và không sửa math của MCTS.

## 3. Dữ liệu và protocol

### 3.1. Arena data mới

- Hai model: R-GCN đi trước, R-GAT đi sau.
- Checkpoint: iteration 60, seed training 7.
- Số ván: 5.
- MCTS: 100 playouts/nước.
- Selection: `mode=data`, `tau=1` trong 10 ply đầu, sau đó greedy.
- Base seed: 8; game index: 1–5.
- Output: `results/h3_pilot/arena/rgcn_vs_rgat_data`.

### 3.2. H1

- 24 state tactical exact, không phụ thuộc đường ván arena.
- 13 checkpoint/model: iteration 0, 5, ..., 60.
- Tổng cộng 312 dòng metric/model.
- MCTS 50 playouts tại các iteration đăng ký: 0, 20, 40, 60.
- Output tái đánh giá: `results/h1/recheck_20260813`.

Các metric chính:

- `policy_top1_correct`: top-1 policy thuộc tập optimal actions exact.
- `policy_optimal_mass`: tổng xác suất policy trên các optimal actions.
- `value_error`: sai số tuyệt đối với solver value.
- `mcts_selected_optimal`: nước greedy sau MCTS có optimal không.
- `mcts_optimal_mass`: visit mass trên các optimal actions.
- `search_gain = mcts_optimal_mass - policy_optimal_mass`.
- Graph alignment được đối chiếu với valid tactical proofs, random baseline và structural baseline.

## 4. Kết quả arena

| Ván | Winner | Độ dài |
|---|---|---:|
| game 01 | R-GCN | 9 |
| game 02 | R-GAT | 14 |
| game 03 | R-GCN | 9 |
| game 04 | R-GCN | 11 |
| game 05 | R-GCN | 11 |

- Ván phân biệt: **5/5 (100%)**.
- Trước sửa: ba ván `game_04–06` trùng hoàn toàn; trong năm ván `game_02–06` chỉ có 3 chuỗi phân biệt.
- Độ dài trung bình mới: **10,8 nước**, khoảng 9–14.
- Kết quả thô: R-GCN thắng 4, R-GAT thắng 1.

Tỷ số 4–1 chưa phải phép so sức mạnh công bằng vì R-GCN luôn đi trước, chỉ có năm ván và 10 ply đầu có sampling.

### 4.1. Policy định hướng MCTS trong arena

| Model | Policy top-1 = MCTS top-1 | MCTS top-1 trong policy top-3 | Hạng policy TB của MCTS top-1 | TV(P,N) |
|---|---:|---:|---:|---:|
| R-GCN | **75,9%** | **93,1%** | **1,52** | 0,303 |
| R-GAT | **52,0%** | **96,0%** | **1,68** | 0,172 |

Policy không vô dụng: 93–96% nước được MCTS xếp đầu vốn đã nằm trong top-3 của network. R-GCN thường xếp đúng top-1 hơn; R-GAT thường khoanh đúng nhóm ứng viên nhưng cần MCTS sắp hạng lại.

## 5. Kết quả H1 checkpoint cuối

| Metric | R-GCN iter 0 | R-GCN iter 60 | R-GAT iter 0 | R-GAT iter 60 |
|---|---:|---:|---:|---:|
| Policy top-1 correct | 37,5% | **54,2% (13/24)** | 37,5% | **66,7% (16/24)** |
| Policy optimal mass | 0,343 | **0,466** | 0,342 | **0,540** |
| Value MAE | 0,827 | **0,651** | 0,969 | **0,604** |
| MCTS top-1 correct | 95,8% | **100% (24/24)** | 100% | **100% (24/24)** |
| MCTS optimal mass | 0,860 | **0,933** | 0,844 | **0,918** |
| Search gain | +0,518 | **+0,467** | +0,502 | **+0,378** |

### 5.1. Diễn tiến học

- R-GCN policy tăng từ 9/24 lên 13/24, đạt plateau 13/24 từ iteration 30.
- R-GAT policy tăng từ 9/24 lên 16/24; đỉnh 17/24 tại iteration 45.
- Optimal mass và value của cả hai tiếp tục cải thiện đến iteration 60.
- Tại iteration 60, policy R-GCN sai 11 state và R-GAT sai 8 state; MCTS sửa đúng toàn bộ các quyết định top-1.

## 6. Graph alignment tại iteration 60

| Metric | R-GCN | R-GAT |
|---|---:|---:|
| Graph critical mass | 0,03121 | 0,03105 |
| Random critical mass | 0,03441 | 0,03441 |
| Structural critical mass | 0,03121 | 0,03121 |
| Alignment minus random | **−0,00320** | **−0,00336** |
| Alignment minus structural | **0,00000** | **−0,00016** |
| Graph AUPRC | 0,04718 | 0,04468 |

R-GCN dùng structural score cố định trong evaluator, nên alignment bằng structural baseline và không thay đổi theo training. Attention R-GAT không vượt random hoặc structural baseline; AUPRC gần như phẳng từ 0,0442 ở iteration 0 đến 0,0447 ở iteration 60.

Do đó chưa có bằng chứng rằng attention edges đã học cách tập trung vào quan hệ thuộc tactical proof. Kết quả này không phủ định khả năng graph architecture giúp representation/policy; nó chỉ bác bỏ diễn giải mạnh rằng attention hiện tại là tactical explanation đã được xác nhận.

## 7. Đối chiếu Arena và H1

Arena và H1 đo hai lớp khác nhau:

- Arena cho thấy model đã định hướng search: MCTS-top1 hầu như nằm trong policy top-3.
- H1 exact cho thấy policy-only vẫn chưa đủ tin cậy: top-1 chỉ đạt 54–67%, trong khi MCTS đạt 100%.
- Vì vậy mô tả phù hợp nhất là **model khoanh vùng và xếp hạng sơ bộ; MCTS kiểm chứng và sửa quyết định chiến thuật**.
- R-GAT có policy/value H1 tốt hơn R-GCN, nhưng arena nhỏ lại nghiêng 4–1 về R-GCN. Khác biệt này có thể do màu đi, sampling, kích thước mẫu và phân bố state.

## 8. Kết luận

1. Lỗi lặp ván của arena đã được sửa; nguồn `data` mới đạt 100% ván phân biệt trong mẫu năm ván.
2. Model đã học: policy optimal mass tăng và value error giảm ở cả hai kiến trúc.
3. MCTS **vẫn đang gánh đáng kể tactical correctness**: policy-only 13/24 và 16/24, MCTS 24/24.
4. R-GAT học policy/value H1 tốt hơn R-GCN ở checkpoint cuối, dù kết quả arena nhỏ chưa phản ánh điều này.
5. Graph attention chưa thể được xem là tactical explanation: alignment không vượt baseline.

## 9. Giới hạn và bước đánh giá tiếp theo

- Arena cần 50–100 ván mỗi màu, dùng cùng tập opening trong `mode=eval`, để so sức mạnh agent.
- Cần báo cáo confidence interval thay vì kết luận từ tỷ số 4–1.
- Nên chạy ablation theo budget MCTS: policy-only, 5, 10, 25, 50 và 100 playouts. Đường accuracy theo budget sẽ định lượng trực tiếp MCTS đang gánh bao nhiêu.
- Muốn kết luận causal về graph/attention cần ablation cạnh, relation hoặc attention, không chỉ quan sát heatmap.
- D2c Task 1 phải lấy population từ `mode=data`; các population sinh từ arena tất định cũ cần đánh dấu không hợp lệ.

## 10. Artifact bàn giao

- `results/h1/recheck_20260813/rgcn_developmental_metrics.csv`
- `results/h1/recheck_20260813/rgat_developmental_metrics.csv`
- `results/h1/recheck_20260813/rgcn_developmental_metrics_runtime.json`
- `results/h1/recheck_20260813/rgat_developmental_metrics_runtime.json`
- `results/h3_pilot/arena/rgcn_vs_rgat_data/game_01..05`

Kiểm thử regression sau sửa arena: **54 test passed**. Cảnh báo duy nhất là pytest không ghi được `.pytest_cache`, không ảnh hưởng kết quả.
