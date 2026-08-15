# Phase E-3a.2 — Mở rộng gold set 6×6 mid-game

Ngày chạy: 2026-08-14  
Phạm vi: chỉ thu thêm `exact_complete` trước E-3b; không sửa evaluator, graph, benchmark freeze hoặc model evaluation.

## Kết quả chốt

| Pha | Gold trước E-3a.2 | Gold mới | Gold sau E-3a.2 |
|---|---:|---:|---:|
| Mid, ply 5–9 | 8 | 15 | **23** |
| Late, ply ≥10 | 71 | 0 | **71** |
| Tổng | 79 | 15 | **94** |

Tỷ trọng mid tăng từ 8/79 = 10.13% lên 23/94 = **24.47%**. Gold vẫn thiên late: 71/94 = **75.53%**.

## Task 1 — Multi-seed, solver gốc 2 giây

- Nguồn state: R-GAT self-play `mode=data`, checkpoint `results/h3_pilot_v2/rgat/seed_7/model.pt`, MCTS 4 playouts.
- Đã đi hết K=20 seed, tối đa 6 game/seed; chỉ lấy prefix ply 5–9.
- Dedup D4 với toàn bộ 300 candidate 6×6 của E-3a.1 và giữa các state mới.
- Thu được 159 candidate mới: **13 exact-complete**, 83 exact-partial, 63 unknown.
- Chỉ 13 exact-complete đi tiếp vào gold; partial/unknown không được đưa vào artifact gold-only.

## Task 2 — Enhanced proposer, solver gốc phê chuẩn

Task 1 chỉ đưa tổng mid lên 21/30 nên đã chạy nhánh proposer:

| Kết quả | Số state |
|---|---:|
| Candidate chưa complete được sàng | 146 |
| Enhanced trả `exact` | 11 |
| Enhanced abstain/unknown | 135 |
| Solver gốc 8 giây trả `exact_complete` | 2 |
| Solver gốc không complete | 9 |
| Nhãn được nhận sau khi hai bên trùng | **2** |
| Enhanced-vs-gốc mismatch | **0** |
| Nhãn enhanced-only | **0** |

Hai nhãn nhận vào đều lấy `(value, optimal_actions)` từ solver gốc verified. Enhanced chỉ dùng để đề xuất state đáng thử.

## Agreement và soundness gates

- Tất cả 15 gold mới được chạy lại bằng solver gốc với cổng generous 30 giây, 5,000,000 node.
- Kết quả: **15/15 trùng hoàn toàn**, mismatch **0**, rejected **0**.
- D4 uniqueness trên 94 gold: PASS.
- Schema v2 và `exact_complete/full_minimax` semantics trên artifact: PASS.
- Legacy router agreement: **24/24**, mismatch **0**; artifact gate: **94/94** valid và eligible.
- Full regression suite: **76 passed**.

Một lượt kiểm tra ban đầu ở budget 8 giây có 1 lần không qua cổng boolean. Không dùng kết quả đó để hạ chuẩn; lượt xác nhận generous độc lập sau cùng hoàn tất và trùng trên cả 15/15. Chi tiết từng state nằm trong `results/h1_integration/e3a2/original_agreement.json`.

## Phân bố ply của gold sau mở rộng

| Ply | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Gold | 1 | 2 | 2 | 18 | 5 | 9 | 9 | 9 | 11 | 10 | 8 | 7 | 2 | 1 |

Min=6, median=13, mean=12.53, max=19. Mid representation đã tốt hơn nhưng chủ yếu nằm ở ply 9 (18/23), nên E-3b vẫn phải công bố số mẫu theo ply/phase.

## Nhánh E-3b được kích hoạt

`10 ≤ |mid_complete|=23 < 30` ⇒ **mid_suggestive**.

E-3b được phép báo kết quả mid dưới dạng “gợi ý, n=23”, không kết luận mạnh cho mid-game. Kết luận chính vẫn dựa trên late-game 6×6, kèm caveat phase rõ ràng. Không được phát biểu phát hiện chính cho cả hai phase như khi `mid_complete ≥ 30`.

## Ghi chú runtime CPU/GPU

CUDA đã được nối vào nguồn self-play qua `--device`. WSL nhận RTX 3050 6GB, nhưng benchmark cùng trajectory 5 ply cho kết quả CPU 3.65 giây và CUDA 6.81 giây; CUDA chậm hơn do MCTS suy luận tuần tự batch=1 trên model nhỏ. Lượt production vì vậy giữ inference trên CPU. Exact solver luôn chạy CPU.

## Artifacts

- `results/h1_integration/e3a2/expanded_gold.jsonl`: 94 record gold-only, chưa freeze thành benchmark E-3b.
- `results/h1_integration/e3a2/summary.json`: số liệu tổng hợp và nhánh quyết định.
- `results/h1_integration/e3a2/task1_progress.json`: cache multi-seed/router.
- `results/h1_integration/e3a2/task2_progress.json`: log proposer/phê chuẩn.
- `results/h1_integration/e3a2/original_agreement.json`: agreement từng gold mới.
- `results/h1_integration/e3a2/gate.json`: schema + legacy agreement gate.

Kết luận: E-3a.2 thu thêm 15 mid complete sound, nâng mid gold từ 8 lên **23** nhưng chưa đạt N_min=30. Dừng trước E-3b đúng phạm vi.
