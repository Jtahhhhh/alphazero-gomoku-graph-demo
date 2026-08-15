# Phase E-3a.1 — Budget calibration và production distribution

Ngày đo: 2026-08-14  
Phạm vi: đo và chỉnh budget/skip-policy của benchmark generation; không sửa evaluator, graph, benchmark freeze hoặc model evaluation.

## 1. Population và phương pháp

- Checkpoint R-GAT: `results/h3_pilot_v2/rgat/seed_7/model.pt`.
- State source: self-play `mode=data`, MCTS 4 playouts, seed gốc 7; từng game có seed dẫn xuất cố định.
- Thu toàn bộ prefix mid/late và dedup D4 sau round-trip gate.
- 300 state mỗi board: tổng 900 state.
- Bucket: 6×6 dùng `5–9`/`10+`; 10×10 và 15×15 dùng `10–24`/`25+`.
- Exact luôn là solver gốc verified. Enhanced exact solver không được gọi.

## 2. Task 1 — Calibration budget

Calibration dùng cùng một mẫu stratified 16 state/board và ladder 0.5s, 2s, 8s; node cap 1,000,000.

| Board | 0.5s | 2s | 8s | BUDGET* |
|---|---:|---:|---:|---:|
| 6×6 | 2/16 = 12.50% | 3/16 = 18.75% | 3/16 = 18.75% | **2s** |
| 10×10 | 0/16 = 0% | 0/16 = 0% | 0/16 = 0% | **0.5s** |
| 15×15 | 0/16 = 0% | 0/16 = 0% | 0/16 = 0% | **0.5s** |

6×6 tăng 6.25 điểm phần trăm từ 0.5s lên 2s rồi bão hòa; 8s không thu thêm complete. Bàn lớn không thu được complete khi tăng tới 8s, nên giữ 0.5s để tránh chi phí vô ích.

Skip-policy benchmark đã được nới thành **luôn thử exact với BUDGET*** rồi mới fallback VCF. Trên mẫu 6×6, E*=15 runtime sẽ skip cả ba complete tìm thấy ngoài bound; baseline complete theo skip cũ là 0%, còn benchmark always-try ở BUDGET*=2s là 18.75%. Runtime router và E*=15 không thay đổi.

Biểu đồ: `results/h1_integration/e3a1/complete_rate_vs_budget.svg`. Dữ liệu: `complete_rate_vs_budget.csv` và `calibration_summary.json`.

## 3. Task 2 — Production distribution

| Board | Ply bucket | Complete | Partial | Unknown | Tổng |
|---|---|---:|---:|---:|---:|
| 6×6 | 5–9 | 8 | 54 | 93 | 155 |
| 6×6 | 10+ | 71 | 42 | 32 | 145 |
| 10×10 | 10–24 | 0 | 12 | 242 | 254 |
| 10×10 | 25+ | 0 | 7 | 39 | 46 |
| 15×15 | 10–24 | 0 | 20 | 277 | 297 |
| 15×15 | 25+ | 0 | 0 | 3 | 3 |

### Tổng theo board

| Board | Complete | Partial | Unknown | Complete rate | Partial rate | Unknown rate |
|---|---:|---:|---:|---:|---:|---:|
| 6×6 | 79 | 96 | 125 | 26.33% | 32.00% | 41.67% |
| 10×10 | 0 | 19 | 281 | 0% | 6.33% | 93.67% |
| 15×15 | 0 | 20 | 280 | 0% | 6.67% | 93.33% |

Partial replay: **135/135 PASS (100%)**. Reader gate: 900/900 valid, 0 invalid; 686 unknown đều ngoài denominator. Legacy exact agreement: 24/24, zero mismatch.

### Wall-time

| Board | Mean/state | Median/state |
|---|---:|---:|
| 6×6 | 1.571s | 2.022s |
| 10×10 | 0.712s | 0.604s |
| 15×15 | 1.124s | 1.028s |

Tổng measured routing wall-time theo từng record là **1,022.06s ≈ 17.03 phút** cho 900 state. Đây là estimate trực tiếp phù hợp cho một lần benchmark generation; chưa gồm self-play population collection và process startup.

## 4. Task 3 — Ba số E-3b cần

### Gold complete theo board × phase

| Board | Mid | Late | Complete tổng |
|---|---:|---:|---:|
| 6×6 | 8 (`5–9`) | 71 (`10+`) | **79** |
| 10×10 | 0 (`10–24`) | 0 (`25+`) | **0** |
| 15×15 | 0 (`10–24`) | 0 (`25+`) | **0** |

### Partial theo board

- 6×6: **96** — đủ dày cho phân tích partial có stratification.
- 10×10: **19** — rổ mỏng, E-3b phải báo cáo có điều kiện.
- 15×15: **20** — rổ mỏng, E-3b phải báo cáo có điều kiện.

### Confound game-phase của complete

Complete chỉ xuất hiện ở 6×6: 71/79 = **89.87%** thuộc `ply≥10`, chỉ 8/79 = 10.13% thuộc `ply 5–9`. Ply complete có min 7, median 13, mean 13.27, max 19. Gold set vì vậy thiên mạnh về late tactical states. 10×10/15×15 không có complete nên không được phép phát biểu metric complete cho bàn lớn; E-3b phải tách phase và dùng partial metrics có điều kiện.

## 5. Gate và artifacts

- `production_gate.json`: PASS; 24 legacy mismatch=0; 900 v2 invalid=0.
- `production_summary.json`: distribution/wall-time/phase confound.
- `production_candidates.jsonl`: measurement candidates, **chưa freeze thành benchmark**.
- `routing_progress.json` và `population.pt`: cache/resume.

Kết luận: BUDGET* đã thu thêm complete 6×6 (production 500ms trước calibration: 68; 2s: 79), large-board exact vẫn không khả thi trong vùng budget đã đo. Dừng trước E-3b.
