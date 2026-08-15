# Phase D2c — Self-play Coverage and Unknown Triage

Ngày đo: 2026-08-13  
Phạm vi: measurement-only; không thay đổi VCF search, ground-truth label hay pipeline sinh dữ liệu. Task 0 chỉ chuẩn hóa diagnostic `unknown_reason` thành `budget|exhausted`.

## 1. Inputs thực tế

- CKPT: `results/h3_pilot_v2/rgat/seed_7/model.pt`
- SHA-256: `62E6FF9E5D973DBF7C56A66A68EC6E454F6BC22187D7F1BE4CB927A05986B9BC`
- Mtime: 2026-08-13 14:39:49 local; size 460,303 bytes.
- 6×6/k=4: `G=50`, `SIMS=64`.
- 10×10/k=5 và 15×15/k=5: dùng fallback được spec cho phép, `G=20`, `SIMS=32`, do R-GAT MCTS trên bàn lớn chậm.
- Temperature: `1.0` cho ply `<8`, sau đó argmax; seed mỗi game `0..G-1`.
- Dedupe bằng `state_id`; cap `S=500`; subsample seed 12345. Population thực tế đều dưới cap.
- VCF budget giữ như D2a: `node_cap=10,000`, `time_cap_ms=250`.
- Checkpoint học trên 6×6. Khi đo bàn lớn, toàn bộ learned parameter tensor được giữ nguyên; chỉ size-specific graph buffers được tái sinh cho 10×10/15×15. Policy/value heads là node-wise/mean-pooled nên không có tensor học phụ thuộc board size.
- Cache đo: `results/h3_pilot_v2/rgat/seed_7/d2c_measurement.cache.pt`.

## 2. Task 1 — Coverage trên self-play states

| Board | POP states | VCF solved | Coverage self-play | Coverage random D2a |
|---|---:|---:|---:|---:|
| 6×6, k=4 | 416 | 129 | **31.01%** | 91.67% |
| 10×10, k=5 | 254 | 28 | **11.02%** | 41.67% |
| 15×15, k=5 | 209 | 23 | **11.00%** | 41.67% |

Random D2a và self-play không cùng population; cột random chỉ là đối chiếu smoke theo yêu cầu.

### Phân rã theo ply

| Board | Ply bucket | States | Coverage |
|---|---|---:|---:|
| 6×6 | 0–4 | 144 | 2.78% |
| 6×6 | 5–9 | 183 | 40.44% |
| 6×6 | 10+ | 89 | 57.30% |
| 10×10 | 0–9 | 150 | 3.33% |
| 10×10 | 10–24 | 90 | 15.56% |
| 10×10 | 25+ | 14 | 64.29% |
| 15×15 | 0–9 | 144 | 3.47% |
| 15×15 | 10–24 | 65 | 27.69% |
| 15×15 | 25+ | 0 | N/A |

Coverage tăng rõ theo tiến trình ván. Coverage tổng thấp một phần lớn vì POP chứa nhiều opening states chưa có four-chain để chứng minh.

VCF outcomes:

- 6×6: 129 proven, 282 exhausted, 5 budget.
- 10×10: 28 proven, 204 exhausted, 22 budget.
- 15×15: 23 proven, 141 exhausted, 45 budget.

## 3. Task 2 — Unknown triage 6×6

`|U6| = 287`: 282 exhausted + 5 budget. Theo ưu tiên trong spec, `C=5`, nên `frac_C=5/287=1.74%`.

### Blocker exact-oracle

Spec giả định mọi self-play state 6×6 “exact-solvable miễn phí”. Giả định này không đúng với exact solver hiện tại, nhất quán với `docs/audits/phase_a_ground_truth.md`: near-opening 6×6 có cây quá lớn.

Một lần gọi unbounded bị kẹt ở opening state và phải dừng sau hơn 20 phút. Audit hữu hạn sau đó chạy exact solver với 1,000 ms và 1,000,000 nodes trên toàn bộ 282 exhausted states:

- Exact hoàn tất: 23/282.
- Trong phần hoàn tất: `A_known=1`, `B_known=22`.
- Chưa exact: 259/282.
- Artifact audit: `results/h3_pilot_v2/rgat/seed_7/d2c_exact_audit.json`.

Vì 259 state không có exact value, không được phép gán chúng vào A hoặc B. Do đó:

```text
A = chưa xác định hoàn toàn
B = chưa xác định hoàn toàn
C = 5
U6 = 287
P(A) = KHÔNG XÁC ĐỊNH
frac_C = 1.74%
```

Cận bảo thủ duy nhất có thể báo mà không dùng heuristic:

```text
1/287 ≤ P(A) ≤ (1+259)/287
0.35% ≤ P(A) ≤ 90.59%
```

Cận này quá rộng, không dùng được cho quyết định VCT. Việc gọi 259 state timeout là B, C hay A sẽ vi phạm định nghĩa operational và soundness của spec.

## 4. Task 3 — Kiểm tra trùng khớp 10×10 và 15×15

- Solved signatures: 28 cho 10×10, 23 cho 15×15.
- Multiset Jaccard: **`J=0.000`**, dưới ngưỡng 0.70.
- Median bounding box `(height, width, min_row, min_col)`:
  - 10×10: `(4, 10, 0, 0)`
  - 15×15: `(4, 15, 0, 0)`
- Height và origin giống nhau, nhưng width chênh 5 ô, lớn hơn ngưỡng 1. Vì vậy điều kiện generator-size-independent không đạt.

Kết luận cứng theo spec:

```text
generator_size_independent = false
artifact = (J >= 0.70) OR generator_size_independent = false
```

**Task 3: `artifact=false`.** Hai coverage 41.7% trước đây không được giải thích bởi tiêu chí artifact đã định nghĩa; dù vậy coverage self-play vẫn là số ưu tiên.

## 5. Khuyến nghị GO/NO-GO cho VCT

**NO-GO tạm thời / không mở D3.** `frac_C=1.74%` cho thấy VCF budget không chi phối U6, nhưng `P(A)` chưa xác định do exact oracle không giải được 259 opening/mid states. Áp ngưỡng §6 khi thiếu `P(A)` sẽ là suy diễn ngoài spec.

Bước hợp lệ tiếp theo phải là sửa feasibility của Task 2 — ví dụ predefine một exact-solvable late-state triage population hoặc cung cấp exact oracle mạnh hơn — rồi khóa lại denominator trước khi đo. Không được dùng VCF, game outcome hay heuristic để tự gán 259 state.
