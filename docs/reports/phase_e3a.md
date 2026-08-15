# Phase E-3a — H1 integration: Router → Schema v2 → Generator

Ngày hoàn tất: 2026-08-13  
Phạm vi: chỉ nửa đầu integration H1; không sửa evaluator, graph evidence, benchmark đóng băng hoặc đánh giá model.

## 1. Ground-truth router

`azgomoku/ground_truth.py` thực hiện thứ tự:

1. Thử exact solver gốc `azgomoku.solver.solve_actions` nếu state không vượt E* đã hiệu chuẩn.
2. Nếu exact abstain hoặc được bỏ qua vì tối ưu budget, thử VCF.
3. Chỉ trả `exact_partial` khi certificate replay thành công; ngược lại trả `unknown` không action/proof.

Chỉ 6×6 có bound hiệu chuẩn `E*=15`. Board khác không bị cắt theo kích thước và mặc định vẫn thử exact. Router có `RouterStats` log count/rate complete, partial và unknown.

Gate tổng trên 24 legacy H1 exact states: **24/24 exact_complete, zero mismatch** về value và optimal actions. Artifact: `results/h1_integration/e3a_gate.json`.

## 2. Schema v2 và validator fail-closed

`azgomoku/h1_schema.py` cung cấp writer, parser proof tree, validator và structured rejection. Schema ghi đầy đủ status/method/completeness/budget/runtime/proof/provenance/perspective v2.

Các trường hợp đã có test:

- v1 exact tiếp tục normalize thành complete;
- thiếu hoặc status lạ bị reject;
- thiếu completeness bị xử lý như partial;
- sai perspective hoặc thiếu field bắt buộc bị reject;
- partial proof bị sửa/xóa nhánh bị reject ngay lúc đọc;
- unknown được parse nhưng ngoài ground-truth denominator.

## 3. Generator H1 candidates

`investigation/generate_h1_benchmark.py` hiện:

- hỗ trợ 6×6/k4, 10×10/k5 và 15×15/k5;
- loại opening thưa, lấy bucket mid/late theo D2c: 6×6 từ ply 5, board lớn từ ply 10;
- CLI bắt buộc checkpoint và dùng đúng stochastic selection `mode=data`;
- chạy mọi state qua router, không gọi enhanced exact solver;
- lưu seed, history, budget, generator version, checkpoint SHA-256, board/ply/empty count và dedup mode;
- giữ unknown cho coverage nhưng validator loại khỏi denominator;
- dedup D4 chỉ bật sau round-trip self-check; canonical key gồm board, player, win length và last move.

D4 tests phủ invariant canonical qua 8 symmetry, action mapping, và round-trip `critical_cells/relations/windows`.

## 4. Smoke thật

Input checkpoint: `results/h3_pilot_v2/rgat/seed_7/model.pt`; `mode=data`; 1 MCTS playout; budget 500 ms/100,000 nodes; seed 7.

- 3 records: một record cho mỗi board 6/10/15.
- 3/3 `exact_partial`; 3/3 certificate replay PASS; 0 invalid.
- D4 dedup enabled và round-trip PASS.
- Bucket: 6×6 `10+`, 10×10 `10–24`, 15×15 `10–24`.

Tỷ lệ partial 100% chỉ là smoke n=3 với budget thấp, không phải benchmark coverage hay dấu hiệu production router phình partial. Gate prefer-complete riêng trên 24 exact states cho 24/24 complete.

Artifacts:

- `results/h1_integration/e3a_smoke.jsonl`
- `results/h1_integration/e3a_smoke.summary.json`
- `results/h1_integration/e3a_gate.json`

## 5. Ranh giới bàn giao E-3b

E-3a chỉ tạo candidates có nhãn/provenance/dedup mode. Chưa thay evaluator, chưa nối proof vào graph/SVG, chưa đóng băng/hash benchmark và chưa đánh giá R-GCN/R-GAT.
