# Phase D2c-v2 — P(A) trên tập exact-decidable

Ngày đo: 2026-08-13  
Phạm vi: triage offline trên cache D2c; không sửa VCF search, schema, H1 metrics, pipeline sinh nhãn hay D3.

## Kết luận

**NO-GO VCT / đóng Task 3.** Kết quả chính thức từ Track 1 là `P(A)_dec = 0/1 = 0.00`, thấp hơn ngưỡng `0.15`. Scope của số này là các unknown self-play 6×6 có **không quá 15 ô trống**, tức `exact-decidable ∩ tactically-relevant` theo calibration 60 giây/state. Mẫu rất nhỏ (`n=1`), nhưng đây là quy tắc quyết định operational đã khóa trong spec; không suy rộng con số này sang opening hoặc toàn bộ U6.

Track 2 mở rộng được vùng đo đến `E*=17` và cho số tham khảo `P(A)=1/4=0.25`, nhưng **không được dùng làm ground truth chính thức**: gate VCF đạt 127/129, còn 2 timeout dù không có mâu thuẫn. Theo fail-closed contract, timeout không phải pass.

## Task 0 — exact solver cũ có gì

Audit và mã sản phẩm `azgomoku/solver.py` xác nhận:

- Có negamax + alpha-beta trong `_negamax(state, alpha, beta, ctx)`.
- Có TT khóa bằng `(board.tobytes(), to_play, win_length)`; chỉ cache node không cutoff nên entry là exact.
- Chưa có canonicalization D4/symmetry reduction.
- `solve_actions()` duyệt full-width ở root và trả `action_values` cho mọi root move khi hoàn tất, cùng đầy đủ `optimal_actions`.

Vì vậy Track 2 không tái triển khai negamax/AB/full-width root; phần thêm mới tập trung vào canonical TT, bound flags, ordering và chế độ root-value nhanh.

## Track 1 — calibration bằng solver hiện hành

Input là `U6=287` unknown: 282 `exhausted`, 5 `budget`. Budget exact là 60,000 ms/state, node cap 10,000,000; `E*` là bucket empty-count lớn nhất đạt completion ≥95%.

| Empty cells | States | Exact | Completion | Median time |
|---:|---:|---:|---:|---:|
| 15 | 1 | 1 | 100% | 4,876.82 ms |
| 16 | 1 | 0 | 0% | 60,000.14 ms |

Do đó `E*=15`.

| Scope | A | B | C | Total | P(A) |
|---|---:|---:|---:|---:|---:|
| `U6_dec`, empty ≤15 | 0 | 1 | 0 | 1 | **0.00** |

Artifact:

- `results/h3_pilot_v2/rgat/seed_7/d2c_v2/solve_time_vs_empty.svg`
- `results/h3_pilot_v2/rgat/seed_7/d2c_v2/solve_time_vs_empty.csv`
- `results/h3_pilot_v2/rgat/seed_7/d2c_v2/track1_summary.json`

## Track 2 — enhanced offline exact solver

Track 2 được triển khai vì Track 1 chỉ có một state trong denominator. `azgomoku/offline_solver.py` bổ sung:

- canonical board qua đủ 8 phép D4 trong TT;
- TT `exact/lower/upper` an toàn với alpha-beta;
- root-value triage dùng zero-window hai pass; H1 dùng root full-width riêng;
- ordering: thắng ngay, block thắng ngay, tạo four, rồi khoảng cách tới tâm;
- tùy chọn preferred root action chỉ để ordering; child đó vẫn phải được exact negamax chứng minh;
- abstain `unknown` khi hết time/node budget.

Calibration sau nâng cấp:

| Empty cells | States | Exact | Completion |
|---:|---:|---:|---:|
| 15 | 1 | 1 | 100% |
| 16 | 1 | 1 | 100% |
| 17 | 2 | 2 | 100% |
| 18 | 3 | 2 | 66.67% |

`E*=17`; `A=1`, `B=2`, `C=1`, total `4`, số tham khảo `P(A)=0.25`. Đây không phải số certified vì gate tổng chưa pass.

### Bốn soundness gate

| Gate | Result | Checks | Failures |
|---|---|---:|---:|
| Agreement solver cũ + H1 exact_complete | PASS | 47 | 0 |
| D4 self-consistency/action mapping | PASS | 40 | 0 |
| VCF replay-verified positives ⇒ exact `+1` | **INCONCLUSIVE/FAIL-CLOSED** | 129 | 2 timeout |
| TT on/off | PASS | 10 | 0 |

VCF cross-check có 127 exact `+1`, 0 contradiction, 2 timeout sau tối đa 300 giây/state. Vì không đủ bốn ô xanh, enhanced solver chưa được phép ghi nhãn H1; số state mới materialize thành `exact_complete` là **0**.

Artifacts:

- `results/h3_pilot_v2/rgat/seed_7/d2c_v2_enhanced_bounds/track1_summary.json`
- `results/h3_pilot_v2/rgat/seed_7/d2c_v2_enhanced/soundness_gates_final.json`
- `results/h3_pilot_v2/rgat/seed_7/d2c_v2_enhanced/vcf_offline_crosscheck.json`

## Coverage self-play và quyết định

Coverage VCF giữ nguyên D2c: 6×6 **31.01%**, 10×10 **11.02%**, 15×15 **11.00%**. Mid/late coverage tương ứng đạt 57.30% ở 6×6 `ply≥10` và 64.29% ở 10×10 `ply≥25`; 15×15 chưa có state `ply≥25` trong mẫu.

Áp ngưỡng đã khóa lên số Track 1 certified: `P(A)_dec=0.00 < 0.15` ⇒ **đóng VCT, không mở D3**. H1 tiếp tục với scope VCF-only trên tập tactically-decided.
