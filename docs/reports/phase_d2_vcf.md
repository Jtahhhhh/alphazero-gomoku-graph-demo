# Phase D2 — VCF Solver Report

Ngày chạy: 2026-08-13

## Phạm vi đã hoàn thành

- D2a one-sided VCF cho người sắp đi.
- Chỉ sinh move tạo five/four; không đưa three/open-three vào VCF.
- AND-node phân biệt `NO_OBLIGATION` và `UNSTOPPABLE`.
- Defender counter-five được kiểm tra trước shortcut double-threat; counter-four làm mất tempo và nhánh bị bác bảo thủ.
- Budget node/time lan thành `unknown`.
- Kết quả proven luôn là `exact_partial`, `value=+1`, `optimal_actions_complete=false`.
- Proof tree serialize được, replay mọi AND-child và reduce được về `valid_proofs[]`.
- `solve_vcf()` replay-verify certificate trước khi phát nhãn; proof lỗi trở thành `unknown`.

D2b defensive VCF là tùy chọn và chưa triển khai.

## Sự cố soundness được cổng phát hiện

Lần chạy oracle đầu tiên phát hiện một false-positive tại seed 71: VCF dùng shortcut `UNSTOPPABLE` trước khi kiểm tra defender có immediate five. Exact oracle đánh action 14 là `-1` trong khi VCF ban đầu tuyên bố `+1`.

Đã sửa thứ tự AND-node: defender immediate five được kiểm tra trước double-threat shortcut. Case này được giữ thành regression test `test_defender_counter_five_precedes_double_threat_shortcut`.

## Hai cổng soundness cuối

- Oracle-agreement: 24 legal deterministic 6×6/k=4 states, exact oracle budget 2,000 ms; VCF budget 10,000 nodes/250 ms.
- Kết quả cuối: **0 false-positive / 22 claimed actions**.
- Proof replay: **0 replay failure** trên toàn bộ proof phát ra trong phép đo coverage.
- Replay negative control xóa AND-child: verifier từ chối certificate.

## Coverage smoke measurement

Mỗi board size dùng 24 legal nonterminal random-history states với seed cố định. VCF budget là 10,000 nodes/250 ms mỗi state. Các tập có độ sâu lịch sử khác nhau nên đây là smoke coverage định hướng, không phải ước lượng population-unbiased.

| Board | States | `exact_partial` | Coverage | Replay failures | Unknown breakdown |
|---|---:|---:|---:|---:|---|
| 6×6, k=4 | 24 | 22 | 91.7% | 0 | 2 abstain |
| 10×10, k=5 | 24 | 10 | 41.7% | 0 | 9 time cap, 5 not proven |
| 15×15, k=5 | 24 | 10 | 41.7% | 0 | 13 time cap, 1 not proven |

## Cổng quyết định trước D3

VCF cho coverage đáng kể nhưng chỉ khoảng 41.7% trên hai smoke set bàn lớn; hơn một nửa state vẫn abstain, chủ yếu do time cap. Con số này chưa đủ để tự động quyết định mở VCT vì sampling chưa được chuẩn hóa theo tactical category và N còn nhỏ.

Theo spec, dừng tại đây để review hai số: **false-positive = 0**, coverage lần lượt **91.7% / 41.7% / 41.7%** cho 6×6 / 10×10 / 15×15. Không mở Phase D3 trong commit này.
