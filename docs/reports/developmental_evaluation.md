# Developmental evaluation trên frozen H1 benchmark

Ngày chạy: 2026-08-14  
Benchmark SHA-256: `9abd52ef4991489586682e881e495fcb4c2ffe00fb55dc9dee1d9008aca4ff02`  
Phạm vi: inference trên 13 checkpoint có sẵn × 2 model; không train, không self-play, không sinh state và không sửa benchmark.

## Phán quyết

**H_null “alignment phẳng chỉ vì chưa train đủ/chưa hội tụ” bị bác bỏ theo tiêu chí đã định trước.**

Đuôi hội tụ late của R-GAT là iter **55 và 60**:

- `policy_optimal_mass ≥ 0.95×max`, với max=0.52164 và ngưỡng=0.49556.
- `hard_collapse_rate ≤ 5%`: cả hai checkpoint bằng **0%**.
- Critical mass vẫn thấp hơn strongest baseline ở cả hai checkpoint.
- Topology correlation vẫn cao: 0.96175 và 0.96872.
- Alignment excess không tăng trong đuôi: thay đổi từ -0.0021561 xuống -0.0021824, gain=-0.0000263; thấp hơn ngưỡng tăng có ý nghĩa 0.001.

Câu dùng cho luận văn:

> R-GAT attention không trở thành tactical explanation dù competence đã vào đuôi hội tụ và hard collapse đã biến mất; attention vẫn bám graph topology (correlation khoảng 0.96–0.97) và không vượt structural/random baseline xuyên suốt đuôi training.

Đây là kết luận developmental tương quan, chưa phải nhân quả. Causal topology/tactic ablation vẫn là bước riêng.

## Gates

| Gate | Kết quả |
|---|---:|
| Checkpoint | 13 R-GCN + 13 R-GAT, iter 0,5,…,60 |
| Per-state inference rows | 2,444 |
| Aggregated model × checkpoint × phase rows | 52 |
| Benchmark hashes trong output | 1, khớp manifest |
| Structural baseline drift | 0 |
| Random baseline drift | 0 |
| Iter-60 critical/topology so với E-3b | Khớp, sai số ≤2.1e-17 |
| Full regression suite | **86 passed** |

Baseline late hằng qua mọi checkpoint: structural=0.0330864, random=0.0351212. Baseline mid hằng: structural=0.0245536, random=0.0268162.

## Quỹ đạo R-GAT late — n=71, kết luận chính

| Iter | Policy mass | Critical mass | Δ structural | Δ random | Topology corr | Hard collapse |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.49713 | 0.033084 | -0.000002 | -0.002037 | 0.99999 | 100% |
| 5 | 0.50011 | 0.033086 | -0.000001 | -0.002035 | 1.00000 | 100% |
| 10 | 0.50357 | 0.033087 | +0.0000004 | -0.002034 | 1.00000 | 100% |
| 15 | 0.50667 | 0.033080 | -0.000006 | -0.002041 | 1.00000 | 100% |
| 20 | 0.50996 | 0.033030 | -0.000057 | -0.002092 | 0.99889 | 100% |
| 25 | 0.50648 | 0.032873 | -0.000214 | -0.002249 | 0.99210 | 100% |
| 30 | 0.51203 | 0.032912 | -0.000175 | -0.002209 | 0.99479 | 100% |
| 35 | 0.51528 | 0.033028 | -0.000059 | -0.002093 | 0.99774 | 100% |
| 40 | 0.51138 | 0.033025 | -0.000062 | -0.002097 | 0.99493 | 100% |
| 45 | 0.51313 | 0.032987 | -0.000099 | -0.002134 | 0.98729 | 52.11% |
| 50 | 0.50905 | 0.033038 | -0.000049 | -0.002084 | 0.97217 | 8.45% |
| **55** | **0.52164** | **0.032965** | **-0.000121** | **-0.002156** | **0.96175** | **0%** |
| **60** | **0.51482** | **0.032939** | **-0.000148** | **-0.002182** | **0.96872** | **0%** |

Critical mass không vượt strongest baseline ở bất kỳ checkpoint nào. Blip tốt nhất là iter 10 nhưng vẫn thấp hơn random baseline 0.00203445. Vì vậy không có bằng chứng attention từng hình thành tactical alignment rồi mất đi.

## Ba quỹ đạo

### 1. Decoupling

Policy optimal mass late tăng nhẹ từ 0.4971 lên vùng 0.51–0.52, trong khi critical mass nằm quanh structural và luôn dưới random. Competence và attention alignment vì vậy bị decouple theo thời gian.

### 2. Topology mechanism

Kịch bản quan sát được là **high-from-initialization**, không phải drift-toward-topology: topology correlation bắt đầu ở 0.99999, luôn cao hơn 0.96, và kết thúc ở 0.96872. Attention được neo vào topology ngay từ initialization rồi chỉ tách nhẹ khi train.

### 3. Collapse confound

Late hard-collapse là 100% tại iter 0–40, giảm còn 52.11% ở iter 45, 8.45% ở iter 50 và 0% ở iter 55–60. Điều này tách được hai vùng:

- Đầu training: flat alignment bị confound bởi collapse.
- Đuôi training: collapse đã hết nhưng alignment vẫn không vượt baseline và topology correlation vẫn cao.

Chính vùng đuôi thứ hai bác bỏ giải thích “chỉ vì collapse/chưa train đủ”.

## MCTS subset — late n=71

| Iter | R-GCN mass | R-GCN gain | R-GAT mass | R-GAT gain |
|---:|---:|---:|---:|---:|
| 0 | 0.8735 | +0.3769 | 0.8718 | +0.3747 |
| 20 | 0.9020 | +0.3965 | 0.8918 | +0.3819 |
| 40 | 0.8530 | +0.3497 | 0.8721 | +0.3607 |
| 60 | 0.8113 | +0.3141 | 0.8808 | +0.3660 |

MCTS tiếp tục tạo search gain lớn cho cả hai kiến trúc. Đây chỉ là subset kiểm tra 0/20/40/60, không phải MCTS-budget ablation.

## So sánh competence R-GCN/R-GAT

So sánh policy/value theo thời gian là hợp lệ vì đây là hai kiến trúc thật. Ở late endpoint, R-GAT có policy mass 0.51482 so với R-GCN 0.49718, và value error 0.87928 so với 0.92208. Quỹ đạo không đơn điệu và chưa có CI, nên chỉ diễn giải mô tả.

Ở trục alignment, R-GCN luôn chỉ là structural control theo thiết kế; không được diễn giải “R-GCN không align” như một kết quả học được.

## Mid-game — n=23, chỉ gợi ý

Mid không tham gia phán quyết mạnh. Ở iter 55→60, topology correlation 0.97446→0.98080, critical mass 0.024346→0.024329 và vẫn dưới structural/random. Hard-collapse là 8.70%→13.04%, nên confound mid chưa sạch như late; chỉ ghi nhận pattern gợi ý.

## Artifacts

- `results/h1_integration/developmental/developmental_metrics.csv`: 52 hàng checkpoint × model × phase.
- `results/h1_integration/developmental/developmental_per_state.json`: 2,444 hàng nguồn có cache/resume.
- `results/h1_integration/developmental/developmental_gates.json`: baseline + endpoint consistency gates.
- `results/h1_integration/developmental/developmental_analysis.json`: tail definitions và H_null verdict.
- `results/h1_integration/developmental/figures/decoupling.svg`.
- `results/h1_integration/developmental/figures/topology-correlation.svg`.
- `results/h1_integration/developmental/figures/hard-collapse.svg`.

Không chạy train thêm, causal ablation, arena, MCTS budget ablation hoặc state generation.
