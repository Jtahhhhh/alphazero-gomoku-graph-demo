# Source Overview

## Phạm vi hiện tại

Repository chỉ giữ pipeline nghiên cứu R-GCN/R-GAT cho AlphaZero-style Gomoku. Cấu hình chuẩn hiện tại là bàn 6×6, `win_length=4`.

## Luồng chính

1. `azgomoku/game.py`: trạng thái, legal moves, apply move, terminal/winner và feature tensor.
2. `azgomoku/graph.py`: cell graph với bốn quan hệ horizontal, vertical và hai diagonal; stable edge ID.
3. `models/rgcn.py`, `models/rgat.py`: hai policy/value encoder được nghiên cứu.
4. `azgomoku/mcts.py`: MCTS convention v2, Q theo người chọn action ở parent.
5. `azgomoku/training.py`: self-play, replay samples, symmetry augmentation, opening exploration và optimization.
6. `experiments/run_h3_pilot.py`: train/resume/checkpoint H3 từ config.
7. `investigation/h3_evaluate.py`: đánh giá checkpoint trajectory trên frozen H1 benchmark.
8. `azgomoku/explanation/game_export.py`: arena model-vs-model và export JSON/SVG theo từng nước.

## Cơ chế chống collapse

- Tám phép D4 symmetry augmentation cho features và policy.
- Root Dirichlet noise chỉ trong self-play.
- Temperature schedule: sampling ở opening, greedy ở late game.
- Log opening diversity, entropy, corner mass và edge mass.
- Arena/evaluation không tự thêm noise.

## Ground truth và soundness

- `azgomoku/solver.py`: bounded exact negamax/alpha-beta.
- `azgomoku/tactics.py`: immediate win, mandatory block, fork và threat primitives.
- `azgomoku/vcf.py`: one-sided replay-verified VCF với abstention.
- `azgomoku/oracle_agreement.py`: exact-oracle false-positive gate trên 6×6.
- `diagnostic/h1_tactical.jsonl`: 24 frozen exact tactical states.
- `investigation/generate_h1_benchmark.py`: generator H1.
- `investigation/validate_solver.py`: exhaustive independent validation.

Threat solver chỉ phát nhãn thắng đã chứng minh; không chứng minh được trả `unknown`. VCF label luôn có optimal action set chưa đầy đủ.

## Evidence và visualization

- R-GCN export stable structural edges, không gắn learned attention.
- R-GAT export attention từng edge/head và mean attention.
- Mỗi arena ply có board SVG, graph SVG, decision SVG và explanation JSON.
- Attention là descriptive model evidence, không phải causal proof.

## Artifact được giữ

- `results/h1`: kết quả H1 hiện có.
- `results/h3_pilot`: checkpoint/run cũ để đối chiếu.
- `results/h3_pilot_v2`: run mới sau chống collapse.

Legacy CNN/HAN, static graph demo, optimization probes, báo cáo trung gian và result cũ đã được loại khỏi pipeline.

## Correctness hiện tại

Full suite sau cleanup: 50 tests passed. Test bao phủ core game/graph/model, explanation/arena, checkpoint/resume, collapse controls, H1 ground truth, exact solver, tactics, oracle agreement và replay-verified VCF.

## Giới hạn nghiên cứu

- H1 hiện thiên về tactic ngắn; không đại diện đầy đủ opening/whole-game strength.
- Run cũ cho thấy MCTS cứu policy đáng kể nhưng R-GAT alignment gần random.
- Cần train mới từ đầu sau anti-collapse, nhiều seed và arena đổi bên trước khi so sánh kiến trúc.
- Bàn lớn chỉ được kết luận có điều kiện trên tập `tactically-decided`, kèm coverage.
