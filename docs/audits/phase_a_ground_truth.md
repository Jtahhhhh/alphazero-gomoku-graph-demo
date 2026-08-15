# Phase A — Audit nền tảng Ground Truth

Ngày audit: 2026-08-13  
Phạm vi: chỉ đọc mã nguồn và chạy phép đo chẩn đoán; chưa thay đổi solver, schema, generator, evaluator hay pipeline huấn luyện.

## 1. Luật chơi và biểu diễn trạng thái

- Trạng thái nằm ở `azgomoku/game.py`, lớp `GomokuState` (`@dataclass(frozen=True)`). Bàn cờ là ma trận `numpy`, quân hiện tại là `to_play`, nước cuối là chỉ số phẳng `last_move`, và độ dài thắng là `win_length`.
- `GomokuState.initial(size=6, win_length=4)` cho thấy mặc định hiện tại là 6×6, k=4; kích thước và k có thể truyền vào nên lớp trạng thái không bị khóa cứng ở 6×6.
- `GomokuState.legal_actions()` lấy các ô bằng 0 theo thứ tự row-major bằng `np.flatnonzero(board.reshape(-1) == 0)`.
- `GomokuState.play(action)` kiểm tra biên và ô đã chiếm, tạo bản sao bàn cờ, đặt quân, rồi đổi lượt.
- `GomokuState.winner()` quét toàn bộ ô theo bốn hướng `(0,1)`, `(1,0)`, `(1,1)`, `(1,-1)` và kiểm tra đúng `win_length` ô liên tiếp. Chi phí xấp xỉ `O(size² × win_length)` cho mỗi lần gọi.
- `terminal()` và `outcome_for(player)` dựa trên `winner()` và số nước hợp lệ. Không thấy luật cấm, overline hay biến thể Renju; semantics hiện tại là Gomoku tự do, có chuỗi dài ít nhất k thì thắng vì mọi chuỗi dài hơn chứa một cửa sổ dài k.

## 2. Solver exact hiện tại

- Entry point sản phẩm là `azgomoku/solver.py::solve_actions(state, deadline_ms=None, node_budget=None)`; `solve_state` chỉ là alias.
- Thuật toán lõi là `azgomoku/solver.py::_negamax(state, alpha, beta, ctx)`: negamax + alpha-beta + transposition table, khóa bởi `(board.tobytes(), to_play, win_length)`.
- `azgomoku/solver.py::_ordered_actions(state)` ưu tiên nước thắng ngay, sau đó sắp theo khoảng cách Manhattan tới tâm.
- `_Context.enter()` thi hành cả deadline và node budget. Kết quả có status `exact`, `timeout`, hoặc `node_budget`.
- `SolverResult` trả `status`, `value`, `optimal_actions`, `action_values`, `nodes`, `elapsed_ms`. Khi bị giới hạn, solver giữ các root action đã hoàn tất trong `action_values`, nhưng chủ động đặt `value=None` và `optimal_actions=()`. Đây là hành vi abstain an toàn ở tầng kết quả solver.
- Transposition table chỉ lưu giá trị khi node không bị alpha-beta cutoff, vì vậy entry đã lưu là giá trị exact chứ không phải bound bị gắn nhầm là exact.
- Điểm nghẽn chính: `solve_actions()` giải riêng mọi root action bằng cửa sổ đầy đủ; `_ordered_actions()` còn gọi `state.play(a).winner()` cho từng nước hợp lệ, trong khi `winner()` quét cả bàn. Branching factor và chi phí kiểm tra thắng tăng nhanh theo kích thước bàn.
- `investigation/solver_benchmark.py::solve(...)` là probe độc lập cũ, dùng memoized negamax và deadline nhưng không phải solver sản phẩm; tài liệu đầu file cũng ghi rõ điều này.

## 3. Tactical proofs và nền móng threat hiện có

- `azgomoku/tactics.py::windows(size, length)` liệt kê các cửa sổ dài k ở bốn hướng.
- `azgomoku/tactics.py::immediate_threats(board, player, win_length)` chỉ nhận diện cửa sổ có đúng k−1 quân của người chơi và một ô trống — tức threat thắng ở nước kế tiếp.
- `azgomoku/tactics.py::extract_tactical_proofs(state)` hiện sinh ba concept:
  - `immediate_win` + `winning_line`;
  - `mandatory_block`, chỉ khi mọi immediate threat của đối phương cùng quy về một ô chặn;
  - `simple_fork`, khi một nước tạo ít nhất hai ô thắng kế tiếp khác nhau.
- Proof hiện có chứa `action`, `concepts`, `critical_cells`, `critical_relations`, `windows` và được sort ổn định.
- Tìm kiếm mã nguồn trong `azgomoku`, `investigation`, `tests` không thấy VCF, VCT, DF-PN/proof-number search, AND/OR threat search, open-three, double-four hay forcing sequence tổng quát. Vì vậy đây mới là bộ phát hiện tactic cục bộ một ply, chưa phải threat solver.
- Chưa có bộ phân loại open-three/closed-three/four tổng quát; `immediate_threats()` là nền móng gần nhất có thể tái sử dụng cho detector “four”.

## 4. Graph và ID ổn định

- `azgomoku/graph.py::RELATIONS` định nghĩa bốn quan hệ `horizontal`, `vertical`, `diagonal_down`, `diagonal_up`.
- `cell_graph(size)`, `line_memberships(size)` và `metapath_edges(size)` được cache theo kích thước và sinh theo vòng lặp xác định.
- `cell_edge_records(size)` và `metapath_edge_records(size)` tạo edge ID theo công thức ngữ nghĩa `"{relation}:{source}:{target}"`; metapath record còn có `line_id`. Với cùng board size và implementation, thứ tự/ID là ổn định.
- `tests/test_h1_benchmark.py` kiểm tra evidence edge ID là duy nhất và ánh xạ tọa độ nhất quán. Chưa thấy registry/version riêng cho graph ID, nên thay đổi quy tắc dựng graph trong tương lai vẫn có thể làm thay đổi tập cạnh dù chuỗi ID có dạng ổn định.

## 5. Dataset H1 và pipeline đánh giá

- Generator hiện tại là `investigation/generate_h1_benchmark.py::generate(target=24, seed=7, attempts=30000, deadline_ms=2000)`.
- Generator gọi `GomokuState.initial()` không truyền tham số, nên dữ liệu H1 hiện bị cố định thực tế ở 6×6, k=4. CLI không có `board_size` hoặc `win_length`.
- Record schema v1 được tạo bởi `investigation/generate_h1_benchmark.py::record(...)`, gồm `schema_version`, `state_id`, `state`, `provenance`, `solver`, `valid_proofs`.
- Generator chỉ nhận record khi `solver.status == "exact"`, sau đó chỉ giữ proofs gắn với `optimal_actions`; trạng thái timeout/node-budget không bị dùng làm ground truth.
- `investigation/evaluate_h1.py::load_records(path)` đọc JSONL trực tiếp, chưa validate schema. `evaluate_record(...)` ưu tiên proofs của action mà model chọn nếu action đó optimal, nếu không thì dùng optimal action nhỏ nhất; nếu không có proof tương ứng thì fallback sang mọi proof thuộc optimal actions.
- `docs/contracts/h1_correctness.md` đã quy định exact player-to-move value, đầy đủ optimal actions, timeout/node-budget không phải ground truth, proofs tách khỏi solver và alignment theo action. Đây là nền hợp lý nhưng chưa mô tả `unknown_reason`, threat certificate hoặc ngân sách threat solver.
- Tests hiện có:
  - `tests/test_solver.py`: perspective terminal/draw, immediate win, giới hạn không được báo exact, tính deterministic, và so khớp exhaustive solver độc lập trên 3×3;
  - `tests/test_tactics.py`: bốn hướng immediate win, mandatory block, simple fork, nhiều proofs;
  - `tests/test_h1_benchmark.py`: replay hợp lệ, exact solver output, symmetry dedup, action/proof hợp lệ, edge ID ổn định.

## 6. Phép đo khả năng mở rộng

Lệnh đo gọi trực tiếp `azgomoku.solver.solve_actions`, dùng `node_budget=5000`. Mẫu 6×6 là late tactical với 7 ô trống và deadline 2000 ms. Mẫu 10×10/15×15 là trạng thái gần đầu ván có hai quân, k=5, deadline 200 ms. Đây là smoke measurement định hướng, không phải benchmark thống kê.

| Board | Legal root actions | Deadline | Status | Nodes | Elapsed | Root actions hoàn tất |
|---|---:|---:|---|---:|---:|---:|
| 6×6, k=4 | 7 | 2000 ms | exact | 11 | 11.32 ms | 7 |
| 10×10, k=5 | 98 | 200 ms | timeout | 9 | 203.78 ms | 0 |
| 15×15, k=5 | 223 | 200 ms | timeout | 3 | 224.27 ms | 0 |

Diễn giải: exact solver phù hợp với tactical state nhỏ/gần cuối ván, nhưng không tạo được một root label nào cho hai trạng thái bàn lớn gần đầu ván trong ngân sách ngắn. Số node rất thấp ở bàn lớn không biểu thị tìm kiếm hiệu quả; phần đáng kể thời gian bị tiêu tốn bởi việc tạo/sắp nước và quét `winner()` trên bàn lớn.

## 7. Khoảng trống so với mục tiêu Threat-Solver + Abstention

1. Chưa có định nghĩa threat chuẩn hóa theo k=5 (four/open-four/three/open-three) và chưa có unit test chống false positive.
2. Chưa có AND/OR search, VCF/VCT search, transposition semantics cho proof/disproof, hoặc certificate có thể replay/verify độc lập.
3. `SolverResult` có abstention theo resource limit, nhưng schema dataset chưa có nhãn `unknown` có cấu trúc, `unknown_reason`, method, budgets, độ sâu hay certificate.
4. Generator H1 cố định 6×6/k=4 và thiên về late-game exact states; chưa có sampling policy cho 10×10/15×15.
5. Evaluator chưa validate schema/version và chưa có policy loại unknown khỏi metric ground-truth.
6. Chưa có verifier bảo đảm certificate chỉ dùng legal moves, đúng lượt, đúng threat transition và kết thúc bằng win bắt buộc.

## 8. Kết luận Phase A

Nền tảng hiện tại đủ để tái sử dụng các thành phần: luật chơi tổng quát theo size/k, immediate-threat windows, bounded exact solver với abstention an toàn, proof-to-graph mapping và stable semantic edge IDs. Tuy nhiên, chưa có threat solver theo nghĩa VCF/VCT và exact solver không mở rộng được trực tiếp tới trạng thái bàn lớn có branching cao.

Phase A không đưa ra quyết định GO/NO-GO. Bước kế tiếp theo plan phải là Phase B: chốt threat semantics, soundness contract, certificate format, quan hệ giữa exact/threat/unknown, budgets và acceptance criteria trước khi viết product code.
