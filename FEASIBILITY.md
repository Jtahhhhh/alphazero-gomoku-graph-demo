# Phase B — Feasibility Gate cho Threat-Solver + Abstention

Ngày đánh giá: 2026-08-13  
Đầu vào: `AUDIT.md` và mã nguồn hiện tại.  
Phạm vi: quyết định khả thi; không viết hoặc sửa code sản phẩm.

## Quyết định

**GO — có điều kiện soundness bắt buộc.**

Ba tiêu chí GO đều đạt:

1. Có thể thêm threat solver như module song song mà không sửa thuật toán exact solver hiện tại.
2. Có thể tái sử dụng bộ sinh window/directional relation và detector immediate threat hiện có.
3. Có thể thêm schema v2 và nhánh đọc tương thích schema v1 mà không làm thay đổi artifact v1.

Quyết định GO chỉ cho phép chuyển sang **Phase C — Design**. Nó chưa cho phép dùng bất kỳ kết quả VCF/VCT nào làm ground truth trước khi correctness harness chứng minh zero false-positive trên oracle 6×6.

## 1. Khả năng tái sử dụng

### Kết luận

Tái sử dụng được phần hình học line/window và immediate-four detector; tái sử dụng được cấu trúc tactical proof đầu ra. Không tái sử dụng được một threat-search engine vì repo chưa có thành phần đó.

### Bằng chứng mã nguồn

- `azgomoku/tactics.py::windows(size: int, length: int)` sinh mọi cửa sổ đúng `length` theo `DIRECTIONS`.
- `azgomoku/tactics.py::DIRECTIONS` ghép trực tiếp bốn tên quan hệ từ `azgomoku.graph.RELATIONS` với `(0,1)`, `(1,0)`, `(1,1)`, `(1,-1)`.
- `azgomoku/tactics.py::immediate_threats(board, player, win_length)` nhận diện cửa sổ có `win_length - 1` quân và đúng một ô trống. Với k=5, đây chính là primitive “đánh một nước thành five”, phù hợp làm nền cho detector four.
- `azgomoku/tactics.py::extract_tactical_proofs(state)` đã chuẩn hóa proof thành `action`, `concepts`, `critical_cells`, `critical_relations`, `windows`, đồng thời sort kết quả ổn định.
- `azgomoku/graph.py::RELATIONS` và `cell_edge_records(size)` dùng cùng bốn quan hệ và edge ID `"{relation}:{source}:{target}"`, nên threat certificate có thể ánh xạ sang graph evidence mà không đổi graph topology.

### Phần không được xem là đã có

`immediate_threats()` không phải detector hoàn chỉnh cho three/open-three. `simple_fork` chỉ đếm ít nhất hai ô thắng ngay sau một nước; nó không chứng minh mọi phản ứng của defender trong chuỗi nhiều ply. Tìm kiếm mã nguồn không thấy VCF, VCT, AND/OR search, DF-PN, proof/disproof number hoặc certificate verifier.

Do đó, Phase C/D phải gọi lại `windows()` hoặc tách primitive chung có hành vi tương đương; không được suy diễn rằng `simple_fork` đã là VCT.

## 2. Điểm chèn solver song song

### Kết luận

Có điểm chèn sạch ở cấp module và ở generator. Không cần sửa thuật toán trong solver cũ.

### Bằng chứng mã nguồn

- Exact solver có API độc lập tại `azgomoku/solver.py::solve_actions(state, deadline_ms=None, node_budget=None) -> SolverResult` và `solve_state(...)` là alias.
- Solver chỉ phụ thuộc `GomokuState`; không được nhúng trong model, MCTS hoặc graph builder.
- Generator gọi solver tại `investigation/generate_h1_benchmark.py::generate(...)`, cụ thể kiểm tra `result.status != "exact"` trước khi nhận record. Đây là nơi có thể thay lời gọi trực tiếp bằng router ở Phase D5.
- Kích thước bàn có sẵn qua `state.size`; `GomokuState.initial(size=6, win_length=4)` ở `azgomoku/game.py` đã nhận size và k, dù generator hiện gọi mặc định bằng `GomokuState.initial()`.

### Hình dạng tích hợp được phép

Phase C nên thiết kế một interface/router mới, ví dụ một module riêng nhận `GomokuState` và budget:

```text
ground-truth router
├── board nhỏ  → azgomoku.solver.solve_actions (giữ nguyên)
└── board lớn  → VCF → VCT → unknown
```

Router chọn method bằng cấu hình/policy theo board size; không đặt điều kiện board-size vào `_negamax()` và không thay semantics của `SolverResult` v1. Exact solver cũ tiếp tục là oracle vàng cho 6×6 và test agreement.

### Điều kiện kỹ thuật

- Threat solver mới chỉ được công bố `exact_partial` khi có certificate thắng cưỡng bức có thể verify độc lập.
- Hết node/time budget ở bất kỳ nhánh chưa giải quyết nào phải lan truyền thành `unknown`, không được biến thành nhánh thua.
- Không giới hạn candidate theo bán kính hình học; chỉ được giới hạn bằng định nghĩa threat đã chốt.
- Router không được fallback sang heuristic/depth-limited evaluation để tạo nhãn.

## 3. Khả năng mở rộng schema tương thích ngược

### Kết luận

Khả thi với schema v2 và parser phân nhánh rõ theo version. Schema hiện nhỏ và được tạo tập trung, nhưng chưa có validator nên Phase D phải bổ sung validation thay vì chỉnh ad-hoc ở evaluator.

### Bằng chứng mã nguồn

- `investigation/generate_h1_benchmark.py::record(...)` là điểm tạo record tập trung và hiện ghi `"schema_version": 1`.
- Cùng hàm đó ghi nguyên `"solver": result.dict()` và `"valid_proofs": proofs`; vì vậy có thể thêm metadata ở v2 mà giữ hình dạng proof lõi.
- `investigation/evaluate_h1.py::load_records(path)` hiện chỉ gọi `json.loads` từng dòng, không dùng Pydantic/dataclass/JSON Schema.
- `investigation/evaluate_h1.py::evaluate_record(...)` đọc trực tiếp `record["solver"]` và `record["valid_proofs"]`, nên cần normalization layer trước evaluator để v1/v2 cùng đi qua một contract nội bộ.

### Chiến lược tương thích bắt buộc cho Phase C

- Artifact đang có giữ nguyên `schema_version = 1`; không rewrite tại chỗ.
- Artifact mới dùng `schema_version = 2`.
- Reader thực hiện `parse_v1` hoặc `parse_v2`, rồi normalize về representation nội bộ chung.
- Mapping v1: `solver.status == "exact"` tương ứng full exact/complete; `timeout` và `node_budget` tương ứng unknown với lý do giữ nguyên. Không bịa `method`, budget hoặc certificate nếu artifact cũ không chứa chúng.
- Writer v2 không phát trạng thái partial/exact nếu thiếu certificate hợp lệ.
- H1 lọc unknown trước metric và báo coverage theo board size. V1 exact records vẫn được đánh giá như trước.

Thiết kế field/enums chính xác thuộc Phase C; Phase B chỉ xác nhận extension này khả thi và không đòi phá v1.

## 4. Rủi ro soundness

### Rủi ro lớn nhất

Rủi ro nghiêm trọng nhất là **bỏ sót một phản ứng hợp lệ của defender**, làm một nhánh OR bị xử lý nhầm như forced move hoặc một node AND chưa duyệt hết bị coi là đã chứng minh. Nguồn lỗi điển hình:

1. Một “four” có nhiều ô hóa giải, hoặc một nước defender vừa chặn vừa tạo counter-threat/win.
2. Hai threat window giao nhau bị deduplicate sai, dẫn đến kết luận nhầm double-four.
3. Four-three/open-three được phân loại chỉ bằng số window mà không kiểm tra legality và mọi response sau đó.
4. Overlapping windows hoặc chuỗi dài hơn k tạo nhiều representation của cùng một threat.
5. Budget hết giữa node AND nhưng trạng thái chưa giải quyết bị coi là false thay vì unknown.
6. Certificate lưu một principal variation duy nhất trong tình huống defender có nhiều response; như vậy không chứng minh thắng cưỡng bức.
7. Dùng giới hạn khoảng cách quanh last move làm mất một nước chặn/counter-threat ở xa.

### Soundness contract tối thiểu

Một kết quả thắng chỉ được chấp nhận khi:

- mọi nước trong certificate hợp lệ theo `GomokuState.legal_actions()` và được áp dụng bằng `GomokuState.play(action)`;
- tại node attacker tồn tại ít nhất một continuation được chứng minh;
- tại node defender, **mọi** response hợp lệ theo threat semantics đều có continuation thắng đã chứng minh;
- leaf thắng được kiểm tra bằng `GomokuState.winner()`, không bằng điểm heuristic;
- node bị timeout/node cap trả unknown và unknown lan truyền bảo thủ;
- certificate verifier độc lập replay toàn bộ cây proof;
- trên miền 6×6, mọi action được threat solver tuyên bố thắng phải có `action_values[action] == +1` từ `azgomoku.solver.solve_actions(...)`.

Oracle-agreement 6×6 với **zero false-positive** là merge gate. Coverage thấp hoặc false-negative được chấp nhận; một false-positive là blocker.

## 5. Đánh giá theo tiêu chí GO

| Tiêu chí | Bằng chứng | Kết quả |
|---|---|---|
| Solver mới chạy song song, không đụng solver cũ | `azgomoku/solver.py::solve_actions(...)` là API module độc lập; generator có một call site rõ ở `investigation/generate_h1_benchmark.py::generate(...)` | Đạt |
| Tái sử dụng line/window detector | `azgomoku/tactics.py::windows(...)`, `immediate_threats(...)`, `DIRECTIONS` | Đạt |
| Schema mở rộng an toàn | Record được tạo tập trung ở `record(...)`; có `schema_version=1`; evaluator có thể đặt parser/normalizer trước `evaluate_record(...)` | Đạt, với điều kiện thêm validator/reader v1-v2 |

## 6. Các điều kiện trước khi triển khai

Phase C phải chốt bằng văn bản, trước khi viết code:

1. Semantics chính xác của five, four, open-four, three/open-three, defense set và counter-win.
2. AND/OR semantics và quy tắc lan truyền `proven`, `disproven`, `unknown`.
3. Interface kết quả thống nhất giữa full solver và threat solver.
4. Proof-tree/certificate schema đủ biểu diễn mọi defender response, không chỉ một line.
5. Schema v2, mapping v1, unknown filtering và coverage reporting.
6. Budget semantics có thể tái lập và lý do abstention.
7. Correctness tests, đặc biệt oracle-agreement 6×6, adversarial distant defense và tiny-budget abstention.

## 7. Phạm vi quyết định

GO không khẳng định coverage sẽ cao trên 10×10/15×15 và không khẳng định VCT/DF-PN sẽ đủ nhanh. GO chỉ khẳng định kiến trúc repo cho phép triển khai hướng threat-solver sound + abstention mà không phá exact solver, graph evidence hoặc artifact v1.

Kết luận khoa học trên bàn lớn sau này bắt buộc ghi: **có điều kiện trên tập tactically-decided**, kèm coverage theo board size. Chuẩn vàng 6×6 full exact được giữ nguyên.
