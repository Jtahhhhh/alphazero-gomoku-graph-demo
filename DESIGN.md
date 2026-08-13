# Phase C — Design: Sound Threat-Solver + Abstention

Ngày khóa thiết kế: 2026-08-13  
Trạng thái: **LOCKED**  
Đầu vào: `AUDIT.md`, `FEASIBILITY.md`, PRE-PHASE-C Design Lock.  
Phạm vi: đặc tả; chưa triển khai source code sản phẩm.

## 0. Mục tiêu và bất biến

Thiết kế thêm một ground-truth solver song song với `azgomoku/solver.py`, dùng full minimax cho bàn nhỏ và threat proof cho bàn lớn. Soundness quan trọng hơn coverage. Không dùng heuristic/depth-limited evaluation làm nhãn, không cắt nước theo khoảng cách hình học, và mọi trường hợp chưa chứng minh trong budget phải abstain.

Exact solver hiện tại được giữ nguyên làm oracle 6×6. Kết quả bàn lớn chỉ có ý nghĩa trên tập `tactically-decided`, luôn báo coverage.

## 1. Interface solver thống nhất

### Quyết định

Thêm interface mới song song; không thay `azgomoku.solver.SolverResult` v1:

```python
solve(state: GomokuState, budget: SolverBudget, method: SolverMethod) -> GroundTruthResult
```

```text
SolverBudget {
  node_cap: int | null,
  time_cap_ms: int | null
}

GroundTruthResult {
  status: "exact_complete" | "exact_partial" | "unknown",
  value: -1 | 0 | +1 | null,
  optimal_actions: list[int] | null,
  optimal_actions_complete: bool,
  action_values: map[int, -1|0|+1] | null,
  method: "full_minimax" | "vcf" | "vct" | "dfpn",
  proof: ProofNode | null,
  nodes: int,
  elapsed_ms: float,
  budget: SolverBudget,
  unknown_reason: string | null,
  coverage_note: string
}
```

`SolverMethod` là enum đóng. `dfpn` được dành chỗ trong contract nhưng chưa bắt buộc triển khai.

### Luật kết quả

- `full_minimax` thành công: `exact_complete`, value thuộc `{-1,0,+1}`, toàn bộ `action_values`, toàn bộ `optimal_actions`, `optimal_actions_complete=true`, `proof` có thể null.
- `vcf|vct|dfpn` chứng minh ít nhất một action thắng: `exact_partial`, `value=+1`, chỉ các action đã có certificate nằm trong `optimal_actions` và `action_values[action]=+1`, `optimal_actions_complete=false`, `proof` bắt buộc khác null.
- Không chứng minh được hoặc hết budget: `unknown`, `value=null`, `optimal_actions=null`, `action_values=null`, `optimal_actions_complete=false`, `proof=null`, `unknown_reason` bắt buộc.
- Threat solver không phát `value=0` hoặc `value=-1`: thất bại tìm proof không phải là proof hòa/thua.

Lý do: existential winning proof đủ xác nhận `value=+1` và action đã chứng minh là optimal, nhưng không đủ phủ mọi optimal action hay kết luận hòa/thua.

### Router

Router được cấu hình bằng board-size policy:

```text
small board → full_minimax
large board → vcf → nếu unknown thì vct → nếu bật thì dfpn → unknown
```

Mỗi attempt có budget riêng được ghi lại. Router chỉ nâng `unknown` thành `exact_partial` khi tầng sau trả proof hợp lệ; tuyệt đối không dùng heuristic fallback.

## 2. OR/AND semantics và threat primitives

### Quyết định

Search nội bộ dùng ba trạng thái `PROVEN`, `DISPROVEN`, `UNKNOWN`. Attacker là `state.to_play` tại root và không đổi danh tính trong toàn proof. OR-node là lượt attacker; AND-node là lượt defender.

### Primitive và nguồn tái sử dụng

- `creates_five(state, move, player)`: yêu cầu move legal, gọi `state.play(move)`, rồi xác nhận `child.winner() == player`. Winning windows được tìm bằng `azgomoku.tactics.windows(size, win_length)`.
- `winning_completions(state, player)`: nhóm kết quả từ `azgomoku.tactics.immediate_threats(board, player, win_length)` theo ô hoàn thành; luôn xác minh lại bằng `creates_five` để tránh tin pattern chưa replay.
- `four(state, player)`: một threat có tập `winning_completions` không rỗng. Double-four là khi không tồn tại một legal defender move duy nhất loại bỏ toàn bộ winning completions còn lại.
- `three_candidate(state, move, player)`: sau move hợp lệ, tồn tại ít nhất một extension hợp lệ ở lượt tấn công kế tiếp tạo four; open-three là trường hợp có ít nhất hai extension khác nhau tạo four. Đây chỉ là bộ sinh candidate, không phải certificate thắng.
- `threat_moves(state, method)`: luôn gồm immediate five; VCF thêm các move tạo four; VCT thêm các `three_candidate`. Candidate generation có thể thiếu nước tấn công và làm giảm coverage, nhưng không được dùng để lược defender response thiếu bảo chứng.

Lý do chọn định nghĩa vận hành theo legal move + terminal check thay vì chỉ tên pattern: pattern có thể sinh candidate rộng, còn thắng chỉ được xác nhận bằng replay đến `winner()`.

### OR-node

Với attacker sắp đi:

1. Duyệt `threat_moves` theo thứ tự deterministic.
2. Move tạo five hợp lệ trả `PROVEN` với leaf terminal.
3. Nếu có một child `PROVEN`, OR-node trả `PROVEN`.
4. Nếu không có `PROVEN` nhưng có `UNKNOWN`, trả `UNKNOWN`.
5. Chỉ trả `DISPROVEN` khi toàn bộ candidate đã duyệt đều `DISPROVEN` và không hết budget.

Bỏ sót attacker candidate chỉ gây false-negative/giảm coverage, không đủ tạo proof sai.

### AND-node: completeness nằm ở defender

AND-node chỉ `PROVEN` khi mọi reply thuộc tập response đầy đủ đã được chứng minh dẫn tới attacker thắng.

Ba quy tắc bắt buộc:

1. **Không giả định defender ngoan.** Mặc định response set là toàn bộ `state.legal_actions()`.
2. **Chỉ thu hẹp có bảo chứng:**
   - Nếu attacker đang đe dọa five ngay, tính mọi ô winning completion.
   - Một defender reply được coi là blocking reply khi sau khi áp dụng, attacker không còn immediate five từ threat hiện hữu.
   - Mọi reply không chặn và không tự thắng cho defender thua ngay ở lượt attacker kế; chúng có thể được đóng bằng terminal continuation xác minh được.
   - Nếu không tồn tại một reply chặn hết mọi completion, node là unstoppable và `PROVEN`, nhưng certificate vẫn phải ghi các completion/winning leaves đủ để verifier xác nhận.
   - Với open-three/VCT, **không thu hẹp**: vét toàn bộ legal defender actions. Đây là quyết định khóa nhằm bảo toàn soundness; tối ưu hóa chỉ được thêm sau nếu có định lý/pruning rule và test riêng.
3. **Counter-threat tường minh:** sau mỗi defender reply, kiểm tra theo thứ tự:
   - defender tạo five → nhánh `DISPROVEN` ngay;
   - defender tạo immediate four/counter-threat khiến attacker phải bỏ chuỗi tấn công → nhánh được xử lý bảo thủ là `DISPROVEN`, trừ khi attacker có immediate five và kết thúc trước;
   - nếu không, tiếp tục proof search.

Một child `DISPROVEN` làm AND-node `DISPROVEN`. Nếu không child nào `DISPROVEN` nhưng có `UNKNOWN`, node `UNKNOWN`. Chỉ khi tất cả child `PROVEN`, node mới `PROVEN`.

Lý do: false-positive chủ yếu xuất hiện khi bỏ sót hoặc đơn giản hóa phản ứng defender. Xử lý counter-four bảo thủ có thể mất proof thật nhưng không tạo nhãn sai.

## 3. Hợp đồng perspective và MCTS v2

### Quyết định

Mọi `value` trong `GroundTruthResult` là từ góc nhìn **người sắp đi tại state được đánh giá**. Mọi `action_values[a]` cũng từ góc nhìn người chọn action ở state cha đó:

```text
action_value(s, a) = -value(play(s, a))
```

`+1` nghĩa là người sắp đi tại `s` có thắng cưỡng bức.

Điều này khớp `azgomoku/mcts.py::Node`, nơi docstring quy định Q của action edge theo perspective người chơi ở parent, và khớp `azgomoku/explanation/mcts_trace.py`:

```text
MCTS_VALUE_CONVENTION_VERSION = 2
MCTS_Q_PERSPECTIVE = "player_who_selects_action_at_parent"
```

### Điểm chuyển đổi duy nhất

Không đổi dấu khi join root solver action với root MCTS candidate: `solver.action_values[a]` và `root.children[a].q` cùng perspective. Phép đổi dấu duy nhất nằm khi đi qua một ply trong recursion/backup:

```text
parent_action_value = -child_state_value
```

Normalization/join layer phải assert artifact MCTS có convention version 2 và đúng chuỗi perspective; version khác hoặc thiếu metadata bị fail-closed khỏi phép so Q.

### Assertion bắt lệch dấu

Harness có fixture 6×6 exact `value=+1`. Với search đủ sâu/deterministic theo fixture, ít nhất một `optimal_action` đã được thăm phải có `Q>0`; đồng thời action oracle `-1` không được được diễn giải thành thắng do đổi dấu. Test này là diagnostic contract cho join; oracle-agreement của solver không phụ thuộc MCTS.

Lý do: lưu convention ở đúng một adapter ngăn evaluator tự đổi dấu lần hai.

## 4. Semantics `exact_partial` và chính sách H1

### Quyết định

`exact_partial` nghĩa là đã chứng minh tồn tại ít nhất một winning action, không có nghĩa đã liệt kê đầy đủ optimal set. Field `optimal_actions_complete` là bắt buộc:

| Method/status | `optimal_actions_complete` |
|---|---|
| `full_minimax/exact_complete` | `true` |
| `vcf|vct|dfpn/exact_partial` | `false` |
| `unknown` | `false` |

Threat solver chỉ đưa action có certificate hợp lệ vào `optimal_actions`. Tên field được giữ để reader dùng chung, nhưng report phải gọi đây là `proven_winning_actions` khi flag false.

### H1 metric policy

- Complete: giữ policy/value optimal-set metrics và graph metrics hiện có gồm critical mass, precision@K, recall@K, AUPRC.
- Partial:
  - được báo `proven-action probability mass`, top-1 có thuộc tập action đã chứng minh hay không, proof critical mass và precision@K;
  - không báo optimal-set recall, không phạt model vì chọn action ngoài danh sách partial;
  - không công bố recall@K hoặc AUPRC như metric so sánh chính trên partial records, vì chúng có thành phần recall; giá trị chẩn đoán nếu lưu phải gắn `partial_non_penalizing=false` và không aggregate chung;
  - value target `+1` vẫn hợp lệ vì đã có một thắng cưỡng bức.
- Unknown: loại trước mọi metric ground-truth.

Mọi bảng H1 tách `board_size`, `status`, `optimal_actions_complete`; không trộn complete và partial trong cùng aggregate không phân tầng.

Lý do: tập winning action đã chứng minh là lower bound của optimal set. Precision trên proof đã biết có nghĩa; recall của một tập chưa đầy đủ không có nghĩa.

## 5. Proof tree, replay và reduce

### Quyết định schema

Proof tree dùng node tối thiểu sau:

```text
ProofNode {
  player_to_move: 1 | -1,
  move: int | null,
  node_type: "OR" | "AND",
  children: list[ProofNode],
  terminal: "five" | null
}
```

Semantics khóa:

- Root có `move=null` và `player_to_move=state.to_play`.
- Với node khác root, `move` là nước đã áp dụng từ parent để đạt state của node đó.
- `node_type` được xác định từ `player_to_move`: attacker → OR, defender → AND.
- Leaf thắng có `terminal="five"`, `children=[]`; state tại leaf phải có `winner()==attacker`.
- Không serialize node UNKNOWN/DISPROVEN vào winning certificate. Chỉ cây `PROVEN` hoàn chỉnh được xuất.
- Mỗi AND-node chứa child cho toàn bộ response cần xét theo semantics mục 2; một principal variation đơn không hợp lệ.

### Replay contract

`replay(state, proof)`:

1. Kiểm tra root metadata khớp state và attacker.
2. Với từng child, kiểm tra `child.move` thuộc legal actions, áp dụng bằng `GomokuState.play`, và kiểm tra `player_to_move`/`node_type` của child khớp state mới.
3. OR-node phải có ít nhất một child PROVEN trong certificate; canonical writer chỉ giữ một winning child để proof gọn.
4. AND-node phải có đúng tập child move mà `mandatory_replies` tái tính từ state yêu cầu; thiếu, thừa, trùng hoặc illegal đều fail.
5. Tại leaf, yêu cầu `terminal="five"` và `winner()==attacker`.
6. Vét đệ quy mọi child của AND-node. Replay chỉ thành công nếu mọi leaf là attacker five.

### Reduce sang `valid_proofs[]`

Không đổi các field v1. Với mỗi root proven action, reducer tạo một proof:

```text
{
  action,
  concepts,
  critical_cells,
  critical_relations,
  windows,
  proof_method: "vcf"|"vct"|"dfpn"|"full_minimax",
  proof_status: "exact"
}
```

- `action` là move của winning child tại root OR.
- `critical_cells` là hợp các attacker moves: các child moves được chọn từ OR parents trên mọi nhánh defender của cây proof. Không đưa defender moves vào tập này.
- `windows` là mọi cửa sổ dài `win_length` xác nhận five tại terminal leaves, tái tính bằng `azgomoku.tactics.windows()`; sort/deduplicate deterministic.
- `critical_relations` là relation của các terminal windows.
- `concepts` gồm `winning_line` và `forced_sequence`, cộng `vcf` hoặc `vct` theo method; reader v1 vẫn có thể bỏ qua concept chưa biết nếu chỉ dùng geometry.
- `proof_method` và `proof_status="exact"` bắt buộc ở schema v2.

Lý do: tree phục vụ audit/replay; projection phẳng giữ nguyên join hiện tại của H1.

## 6. Schema v2 và reader fail-closed

### Quyết định field-by-field

Top-level v2:

```text
schema_version: 2
state_id: string
state: {
  board_size: int,
  win_length: int,
  current_player: 1|-1,
  last_move: int,
  board: list[list[int]],
  legal_actions: list[int]
}
provenance: object
solver: GroundTruthResult
valid_proofs: list[Proof]
```

Field `solver.status`, `method`, `value`, `optimal_actions`, `optimal_actions_complete`, `action_values`, `proof`, `nodes`, `elapsed_ms`, `budget`, `unknown_reason`, `coverage_note` đều bắt buộc ở v2, kể cả khi giá trị là null. `budget.node_cap` và `budget.time_cap_ms` bắt buộc hiện diện. `valid_proofs[*].proof_method` và `proof_status` bắt buộc.

Ràng buộc chéo:

- `exact_complete` ↔ `method=full_minimax`, `optimal_actions_complete=true`, value khác null, full action map.
- `exact_partial` ↔ threat method, `value=+1`, `optimal_actions_complete=false`, proof khác null, mọi listed action có `action_values=+1` và proof exact.
- `unknown` ↔ value/actions/proof null, flag false, `unknown_reason` khác null, `valid_proofs=[]`.
- Mọi action phải legal; proof replay phải qua trước khi record được eligible cho H1.

### Mapping v1

- v1 `status="exact"` normalize thành `exact_complete`, `method="full_minimax"`, `optimal_actions_complete=true`; các field không tồn tại được đánh dấu `legacy_missing`, không bịa budget/proof tree.
- v1 `timeout|node_budget` normalize thành `unknown` với `unknown_reason` tương ứng và bị loại H1.
- Artifact v1 không bị rewrite.

### Bảng reader → hành động

| Tình huống đọc | Hành động |
|---|---|
| `schema_version=1`, status exact hợp lệ | Parse v1, normalize complete, cho vào H1 theo contract v1 |
| `schema_version=1`, timeout/node_budget | Normalize unknown, loại H1, tính vào coverage denominator nếu thuộc sampling manifest |
| `schema_version=2`, đủ field và cross-field validation | Parse; chỉ exact complete/partial mới eligible |
| v2 thiếu bất kỳ field bắt buộc | Reject record khỏi H1, ghi validation error |
| v2 status ngoài enum | Reject khỏi H1, ghi validation error |
| v2 unknown | Parse hợp lệ nhưng loại H1; ghi coverage/unknown reason |
| v2 exact_partial thiếu proof hoặc replay fail | Reject khỏi H1 như invalid, không hạ cấp im lặng thành trusted unknown |
| v2 exact_complete có action map thiếu legal root action | Reject khỏi H1 |
| version lớn hơn 2 | Unsupported/reject khỏi H1; không đoán forward compatibility |
| thiếu `schema_version` hoặc JSON hỏng | Reject và báo lỗi |
| state/proof chứa action illegal, ID mismatch hoặc enum lạ | Reject khỏi H1 |

Reader trả cả accepted records và structured rejection report; không bỏ record im lặng.

Lý do: ground truth thiếu metadata phải được xem là không đáng tin, không được lấp mặc định.

## 7. Test-first oracle-agreement harness

### Quyết định

Trước D2, tạo harness E.1 với adapter `threat_solver_stub(state, budget)` luôn trả `unknown`. Harness phải xanh ở trạng thái stub; D2 chỉ thay adapter bằng VCF implementation.

Harness:

1. Sinh deterministic N legal 6×6/k=4 states bằng logic `legal_random_history_v1`, lưu seed/history/state ID.
2. Chỉ giữ state mà `azgomoku.solver.solve_actions(...)` trả exact trong oracle budget.
3. Lưu oracle `value`, toàn bộ `optimal_actions`, `action_values`.
4. Gọi threat solver qua một interface injection duy nhất.
5. Với mọi action threat solver tuyên bố proven `+1`, assert `oracle.action_values[action] == +1`.
6. Assert `exact_partial` có proof replay thành công, action legal, flag incomplete và value +1.
7. Stub unknown không tạo positive và vì vậy pass.

Bất kỳ false-positive nào fail test và chặn merge. Test báo seed, history, board, method, action, certificate để tái hiện.

Harness bổ sung assertion perspective mục 3, nhưng không dùng MCTS làm oracle correctness cho threat solver.

Lý do: hàng rào soundness tồn tại trước dòng VCF đầu tiên, tránh implementation tự định nghĩa lại tiêu chí đúng.

## 8. Budget và abstention propagation

### Quyết định

Một context dùng chung giữ start time, node counter và immutable caps. Mọi recursive entry kiểm tra budget. Không reset budget tại child và không loại thời gian candidate generation/reply enumeration khỏi elapsed time.

Quy tắc ternary:

| Node | Điều kiện | Kết quả |
|---|---|---|
| OR | có ít nhất một child PROVEN | PROVEN |
| OR | không PROVEN, có ít nhất một UNKNOWN | UNKNOWN |
| OR | mọi child DISPROVEN | DISPROVEN |
| AND | có ít nhất một child DISPROVEN | DISPROVEN |
| AND | không DISPROVEN, có ít nhất một UNKNOWN | UNKNOWN |
| AND | mọi child PROVEN | PROVEN |

Budget vượt tại bất kỳ điểm nào tạo `UNKNOWN`, với `unknown_reason="time_cap"` hoặc `"node_cap"`. Nếu đồng thời chạm cả hai, reason theo check deterministic `time_cap` trước rồi `node_cap`, và có thể ghi secondary diagnostics ngoài contract.

`UNKNOWN` không được cache như PROVEN/DISPROVEN. Transposition table nếu cache ternary phải gắn method, attacker, board, player-to-move và chỉ tái dùng kết quả phù hợp budget semantics; Phase D ban đầu chỉ cache PROVEN/DISPROVEN đã hoàn tất để tránh nâng cấp nhầm unknown.

Router có thể thử method tiếp theo sau unknown bằng budget riêng được ghi riêng, nhưng chỉ kết quả proof của method thành công được xuất. Nếu tất cả unknown, final status unknown và H1 loại record; state vẫn nằm trong coverage denominator.

Lý do: unknown là thiếu chứng minh, không phải giá trị game. Lan truyền ternary ngăn timeout bị biến thành false hoặc thành nhãn thắng.

## 9. Definition of Done Phase C

- [x] Interface thống nhất, status/method enum và router.
- [x] OR/AND semantics, đầy đủ ba quy tắc defender và counter-threat.
- [x] Perspective khớp MCTS convention v2 và assertion bắt lệch dấu.
- [x] `exact_partial`, `optimal_actions_complete` và H1 policy.
- [x] `ProofNode`, replay contract và reduce sang `valid_proofs[]`.
- [x] Schema v2 field-by-field và bảng reader fail-closed.
- [x] Kế hoạch harness E.1 với stub injection trước VCF.
- [x] Budget/abstention ternary propagation.

**Phase C hoàn tất về mặt đặc tả. Cho phép chuyển sang Phase D1, nhưng chưa cho phép dùng threat labels làm ground truth cho đến khi D2/D3 qua oracle-agreement và certificate replay.**
