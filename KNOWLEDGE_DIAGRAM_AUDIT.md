# AUDIT — Contrast Knowledge-Diagram

Ngày audit: 2026-08-14  
Phạm vi: chỉ đọc code, test và artifact hiện có; không chạy evaluator/freeze/đánh giá E-3b; không sửa code, schema hay SVG.

## Tóm tắt điều hành

Repo **đã đủ dữ kiện để dựng contrast knowledge-diagram trên tập gold có proof phẳng**, và cổng tọa độ D4 hiện có artifact xanh. Tuy nhiên, chưa thể gọi toàn bộ lớp nền là một “proof graph/cây lập luận đầy đủ” cho mọi state: 242/243 proof chỉ là tactical proof phẳng; chỉ 1 proof có VCF certificate dạng cây OR/AND. Ngoài ra, nhãn khái niệm hiện gắn ở mức proof, không gắn trực tiếp theo từng edge/window; R-GCN structural score được dựng trong pipeline E-3b nhưng chưa được export ở cùng schema evidence như R-GAT.

Phán quyết ngắn: **khả thi ngay cho diagram đối chiếu proof phẳng trên gold; khả thi sau khi dựng cầu biểu diễn nếu yêu cầu proof-tree đầy đủ, concept-per-edge và một export thống nhất R-GAT/R-GCN.**

## A. Lớp nền — tri thức solver (proof graph)

### A1. Node tri thức

**ĐÃ CÓ cho proof phẳng.** Mỗi phần tử `valid_proofs[]` có:

- `action`: nước đi mà proof chứng minh;
- `concepts`: ví dụ `immediate_win`, `winning_line`, `mandatory_block`, `simple_fork`; VCF có thêm `forced_sequence`, `vcf`;
- `critical_cells`: danh sách action index phẳng của các ô liên quan;
- `critical_relations`: tập quan hệ hình học;
- `windows`: danh sách cửa sổ, mỗi cửa sổ là đầy đủ các action index của k ô.

Thực thể tạo tactical proof nằm ở `azgomoku/tactics.py:210-224`: immediate win ghi `concepts=["immediate_win","winning_line"]`; mandatory block và simple fork đều ghi `critical_cells`, `critical_relations`, `windows`. `azgomoku/h1_schema.py:51-67` ghi toàn bộ `result.valid_proofs` ra record. Artifact thật `diagnostic/h1_benchmark_v1/h1_benchmark_v1.jsonl` có 94 record, 83 record mang proof, tổng 243 proof.

Ví dụ thật trong benchmark: state `b3a6c7628630359d`, action `14`, concept `winning_line/forced_sequence/vcf`, `critical_cells=[13,14,15]`, relation `horizontal`, hai windows `[12,13,14,15]` và `[13,14,15,16]`.

### A2. Edge có nhãn khái niệm

**CÓ dữ kiện để suy ra edge hình học, nhưng KHÔNG CÓ nhãn khái niệm trực tiếp trên từng edge.**

- Nhãn khái niệm nằm ở `valid_proofs[i].concepts`.
- Quan hệ nằm riêng ở `valid_proofs[i].critical_relations`.
- Vùng/cửa sổ nằm riêng ở `valid_proofs[i].windows`.
- Graph edge chuẩn có `edge_id`, `relation`, `source`, `target` tại `azgomoku/graph.py:17`.
- Cầu join hiện tại ở `investigation/evaluate_h1.py:43-46`: edge được coi là critical khi relation thuộc `critical_relations` và source/target cùng nằm trong ít nhất một proof window.

Do đó renderer có thể gắn concept của **proof** lên các edge được suy ra, nhưng với proof có nhiều concept/relation/window thì schema không chỉ rõ concept nào thuộc riêng window/edge nào. Field concept-per-edge/window: **KHÔNG TÌM THẤY**.

### A3. Vùng

**ĐÃ CÓ.** `windows` chứa đầy đủ action index của k ô. Bộ sinh hình học `azgomoku/tactics.py:65-71` tạo tuple ô theo bốn relation; gate replay kiểm tra window/relation với hình học thật tại `investigation/e3b_common.py:73-86`. Renderer so sánh E-3b đã vẽ mỗi window thành polyline tại `investigation/e3b_graph.py:153-160`.

### A4. Mạch lập luận OR/AND

**CÓ serialize, nhưng không phủ toàn bộ gold.**

- `azgomoku/h1_schema.py:40-48` đọc cây gồm `player_to_move`, `move`, `node_type`, `children`, `terminal`.
- `azgomoku/ground_truth.py:35-41` serialize `proof` bằng `ProofNode.dict()`.
- `investigation/e3b_common.py:153-169` đặt cây VCF vào `proof_certificates[].tree` và nối proof phẳng bằng `certificate_id`.

Artifact frozen có đúng **1 VCF certificate**; ví dụ state `b3a6c7628630359d` chứa cây OR/AND nhiều tầng kết thúc ở `terminal="unstoppable_double_threat"`. Manifest `diagnostic/h1_benchmark_v1/manifest.json:33-44` xác nhận 242 tactical proofs + 1 VCF proof = 243. Vì vậy cây lập luận cho 242 tactical proof: **KHÔNG TÌM THẤY**; chúng chỉ có proof phẳng đã replay.

### A5. Nhiều proof/optimal

**ĐÃ CÓ và không bị ép còn một proof trong dữ liệu.** `valid_proofs` là list; `annotate_gold` thu thập, canonicalize, deduplicate rồi giữ toàn bộ tại `investigation/e3b_common.py:140-190`. Benchmark có 243 proof trên 83 proof-bearing states. Tuy nhiên renderer comparison hiện chọn `record["valid_proofs"][0]` tại `investigation/e3b_graph.py:164-170`, nên **dữ liệu có đủ nhiều proof nhưng SVG hiện tại chỉ vẽ proof đầu tiên**.

### Kết luận A

Lớp nền đủ trực tiếp cho **knowledge layer phẳng**: ô, action, relation, window, concept và nhiều proof. Thiếu chính xác:

1. concept-per-window/edge (hiện chỉ có `proof.concepts` tổng quát);
2. proof tree cho 242 tactical proof;
3. renderer duyệt toàn bộ `valid_proofs` thay vì `[0]` nếu mục tiêu là vẽ đủ mọi proof.

## B. Lớp phủ — model evidence và phép join

### B1. Stable identity

**ĐÃ CÓ cơ chế join ổn định theo action index.**

- Node identity thực tế là action phẳng `row * size + col`; object node export có `action`, `row`, `col` (`azgomoku/explanation/model_evidence.py:11-12`). Không có field tên riêng `node_id`; `action` đóng vai trò node id.
- Edge identity là `<relation>:<source>:<target>` tại `azgomoku/graph.py:17`.
- `collect_model_evidence` tra edge ổn định bằng `(relation,source,target)` rồi xuất đúng `edge_id` tại `azgomoku/explanation/model_evidence.py:17-35`.
- Policy dùng vector theo action index; MCTS root children cũng dùng action. `investigation/e3b_graph.py:45-84` kiểm tra proof action ∈ legal ∩ optimal ∩ graph nodes và identity `divmod`; `runtime_mcts_action_gate` tại dòng 87-102 kiểm tra MCTS children bằng đúng tập legal action và proof action nằm trong children.

Join proof critical cells ↔ graph nodes ↔ policy index là trực tiếp, không cần bảng tra ngoài.

### B2. R-GAT attention

**ĐÃ CÓ per-edge và join được ngay.** `azgomoku/explanation/model_evidence.py:30-35` xuất mỗi directed edge với `edge_id`, relation, source/target action, `head_attention`, mean `attention`, aggregation và layer. Test `tests/test_explanation.py:31-36` đối chiếu giá trị export với tensor `relation_attention`. Attention có thể phủ đúng graph edge rồi so với proof edge qua cầu join ở A2.

### B3. R-GCN structural

**CÓ structural anchor trong pipeline E-3b, nhưng chưa có cùng contract export.**

- Export chuẩn của R-GCN ghi `evidence_kind="structural_relations"`, edge có `attention=None`, và limitation “no learned attention coefficients” (`azgomoku/explanation/model_evidence.py:22-37`; test `tests/test_explanation.py:25-28`).
- E-3b dựng score cố định `1 / indegree(relation,target)` ở mức edge tại `investigation/e3b_graph.py:27-42`, gắn cùng `edge_id`, source, target. Đây đúng là structural baseline by design, không phải evidence học được.

Vì vậy overlay R-GCN khả thi ngay trong E-3b, nhưng field structural score trong `graph_evidence.edges`: **KHÔNG TÌM THẤY**. Cần một adapter/contract thống nhất nếu diagram chỉ đọc explanation export chuẩn.

### B4. Cùng hệ tọa độ

**ĐÃ CÓ.** Proof cells, policy, graph nodes, R-GAT/R-GCN edges và MCTS children đều dùng action index phẳng trên immutable pre-move board; row/col được suy bằng `divmod(action,size)`.

### Kết luận B

R-GAT join được ngay. R-GCN join được trong pipeline E-3b hiện tại, nhưng thiếu structural-score field trong explanation export chung. Ngoài ra cần materialize mapping proof → critical edge nếu renderer không muốn tự dùng quy tắc relation + cùng window.

## C. Cổng tọa độ / D4

### C1. Test và artifact xanh

**ĐÃ CÓ VÀ XANH.**

- Transform action, relation, critical cells và windows: `azgomoku/symmetry.py:10-58`.
- Self-check toàn cục: `azgomoku/symmetry.py:72-88`.
- Test round-trip state/proof/optimal actions cho cả 8 symmetry: `tests/test_h1_generator_v2.py:7-19`.
- Gate trên benchmark: `investigation/e3b_graph.py:45-84`.
- Artifact thật `results/h1_integration/e3b/graph_gate.json`: `passed=true`, 94 records, 243 proofs, **1.944 D4 proof round-trips** và **1.944 action-alignment checks**.

Gate transform toàn bộ flat proof, nên phủ `action`, `critical_cells`, `critical_relations` và `windows`; relation được transform bằng vector hướng. Test `tests/test_e3b_pipeline.py:58-70` còn chứng minh gate fail-closed khi proof không canonical.

### C2. Action ↔ policy ↔ MCTS ↔ graph

Phần proof/policy/graph được kiểm tra cho mọi 243 proof qua 1.944 checks. Runtime MCTS gate trong artifact chỉ chạy trên **2 representative states** (mid `42be4c9c478fcf87`, late `4dca2566ec2be9b6`), mỗi state có MCTS children đúng legal actions và proof actions đều hiện diện.

Do đó cổng tọa độ **xanh cho proof/D4 toàn bộ gold**, nhưng coverage runtime MCTS end-to-end chỉ là representative, chưa phải 94/94 state. Đây không phải blocker cho overlay tĩnh dùng action identity; nếu tuyên bố cổng MCTS runtime toàn tập thì cần mở rộng coverage.

### Kết luận C

**Cổng D4 sẵn và xanh; không phải blocker bắt buộc hiện tại.** Hạn chế duy nhất cần ghi đúng là runtime MCTS gate mới phủ 2 representative states.

## D. Phân biệt “trỏ lệch” và “attention collapse”

### D1. Dữ liệu và metric

**ĐÃ CÓ đủ dữ liệu và đã có metric riêng.** Per-edge/per-head attention cho phép đo phân tán theo nhóm softmax `(relation,target)`. `_collapse_metrics` tại `investigation/e3b_pipeline.py:146-184` tính:

- `attention_normalized_entropy`;
- `attention_structural_mae` so với `1/indegree`;
- `attention_head_diversity`;
- `attention_topology_correlation`;
- `attention_collapse_flag`, bật khi entropy ≥ 0,98, structural MAE ≤ 0,02 và head diversity ≤ 0,02.

Các field này nằm trong `METRIC_FIELDS` tại `investigation/e3b_pipeline.py:44-71`; test `tests/test_e3b_pipeline.py:42-55` xác nhận uniform attention tạo entropy 1, MAE 0 và collapse flag 1.

No-align thật được đo độc lập bằng proof alignment (`graph_critical_mass`, precision/recall@k, AUPRC, `alignment_minus_random`, `alignment_minus_structural`). Vì vậy:

- collapse: attention gần uniform/structural, entropy cao, diversity thấp;
- trỏ lệch: collapse flag không bật nhưng proof-alignment thấp.

### Kết luận D

**Phân biệt được ngay.** Không cần thêm đại lượng bắt buộc. Diagram cần đọc/hiển thị collapse flag hoặc các metric thành phần; SVG comparison hiện tại chưa hiển thị indicator này.

## E. Chỉ dựng trên gold complete

### E1. Cờ và bộ lọc

**ĐÃ CÓ và fail-closed.** Schema yêu cầu `status` và `optimal_actions_complete` tại `azgomoku/h1_schema.py:14-20`; semantics complete/partial/unknown được kiểm tra tại dòng 118-148. Reader gold `investigation/e3b_common.py:42-70` từ chối record nếu không đồng thời là `status=exact_complete`, `method=full_minimax`, `optimal_actions_complete=true`, eligible và đúng 6x6/k=4. Test `tests/test_e3b_pipeline.py:73-94` xác nhận một partial hợp lệ vẫn bị gold reader từ chối.

Partial vẫn có thể render như “tri thức một phần” vì có `status=exact_partial`, `optimal_actions_complete=false`, `coverage_note` và `_validation.label_kind`; nhưng SVG hiện tại chưa render nhãn partial. Field dữ liệu: có. Nhãn render: **KHÔNG TÌM THẤY**.

### E2. Gold set 94

**ĐÃ Ở dạng đọc được.**

- Nguồn: `results/h1_integration/e3a2/expanded_gold.jsonl`.
- Prepared: `results/h1_integration/e3b/prepared_gold.jsonl`.
- Frozen: `diagnostic/h1_benchmark_v1/h1_benchmark_v1.jsonl`.
- Manifest `diagnostic/h1_benchmark_v1/manifest.json:1-44`: n=94, 23 mid + 71 late, exact agreement 94/94, 83 proof-bearing states, 243 proof replay passed.

### Kết luận E

Gold-only lọc được ngay. Partial có đủ cờ để gắn nhãn “tri thức một phần”, nhưng renderer phải hiển thị cờ; không được dùng partial như complete.

## F. Hiện trạng SVG và khoảng cách

### F1. SVG explainer quyết định hiện tại

Renderer chung:

- `azgomoku/explanation/rendering/board_svg.py`: board và selected move;
- `azgomoku/explanation/rendering/decision_svg.py`: board + policy/MCTS + relational evidence;
- `azgomoku/explanation/rendering/graph_svg.py`: relational edges, top-k attention và chú thích R-GCN/R-GAT;
- entry point `azgomoku/explanation/explanation_export.py:25-53`.

Nó vẽ immutable pre-move state, selected action, policy/MCTS trace và attention/structural relations; chưa nhận ground-truth proof.

Renderer E-3b gần mục tiêu hơn: `investigation/e3b_graph.py:110-192` đã chồng critical cells, proof windows, R-GAT learned attention và R-GCN structural baseline trên cùng board, đồng thời ghi action/relation. Hai artifact thật ở `results/h1_integration/e3b/figures/mid_42be4c9c478fcf87.svg` và `.../late_4dca2566ec2be9b6.svg`.

### F2. Phần phải thêm để thành knowledge-diagram đầy đủ

1. Vẽ tất cả proof hoặc có bộ chọn proof; hiện chỉ `[0]`.
2. Materialize/hiển thị proof critical edges và concept label; hiện chỉ tô cell/window và ghi relation tổng quát.
3. Nếu dùng VCF: panel cây OR/AND/terminal; dữ liệu chỉ có cho 1 certificate.
4. Collapse indicator đọc metric đã có.
5. Gold/partial badge đọc `status` + `optimal_actions_complete`.
6. Export/adapter thống nhất R-GCN structural score với R-GAT `graph_evidence`.
7. Ghi rõ 11/94 gold states không có replayed proof nên không đủ lớp nền tactical để vẽ contrast proof-aware; manifest cho biết 83 proof-bearing states.

### F3. Ước lượng tái sử dụng

Đây là ước lượng theo bề mặt renderer, không phải đo LOC sản phẩm:

- Nếu phát triển từ **E-3b comparison SVG**: tái sử dụng khoảng **65–75%** (board geometry, node/edge coordinates, attention ranking/style, critical cells, windows, hai panel). Cần thêm 25–35% cho proof enumeration, concept/critical-edge semantics, tree/collapse/status UI và export adapter.
- Nếu phát triển từ **decision SVG chung**: tái sử dụng khoảng **40–50%** vì có board/policy/MCTS/graph primitives nhưng thiếu toàn bộ proof layer.

## Kết luận audit bắt buộc

1. **Lớp nền solver:** đủ dữ kiện ngay cho proof phẳng trên 83/94 gold states; thiếu concept-per-edge/window và thiếu proof tree cho 242/243 tactical proof.
2. **Lớp phủ + join:** R-GAT join được ngay bằng action/edge identity; R-GCN join được qua `structural_edges`, nhưng thiếu structural-score field trong explanation export chung; critical proof edge hiện là phép suy relation + window, chưa là field materialized.
3. **Cổng tọa độ D4:** xanh sẵn — artifact `graph_gate.json` pass 1.944/1.944 proof/action round-trips. Runtime MCTS end-to-end mới kiểm tra 2 representative states.
4. **Collapse vs no-align:** phân biệt được bằng metric hiện có; renderer cần hiển thị chúng.
5. **Gold-only:** lọc được ngay, fail-closed bằng `status=exact_complete` + `optimal_actions_complete=true`; partial có cờ để gắn nhãn nhưng SVG chưa hiển thị.
6. **Phán quyết:** contrast knowledge-diagram **khả thi ngay cho phiên bản proof-phẳng trên gold proof-bearing states**. Phiên bản tuyên bố là proof-graph/cây lập luận đầy đủ chỉ khả thi sau khi dựng: (a) concept/critical-edge mapping rõ ràng, (b) UI nhiều proof, (c) proof-tree coverage hoặc cách diễn đạt trung thực rằng tactical proof là phẳng, và (d) contract export structural score thống nhất. Repo **không có knowledge graph của model**; audit này chỉ xác nhận evidence overlay, đúng theo ranh giới đề bài.

Không có kết luận “nên làm hay không”; báo cáo chỉ xác định dữ kiện hiện có và khoảng thiếu.
