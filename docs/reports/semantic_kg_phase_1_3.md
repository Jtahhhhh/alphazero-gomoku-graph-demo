# Semantic KG Phase 1–3 Report

Ngày hoàn tất: 2026-08-14

## Kết luận

Phase 1–3 đã được triển khai thành một lớp semantic adapter độc lập, không sửa
solver, tactics, VCF, kiến trúc model, MCTS hay SVG renderer. Frozen H1 benchmark
được đọc như nguồn bất biến và đã được xuất thành Semantic KG v1.

Trạng thái: **PASS — Level 3, provenance-aware Semantic KG cho dữ liệu frozen hiện có.**

Semantic experiment, query layer và ontology `Pattern`/`OpenEnd` chưa được triển
khai, đúng với phạm vi đã khóa.

## Thành phần đã thêm

### Phase 1 — contract và validation

- `azgomoku/semantic/schema.py`
- `azgomoku/semantic/predicates.py`
- `azgomoku/semantic/epistemic.py`
- `azgomoku/semantic/validation.py`
- `docs/contracts/semantic_kg_v1.md`
- `tests/test_semantic_schema.py`
- `tests/test_semantic_validation.py`

Schema dùng closed vocabulary cho entity type, predicate và năm epistemic class:
`EXACT`, `CERTIFIED`, `DERIVED`, `HEURISTIC`, `LEARNED`. Validator áp dụng XOR
`object_id`/`value`, referential integrity, provenance bắt buộc và các hard gate
epistemic. `Pattern`, `OpenEnd`, `EXTENDS` và `HAS_OPEN_END` không tồn tại trong
vocabulary v1.

### Phase 2 — stable identity

- `azgomoku/semantic/identity.py`
- `docs/contracts/semantic_identity.md`
- `tests/test_semantic_identity.py`
- `tests/test_semantic_d4_identity.py`

Mỗi object có raw ID scope theo state và canonical key dưới D4. Canonical key
được chọn từ cặp `(transformed state, transformed semantic payload)` qua đủ tám
phép đối xứng; cách này tránh nhập nhầm hai cell/move khác nhau trên một bàn cờ
bất đối xứng. Proof ID được băm từ nội dung chuẩn hóa, không phụ thuộc list order
hay nhãn hiển thị P1/P2.

### Phase 3 — extraction và export

- `azgomoku/semantic/extract_state.py`
- `azgomoku/semantic/extract_tactics.py`
- `azgomoku/semantic/extract_proofs.py`
- `azgomoku/semantic/extract_evidence.py`
- `azgomoku/semantic/export_kg.py`
- `tests/test_semantic_extraction.py`
- `tests/test_semantic_export.py`
- `semantic_kg/entities.jsonl`
- `semantic_kg/facts.jsonl`
- `semantic_kg/provenance.jsonl`
- `semantic_kg/manifest.json`
- `semantic_kg/pilot/*`

Extractor tactics chỉ gọi logic tactics hiện có. Proof tactical và VCF đều được
replay trước khi phát sinh fact `CERTIFIED`; không có proof giả cho no-proof
state. Geometry/membership được giữ ở `DERIVED`, còn full-minimax action truth là
`EXACT`. Learned evidence chỉ được phép phát sinh `LEARNED`/`DERIVED`, không được
phát sinh tactical truth.

## Pilot gate

Pilot cố định gồm:

- tactical proof: `4dca2566ec2be9b6`
- VCF certificate: `b3a6c7628630359d`
- exact-complete no-proof: `74c55e1c7c911cc9`

Kết quả:

- 3 states, trong đó 2 proof-bearing và 1 no-proof
- 3 replay-backed proofs
- 1.035 entities
- 2.418 facts
- `HEURISTIC = 0`
- D4 gate: 24 record transforms, 8.280 entity canonical checks và 19.296 fact
  transform checks, tất cả đều pass

## Full frozen export

### Cardinality

| Chỉ số | Giá trị |
|---|---:|
| States | 94 |
| Proof-bearing states | 83 |
| No-proof states | 11 |
| Replay-backed proofs | 243 |
| Entities | 32.495 |
| Facts | 78.840 |
| Provenance records | 1.144 |

### Entity types

| Entity type | Count |
|---|---:|
| BoardState | 94 |
| Cell | 3.384 |
| Move | 2.206 |
| LineWindow | 5.076 |
| WinningThreat | 654 |
| DefenseSet | 94 |
| ForcedResponse | 58 |
| Proof | 243 |
| ProofNode | 6 |
| StructuralEdge | 20.680 |

`AttentionObservation` và `MCTSCandidate` không xuất hiện trong frozen artifact vì
H1 benchmark không lưu learned/search trace; adapter cho hai loại này được kiểm
thử bằng fixture riêng.

### Epistemic counts

| Class | Count |
|---|---:|
| EXACT | 3.324 |
| CERTIFIED | 243 |
| DERIVED | 75.273 |
| HEURISTIC | 0 |
| LEARNED | 0 |

### Predicate counts

| Predicate | Count |
|---|---:|
| BLOCKS | 56 |
| CONNECTS | 41.360 |
| CONTAINS | 3.390 |
| CREATES | 1.754 |
| FORCES | 56 |
| HAS_ACTION_VALUE | 2.206 |
| HAS_COMPLETION | 716 |
| HAS_DIRECTION | 716 |
| HAS_WINDOW | 386 |
| OPTIMAL_IN | 1.118 |
| PLAYED_AT | 2.206 |
| REQUIRES | 2 |
| SUPPORTS | 243 |
| USES_CELL | 24.631 |

`HAS_WEIGHT` và `OVERLAPS` chỉ phát sinh khi có learned evidence, nên count bằng
0 trong frozen export. Các predicate ngoài contract không được emit.

## Integrity và tests

- Referential integrity: PASS
- Epistemic integrity: PASS
- Certified-proof replay lineage: PASS, 243/243
- Pilot D4 semantic equivalence: PASS
- Deterministic export: PASS, hai lần xuất cùng input byte-identical cho cả bốn file
- Frozen benchmark byte integrity: PASS cho cả JSONL và source manifest
- JSONL SHA-256 trước và sau:
  `9abd52ef4991489586682e881e495fcb4c2ffe00fb55dc9dee1d9008aca4ff02`
- Source manifest SHA-256 trước và sau:
  `7498c6379f139b470cf8fd3b273e5085c28d4ab5427c134f59aa0df8fca2f8b1`
- Semantic tests: 17/17 pass
- Full repository regression: 105/105 pass với `PYTHONUTF8=1`

Lần chạy full suite đầu tiên dưới code page CP1252 fail tại một test SVG có sẵn,
do test gọi `Path.read_text()` mà không chỉ định UTF-8. Bật UTF-8 mode làm toàn bộ
suite pass; không có thay đổi nào được thực hiện lên renderer/test cũ để né lỗi.

## Các điểm cần bàn trước semantic experiments

1. **Learned evidence bundle.** Frozen H1 không lưu policy, value, attention,
   checkpoint hoặc MCTS search trace, nên artifact hiện tại trung thực có
   `LEARNED = 0`. Thí nghiệm alignment cần một input bundle versioned chứa model
   checkpoint và search budget/config.
2. **DefenseSet linkage.** `DefenseSet` hiện là entity có payload xác định nhưng
   vocabulary v1 chưa có predicate membership/link riêng cho nó. Không tự thêm
   predicate trong Phase 3; cần quyết định contract nếu query layer cần truy vấn
   trực tiếp defense-set membership.
3. **D4 gate trên toàn corpus.** Gate bắt buộc đã chạy full semantic extraction
   trên 3 pilot × 8 transforms; identity unit tests bao phủ các loại còn lại.
   Chạy 94 × 8 là một gate mở rộng hữu ích trước khi đóng băng một bản release,
   nhưng không nằm trong acceptance target pilot của plan hiện tại.
4. **Windows test encoding.** Nên cấu hình `PYTHONUTF8=1` trong test runner hoặc
   sửa test SVG hiện có để đọc explicit UTF-8 ở một thay đổi riêng, ngoài phạm vi
   Semantic KG.

Không có blocker đối với việc đọc/query Semantic KG hiện tại. Blocker thực sự
cho experiment learned-vs-certified là thiếu learned evidence artifact có
provenance đầy đủ.
