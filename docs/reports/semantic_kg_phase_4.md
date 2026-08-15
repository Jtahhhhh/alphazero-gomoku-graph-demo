# Semantic KG Phase 4 — Learned Evidence Grounding

Ngày chạy: 2026-08-14

## Kết luận

Phase 4 tạo thành công một learned/search evidence overlay riêng, join chính xác
vào Semantic KG v1 mà không sửa bốn file base hoặc frozen H1 benchmark.

Trạng thái: **PASS**.

## Immutable base

Bốn file base được khóa tại `semantic_kg/base_freeze.json`:

| File | SHA-256 |
|---|---|
| `entities.jsonl` | `784f1cb9c883f38e8d19bb779ef1a6be1b9175743fc9bd79c7e9f9c6f4b3d1d7` |
| `facts.jsonl` | `da793f2e66c5a953686175a22549618974cccd38712ee59e7e93e83c1284355e` |
| `provenance.jsonl` | `00c5f2fa2aa4ead063efdbe82db43133bd1d33776a45b0f8334a406031481594` |
| `manifest.json` | `28c2715fbc9b16a75dc7e849a907a807c5097918ff49a92fd0b76583533998a8` |

Các hash được kiểm tra lại sau D4 gate và sau evidence export; tất cả giữ nguyên.

Frozen source cũng giữ nguyên:

- H1 JSONL: `9abd52ef4991489586682e881e495fcb4c2ffe00fb55dc9dee1d9008aca4ff02`
- H1 manifest: `7498c6379f139b470cf8fd3b273e5085c28d4ab5427c134f59aa0df8fca2f8b1`

## Full D4 release gate

`semantic_kg/d4_release_gate.json` ghi nhận:

- 94 states × 8 transforms = 752 transform checks
- 259.960 canonical entity checks
- 627.744 semantic fact-transform checks
- 630.720 referential-integrity checks
- 630.720 epistemic-invariance checks
- 752 fact-count invariance checks
- 1.944 certified proof-lineage checks
- 752 semantic round-trips
- raw IDs được tái dựng từ transformed state và mọi entity giữ đúng transformed
  `state_id` scope

Tất cả đều PASS. Raw ID không bị yêu cầu giống orientation gốc; canonical key
phải và đã invariant.

## Evidence contract v1.1

Contract nằm tại `docs/contracts/semantic_evidence_v1.md`.

Các đại lượng được tách riêng:

- network policy: `HAS_POLICY_PROB`
- network value: `HAS_STATE_VALUE`
- attention: `OBSERVES`, `HAS_ATTENTION_WEIGHT`
- MCTS: `HAS_MCTS_PRIOR`, `HAS_VISITS`, `HAS_Q`, `HAS_SEARCH_PROB`, `IS_SELECTED`

Không reuse solver `HAS_ACTION_VALUE`. Overlay không có tactical truth predicate,
`Pattern`, `OpenEnd`, `EXTENDS`, `HAS_OPEN_END`, hoặc heuristic fact.

## Evidence sources

- Models: R-GAT và R-GCN, seed 7
- Network checkpoints: iter 0, 5, ..., 60 cho mỗi model; tổng cộng 26
- R-GAT attention checkpoints: 13
- MCTS checkpoints/model: iter 0, 20, 40, 60
- MCTS budget: 50 playouts, `c_puct=1.5`, temperature 1.0, no root noise
- Endpoint: iter60 R-GAT và R-GCN
- States/checkpoint: toàn bộ 94 frozen states

R-GCN không expose learned attention coefficients, nên Phase 4 không tạo pseudo-
attention cho R-GCN. R-GCN vẫn có policy, value và MCTS evidence; structural graph
được join từ base `StructuralEdge`.

## Overlay cardinality

### Full developmental + endpoint overlay

| Chỉ số | Count |
|---|---:|
| Entities | 286.488 |
| Facts | 703.368 |
| Provenance | 272.036 |
| External base references | 346.288 |
| LEARNED | 416.880 |
| DERIVED joins | 286.488 |
| EXACT | 0 |
| CERTIFIED | 0 |
| HEURISTIC | 0 |

Entity types:

- `AttentionObservation`: 268.840 = 20.680 edges × 13 R-GAT checkpoints
- `MCTSCandidate`: 17.648 = 2.206 moves × 8 model-checkpoint searches

Predicate counts:

| Predicate | Count |
|---|---:|
| HAS_POLICY_PROB | 57.356 |
| HAS_STATE_VALUE | 2.444 |
| OBSERVES | 268.840 |
| HAS_ATTENTION_WEIGHT | 268.840 |
| REFERS_TO_MOVE | 17.648 |
| HAS_MCTS_PRIOR | 17.648 |
| HAS_VISITS | 17.648 |
| HAS_Q | 17.648 |
| HAS_SEARCH_PROB | 17.648 |
| IS_SELECTED | 17.648 |

### Endpoint overlay

| Chỉ số | Count |
|---|---:|
| Entities | 25.092 |
| Facts | 72.432 |
| Provenance | 21.056 |
| External base references | 29.692 |
| LEARNED | 47.340 |
| DERIVED joins | 25.092 |

## Provenance và join integrity

Mỗi network observation lưu model type/mode, checkpoint path/SHA/iteration,
training seed, generator version, board size và win length.

Attention lưu thêm exact edge ID, final layer, head scope `all` và locked
aggregation `mean across attention heads`. MCTS lưu playouts, search seed,
temperature, selection mode, root convention và `c_puct`.

Các join fail closed:

- policy fact → base `Move` → base `Cell`/`BoardState`
- attention observation → base `StructuralEdge`
- MCTS candidate → base `Move`
- không fuzzy coordinate matching

`semantic_evidence_v1/evidence_release_gate.json` đối chiếu count bằng công thức
từ base `2.206 Move` và `20.680 StructuralEdge`, đồng thời kiểm tra hash của cả
full và endpoint artifact.

## Files

- `azgomoku/semantic/evidence_schema.py`
- `investigation/semantic_evidence_export.py`
- `docs/contracts/semantic_evidence_v1.md`
- `tests/test_semantic_evidence.py`
- `tests/test_semantic_evidence_join.py`
- `tests/test_semantic_evidence_provenance.py`
- `tests/test_semantic_full_d4.py`
- `semantic_evidence_v1/entities.jsonl`
- `semantic_evidence_v1/facts.jsonl`
- `semantic_evidence_v1/provenance.jsonl`
- `semantic_evidence_v1/manifest.json`
- `semantic_evidence_v1/endpoint/*`
- `semantic_evidence_v1/evidence_release_gate.json`

## Giới hạn được giữ rõ

- Attention là model observation, không phải causal proof.
- MCTS result là observational fixed-budget evidence.
- R-GCN structural reference không được báo như learned-attention finding.
- Phase 4 không thực hiện statistical interpretation; việc đó thuộc Phase 5.

## Xác minh phát hành

- Full D4 release gate: PASS trên `94 × 8 = 752` transforms, gồm canonical
  identity, fact semantics/count, referential integrity, epistemic labels, proof
  lineage, raw-state scope và round-trip.
- Evidence release gate: PASS; hash của bốn base-KG files khớp freeze record sau
  export.
- Quét cả full và endpoint overlay: không có tactical-truth predicate bị cấm.
- Semantic/evidence tests: `26 passed`; toàn bộ repository: `114 passed` với
  UTF-8 mode.
