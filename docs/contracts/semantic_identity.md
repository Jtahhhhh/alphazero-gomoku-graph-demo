# Semantic Identity Contract

Version: `semantic_identity_v1`

## Raw identity and canonical identity

Every Semantic KG entity has two independent identifiers:

- `entity_id` identifies the object in the current state and orientation.
- `canonical_key` identifies the D4-equivalence class of the pair `(state, semantic payload)`.

Canonicalizing the semantic payload alone is forbidden because it can collapse unrelated cells or moves on an asymmetric board. The implementation transforms both state and payload through all eight D4 symmetries, serializes each pair deterministically, and hashes the lexicographic minimum.

## Raw ID formats

```text
state:{state_id}
cell:{state_id}:r{row}c{col}
move:{state_id}:a{action}
line:{state_id}:{content_hash}
threat:{state_id}:{content_hash}
defense_set:{state_id}:{content_hash}
forced_response:{threat_or_proof_id}:a{action}
proof:{state_id}:{normalized_proof_hash}
proofnode:{proof_id}:{tree_path}
edge:{state_id}:{relation}:{source}:{target}
attention:{state_id}:{checkpoint_sha}:{legacy_edge_id}:{layer}
mcts:{state_id}:{search_config_hash}:a{action}
```

Candidate Move identity is implemented in v1. Historical move-event identity is reserved for a later contract.

## Normalization

Flat proof identity uses `action`, sorted concepts, sorted critical cells/relations, normalized/sorted windows, proof method, and proof status. Certificate IDs and renderer list positions are excluded. Reordering a proof list or changing `P1/P2` display order cannot change semantic proof identity.

Line windows normalize reversal. Threat identity includes state, player, relation, normalized window, and completion. Forced responses are scoped to a Threat or Proof identity so one coordinate may represent distinct responses in distinct forcing contexts.

Structural semantic edge IDs are state-scoped while retaining the legacy `relation:source:target` key as an attribute. Attention observations are distinct from structural edges and include checkpoint and layer identity. MCTS candidates include a deterministic search-config hash.

## D4 gates

For every supported semantic object:

```text
raw_id(original) may differ from raw_id(transformed)
canonical_key(original) == canonical_key(transformed)
```

The gate covers state, Cell, Move, LineWindow, WinningThreat, ForcedResponse, flat Proof, ProofNode subtree, StructuralEdge, AttentionObservation, and MCTSCandidate. It extends the existing coordinate/proof round-trip gate; it does not replace it.
