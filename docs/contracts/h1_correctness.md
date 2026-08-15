# H1 Correctness Contract

- `V*(s)` is always from `state.to_play`'s perspective: `+1` forced win, `0` draw, `-1` forced loss.
- `A*(s)` contains **all** legal actions attaining the maximum exact game-theoretic value.
- Solver APIs return `optimal_actions` and `action_values`; a single `best_action` API is prohibited.
- `status == "exact"` is required for benchmark inclusion. Timeout and node-budget results never become ground truth.
- Solver results and explanation proofs are distinct. Tactical proofs are independently extracted and certified from board geometry.
- Multiple valid proofs are stored in `valid_proofs[]`; no unique ground-truth mask is assumed.
- Explanation matching is action-conditioned. For a selected optimal action, report both best-valid-proof and mean-valid-proof alignment over all preregistered proofs for that action.
- HAN global β is model-level metadata, never state-level explanation evidence.
