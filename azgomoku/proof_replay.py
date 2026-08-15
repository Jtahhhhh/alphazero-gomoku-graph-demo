"""Fail-closed replay primitives for tactical and VCF proofs."""

from __future__ import annotations

from azgomoku.h1_schema import proof_node_from_dict, state_from_record
from azgomoku.tactics import mandatory_defenses, winning_completions, windows
from azgomoku.vcf import replay_vcf_proof


def _geometry_ok(state, proof: dict) -> bool:
    size = state.size
    cells = list(map(int, proof.get("critical_cells", [])))
    action = int(proof.get("action", -1))
    if action not in set(map(int, state.legal_actions())):
        return False
    if any(cell < 0 or cell >= size * size for cell in cells):
        return False
    known = {(relation, tuple(cell_list)) for relation, cell_list in windows(size, state.win_length)}
    relations = set(proof.get("critical_relations", []))
    proof_windows = [tuple(map(int, item)) for item in proof.get("windows", [])]
    if not proof_windows or not relations:
        return False
    return all(any((relation, window) in known for relation in relations) for window in proof_windows)


def replay_flat_proof(state, proof: dict, certificate: dict | None = None) -> bool:
    """Replay a reduced tactical/VCF annotation without trusting its text labels."""
    if not _geometry_ok(state, proof):
        return False
    action = int(proof["action"])
    method = proof.get("proof_method")
    if method == "vcf":
        if certificate is None:
            return False
        try:
            tree = proof_node_from_dict(certificate["tree"])
        except (KeyError, TypeError, ValueError):
            return False
        return int(certificate.get("action", -1)) == action and replay_vcf_proof(state, tree)

    concepts = set(proof.get("concepts", []))
    player = int(state.to_play)
    child = state.play(action)
    if "immediate_win" in concepts:
        return child.winner() == player
    if "mandatory_block" in concepts:
        defenses = mandatory_defenses(state, -player)
        return not defenses.unstoppable and action in defenses.blocking_moves
    if "simple_fork" in concepts:
        completions = {item.completion for item in winning_completions(child, player)}
        return len(completions) >= 2
    return False


def replay_record_proofs(record: dict) -> tuple[int, int]:
    state = state_from_record(record)
    certificates = {item["certificate_id"]: item for item in record.get("proof_certificates", [])}
    passed = 0
    for proof in record.get("valid_proofs", []):
        certificate = certificates.get(proof.get("certificate_id"))
        if replay_flat_proof(state, proof, certificate):
            passed += 1
    return passed, len(record.get("valid_proofs", []))
