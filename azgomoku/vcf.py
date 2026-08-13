"""Sound, one-sided Victory by Continuous Fours solver."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Literal

from .game import GomokuState
from .tactics import (
    creates_five,
    mandatory_defenses,
    threat_moves,
    windows,
    winning_completions,
)


class ProofState(Enum):
    PROVEN = "proven"
    DISPROVEN = "disproven"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class SolverBudget:
    node_cap: int | None = None
    time_cap_ms: int | None = None


@dataclass(frozen=True)
class ProofNode:
    player_to_move: int
    move: int | None
    node_type: Literal["OR", "AND"]
    children: tuple["ProofNode", ...] = ()
    terminal: Literal["five", "unstoppable_double_threat"] | None = None

    def dict(self) -> dict:
        result = asdict(self)
        result["children"] = [child.dict() for child in self.children]
        return result


@dataclass(frozen=True)
class VCFResult:
    status: Literal["exact_partial", "unknown"]
    value: int | None
    optimal_actions: tuple[int, ...] | None
    optimal_actions_complete: bool
    action_values: dict[int, int] | None
    method: Literal["vcf"]
    proof: ProofNode | None
    valid_proofs: tuple[dict, ...]
    nodes: int
    elapsed_ms: float
    budget: SolverBudget
    unknown_reason: str | None
    coverage_note: str

    def dict(self) -> dict:
        result = asdict(self)
        result["optimal_actions"] = (
            None if self.optimal_actions is None else list(self.optimal_actions)
        )
        result["action_values"] = (
            None
            if self.action_values is None
            else {str(key): value for key, value in self.action_values.items()}
        )
        result["proof"] = None if self.proof is None else self.proof.dict()
        result["valid_proofs"] = list(self.valid_proofs)
        return result


class _BudgetExceeded(RuntimeError):
    def __init__(self, reason: Literal["time_cap", "node_cap"]):
        self.reason = reason


class _Context:
    def __init__(self, budget: SolverBudget, attacker: int):
        self.budget = budget
        self.attacker = attacker
        self.start = time.perf_counter()
        self.nodes = 0
        self.limit_reason: str | None = None

    def enter(self) -> None:
        elapsed_ms = (time.perf_counter() - self.start) * 1000
        if self.budget.time_cap_ms is not None and elapsed_ms >= self.budget.time_cap_ms:
            self.limit_reason = "time_cap"
            raise _BudgetExceeded("time_cap")
        if self.budget.node_cap is not None and self.nodes >= self.budget.node_cap:
            self.limit_reason = "node_cap"
            raise _BudgetExceeded("node_cap")
        self.nodes += 1

    @property
    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self.start) * 1000


def _node_type(state: GomokuState, attacker: int) -> Literal["OR", "AND"]:
    return "OR" if state.to_play == attacker else "AND"


def _terminal_node(
    state: GomokuState,
    attacker: int,
    move: int,
    terminal: Literal["five", "unstoppable_double_threat"],
) -> ProofNode:
    return ProofNode(
        player_to_move=int(state.to_play),
        move=int(move),
        node_type=_node_type(state, attacker),
        terminal=terminal,
    )


def _prove_or(state: GomokuState, ctx: _Context) -> tuple[ProofState, ProofNode | None]:
    ctx.enter()
    if state.to_play != ctx.attacker or state.terminal():
        return ProofState.DISPROVEN, None
    saw_unknown = False
    # D2 tempo lock: the VCF OR-node consumes only five/four candidates.
    for candidate in threat_moves(state, ctx.attacker, "vcf"):
        move = candidate.move
        child = state.play(move)
        if child.winner() == ctx.attacker:
            return ProofState.PROVEN, _terminal_node(child, ctx.attacker, move, "five")
        try:
            result, certificate = _prove_and(child, ctx, move)
        except _BudgetExceeded:
            result, certificate = ProofState.UNKNOWN, None
        if result is ProofState.PROVEN:
            return ProofState.PROVEN, certificate
        if result is ProofState.UNKNOWN:
            saw_unknown = True
    return (ProofState.UNKNOWN if saw_unknown else ProofState.DISPROVEN), None


def _prove_and(
    state: GomokuState, ctx: _Context, attacker_move: int
) -> tuple[ProofState, ProofNode | None]:
    ctx.enter()
    if state.to_play == ctx.attacker:
        return ProofState.DISPROVEN, None
    defender = int(state.to_play)
    # A defender five is played now, before any pending attacker completion.
    # This check must precede the double-threat shortcut.
    if winning_completions(state, defender):
        return ProofState.DISPROVEN, None
    defenses = mandatory_defenses(state, ctx.attacker)
    if defenses.unstoppable:
        return (
            ProofState.PROVEN,
            _terminal_node(
                state, ctx.attacker, attacker_move, "unstoppable_double_threat"
            ),
        )
    if not defenses.blocking_moves:
        return ProofState.DISPROVEN, None

    children = []
    saw_unknown = False
    for defense in defenses.blocking_moves:
        child = state.play(defense)
        if child.winner() == defender:
            return ProofState.DISPROVEN, None
        attacker_can_finish = bool(winning_completions(child, ctx.attacker))
        defender_counter_four = bool(winning_completions(child, defender))
        if defender_counter_four and not attacker_can_finish:
            return ProofState.DISPROVEN, None
        try:
            result, continuation = _prove_or(child, ctx)
        except _BudgetExceeded:
            result, continuation = ProofState.UNKNOWN, None
        if result is ProofState.DISPROVEN:
            return ProofState.DISPROVEN, None
        if result is ProofState.UNKNOWN:
            saw_unknown = True
            continue
        if continuation is None:
            raise AssertionError("PROVEN branch lacks a certificate")
        children.append(
            ProofNode(
                player_to_move=int(child.to_play),
                move=int(defense),
                node_type="OR",
                children=(continuation,),
            )
        )
    if saw_unknown:
        return ProofState.UNKNOWN, None
    return (
        ProofState.PROVEN,
        ProofNode(
            player_to_move=int(state.to_play),
            move=int(attacker_move),
            node_type="AND",
            children=tuple(children),
        ),
    )


def _required_and_moves(state: GomokuState, attacker: int) -> tuple[int, ...] | None:
    defenses = mandatory_defenses(state, attacker)
    if defenses.unstoppable:
        return None
    return defenses.blocking_moves


def replay_vcf_proof(state: GomokuState, proof: ProofNode) -> bool:
    """Replay all certified defender branches; fail closed on any mismatch."""

    attacker = int(state.to_play)
    if proof.move is not None or proof.player_to_move != attacker or proof.node_type != "OR":
        return False

    def replay_node(current: GomokuState, node: ProofNode) -> bool:
        if node.player_to_move != current.to_play or node.node_type != _node_type(current, attacker):
            return False
        if node.terminal == "five":
            return not node.children and current.winner() == attacker
        if node.terminal == "unstoppable_double_threat":
            return (
                not node.children
                and current.to_play == -attacker
                and not winning_completions(current, -attacker)
                and mandatory_defenses(current, attacker).unstoppable
            )
        if node.terminal is not None or not node.children:
            return False
        moves = [child.move for child in node.children]
        if any(move is None for move in moves) or len(set(moves)) != len(moves):
            return False
        legal = set(map(int, current.legal_actions()))
        if not set(moves) <= legal:
            return False
        if node.node_type == "OR":
            if len(node.children) != 1:
                return False
        else:
            required = _required_and_moves(current, attacker)
            if required is None or set(moves) != set(required):
                return False
        for child in node.children:
            next_state = current.play(int(child.move))
            if not replay_node(next_state, child):
                return False
        return True

    return replay_node(state, proof)


def reduce_vcf_proof(state: GomokuState, proof: ProofNode) -> dict:
    """Project a replayable tree onto the existing flat H1 proof shape."""

    if not replay_vcf_proof(state, proof):
        raise ValueError("cannot reduce an invalid VCF proof")
    attacker = int(state.to_play)
    attacker_moves = set()
    terminal_windows = set()

    def visit(current: GomokuState, node: ProofNode) -> None:
        if node.terminal == "five":
            flat = current.board.reshape(-1)
            for relation, cells in windows(current.size, current.win_length):
                if all(flat[cell] == attacker for cell in cells):
                    terminal_windows.add((relation, tuple(cells)))
        elif node.terminal == "unstoppable_double_threat":
            for threat in winning_completions(current, attacker):
                terminal_windows.add((threat.relation, threat.window))
        for child in node.children:
            if node.node_type == "OR":
                attacker_moves.add(int(child.move))
            visit(current.play(int(child.move)), child)

    visit(state, proof)
    root_action = int(proof.children[0].move)
    ordered_windows = sorted({cells for _, cells in terminal_windows})
    relations = sorted({relation for relation, _ in terminal_windows})
    return {
        "action": root_action,
        "concepts": ["winning_line", "forced_sequence", "vcf"],
        "critical_cells": sorted(attacker_moves),
        "critical_relations": relations,
        "windows": [list(cells) for cells in ordered_windows],
        "proof_method": "vcf",
        "proof_status": "exact",
    }


def solve_vcf(
    state: GomokuState, node_cap: int = 100_000, time_cap_ms: int = 1_000
) -> VCFResult:
    """Prove one attacker VCF or abstain; never infer draw/loss."""

    budget = SolverBudget(node_cap=node_cap, time_cap_ms=time_cap_ms)
    ctx = _Context(budget, int(state.to_play))
    root = ProofNode(int(state.to_play), None, "OR")
    try:
        result, continuation = _prove_or(state, ctx)
    except _BudgetExceeded:
        result, continuation = ProofState.UNKNOWN, None
    proof = None
    if result is ProofState.PROVEN and continuation is not None:
        proof = ProofNode(root.player_to_move, None, "OR", (continuation,))
        # Production gate: no certificate can become a label without replay.
        if not replay_vcf_proof(state, proof):
            result = ProofState.UNKNOWN
            proof = None
            ctx.limit_reason = "proof_replay_failed"
    if result is ProofState.PROVEN and proof is not None:
        action = int(proof.children[0].move)
        flat_proof = reduce_vcf_proof(state, proof)
        return VCFResult(
            "exact_partial",
            1,
            (action,),
            False,
            {action: 1},
            "vcf",
            proof,
            (flat_proof,),
            ctx.nodes,
            ctx.elapsed_ms,
            budget,
            None,
            "sound VCF existential win; optimal action set is incomplete",
        )
    reason = ctx.limit_reason or (
        "not_proven" if result is ProofState.DISPROVEN else "search_unknown"
    )
    return VCFResult(
        "unknown",
        None,
        None,
        False,
        None,
        "vcf",
        None,
        (),
        ctx.nodes,
        ctx.elapsed_ms,
        budget,
        reason,
        "VCF did not prove a win; no ground-truth label emitted",
    )
