"""Oracle-agreement gate for sound threat solvers.

The default threat solver abstains. Future VCF/VCT implementations plug into the
same callable interface; any claimed winning action is checked against the full
6x6 exact solver before it can be treated as ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Protocol, Sequence

import numpy as np

from .game import GomokuState
from .solver import SolverResult, solve_actions


ThreatStatus = Literal["exact_partial", "unknown"]


@dataclass(frozen=True)
class ThreatClaim:
    status: ThreatStatus
    value: int | None
    optimal_actions: tuple[int, ...] | None
    optimal_actions_complete: bool
    action_values: dict[int, int] | None
    proof: object | None
    method: Literal["vcf", "vct", "dfpn"]
    unknown_reason: str | None


class ThreatSolver(Protocol):
    def __call__(self, state: GomokuState, node_cap: int, time_cap_ms: int) -> ThreatClaim: ...


@dataclass(frozen=True)
class OracleCase:
    state: GomokuState
    history: tuple[int, ...]
    seed: int
    oracle: SolverResult


@dataclass(frozen=True)
class AgreementSummary:
    checked_states: int
    abstained_states: int
    claimed_states: int
    checked_actions: int


def threat_solver_stub(
    state: GomokuState, node_cap: int, time_cap_ms: int
) -> ThreatClaim:
    """Pre-D2 injection point: sound by abstaining on every state."""

    del state, node_cap, time_cap_ms
    return ThreatClaim(
        status="unknown",
        value=None,
        optimal_actions=None,
        optimal_actions_complete=False,
        action_values=None,
        proof=None,
        method="vcf",
        unknown_reason="stub",
    )


def generate_oracle_cases(
    count: int,
    *,
    seed: int = 7,
    min_history: int = 26,
    max_history: int = 32,
    attempts: int = 10_000,
    oracle_node_cap: int = 1_000_000,
    oracle_time_cap_ms: int = 2_000,
) -> tuple[OracleCase, ...]:
    """Generate deterministic legal 6x6/k=4 histories accepted by the oracle."""

    if count < 0:
        raise ValueError("count must be non-negative")
    rng = np.random.default_rng(seed)
    cases = []
    for _ in range(attempts):
        if len(cases) >= count:
            break
        state = GomokuState.initial(size=6, win_length=4)
        history = []
        target = int(rng.integers(min_history, max_history + 1))
        while len(history) < target and not state.terminal():
            legal = state.legal_actions()
            action = int(rng.choice(legal))
            history.append(action)
            state = state.play(action)
        if state.terminal():
            continue
        oracle = solve_actions(
            state,
            deadline_ms=oracle_time_cap_ms,
            node_budget=oracle_node_cap,
        )
        if oracle.status == "exact":
            cases.append(OracleCase(state, tuple(history), seed, oracle))
    if len(cases) != count:
        raise RuntimeError(f"generated {len(cases)} exact cases, expected {count}")
    return tuple(cases)


def check_oracle_agreement(
    cases: Sequence[OracleCase],
    threat_solver: ThreatSolver = threat_solver_stub,
    *,
    threat_node_cap: int = 100_000,
    threat_time_cap_ms: int = 1_000,
    proof_validator: Callable[[GomokuState, object], bool] | None = None,
) -> AgreementSummary:
    """Fail on malformed claims or any threat-solver false-positive."""

    abstained = claimed = checked_actions = 0
    for index, case in enumerate(cases):
        if case.oracle.status != "exact":
            raise AssertionError(f"case {index}: oracle is not exact")
        claim = threat_solver(case.state, threat_node_cap, threat_time_cap_ms)
        if claim.status == "unknown":
            if any(
                value is not None
                for value in (claim.value, claim.optimal_actions, claim.action_values, claim.proof)
            ):
                raise AssertionError(f"case {index}: unknown claim carries a label or proof")
            if claim.optimal_actions_complete or not claim.unknown_reason:
                raise AssertionError(f"case {index}: malformed unknown claim")
            abstained += 1
            continue
        if claim.status != "exact_partial":
            raise AssertionError(f"case {index}: unsupported threat status {claim.status!r}")
        claimed += 1
        if claim.value != 1 or claim.optimal_actions_complete:
            raise AssertionError(f"case {index}: malformed exact_partial semantics")
        if not claim.optimal_actions or claim.action_values is None or claim.proof is None:
            raise AssertionError(f"case {index}: exact_partial lacks actions, values, or proof")
        if proof_validator is None or not proof_validator(case.state, claim.proof):
            raise AssertionError(f"case {index}: proof did not replay successfully")
        legal = set(map(int, case.state.legal_actions()))
        for action in claim.optimal_actions:
            if action not in legal or claim.action_values.get(action) != 1:
                raise AssertionError(f"case {index}: malformed proven action {action}")
            oracle_value = case.oracle.action_values.get(action)
            if oracle_value != 1:
                raise AssertionError(
                    "false-positive: "
                    f"case={index}, seed={case.seed}, history={list(case.history)}, "
                    f"action={action}, oracle_value={oracle_value}, method={claim.method}"
                )
            checked_actions += 1
    return AgreementSummary(len(cases), abstained, claimed, checked_actions)
