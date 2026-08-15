"""Sound ground-truth routing: verified exact solver first, then replayed VCF."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass

from .game import GomokuState
from .solver import solve_actions
from .vcf import ProofNode, replay_vcf_proof, solve_vcf


@dataclass(frozen=True)
class GroundTruthBudget:
    node_cap: int = 1_000_000
    time_cap_ms: int = 2_000


@dataclass(frozen=True)
class GroundTruthResult:
    status: str
    method: str
    value: int | None
    optimal_actions: tuple[int, ...] | None
    optimal_actions_complete: bool
    action_values: dict[int, int] | None
    proof: ProofNode | None
    valid_proofs: tuple[dict, ...]
    nodes: int
    elapsed_ms: float
    budget: GroundTruthBudget
    unknown_reason: str | None
    coverage_note: str

    def dict(self) -> dict:
        result=asdict(self)
        result["optimal_actions"]=None if self.optimal_actions is None else list(self.optimal_actions)
        result["action_values"]=None if self.action_values is None else {str(k):v for k,v in self.action_values.items()}
        result["proof"]=None if self.proof is None else self.proof.dict()
        result["valid_proofs"]=list(self.valid_proofs)
        result["perspective"]={"convention_version":2,"value":"player_to_move_at_state"}
        return result


class RouterStats:
    def __init__(self): self.counts=Counter()

    def observe(self,result: GroundTruthResult) -> None: self.counts[result.status]+=1

    def dict(self) -> dict:
        total=sum(self.counts.values())
        return {
            "total":total,
            "counts":{key:int(self.counts.get(key,0)) for key in ("exact_complete","exact_partial","unknown")},
            "rates":{key:(self.counts.get(key,0)/total if total else 0.) for key in ("exact_complete","exact_partial","unknown")},
        }


# Only 6x6 has a calibrated bound from D2c-v2. Missing entries mean "try exact";
# board size is never itself a soundness decision.
DEFAULT_EMPTY_BOUNDS={6:15}


def route_ground_truth(
    state: GomokuState,
    budget: GroundTruthBudget=GroundTruthBudget(),
    *,
    empty_bounds: dict[int,int] | None=None,
    stats: RouterStats | None=None,
) -> GroundTruthResult:
    bounds=DEFAULT_EMPTY_BOUNDS if empty_bounds is None else empty_bounds
    empty_count=len(state.legal_actions())
    skip_exact=state.size in bounds and empty_count>bounds[state.size]
    if not skip_exact:
        # This is deliberately the original verified solver, never offline_solver.
        exact=solve_actions(state,deadline_ms=budget.time_cap_ms,node_budget=budget.node_cap)
        if exact.status=="exact":
            result=GroundTruthResult(
                "exact_complete","full_minimax",exact.value,exact.optimal_actions,True,
                exact.action_values,None,(),exact.nodes,exact.elapsed_ms,budget,None,
                "verified full-width root action values",
            )
            if stats is not None: stats.observe(result)
            return result

    vcf=solve_vcf(state,node_cap=budget.node_cap,time_cap_ms=budget.time_cap_ms)
    if vcf.status=="exact_partial" and vcf.proof is not None and replay_vcf_proof(state,vcf.proof):
        result=GroundTruthResult(
            "exact_partial","vcf",vcf.value,vcf.optimal_actions,False,vcf.action_values,
            vcf.proof,vcf.valid_proofs,vcf.nodes,vcf.elapsed_ms,budget,None,
            "replay-verified existential VCF win; proven action set is incomplete",
        )
    else:
        reason=vcf.unknown_reason if vcf.unknown_reason in ("budget","exhausted") else "exhausted"
        result=GroundTruthResult(
            "unknown","vcf",None,None,False,None,None,(),vcf.nodes,vcf.elapsed_ms,
            budget,reason,"no replay-verified label emitted",
        )
    if stats is not None: stats.observe(result)
    return result
