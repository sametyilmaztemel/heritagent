"""Frozen CIG allocations for EXP-0001 (issue #13 Phase A; manifest
``inheritance.stages``; ADR-0007 E6; recalibrated by ADR-0012, owner-accepted
2026-10-19).

- Stage 1 (ADR-0012 D3): the candidate's source task, the T_mine task of
  its source trajectory, alone.
- Stage 2 (D4): 192 benefit-bank games over the candidate's frozen
  applicability families in name order: allocation game j is the
  ⌊j/m⌋-th game, in bank order, of the (j mod m)-th family. Every candidate
  that needs n games of a family gets the same first n (cross-trait reuse
  is allowed and logged; within a trait there is no replacement).
- Stage 3 (D5): the in-scope child records stage 2's games by reference;
  out of scope, 24 benefit-bank games over the families the applicability
  omits, round-robin in name order, each family's first games in bank
  order. A candidate whose applicability names every family has no
  out-of-scope family: its out-of-scope stage is vacuous (ADR-0007 E6).
- Interaction: the 10-game interaction pool, shared by every pair.
- Regression: the one shared T_reg (ADR-0006 D8; ADR-0007 E3).

The bank's order is the manifest's list order (ascending by the bank's
selection hash); it is never re-sorted. ``replay_bank``,
``generalization_bank`` and the ablation pool stay in the split manifest,
unused by the gate (ADR-0012 D1); the ablation pool is the final
evaluation's transfer subset (ADR-0007 E4)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from evaluation.benchmarks.alfworld.splits import ALFWORLD_TASK_TYPES
from inheritance.ablation import ALLOCATION_SIZE, AblationAllocation
from inheritance.generalization import (
    OUT_SCOPE_COUNT,
    OutOfScopeAllocation,
    resolve_frozen_applicability,
)
from inheritance.interaction import InteractionAllocation
from inheritance.replay import ReplayAllocation
from inheritance.runner import EvaluationTask

BANKED_POOLS = ("replay_bank", "generalization_bank", "benefit_bank")
POOL_SPLITS = {"t_mine": "train", "ablation": "train", "interaction": "train",
               "replay_bank": "train", "generalization_bank": "train",
               "benefit_bank": "train", "t_eval": "valid_unseen"}


class AllocationError(ValueError):
    """An allocation the protocol requires cannot be built."""


def evaluation_task(game: Mapping, pool: str) -> EvaluationTask:
    bank_id = f"{pool}[{game['family']}]" if pool in BANKED_POOLS else pool
    return EvaluationTask(task_id=game["game_id"], family=game["family"],
                          payload={"game_id": game["game_id"],
                                   "content_sha256": game["content_sha256"],
                                   "split": POOL_SPLITS[pool], "pool": pool},
                          bank_id=bank_id)


def pool_tasks(split_manifest: Mapping, pool: str) -> tuple[EvaluationTask, ...]:
    """A pool's games as evaluation tasks, in game_id order."""
    games = sorted(split_manifest["pools"][pool]["games"], key=lambda g: g["game_id"])
    return tuple(evaluation_task(g, pool) for g in games)


def replay_allocation(source_task: EvaluationTask) -> ReplayAllocation:
    """Stage 1: the source task only (ADR-0012 D3)."""
    return ReplayAllocation(source_task=source_task)


def interaction_allocation(split_manifest: Mapping) -> InteractionAllocation:
    return InteractionAllocation(tasks=pool_tasks(split_manifest, "interaction"))


def even_split(total: int, families: Sequence[str]) -> dict[str, int]:
    """``total`` over ``families`` in name order, the remainder one each to
    the first families: what a round-robin over them takes per family."""
    families = sorted(families)
    if not families:
        raise AllocationError("no family to split over")
    base, remainder = divmod(total, len(families))
    return {f: base + (1 if i < remainder else 0) for i, f in enumerate(families)
            if base + (1 if i < remainder else 0) > 0}


def interleave(bank: Mapping, families: Sequence[str], total: int) -> tuple[EvaluationTask, ...]:
    """``total`` bank games round-robin over ``families`` in name order: game
    j is the ⌊j/m⌋-th game, in bank order, of the (j mod m)-th family."""
    families = sorted(families)
    need = even_split(total, families)
    for family in families:
        held = len(bank["families"].get(family, ()))
        if held < need[family]:
            raise AllocationError(f"benefit_bank[{family}] holds {held} games, {need[family]} "
                                  f"are needed")
    m = len(families)
    return tuple(evaluation_task(bank["families"][families[j % m]][j // m], "benefit_bank")
                 for j in range(total))


def _applicability(envelope: dict, registry) -> tuple[str, ...]:
    applicability = resolve_frozen_applicability(envelope, registry)
    unknown = sorted(set(applicability) - set(ALFWORLD_TASK_TYPES))
    if unknown:
        raise AllocationError(f"applicability names families outside ALFWorld: {unknown}")
    return tuple(sorted(applicability))


def stage2_allocation(bank: Mapping, envelope: dict, registry) -> AblationAllocation:
    """ADR-0012 D4: 192 bank games over the frozen applicability."""
    families = _applicability(envelope, registry)
    return AblationAllocation(tasks=interleave(bank, families, ALLOCATION_SIZE),
                              families=families, bank_sha256=bank["bank_sha256"])


def out_of_scope_allocation(bank: Mapping, envelope: dict, registry) -> OutOfScopeAllocation:
    """ADR-0012 D5: 24 bank games over the omitted families, or none (vacuous)."""
    families = _applicability(envelope, registry)
    omitted = [f for f in sorted(ALFWORLD_TASK_TYPES) if f not in families]
    tasks = interleave(bank, omitted, OUT_SCOPE_COUNT) if omitted else ()
    return OutOfScopeAllocation(tasks=tasks, applicability=families,
                                family_universe=tuple(ALFWORLD_TASK_TYPES),
                                bank_sha256=bank["bank_sha256"])
