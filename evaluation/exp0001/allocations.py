"""Frozen CIG allocations for EXP-0001 from the split manifest (issue #13
Phase A; manifest ``inheritance.stages``; ADR-0007 E6, proposed).

- Replay (design section 4 stage 1): the candidate's source task, the
  T_mine task of its source trajectory, and the 4 games of
  ``replay_bank[family of the source task]``.
- Ablation: the 20-game ablation pool, stratified as the manifest froze it.
- Generalization (E6, proposed): 10 in-scope games split as evenly as
  possible over the candidate's frozen applicability families, and 10
  out-of-scope games split the same way over the other families. A
  family's games are taken in the order of a keyed hash, so every
  candidate that needs n games of a family gets the same n (cross-trait
  reuse is allowed and logged; within a trait there is no replacement).
  A candidate whose applicability names every family has no out-of-scope
  family: its out-of-scope stage is vacuous (E6, owner decision), and its
  10 in-scope games cover every family.
- Interaction: the 10-game interaction pool, shared by every pair.
- Regression: the one shared T_reg (ADR-0006 D8; ADR-0007 E3)."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence

from evaluation.benchmarks.alfworld.splits import ALFWORLD_TASK_TYPES
from inheritance.ablation import AblationAllocation
from inheritance.generalization import (
    IN_SCOPE_COUNT,
    OUT_SCOPE_COUNT,
    GeneralizationAllocation,
    resolve_frozen_applicability,
)
from inheritance.interaction import InteractionAllocation
from inheritance.replay import ReplayAllocation
from inheritance.runner import EvaluationTask

GENERALIZATION_DRAW_SEED = 0
GENERALIZATION_RULE = {
    "split": "10 per scope, as even as possible over the scope's families; the remainder "
             "goes one game each to the first families in name order",
    "draw": "per family, the games with the smallest "
            "sha256('heritagent-gen|<seed>|<family>|<game_id>'), seed 0",
    "no_out_of_scope_family": "a vacuous out-of-scope stage; the in-scope games cover every "
                              "family",
}
BANKED_POOLS = ("replay_bank", "generalization_bank")
POOL_SPLITS = {"t_mine": "train", "ablation": "train", "interaction": "train",
               "replay_bank": "train", "generalization_bank": "train", "t_eval": "valid_unseen"}


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


def _family_games(split_manifest: Mapping, pool: str, family: str) -> list[Mapping]:
    return sorted((g for g in split_manifest["pools"][pool]["games"] if g["family"] == family),
                  key=lambda g: g["game_id"])


def replay_allocation(split_manifest: Mapping, source_task: EvaluationTask) -> ReplayAllocation:
    family = source_task.family
    games = _family_games(split_manifest, "replay_bank", family)
    if len(games) != 4:
        raise AllocationError(f"replay_bank[{family}] holds {len(games)} games, the replay stage "
                              f"needs 4")
    return ReplayAllocation(source_task=source_task,
                            bank_variants=tuple(evaluation_task(g, "replay_bank") for g in games),
                            source_family=family, replay_bank_id=f"replay_bank[{family}]")


def ablation_allocation(split_manifest: Mapping) -> AblationAllocation:
    pool = split_manifest["pools"]["ablation"]
    return AblationAllocation(tasks=pool_tasks(split_manifest, "ablation"),
                              frozen_strata=dict(pool["family_counts"]))


def interaction_allocation(split_manifest: Mapping) -> InteractionAllocation:
    return InteractionAllocation(tasks=pool_tasks(split_manifest, "interaction"))


def even_split(total: int, families: Sequence[str]) -> dict[str, int]:
    families = sorted(families)
    if not families:
        raise AllocationError("no family to split over")
    base, remainder = divmod(total, len(families))
    return {f: base + (1 if i < remainder else 0) for i, f in enumerate(families)
            if base + (1 if i < remainder else 0) > 0}


def generalization_rank(family: str, game_id: str, seed: int = GENERALIZATION_DRAW_SEED) -> str:
    return hashlib.sha256(f"heritagent-gen|{seed}|{family}|{game_id}".encode("utf-8")).hexdigest()


def _draw(split_manifest: Mapping, plan: Mapping[str, int]) -> tuple[EvaluationTask, ...]:
    tasks = []
    for family in sorted(plan):
        games = sorted(_family_games(split_manifest, "generalization_bank", family),
                       key=lambda g: (generalization_rank(family, g["game_id"]), g["game_id"]))
        if len(games) < plan[family]:
            raise AllocationError(f"generalization_bank[{family}] holds {len(games)} games, "
                                  f"{plan[family]} are needed")
        tasks.extend(evaluation_task(g, "generalization_bank") for g in games[:plan[family]])
    return tuple(tasks)


def generalization_allocation(split_manifest: Mapping, envelope: dict,
                              registry) -> GeneralizationAllocation:
    applicability = resolve_frozen_applicability(envelope, registry)
    unknown = sorted(set(applicability) - set(ALFWORLD_TASK_TYPES))
    if unknown:
        raise AllocationError(f"applicability names families outside ALFWorld: {unknown}")
    outside = [f for f in ALFWORLD_TASK_TYPES if f not in applicability]
    in_plan = even_split(IN_SCOPE_COUNT, applicability)
    out_plan = even_split(OUT_SCOPE_COUNT, outside) if outside else {}  # empty: vacuous
    return GeneralizationAllocation(
        in_scope_tasks=_draw(split_manifest, in_plan), out_scope_tasks=_draw(split_manifest, out_plan),
        frozen_in_scope_plan=in_plan, frozen_out_scope_plan=out_plan,
        declared_frozen_applicability=tuple(applicability),
        family_universe=ALFWORLD_TASK_TYPES)
