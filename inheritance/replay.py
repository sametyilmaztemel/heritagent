"""Replay evaluator (CIG stage 1; issue #11 criteria 5-6, 10-11, 13).

Replay evaluation set = exactly 5 distinct task instances: the source task
plus 4 valid unseen same-family variants from the frozen replay bank. For
each instance: ONE with-trait execution + ONE without-trait execution at
eval temperature 0 (paired intervention; no pseudo-replication).

Locked pass rule (all must hold):
1. successes_with >= 3 of 5;
2. delta_count >= 2 (Δsuccess >= +0.4);
3. expression check — target gene appears in skills_expressed in EVERY
   successful with-trait episode, and in NO without-trait episode; failed
   with-trait episodes may legitimately not express (regulation semantics).

Emits a schema-valid ``cig/0.1`` child record
(``<parent>/S<mining_seed>-replay``). No aggregate/verdict, no somatic
terminal decision, no germline mutation — #12 owns those.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

from genome.validation.loader import load_cig
from runtime.expression import RuntimeConfig
from runtime.loop import Budgets

from inheritance.overlay import (
    EvaluationConfigs,
    evaluation_settings,
    build_evaluation_configs,
)
from inheritance.runner import (
    EvaluationIntegrityError,
    GateEpisodeRunner,
    RunnerContractError,
    validate_gate_setup,
    validate_outcome,
)

REPLAY_INSTANCE_COUNT = 5          # source + 4 bank variants
REPLAY_MIN_SUCCESSES_WITH = 3      # >= 3/5
REPLAY_MIN_DELTA_COUNT = 2         # Δsuccess >= +2/5
REPLAY_THRESHOLD_TEXT = ("successes_with >= 3/5 and delta_count >= +2/5 "
                          "and expression check")


@dataclass(frozen=True)
class ReplayAllocation:
    """The frozen replay allocation validated fail-closed before execution:
    exactly source + 4 unseen same-family bank variants, all distinct task
    ids, no replacement within the trait. Every variant must carry the
    declared frozen replay-bank provenance. Tasks and metadata are
    deep-frozen at construction: caller mutation cannot change the
    executed task definitions."""

    source_task: EvaluationTask
    bank_variants: tuple[EvaluationTask, ...]
    source_family: str
    replay_bank_id: str   # frozen canonical bank id, e.g. replay_bank[heat_and_place]

    def __post_init__(self):
        import copy as _copy
        object.__setattr__(self, "source_task",
                            _copy.deepcopy(self.source_task))
        object.__setattr__(self, "bank_variants",
                            _copy.deepcopy(self.bank_variants))
        if not isinstance(self.replay_bank_id, str) or not self.replay_bank_id.strip():
            raise ValueError(
                "replay_bank_id must be a non-empty frozen bank identifier")
        if len(self.bank_variants) != REPLAY_INSTANCE_COUNT - 1:
            raise ValueError(
                f"replay allocation requires exactly {REPLAY_INSTANCE_COUNT - 1} "
                f"bank variants, got {len(self.bank_variants)}")
        ids = [self.source_task.task_id] + [t.task_id for t in self.bank_variants]
        if len(set(ids)) != len(ids):
            raise ValueError(f"replay task ids must be distinct, got {ids}")
        for variant in self.bank_variants:
            if variant.family != self.source_family:
                raise ValueError(
                    f"bank variant {variant.task_id!r} family "
                    f"{variant.family!r} != source family {self.source_family!r}")
            if not variant.bank_id or variant.bank_id != self.replay_bank_id:
                raise ValueError(
                    f"bank variant {variant.task_id!r} bank_id "
                    f"{variant.bank_id!r} does not match the declared frozen "
                    f"replay bank {self.replay_bank_id!r}")
        if self.source_task.family != self.source_family:
            raise ValueError(
                f"source task family {self.source_task.family!r} != "
                f"source_family {self.source_family!r}")

    @property
    def tasks(self) -> tuple[EvaluationTask, ...]:
        return (self.source_task, *self.bank_variants)

    def allocation_metadata(self) -> dict:
        return {
            "source_task_id": self.source_task.task_id,
            "source_family": self.source_family,
            "replay_bank_id": self.replay_bank_id,
            "bank_variant_ids": [t.task_id for t in self.bank_variants],
            "bank_ids": [t.bank_id for t in self.bank_variants],
            "instance_count": len(self.tasks),
        }


@dataclass(frozen=True)
class ReplayEvaluation:
    """Immutable stage-1 evidence (deep-copy accessors) plus the emitted
    schema-valid child CIG record."""

    _passed: bool
    _successes_with: int
    _successes_without: int
    _delta_count: int
    _delta_success: float
    _expression_ok: bool
    _episodes: tuple[dict, ...]
    _child_record: dict
    _allocation: dict

    @property
    def passed(self) -> bool:
        return self._passed

    @property
    def successes_with(self) -> int:
        return self._successes_with

    @property
    def successes_without(self) -> int:
        return self._successes_without

    @property
    def delta_count(self) -> int:
        return self._delta_count

    @property
    def delta_success(self) -> float:
        return self._delta_success

    @property
    def expression_ok(self) -> bool:
        return self._expression_ok

    @property
    def episodes(self) -> tuple[dict, ...]:
        return copy.deepcopy(self._episodes)

    @property
    def child_record(self) -> dict:
        return copy.deepcopy(self._child_record)

    @property
    def allocation(self) -> dict:
        return copy.deepcopy(self._allocation)


class ReplayEvaluator:
    """Paired replay evaluation of one somatic candidate over 5 instances."""

    def __init__(self, *, candidate_envelope: dict, registry,
                 base_config: RuntimeConfig, runner: GateEpisodeRunner,
                 budgets: Budgets, parent_cig_id: str, mining_seed: int,
                 target_regulation=None):
        # central preflight BEFORE any episode (zero runner calls on invalid
        # setup): parent grammar, seed representation, child coherence
        validate_gate_setup(parent_cig_id, mining_seed)
        self._registry = registry
        self._base_config = base_config
        self._runner = runner
        self._budgets = budgets
        self._parent_cig_id = parent_cig_id
        self._mining_seed = mining_seed
        self._target_regulation = target_regulation
        self._configs: EvaluationConfigs | None = None
        self._candidate_gene_id: str | None = None
        # locked evaluation settings derived from the fixed budget:
        # temperature 0, max_tokens = budgets.max_tokens_per_request, seed =
        # the mining seed, no stop — ONE deterministic run per (task
        # instance, condition); both conditions share this exact object
        self._locked_settings = evaluation_settings(
            max_tokens=budgets.max_tokens_per_request, seed=mining_seed)
        # fail-closed setup (criterion 13): state/origin/binding/duplicate
        self._configs = build_evaluation_configs(
            base_config, candidate_envelope, registry,
            target_regulation=target_regulation)
        self._candidate_gene_id = self._configs.target_gene_id

    def evaluate(self, allocation: ReplayAllocation) -> ReplayEvaluation:
        """Run the paired 5-instance replay; returns immutable stage
        evidence + the emitted child CIG record."""
        gene = self._candidate_gene_id
        episodes: list[dict] = []
        successes_with = 0
        successes_without = 0
        expression_violations: list[str] = []
        seen_pairs: set[tuple[str, str]] = set()

        for task in allocation.tasks:
            for condition, config in (("with_trait", self._configs.with_trait),
                                       ("without_trait", self._configs.without_trait)):
                pair_key = (task.task_id, condition)
                if pair_key in seen_pairs:
                    raise RunnerContractError(
                        f"duplicate execution for {pair_key}; exactly one run "
                        f"per (task instance, condition) is allowed")
                seen_pairs.add(pair_key)
                outcome = self._runner.run_episode(task, config, self._budgets, condition, self._locked_settings)
                validate_outcome(outcome, task.task_id, condition,
                                  settings=self._locked_settings)
                episodes.append({
                    "task_id": task.task_id,
                    "family": task.family,
                    "condition": condition,
                    "success": outcome.success,
                    "expressed_gene_ids": list(outcome.expressed_gene_ids),
                    "status": outcome.status,
                    "trajectory_id": outcome.trajectory_id,
                    "prompt_tokens": outcome.prompt_tokens,
                    "completion_tokens": outcome.completion_tokens,
                    "steps": outcome.steps,
                })
                if condition == "with_trait":
                    if outcome.success:
                        successes_with += 1
                        if gene not in outcome.expressed_gene_ids:
                            expression_violations.append(
                                f"successful with-trait episode on "
                                f"{task.task_id!r} never expressed {gene!r}")
                else:
                    if outcome.success:
                        successes_without += 1
                    if gene in outcome.expressed_gene_ids:
                        # contaminated intervention: INVALID evidence, not a
                        # candidate failure — abort with no completed child
                        raise EvaluationIntegrityError(
                            f"without-trait episode on {task.task_id!r} "
                            f"expressed the target gene {gene!r}: the paired "
                            f"intervention is contaminated and produced no "
                            f"child CIG record")

        delta_count = successes_with - successes_without
        expression_ok = not expression_violations
        passed = (successes_with >= REPLAY_MIN_SUCCESSES_WITH
                   and delta_count >= REPLAY_MIN_DELTA_COUNT
                   and expression_ok)
        if expression_violations:
            episodes.append({"expression_violations": expression_violations})

        child = self._build_child_record(
            allocation=allocation, episodes=episodes,
            successes_with=successes_with, successes_without=successes_without,
            delta_count=delta_count,
            delta_success=delta_count / len(allocation.tasks),
            expression_ok=expression_ok, passed=passed)
        return ReplayEvaluation(
            _passed=passed, _successes_with=successes_with,
            _successes_without=successes_without, _delta_count=delta_count,
            _delta_success=delta_count / len(allocation.tasks),
            _expression_ok=expression_ok,
            _episodes=tuple(copy.deepcopy(episodes)),
            _child_record=copy.deepcopy(child),
            _allocation=copy.deepcopy(allocation.allocation_metadata()))

    def _build_child_record(self, *, allocation: ReplayAllocation, episodes: list[dict],
                             successes_with: int, successes_without: int,
                             delta_count: int, delta_success: float,
                             expression_ok: bool, passed: bool) -> dict:
        child_id = f"{self._parent_cig_id}/S{self._mining_seed}-replay"
        child = {
            "cig_id": child_id,
            "parent_cig_id": self._parent_cig_id,
            "seed": self._mining_seed,
            "stage": "replay",
            "measurements": {
                "candidate_gene_id": self._candidate_gene_id,
                "allocation": allocation.allocation_metadata(),
                "episodes": episodes,
                "successes_with": successes_with,
                "successes_without": successes_without,
                "delta_count": delta_count,
                "delta_success": delta_success,
                "thresholds": {
                    "min_successes_with": REPLAY_MIN_SUCCESSES_WITH,
                    "min_delta_count": REPLAY_MIN_DELTA_COUNT,
                    "rule": REPLAY_THRESHOLD_TEXT,
                },
                "expression_check": {"passed": expression_ok},
                "deterministic_execution_policy": {
                    "runs_per_task_condition": 1,
                    "eval_temperature": 0.0,
                    "paired_conditions": ["with_trait", "without_trait"],
                },
                "stage_pass": passed,
            },
        }
        load_cig(child)  # schema-valid or fail closed
        return child
