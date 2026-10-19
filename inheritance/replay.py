"""Replay evaluator (CIG stage 1; ADR-0012 D3, owner-accepted 2026-10-19).

Stage 1 is an expression check on the source task: the candidate plays its
source task (the T_mine game of its mining pair's success, ADR-0007 E6)
once with the trait, at the gate's settings, in round R1 (ADR-0012 D2). It
passes iff the target gene is expressed in that episode or the episode
fails: the locked expression check applied to one episode. There is no
outcome threshold and no without-trait execution.

What it tests: that a successful replay of the trait expresses its gene.
It does **not** test that the trait reproduces its outcome where it was
found: a failed episode passes, and nothing is compared (ADR-0012
"Research implication"). Outcome evidence is stage 2's, on unseen in-scope
games.

Emits a schema-valid ``cig/0.1`` child record
(``<parent>/S<mining_seed>-replay``). No aggregate/verdict, no somatic
terminal decision, no germline mutation — the gate owns those.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

from genome.validation.loader import load_cig
from runtime.expression import RuntimeConfig
from runtime.loop import Budgets

from inheritance.overlay import (
    EvaluationConfigs,
    evaluation_settings,
    build_evaluation_configs,
)
from inheritance.runner import (
    EvaluationTask,
    GateEpisodeRunner,
    RunnerContractError,
    validate_gate_setup,
    validate_outcome,
)

REPLAY_ROUND = 1                   # stage 1 plays in round R1 (ADR-0012 D2)
REPLAY_THRESHOLD_TEXT = ("source task once with the trait: target gene expressed or the "
                          "episode failed (ADR-0012 D3)")


@dataclass(frozen=True)
class ReplayAllocation:
    """Stage 1's allocation: the candidate's source task only (ADR-0012 D3),
    the T_mine game of its mining pair's success. Deep-frozen at
    construction: caller mutation cannot change the executed task."""

    source_task: EvaluationTask

    def __post_init__(self):
        object.__setattr__(self, "source_task", copy.deepcopy(self.source_task))
        if not isinstance(self.source_task, EvaluationTask):
            raise ValueError("stage 1 needs the source task as an EvaluationTask")

    @property
    def tasks(self) -> tuple[EvaluationTask, ...]:
        return (self.source_task,)

    def allocation_metadata(self) -> dict:
        return {"source_task_id": self.source_task.task_id,
                "source_family": self.source_task.family,
                "bank_id": self.source_task.bank_id, "instance_count": 1}


@dataclass(frozen=True)
class ReplayEvaluation:
    """Immutable stage-1 evidence (deep-copy accessors) plus the emitted
    schema-valid child CIG record."""

    _passed: bool
    _success: bool
    _expressed: bool
    _episode: dict
    _child_record: dict
    _allocation: dict

    @property
    def passed(self) -> bool:
        return self._passed

    @property
    def success(self) -> bool:
        """Whether the with-trait episode solved the source task."""
        return self._success

    @property
    def expressed(self) -> bool:
        """Whether the target gene was expressed in that episode."""
        return self._expressed

    @property
    def episode(self) -> dict:
        return copy.deepcopy(self._episode)

    @property
    def child_record(self) -> dict:
        return copy.deepcopy(self._child_record)

    @property
    def allocation(self) -> dict:
        return copy.deepcopy(self._allocation)


class ReplayEvaluator:
    """Stage 1 of one somatic candidate: one with-trait episode on its
    source task (ADR-0012 D3)."""

    def __init__(self, *, candidate_envelope: dict, registry,
                 base_config: RuntimeConfig, runner: GateEpisodeRunner,
                 budgets: Budgets, parent_cig_id: str, mining_seed: int,
                 target_regulation=None):
        # central preflight BEFORE any episode (zero runner calls on invalid
        # setup): parent grammar, seed representation, child coherence
        validate_gate_setup(parent_cig_id, mining_seed)
        self._runner = runner
        self._budgets = budgets
        self._parent_cig_id = parent_cig_id
        self._mining_seed = mining_seed
        # locked evaluation settings derived from the fixed budget:
        # temperature 0, max_tokens = budgets.max_tokens_per_request, seed =
        # the mining seed, the ReAct protocol stop (ADR-0003 A2)
        self._locked_settings = evaluation_settings(
            max_tokens=budgets.max_tokens_per_request, seed=mining_seed)
        # fail-closed setup (criterion 13): state/origin/binding/duplicate
        self._configs: EvaluationConfigs = build_evaluation_configs(
            base_config, candidate_envelope, registry,
            target_regulation=target_regulation)
        self._candidate_gene_id = self._configs.target_gene_id

    def evaluate(self, allocation: ReplayAllocation) -> ReplayEvaluation:
        """Play the source task once with the trait and apply the expression
        check; returns immutable stage evidence + the emitted child record."""
        if not isinstance(allocation, ReplayAllocation):
            raise RunnerContractError("stage 1 needs a ReplayAllocation (the source task)")
        task = allocation.source_task
        outcome = self._runner.run_episode(task, self._configs.with_trait, self._budgets,
                                           "with_trait", self._locked_settings)
        validate_outcome(outcome, task.task_id, "with_trait", settings=self._locked_settings)
        expressed = self._candidate_gene_id in outcome.expressed_gene_ids
        passed = expressed or not outcome.success
        episode = {
            "task_id": task.task_id, "family": task.family, "condition": "with_trait",
            "round": REPLAY_ROUND, "success": outcome.success,
            "expressed_gene_ids": list(outcome.expressed_gene_ids), "status": outcome.status,
            "trajectory_id": outcome.trajectory_id, "prompt_tokens": outcome.prompt_tokens,
            "completion_tokens": outcome.completion_tokens, "steps": outcome.steps,
        }
        child = self._build_child_record(allocation=allocation, episode=episode,
                                         expressed=expressed, passed=passed)
        return ReplayEvaluation(
            _passed=passed, _success=outcome.success, _expressed=expressed,
            _episode=copy.deepcopy(episode), _child_record=copy.deepcopy(child),
            _allocation=allocation.allocation_metadata())

    def _build_child_record(self, *, allocation: ReplayAllocation, episode: dict,
                            expressed: bool, passed: bool) -> dict:
        child = {
            "cig_id": f"{self._parent_cig_id}/S{self._mining_seed}-replay",
            "parent_cig_id": self._parent_cig_id,
            "seed": self._mining_seed,
            "stage": "replay",
            "measurements": {
                "candidate_gene_id": self._candidate_gene_id,
                **self._configs.binding(),  # ADR-0010
                "allocation": allocation.allocation_metadata(),
                "episodes": [episode],
                "expression_check": {"expressed": expressed, "success": episode["success"],
                                     "passed": passed, "rule": REPLAY_THRESHOLD_TEXT},
                "thresholds": {"rule": REPLAY_THRESHOLD_TEXT},
                "deterministic_execution_policy": {
                    "runs_per_task_condition": 1,
                    "eval_temperature": 0.0,
                    "paired_conditions": ["with_trait"],
                },
                "eval_generation_settings": {
                    "temperature": self._locked_settings.temperature,
                    "max_tokens": self._locked_settings.max_tokens,
                    "seed": self._locked_settings.seed,
                    "stop": list(self._locked_settings.stop),
                },
                "stage_pass": passed,
            },
        }
        load_cig(child)  # schema-valid or fail closed
        return child
