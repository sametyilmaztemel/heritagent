"""Stage 4A — T_reg regression evaluator + stage 4B — expression overhead
measurement (issue #12; merged ADR-0002).

Regression: T_reg is the frozen 40-task subset of T_mine that G0 already
solved. Every task carries immutable baseline provenance (original
successful G0 trajectory id, baseline_success == True). The baseline is NOT
rerun — the prereg design re-runs T_reg with ``genome ∪ trait`` and uses the
already-observed G0 success as baseline evidence. Locked threshold:
``regression_drop = 1 - success_with <= 0.03`` (tau_r; with 40 binary tasks
this means at least 39/40 remain successful).

Overhead: expression overhead is measured on the ACTUAL model-visible
projected single-skill block using the same runtime rendering path, counted
by an explicit deterministic TokenCounter whose identity is recorded.
Locked threshold: tau_k <= 512 tokens/trait. No model generation required.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Protocol

from runtime.expression import RuntimeConfig
from runtime.loop import Budgets
from runtime.model_adapters import GenerationSettings
from runtime.skills import skill_block

from inheritance.overlay import (
    EvaluationConfigs,
    build_evaluation_configs,
)
from inheritance.runner import (
    EvaluationTask,
    GateEpisodeRunner,
    RunnerContractError,
    validate_outcome,
)

REGRESSION_TASK_COUNT = 40
TAU_R = 0.03
REGRESSION_THRESHOLD_TEXT = "regression_drop = 1 - success_with <= 0.03 (tau_r)"
TAU_K_TOKENS = 512
OVERHEAD_THRESHOLD_TEXT = "expression_tokens <= 512 (tau_k)"


class TokenCounter(Protocol):
    """Provider-neutral deterministic token counter. Primary M0 evidence
    must record the counter identity; the generic chars/4 adapter estimate
    is NOT an exact tokenizer count and must not be presented as one."""

    @property
    def identity(self) -> str: ...

    def count_tokens(self, text: str) -> int: ...


@dataclass(frozen=True)
class TRegTask:
    """One frozen T_reg task with immutable baseline provenance: G0 already
    solved it, and ``baseline_trajectory_id`` points at that successful #8
    trajectory. The baseline is never rerun."""

    task_id: str
    family: str
    baseline_trajectory_id: str
    payload: dict = field(default_factory=dict)

    def __post_init__(self):
        import copy as _copy
        object.__setattr__(self, "payload", _copy.deepcopy(self.payload))
        if not self.baseline_trajectory_id or \
                not self.baseline_trajectory_id.startswith("T-"):
            raise ValueError(
                f"T_reg task {self.task_id!r} requires the original successful "
                f"G0 trajectory id (non-empty 'T-...' reference)")


@dataclass(frozen=True)
class RegressionAllocation:
    """Exactly 40 distinct frozen T_reg tasks, all with valid baseline
    provenance. Deep-frozen at construction."""

    tasks: tuple[TRegTask, ...]

    def __post_init__(self):
        import copy as _copy
        object.__setattr__(self, "tasks", _copy.deepcopy(self.tasks))
        if len(self.tasks) != REGRESSION_TASK_COUNT:
            raise ValueError(
                f"T_reg allocation requires exactly {REGRESSION_TASK_COUNT} "
                f"tasks, got {len(self.tasks)}")
        ids = [t.task_id for t in self.tasks]
        if len(set(ids)) != len(ids):
            raise ValueError("T_reg task ids must be distinct")
        for task in self.tasks:
            if not isinstance(task, TRegTask):
                raise ValueError("T_reg allocation accepts only TRegTask entries")

    def allocation_metadata(self) -> dict:
        return {
            "task_count": len(self.tasks),
            "task_ids": [t.task_id for t in self.tasks],
            "baseline_trajectory_ids": [t.baseline_trajectory_id
                                         for t in self.tasks],
            "baseline_success": True,
            "baseline_rerun": False,
        }


@dataclass(frozen=True)
class RegressionEvaluation:
    """Immutable stage-4A evidence."""

    _passed: bool
    _successes_with: int
    _regression_drop: float
    _episodes: tuple[dict, ...]
    _allocation: dict

    @property
    def passed(self) -> bool:
        return self._passed

    @property
    def successes_with(self) -> int:
        return self._successes_with

    @property
    def regression_drop(self) -> float:
        return self._regression_drop

    @property
    def episodes(self) -> tuple[dict, ...]:
        return copy.deepcopy(self._episodes)

    @property
    def allocation(self) -> dict:
        return copy.deepcopy(self._allocation)


class RegressionEvaluator:
    """Runs T_reg with the trait overlay (genome ∪ trait semantics); the
    baseline is the frozen G0 provenance, never rerun."""

    def __init__(self, *, candidate_envelope: dict, registry,
                 base_config: RuntimeConfig, runner: GateEpisodeRunner,
                 budgets: Budgets, target_regulation=None):
        self._configs: EvaluationConfigs = build_evaluation_configs(
            base_config, candidate_envelope, registry,
            target_regulation=target_regulation)
        self._candidate_gene_id = self._configs.target_gene_id
        self._runner = runner
        self._budgets = budgets
        self._settings = GenerationSettings(
            temperature=0.0, max_tokens=budgets.max_tokens_per_request)

    def evaluate(self, allocation: RegressionAllocation) -> RegressionEvaluation:
        episodes: list[dict] = []
        successes_with = 0
        for task in allocation.tasks:
            outcome = self._runner.run_episode(
                EvaluationTask(task_id=task.task_id, family=task.family,
                                payload=dict(task.payload)),
                self._configs.with_trait, self._budgets, "with_trait",
                self._settings)
            validate_outcome(outcome, task.task_id, "with_trait",
                              settings=self._settings)
            if outcome.success:
                successes_with += 1
            episodes.append({
                "task_id": task.task_id, "family": task.family,
                "success": outcome.success,
                "baseline_trajectory_id": task.baseline_trajectory_id,
                "trajectory_id": outcome.trajectory_id,
            })
        successes = successes_with
        drop = 1.0 - successes / len(allocation.tasks)
        passed = drop <= TAU_R
        return RegressionEvaluation(
            _passed=passed, _successes_with=successes,
            _regression_drop=drop, _episodes=tuple(copy.deepcopy(episodes)),
            _allocation=copy.deepcopy(allocation.allocation_metadata()))


@dataclass(frozen=True)
class OverheadEvaluation:
    """Immutable stage-4B evidence: rendered skill block hash, exact token
    count, counter identity, locked threshold, pass."""

    _passed: bool
    _tokens: int
    _rendered_sha256: str
    _counter_identity: str
    _rendered_block: str
    _threshold: int

    @property
    def passed(self) -> bool:
        return self._passed

    @property
    def tokens(self) -> int:
        return self._tokens

    @property
    def rendered_sha256(self) -> str:
        return self._rendered_sha256

    @property
    def counter_identity(self) -> str:
        return self._counter_identity

    @property
    def rendered_block(self) -> str:
        return self._rendered_block

    @property
    def threshold(self) -> int:
        return self._threshold


def measure_expression_overhead(runtime_skill, counter: TokenCounter) -> OverheadEvaluation:
    """Render ONLY the runtime-approved single-skill block via the same
    runtime rendering path used in prompts, count tokens with the supplied
    deterministic counter, and evaluate the locked tau_k threshold. No model
    generation; no evaluation-only metadata in the rendered text."""
    from trajectory.storage.canonical import canonical_json, sha256_hex
    rendered = skill_block([runtime_skill])
    rendered_sha256 = sha256_hex(rendered.encode("utf-8"))
    tokens = counter.count_tokens(rendered)
    # no evaluation-only metadata in the rendered block
    lowered = rendered.lower()
    for forbidden in ("applicability", "task_families", "estimated_generality",
                       "evidence_rationale", "cig_record", "provenance"):
        if forbidden in lowered:
            raise RunnerContractError(
                f"rendered expression block contains evaluation-only "
                f"metadata {forbidden!r}")
    passed = tokens <= TAU_K_TOKENS
    return OverheadEvaluation(
        _passed=passed, _tokens=tokens, _rendered_sha256=rendered_sha256,
        _counter_identity=counter.identity, _rendered_block=rendered,
        _threshold=TAU_K_TOKENS)
