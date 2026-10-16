"""The EXP-0001 ``GateEpisodeRunner`` (issue #13 Phase A; ADR-0008).

Every gate episode is a ledger-admitted episode of the EpisodeExecutor, so
hard caps, resume and trajectory recording hold in the gate as in the
lifetime. Each call gets one ledger key from its scope, task, condition and
the configuration's skill set; a repeated identical call in the same scope
gets the next occurrence index, so a resumed run, which repeats the same
calls in the same order, maps every call to its first execution.

``PrefetchGateRunner`` serves a dry pass of the same gate code: a recorded
episode returns its recorded outcome, an unrecorded one is collected and
answered with a placeholder. The pipeline uses it only over stretches of
the gate whose episodes do not depend on each other's outcomes, then runs
the collected episodes in parallel (ADR-0008)."""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Callable

from evaluation.exp0001.episodes import EpisodeExecutor, EpisodeResult, EpisodeSpec
from inheritance.runner import EpisodeOutcome, EvaluationTask, RunnerContractError
from runtime.expression import RuntimeConfig
from runtime.loop import Budgets
from runtime.model_adapters.base import GenerationSettings

PLACEHOLDER_TRAJECTORY = "T-prefetch-placeholder"


def config_fingerprint(config: RuntimeConfig) -> str:
    skills = ",".join(s.gene_id for s in config.skills)
    return hashlib.sha256(f"{config.genome_id}|{skills}".encode("utf-8")).hexdigest()[:8]


class PrefetchStop(Exception):
    """A dry pass reached an episode it must not collect."""


def _outcome(result: EpisodeResult, condition: str,
             settings: GenerationSettings) -> EpisodeOutcome:
    return EpisodeOutcome(
        task_id=result.task_id, condition=condition, success=result.success,
        expressed_gene_ids=tuple(result.expressed_gene_ids), status=result.status,
        trajectory_id=result.trajectory_id, prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens, steps=result.steps,
        effective_settings=settings)


class LedgerGateRunner:
    def __init__(self, executor: EpisodeExecutor, scope: str, budgets: Budgets, *,
                 requested: list | None = None):
        """``requested``, when given, collects the key of every episode asked for."""
        self._executor = executor
        self._scope = scope
        self._budgets = budgets
        self._seen: Counter = Counter()
        self._requested = requested

    def _spec(self, task: EvaluationTask, config: RuntimeConfig, budgets: Budgets,
              condition: str, settings: GenerationSettings) -> EpisodeSpec:
        if budgets != self._budgets:
            raise RunnerContractError(f"gate budgets {budgets} differ from the protocol "
                                      f"budgets {self._budgets} (ADR-0007 E5)")
        if settings is None or settings.temperature != 0.0:
            raise RunnerContractError("gate episodes run at the locked temperature 0")
        split = task.payload.get("split")
        if split != "train" or task.payload.get("pool") == "t_eval":
            # contamination rule: T_eval is never visible to the gate
            raise RunnerContractError(f"gate task {task.task_id!r} must be a train game, got "
                                      f"split {split!r}; T_eval never enters the gate")
        identity = (task.task_id, condition, config_fingerprint(config))
        occurrence = self._seen[identity]
        self._seen[identity] += 1
        key = f"{self._scope}/{task.task_id}/{condition}/{identity[2]}-{occurrence}"
        return EpisodeSpec(
            key=key, task_id=task.task_id, family=task.family, split="train", config=config,
            settings=settings,
            tags={"stage": self._scope, "condition": condition, "bank_id": task.bank_id,
                  "skills": [s.gene_id for s in config.skills]})

    def run_episode(self, task: EvaluationTask, config: RuntimeConfig, budgets: Budgets,
                    condition: str, settings: GenerationSettings) -> EpisodeOutcome:
        spec = self._spec(task, config, budgets, condition, settings)
        if self._requested is not None:
            self._requested.append(spec.key)
        return _outcome(self._executor.run(spec), condition, settings)


class PrefetchGateRunner(LedgerGateRunner):
    """A dry-pass runner: it never runs an episode. ``collect(condition)``
    says which unrecorded episodes this pass may collect; reaching any
    other unrecorded episode ends the pass with PrefetchStop."""

    def __init__(self, executor: EpisodeExecutor, scope: str, budgets: Budgets, *,
                 collected: list, collect: Callable[[str], bool]):
        super().__init__(executor, scope, budgets)
        self._collected = collected
        self._collect = collect

    def run_episode(self, task: EvaluationTask, config: RuntimeConfig, budgets: Budgets,
                    condition: str, settings: GenerationSettings) -> EpisodeOutcome:
        spec = self._spec(task, config, budgets, condition, settings)
        recorded = self._executor.recorded(spec)
        if recorded is not None:
            return _outcome(recorded, condition, settings)
        if not self._collect(condition):
            raise PrefetchStop(spec.key)
        self._collected.append(spec)
        # a placeholder no collected episode depends on (see the pipeline)
        return EpisodeOutcome(task_id=task.task_id, condition=condition, success=False,
                              expressed_gene_ids=(), status="prefetch-placeholder",
                              trajectory_id=PLACEHOLDER_TRAJECTORY, prompt_tokens=0,
                              completion_tokens=0, steps=0, effective_settings=settings)
