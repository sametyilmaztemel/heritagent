"""Stage 4A — T_reg regression evaluator + stage 4B — expression overhead
measurement (issue #12; merged ADR-0002).

Regression: T_reg is the frozen 40-task subset of T_mine that G0 solved in
a lifetime episode. Every task carries immutable provenance (the original
successful G0 trajectory id, baseline_success == True). Since ADR-0009 D1
(owner, 2026-10-15) the stage is paired: G0 without the trait plays every
T_reg task once at the gate's locked settings, shared by every candidate of
the gate run (``run_regression_baseline``), and each candidate runs T_reg
with ``genome ∪ trait`` at the same settings. Locked threshold:
``regression_drop = (success_without - success_with) / 40 <= 0.03`` (tau_r;
with 40 binary tasks a net loss of at most one task; a net gain passes).
The lifetime success at temperature 0.7 selects T_reg (ADR-0007 E3); it is
not the comparison the stage decides on.

Overhead: expression overhead is measured on the ACTUAL model-visible
projected single-skill block using the same runtime rendering path, counted
by an explicit deterministic TokenCounter whose identity is recorded.
Locked threshold: tau_k <= 512 tokens/trait. No model generation required.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Protocol

from runtime.expression import RuntimeConfig
from runtime.loop import Budgets
from runtime.skills import skill_block

from inheritance.overlay import (
    EvaluationConfigs,
    build_evaluation_configs,
    evaluation_settings,
)
from inheritance.runner import (
    EvaluationTask,
    GateEpisodeRunner,
    RunnerContractError,
    validate_outcome,
)

REGRESSION_TASK_COUNT = 40
TAU_R = 0.03
REGRESSION_THRESHOLD_TEXT = ("regression_drop = (success_without - success_with) / 40 <= 0.03 "
                             "(tau_r), paired against one shared G0 run (ADR-0009 D1)")
TAU_K_TOKENS = 512
OVERHEAD_THRESHOLD_TEXT = "expression_tokens <= 512 (tau_k)"


def drop_within(successes: int, total: int, tau: float) -> bool:
    """Exact ``1 - successes/total <= tau`` in rational arithmetic.

    A drop of exactly tau must pass: with binary floats 1 - 38/40 is
    0.050000000000000044, which would make the tau_r = 0.05 sensitivity cell
    reject 2/40 and collapse it onto the tau_r = 0.03 cell."""
    return Fraction(total - successes, total) <= Fraction(repr(tau))


def paired_drop_within(successes_with: int, successes_without: int, total: int,
                       tau: float) -> bool:
    """Exact ``(successes_without - successes_with) / total <= tau`` (ADR-0009
    D1): the trait's net loss against G0 at the same settings."""
    return Fraction(successes_without - successes_with, total) <= Fraction(repr(tau))


class TokenCounter(Protocol):
    """Provider-neutral deterministic token counter. Primary M0 evidence
    must record the counter identity; the generic chars/4 adapter estimate
    is NOT an exact tokenizer count and must not be presented as one."""

    @property
    def identity(self) -> str: ...

    def count_tokens(self, text: str) -> int: ...


def validate_token_counter(counter) -> None:
    """Fail closed on invalid counter contracts: empty/non-string identity
    or a non-callable/non-negative count_tokens."""
    identity = getattr(counter, "identity", None)
    if not isinstance(identity, str) or not identity.strip():
        raise RunnerContractError(
            f"TokenCounter.identity must be a non-empty stable string, got "
            f"{identity!r}")
    count = getattr(counter, "count_tokens", None)
    if not callable(count):
        raise RunnerContractError("TokenCounter.count_tokens must be callable")
    probe = count("")
    if not isinstance(probe, int) or isinstance(probe, bool) or probe < 0:
        raise RunnerContractError(
            f"TokenCounter.count_tokens must return a non-negative integer, "
            f"got {probe!r}")


def deep_freeze_json(obj):
    """Recursive freeze for JSON-like payloads (TRegTask hardening)."""
    import types
    if isinstance(obj, dict):
        import types as _t
        return _t.MappingProxyType({k: deep_freeze_json(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return tuple(deep_freeze_json(v) for v in obj)
    return obj


@dataclass(frozen=True)
class TRegTask:
    """One frozen T_reg task with immutable baseline provenance: G0 already
    solved it, and ``baseline_trajectory_id`` points at that successful #8
    trajectory. The payload is deep-frozen at construction and
    ``baseline_success`` is explicitly True per task."""

    task_id: str
    family: str
    baseline_trajectory_id: str
    baseline_success: bool = True
    payload: dict = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "payload", deep_freeze_json(self.payload))
        if self.baseline_success is not True:
            raise ValueError(
                f"T_reg task {self.task_id!r} must carry baseline_success=True "
                f"(the original G0 successful trajectory), got "
                f"{self.baseline_success!r}")
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
        # TRegTask objects are already deeply immutable; reuse them as-is so
        # the frozen (mappingproxy) payload trees are never pickled/copied
        object.__setattr__(self, "tasks", tuple(self.tasks))
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
            # the selection's lifetime trajectories are never re-run; the
            # stage compares against one shared G0 run (ADR-0009 D1)
            "baseline_rerun": False,
            "paired_g0_run": True,
        }


@dataclass(frozen=True)
class RegressionBaseline:
    """G0 without the trait on every T_reg task at the gate's locked
    settings, once per mining seed in a gate run and shared by every
    candidate of that seed (ADR-0009 D1)."""

    episodes: tuple[dict, ...]
    locked_settings: dict

    @property
    def task_ids(self) -> tuple[str, ...]:
        return tuple(e["task_id"] for e in self.episodes)

    @property
    def successes(self) -> int:
        return sum(1 for e in self.episodes if e["success"])

    def record(self) -> dict:
        return {"condition": "without_trait", "shared_across_candidates": True,
                "episodes": copy.deepcopy(list(self.episodes)),
                "successes_without": self.successes,
                "eval_generation_settings": copy.deepcopy(self.locked_settings)}


def run_regression_baseline(*, base_config: RuntimeConfig, runner: GateEpisodeRunner,
                            budgets: Budgets, mining_seed: int,
                            allocation: RegressionAllocation) -> RegressionBaseline:
    settings = evaluation_settings(max_tokens=budgets.max_tokens_per_request, seed=mining_seed)
    episodes = []
    for task in allocation.tasks:
        outcome = runner.run_episode(
            EvaluationTask(task_id=task.task_id, family=task.family, payload=dict(task.payload)),
            base_config, budgets, "without_trait", settings)
        validate_outcome(outcome, task.task_id, "without_trait", settings=settings)
        episodes.append({"task_id": task.task_id, "family": task.family,
                         "success": outcome.success, "trajectory_id": outcome.trajectory_id})
    return RegressionBaseline(
        episodes=tuple(episodes),
        locked_settings={"temperature": settings.temperature, "max_tokens": settings.max_tokens,
                         "seed": settings.seed, "stop": list(settings.stop)})


@dataclass(frozen=True)
class RegressionEvaluation:
    """Immutable stage-4A evidence."""

    _passed: bool
    _successes_with: int
    _task_count: int
    _regression_drop: float
    _episodes: tuple[dict, ...]
    _allocation: dict
    _locked_settings: dict
    _successes_without: int = 0
    _baseline: dict = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self._passed

    @property
    def locked_settings(self) -> dict:
        """The exact generation settings every T_reg episode ran with
        (persisted verbatim in the stage-4 child)."""
        return copy.deepcopy(self._locked_settings)

    @property
    def successes_with(self) -> int:
        return self._successes_with

    @property
    def successes_without(self) -> int:
        """G0's successes on T_reg in the shared paired run."""
        return self._successes_without

    @property
    def baseline(self) -> dict:
        """The shared G0 run's record (ADR-0009 D1)."""
        return copy.deepcopy(self._baseline)

    @property
    def task_count(self) -> int:
        """Number of T_reg tasks run (the denominator of the drop)."""
        return self._task_count

    @property
    def regression_drop(self) -> float:
        """Nearest float to the exact paired drop; decisions use paired_drop_within()."""
        return self._regression_drop

    @property
    def episodes(self) -> tuple[dict, ...]:
        return copy.deepcopy(self._episodes)

    @property
    def allocation(self) -> dict:
        return copy.deepcopy(self._allocation)


class RegressionEvaluator:
    """Runs T_reg with the trait overlay (genome ∪ trait semantics) and
    compares it with the shared G0 run at the same settings (ADR-0009 D1)."""

    def __init__(self, *, candidate_envelope: dict, registry,
                 base_config: RuntimeConfig, runner: GateEpisodeRunner,
                 budgets: Budgets, parent_cig_id: str, mining_seed: int,
                 target_regulation=None):
        self._configs: EvaluationConfigs = build_evaluation_configs(
            base_config, candidate_envelope, registry,
            target_regulation=target_regulation)
        self._candidate_gene_id = self._configs.target_gene_id
        self._runner = runner
        self._budgets = budgets
        self._parent_cig_id = parent_cig_id
        self._mining_seed = mining_seed
        # the same locked settings as every other stage: mining seed, the
        # request budget and the ReAct protocol stop (ADR-0003 A2)
        self._settings = evaluation_settings(
            max_tokens=budgets.max_tokens_per_request, seed=mining_seed)
        self._locked = {"temperature": self._settings.temperature,
                         "max_tokens": self._settings.max_tokens,
                         "seed": self._settings.seed,
                         "stop": list(self._settings.stop)}

    def evaluate(self, allocation: RegressionAllocation,
                 baseline: RegressionBaseline) -> RegressionEvaluation:
        if baseline.task_ids != tuple(t.task_id for t in allocation.tasks):
            raise RunnerContractError("the shared G0 run covers other T_reg tasks than the "
                                      "allocation (ADR-0009 D1)")
        if baseline.locked_settings != self._locked:
            raise RunnerContractError(f"the shared G0 run used settings {baseline.locked_settings}, "
                                      f"the stage locks {self._locked}")
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
        successes_without = baseline.successes
        total = len(allocation.tasks)
        drop = (successes_without - successes_with) / total
        passed = paired_drop_within(successes_with, successes_without, total, TAU_R)
        return RegressionEvaluation(
            _passed=passed, _successes_with=successes_with, _task_count=total,
            _regression_drop=drop, _episodes=tuple(copy.deepcopy(episodes)),
            _allocation=copy.deepcopy(allocation.allocation_metadata()),
            _locked_settings=copy.deepcopy(self._locked),
            _successes_without=successes_without, _baseline=baseline.record())


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
    # fail closed on the ACTUAL count for the real rendered block (the
    # construction-time probe cannot cover this input): invalid counts
    # never reach the evidence
    if not isinstance(tokens, int) or isinstance(tokens, bool) or tokens < 0:
        raise RunnerContractError(
            f"TokenCounter {counter.identity!r} returned an invalid actual "
            f"token count {tokens!r} for the rendered block (must be a "
            f"non-negative int, not bool); no overhead evidence written")
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
