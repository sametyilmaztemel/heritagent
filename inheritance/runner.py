"""Generic benchmark-agnostic episode-runner contract for CIG stages (issue
#11 criterion 3; hardened per critic re-review).

The evaluators never touch a benchmark directly: they drive a
``GateEpisodeRunner`` with typed tasks and consume typed outcomes.
Benchmark success comes from the runner/environment outcome — never from
``RunResult.status == "finished"``.

Contract hardening:
- every episode call receives the evaluator's LOCKED ``GenerationSettings``
  (temperature 0.0); the outcome must ACKNOWLEDGE the exact effective
  settings it used — mismatches and nonzero temperatures fail closed;
- completed episodes must carry a non-empty ``trajectory_id`` (``T-...``)
  so child CIG records are traceable to #8 phenotype evidence;
- target-gene expression in the WITHOUT-trait arm is an
  ``EvaluationIntegrityError`` (contaminated intervention — not candidate
  evidence), aborting the stage with no completed child record;
- central gate-setup preflight (parent CIG grammar + mining seed + child
  coherence) shared by replay and ablation, run before any episode.
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from typing import Protocol

from runtime.expression import RuntimeConfig
from runtime.loop import Budgets
from runtime.model_adapters import GenerationSettings

CONDITIONS = ("with_trait", "without_trait")

_TRAJECTORY_ID_PATTERN = re.compile(r"^T-[0-9a-zA-Z_-]+$")
_PARENT_CIG_PATTERN = re.compile(r"^CIG-[0-9]{4,}$")


class RunnerContractError(ValueError):
    """A runner returned a malformed/duplicate/mismatched outcome or ignored
    the locked evaluation settings (fail closed; no completed stage evidence
    may be built from it)."""


class EvaluationIntegrityError(Exception):
    """Target-gene expression appeared in the WITHOUT-trait arm: the paired
    intervention is contaminated and cannot produce valid evidence. The
    stage aborts with no completed child record (both stages)."""


def validate_gate_setup(parent_cig_id: str, mining_seed: int) -> None:
    """Central preflight shared by replay + ablation: parent CIG id must
    match the aggregate grammar ``^CIG-[0-9]{4,}$``, the mining seed must be
    a non-negative integer, and the derived child ids must be coherent.
    Callers run this BEFORE any episode (zero runner calls on invalid
    setup)."""
    issues: list[str] = []
    if not isinstance(parent_cig_id, str) or \
            not _PARENT_CIG_PATTERN.match(parent_cig_id):
        issues.append(f"parent CIG id {parent_cig_id!r} does not match "
                       f"^CIG-[0-9]{{4,}}$")
    if isinstance(mining_seed, bool) or not isinstance(mining_seed, int) \
            or mining_seed < 0:
        issues.append(f"mining seed {mining_seed!r} must be a non-negative integer")
    else:
        for stage in ("replay", "ablation"):
            child = f"{parent_cig_id}/S{mining_seed}-{stage}"
            if not re.match(r"^CIG-[0-9]{4,}/S[0-9]+-", child):
                issues.append(f"child id {child!r} is not coherent with the "
                               f"parent/seed")
    if issues:
        raise RunnerContractError(sorted(issues))


def deep_freeze(obj):
    """Recursively freeze JSON-like structures: dicts become MappingProxyType
    over frozen values, lists become tuples. Reads work unchanged; any
    mutation (including nested) raises TypeError. Serializers that reject
    mapping proxies should convert via ``dict(obj)`` first."""
    import types
    if isinstance(obj, dict):
        return types.MappingProxyType({k: deep_freeze(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return tuple(deep_freeze(v) for v in obj)
    return obj


class EvaluationTask:
    """One frozen task instance from a bank/allocation.

    The payload is deep-frozen at construction (nested dicts become
    mapping proxies): ``task.payload["goal"] = ...`` and nested mutations
    raise TypeError, so caller mutation cannot change the executed task
    definition. ``payload`` returns the frozen view — convert with
    ``dict(task.payload)`` if a mutable copy is needed."""

    __slots__ = ("_task_id", "_family", "_bank_id", "_payload")

    def __init__(self, task_id: str, family: str, payload: dict | None = None,
                 bank_id: str | None = None):
        object.__setattr__(self, "_task_id", task_id)
        object.__setattr__(self, "_family", family)
        object.__setattr__(self, "_bank_id", bank_id)
        object.__setattr__(self, "_payload", deep_freeze(copy.deepcopy(payload or {})))

    @property
    def task_id(self) -> str:
        return self._task_id

    @property
    def family(self) -> str:
        return self._family

    @property
    def bank_id(self) -> str | None:
        return self._bank_id

    @property
    def payload(self):
        return self._payload  # deep-frozen view; mutation raises TypeError

    def __setattr__(self, name, value):
        raise AttributeError("EvaluationTask is immutable after construction")

    def __delattr__(self, name):
        raise AttributeError("EvaluationTask is immutable after construction")

    def __deepcopy__(self, memo):
        return EvaluationTask(self._task_id, self._family,
                               copy.deepcopy(dict(self._payload)), self._bank_id)

    def __eq__(self, other):
        if not isinstance(other, EvaluationTask):
            return NotImplemented
        return (self._task_id, self._family, self._bank_id) == \
            (other._task_id, other._family, other._bank_id)

    def __hash__(self):
        return hash((self._task_id, self._family, self._bank_id))

    def __repr__(self):
        return (f"EvaluationTask(task_id={self._task_id!r}, "
                 f"family={self._family!r}, bank_id={self._bank_id!r})")


@dataclass(frozen=True)
class EpisodeOutcome:
    """Result of exactly one deterministic episode for one (task, condition).

    ``success`` is the BENCHMARK outcome supplied by the runner (never
    inferred from runtime status); ``expressed_gene_ids`` are the gene ids
    observed in ``skills_expressed`` events during the episode;
    ``effective_settings`` acknowledges the exact generation settings the
    runner used (must equal the evaluator's locked settings);
    ``trajectory_id`` (``T-...``) is REQUIRED for completed episodes so
    child CIG records trace back to #8 phenotype evidence."""

    task_id: str
    condition: str                # "with_trait" | "without_trait"
    success: bool
    expressed_gene_ids: tuple[str, ...]
    status: str                   # runtime terminal status
    trajectory_id: str | None = None   # REQUIRED non-empty T-... reference
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    steps: int | None = None
    effective_settings: GenerationSettings | None = None


class GateEpisodeRunner(Protocol):
    """Run EXACTLY ONE deterministic episode for a (task, condition) pair
    using the evaluator's LOCKED settings (temperature 0) and fixed budgets.
    The outcome must acknowledge the effective settings and carry a
    trajectory reference. Any retry inside the episode is part of that
    episode — the runner never produces replicates."""

    def run_episode(self, task: EvaluationTask, config: RuntimeConfig,
                    budgets: Budgets, condition: str,
                    settings: GenerationSettings) -> EpisodeOutcome: ...


def validate_outcome(outcome: EpisodeOutcome, expected_task_id: str,
                     condition: str,
                     settings: GenerationSettings | None = None) -> None:
    """Fail closed on malformed/duplicate/mismatched runner outcomes,
    ignored evaluation settings, or missing trajectory provenance."""
    if outcome.task_id != expected_task_id:
        raise RunnerContractError(
            f"runner returned task_id {outcome.task_id!r}, expected "
            f"{expected_task_id!r}")
    if outcome.condition != condition:
        raise RunnerContractError(
            f"runner returned condition {outcome.condition!r}, expected "
            f"{condition!r}")
    if outcome.condition not in CONDITIONS:
        raise RunnerContractError(f"unknown condition {outcome.condition!r}")
    if not isinstance(outcome.success, bool):
        raise RunnerContractError(
            f"runner outcome for {expected_task_id!r} has non-boolean success")
    # locked eval settings: acknowledged AND temperature must be zero
    if outcome.effective_settings is None:
        raise RunnerContractError(
            f"runner outcome for {expected_task_id!r} does not acknowledge "
            f"the effective generation settings")
    if outcome.effective_settings.temperature != 0.0:
        raise RunnerContractError(
            f"runner executed {expected_task_id!r} at temperature "
            f"{outcome.effective_settings.temperature!r}; CIG episodes are "
            f"locked to temperature 0.0")
    if settings is not None and outcome.effective_settings != settings:
        raise RunnerContractError(
            f"runner effective settings differ from the evaluator's locked "
            f"settings for {expected_task_id!r}: locked "
            f"{settings!r} vs acknowledged {outcome.effective_settings!r} "
            f"(temperature/max_tokens/seed/stop must match exactly)")
    # trajectory provenance is REQUIRED for completed episodes
    trajectory = outcome.trajectory_id
    if not isinstance(trajectory, str) or not _TRAJECTORY_ID_PATTERN.match(trajectory):
        raise RunnerContractError(
            f"runner outcome for {expected_task_id!r} carries no valid "
            f"trajectory_id (expected non-empty 'T-...' reference)")
