"""Generic benchmark-agnostic episode-runner contract for CIG stages (issue
#11 criterion 3).

The evaluators never touch a benchmark directly: they drive a
``GateEpisodeRunner`` with typed tasks and consume typed outcomes.
Benchmark success comes from the runner/environment outcome — never from
``RunResult.status == "finished"``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from runtime.expression import RuntimeConfig
from runtime.loop import Budgets

CONDITIONS = ("with_trait", "without_trait")


class RunnerContractError(ValueError):
    """A runner returned a malformed/duplicate/mismatched outcome
    (fail closed; no completed stage evidence may be built from it)."""


@dataclass(frozen=True)
class EvaluationTask:
    """One frozen task instance from a bank/allocation."""

    task_id: str
    family: str
    payload: dict = field(default_factory=dict)   # benchmark-agnostic task content/ref
    bank_id: str | None = None                    # bank/allocation provenance


@dataclass(frozen=True)
class EpisodeOutcome:
    """Result of exactly one deterministic episode for one (task, condition).

    ``success`` is the BENCHMARK outcome supplied by the runner (never
    inferred from runtime status); ``expressed_gene_ids`` are the gene ids
    observed in ``skills_expressed`` events during the episode."""

    task_id: str
    condition: str                # "with_trait" | "without_trait"
    success: bool
    expressed_gene_ids: tuple[str, ...]
    status: str                   # runtime terminal status
    trajectory_id: str | None = None   # #8 recorder reference when captured
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    steps: int | None = None


class GateEpisodeRunner(Protocol):
    """Run EXACTLY ONE deterministic episode for a (task, condition) pair at
    eval temperature 0 with fixed budgets. The evaluator passes the
    condition explicitly alongside the paired config. Any retry inside the
    episode is part of that episode — the runner never produces replicates."""

    def run_episode(self, task: EvaluationTask, config: RuntimeConfig,
                    budgets: Budgets, condition: str) -> EpisodeOutcome: ...


def validate_outcome(outcome: EpisodeOutcome, expected_task_id: str,
                      condition: str) -> None:
    """Fail closed on malformed/duplicate/mismatched runner outcomes."""
    if outcome.task_id != expected_task_id:
        raise RunnerContractError(
            f"runner returned task_id {outcome.task_id!r}, expected "
            f"{expected_task_id!r}")
    if outcome.condition != condition:
        raise RunnerContractError(
            f"runner returned condition {outcome.condition!r}, expected "
            f"{condition!r}")
    if not isinstance(outcome.success, bool):
        raise RunnerContractError(
            f"runner outcome for {expected_task_id!r} has non-boolean success")
    if outcome.condition not in CONDITIONS:
        raise RunnerContractError(f"unknown condition {outcome.condition!r}")
