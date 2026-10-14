"""One recorded EXP-0001 episode (issue #13 Phase A).

An episode runs only when the ledger admits it: no hard cap would be
exceeded, and the ledger does not already hold its key (resume). It plays
one ALFWorld game through ``run_agent`` with a ``TrajectoryRecorder``
attached, annotates the trajectory with ALFWorld's ``won`` (ADR-0003 D6:
never the model's text, never the runtime status), and records the episode
and its model calls in the ledger."""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from evaluation.exp0001.ledger import LedgerError, RunLedger
from evaluation.exp0001.protocol import PROTOCOL
from runtime.expression import RuntimeConfig
from runtime.loop import Budgets, run_agent
from runtime.model_adapters.base import GenerationSettings, ModelAdapter
from trajectory.recorder.context import (
    BudgetSnapshot,
    ModelProvenance,
    OutcomeAnnotation,
    TrajectoryContext,
)
from trajectory.recorder.recorder import TrajectoryRecorder

EVALUATOR_ID = "alfworld-won"
EVALUATOR_VERSION = "ADR-0003-D6"


def trajectory_id_for(run_id: str, key: str) -> str:
    """A trajectory id fixed by the run and the episode key, so the ledger,
    the trajectory file and later references agree without a random id."""
    return "T-" + hashlib.sha256(f"{run_id}|{key}".encode("utf-8")).hexdigest()[:12]


def _settings_record(settings: GenerationSettings) -> dict:
    return {"temperature": settings.temperature, "max_tokens": settings.max_tokens,
            "seed": settings.seed, "stop": list(settings.stop)}


def protocol_budgets() -> Budgets:
    b = PROTOCOL["budgets"]
    return Budgets(max_steps=b["max_steps"], max_total_retries=b["max_total_retries"],
                   max_tokens_per_request=b["max_tokens_per_request"])


class EpisodeEnvironment(Protocol):
    """What ``runtime.tool_adapters.alfworld.AlfworldEnvironment`` offers."""

    def start(self) -> str: ...

    def list_tools(self): ...

    def call(self, name: str, arguments: dict): ...

    def record(self): ...


@dataclass(frozen=True)
class EpisodeSpec:
    key: str
    task_id: str
    family: str
    split: str
    config: RuntimeConfig
    settings: GenerationSettings
    tags: Mapping = field(default_factory=dict)


@dataclass(frozen=True)
class EpisodeResult:
    key: str
    task_id: str
    family: str
    success: bool
    trajectory_id: str
    trajectory_path: str
    model_calls: int
    steps: int
    invalid_commands: int
    status: str
    resumed: bool
    expressed_gene_ids: tuple = ()
    prompt_tokens: int = 0
    completion_tokens: int = 0


class EpisodeExecutor:
    def __init__(self, *, model: ModelAdapter,
                 env_factory: Callable[[str, str], EpisodeEnvironment],
                 ledger: RunLedger, trajectories_dir: Path, run_id: str, code_commit: str,
                 budgets: Budgets | None = None, clock: Callable[[], float] | None = None,
                 sleep: Callable[[float], None] = time.sleep):
        self._model = model
        self._env_factory = env_factory
        self._ledger = ledger
        self._dir = Path(trajectories_dir)
        self._run_id = run_id
        self._code_commit = code_commit
        self._budgets = budgets or protocol_budgets()
        self._clock = clock
        self._sleep = sleep

    def _context(self, spec: EpisodeSpec) -> TrajectoryContext:
        metadata = self._model.metadata()
        return TrajectoryContext(
            genome_id=spec.config.genome_id, run_id=self._run_id, experiment_id="EXP-0001",
            model=ModelProvenance.from_metadata(metadata), seed=spec.settings.seed,
            code_commit=self._code_commit, environment="alfworld", benchmark="ALFWorld",
            split=spec.split,
            budget=BudgetSnapshot(max_steps=self._budgets.max_steps,
                                  max_total_retries=self._budgets.max_total_retries,
                                  max_tokens_per_request=self._budgets.max_tokens_per_request),
            tags={"episode_key": spec.key, "task_id": spec.task_id, "family": spec.family,
                  **dict(spec.tags)})

    def run(self, spec: EpisodeSpec) -> EpisodeResult:
        if not self._ledger.admit(spec.key):  # recorded already: resume
            outcome = dict(self._ledger.done_record(spec.key)["outcome"])
            recorded = outcome.pop("settings")
            if recorded != _settings_record(spec.settings):
                # the ledger's outcome would be acknowledged under settings it
                # never ran with (GateEpisodeRunner contract)
                raise LedgerError(f"{spec.key!r} was recorded with settings {recorded}, not "
                                  f"{_settings_record(spec.settings)}")
            outcome["trajectory_path"] = str(self._dir / outcome.pop("trajectory_file"))
            outcome["expressed_gene_ids"] = tuple(outcome.get("expressed_gene_ids", ()))
            return EpisodeResult(key=spec.key, resumed=True, **outcome)
        trajectory_id = trajectory_id_for(self._run_id, spec.key)
        # a fixed-length file name: a key embeds game and gene ids of any length
        path = self._dir / f"{trajectory_id}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            # a crash after the trajectory was written but before the ledger
            # record: the episode is not counted, so it is run again cleanly
            path.unlink()
        env = self._env_factory(spec.split, spec.task_id)  # (split, game_id)
        try:
            task = env.start()
            recorder = TrajectoryRecorder(self._context(spec), path, trajectory_id=trajectory_id)
            try:
                kwargs = {"clock": self._clock} if self._clock is not None else {}
                result = run_agent(config=spec.config, model=self._model, env=env, task=task,
                                   budgets=self._budgets, settings=spec.settings,
                                   hooks=(recorder,), sleep=self._sleep, **kwargs)
                episode = env.record()
                recorder.attach_outcome(OutcomeAnnotation(
                    success=bool(episode.won), evaluator_id=EVALUATOR_ID,
                    evaluator_version=EVALUATOR_VERSION,
                    metadata={"game_id": spec.task_id, "steps": episode.steps,
                              "invalid_commands": episode.invalid_commands}))
                finalized = recorder.finalize()
            except BaseException:
                # nothing reaches the ledger; the prefix stays on disk until
                # the resumed run replaces it
                recorder.abort()
                raise
        finally:
            close = getattr(env, "close", None)
            if callable(close):
                close()
        calls = result.events.of_kind("model_called")
        expressed: list[str] = []
        for event in result.events.of_kind("skills_expressed"):
            expressed.extend(g for g in event.data["skills"] if g not in expressed)
        outcome = {"task_id": spec.task_id, "family": spec.family, "success": bool(episode.won),
                   "trajectory_id": finalized.trajectory_id,
                   "trajectory_file": path.name,  # relative: the run directory may move
                   "model_calls": len(calls), "steps": episode.steps,
                   "invalid_commands": episode.invalid_commands, "status": result.status,
                   "expressed_gene_ids": expressed,
                   "prompt_tokens": calls[-1].data["cumulative_prompt_tokens"] if calls else 0,
                   "completion_tokens":
                       calls[-1].data["cumulative_completion_tokens"] if calls else 0,
                   "settings": _settings_record(spec.settings)}
        self._ledger.record(spec.key, model_calls=len(calls), outcome=outcome)
        result = {k: v for k, v in outcome.items() if k not in ("settings", "trajectory_file")}
        return EpisodeResult(key=spec.key, resumed=False, trajectory_path=str(path),
                             **{**result, "expressed_gene_ids": tuple(expressed)})
