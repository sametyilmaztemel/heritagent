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

from evaluation.exp0001.ledger import AdmissionDeferred, LedgerError, RunLedger
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


class RunPaused(Exception):
    """The operator's pause condition held before an admission: no new
    episode was admitted, every admitted one is recorded, and the run is
    still running. A later process resumes it."""


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
    parse_failures: int = 0
    retries: int = 0


def play_episode(*, model: ModelAdapter, env_factory: Callable[[str, str], EpisodeEnvironment],
                 spec: EpisodeSpec, context: TrajectoryContext, path: Path, trajectory_id: str,
                 budgets: Budgets, clock: Callable[[], float] | None = None,
                 sleep: Callable[[float], None] = time.sleep) -> dict:
    """Play one admitted episode and write its trajectory; returns the
    ledger outcome. It touches no ledger, so it can run in a worker
    process while the parent stays the ledger's only writer."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        # a crash after the trajectory was written but before the ledger
        # record: the episode is not counted, so it is run again cleanly
        path.unlink()
    env = env_factory(spec.split, spec.task_id)  # (split, game_id)
    try:
        task = env.start()
        recorder = TrajectoryRecorder(context, path, trajectory_id=trajectory_id)
        try:
            kwargs = {"clock": clock} if clock is not None else {}
            result = run_agent(config=spec.config, model=model, env=env, task=task,
                               budgets=budgets, settings=spec.settings,
                               hooks=(recorder,), sleep=sleep, **kwargs)
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
    return {"task_id": spec.task_id, "family": spec.family, "success": bool(episode.won),
            "trajectory_id": finalized.trajectory_id,
            "trajectory_file": path.name,  # relative: the run directory may move
            "model_calls": len(calls), "steps": episode.steps,
            "invalid_commands": episode.invalid_commands, "status": result.status,
            "expressed_gene_ids": expressed,
            "prompt_tokens": calls[-1].data["cumulative_prompt_tokens"] if calls else 0,
            "completion_tokens": calls[-1].data["cumulative_completion_tokens"] if calls else 0,
            "parse_failures": len(result.events.of_kind("plan_parse_failed")),
            "retries": len(result.events.of_kind("retry_scheduled")),
            "settings": _settings_record(spec.settings)}


# -- worker processes ---------------------------------------------------------
# Parallel episodes run in worker processes (one TextWorld game per process
# at a time; TextWorld is not documented as thread-safe). Each worker holds
# its own model client and environment factory, set once by the initializer.
_WORKER: dict = {}
WORKER_TASKS = 50  # episodes a worker plays before it is replaced


def _init_worker(model, env_factory, budgets, sleep_between_retries: bool) -> None:
    _WORKER.update(model=model, env_factory=env_factory, budgets=budgets,
                   sleep=time.sleep if sleep_between_retries else _no_sleep)


def _no_sleep(_seconds: float) -> None:
    return None


def _play_in_worker(spec: EpisodeSpec, context: TrajectoryContext, path: str,
                    trajectory_id: str) -> dict:
    return play_episode(model=_WORKER["model"], env_factory=_WORKER["env_factory"], spec=spec,
                        context=context, path=Path(path), trajectory_id=trajectory_id,
                        budgets=_WORKER["budgets"], sleep=_WORKER["sleep"])


class EpisodeExecutor:
    """Runs ledger-admitted episodes, one at a time (``run``) or several at
    once in worker processes (``run_many`` with ``workers`` > 1). Either
    way the parent admits every episode before it starts and records it
    after it ends, so the ledger has one writer, and the caps count the
    worst case of every episode in flight."""

    def __init__(self, *, model: ModelAdapter,
                 env_factory: Callable[[str, str], EpisodeEnvironment],
                 ledger: RunLedger, trajectories_dir: Path, run_id: str, code_commit: str,
                 budgets: Budgets | None = None, clock: Callable[[], float] | None = None,
                 sleep: Callable[[float], None] = time.sleep, workers: int = 1,
                 sleep_between_retries: bool = True,
                 pause_when: Callable[[], bool] | None = None):
        """``pause_when`` is asked before every new admission; when it
        holds, the episodes in flight finish and RunPaused is raised."""
        if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
            raise ValueError(f"workers must be a positive int, got {workers!r}")
        self._model = model
        self._env_factory = env_factory
        self._ledger = ledger
        self._dir = Path(trajectories_dir)
        self._run_id = run_id
        self._code_commit = code_commit
        self._budgets = budgets or protocol_budgets()
        self._clock = clock
        self._sleep = sleep
        self._workers = workers
        self._sleep_between_retries = sleep_between_retries
        self._pause_when = pause_when
        self._pool = None

    @property
    def workers(self) -> int:
        return self._workers

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

    def _resumed(self, spec: EpisodeSpec) -> EpisodeResult:
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

    def recorded(self, spec: EpisodeSpec) -> EpisodeResult | None:
        """The recorded result of an episode, or None if the ledger lacks it."""
        return self._resumed(spec) if self._ledger.is_done(spec.key) else None

    def _target(self, spec: EpisodeSpec) -> tuple[Path, str]:
        trajectory_id = trajectory_id_for(self._run_id, spec.key)
        # a fixed-length file name: a key embeds game and gene ids of any length
        return self._dir / f"{trajectory_id}.jsonl", trajectory_id

    def _record(self, spec: EpisodeSpec, outcome: dict) -> EpisodeResult:
        self._ledger.record(spec.key, model_calls=outcome["model_calls"], outcome=outcome)
        result = {k: v for k, v in outcome.items() if k not in ("settings", "trajectory_file")}
        return EpisodeResult(key=spec.key, resumed=False,
                             trajectory_path=str(self._dir / outcome["trajectory_file"]),
                             **{**result, "expressed_gene_ids": tuple(result["expressed_gene_ids"])})

    def _paused(self) -> bool:
        return self._pause_when is not None and bool(self._pause_when())

    def run(self, spec: EpisodeSpec) -> EpisodeResult:
        if self._ledger.is_done(spec.key):  # recorded already: resume
            return self._resumed(spec)
        if self._paused():
            raise RunPaused(f"paused before {spec.key!r}")
        self._ledger.admit(spec.key)
        path, trajectory_id = self._target(spec)
        try:
            outcome = play_episode(model=self._model, env_factory=self._env_factory, spec=spec,
                                   context=self._context(spec), path=path,
                                   trajectory_id=trajectory_id, budgets=self._budgets,
                                   clock=self._clock, sleep=self._sleep)
        except BaseException:
            self._ledger.abort(spec.key)  # charged now, as a reload would charge it
            raise
        return self._record(spec, outcome)

    # -- several at once ------------------------------------------------------
    def _get_pool(self):
        if self._pool is None:
            import multiprocessing
            import sys
            from concurrent.futures import ProcessPoolExecutor
            # a worker is replaced after a number of games: TextWorld's caches
            # grow with every game a process opens (Python 3.11+)
            recycle = {"max_tasks_per_child": WORKER_TASKS} if sys.version_info >= (3, 11) else {}
            self._pool = ProcessPoolExecutor(
                max_workers=self._workers, mp_context=multiprocessing.get_context("spawn"),
                initializer=_init_worker,
                initargs=(self._model, self._env_factory, self._budgets,
                          self._sleep_between_retries), **recycle)
        return self._pool

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None

    def run_many(self, specs) -> tuple[EpisodeResult, ...]:
        """Run independent episodes, ``workers`` at a time; results come back
        in the order of ``specs``. Before an admission that a cap would block
        while episodes are in flight, the in-flight episodes are finished and
        recorded first; only then may the admission stop the run."""
        specs = list(specs)
        if len({s.key for s in specs}) != len(specs):
            raise ValueError("run_many needs distinct episode keys")
        if self._workers == 1:
            return tuple(self.run(spec) for spec in specs)
        from concurrent.futures import ALL_COMPLETED, FIRST_COMPLETED, wait
        results: dict[str, EpisodeResult] = {}
        pending: dict = {}  # future -> spec
        failure: BaseException | None = None

        def drain(block: bool) -> None:
            nonlocal failure
            done, _ = wait(list(pending), return_when=FIRST_COMPLETED if block else ALL_COMPLETED)
            for future in done:
                spec = pending.pop(future)
                try:
                    outcome = future.result()
                except BaseException as exc:  # the episode raised in its worker
                    self._ledger.abort(spec.key)  # charged now, as a reload would charge it
                    failure = failure or exc
                    continue
                try:
                    results[spec.key] = self._record(spec, outcome)
                except BaseException as exc:
                    failure = failure or exc

        try:
            for spec in specs:
                if failure is not None:
                    break
                if self._ledger.is_done(spec.key):
                    results[spec.key] = self._resumed(spec)
                    continue
                while True:
                    if self._paused():
                        if pending:
                            drain(block=True)
                            if failure is not None:
                                break
                            continue
                        failure = RunPaused(f"paused before {spec.key!r}")
                        break
                    try:
                        # with episodes in flight a cap defers the admission
                        # instead of stopping the run; without, it stops it
                        self._ledger.admit(spec.key, defer=bool(pending))
                    except AdmissionDeferred:
                        drain(block=True)
                        if failure is not None:
                            break
                        continue
                    break
                if failure is not None:
                    break
                path, trajectory_id = self._target(spec)
                try:
                    future = self._get_pool().submit(_play_in_worker, spec, self._context(spec),
                                                     str(path), trajectory_id)
                except BaseException:
                    self._ledger.abort(spec.key)  # no worker took it: charged as a reload would
                    raise
                pending[future] = spec
                while len(pending) >= self._workers:
                    drain(block=True)
        except BaseException as exc:
            while pending:
                drain(block=False)
            if failure is not None and failure is not exc and hasattr(exc, "add_note"):
                exc.add_note(f"while stopping, an episode in flight also failed: {failure!r}")
            raise
        while pending:
            drain(block=False)
        if failure is not None:
            raise failure
        return tuple(results[spec.key] for spec in specs)
