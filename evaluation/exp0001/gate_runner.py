"""The EXP-0001 gate runners (issue #13 Phase A; ADR-0008; recalibrated by
ADR-0012 D2, owner-accepted 2026-10-19).

A seed's gate runs in six rounds. In each round, G0 plays once, at the
gate's locked settings, every game that some with-trait execution of the
round is compared with: that execution is the game's **cell** for (seed,
round). Every without-trait and neither request of the round reads it; no
other round does. Cells are played lazily, so the gate records no episode
it did not request.

Every gate episode is a ledger-admitted episode of the EpisodeExecutor, so
hard caps, resume and trajectory recording hold in the gate as in the
lifetime. Keys name the seed and the round:

- with-trait (and both-trait) executions:
  ``gate/<seed>/R<round>/<scope>/<task>/<condition>/<config fingerprint>``;
- cells: ``gate/<seed>/R<round>/cell/<task>/<G0 fingerprint>``.

The pipeline drives the gate through a ``GatePass``. A **dry** pass runs
the real gate code but no episode: a recorded episode returns its outcome,
an unrecorded one of the round being collected is collected and answered
with a placeholder, and an unrecorded one of any other round ends the pass
(``PrefetchStop``). The round's collected episodes then run as one queue
in ascending sha256 of their keys. After the last round a **read** pass
runs the gate code against recorded episodes only: an unrecorded request
fails the run closed, so no gate episode ever runs outside its round's
queue."""

from __future__ import annotations

import dataclasses
import hashlib
from collections.abc import Callable

from evaluation.exp0001.episodes import EpisodeExecutor, EpisodeResult, EpisodeSpec
from inheritance.runner import EpisodeOutcome, EvaluationTask, RunnerContractError
from runtime.expression import RuntimeConfig
from runtime.loop import Budgets
from runtime.model_adapters.base import GenerationSettings

PLACEHOLDER_TRAJECTORY = "T-prefetch-placeholder"
CELL_CONDITIONS = ("without_trait", "neither")
CELL_CONDITION_TAG = "g0_cell"


def config_fingerprint(config: RuntimeConfig) -> str:
    skills = ",".join(s.gene_id for s in config.skills)
    return hashlib.sha256(f"{config.genome_id}|{skills}".encode("utf-8")).hexdigest()[:8]


def queue_order(spec: EpisodeSpec) -> str:
    """A round's queue order: ascending sha256 of the ledger key (ADR-0012 D2)."""
    return hashlib.sha256(spec.key.encode("utf-8")).hexdigest()


def key_round(key: str) -> int | None:
    """The gate round a gate ledger key names, or None for any other key."""
    parts = key.split("/")
    if len(parts) > 2 and parts[0] == "gate" and parts[2][:1] == "R" and parts[2][1:].isdigit():
        return int(parts[2][1:])
    return None


class PrefetchStop(Exception):
    """A dry pass reached an episode it must not collect."""


class GateQueueError(RuntimeError):
    """The read pass requested an episode its round's queue did not run: the
    gate left the protocol (ADR-0012 D2) and needs a human."""


def _outcome(result: EpisodeResult, condition: str,
             settings: GenerationSettings) -> EpisodeOutcome:
    return EpisodeOutcome(
        task_id=result.task_id, condition=condition, success=result.success,
        expressed_gene_ids=tuple(result.expressed_gene_ids), status=result.status,
        trajectory_id=result.trajectory_id, prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens, steps=result.steps,
        effective_settings=settings)


def _placeholder(task: EvaluationTask, condition: str,
                 settings: GenerationSettings) -> EpisodeOutcome:
    # a placeholder no other episode of its round depends on (see the pipeline)
    return EpisodeOutcome(task_id=task.task_id, condition=condition, success=False,
                          expressed_gene_ids=(), status="prefetch-placeholder",
                          trajectory_id=PLACEHOLDER_TRAJECTORY, prompt_tokens=0,
                          completion_tokens=0, steps=0, effective_settings=settings)


class GatePass:
    """One pass of a seed's gate code (dry or read). ``runner(round, scope)``
    gives the runner of a round; every runner of the pass shares the pass's
    cells, so a game's cell is requested once per (round, pass)."""

    def __init__(self, executor: EpisodeExecutor, *, seed: int, g0_config: RuntimeConfig,
                 budgets: Budgets, mode: str, collect_round: int | None = None):
        if mode not in ("dry", "read"):
            raise ValueError(f"a gate pass is dry or read, not {mode!r}")
        if (mode == "dry") != (collect_round is not None):
            raise ValueError("a dry pass collects exactly one round")
        self._executor = executor
        self.seed = seed
        self._g0 = g0_config
        self._g0_fingerprint = config_fingerprint(g0_config)
        self._budgets = budgets
        self.mode = mode
        self.collect_round = collect_round
        self.collected: list[EpisodeSpec] = []
        self.requested: list[str] = []
        self._cells: dict[tuple[int, str], tuple[EpisodeOutcome, GenerationSettings]] = {}
        self._keys: set[str] = set()
        self.cell_reads: dict[tuple[int, str], int] = {}  # (round, task) -> readers

    def runner(self, gate_round: int, scope: str) -> "RoundRunner":
        return RoundRunner(self, gate_round, scope)

    def _execute(self, spec: EpisodeSpec, gate_round: int, task: EvaluationTask,
                 condition: str, settings: GenerationSettings) -> EpisodeOutcome:
        recorded = self._executor.recorded(spec)
        if recorded is not None:
            self.requested.append(spec.key)
            return _outcome(recorded, condition, settings)
        if self.mode == "read":
            raise GateQueueError(f"{spec.key} was not run in round R{gate_round}'s queue; the "
                                 f"gate requested an episode outside its round (ADR-0012 D2)")
        if gate_round != self.collect_round:
            raise PrefetchStop(spec.key)
        self.collected.append(spec)
        return _placeholder(task, condition, settings)

    def _spec(self, key: str, task: EvaluationTask, config: RuntimeConfig,
              settings: GenerationSettings, tags: dict) -> EpisodeSpec:
        return EpisodeSpec(key=key, task_id=task.task_id, family=task.family, split="train",
                           config=config, settings=settings, tags=tags)

    def run_with(self, gate_round: int, scope: str, task: EvaluationTask,
                 config: RuntimeConfig, condition: str,
                 settings: GenerationSettings) -> EpisodeOutcome:
        fingerprint = config_fingerprint(config)
        key = f"gate/{self.seed}/R{gate_round}/{scope}/{task.task_id}/{condition}/{fingerprint}"
        if key in self._keys:
            raise RunnerContractError(f"{key} was requested twice in one pass; exactly one "
                                      f"execution per (seed, round, task, condition) is allowed")
        self._keys.add(key)
        spec = self._spec(key, task, config, settings, {
            "stage": scope, "condition": condition, "bank_id": task.bank_id,
            "skills": [s.gene_id for s in config.skills], "round": gate_round,
            "mining_seed": self.seed, "cell": False})
        return self._execute(spec, gate_round, task, condition, settings)

    def read_cell(self, gate_round: int, task: EvaluationTask, config: RuntimeConfig,
                  condition: str, settings: GenerationSettings) -> EpisodeOutcome:
        if config != self._g0 or config_fingerprint(config) != self._g0_fingerprint:
            raise RunnerContractError(f"a {condition} request on {task.task_id!r} must run G0's "
                                      f"exact config: it is the round's G0 cell (ADR-0012 D2)")
        cached = self._cells.get((gate_round, task.task_id))
        if cached is not None:
            outcome, cell_settings = cached
            if cell_settings != settings:
                raise RunnerContractError(f"the R{gate_round} cell of {task.task_id!r} ran at "
                                          f"{cell_settings!r}, a reader asks for {settings!r}")
            self.cell_reads[(gate_round, task.task_id)] += 1
            return dataclasses.replace(outcome, condition=condition)
        key = f"gate/{self.seed}/R{gate_round}/cell/{task.task_id}/{self._g0_fingerprint}"
        spec = self._spec(key, task, config, settings, {
            "stage": "cell", "condition": CELL_CONDITION_TAG, "bank_id": task.bank_id,
            "skills": [s.gene_id for s in config.skills], "round": gate_round,
            "mining_seed": self.seed, "cell": True})
        outcome = self._execute(spec, gate_round, task, condition, settings)
        self._cells[(gate_round, task.task_id)] = (outcome, settings)
        self.cell_reads[(gate_round, task.task_id)] = 1
        return outcome


class RoundRunner:
    """The ``GateEpisodeRunner`` of one round and scope within a pass. A
    without-trait or neither request is the round's G0 cell of the game; any
    other request is the scope's own execution."""

    def __init__(self, gate_pass: GatePass, gate_round: int, scope: str):
        self._pass = gate_pass
        self.round = gate_round
        self._scope = scope

    def run_episode(self, task: EvaluationTask, config: RuntimeConfig, budgets: Budgets,
                    condition: str, settings: GenerationSettings) -> EpisodeOutcome:
        if budgets != self._pass._budgets:
            raise RunnerContractError(f"gate budgets {budgets} differ from the protocol "
                                      f"budgets {self._pass._budgets} (ADR-0007 E5)")
        if settings is None or settings.temperature != 0.0:
            raise RunnerContractError("gate episodes run at the locked temperature 0")
        if settings.seed != self._pass.seed:
            raise RunnerContractError(f"gate episodes of seed {self._pass.seed} run at that "
                                      f"seed, not {settings.seed!r}")
        split = task.payload.get("split")
        if split != "train" or task.payload.get("pool") == "t_eval":
            # contamination rule: T_eval is never visible to the gate
            raise RunnerContractError(f"gate task {task.task_id!r} must be a train game, got "
                                      f"split {split!r}; T_eval never enters the gate")
        if condition in CELL_CONDITIONS:
            return self._pass.read_cell(self.round, task, config, condition, settings)
        return self._pass.run_with(self.round, self._scope, task, config, condition, settings)


RunnerFactory = Callable[[int, str], RoundRunner]
