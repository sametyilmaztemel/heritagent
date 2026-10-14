"""G0 lifetime, mining pairs and T_reg (issue #13 Phase A; ADR-0007 E1, E3).

In each mining seed's lifetime G0 plays every T_mine task 3 times at the
mining temperature, in game_id order, attempts 0, 1, 2, each with its
pre-registered sampling seed (E1). A task with a success and a failure
gives one mining pair: its first success and its first failure. T_reg is
drawn once from every seed's lifetime by a keyed hash, with each task's
earliest success as its baseline (E3); fewer than 40 eligible tasks fail
closed."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from evaluation.benchmarks.alfworld.splits import ALFWORLD_TASK_TYPES
from evaluation.exp0001.episodes import EpisodeExecutor, EpisodeResult, EpisodeSpec
from evaluation.exp0001.protocol import (
    LIFETIME_ATTEMPTS_PER_TASK,
    MAX_TOKENS_PER_REQUEST,
    MINING_SEEDS,
    MINING_TEMPERATURE,
    PROTOCOL,
    T_REG_SEED,
    T_REG_SIZE,
    attempt_sampling_seed,
)
from evaluation.exp0001.provenance import canonical_sha256, plain
from inheritance.regression import RegressionAllocation, TRegTask
from runtime.expression import RuntimeConfig
from runtime.model_adapters.base import GenerationSettings
from trajectory.normalization import load_normalized
from traits.miner.inputs import MiningBatch, MiningPair

LIFETIME_SPLIT = "train"  # T_mine is drawn from ALFWorld train (splits.py)
T_REG_SCHEMA = "heritagent/exp0001-t-reg/0.1"


class LifetimeError(ValueError):
    """The lifetime results are incomplete or inconsistent, or a rule of
    ADR-0007 cannot be met (no pair in a seed, fewer than 40 T_reg tasks)."""


@dataclass(frozen=True)
class LifetimeTask:
    task_id: str        # the ALFWorld game_id
    family: str
    content_sha256: str


def t_mine_tasks(split_manifest: Mapping) -> tuple[LifetimeTask, ...]:
    """The T_mine pool of the frozen split manifest, in game_id order."""
    games = split_manifest["pools"]["t_mine"]["games"]
    tasks = tuple(LifetimeTask(g["game_id"], g["family"], g["content_sha256"])
                  for g in sorted(games, key=lambda g: g["game_id"]))
    if len({t.task_id for t in tasks}) != len(tasks):
        raise LifetimeError("t_mine holds a game twice")
    return tasks


def lifetime_key(mining_seed: int, task_id: str, attempt: int) -> str:
    return f"lifetime/{mining_seed}/{task_id}/{attempt}"


def lifetime_settings(mining_seed: int, task_id: str, attempt: int) -> GenerationSettings:
    from runtime.policies.planner_react import STOP_SEQUENCES
    return GenerationSettings(temperature=MINING_TEMPERATURE, max_tokens=MAX_TOKENS_PER_REQUEST,
                              seed=attempt_sampling_seed(mining_seed, task_id, attempt),
                              stop=STOP_SEQUENCES)


def lifetime_specs(tasks: Sequence[LifetimeTask], config: RuntimeConfig,
                   mining_seed: int) -> tuple[EpisodeSpec, ...]:
    if mining_seed not in MINING_SEEDS:
        raise LifetimeError(f"{mining_seed} is not a pre-registered mining seed {MINING_SEEDS}")
    return tuple(
        EpisodeSpec(key=lifetime_key(mining_seed, task.task_id, attempt), task_id=task.task_id,
                    family=task.family, split=LIFETIME_SPLIT, config=config,
                    settings=lifetime_settings(mining_seed, task.task_id, attempt),
                    tags={"stage": "lifetime", "mining_seed": mining_seed, "attempt": attempt})
        for task in sorted(tasks, key=lambda t: t.task_id)
        for attempt in range(LIFETIME_ATTEMPTS_PER_TASK))


def run_lifetime(executor: EpisodeExecutor, tasks: Sequence[LifetimeTask],
                 config: RuntimeConfig, mining_seed: int) -> tuple[EpisodeResult, ...]:
    """Run (or resume) one seed's lifetime; a hard cap stops it by raising
    the ledger's CapExceeded."""
    return tuple(executor.run(spec) for spec in lifetime_specs(tasks, config, mining_seed))


def _attempt(result: EpisodeResult, mining_seed: int) -> int:
    prefix = f"lifetime/{mining_seed}/{result.task_id}/"
    if not result.key.startswith(prefix):
        raise LifetimeError(f"{result.key!r} is not a lifetime episode of seed {mining_seed}")
    return int(result.key[len(prefix):])


def _by_task(results: Sequence[EpisodeResult], tasks: Sequence[LifetimeTask],
             mining_seed: int) -> dict[str, list[EpisodeResult]]:
    """Every (task, attempt) of the seed exactly once, attempts in order."""
    expected = {lifetime_key(mining_seed, t.task_id, a)
                for t in tasks for a in range(LIFETIME_ATTEMPTS_PER_TASK)}
    keys = [r.key for r in results]
    if len(set(keys)) != len(keys) or set(keys) != expected:
        missing, extra = sorted(expected - set(keys)), sorted(set(keys) - expected)
        raise LifetimeError(f"seed {mining_seed}'s lifetime is not complete and exact "
                            f"(missing {missing[:3]}, unexpected {extra[:3]}, "
                            f"{len(keys) - len(set(keys))} duplicates)")
    families = {t.task_id: t.family for t in tasks}
    grouped: dict[str, list[EpisodeResult]] = {t.task_id: [] for t in tasks}
    for result in results:
        if result.family != families[result.task_id]:
            raise LifetimeError(f"{result.key!r} carries family {result.family!r}, the split "
                                f"manifest says {families[result.task_id]!r}")
        grouped[result.task_id].append(result)
    for task_id in grouped:
        grouped[task_id].sort(key=lambda r: _attempt(r, mining_seed))
    return grouped


def mining_pairs(results: Sequence[EpisodeResult], tasks: Sequence[LifetimeTask],
                 mining_seed: int) -> tuple[MiningPair, ...]:
    """Per task, in game_id order: the first success and the first failure
    (E1). The success labels come from the trajectories' outcome
    annotations, which MiningPair checks again."""
    grouped = _by_task(results, tasks, mining_seed)
    pairs = []
    for task in sorted(tasks, key=lambda t: t.task_id):
        attempts = grouped[task.task_id]
        success = next((r for r in attempts if r.success), None)
        failure = next((r for r in attempts if not r.success), None)
        if success is None or failure is None:
            continue
        pairs.append(MiningPair(success=load_normalized(success.trajectory_path),
                                failure=load_normalized(failure.trajectory_path),
                                task_family=task.family))
    return tuple(pairs)


def mining_batch(results: Sequence[EpisodeResult], tasks: Sequence[LifetimeTask],
                 mining_seed: int) -> MiningBatch:
    pairs = mining_pairs(results, tasks, mining_seed)
    if not pairs:
        raise LifetimeError(f"seed {mining_seed}'s lifetime gave no mining pair: no T_mine task "
                            f"had both a success and a failure")
    return MiningBatch(pairs=pairs, family_universe=ALFWORLD_TASK_TYPES)


def t_reg_rank(task_id: str, seed: int = T_REG_SEED) -> str:
    return hashlib.sha256(f"heritagent-treg|{seed}|{task_id}".encode("utf-8")).hexdigest()


def derive_t_reg(results_by_seed: Mapping[int, Sequence[EpisodeResult]],
                 tasks: Sequence[LifetimeTask]) -> RegressionAllocation:
    """E3: one T_reg for every seed, from every seed's complete lifetime."""
    if tuple(sorted(results_by_seed)) != tuple(sorted(MINING_SEEDS)):
        raise LifetimeError(f"T_reg needs the lifetime of every mining seed {MINING_SEEDS}, "
                            f"got {sorted(results_by_seed)}")
    grouped = {seed: _by_task(results_by_seed[seed], tasks, seed) for seed in MINING_SEEDS}
    baselines: dict[str, tuple[int, EpisodeResult]] = {}
    for seed in MINING_SEEDS:  # seed order, then attempt order: the earliest success
        for task_id, attempts in grouped[seed].items():
            if task_id in baselines:
                continue
            success = next((r for r in attempts if r.success), None)
            if success is not None:
                baselines[task_id] = (seed, success)
    if len(baselines) < T_REG_SIZE:
        raise LifetimeError(f"only {len(baselines)} T_mine tasks were solved in any lifetime "
                            f"episode; T_reg needs {T_REG_SIZE} (ADR-0007 E3 fails closed: "
                            f"changing the rule needs a pre-execution amendment)")
    drawn = sorted(baselines, key=lambda task_id: (t_reg_rank(task_id), task_id))[:T_REG_SIZE]
    by_id = {t.task_id: t for t in tasks}
    allocation = []
    for task_id in sorted(drawn):
        seed, success = baselines[task_id]
        allocation.append(TRegTask(
            task_id=task_id, family=by_id[task_id].family,
            baseline_trajectory_id=success.trajectory_id,
            payload={"game_id": task_id, "content_sha256": by_id[task_id].content_sha256,
                     "split": LIFETIME_SPLIT, "pool": "t_mine",
                     "baseline_key": success.key, "baseline_seed": seed,
                     "baseline_attempt": _attempt(success, seed)}))
    return RegressionAllocation(tasks=tuple(allocation))


def t_reg_record(allocation: RegressionAllocation, eligible: int) -> dict:
    """The frozen T_reg artifact the run records before any gate episode."""
    record = {"schema": T_REG_SCHEMA, "experiment": "EXP-0001",
              "rule": plain(PROTOCOL["t_reg"]), "eligible": eligible,
              "tasks": [{"task_id": t.task_id, "family": t.family,
                         "baseline_trajectory_id": t.baseline_trajectory_id,
                         **plain(t.payload)} for t in allocation.tasks]}
    record["t_reg_sha256"] = canonical_sha256(record)
    return record


def eligible_count(results_by_seed: Mapping[int, Sequence[EpisodeResult]]) -> int:
    return len({r.task_id for results in results_by_seed.values() for r in results if r.success})
