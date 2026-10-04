"""Shared fixtures/helpers for Trait Miner tests — ScriptedAdapter only.

Episodes are real #7-runtime runs captured by the #8 recorder and reduced
through the merged load_normalized() surface, so mining tests exercise the
exact production path."""

import pytest

from genome.validation.registry import TraitRegistry
from runtime.loop import Budgets, ToolObservation, run_agent
from runtime.model_adapters import ScriptedAdapter
from runtime.seed_g0 import g0_runtime_config
from trajectory.normalization import load_normalized
from trajectory.recorder.context import OutcomeAnnotation, TrajectoryContext
from tests.runtime.conftest import ScriptedEnv
from trajectory.recorder.recorder import TrajectoryRecorder

FAMILIES = ("heat_and_place", "clean_surface", "navigate")
TASK = "Heat the plate."


def react(tool: str, arguments: dict, thought: str = "Proceeding.") -> str:
    import json as _json
    return (f"Thought: {thought}\nAction: {tool}\n"
            f"Action Input: {_json.dumps(arguments, sort_keys=True)}")


def final(answer: str, thought: str = "Done.") -> str:
    return f"Thought: {thought}\nFinal Answer: {answer}"


@pytest.fixture
def registry(tmp_path):
    return TraitRegistry(tmp_path / "registry")


@pytest.fixture
def context():
    return TrajectoryContext(genome_id="G-0f1e2d3c", run_id="R-miner", seed=11)


def run_episode(tmp_path, registry, *, name: str, success: bool, task: str = TASK,
                trajectory_id: str | None = None):
    """One scripted episode -> verified NormalizedTrajectory carrying a
    generic outcome annotation (success=True/False)."""
    config = g0_runtime_config(registry)
    adapter = ScriptedAdapter([])
    if success:
        adapter.enqueue_text(react("heat_object", {"object": "plate"}))
        adapter.enqueue_text(final("The plate is heated."))
        env_results = {"heat_object": [ToolObservation(ok=True, content="heated")]}
    else:
        adapter.enqueue_text(react("heat_object", {"object": "plate"}))
        env_results = {"heat_object": [ToolObservation(ok=False, content="heater offline")]}
    env = ScriptedEnv(env_results)
    path = tmp_path / f"{name}.jsonl"
    recorder = TrajectoryRecorder(TrajectoryContext(genome_id="G-0f1e2d3c",
                                                     run_id=f"R-{name}"), path,
                                   trajectory_id=trajectory_id)
    budgets = Budgets(max_steps=4 if success else 1,
                      max_total_retries=10 if success else 0,
                      max_tokens_per_request=64)
    result = run_agent(config=config, model=adapter, env=env, task=task,
                       budgets=budgets, hooks=(recorder,))
    expected_status = "finished" if success else "budget_exhausted"
    assert result.status == expected_status
    recorder.attach_outcome(OutcomeAnnotation(
        success=success, reward=1.0 if success else 0.0,
        evaluator_id="scripted-evaluator", evaluator_version="1.0"))
    assert recorder.finalize().status == expected_status
    return load_normalized(path)


def make_pair(tmp_path, registry, *, family="heat_and_place",
              success_task=TASK, failure_task=TASK,
              success_trajectory_id=None, failure_trajectory_id=None) -> "MiningPair":
    from traits.miner.inputs import MiningPair
    success = run_episode(tmp_path, registry, name="success-ep", success=True,
                           task=success_task, trajectory_id=success_trajectory_id)
    failure = run_episode(tmp_path, registry, name="failure-ep", success=False,
                           task=failure_task, trajectory_id=failure_trajectory_id)
    return MiningPair(success=success, failure=failure, task_family=family)


def make_batch(tmp_path, registry, *, family="heat_and_place", families=FAMILIES,
               success_task=TASK, failure_task=TASK,
               success_trajectory_id=None, failure_trajectory_id=None):
    from traits.miner.inputs import MiningBatch
    pair = make_pair(tmp_path, registry, family=family,
                     success_task=success_task, failure_task=failure_task,
                     success_trajectory_id=success_trajectory_id,
                     failure_trajectory_id=failure_trajectory_id)
    return MiningBatch(pairs=(pair,), family_universe=tuple(families)), pair
