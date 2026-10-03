"""Shared fixtures/helpers for trajectory tests — ScriptedAdapter only."""

import pytest

from genome.validation.registry import TraitRegistry
from runtime.loop import Budgets, run_agent
from runtime.model_adapters import ScriptedAdapter
from runtime.seed_g0 import g0_runtime_config
from trajectory.recorder.context import (
    BudgetSnapshot,
    ModelProvenance,
    TrajectoryContext,
)
from trajectory.recorder.recorder import TrajectoryRecorder
from tests.runtime.conftest import ScriptedEnv

GOLDEN_SCRIPT = None  # built per test via helpers below


@pytest.fixture
def registry(tmp_path):
    return TraitRegistry(tmp_path / "registry")


@pytest.fixture
def context():
    return TrajectoryContext(
        genome_id="G-0f1e2d3c", run_id="R-test", seed=11,
        model=ModelProvenance(model_id="scripted-test-model", revision="test-revision",
                              backend="scripted"),
        code_commit="deadbeefcafe", benchmark="scripted-bench", split="golden",
        budget=BudgetSnapshot(max_steps=4, max_total_retries=10, max_tokens_per_request=64))


class SteppedClock:
    """Deterministic monotonic clock: 0.25 s per read."""

    def __init__(self):
        self.now = 0.0

    def __call__(self):
        self.now += 0.25
        return self.now


def run_captured(tmp_path, registry, context, *, script, env_results,
                 task="Heat the plate.", budgets=None, name="trajectory",
                 clock=None, extra_hooks=()):
    """Run one scripted episode with a TrajectoryRecorder attached as a hook."""
    config = g0_runtime_config(registry)
    adapter = ScriptedAdapter([])
    for item in script:
        if isinstance(item, tuple):
            adapter.enqueue_text(item[0], completion_tokens=item[1])
        else:
            adapter.enqueue_text(item)
    env = ScriptedEnv(env_results)
    path = tmp_path / f"{name}.jsonl"
    recorder = TrajectoryRecorder(context, path)
    budgets = budgets or Budgets(max_steps=4, max_total_retries=10, max_tokens_per_request=64)
    kwargs = {"clock": clock} if clock is not None else {}
    result = run_agent(config=config, model=adapter, env=env, task=task, budgets=budgets,
                       hooks=(recorder, *extra_hooks), **kwargs)
    return result, recorder, path, env, adapter
