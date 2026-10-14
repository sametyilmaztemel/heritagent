"""Nested task payloads (module audit, 2026-10-13): a TRegTask freezes its
payload, so nested dicts arrive as mapping proxies. EvaluationTask could
not deep-copy them, and RegressionEvaluator raised TypeError on the first
task whose payload held a nested dict or a list of dicts."""

import pytest

from inheritance.regression import RegressionAllocation, TRegTask
from inheritance.runner import EvaluationTask, deep_freeze
from tests.inheritance.test_regression_overhead import ScriptedRegRunner, make_evaluator

PAYLOAD = {"goal": "heat the plate",
           "objects": [{"name": "plate", "state": {"hot": False}}],
           "room": {"name": "kitchen", "receptacles": ["microwave 1", "fridge 1"]}}


class PayloadRecordingRunner(ScriptedRegRunner):
    def __init__(self, successes: int):
        super().__init__(successes)
        self.payloads: list = []

    def run_episode(self, task, config, budgets, condition, settings=None):
        self.payloads.append(task.payload)
        return super().run_episode(task, config, budgets, condition, settings)


def test_regression_runs_tasks_with_nested_payloads(tmp_path, registry, base_config, envelope):
    runner = PayloadRecordingRunner(successes=40)
    allocation = RegressionAllocation(tasks=tuple(
        TRegTask(task_id=f"treg-{i}", family="heat_and_place",
                 baseline_trajectory_id=f"T-{i:04d}aaaa", payload=PAYLOAD)
        for i in range(40)))
    result = make_evaluator(tmp_path, registry, base_config, envelope, runner).evaluate(allocation)
    assert result.successes_with == 40
    seen = runner.payloads[0]
    assert seen["objects"][0]["state"]["hot"] is False
    assert list(seen["room"]["receptacles"]) == ["microwave 1", "fridge 1"]
    with pytest.raises(TypeError):
        seen["room"]["name"] = "bathroom"  # still deep-frozen for the runner


@pytest.mark.parametrize("payload", [PAYLOAD, deep_freeze(PAYLOAD)])
def test_an_evaluation_task_accepts_plain_and_frozen_payloads(payload):
    task = EvaluationTask(task_id="t-1", family="heat_and_place", payload=payload)
    assert task.payload["objects"][0]["state"]["hot"] is False
    with pytest.raises(TypeError):
        task.payload["objects"][0]["state"]["hot"] = True


def test_a_tuple_of_dicts_is_frozen_too():
    task = EvaluationTask(task_id="t-1", family="heat_and_place",
                          payload={"steps": ({"id": "S1"},)})
    with pytest.raises(TypeError):
        task.payload["steps"][0]["id"] = "S2"
