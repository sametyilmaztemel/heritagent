"""Frozen allocation contract tests for the stage-2 evaluator (issue #10
final re-review, groups 4-5; the allocation of ADR-0012 D4)."""

import copy

import pytest

from inheritance.runner import EvaluationTask
from tests.inheritance.test_ablation import make_allocation


def test_task_payload_snapshot_survives_caller_mutation():
    """Caller mutation of a task payload after allocation construction cannot
    change the executed task definition."""
    allocation = make_allocation()
    with pytest.raises(TypeError):
        allocation.tasks[0].payload["goal"] = "TAMPERED"
    assert dict(allocation.tasks[0].payload) == {"goal": "game 0"}


def test_the_allocation_is_a_snapshot_of_its_tasks():
    tasks = list(make_allocation().tasks)
    from inheritance.ablation import AblationAllocation
    allocation = AblationAllocation(tasks=tuple(tasks), families=("heat_and_place",),
                                    bank_sha256="b" * 64)
    tasks.reverse()  # the caller's list, not the allocation's tuple
    assert allocation.tasks[0].task_id == "bank-heat_and_place-000"
    with pytest.raises(Exception):
        allocation.tasks = ()  # frozen dataclass


def test_nested_payload_dict_list_dict_frozen():
    """Nested payload (dict -> list -> dict) is frozen; nested mutation is
    impossible through any copy."""
    task = EvaluationTask(task_id="nested-1", family="heat_and_place",
                          payload={"outer": {"list": [{"inner": {"k": "v"}}]}},
                          bank_id="benefit_bank[heat_and_place]")
    with pytest.raises(TypeError):
        task.payload["outer"]["list"][0]["inner"]["k"] = "hacked"
    with pytest.raises(TypeError):
        task.payload["outer"]["new"] = 1
    assert task.payload["outer"]["list"][0]["inner"]["k"] == "v"


def test_deepcopy_returns_self_for_immutable_task():
    task = EvaluationTask(task_id="t-1", family="heat_and_place",
                          payload={"outer": [{"inner": 1}]}, bank_id="b")
    assert copy.deepcopy(task) is task  # deeply immutable: deepcopy is identity
