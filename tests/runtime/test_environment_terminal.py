"""Environment-ended episodes and deterministic tool failures (ADR-0003 D3,
D5): ToolObservation.terminal ends the run with status
``environment_terminal``; ``retryable=False`` skips the retry policy."""

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from runtime.loop import Budgets, ToolObservation, run_agent
from runtime.model_adapters import ScriptedAdapter
from tests.runtime.conftest import ScriptedEnv, react_action, react_final

BUDGETS = Budgets(max_steps=4, max_total_retries=10, max_tokens_per_request=64)


def test_terminal_observation_ends_the_run(g0_config):
    adapter = ScriptedAdapter([react_action("act", {"command": "put egg in fridge"}),
                               react_final("should never be asked")])
    env = ScriptedEnv({"act": [ToolObservation(ok=True, content="You put the egg.",
                                               terminal=True)]})
    result = run_agent(config=g0_config, model=adapter, env=env, task="t", budgets=BUDGETS)
    assert result.status == "environment_terminal"
    assert result.answer is None and result.steps == 1
    assert result.events.kinds()[-3:] == ["tool_called", "tool_result", "environment_terminal"]
    assert len(result.events.of_kind("model_called")) == 1  # no step after the end


def test_a_terminal_failure_is_not_retried(g0_config):
    """terminal takes precedence over retryable (module audit, 2026-10-13):
    an episode that ended in a failure, with retryable left at its default,
    is never called again."""
    adapter = ScriptedAdapter([react_action("act", {"command": "open fridge"}),
                               react_final("should never be asked")])
    env = ScriptedEnv({"act": [ToolObservation(ok=False, content="You lost.", terminal=True),
                               ToolObservation(ok=True, content="never sent")]})
    delays = []
    result = run_agent(config=g0_config, model=adapter, env=env, task="t", budgets=BUDGETS,
                       sleep=delays.append)
    assert result.status == "environment_terminal"
    assert env.calls == [("act", {"command": "open fridge"})]
    assert not result.events.of_kind("retry_scheduled") and delays == []
    assert result.events.kinds()[-3:] == ["tool_called", "tool_result", "environment_terminal"]
    assert result.counters.failing_tools == 1


def test_non_retryable_failure_is_not_retried(g0_config):
    adapter = ScriptedAdapter([react_action("act", {"command": "fly"}),
                               react_final("gave up")])
    env = ScriptedEnv({"act": [ToolObservation(ok=False, content="Nothing happens.",
                                               retryable=False)]})
    delays = []
    result = run_agent(config=g0_config, model=adapter, env=env, task="t", budgets=BUDGETS,
                       sleep=delays.append)
    assert env.calls == [("act", {"command": "fly"})]
    assert not result.events.of_kind("retry_scheduled") and delays == []
    # still an agent-observable failure for the regulation counters
    assert result.counters.failing_tools == 1
    assert result.counters.steps_without_progress == 1


def test_retryable_failure_still_retries(g0_config):
    adapter = ScriptedAdapter([react_action("act", {"command": "look"}),
                               react_final("ok")])
    env = ScriptedEnv({"act": [ToolObservation(ok=False, content="transient"),
                               ToolObservation(ok=True, content="You see a fridge.")]})
    result = run_agent(config=g0_config, model=adapter, env=env, task="t", budgets=BUDGETS,
                       sleep=lambda s: None)
    assert len(result.events.of_kind("retry_scheduled")) == 1
    assert result.status == "finished"


def test_planner_schema_allows_the_adr0003_step_budget():
    schema = json.loads(Path(
        "runtime/expression/payload_schemas/planner_policy.schema.json").read_text())
    validator = Draft202012Validator(schema)
    assert not list(validator.iter_errors({"style": "react", "max_plan_steps": 50}))
    assert list(validator.iter_errors({"style": "react", "max_plan_steps": 65}))
