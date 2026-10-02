"""Single-agent execution loop tests (issue #7 criteria 7, 8, 10) — fully scripted."""

import json

import pytest

from runtime.loop import Budgets, ToolObservation, run_agent
from tests.runtime.conftest import (
    SKILL_GENE_ID,
    ScriptedEnv,
    config_with_skill,
    react_action,
    react_final,
)

GOLDEN_BUDGETS = Budgets(max_steps=4, max_total_retries=10, max_tokens_per_request=64)


def golden_script(adapter):
    adapter.enqueue_text(react_action("heat_object", {"object": "plate"},
                                      thought="I should inspect then heat the plate."))
    adapter.enqueue_text(react_final("The plate is heated.", thought="The tool reported success."))


def test_golden_scripted_run_event_ordering(g0_config):
    adapter, seen = None, []

    from runtime.model_adapters import ScriptedAdapter
    adapter = ScriptedAdapter([])
    golden_script(adapter)
    env = ScriptedEnv({"heat_object": [ToolObservation(ok=True, content="The plate is now heated.")]})
    result = run_agent(config=g0_config, model=adapter, env=env, task="Heat the plate.",
                       budgets=GOLDEN_BUDGETS, hooks=(lambda event: seen.append(event),))

    assert result.status == "finished"
    assert result.answer == "The plate is heated."
    assert result.steps == 2
    assert env.calls == [("heat_object", {"object": "plate"})]
    expected = ["run_started",
                "step_started", "skills_expressed", "model_called", "step_decision",
                "tool_called", "tool_result",
                "step_started", "skills_expressed", "model_called", "step_decision",
                "finished"]
    assert result.events.kinds() == expected
    assert [e.kind for e in seen] == expected  # hooks see the same total order
    assert seen[0].seq == 1 and result.events.events[0].seq == 1


def test_model_messages_contain_no_evaluation_only_metadata(registry):
    from runtime.model_adapters import ScriptedAdapter
    config = config_with_skill(registry)  # skill payload carries applicability.task_families
    adapter = ScriptedAdapter([react_action("heat_object", {"object": "plate"}),
                               react_final("done")])
    env = ScriptedEnv({"heat_object": [ToolObservation(ok=True, content="heated")]})
    result = run_agent(config=config, model=adapter, env=env, task="Heat the plate.",
                       budgets=GOLDEN_BUDGETS)
    assert result.status == "finished"
    rendered = json.dumps([list(r.messages) for r in adapter.requests], default=str)
    for forbidden in ("applicability", "task_families", "cig_record", "gate_reports", "provenance"):
        assert forbidden not in rendered, f"{forbidden} leaked into model messages"
    assert "look_before_heat_v1" in rendered  # expressed skills ARE in the prompt


def test_generation_settings_propagated_explicitly(g0_config):
    from runtime.model_adapters import ScriptedAdapter
    adapter = ScriptedAdapter([react_final("done")])
    budgets = Budgets(max_steps=1, max_total_retries=2, max_tokens_per_request=123)
    run_agent(config=g0_config, model=adapter, env=ScriptedEnv({}), task="t", budgets=budgets)
    settings = adapter.requests[0].settings
    assert settings.temperature == 0.0  # EXP-0001 evaluation mode
    assert settings.max_tokens == 123


def test_constitutive_skill_expressed_conditional_not_until_threshold(registry):
    from runtime.model_adapters import ScriptedAdapter
    config = config_with_skill(
        registry, express_when={"all_of": [{"metric": "consecutive_failures", "op": ">=", "value": 2}]})
    adapter = ScriptedAdapter([
        react_action("heat_object", {"object": "plate"}),
        react_action("heat_object", {"object": "plate"}),
        react_final("gave up after failures"),
    ])
    env = ScriptedEnv({"heat_object": [ToolObservation(ok=False, content="heater offline"),
                                       ToolObservation(ok=False, content="heater offline")]})
    result = run_agent(config=config, model=adapter, env=env, task="Heat the plate.",
                       budgets=Budgets(max_steps=3, max_total_retries=0, max_tokens_per_request=64))
    # expression is evaluated at the START of each step: counters are 0, 1, 2
    expression_events = [e.data["skills"] for e in result.events.of_kind("skills_expressed")]
    assert expression_events[0] == []
    assert expression_events[1] == []
    assert expression_events[2] == [SKILL_GENE_ID]


def test_counter_semantics_after_failures(registry):
    from runtime.model_adapters import ScriptedAdapter
    config = config_with_skill(registry)
    adapter = ScriptedAdapter([react_action("heat_object", {"object": "plate"}),
                               react_final("gave up")])
    env = ScriptedEnv({"heat_object": [ToolObservation(ok=False, content="offline")]})
    result = run_agent(config=config, model=adapter, env=env, task="t",
                       budgets=Budgets(max_steps=2, max_total_retries=0, max_tokens_per_request=64))
    assert result.counters.consecutive_failures == 1
    assert result.counters.failing_tools == 1
    assert result.counters.repeated_tool_error == 1  # same tool failed
    assert result.counters.steps_without_progress == 1  # step 2 ended with Final Answer, no tool success


def test_repeated_tool_error_resets_on_other_tool_failure(g0_config):
    from runtime.model_adapters import ScriptedAdapter
    adapter = ScriptedAdapter([react_action("tool_a", {}),
                               react_action("tool_b", {}),
                               react_final("done")])
    env = ScriptedEnv({"tool_a": [ToolObservation(ok=False, content="a failed")],
                       "tool_b": [ToolObservation(ok=False, content="b failed")]})
    result = run_agent(config=g0_config, model=adapter, env=env, task="t",
                       budgets=Budgets(max_steps=3, max_total_retries=0, max_tokens_per_request=64))
    assert result.counters.repeated_tool_error == 1  # different tool -> reset to 1
    assert result.counters.failing_tools == 2


def test_tool_retry_with_backoff_then_success(g0_config):
    from runtime.model_adapters import ScriptedAdapter
    adapter = ScriptedAdapter([react_action("heat_object", {"object": "plate"}),
                               react_final("heated after retry")])
    env = ScriptedEnv({"heat_object": [ToolObservation(ok=False, content="transient"),
                                       ToolObservation(ok=True, content="heated")]})
    delays = []
    result = run_agent(config=g0_config, model=adapter, env=env, task="t",
                       budgets=Budgets(max_steps=2, max_total_retries=5, max_tokens_per_request=64),
                       sleep=delays.append)
    assert result.status == "finished"
    retry_events = result.events.of_kind("retry_scheduled")
    assert len(retry_events) == 1
    assert retry_events[0].data["reason"] == "tool_failure"
    assert delays == [1.0]  # exponential backoff, first retry: base * 2^0


def test_plan_parse_failures_retried_then_replan_succeeds(g0_config):
    from runtime.model_adapters import ScriptedAdapter
    adapter = ScriptedAdapter(["I will heat the plate now.",       # malformed
                              "Action: heat_object",               # malformed (missing parts)
                              react_action("heat_object", {"object": "plate"}),
                              react_final("done")])
    env = ScriptedEnv({"heat_object": [ToolObservation(ok=True, content="heated")]})
    delays = []
    result = run_agent(config=g0_config, model=adapter, env=env, task="t",
                       budgets=Budgets(max_steps=2, max_total_retries=5, max_tokens_per_request=64),
                       sleep=delays.append)
    assert result.status == "finished"
    assert len(result.events.of_kind("plan_parse_failed")) == 2
    assert len(result.events.of_kind("retry_scheduled")) == 2
    assert delays == [1.0, 2.0]  # exponential: 1, 2


def test_max_steps_budget_enforced(g0_config):
    from runtime.model_adapters import ScriptedAdapter
    adapter = ScriptedAdapter([react_action("heat_object", {"object": "plate"}) for _ in range(3)])
    env = ScriptedEnv({"heat_object": [ToolObservation(ok=True, content="ok")] * 3})
    result = run_agent(config=g0_config, model=adapter, env=env, task="t",
                       budgets=Budgets(max_steps=2, max_total_retries=0, max_tokens_per_request=64))
    assert result.status == "budget_exhausted"
    assert result.events.of_kind("budget_exhausted")[-1].data["budget"] == "max_steps"


def test_max_total_retries_budget_enforced(g0_config):
    from runtime.model_adapters import ScriptedAdapter
    adapter = ScriptedAdapter(["malformed", "malformed again"])
    result = run_agent(config=g0_config, model=adapter, env=ScriptedEnv({}), task="t",
                       budgets=Budgets(max_steps=3, max_total_retries=1, max_tokens_per_request=64))
    assert result.status == "budget_exhausted"
    assert result.events.of_kind("budget_exhausted")[-1].data["budget"] == "max_total_retries"


def test_max_total_tokens_budget_enforced(g0_config):
    from runtime.model_adapters import ScriptedAdapter
    adapter = ScriptedAdapter([])
    adapter.enqueue_text(react_action("heat_object", {"object": "plate"}), completion_tokens=8)
    adapter.enqueue_text("garbage two", completion_tokens=8)  # parse will fail, but budget hits first
    env = ScriptedEnv({"heat_object": [ToolObservation(ok=True, content="ok")]})
    result = run_agent(config=g0_config, model=adapter, env=env, task="t",
                       budgets=Budgets(max_steps=3, max_total_retries=5, max_tokens_per_request=64,
                                       max_total_tokens=10))
    assert result.status == "budget_exhausted"
    assert result.events.of_kind("budget_exhausted")[-1].data["budget"] == "max_total_tokens"


def test_policy_retry_cap_bounds_plan_retries(g0_config):
    # G0 retry policy allows 3 retries; even a larger loop budget stops at 3
    from runtime.model_adapters import ScriptedAdapter
    adapter = ScriptedAdapter(["bad"] * 5)
    result = run_agent(config=g0_config, model=adapter, env=ScriptedEnv({}), task="t",
                       budgets=Budgets(max_steps=2, max_total_retries=99, max_tokens_per_request=64))
    assert result.status == "budget_exhausted"
    assert len(result.events.of_kind("retry_scheduled")) == 3
    assert result.events.of_kind("budget_exhausted")[-1].data["budget"] == "max_total_retries"
