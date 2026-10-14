"""run_agent's generation settings are an explicit input (issue #7
criterion 1; module audit, 2026-10-13). A gate runner passes the
evaluator's locked settings, so their seed and stop reach every model call
and are recorded in the model_called events."""

import pytest

from inheritance.overlay import evaluation_settings
from runtime.loop import Budgets, ToolObservation, run_agent
from runtime.model_adapters import GenerationSettings, ScriptedAdapter
from tests.runtime.conftest import ScriptedEnv, react_action, react_final

BUDGETS = Budgets(max_steps=3, max_total_retries=2, max_tokens_per_request=64)


def run(g0_config, settings=None, replies=None):
    adapter = ScriptedAdapter(replies or [react_action("act", {"command": "look"}),
                                          react_final("done")])
    env = ScriptedEnv({"act": [ToolObservation(ok=True, content="You see a fridge.")]})
    result = run_agent(config=g0_config, model=adapter, env=env, task="t", budgets=BUDGETS,
                       settings=settings, sleep=lambda _delay: None)
    return adapter, result


def test_default_settings_are_greedy_with_the_request_budget(g0_config):
    adapter, result = run(g0_config)
    assert result.status == "finished"
    assert [r.settings for r in adapter.requests] == [
        GenerationSettings(temperature=0.0, max_tokens=64, seed=None, stop=())] * 2
    assert [(e.data["seed"], e.data["stop"]) for e in result.events.of_kind("model_called")] == [
        (None, [])] * 2


def test_passed_settings_reach_every_model_call_and_the_events(g0_config):
    settings = GenerationSettings(temperature=0.0, max_tokens=64, seed=1234, stop=("Observation:",))
    adapter, result = run(g0_config, settings,
                          replies=["not ReAct", react_action("act", {"command": "look"}),
                                   react_final("done")])
    assert result.status == "finished"
    assert len(adapter.requests) == 3  # including the reprompt after the parse failure
    assert all(r.settings == settings for r in adapter.requests)
    assert [(e.data["seed"], e.data["stop"], e.data["max_tokens"])
            for e in result.events.of_kind("model_called")] == [(1234, ["Observation:"], 64)] * 3


def test_the_locked_evaluation_settings_are_the_settings_used(g0_config):
    """What a gate runner acknowledges to validate_outcome is what the model
    received, seed included."""
    locked = evaluation_settings(max_tokens=64, seed=7)
    adapter, _ = run(g0_config, locked)
    assert {r.settings for r in adapter.requests} == {locked}


def test_max_tokens_disagreeing_with_the_request_budget_fails_closed(g0_config):
    adapter = ScriptedAdapter([react_final("never asked")])
    with pytest.raises(ValueError, match=r"settings.max_tokens \(128\) must equal "
                                         r"budgets.max_tokens_per_request \(64\)"):
        run_agent(config=g0_config, model=adapter, env=ScriptedEnv({}), task="t",
                  budgets=BUDGETS, settings=GenerationSettings(max_tokens=128))
    assert adapter.requests == []  # rejected before any model call
