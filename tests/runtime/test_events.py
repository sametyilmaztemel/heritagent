"""Event contract tests (critic review round 2, group 3): immutability,
reconstructability, latency, raw responses."""

import json

from runtime.loop import Budgets, ToolObservation, run_agent
from tests.runtime.conftest import ScriptedEnv, react_action, react_final


def test_hook_mutation_cannot_alter_eventlog_history(g0_config):
    from runtime.model_adapters import ScriptedAdapter
    adapter = ScriptedAdapter([react_final("done")])

    def hostile_hook(event):
        event.data["answer"] = "hacked"
        event.data["injected"] = True

    result = run_agent(config=g0_config, model=adapter, env=ScriptedEnv({}), task="t",
                       budgets=Budgets(max_steps=1, max_total_retries=1, max_tokens_per_request=64),
                       hooks=(hostile_hook,))
    finished = result.events.of_kind("finished")[0]
    assert finished.data["answer"] == "done"
    assert "injected" not in finished.data


def test_accessor_mutation_cannot_alter_eventlog_history(g0_config):
    from runtime.model_adapters import ScriptedAdapter
    adapter = ScriptedAdapter([react_final("done")])
    result = run_agent(config=g0_config, model=adapter, env=ScriptedEnv({}), task="t",
                       budgets=Budgets(max_steps=1, max_total_retries=1, max_tokens_per_request=64))
    snapshot = result.events.events
    snapshot[0].data["task"] = "mutated"
    snapshot[0].data["extra"] = True
    fresh = result.events.of_kind("run_started")[0]
    assert fresh.data["task"] == "t"
    assert "extra" not in fresh.data


def test_golden_trajectory_reconstructed_from_events_alone(g0_config):
    """#8 must be able to rebuild the full phenotype from the event stream
    alone — this test uses ONLY emitted events (hook copies)."""
    from runtime.model_adapters import ScriptedAdapter
    adapter = ScriptedAdapter([react_action("heat_object", {"object": "plate"},
                                            thought="Inspect then heat."),
                               react_final("The plate is heated.", thought="Success reported.")])
    seen = []
    env = ScriptedEnv({"heat_object": [ToolObservation(ok=True, content="The plate is now heated.")]})
    run_agent(config=g0_config, model=adapter, env=env, task="Heat the plate.",
              budgets=Budgets(max_steps=4, max_total_retries=10, max_tokens_per_request=64),
              hooks=(seen.append,))

    traj = {"task": None, "steps": [], "final": None}
    for event in seen:
        if event.kind == "run_started":
            traj["task"] = event.data["task"]
        elif event.kind == "step_decision" and event.data["tool"] is not None:
            traj["steps"].append({"thought": event.data["thought"]})
        elif event.kind == "tool_called":
            traj["steps"][-1]["action"] = {"name": event.data["tool"],
                                            "arguments": event.data["arguments"]}
        elif event.kind == "tool_result":
            traj["steps"][-1]["result"] = {"ok": event.data["ok"],
                                            "content": event.data["content"]}
        elif event.kind == "finished":
            traj["final"] = event.data["answer"]

    assert traj == {
        "task": "Heat the plate.",
        "steps": [{"thought": "Inspect then heat.",
                    "action": {"name": "heat_object", "arguments": {"object": "plate"}},
                    "result": {"ok": True, "content": "The plate is now heated."}}],
        "final": "The plate is heated.",
    }


def test_raw_response_and_messages_present_in_events(g0_config):
    from runtime.model_adapters import ScriptedAdapter
    adapter = ScriptedAdapter(["this is not valid ReAct output at all",
                               react_final("recovered")])
    result = run_agent(config=g0_config, model=adapter, env=ScriptedEnv({}), task="t",
                       budgets=Budgets(max_steps=2, max_total_retries=2, max_tokens_per_request=64))
    called = result.events.of_kind("model_called")[0].data
    assert called["raw_response"] == "this is not valid ReAct output at all"
    assert called["messages"][0]["role"] == "system"
    assert called["messages"][1]["role"] == "user"
    assert called["total_tokens"] > 0
    # cumulative accounting: prompt + completion tracked separately AND combined
    assert called["cumulative_prompt_tokens"] + called["cumulative_completion_tokens"] \
        == called["total_tokens"]
    parse_failed = result.events.of_kind("plan_parse_failed")[0].data
    assert parse_failed["raw_response"] == "this is not valid ReAct output at all"


def test_event_timestamps_from_injected_monotonic_clock(g0_config):
    """Every event carries monotonic_s from the SAME injected clock domain;
    stamps are strictly increasing and quantized to the fake clock steps."""
    from runtime.model_adapters import ScriptedAdapter

    class SteppedClock:
        def __init__(self):
            self.now = 0.0

        def __call__(self):
            self.now += 0.25
            return self.now

    clock = SteppedClock()
    adapter = ScriptedAdapter([react_action("heat_object", {"object": "plate"}),
                               react_final("done")])
    seen = []
    env = ScriptedEnv({"heat_object": [ToolObservation(ok=True, content="ok")]})
    result = run_agent(config=g0_config, model=adapter, env=env, task="t",
                       budgets=Budgets(max_steps=2, max_total_retries=2, max_tokens_per_request=64),
                       hooks=(seen.append,), clock=clock)
    stamps = [event.monotonic_s for event in seen]
    assert stamps == sorted(stamps)                      # ordered
    assert all(b > a for a, b in zip(stamps, stamps[1:]))  # strictly increasing
    assert all(abs(s / 0.25 - round(s / 0.25)) < 1e-9 for s in stamps)  # same clock domain
    assert stamps[0] == 0.25                             # first clock read stamps the first event
    # latency and event timestamps share the domain: model latency is still one step (250 ms)
    assert result.events.of_kind("model_called")[0].data["latency_ms"] == 250.0


def test_latency_fields_present_with_injected_clock(g0_config):
    from runtime.model_adapters import ScriptedAdapter

    class FakeClock:
        """Advances 0.25 s per read; two reads per measurement = 250 ms."""

        def __init__(self):
            self.now = 0.0

        def __call__(self):
            self.now += 0.25
            return self.now

    adapter = ScriptedAdapter([react_action("heat_object", {"object": "plate"}),
                               react_final("done")])
    env = ScriptedEnv({"heat_object": [ToolObservation(ok=True, content="ok")]})
    result = run_agent(config=g0_config, model=adapter, env=env, task="t",
                       budgets=Budgets(max_steps=2, max_total_retries=2, max_tokens_per_request=64),
                       clock=FakeClock())
    assert result.events.of_kind("model_called")[0].data["latency_ms"] == 250.0
    assert result.events.of_kind("tool_result")[0].data["latency_ms"] == 250.0
