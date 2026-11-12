"""ReAct reply parsing (module audit, 2026-10-13). A reply is exactly one
step: an Action with its Action Input, or a non-empty Final Answer. The
"Final Answer:" marker counts only at the start of a line, and a reply that
both acts and finishes fails closed, so the retry policy reprompts instead
of ending the run without sending the action."""

import json
import sys

import pytest

from runtime.loop import Budgets, ToolObservation, run_agent
from runtime.policies.errors import PlanParseError
from runtime.policies.planner_react import ReActPlanner
from tests.runtime.conftest import ScriptedEnv, react_action, react_final

ACTION = react_action("act", {"command": "put egg 1 in/on microwave 1"}, thought="Place the egg.")


def test_a_final_answer_parses():
    decision = ReActPlanner.parse(react_final("The egg is in the microwave.", thought="All done."))
    assert decision.action is None
    assert decision.final_answer == "The egg is in the microwave."
    assert decision.thought == "All done."


def test_a_multi_line_final_answer_is_kept_whole():
    decision = ReActPlanner.parse("Thought: Done.\nFinal Answer: first line\nsecond line\n")
    assert decision.final_answer == "first line\nsecond line"


def test_an_indented_final_answer_marker_counts():
    assert ReActPlanner.parse("Thought: Done.\n  Final Answer: yes").final_answer == "yes"


@pytest.mark.parametrize("reply", [
    "Thought: Done.\nFinal Answer:",
    "Thought: Done.\nFinal Answer: ",
    "Thought: Done.\nFinal Answer:   \n",
    "Thought: Done.\nFinal Answer:\n\n\t\n",
])
def test_an_empty_final_answer_is_rejected(reply):
    with pytest.raises(PlanParseError, match="Final Answer present but empty"):
        ReActPlanner.parse(reply)


@pytest.mark.parametrize("reply", [
    f"{ACTION}\n\nFinal Answer: The egg has been placed in the microwave.",
    "Thought: Done.\nFinal Answer: placed\nAction: act\nAction Input: {\"command\": \"look\"}",
    "Thought: Done.\nAction Input: {\"command\": \"look\"}\nFinal Answer: placed",
    "Thought: Done.\nAction: act\nFinal Answer: placed",
])
def test_a_reply_that_both_acts_and_finishes_is_rejected(reply):
    with pytest.raises(PlanParseError, match="both an Action and a Final Answer"):
        ReActPlanner.parse(reply)


DECORATIONS = ["  ", "\t", "- ", "* ", "> ", ">> ", "1. ", "2) ", "# ", "`", "**",
               "\u00a0", "\u3000", "\u200b", "\ufeff", "\u2022 "]


@pytest.mark.parametrize("prefix", DECORATIONS)
def test_a_decorated_action_with_a_final_answer_is_rejected(prefix):
    """Review of #55: the action path accepts Action lines behind list or
    quote decoration, so the mixed-step check must see them too."""
    action = (f"Thought: Place the egg.\n{prefix}Action: act\n"
              f"{prefix}Action Input: {{\"command\": \"look\"}}")
    assert ReActPlanner.parse(action).action.name == "act"  # one step: still an action
    with pytest.raises(PlanParseError, match="both an Action and a Final Answer"):
        ReActPlanner.parse(f"{action}\nFinal Answer: The egg has been placed.")


def test_a_final_answer_marker_inside_a_thought_is_prose():
    reply = ("Thought: I am not ready for a Final Answer: yet.\n"
             "Action: act\nAction Input: {\"command\": \"look\"}")
    decision = ReActPlanner.parse(reply)
    assert decision.final_answer is None
    assert decision.action.name == "act" and decision.action.arguments == {"command": "look"}


def test_a_final_answer_marker_inside_action_input_is_prose():
    arguments = {"command": "say Final Answer: hello"}
    decision = ReActPlanner.parse(react_action("act", arguments))
    assert decision.final_answer is None
    assert decision.action.arguments == arguments


def test_an_action_marker_inside_a_final_answer_line_is_prose():
    decision = ReActPlanner.parse("Thought: No further Action: needed.\nFinal Answer: done")
    assert decision.final_answer == "done"


def test_a_mixed_reply_is_reprompted_and_the_action_is_sent(g0_config):
    """Through the loop: the mixed reply no longer ends the run with the
    action unsent. The retry policy reprompts, and the next reply's action
    reaches the environment."""
    from runtime.model_adapters import ScriptedAdapter
    adapter = ScriptedAdapter([f"{ACTION}\n\nFinal Answer: The egg has been placed.",
                               ACTION, react_final("done")])
    env = ScriptedEnv({"act": [ToolObservation(ok=True, content="You put the egg 1 in the microwave 1.")]})
    result = run_agent(config=g0_config, model=adapter, env=env, task="t",
                       budgets=Budgets(max_steps=2, max_total_retries=2, max_tokens_per_request=64),
                       sleep=lambda _delay: None)
    assert [e.data["error"] for e in result.events.of_kind("plan_parse_failed")] == [
        "reply contains both an Action and a Final Answer; exactly one step is allowed"]
    assert env.calls == [("act", json.loads(ACTION.split("Action Input: ", 1)[1]))]
    assert result.status == "finished" and result.answer == "done"


@pytest.mark.skipif(not getattr(sys, "get_int_max_str_digits", lambda: 0)(),
                    reason="this interpreter has no integer digit limit")
def test_an_integer_past_pythons_digit_limit_is_a_parse_failure_that_is_retried(g0_config):
    """json.loads raises a plain ValueError for an integer of more than 4,300
    digits; the loop retries it as any other malformed Action Input."""
    from runtime.model_adapters import ScriptedAdapter
    huge = "Thought: t\nAction: act\nAction Input: {\"command\": " + "1" * 5000 + "}"
    with pytest.raises(PlanParseError, match="Action Input is not valid JSON"):
        ReActPlanner.parse(huge)
    adapter = ScriptedAdapter([huge, react_final("done")])
    result = run_agent(config=g0_config, model=adapter, env=ScriptedEnv({"act": []}), task="t",
                       budgets=Budgets(max_steps=1, max_total_retries=1, max_tokens_per_request=64),
                       sleep=lambda _delay: None)
    assert len(result.events.of_kind("plan_parse_failed")) == 1
    assert result.status == "finished"


@pytest.mark.parametrize("value", ["NaN", "Infinity", "-Infinity", "1e999", '"\\ud800"'])
def test_an_action_input_that_parses_but_cannot_be_stored_is_a_parse_failure(value):
    """NaN, an overflowing literal and a lone surrogate escape load in Python
    but cannot be written to the trajectory as strict UTF-8 JSON."""
    with pytest.raises(PlanParseError, match="Action Input is not valid JSON"):
        ReActPlanner.parse(f'Thought: t\nAction: act\nAction Input: {{"command": {value}}}')


@pytest.mark.parametrize("value", ["NaN", "1e999", '"\\ud800"'])
def test_a_recorded_run_retries_an_action_input_that_cannot_be_stored(g0_config, tmp_path, value):
    """Through the loop with a recorder attached: the reply is retried with the
    existing message instead of stopping the episode at the recorder."""
    from runtime.model_adapters import ScriptedAdapter
    from trajectory.recorder.context import TrajectoryContext
    from trajectory.recorder.recorder import TrajectoryRecorder
    recorder = TrajectoryRecorder(TrajectoryContext(genome_id=g0_config.genome_id),
                                  tmp_path / "t.jsonl", trajectory_id="T-0123456789ab")
    adapter = ScriptedAdapter([f'Thought: t\nAction: act\nAction Input: {{"command": {value}}}',
                               react_final("done")])
    result = run_agent(config=g0_config, model=adapter, env=ScriptedEnv({"act": []}), task="t",
                       budgets=Budgets(max_steps=1, max_total_retries=1, max_tokens_per_request=64),
                       hooks=(recorder,), sleep=lambda _delay: None)
    assert len(result.events.of_kind("plan_parse_failed")) == 1
    assert result.status == "finished"
    assert recorder.finalize().document["events"]


def nested_action(depth):
    """A reply whose Action Input nests ``depth`` containers deep."""
    inner = "[" * (depth - 1) + '"x"' + "]" * (depth - 1)
    return f'Thought: t\nAction: act\nAction Input: {{"command": {inner}}}'


def test_an_action_input_nested_up_to_the_bound_parses():
    from runtime.model_adapters.base import MAX_JSON_DEPTH
    decision = ReActPlanner.parse(nested_action(MAX_JSON_DEPTH))
    assert decision.action.name == "act"


@pytest.mark.parametrize("extra", [1, 1000, 100000])
def test_an_action_input_nested_past_the_bound_is_a_parse_failure(extra):
    """At 100,000 levels json.loads itself raises RecursionError; at about
    1,000 the recorder's deep copy did. Both are parse failures."""
    from runtime.model_adapters.base import MAX_JSON_DEPTH
    with pytest.raises(PlanParseError, match="Action Input nests deeper than"):
        ReActPlanner.parse(nested_action(MAX_JSON_DEPTH + extra))


@pytest.mark.parametrize("depth", [1000, 100000])
def test_a_recorded_run_retries_an_action_input_nested_too_deep(g0_config, tmp_path, depth):
    from runtime.model_adapters import ScriptedAdapter
    from trajectory.recorder.context import TrajectoryContext
    from trajectory.recorder.recorder import TrajectoryRecorder
    recorder = TrajectoryRecorder(TrajectoryContext(genome_id=g0_config.genome_id),
                                  tmp_path / "t.jsonl", trajectory_id="T-0123456789ab")
    adapter = ScriptedAdapter([nested_action(depth), react_final("done")])
    result = run_agent(config=g0_config, model=adapter, env=ScriptedEnv({"act": []}), task="t",
                       budgets=Budgets(max_steps=1, max_total_retries=1, max_tokens_per_request=64),
                       hooks=(recorder,), sleep=lambda _delay: None)
    assert len(result.events.of_kind("plan_parse_failed")) == 1
    assert result.status == "finished"
    assert recorder.finalize().document["events"]


# sha256 of the messages of every model call in the episode below, at the code
# before unstorable Action Inputs were retried
MESSAGES_SHA256_OF_A_RETRIED_EPISODE = "4ceff25c247814de2c22b071dd606666129036814fa4f110b0a9d0e5f28bcd9d"


def test_the_messages_of_a_retried_episode_are_byte_for_byte_unchanged(g0_config):
    """Retrying an unstorable Action Input changes nothing the model sees: the
    prompt and the retry message of this scripted episode keep their bytes."""
    import hashlib

    from runtime.model_adapters import ScriptedAdapter
    adapter = ScriptedAdapter(["Thought: t\nAction: act\nAction Input: not json", ACTION,
                               react_final("done")])
    env = ScriptedEnv({"act": [ToolObservation(ok=True, content="You put the egg 1 in the microwave 1.")]})
    result = run_agent(config=g0_config, model=adapter, env=env, task="Put an egg in the microwave.",
                       budgets=Budgets(max_steps=2, max_total_retries=2, max_tokens_per_request=64),
                       sleep=lambda _delay: None)
    assert result.status == "finished"
    messages = [e.data["messages"] for e in result.events.of_kind("model_called")]
    assert len(messages) == 3
    blob = json.dumps(messages, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    assert hashlib.sha256(blob.encode("utf-8")).hexdigest() == MESSAGES_SHA256_OF_A_RETRIED_EPISODE
