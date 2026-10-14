"""ReAct reply parsing (module audit, 2026-10-13). A reply is exactly one
step: an Action with its Action Input, or a non-empty Final Answer. The
"Final Answer:" marker counts only at the start of a line, and a reply that
both acts and finishes fails closed, so the retry policy reprompts instead
of ending the run without sending the action."""

import json

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
