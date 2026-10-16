"""The context-window rule (ADR-0009 D4) and A2's backstop (ADR-0003):
a prompt that does not fit ends the episode as budget_exhausted
(context_window), never truncated or retried; a reply that runs on past
its Action Input into an invented observation fails closed and is
reprompted."""

import json

from runtime.loop import Budgets, ToolObservation, run_agent
from runtime.model_adapters import ScriptedAdapter
from runtime.model_adapters.errors import ContextWindowExceeded
from tests.runtime.conftest import ScriptedEnv, react_action, react_final

BUDGETS = Budgets(max_steps=5, max_total_retries=3, max_tokens_per_request=64)
LOOK = react_action("act", {"command": "look"})


class Overflows(ScriptedAdapter):
    """Answers LOOK until call ``at``, where the prompt no longer fits."""

    def __init__(self, at):
        super().__init__()
        self.at, self.calls = at, 0

    def generate(self, messages, settings):
        self.calls += 1
        if self.calls == self.at:
            raise ContextWindowExceeded("This model's maximum context length is 32768 tokens.")
        from runtime.model_adapters.base import GenerationResult
        return GenerationResult(text=LOOK, finish_reason="stop")


def test_a_prompt_that_does_not_fit_ends_the_episode_as_a_failure(g0_config):
    model = Overflows(at=3)
    env = ScriptedEnv({"act": [ToolObservation(ok=True, content="You see a fridge.")] * 5})
    result = run_agent(config=g0_config, model=model, env=env, task="t", budgets=BUDGETS,
                       sleep=lambda _delay: None)
    assert result.status == "budget_exhausted" and result.answer is None
    exhausted = result.events.of_kind("budget_exhausted")
    assert [e.data["budget"] for e in exhausted] == ["context_window"]
    assert exhausted[0].step == 3 and model.calls == 3  # never retried
    assert len(result.events.of_kind("model_called")) == 2
    assert not result.events.of_kind("retry_scheduled")


def test_a_reply_that_invents_an_observation_is_reprompted(g0_config):
    runaway = (LOOK + "\nObservation: You arrive at fridge 1.\nThought: open it.\n"
               + react_action("act", {"command": "open fridge 1"}).split("\n", 1)[1])
    model = ScriptedAdapter([runaway, LOOK, react_final("done")])
    env = ScriptedEnv({"act": [ToolObservation(ok=True, content="You see a fridge.")]})
    result = run_agent(config=g0_config, model=model, env=env, task="t", budgets=BUDGETS,
                       sleep=lambda _delay: None)
    assert len(result.events.of_kind("plan_parse_failed")) == 1
    assert env.calls == [("act", json.loads(LOOK.split("Action Input: ", 1)[1]))]
    assert result.status == "finished"
