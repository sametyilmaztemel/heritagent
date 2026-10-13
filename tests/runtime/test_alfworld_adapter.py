"""ALFWorld ToolEnvironment adapter (ADR-0003) against a scripted fake of
TextWorld's single-game gym API; no ALFWorld data or install needed."""

import os

import pytest

from runtime.loop import Budgets, run_agent
from runtime.model_adapters import ScriptedAdapter
from runtime.tool_adapters.alfworld import (
    ACT_TOOL,
    ADMISSIBLE_HEADER,
    BANNER,
    AlfworldAdapterError,
    AlfworldEnvironment,
    render_observation,
    task_text,
)
from tests.runtime.conftest import react_action, react_final

GAME_ID = "pick_heat_then_place_in_recep-Egg-None-Fridge-13/trial_T0001"
INITIAL = (f"{BANNER}\n\nYou are in the middle of a room. Looking quickly around you, "
           "you see a fridge 1 and a microwave 1.\n\nYour task is to: put a hot egg in fridge.")


class FakeSession:
    """reset() -> (obs, infos); step(cmd) -> (obs, score, done, infos)."""

    def __init__(self, script: dict[str, tuple[str, bool, bool]]):
        self.script = script
        self.commands = []

    def reset(self):
        return INITIAL, {"won": False, "admissible_commands": ["look", "go to fridge 1"],
                         "extra.gamefile": "/data/" + GAME_ID}

    def step(self, command):
        self.commands.append(command)
        text, done, won = self.script.get(command, ("Nothing happens.", False, False))
        admissible = [] if done else ["look", "go to microwave 1", "go to fridge 1"]
        return text, (1.0 if won else 0.0), done, {
            "won": won, "admissible_commands": admissible, "extra.gamefile": "/data/" + GAME_ID}


SCRIPT = {
    "go to fridge 1": ("You arrive at fridge 1.", False, False),
    "put egg 1 in/on fridge 1": ("You put the egg 1 in/on the fridge 1.", True, True),
}


def test_task_text_strips_only_the_banner():
    assert task_text(INITIAL) == INITIAL[len(BANNER):].strip()
    assert task_text("no banner here") == "no banner here"


def test_observation_lists_sorted_admissible_commands():
    rendered = render_observation("You arrive at fridge 1.  ", ["look", "go to fridge 1"])
    assert rendered == f"You arrive at fridge 1.\n\n{ADMISSIBLE_HEADER}\ngo to fridge 1\nlook"


def test_single_act_tool():
    env = AlfworldEnvironment(FakeSession(SCRIPT), game_id=GAME_ID)
    assert env.list_tools() == [ACT_TOOL]
    assert ACT_TOOL.parameters_schema["required"] == ["command"]


def test_full_episode_through_the_runtime(g0_config):
    session = FakeSession(SCRIPT)
    env = AlfworldEnvironment(session, game_id=GAME_ID)
    adapter = ScriptedAdapter([
        react_action("act", {"command": "fly to the moon"}),
        react_action("act", {"command": "go to fridge 1"}),
        react_action("act", {"command": "put egg 1 in/on fridge 1"}),
        react_final("never reached"),
    ])
    sleeps = []
    result = run_agent(config=g0_config, model=adapter, env=env, task=env.start(),
                       budgets=Budgets(max_steps=8, max_total_retries=10, max_tokens_per_request=64),
                       sleep=sleeps.append)
    record = env.record()
    assert result.status == "environment_terminal"
    assert record.won is True and record.done is True
    assert record.steps == 3 and record.invalid_commands == 1
    assert session.commands == ["fly to the moon", "go to fridge 1", "put egg 1 in/on fridge 1"]
    # the inadmissible command was not retried (no backoff sleeps)
    assert sleeps == [] and not result.events.of_kind("retry_scheduled")
    assert result.counters.failing_tools == 1
    # evaluation-only metadata never reaches the model
    seen = "\n".join(m.content for r in adapter.requests for m in r.messages)
    assert GAME_ID not in seen and "pick_heat_then_place_in_recep" not in seen
    assert "Your task is to: put a hot egg in fridge." in seen
    assert ADMISSIBLE_HEADER in seen


def test_unfinished_episode_is_not_a_success(g0_config):
    env = AlfworldEnvironment(FakeSession(SCRIPT), game_id=GAME_ID)
    adapter = ScriptedAdapter([react_action("act", {"command": "go to fridge 1"}),
                               react_final("I think I am done.")])
    result = run_agent(config=g0_config, model=adapter, env=env, task=env.start(),
                       budgets=Budgets(max_steps=4, max_total_retries=10, max_tokens_per_request=64))
    assert result.status == "finished"          # the model claimed completion...
    assert env.record().won is False            # ...but the game decides success


def test_calls_after_the_end_are_terminal_and_not_forwarded():
    session = FakeSession(SCRIPT)
    env = AlfworldEnvironment(session, game_id=GAME_ID)
    env.start()
    env.call("act", {"command": "put egg 1 in/on fridge 1"})
    after = env.call("act", {"command": "look"})
    assert after.terminal and not after.ok and not after.retryable
    assert session.commands == ["put egg 1 in/on fridge 1"]


def test_order_is_enforced():
    env = AlfworldEnvironment(FakeSession(SCRIPT), game_id=GAME_ID)
    with pytest.raises(AlfworldAdapterError):
        env.call("act", {"command": "look"})
    with pytest.raises(AlfworldAdapterError):
        env.record()
    env.start()
    with pytest.raises(AlfworldAdapterError):
        env.start()
    unknown = env.call("teleport", {})
    assert not unknown.ok and not unknown.retryable


@pytest.mark.skipif(not os.environ.get("ALFWORLD_GAMEFILE"),
                    reason="needs the alfworld extra and ALFWORLD_GAMEFILE pointing at a game.tw-pddl")
def test_real_game_opens_and_steps():
    from runtime.tool_adapters.alfworld import open_game
    env = AlfworldEnvironment(open_game(os.environ["ALFWORLD_GAMEFILE"], max_episode_steps=50),
                              game_id="real")
    assert "Your task is to:" in env.start()
    observation = env.call("act", {"command": "look"})
    assert ADMISSIBLE_HEADER in observation.content


def test_a_game_that_ends_without_winning_is_a_failure(g0_config):
    """done (e.g. TextWorld's step limit) is not success: only `won` is."""
    session = FakeSession({"go to fridge 1": ("You arrive at fridge 1.", True, False)})
    env = AlfworldEnvironment(session, game_id=GAME_ID)
    adapter = ScriptedAdapter([react_action("act", {"command": "go to fridge 1"})])
    result = run_agent(config=g0_config, model=adapter, env=env, task=env.start(),
                       budgets=Budgets(max_steps=4, max_total_retries=10, max_tokens_per_request=64))
    record = env.record()
    assert result.status == "environment_terminal"
    assert record.done is True and record.won is False
