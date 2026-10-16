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


def test_task_text_carries_the_initial_admissible_commands():
    """ADR-0003 A1: D2 applies to the initial observation too, so the model
    sees the valid commands before its first action."""
    env = AlfworldEnvironment(FakeSession(SCRIPT), game_id=GAME_ID)
    task = env.start()
    assert task == render_observation(task_text(INITIAL), ["look", "go to fridge 1"])
    assert task.endswith(f"{ADMISSIBLE_HEADER}\ngo to fridge 1\nlook")
    assert "Your task is to: put a hot egg in fridge." in task
    assert BANNER not in task and GAME_ID not in task


class _NoAdmissible(FakeSession):
    """Drops admissible_commands (absent key, or None) at reset or at steps."""

    def __init__(self, script, *, at, missing="absent", from_step=1):
        super().__init__(script)
        self.at, self.missing, self.from_step = at, missing, from_step

    def _strip(self, infos):
        if self.missing == "none":
            return {**infos, "admissible_commands": None}
        return {k: v for k, v in infos.items() if k != "admissible_commands"}

    def reset(self):
        obs, infos = super().reset()
        return obs, (self._strip(infos) if self.at == "reset" else infos)

    def step(self, command):
        obs, score, done, infos = super().step(command)
        strip = self.at == "step" and len(self.commands) >= self.from_step
        return obs, score, done, (self._strip(infos) if strip else infos)


@pytest.mark.parametrize("missing", ["absent", "none"])
def test_missing_admissible_commands_at_reset_fail_closed(missing):
    """Review of #48: a game opened without admissible commands would show
    the model an empty command list; fail instead (D2), and stay unstarted."""
    env = AlfworldEnvironment(_NoAdmissible(SCRIPT, at="reset", missing=missing), game_id=GAME_ID)
    with pytest.raises(AlfworldAdapterError, match="admissible_commands"):
        env.start()
    with pytest.raises(AlfworldAdapterError, match="before start"):
        env.call("act", {"command": "look"})


@pytest.mark.parametrize("missing", ["absent", "none"])
@pytest.mark.parametrize("command", [
    "go to fridge 1",              # ordinary step
    "put egg 1 in/on fridge 1",    # the winning, terminal step
    "fly to the moon",             # inadmissible: "Nothing happens."
])
def test_missing_admissible_commands_at_a_step_change_no_state(missing, command):
    """Review of #49: the check runs at every step, the terminal one
    included, before any counter or flag changes."""
    env = AlfworldEnvironment(_NoAdmissible(SCRIPT, at="step", missing=missing), game_id=GAME_ID)
    env.start()
    before = env.record()
    with pytest.raises(AlfworldAdapterError, match="admissible_commands"):
        env.call("act", {"command": command})
    assert env.record() == before


@pytest.mark.parametrize("from_step", [2, 3, 5])
@pytest.mark.parametrize("command", ["put egg 1 in/on fridge 1", "fly to the moon"])
def test_missing_admissible_commands_at_a_later_step_change_no_state(command, from_step):
    """Review of #50/#51: the check runs at every step, not only the first
    ones. Good steps change the record (positive control); the first
    failing step leaves it as it was."""
    env = AlfworldEnvironment(_NoAdmissible(SCRIPT, at="step", from_step=from_step),
                              game_id=GAME_ID)
    env.start()
    for n in range(1, from_step):
        assert env.call("act", {"command": "go to fridge 1"}).ok
        assert env.record().steps == n
    before = env.record()
    with pytest.raises(AlfworldAdapterError, match="admissible_commands"):
        env.call("act", {"command": command})
    assert env.record() == before


def test_the_admissible_header_is_the_frozen_literal():
    """D2 / A1 are part of the frozen G0 prompt: pin the exact text."""
    assert ADMISSIBLE_HEADER == "Admissible commands:"
    assert render_observation("You see a fridge 1.", ["b", "a"]) == (
        "You see a fridge 1.\n\nAdmissible commands:\na\nb")
    assert render_observation("Done.", []) == "Done.\n\nAdmissible commands:\n"


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


class _NoWon(FakeSession):
    """Infos without a boolean ``won`` at reset or at a step."""

    def __init__(self, script, *, at, value=None):
        super().__init__(script)
        self.at, self.value = at, value

    def _strip(self, infos):
        infos = dict(infos)
        if self.value is None:
            infos.pop("won")
        else:
            infos["won"] = self.value
        return infos

    def reset(self):
        obs, infos = super().reset()
        return obs, (self._strip(infos) if self.at == "reset" else infos)

    def step(self, command):
        obs, score, done, infos = super().step(command)
        return obs, score, done, (self._strip(infos) if self.at == "step" else infos)


@pytest.mark.parametrize("value", [None, 1, "true"])
def test_a_game_without_a_boolean_won_fails_closed(value):
    """Critic review of ADR-0003 (2026-10-15): won is the only success
    signal (D6); a game opened without it fails, rather than scoring every
    episode a failure, and the check changes no state."""
    env = AlfworldEnvironment(_NoWon(SCRIPT, at="reset", value=value), game_id=GAME_ID)
    with pytest.raises(AlfworldAdapterError, match="won"):
        env.start()
    env = AlfworldEnvironment(_NoWon(SCRIPT, at="step", value=value), game_id=GAME_ID)
    env.start()
    before = env.record()
    with pytest.raises(AlfworldAdapterError, match="won"):
        env.call("act", {"command": "put egg 1 in/on fridge 1"})
    assert env.record() == before


# -- TextWorld 1.7.0 on Python 3.13 (PEP 667) ---------------------------------------------
TEXTGEN_SOURCE = '''
def shout(text):
    return text.upper()


class TerminalSymbol:
    def __init__(self, value):
        self.value = value


class EvalSymbol:
    def __init__(self, expression, context=None):
        self.expression = expression
        self.context = context or {}

    def derive(self, context=None):  # TextWorld 1.7.0's, verbatim
        context = context or self.context
        locals().update(context["variables"])
        value = eval(self.expression)
        return [TerminalSymbol(value)]
'''


class _Thing:
    def __init__(self, name, indefinite="a", id=""):
        self.name, self.indefinite, self.id = name, indefinite, id


def _textgen():
    import types
    module = types.ModuleType("textgen_stand_in")
    exec(compile(TEXTGEN_SOURCE, "textgen_stand_in", "exec"), module.__dict__)
    return module


# the grammar's expressions in ALFWorld 0.4.2 (alfred.twl2 and the games' own), and the
# names the original resolved besides them: a module global, its own locals, and `value`,
# which it never saw (still unbound when eval refreshed the locals dict)
EXPRESSIONS = ["r.name", "o.indefinite + ' ' + o.name", "r.name or r.id", "{'k': r}['k'].name",
               "shout(o.name)", "self.expression", "len(context['variables'])", "value"]
VARIABLES = {"r": _Thing("", id="drawer 1"), "o": _Thing("apple 1", "an"), "value": "hidden"}


def _derive(module, expression, variables):
    try:
        return module.EvalSymbol(expression).derive({"variables": variables})[0].value
    except NameError:
        return NameError


def test_the_textworld_eval_fix_gives_the_names_the_original_gave():
    import sys
    from runtime.tool_adapters.alfworld import patch_textworld_eval
    patched = _textgen()
    patch_textworld_eval(patched, "1.7.0")
    expected = ["", "an apple 1", "drawer 1", "", "APPLE 1", "self.expression", 3, NameError]
    assert [_derive(patched, e, VARIABLES) for e in EXPRESSIONS] == expected
    original = _textgen()
    if sys.version_info < (3, 13):  # the original still works here: the same answers
        assert [_derive(original, e, VARIABLES) for e in EXPRESSIONS] == expected
    else:  # PEP 667: its locals().update no longer reaches eval (the Colab failure)
        assert _derive(original, "r.name", VARIABLES) is NameError


def test_the_textworld_eval_fix_refuses_another_version_and_applies_once():
    from runtime.tool_adapters.alfworld import AlfworldAdapterError, patch_textworld_eval
    module = _textgen()
    unpatched = module.EvalSymbol.derive
    with pytest.raises(AlfworldAdapterError, match="1.8.0"):
        patch_textworld_eval(module, "1.8.0")
    assert module.EvalSymbol.derive is unpatched
    patch_textworld_eval(module, "1.7.0")
    fixed = module.EvalSymbol.derive
    patch_textworld_eval(module, "1.7.0")
    assert module.EvalSymbol.derive is fixed is not unpatched
