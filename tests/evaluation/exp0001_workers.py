"""Picklable scripted components for EXP-0001 tests that run episodes in
worker processes (the spawn start method pickles the model and the game
factory, so they are module-level classes). Not a test module."""

from evaluation.exp0001.backends import ClosingEnvironment
from runtime.model_adapters import ScriptedAdapter
from runtime.model_adapters.base import GenerationResult
from runtime.tool_adapters.alfworld import AlfworldEnvironment
from tests.evaluation.test_exp0001_lifetime import Game, SeededModel
from tests.runtime.conftest import react_action, react_final


class GameFactory:
    """``env_factory(split, game_id)`` over scripted games."""

    def __call__(self, split, game_id):
        return AlfworldEnvironment(Game(game_id), game_id=game_id)


class ClosingGameFactory:
    def __call__(self, split, game_id):
        return ClosingEnvironment(Game(game_id), game_id)


class FailOnTask(SeededModel):
    """SeededModel that raises on every call of one game's episodes."""

    def __init__(self, wins=(), fail_task=None):
        super().__init__(wins=wins)
        self.fail_task = fail_task

    def generate(self, messages, settings):
        if self.fail_task is not None and any(self.fail_task in m.content for m in messages):
            raise RuntimeError(f"backend died on {self.fail_task}")
        return super().generate(messages, settings)


class WinsOnGames(ScriptedAdapter):
    """At any temperature: wins the games named in ``wins``, else gives a
    final answer without acting (a failure)."""

    def __init__(self, wins=()):
        super().__init__()
        self.wins = tuple(wins)

    def generate(self, messages, settings):
        text = "\n".join(m.content for m in messages)
        won = any(f"solve {game_id}." in text for game_id in self.wins)
        return GenerationResult(text=react_action("act", {"command": "win"}) if won
                                else react_final("I am done."), finish_reason="stop")


class AlsoWinsOnOddSeeds(WinsOnGames):
    """As WinsOnGames, and at a temperature above 0 it also wins every game
    whose sampling seed is odd: the temperature-0.7 attempts form pairs."""

    def generate(self, messages, settings):
        if settings.temperature > 0 and settings.seed is not None and settings.seed % 2:
            return GenerationResult(text=react_action("act", {"command": "win"}),
                                    finish_reason="stop")
        return super().generate(messages, settings)


class WinsOnGamesButFailsOn(WinsOnGames):
    """As WinsOnGames, but the backend dies on every call of one game."""

    def __init__(self, wins=(), fail_task=None):
        super().__init__(wins)
        self.fail_task = fail_task

    def generate(self, messages, settings):
        if any(f"solve {self.fail_task}." in m.content for m in messages):
            raise RuntimeError(f"backend died on {self.fail_task}")
        return super().generate(messages, settings)

