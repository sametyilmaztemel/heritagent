"""ALFWorld ``ToolEnvironment`` adapter (ADR-0003 D1-D3, D6-D8, amendment A1).

One ALFWorld game per episode behind the runtime's provider-neutral tool
boundary:

- D1: a single tool, ``act``, taking one free-text ``command``;
- D2: every observation, the initial one included (A1), is ALFWorld's text
  followed by ``Admissible commands:`` and the sorted admissible commands;
- D3: "Nothing happens." (an inadmissible command) is ``ok=False`` and
  ``retryable=False``, so the loop counts it as a failure without retrying;
- D5/D6: when the game reports done, the observation is ``terminal`` and the
  episode record carries ALFWorld's ``won`` flag, which is the benchmark
  success. The model's text and the runtime status never decide it;
- D7 (as amended by A1): the task text is the initial observation minus the
  TextWorld banner, rendered with the initial admissible commands as in D2,
  so the model sees the valid commands before its first action;
- D8: real games are opened with alfworld's own wrappers,
  ``AlfredDemangler(shuffle=False)`` and ``AlfredInfos``, without expert
  plans (``open_game``; needs the optional ``alfworld`` extra).

The adapter talks to a ``GameSession``: TextWorld's single-game gym API,
``reset() -> (observation, infos)`` and
``step(command) -> (observation, score, done, infos)``, so tests can script
it. The game id and family are evaluation-only and never appear in a
model-visible string.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from runtime.loop import ToolObservation
from runtime.model_adapters import ToolSpec

BANNER = "-= Welcome to TextWorld, ALFRED! =-"
NOTHING_HAPPENS = "Nothing happens."
ADMISSIBLE_HEADER = "Admissible commands:"
ACT_TOOL = ToolSpec(
    name="act",
    description=("Send one text command to the household game, for example "
                 "'go to fridge 1' or 'take apple 1 from countertop 1'."),
    parameters_schema={
        "type": "object",
        "properties": {"command": {"type": "string", "minLength": 1}},
        "required": ["command"],
        "additionalProperties": False,
    },
)


class AlfworldAdapterError(RuntimeError):
    """The adapter was used out of order or the game returned malformed data."""


class GameSession(Protocol):
    def reset(self) -> tuple[str, dict]: ...

    def step(self, command: str) -> tuple[str, float, bool, dict]: ...


@dataclass(frozen=True)
class EpisodeRecord:
    """What the episode runner needs; evaluation-only, never model-visible."""

    game_id: str
    won: bool
    done: bool
    steps: int
    invalid_commands: int


def task_text(initial_observation: str) -> str:
    """D7: the initial observation without the TextWorld banner."""
    text = initial_observation.strip()
    if text.startswith(BANNER):
        text = text[len(BANNER):]
    return text.strip()


def render_observation(text: str, admissible_commands) -> str:
    """D2: the observation, then the sorted admissible commands."""
    commands = sorted(str(c) for c in (admissible_commands or ()))
    return f"{text.strip()}\n\n{ADMISSIBLE_HEADER}\n" + "\n".join(commands)


def _admissible(infos: dict) -> list:
    """The game's admissible commands; fail closed when the game was opened
    without them, since D2 renders them into every observation."""
    if "admissible_commands" not in infos or infos["admissible_commands"] is None:
        raise AlfworldAdapterError(
            "game infos carry no admissible_commands; open the game with "
            "admissible_commands requested (ADR-0003 D2)")
    return infos["admissible_commands"]


def is_inadmissible(observation: str) -> bool:
    return observation.strip() == NOTHING_HAPPENS


class AlfworldEnvironment:
    """``ToolEnvironment`` for one ALFWorld game (one episode)."""

    def __init__(self, session: GameSession, *, game_id: str):
        self._session = session
        self._game_id = game_id
        self._started = False
        self._done = False
        self._won = False
        self._steps = 0
        self._invalid = 0

    def start(self) -> str:
        """Reset the game and return the task text for ``run_agent``: the
        initial observation without the banner, with the initial admissible
        commands (D7 as amended by A1)."""
        if self._started:
            raise AlfworldAdapterError("episode already started; use a new environment")
        observation, infos = self._session.reset()
        task = render_observation(task_text(observation), _admissible(infos))
        self._started = True
        self._won = bool(infos.get("won", False))
        return task

    def list_tools(self) -> list[ToolSpec]:
        return [ACT_TOOL]

    def call(self, name: str, arguments: dict) -> ToolObservation:
        if not self._started:
            raise AlfworldAdapterError("call() before start()")
        if name != ACT_TOOL.name:
            return ToolObservation(ok=False, content=f"unknown tool {name!r}", retryable=False)
        if self._done:
            return ToolObservation(ok=False, content="The episode has ended.",
                                   terminal=True, retryable=False)
        observation, _score, done, infos = self._session.step(arguments["command"])
        commands = _admissible(infos)
        self._steps += 1
        self._done = bool(done)
        self._won = bool(infos.get("won", False))
        invalid = is_inadmissible(observation)
        if invalid:
            self._invalid += 1
        return ToolObservation(
            ok=not invalid,
            content=render_observation(observation, commands),
            terminal=self._done,
            retryable=not invalid,
        )

    def record(self) -> EpisodeRecord:
        if not self._started:
            raise AlfworldAdapterError("record() before start()")
        return EpisodeRecord(game_id=self._game_id, won=self._won, done=self._done,
                             steps=self._steps, invalid_commands=self._invalid)


def open_game(gamefile: str, *, max_episode_steps: int) -> GameSession:
    """Open one real ALFWorld game (D8). Requires ``pip install .[alfworld]``
    and the ALFWorld data; not exercised by the offline test suite."""
    import textworld
    import textworld.gym
    from alfworld.agents.environment.alfred_tw_env import AlfredDemangler, AlfredInfos

    request_infos = textworld.EnvInfos(won=True, admissible_commands=True, extras=["gamefile"])
    env_id = textworld.gym.register_game(
        gamefile, request_infos, max_episode_steps=max_episode_steps,
        wrappers=[AlfredDemangler(shuffle=False), AlfredInfos])
    return textworld.gym.make(env_id)
