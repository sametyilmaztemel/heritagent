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

On Python 3.13, TextWorld 1.7.0 cannot generate a game's text: its
``EvalSymbol.derive`` relies on ``locals().update(...)`` reaching ``eval``,
which PEP 667 ended. ``open_game`` first replaces that one method with the
same evaluation over explicit namespaces (``patch_textworld_eval``), on every
Python version, so all of them run one code path.

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


def _won(infos: dict) -> bool:
    """ALFWorld's ``won``, the only success signal (D6); fail closed when the
    game was opened without it, rather than scoring every episode a failure."""
    if "won" not in infos or not isinstance(infos["won"], bool):
        raise AlfworldAdapterError(
            "game infos carry no boolean 'won'; open the game with won requested "
            "(ADR-0003 D6)")
    return infos["won"]


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
        won = _won(infos)
        self._started = True
        self._won = won
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
        won = _won(infos)  # both checked before any counter changes
        self._steps += 1
        self._done = bool(done)
        self._won = won
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


TEXTWORLD_VERSION = "1.7.0"  # the alfworld extra's pin (ADR-0003)


def eval_namespace(variables: dict, symbol, context) -> dict:
    """The local names TextWorld 1.7.0's ``EvalSymbol.derive`` evaluated its
    expression with up to Python 3.12: the grammar's variables, under the
    method's bound locals ``self`` and ``context``, without its still-unbound
    ``value`` (``eval`` refreshed the updated ``locals()`` dict from the frame
    first)."""
    namespace = dict(variables)
    namespace.pop("value", None)
    namespace.update(self=symbol, context=context)
    return namespace


def patch_textworld_eval(textgen, version: str) -> None:
    """Make TextWorld's ``EvalSymbol.derive`` (``{r.name}`` in a grammar)
    work on Python 3.13, where ``locals()`` in a function is a snapshot
    (PEP 667): the original's ``locals().update(variables)`` no longer reaches
    its ``eval``, and every game fails with ``NameError`` on its first
    description. The replacement evaluates the expression in the module's
    globals and ``eval_namespace``, the names the original saw. It is written
    against TextWorld 1.7.0 and refuses any other version; applying it twice
    changes nothing."""
    if version != TEXTWORLD_VERSION:
        raise AlfworldAdapterError(
            f"textworld {version} is installed; the EvalSymbol fix is written for "
            f"{TEXTWORLD_VERSION}, the alfworld extra's pin")
    symbol_class = textgen.EvalSymbol
    if getattr(symbol_class.derive, "heritagent_eval_fix", False):
        return
    module_globals = vars(textgen)

    def derive(self, context=None):
        context = context or self.context
        value = eval(self.expression, module_globals,
                     eval_namespace(context["variables"], self, context))
        return [textgen.TerminalSymbol(value)]

    derive.heritagent_eval_fix = True
    symbol_class.derive = derive


def open_game(gamefile: str, *, max_episode_steps: int) -> GameSession:
    """Open one real ALFWorld game (D8). Requires ``pip install .[alfworld]``
    and the ALFWorld data; not exercised by the offline test suite."""
    import textworld
    import textworld.gym
    from alfworld.agents.environment.alfred_tw_env import AlfredDemangler, AlfredInfos
    from textworld.envs.pddl import textgen

    patch_textworld_eval(textgen, textworld.__version__)

    request_infos = textworld.EnvInfos(won=True, admissible_commands=True, extras=["gamefile"])
    env_id = textworld.gym.register_game(
        gamefile, request_infos, max_episode_steps=max_episode_steps,
        wrappers=[AlfredDemangler(shuffle=False), AlfredInfos])
    return textworld.gym.make(env_id)


def planner_environments(session) -> list:
    """The TextWorld PDDL environments under an ``open_game`` session (its
    batch environment, their wrappers): each holds its own copy of
    fast-downward's library. Collect them before the session is closed,
    which drops the batch environment."""
    found, seen, stack = [], set(), [session]
    while stack:
        node = stack.pop()
        if node is None or id(node) in seen:
            continue
        seen.add(id(node))
        try:
            own = vars(node)
        except TypeError:  # no instance dictionary: holds nothing of its own
            continue
        if own.get("downward_lib") is not None:  # its own, not a wrapper's forwarded one
            found.append(node)
        stack.extend(own.get("envs") or ())
        stack.extend(own.get(name) for name in ("batch_env", "_wrapped_env"))
    return found


def release_planner(environments) -> int:
    """Unload each environment's copy of fast-downward's library, after its
    game is closed. TextWorld 1.7.0 loads a fresh 32 MB copy for every game
    it opens, from a temporary file it deletes at once, and never unloads
    it: until the process ends, each opened game keeps that much disk (the
    deleted file) and address space. Returns how many copies were unloaded."""
    released = 0
    for env in environments:
        library = vars(env).get("downward_lib")
        if library is None:
            continue
        import fast_downward
        env.downward_lib = None
        fast_downward.close_lib(library)
        released += 1
    return released
