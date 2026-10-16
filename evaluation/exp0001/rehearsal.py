"""A rehearsal of the calibration pilot without a GPU or a model (issue #13
Phase B; ADR-0008): ``cli rehearse``.

Every minute of the A100 runtime costs compute units, so the pilot's path
runs first where it costs almost nothing, on a CPU runtime: the pinned
software on Colab's own Python, the ALFWorld data and games, the worker
processes, the ledger and trajectories, the model's own tokenizer (fetched
without the weights), the Drive mirror and a restore from it, and the
report. Only the model server is a stand-in (``StandInServer``): an HTTP
server in this process that answers ``/version``, ``/v1/models`` and
``/v1/chat/completions`` as ``vllm serve`` does. It renders and counts each
request with the model's own chat template (a template that refuses the
request is a 400, as on the server), compiles a response-format schema with
xgrammar and the model's tokenizer when xgrammar is installed, refuses a
request that breaks the pilot's protocol (``protocol_issues``) or lacks the
protocol's chat-template arguments, and plays ALFWorld's handcoded expert on
the episodes it is set to win.

A rehearsal measures nothing about any model. Its ledger is its own run
(``REHEARSAL_RUN_ID``), its report is not a pilot report, and the command
refuses a directory or a mirror that already holds anything.

The stand-in decides from the task text and the request's seed alone
(``stand_in_wins``), so the same episode gets the same replies, and
``expected_counts`` knows in advance what the report must say."""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from evaluation.exp0001.protocol import CONTEXT_WINDOW, LIFETIME_ATTEMPTS_PER_TASK, \
    MAX_TOKENS_PER_REQUEST, MINER_MAX_TOKENS, MINING_TEMPERATURE, attempt_sampling_seed
from runtime.model_adapters.openai_server import NEUTRAL_SAMPLING
from runtime.policies.planner_react import STOP_SEQUENCES
from runtime.tool_adapters.alfworld import ADMISSIBLE_HEADER

REHEARSAL_RUN_ID = "EXP-0001-pilot-rehearsal"
REHEARSAL_REPORT = "rehearsal-pilot-report.json"
REHEARSAL_REPORT_SCHEMA = "heritagent/exp0001-pilot-rehearsal-report/0.1"
REHEARSAL_PASSED = "rehearsal-passed.json"
REHEARSAL_PASSED_SCHEMA = "heritagent/exp0001-rehearsal-passed/0.1"
STAND_IN_VERSION = "stand-in (rehearsal)"
REHEARSAL_GPU = {"name": "none: a rehearsal on the CPU", "memory_mib": 0, "driver": None}
SHORT_LOSS_STEPS = 3  # a short loss walks this many steps, then gives up
INVALID_EVERY = 9  # a walk sends an inadmissible command every this many steps
_OBSERVATION = re.compile(r"^\[(\d+)\] ", re.MULTILINE)


def _draw(task: str, seed) -> int:
    return int.from_bytes(hashlib.sha256(f"{task}\x00{seed}".encode("utf-8")).digest()[:8], "big")


def stand_in_wins(task: str, seed) -> bool:
    """Whether the stand-in plays the expert's plan in this episode (for a
    game whose plan wins): half of the temperature-0 episodes, two thirds
    of the seeded attempts."""
    draw = _draw(task, seed)
    return draw % 2 == 0 if seed is None else draw % 3 != 0


def _long_loss(task: str, seed) -> bool:
    return (_draw(task, seed) // 3) % 2 == 0


def expert_commands(gamefile: str, *, max_steps: int) -> tuple[list[str], bool]:
    """The commands of ALFWorld's handcoded expert on one game, and whether
    they win it within ``max_steps``."""
    import textworld
    import textworld.gym
    from alfworld.agents.environment.alfred_tw_env import AlfredDemangler, AlfredExpert, \
        AlfredInfos
    from textworld.envs.pddl import textgen

    from runtime.tool_adapters.alfworld import patch_textworld_eval

    patch_textworld_eval(textgen, textworld.__version__)
    request = textworld.EnvInfos(won=True, admissible_commands=True,
                                 extras=["gamefile", "expert_plan"])
    env = textworld.gym.make(textworld.gym.register_game(
        gamefile, request, max_episode_steps=max_steps,
        wrappers=[AlfredDemangler(shuffle=False), AlfredInfos, AlfredExpert("handcoded")]))
    try:
        _, infos = env.reset()
        commands, won, done = [], False, False
        while not done and len(commands) < max_steps:
            plan = infos.get("extra.expert_plan") or []
            if not plan or plan[0] not in infos["admissible_commands"]:
                break
            _, _, done, infos = env.step(plan[0])
            commands.append(plan[0])
            won = bool(infos.get("won"))
        return commands, won
    finally:
        env.close()


def game_plan(env_factory, data_root: str, game_id: str, max_steps: int) -> dict:
    """One pilot game's task text as the episode shows it (through the run's
    own game factory) and its expert plan; run in a worker process. The
    handcoded expert draws from ``random`` unseeded, so it is seeded per game
    here (and the plan pool runs with a fixed hash seed): the same games
    give the same plans in every rehearsal."""
    import random
    random.seed(f"heritagent-rehearsal-expert|{game_id}")
    env = env_factory("train", game_id)
    try:
        task = env.start()
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()
    commands, won = expert_commands(f"{data_root}/train/{game_id}/game.tw-pddl",
                                    max_steps=max_steps)
    return {"game_id": game_id, "task": task, "commands": commands, "won": won}


def expected_counts(plans: Sequence[Mapping]) -> dict:
    """What the pilot report must say when the stand-in plays ``plans``
    (the pilot games in game_id order, PILOT_ATTEMPT_SEED 0)."""
    from evaluation.exp0001.pilot_run import PILOT_ATTEMPT_SEED
    t0 = t07 = pairs = solved = 0
    for plan in plans:
        winnable = plan["won"]
        t0 += winnable and stand_in_wins(plan["task"], None)
        wins = [winnable and stand_in_wins(plan["task"], attempt_sampling_seed(
                    PILOT_ATTEMPT_SEED, plan["game_id"], a))
                for a in range(LIFETIME_ATTEMPTS_PER_TASK)]
        t07 += sum(wins)
        solved += any(wins)
        pairs += any(wins) and not all(wins)
    return {"t0_successes": t0, "t07_successes": t07, "t07_pairs": pairs,
            "t07_solved_in_any_attempt": solved}


def protocol_issues(payload: Mapping) -> list[str]:
    """What in a chat request breaks the pilot's protocol: one choice, the
    neutral sampling values, and per call kind its temperature, seed, stop
    and token budget (a ReAct call: temperature 0 without a seed or the
    mining temperature with one, the stop sequence, at most 512 tokens; a
    teacher call: the mining temperature with a seed, 2,048 tokens). The
    server would serve such a request; the stand-in refuses it, so the
    rehearsal fails instead of the pilot measuring the wrong thing."""
    issues = []
    if payload.get("n") != 1:
        issues.append(f"n is {payload.get('n')!r}, not 1")
    sampling = {key: payload.get(key) for key in NEUTRAL_SAMPLING}
    if sampling != dict(NEUTRAL_SAMPLING):
        issues.append(f"sampling {sampling} is not the neutral {dict(NEUTRAL_SAMPLING)}")
    seed, temperature = payload.get("seed"), payload.get("temperature")
    if payload.get("response_format"):  # the teacher
        if temperature != MINING_TEMPERATURE or not isinstance(seed, int) or \
                payload.get("max_tokens") != MINER_MAX_TOKENS:
            issues.append(f"a teacher call at temperature {temperature!r}, seed {seed!r}, "
                          f"max_tokens {payload.get('max_tokens')!r}")
        unsupported = xgrammar_unsupported(
            payload["response_format"].get("json_schema", {}).get("schema"))
        if unsupported:
            issues.append(f"the schema has {unsupported}, which vLLM 0.29.0 does not give "
                          f"to xgrammar")
    else:  # a ReAct step
        if (seed is None and temperature != 0.0) or \
                (seed is not None and temperature != MINING_TEMPERATURE):
            issues.append(f"a ReAct call at temperature {temperature!r} with seed {seed!r}")
        if payload.get("stop") != list(STOP_SEQUENCES):
            issues.append(f"stop {payload.get('stop')!r} is not {list(STOP_SEQUENCES)!r}")
        if not isinstance(payload.get("max_tokens"), int) or \
                not 0 < payload["max_tokens"] <= MAX_TOKENS_PER_REQUEST:
            issues.append(f"max_tokens {payload.get('max_tokens')!r} is outside 1.."
                          f"{MAX_TOKENS_PER_REQUEST}")
    return issues


# vLLM 0.29.0's vllm/v1/structured_output/backend_xgrammar.py: the string formats xgrammar
# supports, and has_xgrammar_unsupported_json_features below. vLLM keeps such a schema from
# xgrammar before compiling it; with the default "auto" backend it tries guidance instead, so
# that request would run on another grammar backend than every other. xgrammar itself
# compiles such a schema without a word, so the stand-in applies vLLM's rule and refuses it.
XGRAMMAR_STRING_FORMATS = frozenset({
    "email", "date", "time", "date-time", "duration", "ipv4", "ipv6", "hostname", "uuid", "uri",
    "uri-reference", "uri-template", "json-pointer", "relative-json-pointer"})


def xgrammar_unsupported(schema) -> str | None:
    """The first part of ``schema`` that vLLM 0.29.0 does not give to
    xgrammar, or None (a copy of its ``has_xgrammar_unsupported_json_features``)."""
    if not isinstance(schema, dict):
        return None
    kind = schema.get("type")
    if kind in ("integer", "number") and "multipleOf" in schema:
        return f"multipleOf on a {kind}"
    if kind == "array":
        for key in ("uniqueItems", "contains", "minContains", "maxContains"):
            if key in schema:
                return f"{key} on an array"
    if kind == "string":
        if "format" in schema and schema["format"] not in XGRAMMAR_STRING_FORMATS:
            return f"the string format {schema['format']!r}"
        if ("pattern" in schema or "format" in schema) and \
                ("minLength" in schema or "maxLength" in schema):
            return "a string with a pattern or format and a length bound"
    if kind == "object":
        for key in ("patternProperties", "propertyNames"):
            if key in schema:
                return f"{key} on an object"
    for value in schema.values():
        for item in (value if isinstance(value, list) else [value]):
            found = xgrammar_unsupported(item) if isinstance(item, dict) else None
            if found:
                return found
    return None


def xgrammar_check(tokenizer) -> Callable[[dict], None] | None:
    """A compile of a response-format schema with xgrammar and the model's
    own tokenizer, as the server's grammar backend does at request time;
    None when xgrammar is not installed (the notebooks install it)."""
    if tokenizer is None:
        return None
    try:
        import xgrammar
    except ImportError:
        return None
    compiler = xgrammar.GrammarCompiler(xgrammar.TokenizerInfo.from_huggingface(tokenizer))

    def check(schema: dict) -> None:
        compiler.compile_json_schema(json.dumps(schema))

    return check


def _react(thought: str, command: str) -> str:
    return (f"Thought: {thought}\nAction: act\n"
            f"Action Input: {json.dumps({'command': command}, sort_keys=True)}")


def _final(answer: str) -> str:
    return f"Thought: I am stopping here.\nFinal Answer: {answer}"


def _last_admissible(user: str) -> list[str]:
    body = user.rsplit("\n\nStep ", 1)[0]
    if ADMISSIBLE_HEADER not in body:
        return []
    return [line.strip() for line in body.rsplit(ADMISSIBLE_HEADER, 1)[1].splitlines()
            if line.strip()]


class StandInPolicy:
    """The stand-in's replies. ``plans`` maps a task text to its expert
    commands (winning plans only)."""

    def __init__(self, plans: Mapping[str, Sequence[str]]):
        self._plans = {task: list(commands) for task, commands in plans.items()}

    def reply(self, messages: Sequence[Mapping], seed) -> str:
        user = messages[-1]["content"]
        task = user.split("Task: ", 1)[1].split("\n\nObservations so far:", 1)[0] \
            if "Task: " in user else ""
        history = user.split("\n\nObservations so far:\n", 1)[-1].rsplit("\n\nStep ", 1)[0]
        done = max((int(n) for n in _OBSERVATION.findall(history)), default=0)  # [0]: the task
        plan = self._plans.get(task)
        if plan is not None and stand_in_wins(task, seed):
            if done < len(plan):
                return _react("Following the plan.", plan[done])
            return _final("The task is done.")
        if not _long_loss(task, seed) and done >= SHORT_LOSS_STEPS:
            return _final("I could not finish the task.")
        if done and done % INVALID_EVERY == 0:
            return _react("Trying something else.", "jump around")
        goto = sorted(c for c in _last_admissible(user) if c.startswith("go to ")) or ["look"]
        return _react("Looking around.", goto[(_draw(task, seed) + done) % len(goto)])


class StandInServer:
    """``vllm serve``'s HTTP surface for one model, in a thread of this
    process. ``count_prompt`` counts a request's messages as the server's
    tokenizer would; ``outage_seed`` makes every request with that seed fail
    with 503 (a server that died) until it is set back to None."""

    def __init__(self, model_id: str, policy: StandInPolicy, *,
                 count_prompt: Callable[[list[dict]], int],
                 context_window: int = CONTEXT_WINDOW,
                 check_protocol: bool = True,
                 check_schema: Callable[[dict], None] | None = None,
                 template_kwargs: Mapping | None = None):
        self.model_id = model_id
        self.template_kwargs = dict(template_kwargs or {})
        self.outage_seed = None
        self.requests = 0
        self.failed = 0  # the outage's 503s
        self.refused = 0  # every other non-200 answer
        self.schemas_compiled = 0
        self._check_protocol = check_protocol
        self._check_schema = check_schema
        self._policy = policy
        self._count = count_prompt
        self._window = context_window
        self._lock = threading.Lock()
        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self._httpd.daemon_threads = True
        self._thread = threading.Thread(target=self._httpd.serve_forever, name="stand-in",
                                        daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self._httpd.server_address[1]}"

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._httpd.shutdown()
        self._httpd.server_close()

    def _models(self) -> dict:
        return {"object": "list", "data": [{"id": self.model_id, "object": "model",
                                            "max_model_len": self._window}]}

    def _refuse(self, message: str) -> tuple[int, dict]:
        with self._lock:
            self.refused += 1
        return 400, {"object": "error", "message": message}

    def _complete(self, payload: dict) -> tuple[int, dict]:
        with self._lock:
            self.requests += 1
            if self.outage_seed is not None and payload.get("seed") == self.outage_seed:
                self.failed += 1
                return 503, {"object": "error", "message": "the stand-in is down"}
        issues = protocol_issues(payload) if self._check_protocol else []
        if dict(payload.get("chat_template_kwargs") or {}) != self.template_kwargs:
            issues.append(f"chat_template_kwargs {payload.get('chat_template_kwargs')!r} are "
                          f"not the protocol's {self.template_kwargs!r}")
        if issues:
            return self._refuse(f"the stand-in refuses a request that breaks the pilot's "
                                f"protocol: {'; '.join(issues)}")
        error = None
        with self._lock:  # tokenizers and grammar compilers are not shared across threads
            try:
                prompt = self._count(payload["messages"])
            except Exception as exc:  # noqa: BLE001 - the server answers 400 to a template error
                error = f"the chat template refused the request: {exc}"
            else:
                if payload.get("response_format") and self._check_schema is not None:
                    try:
                        self._check_schema(payload["response_format"]["json_schema"]["schema"])
                    except Exception as exc:  # noqa: BLE001 - the server answers 400
                        error = f"the grammar backend refused the schema: {exc}"
                    else:
                        self.schemas_compiled += 1
        if error is not None:  # refused once the lock is free: _refuse takes it again
            return self._refuse(error)
        if prompt + payload["max_tokens"] > self._window:
            return self._refuse(
                f"This model's maximum context length is {self._window} tokens. However, you "
                f"requested {prompt + payload['max_tokens']} tokens.")
        if payload.get("response_format"):
            text = json.dumps({"proposals": []})  # the teacher's schema allows none
        else:
            text = self._policy.reply(payload["messages"], payload.get("seed"))
        completion = max(1, len(text) // 4)
        return 200, {"id": f"stand-in-{self.requests}", "object": "chat.completion",
                     "created": int(time.time()), "model": self.model_id,
                     "choices": [{"index": 0, "finish_reason": "stop",
                                  "message": {"role": "assistant", "content": text}}],
                     "usage": {"prompt_tokens": prompt, "completion_tokens": completion,
                               "total_tokens": prompt + completion}}

    def _handler(self):
        server = self

        class Handler(BaseHTTPRequestHandler):
            def _send(self, status: int, body: dict) -> None:
                data = json.dumps(body).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):  # noqa: N802 - http.server's name
                if self.path == "/version":
                    self._send(200, {"version": STAND_IN_VERSION})
                elif self.path == "/v1/models":
                    self._send(200, server._models())
                else:
                    self._send(404, {"object": "error", "message": f"no route {self.path}"})

            def do_POST(self):  # noqa: N802 - http.server's name
                if self.path != "/v1/chat/completions":
                    self._send(404, {"object": "error", "message": f"no route {self.path}"})
                    return
                length = int(self.headers.get("Content-Length") or 0)
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                self._send(*server._complete(payload))

            def log_message(self, *args):  # no line per request
                return

        return Handler
