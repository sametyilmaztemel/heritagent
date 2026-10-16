"""The pilot's rehearsal without a GPU or a model (``cli rehearse``; issue
#13 Phase B): the stand-in server answers as ``vllm serve`` does, its
replies follow the plans it was given, it knows the report's counts in
advance, and the command runs the pilot through an outage and a restore,
then fails when the report does not say what the stand-in did. Scripted
games; no ALFWorld, no tokenizer download."""

import json

import pytest

from evaluation.exp0001 import cli, rehearsal, serving
from evaluation.exp0001.pilot_run import PILOT_ATTEMPT_SEED
from evaluation.exp0001.protocol import CONTEXT_WINDOW, attempt_sampling_seed
from runtime.model_adapters import ModelMessage
from runtime.model_adapters.base import GenerationSettings
from runtime.model_adapters.errors import BackendUnavailableError, ContextWindowExceeded
from runtime.model_adapters.openai_server import ServerRequestError, VLLMServerAdapter
from runtime.policies.planner_react import ReActPlanner
from runtime.tool_adapters.alfworld import ACT_TOOL
from tests.evaluation.exp0001_workers import GameFactory
from tests.evaluation.test_exp0001_lifetime import Game
from tests.evaluation.test_exp0001_pipeline import PILOTS, splits  # noqa: F401 (fixture)

MODEL = "Org/Model-7B"
TASK = "You are in a room.\n\nYour task is to: solve g1.\n\nAdmissible commands:\ngo to a 1\nlook"


def chars(messages):
    return sum(len(m["content"]) for m in messages) // 4


def messages(task, observations):
    built = ReActPlanner({"style": "react", "max_plan_steps": 50}).build_messages(
        task=task, observations=[task, *observations], skills=(), tools=[ACT_TOOL],
        step=len(observations) + 1, max_steps=50)
    return [{"role": m.role, "content": m.content} for m in built]


def seed_where(task, wins):
    return next(s for s in range(1000) if rehearsal.stand_in_wins(task, s) is wins)


def test_the_policy_follows_the_plan_when_it_wins_and_walks_when_it_loses():
    policy = rehearsal.StandInPolicy({TASK: ["go to a 1", "take x 1"]})
    win = seed_where(TASK, True)
    replies = [policy.reply(messages(TASK, ["obs"] * n), win) for n in range(3)]
    assert [ReActPlanner.parse(r).action.arguments["command"] for r in replies[:2]] == \
        ["go to a 1", "take x 1"]
    assert ReActPlanner.parse(replies[2]).final_answer
    lose = next(s for s in range(1000) if not rehearsal.stand_in_wins(TASK, s)
                and not rehearsal._long_loss(TASK, s))
    walk = [ReActPlanner.parse(policy.reply(messages(TASK, ["obs"] * n), lose))
            for n in range(rehearsal.SHORT_LOSS_STEPS + 1)]
    assert [d.action.arguments["command"] for d in walk[:-1]] == ["go to a 1"] * 3
    assert walk[-1].final_answer  # a short loss gives up
    long = next(s for s in range(1000) if not rehearsal.stand_in_wins(TASK, s)
                and rehearsal._long_loss(TASK, s))
    late = ReActPlanner.parse(policy.reply(messages(TASK, ["obs"] * rehearsal.INVALID_EVERY),
                                           long))
    assert late.action.arguments["command"] == "jump around"  # an inadmissible command
    # a task it has no plan for is never won
    other = TASK.replace("g1", "g2")
    assert ReActPlanner.parse(policy.reply(messages(other, []), win)).action.arguments[
        "command"] != "take x 1"


def test_the_policy_is_a_function_of_the_task_and_the_seed():
    policy = rehearsal.StandInPolicy({TASK: ["go to a 1"]})
    for seed in (None, 1, 2, 3):
        assert policy.reply(messages(TASK, ["o"]), seed) == policy.reply(messages(TASK, ["o"]),
                                                                          seed)
    wins = [rehearsal.stand_in_wins(TASK, s) for s in range(300)]
    assert 0.5 < sum(wins) / len(wins) < 0.8  # two thirds of the seeded attempts


def test_the_expected_counts_follow_the_stand_ins_draws():
    plans = [{"game_id": f"g{i}", "task": f"task {i}", "commands": ["win"], "won": i % 3 != 0}
             for i in range(12)]
    expected = rehearsal.expected_counts(plans)
    t0 = sum(p["won"] and rehearsal.stand_in_wins(p["task"], None) for p in plans)
    wins = {p["game_id"]: [p["won"] and rehearsal.stand_in_wins(
        p["task"], attempt_sampling_seed(PILOT_ATTEMPT_SEED, p["game_id"], a)) for a in range(3)]
        for p in plans}
    assert expected == {"t0_successes": t0, "t07_successes": sum(map(sum, wins.values())),
                        "t07_pairs": sum(any(w) and not all(w) for w in wins.values()),
                        "t07_solved_in_any_attempt": sum(map(any, wins.values()))}
    assert expected["t07_pairs"] > 0


def test_the_stand_in_answers_as_the_server_does():
    policy = rehearsal.StandInPolicy({TASK: ["go to a 1"]})
    with rehearsal.StandInServer(MODEL, policy, count_prompt=chars) as server:
        adapter = VLLMServerAdapter(server.url, MODEL, "b" * 40, transport_retries=1,
                                    backoff_seconds=0.01)
        adapter.require_served()
        assert adapter.served_context_window() == CONTEXT_WINDOW
        assert cli._server_version(server.url) == rehearsal.STAND_IN_VERSION
        settings = GenerationSettings(temperature=0.0, max_tokens=512, seed=None,
                                      stop=("\nObservation:",))
        reply = adapter.generate([ModelMessage(**m) for m in messages(TASK, [])], settings)
        assert reply.prompt_tokens == chars(messages(TASK, [])) and reply.completion_tokens
        assert ReActPlanner.parse(reply.text).action is not None
        preflight = cli._preflight(adapter)
        assert preflight["teacher_reply"] == {"proposals": []}
        server.outage_seed = 7
        with pytest.raises(BackendUnavailableError, match=server.url):
            adapter.generate([ModelMessage("user", "x")],
                             GenerationSettings(temperature=0.7, max_tokens=8, seed=7))
        assert server.failed == 2  # the request and its one retry
        server.outage_seed = None
        with pytest.raises(ContextWindowExceeded):
            adapter.generate([ModelMessage("user", "x" * 4 * CONTEXT_WINDOW)], settings)


def test_the_stand_in_refuses_a_template_or_a_schema_and_keeps_answering():
    """A refusal answers 400 and frees the server for the next request."""
    def refuse_schema(schema):
        raise ValueError("no grammar for this schema")

    def refuse_system_only(messages):
        if all(m["role"] == "system" for m in messages):
            raise ValueError("No user query found in messages.")
        return chars(messages)

    policy = rehearsal.StandInPolicy({TASK: ["go to a 1"]})
    with rehearsal.StandInServer(MODEL, policy, count_prompt=refuse_system_only,
                                 check_schema=refuse_schema) as server:
        adapter = VLLMServerAdapter(server.url, MODEL, "b" * 40, transport_retries=0,
                                    backoff_seconds=0.01, timeout_seconds=10)
        react = GenerationSettings(temperature=0.0, max_tokens=512, seed=None,
                                   stop=("\nObservation:",))
        teacher = GenerationSettings(temperature=0.7, max_tokens=2048, seed=1)
        with pytest.raises(ServerRequestError, match="the grammar backend refused the schema"):
            adapter.structured_generate([ModelMessage("user", "x")], teacher, {"type": "object"})
        with pytest.raises(ServerRequestError, match="No user query found"):
            adapter.generate([ModelMessage("system", "x")], react)
        reply = adapter.generate([ModelMessage(**m) for m in messages(TASK, [])], react)
        assert ReActPlanner.parse(reply.text).action is not None
        assert (server.refused, server.schemas_compiled) == (2, 0)


# -- the command ----------------------------------------------------------------------------
class CharCounter:
    identity = "chars/4"
    tokenizer = None

    def count_chat_tokens(self, messages):
        return chars(messages)

    def count_teacher_prompt_tokens(self, text):
        return len(text) // 4


class Unwinnable(Game):
    def step(self, command):
        text, _, _, infos = super().step("look" if command == "win" else command)
        return text, 0.0, False, infos


class FactoryWithAnUnwinnableGame(GameFactory):
    """Every game but one is won by ``win``; that one never is, although its
    expert plan says it is."""

    def __init__(self, unwinnable):
        self.unwinnable = unwinnable

    def __call__(self, split, game_id):
        from runtime.tool_adapters.alfworld import AlfworldEnvironment
        if game_id == self.unwinnable:
            return AlfworldEnvironment(Unwinnable(game_id), game_id=game_id)
        return super().__call__(split, game_id)


@pytest.fixture
def stubbed(monkeypatch, splits):  # noqa: F811
    pilot = PILOTS[splits["manifest_sha256"]]
    games = sorted(g["game_id"] for g in pilot["games"])
    monkeypatch.setattr(cli, "_manifests", lambda repo: (splits, pilot))
    monkeypatch.setattr(cli, "verify_data", lambda *a: {"release": "0.2.2", "archives": {},
                                                         "games": 0})
    monkeypatch.setattr(cli, "git_state", lambda repo: ("a" * 40, False))
    monkeypatch.setattr(serving, "resolve_revision", lambda model, revision=None: "b" * 40)
    monkeypatch.setattr(cli, "hf_token_counter", lambda *a, **k: CharCounter())
    monkeypatch.setattr(cli, "AlfworldEnvFactory", lambda *a, **k: GameFactory())
    winnable = set(games[::2])
    monkeypatch.setattr(rehearsal, "expert_commands", lambda gamefile, max_steps: (
        (["win"], True) if gamefile.split("/train/")[1].rsplit("/", 1)[0] in winnable
        else ([], False)))
    return games


def rehearse(tmp_path, *extra):
    return cli.main(["rehearse", "--model", MODEL, "--data-root", str(tmp_path / "data"),
                     "--workers", "1", "--progress-every", "0", "--out-dir",
                     str(tmp_path / "out"), *extra])


def test_the_rehearsal_runs_the_pilot_through_an_outage_and_a_restore(tmp_path, stubbed,
                                                                      capsys):
    assert rehearse(tmp_path, "--mirror-dir", str(tmp_path / "mirror")) == 0
    passed = json.loads((tmp_path / "out" / rehearsal.REHEARSAL_PASSED).read_text())
    assert passed["passed"] is True and all(passed["checks"].values())
    assert set(passed["checks"]) >= {"the outage stopped the first pass",
                                     "the restore is on record",
                                     "every pilot episode recorded once",
                                     "the temperature-0.7 pairs"}
    assert passed["code"] == {"commit": "a" * 40}
    assert passed["model"] == {"model_id": MODEL, "revision": "b" * 40, "tokenizer": "chars/4"}
    assert passed["expected"]["t07_pairs"] > 0
    report = json.loads((tmp_path / "out" / rehearsal.REHEARSAL_REPORT).read_text())
    assert report["schema"] == rehearsal.REHEARSAL_REPORT_SCHEMA
    assert report["accelerator"] == rehearsal.REHEARSAL_GPU
    assert report["model"]["backend_version"] == rehearsal.STAND_IN_VERSION
    assert "measures nothing about any model" in report["note"]
    assert not (tmp_path / "out" / "pilot-report.json").exists()
    ledger = (tmp_path / "out" / "ledger.jsonl").read_text()
    assert json.loads(ledger.splitlines()[0])["run_id"] == rehearsal.REHEARSAL_RUN_ID
    assert "PASSED" in capsys.readouterr().out


def test_the_rehearsal_resumes_in_place_without_a_mirror(tmp_path, stubbed):
    assert rehearse(tmp_path) == 0
    passed = json.loads((tmp_path / "out" / rehearsal.REHEARSAL_PASSED).read_text())
    assert all(passed["checks"].values())


def test_the_rehearsal_fails_when_the_report_disagrees_with_the_stand_in(
        tmp_path, stubbed, monkeypatch, capsys):
    # the expert's plan wins this game, but the game never reports won
    monkeypatch.setattr(cli, "AlfworldEnvFactory",
                        lambda *a, **k: FactoryWithAnUnwinnableGame(stubbed[0]))
    assert rehearse(tmp_path) == 2
    assert "the rehearsal failed" in capsys.readouterr().err
    assert not (tmp_path / "out" / rehearsal.REHEARSAL_PASSED).exists()


@pytest.mark.parametrize("busy", ["out", "mirror"])
def test_the_rehearsal_refuses_a_directory_that_holds_anything(tmp_path, stubbed, busy, capsys):
    (tmp_path / busy).mkdir()
    (tmp_path / busy / "ledger.jsonl").write_text("{}\n")
    assert rehearse(tmp_path, "--mirror-dir", str(tmp_path / "mirror")) == 2
    assert "not empty" in capsys.readouterr().err


def test_the_rehearsal_refuses_a_changed_checkout(tmp_path, stubbed, monkeypatch, capsys):
    monkeypatch.setattr(cli, "git_state", lambda repo: ("a" * 40, True))
    assert rehearse(tmp_path) == 2
    assert "has changes" in capsys.readouterr().err



def test_a_pilot_directory_resumes_only_with_its_own_code_and_model(tmp_path, stubbed):
    """A pilot directory (or its mirror) holding another commit's or model's
    pilot, or one from before the start record, is refused before the mirror
    is taken over; the notebooks give each commit and model its own."""
    from argparse import Namespace
    out = tmp_path / "out"
    out.mkdir()
    (out / "ledger.jsonl").write_text("")
    session = Namespace(repo=tmp_path, data_root=tmp_path / "data", out_dir=out, mirror_dir=None,
                        workers=1, server_url="http://127.0.0.1:9", server_info=None,
                        progress_every=0)
    prepared = {"model_id": MODEL, "revision": "c" * 40, "path": None,
                "weights": {"sha256": "d" * 64}}

    def pilot():
        return cli._pilot(session, gpu={}, prepared=prepared, adapter=lambda: None,
                          counter=lambda: None)

    with pytest.raises(cli.CliError, match="unknown code and model"):
        pilot()
    other = {"run_id": "EXP-0001-calibration-pilot", "commit": "a" * 40,
             "model_id": "Qwen/Qwen2.5-7B-Instruct", "revision": "c" * 40,
             "weights_sha256": "d" * 64}
    (out / cli.PILOT_START).write_text(json.dumps(other))
    with pytest.raises(cli.CliError, match="Qwen2.5-7B-Instruct"):
        pilot()
    assert not (out / "UNMIRRORED.json").exists()


def test_a_stopped_rehearsal_stops_and_leaves_no_marker(tmp_path, stubbed, monkeypatch, capsys):
    """The stop button (or a dead kernel) during the first pass ends the
    rehearsal with 130: no second pass, no checks, and no unmirrored marker,
    which would keep the notebook's runtime."""
    passes = []

    def stopped(*args, **kwargs):
        passes.append(1)
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "run_pilot", stopped)
    assert rehearse(tmp_path, "--mirror-dir", str(tmp_path / "mirror")) == 130
    assert passes == [1] and "did not finish" in capsys.readouterr().err
    assert not (tmp_path / "out" / "UNMIRRORED.json").exists()
    assert not (tmp_path / "out" / rehearsal.REHEARSAL_PASSED).exists()


def test_a_rehearsal_stopped_while_its_mirror_fails_still_stops(tmp_path, stubbed, monkeypatch,
                                                                 capsys):
    """A stop whose last snapshot fails is a stop too: no second pass."""
    passes = []

    def stopped(*args, **kwargs):
        passes.append(1)
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "run_pilot", stopped)
    monkeypatch.setattr(cli, "_final_snapshot", lambda *a, **k: False)
    assert rehearse(tmp_path, "--mirror-dir", str(tmp_path / "mirror")) == 130
    err = capsys.readouterr().err
    assert passes == [1] and "did not finish" in err and cli.STOPPED not in err
    assert not (tmp_path / "out" / rehearsal.REHEARSAL_PASSED).exists()


def test_a_rehearsal_stopped_before_its_pilot_says_it_does_not_resume(tmp_path, stubbed,
                                                                      monkeypatch, capsys):
    def stopped(adapter):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "_preflight", stopped)
    assert rehearse(tmp_path) == 130
    err = capsys.readouterr().err
    assert "did not finish" in err and cli.STOPPED not in err


def test_the_stand_in_keeps_from_xgrammar_what_vllm_does():
    """vLLM 0.29.0's rule, on the schema the miner sends and on its full one."""
    from runtime.model_adapters.openai_server import server_constraint
    from traits.miner.miner import output_schema
    assert rehearsal.xgrammar_unsupported(server_constraint(output_schema())) is None
    assert rehearsal.xgrammar_unsupported(output_schema()) == "uniqueItems on an array"
    for schema, found in [({"type": "number", "multipleOf": 2}, "multipleOf"),
                          ({"type": "string", "format": "phone"}, "'phone'"),
                          ({"type": "string", "format": "email"}, None),
                          ({"type": "string", "pattern": "a+", "maxLength": 3}, "length bound"),
                          ({"anyOf": [{"type": "object", "propertyNames": {}}]}, "propertyNames")]:
        result = rehearsal.xgrammar_unsupported(schema)
        assert result is None if found is None else found in result, (schema, result)


def test_a_rehearsals_lock_installs_no_sigterm_marker(tmp_path):
    import signal
    before = signal.getsignal(signal.SIGTERM)
    with cli._run_dir_lock(tmp_path / "d", marker=False):
        assert signal.getsignal(signal.SIGTERM) is before
    with cli._run_dir_lock(tmp_path / "e"):
        assert signal.getsignal(signal.SIGTERM) is not before
    assert signal.getsignal(signal.SIGTERM) is before


def react_payload(**overrides):
    from runtime.model_adapters.openai_server import NEUTRAL_SAMPLING
    return {"n": 1, **NEUTRAL_SAMPLING, "temperature": 0.0, "max_tokens": 512,
            "stop": ["\nObservation:"], **overrides}


@pytest.mark.parametrize("overrides, issue", [
    ({}, None),
    ({"seed": 7, "temperature": 0.7}, None),
    ({"temperature": 0.7}, "temperature 0.7 with seed None"),
    ({"seed": 7, "temperature": 0.0}, "temperature 0.0 with seed 7"),
    ({"stop": []}, "stop []"),
    ({"max_tokens": 1024}, "max_tokens 1024"),
    ({"n": 2}, "n is 2"),
    ({"top_p": 0.8}, "neutral"),
    ({"response_format": {"type": "json_schema"}, "temperature": 0.7, "seed": 11,
      "max_tokens": 2048, "stop": None}, None),
    ({"response_format": {"type": "json_schema"}, "temperature": 0.0, "seed": None,
      "max_tokens": 2048}, "a teacher call at temperature 0.0"),
    ({"response_format": {"type": "json_schema", "json_schema": {"schema": {
        "type": "object", "properties": {"f": {"type": "array", "uniqueItems": True}}}}},
      "temperature": 0.7, "seed": 11, "max_tokens": 2048, "stop": None},
     "uniqueItems on an array, which vLLM 0.29.0 does not give to xgrammar"),
])
def test_the_stand_in_names_what_breaks_the_protocol(overrides, issue):
    found = rehearsal.protocol_issues(react_payload(**overrides))
    assert found == [] if issue is None else any(issue in f for f in found), found
def test_the_stand_in_refuses_what_the_server_would():
    """A request without the protocol's chat-template arguments, or one the
    model's chat template refuses (Qwen3.8's needs a user message), is a 400
    from the stand-in, as from the server: the rehearsal fails on it."""
    from runtime.model_adapters.openai_server import ServerRequestError

    def needs_a_user(messages):
        if not any(m["role"] == "user" for m in messages):
            raise ValueError("No user query found in messages.")
        return chars(messages)

    policy = rehearsal.StandInPolicy({})
    settings = GenerationSettings(temperature=0.0, max_tokens=8, seed=None,
                                  stop=("\nObservation:",))
    with rehearsal.StandInServer(MODEL, policy, count_prompt=needs_a_user,
                                 template_kwargs={"enable_thinking": False}) as server:
        bare = VLLMServerAdapter(server.url, MODEL, "b" * 40, transport_retries=0)
        with pytest.raises(ServerRequestError, match="chat_template_kwargs"):
            bare.generate([ModelMessage("user", "x")], settings)
        adapter = VLLMServerAdapter(server.url, MODEL, "b" * 40, transport_retries=0,
                                    chat_template_kwargs={"enable_thinking": False})
        with pytest.raises(ServerRequestError, match="No user query"):
            adapter.generate([ModelMessage("system", "x")], settings)
        assert adapter.generate([ModelMessage("user", "x")], settings).text
        # preflight sends G0's shape (system and user) and the teacher's (one user message)
        assert cli._preflight(adapter)["teacher_reply"] == {"proposals": []}


def test_the_teacher_prompt_goes_as_one_user_message():
    from traits.miner.miner import TEACHER_ROLE, teacher_messages
    assert TEACHER_ROLE == "user"
    assert [(m.role, m.content) for m in teacher_messages("prompt")] == [("user", "prompt")]
