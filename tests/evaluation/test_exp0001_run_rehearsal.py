"""The run's rehearsal without a GPU or a model (``cli rehearse-run``;
ADR-0012 "Experiment needed"; issue #13 Phase B): the oracle's rule texts
agree with the run's own, the stand-in answers in the scripted world and
refuses what breaks the protocol, and the command runs the whole run through
a blip, a server death, two restores and a pause, then holds it to the
oracle, whose checks fail when the run differs. ``run`` refuses to start
without a passed run rehearsal. Scripted games, the synthetic manifests at
the locked pool sizes; no ALFWorld, no tokenizer download. The outcomes are
the scripted world's, not EXP-0001 results."""

import copy
import functools
import json
import random
import subprocess
import sys
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import pytest

from evaluation.exp0001 import cli, pipeline, provenance, rehearsal, serving
from evaluation.exp0001 import run_rehearsal as rr
from evaluation.exp0001.allocations import interleave
from evaluation.exp0001.lifetime import t_reg_rank
from evaluation.exp0001.protocol import (
    FINAL_EVALUATION_RESERVE,
    MINING_SEEDS,
    PROTOCOL,
    T_REG_SIZE,
    attempt_sampling_seed,
)
from inheritance import ablation, generalization, interaction, regression
from runtime.model_adapters.openai_server import NEUTRAL_SAMPLING
from runtime.policies.planner_react import STOP_SEQUENCES, ReActPlanner
from runtime.tool_adapters.alfworld import ACT_TOOL
from tests.evaluation.exp0001_workers import GameFactory
from tests.evaluation.test_exp0001_pipeline import PILOTS, bank_of, splits  # noqa: F401 (fixture)

MODEL = "Org/Model-7B"
C = [f"rehearsal skill {n}" for n in ("one", "two", "three", "four", "five", "six")]


TOKENIZER = f"hf-tokenizer:{MODEL}@{'b' * 40}"  # what `run` expects of this model and revision


class Counter:
    """Words for the token counts (4B and the completions), characters for
    the prompts; no tokenizer download."""
    identity = TOKENIZER
    tokenizer = None

    def count_tokens(self, text):
        return len(text.split())

    def count_teacher_prompt_tokens(self, text):
        return len(text) // 4

    def count_chat_tokens(self, messages):
        return sum(len(m["content"]) for m in messages) // 4


def start_text(split, game_id):
    return GameFactory()(split, game_id).start()


@pytest.fixture(scope="module")
def script(splits):  # noqa: F811
    return rr.choose_script(splits, PILOTS[splits["manifest_sha256"]], bank_of(splits),
                            start_text=start_text, plan=lambda split, game_id: ["win"])


@pytest.fixture(scope="module")
def expected(script, splits):  # noqa: F811
    return rr.expected_outcome(script, splits, bank_of(splits))


def stub(mp, splits, script, root):  # noqa: F811
    """``rehearse-run``'s machine stubbed: the synthetic manifests, games and
    counter, a clean checkout, the faults at a test's pace."""
    pilot, bank = PILOTS[splits["manifest_sha256"]], bank_of(splits)
    hashes = {"expected_split_sha256": splits["manifest_sha256"],
              "expected_pilot_sha256": pilot["pilot_sha256"],
              "expected_bank_sha256": bank["bank_sha256"]}
    mp.setattr(cli, "_manifests", lambda repo: (splits, pilot, bank))
    mp.setattr(cli, "verify_data", lambda *a: {"release": "0.2.2", "archives": {}, "games": 0})
    mp.setattr(cli, "git_state", lambda repo: ("a" * 40, False))
    mp.setattr(serving, "resolve_revision", lambda model, revision=None: "b" * 40)
    mp.setattr(cli, "hf_token_counter", lambda *a, **k: Counter())
    mp.setattr(cli, "AlfworldEnvFactory", lambda *a, **k: GameFactory())
    # the synthetic manifests stand in for the locked ones
    mp.setattr(cli, "build_run_start_record",
               functools.partial(provenance.build_run_start_record, **hashes))
    mp.setattr(pipeline, "Exp0001Run", functools.partial(pipeline.Exp0001Run, **hashes))
    # retries of milliseconds, and a blip they outlive
    mp.setattr(rr, "ADAPTER_OPTIONS", {"backoff_seconds": 0.002})
    mp.setattr(rr, "BLIP_SECONDS", 0.005)
    path = root / "script.json"
    path.write_text(json.dumps(script), encoding="utf-8")
    return path


def rehearse_run(root, script_path, *extra, workers=1):
    return cli.main(["rehearse-run", "--model", MODEL, "--data-root", str(root / "data"),
                     "--workers", str(workers), "--progress-every", "0",
                     "--out-dir", str(root / "out"), "--mirror-dir", str(root / "mirror"),
                     "--script", str(script_path), *extra])


@pytest.fixture(scope="module")
def rehearsed(tmp_path_factory, splits, script):  # noqa: F811
    """One whole rehearsal, in worker processes."""
    root = tmp_path_factory.mktemp("rehearsed")
    with pytest.MonkeyPatch.context() as mp:
        code = rehearse_run(root, stub(mp, splits, script, root), workers=2)
    return SimpleNamespace(code=code, root=root, out=root / "out", run=root / "out" / "run")


# -- the oracle's rules are the protocol's --------------------------------------------------
def test_the_oracles_constants_are_the_protocols():
    assert rr.LOOKS == ablation.LOOKS and rr.ALPHAS == ablation.ALPHAS
    assert rr.TAU_C == ablation.TAU_C and rr.FUTILITY_DIVISOR == ablation.FUTILITY_DIVISOR
    assert rr.OUT_SCOPE_GAMES == generalization.OUT_SCOPE_COUNT
    # delta_out >= -0.05 over 24 games: a net loss of at most 1
    assert rr.OUT_SCOPE_MAX_NET_LOSS == max(
        n for n in range(25) if Fraction(-n, 24) >= generalization.OUT_SCOPE_MIN_DELTA_EXACT)
    assert rr.T_REG_SIZE == regression.REGRESSION_TASK_COUNT == T_REG_SIZE
    assert rr.T_REG_MAX_NET_DROP == max(
        n for n in range(41) if regression.paired_drop_within(40 - n, 40, 40, regression.TAU_R))
    assert rr.INTERACTION_GAMES == interaction.INTERACTION_TASK_COUNT
    assert rr.HARMFUL_PAIR_DELTA == Fraction(repr(interaction.HARMFUL_PAIR_DELTA))
    assert rr.FINAL_EVALUATION_RESERVE == FINAL_EVALUATION_RESERVE
    assert set(rr.SEED_OFFSET) == set(MINING_SEEDS)


def test_the_oracles_interleave_is_the_allocations(splits):  # noqa: F811
    bank = bank_of(splits)
    families = sorted(bank["families"])
    for chosen in ([rr.SCOPE_FAMILY], families, families[1:], families[:2] + families[4:]):
        for total in (24, 48, 192):
            assert rr.interleave(bank, chosen, total) == \
                [t.task_id for t in interleave(bank, chosen, total)]


def test_the_oracles_sequential_rule_is_ablations():
    draw = random.Random(20261019)
    for _ in range(400):
        rate = draw.choice((0.0, 0.02, 0.05, 0.1, 0.3))
        pairs = [(draw.random() < 0.5 + rate, draw.random() < 0.5) for _ in range(192)]
        looks, stop = ablation.sequential_path(pairs)
        mine = rr.stage2_path(pairs)
        assert mine["stop_look"] == stop["look"] and mine["decision"] == stop["decision"]
        assert mine["reason"] == stop["reason"] and mine["contribution"] == stop["point"]
        assert mine["alpha_met"] is any(look["alpha_met"] for look in looks)
        assert [(x["n_plus"], x["n_minus"]) for x in mine["looks"]] == \
            [(x["n_plus"], x["n_minus"]) for x in looks]


@pytest.mark.parametrize("seed", MINING_SEEDS)
def test_the_oracles_draws_are_the_protocols(seed):
    for task in ("pick_two_obj_and_place-A/trial_T1", "look_at_obj_in_light-B/trial_T2"):
        for attempt in range(3):
            assert rr.attempt_seed(seed, task, attempt) == \
                attempt_sampling_seed(seed, task, attempt)
        assert rr.t_reg_rank(task) == t_reg_rank(task)


def locked():
    root = Path(cli.__file__).resolve().parents[2] / "experiments" / "manifests"
    return tuple(json.loads((root / name).read_text(encoding="utf-8")) for name in (
        cli.SPLITS_FILE, cli.PILOT_FILE, cli.BANK_FILE))


@pytest.fixture(params=["synthetic", "committed"])
def world(request, expected):
    """The synthetic script's expectation, and the committed script's on the
    locked manifests (which ``rehearse-run`` plays on Colab)."""
    if request.param == "synthetic":
        return expected
    splits, pilot, bank = locked()
    script = rr.load_script()
    rr.validate_script(script, splits, pilot, bank)
    assert script["generator"]["planner"].startswith("TextWorld policy_commands")
    return rr.expected_outcome(script, splits, bank)


def test_the_scripted_world_reaches_every_gate_path(world):
    """What the script makes the rule texts decide (ADR-0012 D4-D7), on the
    synthetic manifests and on the locked ones."""
    expected = world
    per = expected["seeds"][11]["per_candidate"]
    paths = {n: (e["stage2"]["decision"], e["stage2"]["reason"], e["stage2"]["stop_look"],
                 e["stage2"]["alpha_met"]) for n, e in per.items()}
    assert paths == {C[0]: ("pass", "efficacy", 1, True), C[1]: ("pass", "efficacy", 2, True),
                     C[2]: ("pass", "efficacy", 3, True),
                     C[3]: ("fail", "no efficacy at the last look", 4, True),
                     C[4]: ("fail", "futility", 1, False)}
    assert per[C[0]]["out_of_scope"] == {"vacuous": True, "pass": True}
    assert per[C[3]]["out_of_scope"]["pass"] is False and per[C[3]]["regression"]["pass"] is False
    assert per[C[4]]["overhead_pass"] is False and per[C[4]]["regression"].get("not_run")
    assert expected["seeds"][11]["harmful"] == [(C[0], C[1])]
    assert per[C[1]]["eliminated_by"] == C[0]
    assert expected["seeds"][11]["promoted"] == [C[0], C[2]]
    assert expected["seeds"][23]["per_candidate"][C[5]]["stage2"]["reason"] == "futility"
    assert expected["seeds"][41]["candidates"] == []
    assert expected["episodes"] == {"lifetime": 540, "gate": 1198, "final": 776, "total": 2514}
    assert expected["cross_round"] == {"pairs": 24, "disagreements": 0}
    assert expected["cross_seed"] == {"pairs": 48, "disagreements": 1}
    assert expected["distinct_genomes"] == 4


# -- the stand-in --------------------------------------------------------------------------
def react_payload(task, *, skills=(), temperature=0.0, seed=None, **overrides):
    built = ReActPlanner({"style": "react", "max_plan_steps": 50}).build_messages(
        task=task, observations=[task], skills=skills, tools=[ACT_TOOL], step=1, max_steps=50)
    payload = {"model": MODEL, "n": 1, "temperature": temperature, "max_tokens": 512,
               "stop": list(STOP_SEQUENCES), **NEUTRAL_SAMPLING,
               "messages": [{"role": m.role, "content": m.content} for m in built]}
    if seed is not None:
        payload["seed"] = seed
    return {**payload, **overrides}


def skill(name, principle="Check first."):
    return SimpleNamespace(name=name, principle=principle, when_to_apply="always",
                           procedure=[{"id": "S1", "text": "look - then act: go"}])


def test_the_stand_in_reads_the_skills_a_react_prompt_expresses():
    payload = react_payload("solve g1.", skills=[skill(C[0]), skill(C[4], rr.LONG_PRINCIPLE)])
    assert rr.prompt_skills(payload["messages"][0]["content"]) == [C[0], C[4]]
    assert rr.prompt_skills(react_payload("solve g1.")["messages"][0]["content"]) == []


def test_the_stand_in_reads_the_evidence_a_teacher_prompt_carries():
    from traits.miner.miner import render_teacher_prompt
    from trajectory.storage.canonical import canonical_json
    document = {"failed_trajectory": {"steps": []}, "form": "without-admissible-commands",
                "kind": "heritagent-differential-evidence", "task": "solve g1."}
    prompt = render_teacher_prompt(evidence_text=canonical_json(document),
                                   family_universe=("a", "b"), max_proposals=10)
    assert rr.evidence_of(prompt) == document
    assert rr.evidence_of("no evidence here") is None


GOOD = {"lifetime": {"temperature": 0.7, "seed": 123456},
        "gate": {"temperature": 0.0, "seed": 11}, "final": {"temperature": 0.0}}


@pytest.mark.parametrize("kind", sorted(GOOD))
def test_the_stand_in_accepts_each_kind_of_the_runs_requests(kind):
    assert rr.run_protocol_issues(react_payload("solve g1.", **GOOD[kind])) == []


@pytest.mark.parametrize("overrides, issue", [
    ({"temperature": 0.0, "seed": 5}, "a ReAct call at temperature 0.0 with seed 5"),
    ({"temperature": 0.7}, "a ReAct call at temperature 0.7 with seed None"),
    ({"temperature": 0.7, "seed": True}, "with seed True"),
    ({"stop": []}, "stop [] is not"),
    ({"max_tokens": 513}, "max_tokens 513 is outside"),
    ({"n": 2}, "n is 2"),
    ({"top_p": 0.9}, "is not the neutral"),
])
def test_the_stand_in_names_what_breaks_the_runs_protocol(overrides, issue):
    issues = rr.run_protocol_issues(react_payload("solve g1.", **overrides))
    assert any(issue in i for i in issues), issues


def teacher_payload(seed, evidence, **overrides):
    from traits.miner.miner import output_schema, render_teacher_prompt
    from trajectory.storage.canonical import canonical_json
    from runtime.model_adapters.openai_server import VLLMServerAdapter, server_constraint
    prompt = render_teacher_prompt(evidence_text=canonical_json(evidence),
                                   family_universe=("a",), max_proposals=10)
    return {"model": MODEL, "n": 1, "temperature": 0.7, "seed": seed, "max_tokens": 2048,
            **NEUTRAL_SAMPLING, "messages": [{"role": "user", "content": prompt}],
            "response_format": VLLMServerAdapter._json_schema_format(
                server_constraint(output_schema())), **overrides}


def test_the_stand_in_refuses_a_teacher_call_outside_the_protocol():
    evidence = {"failed_trajectory": {}, "task": "solve g1."}
    assert rr.run_protocol_issues(teacher_payload(11, evidence)) == []
    for seed, overrides in ((7, {}), (11, {"temperature": 0.0}), (11, {"max_tokens": 4096})):
        assert rr.run_protocol_issues(teacher_payload(seed, evidence, **overrides))


def test_the_stand_in_answers_and_refuses_in_the_scripted_world(script, splits):  # noqa: F811
    policy = rr.RunPolicy(script, splits)
    lifetime = script["lifetime"]["L"]
    task = start_text("train", lifetime[0])  # wins at attempt 0 in seed 11
    win = rr.attempt_seed(11, lifetime[0], 0)
    lose = rr.attempt_seed(11, lifetime[0], 1)
    assert policy.respond(react_payload(task, temperature=0.7, seed=win)) == \
        (200, rehearsal._react("Following the plan.", "win"))
    assert policy.respond(react_payload(task, temperature=0.7, seed=lose))[1].startswith(
        "Thought: Looking around.")
    refused = [
        react_payload("solve no-such-game.", temperature=0.0),  # a text the script lacks
        react_payload(task, temperature=0.7, seed=rr.attempt_seed(11, lifetime[1], 0)),
        react_payload(task, seed=11, skills=[skill(C[5])]),  # seed 23's candidate in seed 11
        react_payload(task, seed=41, skills=[skill(C[0])]),  # seed 41 has none
        react_payload(task, seed=11, skills=[skill(C[0], "task_families: all")]),  # a leak
    ]
    t_eval = next(g["game_id"] for g in splits["pools"]["t_eval"]["games"])
    refused.append(react_payload(start_text("valid_unseen", t_eval), seed=11))
    for payload in refused:
        status, why = policy.respond(payload)
        assert status == 400 and why.startswith("the stand-in refuses"), payload
    assert policy.kinds["lifetime"] == 2 and len(policy.log) == 2 + len(refused)


def evidence(game, **extra):
    return {"failed_trajectory": {}, "task": start_text("train", game), **extra}


def test_the_teacher_proposes_for_the_scripted_pair_and_overflows_once(script, splits):  # noqa: F811
    policy = rr.RunPolicy(script, splits)
    pair, overflow = script["teacher"]["11"]["pair_task"], script["teacher"]["11"]["overflow_task"]
    status, text = policy.respond(teacher_payload(11, evidence(pair)))
    proposals = json.loads(text)["proposals"]
    assert status == 200 and [p["name"] for p in proposals] == C[:5]
    assert all(set(p) <= {"name", "principle", "when_to_apply", "procedure", "applicability",
                          "estimated_generality"} for p in proposals)
    # the same pair in another seed proposes nothing
    assert json.loads(policy.respond(teacher_payload(41, evidence(pair)))[1]) == \
        {"proposals": []}
    status, why = policy.respond(teacher_payload(11, evidence(overflow)))
    assert status == 400 and "maximum context length" in why
    assert policy.respond(teacher_payload(
        11, evidence(overflow, form="without-admissible-commands"))) == \
        (200, json.dumps({"proposals": []}, sort_keys=True))
    assert policy.overflows == 1
    # the preflight's teacher call carries no evidence
    assert policy.respond(teacher_payload(11, {"no": "evidence"}))[0] == 200
    assert policy.kinds["preflight"] == 1 and policy.kinds["teacher"] == 3


def test_the_faults_strike_where_the_plan_says(script, expected, splits):  # noqa: F811
    plan = rr.fault_plan(script, bank_of(splits), expected)
    assert plan["death_skill"] == C[1] and len(plan["death_games"]) == 48
    now = [0.0]
    policy = rr.RunPolicy(script, splits)
    faults = rr.Faults(**{**plan, "blip_seconds": 5.0}, game_of=policy.game,
                       clock=lambda: now[0])
    blip = react_payload("x", temperature=0.7, seed=plan["blip_seed"])
    assert faults(react_payload("x", temperature=0.7, seed=1)) is False
    assert faults(blip) is True
    now[0] = 4.9
    assert faults(react_payload("x", temperature=0.7, seed=1)) is True  # every request, 5 s
    now[0] = 5.0
    assert faults(react_payload("x", temperature=0.7, seed=1)) is False
    death_game = start_text("train", plan["death_games"][0])
    assert faults(react_payload(death_game, seed=11, skills=[skill(C[2])])) is False
    assert faults(react_payload(death_game, seed=11, skills=[skill(C[1])])) is True
    assert faults(react_payload("x")) is True  # dead until cleared
    faults.clear()
    assert faults(react_payload(death_game, seed=11, skills=[skill(C[1])])) is False  # once


# -- the whole rehearsal -------------------------------------------------------------------
def test_the_rehearsal_runs_the_whole_run_and_holds_it_to_the_oracle(rehearsed, expected):
    assert rehearsed.code == 0
    passed = json.loads((rehearsed.out / rr.RUN_REHEARSAL_PASSED).read_text())
    assert passed["schema"] == rr.RUN_REHEARSAL_PASSED_SCHEMA and passed["kind"] == "run"
    assert passed["passed"] is True and all(passed["checks"].values())
    assert len(passed["checks"]) >= 50
    assert passed["episodes"] == expected["episodes"]
    assert passed["code"] == {"commit": "a" * 40} and passed["workers"] == 2
    assert passed["model"] == {"model_id": MODEL, "revision": "b" * 40, "tokenizer": TOKENIZER}
    assert passed["protocol_sha256"] == rr.protocol_sha256(provenance.plain(PROTOCOL))
    assert "measures nothing about any model" in passed["note"]
    assert passed["measures"]["aborted_attempts"] >= 1
    assert passed["measures"]["census_failures"] == []
    start = json.loads((rehearsed.run / "run-start.json").read_text())
    assert start["run_id"] == rr.RUN_REHEARSAL_RUN_ID
    assert start["weights"]["sha256"] == rr.SENTINEL_WEIGHTS_SHA256
    assert start["model"]["backend_version"] == rehearsal.STAND_IN_VERSION
    assert not (rehearsed.run / cli.UNMIRRORED).exists()


def test_the_rehearsals_figures_are_stamped(rehearsed):
    from evaluation.exp0001.figures import OUTPUT_NAMES, REHEARSAL_STAMP
    manifest = json.loads((rehearsed.out / "figures" / "figures.json").read_text())
    assert manifest["rehearsal"] is True and manifest["state"] == "complete"
    assert {f["name"] for f in manifest["figures"] if f["status"] == "drawn"} == \
        set(OUTPUT_NAMES)
    assert REHEARSAL_STAMP in (rehearsed.out / "figures" / "captions.md").read_text()


def perturbed(expected, change):
    copied = copy.deepcopy(expected)
    change(copied)
    return copied


def failed(checks):
    return sorted(name for name, ok in checks.items() if not ok)


@pytest.mark.parametrize("change, check", [
    (lambda e: e["seeds"][11]["per_candidate"][C[2]]["stage2"].update(stop_look=2),
     "seed 11: every stage-2 path (ADR-0012 D4)"),
    (lambda e: e["seeds"][11]["per_candidate"][C[1]].update(eliminated_by=None),
     "seed 11: the eliminations"),
    (lambda e: e["seeds"][11]["per_candidate"][C[2]].update(verdict="reject"),
     "seed 11: the verdicts"),
    (lambda e: e["seeds"][11]["per_candidate"][C[3]]["out_of_scope"].update(net_loss=1),
     "seed 11: R5 and 4B as scripted (D5-D7)"),
    (lambda e: e["seeds"][23]["per_candidate"][C[5]].update(overhead_pass=False),
     "seed 23: R5 and 4B as scripted (D5-D7)"),
    (lambda e: e["rounds"][11]["R5"]["cell_readers"].update(
        {next(iter(e["rounds"][11]["R5"]["cell_readers"])): 9}),
     "seed 11: the rounds, cells and readers (D2)"),
    (lambda e: e["seeds"][11].update(candidates=e["seeds"][11]["candidates"][::-1]),
     "seed 11: the frozen candidates"),
])
def test_the_gate_checks_fail_when_the_run_differs(rehearsed, expected, change, check):
    assert failed(rr.gate_checks(rehearsed.run, perturbed(expected, change),
                                 run_id=rr.RUN_REHEARSAL_RUN_ID, g0_genome_id=json.loads(
                                     (rehearsed.run / "run-start.json").read_text())["g0"][
                                         "genome_id"])) == [check]


def test_the_cell_check_fails_for_another_rounds_cell(rehearsed, expected):
    checks = rr.gate_checks(rehearsed.run, expected, run_id=rr.RUN_REHEARSAL_RUN_ID,
                            g0_genome_id="G-another")
    assert failed(checks) == [f"seed {s}: every comparison read its own round's cell (D2)"
                              for s in (11, 23)]


def ledger_lines(rehearsed):
    return [json.loads(line) for line in
            (rehearsed.run / "ledger.jsonl").read_text().splitlines()]


def test_the_ledger_checks_fail_when_the_ledger_differs(rehearsed, expected):
    lines = ledger_lines(rehearsed)
    interrupted = "gate/11/R2"
    assert failed(rr.ledger_checks(lines, expected, interrupted=interrupted)) == []
    assert failed(rr.ledger_checks(lines, expected, interrupted=None)) == [
        "one reserve check per round, one more for the interrupted round (D9)"]
    attempts = [i for i, r in enumerate(lines) if r.get("type") == "attempt"
                and r["key"].startswith("gate/11/R1/")]
    swapped = list(lines)
    swapped[attempts[3]], swapped[attempts[4]] = swapped[attempts[4]], swapped[attempts[3]]
    assert failed(rr.ledger_checks(swapped, expected, interrupted=interrupted)) == [
        "each round's queue in ascending sha256 of its keys (D2)"]
    reserve = next(i for i, r in enumerate(lines) if r.get("type") == "reserve"
                   and r["label"] == "gate/23/R1")
    assert "one reserve check per round, one more for the interrupted round (D9)" in failed(
        rr.ledger_checks(lines[:reserve] + lines[reserve + 1:], expected,
                         interrupted=interrupted))
    assert failed(rr.ledger_checks(lines, perturbed(
        expected, lambda e: e["episodes"].update(total=2515)), interrupted=interrupted)) == [
        "the episode total"]


@pytest.mark.parametrize("change, check", [
    (lambda e: e["cross_seed"].update(disagreements=0), "the D8 cell measures"),
    (lambda e: e.update(cells=e["cells"] + 1), "the D8 cell measures"),
    (lambda e: next(iter(e["final"].values())).update(t_eval_successes=0),
     "the final cells and their sharing (ADR-0009 D2)"),
    (lambda e: e["seeds"][11].update(promoted=[C[0]]), "the report's gate summaries"),
])
def test_the_report_checks_fail_when_the_report_differs(rehearsed, expected, change, check):
    report = json.loads((rehearsed.run / "report.json").read_text())
    assert failed(rr.report_checks(report, perturbed(expected, change),
                                   ledger_lines(rehearsed))) == [check]


def test_a_rehearsal_whose_run_differs_from_the_oracle_fails_without_a_record(
        tmp_path, splits, script, monkeypatch, capsys):  # noqa: F811
    real = rr.expected_outcome

    def disagreeing(*args):
        return perturbed(real(*args), lambda e: e["seeds"][11]["per_candidate"][C[2]].update(
            verdict="reject"))

    monkeypatch.setattr(rr, "expected_outcome", disagreeing)
    assert rehearse_run(tmp_path, stub(monkeypatch, splits, script, tmp_path)) == 2
    assert "the run rehearsal failed: ['seed 11: the verdicts']" in capsys.readouterr().err
    assert not (tmp_path / "out" / rr.RUN_REHEARSAL_PASSED).exists()


@pytest.mark.parametrize("busy", ["out", "mirror"])
def test_the_rehearsal_refuses_a_directory_that_holds_anything(tmp_path, splits, script,  # noqa: F811
                                                               monkeypatch, busy, capsys):
    path = stub(monkeypatch, splits, script, tmp_path)
    (tmp_path / busy).mkdir()
    (tmp_path / busy / "old").write_text("x")
    assert rehearse_run(tmp_path, path) == 2
    assert "is not empty" in capsys.readouterr().err


def test_the_rehearsal_refuses_a_changed_checkout_and_a_foreign_script(
        tmp_path, splits, script, monkeypatch, capsys):  # noqa: F811
    path = stub(monkeypatch, splits, script, tmp_path)
    monkeypatch.setattr(cli, "git_state", lambda repo: ("a" * 40, True))
    assert rehearse_run(tmp_path, path) == 2
    assert "has changes" in capsys.readouterr().err
    monkeypatch.setattr(cli, "git_state", lambda repo: ("a" * 40, False))
    path.write_text(json.dumps({**script, "manifests": {**script["manifests"],
                                                         "bank_sha256": "c" * 64}}))
    assert rehearse_run(tmp_path, path) == 2
    assert "the script was built for" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def test_rehearse_run_has_no_option_that_reaches_the_protocol():
    """No rehearsal-only scaling: the full protocol runs."""
    sub = next(a for a in cli.parser()._actions if a.dest == "command")
    options = {a.dest for a in sub.choices["rehearse-run"]._actions} - {"help"}
    assert options == {"model", "revision", "data_root", "repo", "workers", "progress_every",
                       "out_dir", "mirror_dir", "script", "max_hours"}


# -- binding a run to a passed rehearsal ----------------------------------------------------
def check(passed, machine=None, **overrides):
    machine = machine or passed  # this machine's Python and software: the rehearsal's
    here = {"commit": "a" * 40, "python": machine["python"], "software": machine["software"],
            "model_id": MODEL, "revision": "b" * 40, "tokenizer": TOKENIZER,
            "protocol": rr.protocol_sha256(provenance.plain(PROTOCOL)), "workers": 2}
    return rr.check_run_rehearsal(passed, **{**here, **overrides})


def test_a_passed_rehearsal_covers_only_its_own_machine(rehearsed):
    passed = json.loads((rehearsed.out / rr.RUN_REHEARSAL_PASSED).read_text())
    assert check(passed) == {}
    for key, value in (("commit", "c" * 40), ("python", "3.0.0"), ("model_id", "Org/Other"),
                       ("revision", "d" * 40), ("tokenizer", "other"), ("protocol", "e" * 64),
                       ("workers", 16), ("software", {"vllm": "0.0"})):
        differ = check(passed, **{key: value})
        assert list(differ) == [{"protocol": "protocol_sha256"}.get(key, key)], key
    assert list(check({**passed, "passed": False})) == ["record"]
    assert list(check({**passed, "kind": "pilot"})) == ["record"]
    pilot = {"schema": rehearsal.REHEARSAL_PASSED_SCHEMA, "passed": True}
    assert list(check(pilot, passed)) == ["record"]


def run_argv(tmp_path, *extra):
    return ["run", "--prepared", "p", "--data-root", "d", "--local-dir", str(tmp_path / "work"),
            "--run-id", "EXP-0001-main", *extra]


def stub_run(monkeypatch, tmp_path, weights="f" * 64):
    monkeypatch.setattr(cli, "gpu_inventory", lambda: [])
    monkeypatch.setattr(cli, "require_a100_80gb", lambda inventory: {"name": "A100"})
    monkeypatch.setattr(cli, "load_prepared", lambda path: {
        "model_id": MODEL, "revision": "b" * 40, "path": str(tmp_path),
        "weights": {"sha256": weights}})
    monkeypatch.setattr(cli, "git_state", lambda repo: ("a" * 40, False))

    def no_run(*args, **kwargs):
        raise AssertionError("the run started")

    monkeypatch.setattr(cli, "_run", no_run)


def test_run_needs_a_passed_run_rehearsal(tmp_path, capsys):
    with pytest.raises(SystemExit):
        cli.parser().parse_args(run_argv(tmp_path))
    assert "--rehearsal-passed" in capsys.readouterr().err


def test_run_refuses_a_rehearsal_that_does_not_cover_it_before_any_write(
        rehearsed, tmp_path, monkeypatch, capsys):
    stub_run(monkeypatch, tmp_path)
    record = rehearsed.out / rr.RUN_REHEARSAL_PASSED
    # the rehearsal ran with 2 workers; the run asks for 16
    assert cli.main(run_argv(tmp_path, "--rehearsal-passed", str(record))) == 2
    err = capsys.readouterr().err
    assert "no passed run rehearsal covers this run" in err and '"workers"' in err
    assert not (tmp_path / "work").exists()
    monkeypatch.setattr(cli, "git_state", lambda repo: ("a" * 40, True))
    assert cli.main(run_argv(tmp_path, "--rehearsal-passed", str(record), "--workers", "2")) == 2
    assert "(dirty)" in capsys.readouterr().err
    (tmp_path / "torn.json").write_text("{")
    assert cli.main(run_argv(tmp_path, "--rehearsal-passed", str(tmp_path / "torn.json"))) == 2
    assert "not a run rehearsal's record" in capsys.readouterr().err


def test_run_starts_with_a_rehearsal_that_covers_it(rehearsed, tmp_path, monkeypatch):
    stub_run(monkeypatch, tmp_path)
    started = []
    monkeypatch.setattr(cli, "_run", lambda args, **kwargs: started.append(args) or 0)
    assert cli.main(run_argv(tmp_path, "--rehearsal-passed",
                             str(rehearsed.out / rr.RUN_REHEARSAL_PASSED), "--workers", "2")) == 0
    assert len(started) == 1


@pytest.mark.parametrize("identity", ["run_id", "weights"])
def test_run_refuses_a_rehearsals_identity(tmp_path, monkeypatch, capsys, identity):
    stub_run(monkeypatch, tmp_path, weights=rr.SENTINEL_WEIGHTS_SHA256 if identity == "weights"
             else "f" * 64)
    argv = run_argv(tmp_path, "--rehearsal-passed", "x")
    if identity == "run_id":
        argv[argv.index("EXP-0001-main")] = rr.RUN_REHEARSAL_RUN_ID
    assert cli.main(argv) == 2
    assert "a rehearsal's identity, not a run's" in capsys.readouterr().err


def test_run_refuses_a_stand_in_server(tmp_path, monkeypatch, capsys):
    stub_run(monkeypatch, tmp_path)
    monkeypatch.setattr(cli, "_require_run_rehearsal", lambda args, prepared: None)
    stand_in = SimpleNamespace(metadata=lambda: SimpleNamespace(
        backend_version=rehearsal.STAND_IN_VERSION))
    monkeypatch.setattr(cli, "_adapter", lambda args, prepared: stand_in)

    def run(args, *, adapter, **kwargs):
        return adapter()

    monkeypatch.setattr(cli, "_run", run)
    assert cli.main(run_argv(tmp_path, "--rehearsal-passed", "x")) == 2
    assert "a rehearsal's stand-in server, not vLLM" in capsys.readouterr().err


def test_rehearsal_check_reads_a_run_record_as_run_does(rehearsed, tmp_path, monkeypatch,
                                                         capsys):
    monkeypatch.setattr(cli, "git_state", lambda repo: ("a" * 40, False))
    record = str(rehearsed.out / rr.RUN_REHEARSAL_PASSED)
    argv = ["rehearsal-check", "--kind", "run", "--passed", record, "--model", MODEL]
    assert cli.main(argv + ["--workers", "2"]) == 0
    assert cli.main(argv + ["--workers", "16"]) == 2
    assert cli.main(argv + ["--revision", "d" * 40]) == 2
    assert cli.main(["rehearsal-check", "--passed", record, "--model", MODEL]) == 2  # pilot kind
    assert "is not a passed rehearsal" in capsys.readouterr().err


def test_the_runs_code_imports_no_rehearsal():
    """The stand-in and its script exist only for ``rehearse-run``: the run's
    own modules never load them."""
    code = ("import sys; import evaluation.exp0001.pipeline, evaluation.exp0001.gate_runner, "
            "evaluation.exp0001.episodes, inheritance.gate; "
            "print(sorted(m for m in sys.modules if 'rehearsal' in m))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         check=True, cwd=Path(cli.__file__).parents[2]).stdout
    assert out.strip() == "[]"
