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
import os
import random
import shutil
import subprocess
import sys
import time
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
from tests.evaluation.exp0001_workers import DyingGameFactory, GameFactory, HangingGameFactory
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
    # scripted games keep no disk: the workers' recycling (Python 3.11+) is not needed
    mp.setattr(cli, "_require_recycled_workers", lambda workers: None)
    mp.setattr(cli, "_require_rehearsal_disk", lambda workers, out_dir=None: None)
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
    return SimpleNamespace(code=code, root=root, out=root / "out", run=root / "out" / "run",
                           script=root / "script.json")


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
    split_manifest, pilot, bank = locked()
    committed = rr.load_script()
    rr.validate_script(committed, split_manifest, pilot, bank)
    assert committed["generator"]["planner"].startswith("TextWorld policy_commands")
    return rr.expected_outcome(committed, split_manifest, bank)


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
    assert per[C[1]]["eliminated_by"] == [C[0]]
    assert expected["seeds"][11]["promoted"] == [C[0], C[2]]
    assert all(e["stage1"]["pass"] for e in per.values()) and per[C[0]]["stage1"]["success"]
    assert per[C[0]]["all_family"] and not any(per[c]["all_family"] for c in C[1:5])
    # D7's grid: skill three would continue past look 3 at 0.08; skill four passes at 0.03
    assert per[C[2]]["sensitivity"][Fraction(2, 25)] == {"identifiable": False}
    assert per[C[3]]["sensitivity"][Fraction(3, 100)]["pass"] is True
    cells = expected["seeds"][11]["sensitivity_cells"]
    assert cells["tau_c=0.05|tau_r=0.03"] == {"identifiable": True, "eligible": sorted(C[:3]),
                                              "eliminated": [C[1]],
                                              "promote": sorted([C[0], C[2]])}
    assert cells["tau_c=0.08|tau_r=0.03"]["identifiable"] is False
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
    # skill two's look-2 games: 8 wins of one command, 40 losses of four requests
    assert plan["death_armed_after"] == (8 * 1 + 40 * 4) // 3
    now = [0.0]
    policy = rr.RunPolicy(script, splits)
    faults = rr.Faults(**{**plan, "blip_seconds": 5.0, "death_armed_after": 2},
                       game_of=policy.game, clock=lambda: now[0])
    blip = react_payload("x", temperature=0.7, seed=plan["blip_seed"])
    assert faults(react_payload("x", temperature=0.7, seed=1)) is False
    assert faults(blip) is True
    now[0] = 4.9
    assert faults(react_payload("x", temperature=0.7, seed=1)) is True  # every request, 5 s
    now[0] = 5.0
    assert faults(react_payload("x", temperature=0.7, seed=1)) is False
    assert faults.blip_hits == 2
    death_game = start_text("train", plan["death_games"][0])
    other_game = start_text("train", script["lifetime"]["L"][0])
    # armed after two of skill two's requests on its look-2 games in seed 11; no other counts
    assert faults(react_payload(death_game, seed=11, skills=[skill(C[1])])) is False
    for other in (react_payload(other_game, seed=11, skills=[skill(C[1])]),
                  react_payload(death_game, seed=23, skills=[skill(C[1])]),
                  react_payload(death_game, seed=11),
                  react_payload(death_game, seed=11, skills=[skill(C[2])])):
        assert faults(other) is False
    assert faults(react_payload(death_game, seed=11, skills=[skill(C[1])])) is False
    assert faults.death_game_requests == 2
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
    # each family of checks is on record, the ones cmd_rehearse_run makes itself included
    for name in (CENSUS_CHECK, "preflight", "the mirror's last snapshot holds the pass's ledger",
                 "the server's death stopped the first pass", "the third pass finished",
                 "both restores are on record", "the run-start record is a rehearsal's",
                 T_REG_CHECK, E1_CHECK, "the server refused only the scripted overflow",
                 "the figures are drawn, stamped as a rehearsal's"):
        assert passed["checks"][name] is True, name
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


CENSUS_CHECK = ("every game a run may play opens through the run's factory, and the scripted "
                "ones show the script's start texts (the census)")
T_REG_CHECK = ("T_reg: the 40 solved tasks of the keyed draw, each with its earliest success "
               "(E3)")
E1_CHECK = "each seed's mining pairs: first success, first failure (E1)"


def perturbed(expected, change):
    copied = copy.deepcopy(expected)
    change(copied)
    return copied


def failed(checks):
    return sorted(name for name, ok in checks.items() if not ok)


@pytest.mark.parametrize("change, check", [
    (lambda e: e["seeds"][11]["per_candidate"][C[2]]["stage2"].update(stop_look=2),
     "seed 11: every stage-2 path (ADR-0012 D4)"),
    (lambda e: e["seeds"][11]["per_candidate"][C[1]].update(eliminated_by=[]),
     "seed 11: the eliminations"),
    (lambda e: e["seeds"][11]["per_candidate"][C[1]].update(eliminated_by=[C[2]]),
     "seed 11: the eliminations"),
    (lambda e: e["seeds"][11]["per_candidate"][C[2]].update(verdict="reject"),
     "seed 11: the verdicts"),
    (lambda e: e["seeds"][11]["per_candidate"][C[3]]["out_of_scope"].update(net_loss=1),
     "seed 11: out of scope (D5, D7)"),
    (lambda e: e["seeds"][11]["per_candidate"][C[1]]["out_of_scope"].update(cell_successes=0),
     "seed 11: out of scope (D5, D7)"),
    (lambda e: e["seeds"][23]["per_candidate"][C[5]].update(overhead_pass=False),
     "seed 23: 4B (D7)"),
    (lambda e: e["seeds"][11]["per_candidate"][C[1]]["stage1"].update(success=True),
     "seed 11: stage 1 (D3)"),
    (lambda e: e["seeds"][11]["per_candidate"][C[0]].update(
        observed=e["seeds"][11]["per_candidate"][C[0]]["observed"][::-1]),
     "seed 11: every stage-2 paired outcome (D4)"),
    (lambda e: e["seeds"][11]["per_candidate"][C[0]].update(
        observed=e["seeds"][11]["per_candidate"][C[0]]["observed"][:-1]),
     "seed 11: every stage-2 paired outcome (D4)"),
    (lambda e: e["seeds"][11]["per_candidate"][C[4]]["in_scope"].update(delta=Fraction(1, 48)),
     "seed 11: in scope by reference (D5)"),
    (lambda e: e["seeds"][11]["per_candidate"][C[1]]["regression"].update(successes_with=3),
     "seed 11: 4A and 4C (D6, D7)"),
    (lambda e: e["seeds"][11]["per_candidate"][C[0]]["pairs"][C[2]].update(
        delta=Fraction(-1, 10)), "seed 11: 4A and 4C (D6, D7)"),
    (lambda e: e["seeds"][11]["per_candidate"][C[2]]["sensitivity"].update(
        {Fraction(2, 25): {"identifiable": True, "pass": False, "stop_look": 3,
                           "reason": "futility", "contribution": Fraction(10, 144)}}),
     "seed 11: the sensitivity grid (D7)"),
    (lambda e: e["seeds"][11]["sensitivity_cells"]["tau_c=0.05|tau_r=0.03"].update(
        promote=[C[0]]), "seed 11: the sensitivity grid (D7)"),
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
    caps = json.loads((rehearsed.run / "run-start.json").read_text())["protocol"]["caps"]
    ledger_checks = functools.partial(rr.ledger_checks, caps=caps)
    assert failed(ledger_checks(lines, expected, interrupted=interrupted)) == []
    assert failed(ledger_checks(lines, expected, interrupted=None)) == [
        "one reserve check per round, one more for the interrupted round (D9)"]
    attempts = [i for i, r in enumerate(lines) if r.get("type") == "attempt"
                and r["key"].startswith("gate/11/R1/")]
    swapped = list(lines)
    swapped[attempts[3]], swapped[attempts[4]] = swapped[attempts[4]], swapped[attempts[3]]
    assert failed(ledger_checks(swapped, expected, interrupted=interrupted)) == [
        "each round's queue in ascending sha256 of its keys (D2)"]
    reserve = next(i for i, r in enumerate(lines) if r.get("type") == "reserve"
                   and r["label"] == "gate/23/R1")
    assert "one reserve check per round, one more for the interrupted round (D9)" in failed(
        ledger_checks(lines[:reserve] + lines[reserve + 1:], expected, interrupted=interrupted))
    assert failed(ledger_checks(lines, perturbed(
        expected, lambda e: e["episodes"].update(total=2515)), interrupted=interrupted)) == [
        "the episode total"]
    arithmetic = "every reserve check's arithmetic, recomputed from the ledger (D9)"
    # a reserve record whose numbers the ledger does not reproduce
    shifted = [dict(r) for r in lines]
    shifted[reserve]["calls_ahead"] += 1
    assert failed(ledger_checks(shifted, expected, interrupted=interrupted)) == [arithmetic]
    # the same records under smaller caps: the check would have stopped the run
    assert failed(rr.ledger_checks(lines, expected, interrupted=interrupted,
                                   caps={**caps, "episodes_max": 3000})) == [arithmetic]
    # a round's check after its first admission
    first = next(i for i, r in enumerate(lines) if r.get("type") == "attempt"
                 and r["key"].startswith("gate/23/R1/"))
    late = lines[:reserve] + lines[reserve + 1:first + 1] + [lines[reserve]] + lines[first + 1:]
    assert "each round's admissions after its reserve check in the session (D9)" in failed(
        ledger_checks(late, expected, interrupted=interrupted))
    # a final episode before the last gate episode
    last_gate = max(i for i, r in enumerate(lines) if r.get("type") == "episode"
                    and r["key"].startswith("gate/"))
    first_final = next(i for i, r in enumerate(lines) if r.get("type") == "episode"
                       and r["key"].startswith("final/"))
    early = list(lines)
    early.insert(last_gate, early.pop(first_final))
    assert failed(ledger_checks(early, expected, interrupted=interrupted)) == [
        "lifetimes, then the gates in seed order, then the final evaluation (E8)"]
    assert failed(ledger_checks(lines, perturbed(
        expected, lambda e: e["final_per_genome"].update(t_eval=135)),
        interrupted=interrupted)) == [
        "every round's and every final genome's episodes (D2, ADR-0009 D2)"]


def test_the_request_checks_count_each_stages_requests(script, expected):
    """A final request that carries a seed is counted as a gate's (ADR-0007
    E8: the final evaluation runs without a sampling seed)."""
    lines = [{"type": "episode", "key": "lifetime/11/x/0", "model_calls": 3},
             {"type": "episode", "key": "gate/11/R1/cell/x/f", "model_calls": 2},
             {"type": "episode", "key": "final/G-1/t_eval/x", "model_calls": 4}]
    policy = SimpleNamespace(log=[], overflows=0, script=script, kinds={
        "lifetime": 3, "gate": 2, "final": 4, "teacher": 0, "preflight": 0})
    key = "every lifetime and final request was its episodes' (E1, E8)"
    assert rr.policy_checks(policy, expected, lines)[key] is True
    policy.kinds.update(gate=3, final=3)
    assert rr.policy_checks(policy, expected, lines)[key] is False
    policy.kinds.update(gate=1, final=4)  # fewer gate requests than its episodes made
    assert rr.policy_checks(policy, expected, lines)[key] is False


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


class FactoryWithABrokenGame(GameFactory):
    def __init__(self, broken):
        self.broken = broken

    def __call__(self, split, game_id):
        if game_id == self.broken:
            raise SystemExit("duplicate object")  # as TextWorld's translator ends
        return super().__call__(split, game_id)


def shift_attempt(seed, pick):
    """Move one L task's winning attempt in ``seed``: ``pick`` chooses the task."""
    def change(e):
        wins = e["lifetime_win_attempt"][seed]
        task = pick(e, wins)
        wins[task] = (wins[task] + 1) % 3
    return change


def swap_t_reg_task(e):
    outside = next(t for t in e["lifetime_win_attempt"][11] if t not in e["t_reg"])
    e["t_reg"] = e["t_reg"][1:] + [outside]


@pytest.mark.parametrize("change, checks", [
    # seed 11 holds every task's earliest success: T_reg's baseline and seed 11's pair move
    (shift_attempt(11, lambda e, wins: e["t_reg"][0]), [E1_CHECK, T_REG_CHECK]),
    # a later seed's win is no baseline: only its mining pair moves
    (shift_attempt(41, lambda e, wins: next(iter(wins))), [E1_CHECK]),
    (swap_t_reg_task, [T_REG_CHECK]),
])
def test_the_lifetime_checks_fail_when_the_run_differs(rehearsed, expected, change, checks):
    assert failed(rr.lifetime_checks(rehearsed.run, perturbed(expected, change),
                                     run_id=rr.RUN_REHEARSAL_RUN_ID)) == sorted(checks)


def test_the_oracle_takes_a_tasks_first_success(script, splits, monkeypatch):  # noqa: F811
    """E1 and E3 read a task's first success in attempt order."""
    monkeypatch.setattr(rr.RunWorld, "lifetime_win", lambda self, seed, task, attempt:
                        attempt > 0)
    e = rr.expected_outcome(script, splits, bank_of(splits))
    assert {a for wins in e["lifetime_win_attempt"].values() for a in wins.values()} == {1}


@pytest.mark.parametrize("refused, overflows, ok", [
    (1, 1, True), (2, 1, False), (0, 0, False), (1, 0, False), (0, 1, False)])
def test_the_server_check_holds_only_the_scripted_overflow(refused, overflows, ok):
    """A refusal the policy never answered (the protocol, the chat template,
    the grammar, the window) fails the rehearsal."""
    assert rr.server_checks(refused, overflows) == {
        "the server refused only the scripted overflow": ok}


def test_the_census_keeps_its_deadline_while_a_game_never_opens(script, monkeypatch):
    monkeypatch.setattr(cli, "CENSUS_WAKE_SECONDS", 0.2)
    games = [("train", g) for g in script["lifetime"]["L"][:4]]
    started = time.monotonic()
    with pytest.raises(cli.CliError, match="census took more than --max-hours"):
        cli._census(HangingGameFactory(games[0][1]), games, 2,
                    overdue=lambda: time.monotonic() > started + 5, progress_every=0)
    assert time.monotonic() - started < 60  # its workers ended: leaving the pool waited on none


def test_a_census_worker_that_dies_fails_the_census_with_advice(script):
    games = [("train", g) for g in script["lifetime"]["L"][:3]]
    with pytest.raises(cli.CliError, match="census worker process ended abruptly.*High-RAM"):
        cli._census(DyingGameFactory(games[1][1]), games, 2, progress_every=0)


class PoolThatBreaksAtASubmit:
    """A census pool whose worker died between two results: the next submit
    raises, as ProcessPoolExecutor.submit does on a broken pool."""

    def __init__(self, *args, **kwargs):
        self.submitted = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def submit(self, function, *args):
        from concurrent.futures import Future
        from concurrent.futures.process import BrokenProcessPool
        self.submitted += 1
        if self.submitted > 2:
            raise BrokenProcessPool("A child process terminated abruptly")
        future = Future()
        future.set_result(("0" * 64, None))
        return future


def test_a_census_pool_broken_at_a_submit_fails_with_advice(monkeypatch):
    import concurrent.futures
    monkeypatch.setattr(concurrent.futures, "ProcessPoolExecutor", PoolThatBreaksAtASubmit)
    games = [("train", f"game-{n}") for n in range(4)]
    with pytest.raises(cli.CliError, match="census worker process ended abruptly.*High-RAM"):
        cli._census(object(), games, 2, progress_every=0)


def test_a_rehearsal_fails_at_its_first_failed_check_without_a_record(
        tmp_path, splits, script, monkeypatch, capsys):  # noqa: F811
    """A failed census (a game that does not open) fails the rehearsal
    before its passes, which take hours on Colab; no record is written."""
    path = stub(monkeypatch, splits, script, tmp_path)
    broken = bank_of(splits)["families"][rr.SCOPE_FAMILY][-1]["game_id"]
    monkeypatch.setattr(cli, "AlfworldEnvFactory", lambda *a, **k: FactoryWithABrokenGame(broken))
    assert rehearse_run(tmp_path, path) == 2
    err = capsys.readouterr().err
    assert "the run rehearsal failed the census" in err
    assert not (tmp_path / "out" / "run").exists()
    assert not (tmp_path / "out" / rr.RUN_REHEARSAL_PASSED).exists()


def test_the_census_names_a_game_that_fails_or_shows_another_text(script):
    games = [("train", script["lifetime"]["L"][0]), ("train", script["lifetime"]["L"][1]),
             ("train", "unplayed/trial_T1"), ("train", "broken/trial_T2")]
    want = script["texts"]
    results = [(want[f"train/{script['lifetime']['L'][0]}"], None), ("0" * 64, None),
               ("1" * 64, None), (None, "SystemExit: duplicate object")]
    assert rr.census_failures(games, results, script) == [
        f"train/{script['lifetime']['L'][1]}: its start text is not the script's",
        "train/broken/trial_T2: SystemExit: duplicate object"]


@pytest.mark.parametrize("busy", ["out", "mirror"])
def test_the_rehearsal_refuses_a_directory_that_holds_anything(tmp_path, splits, script,  # noqa: F811
                                                               monkeypatch, busy, capsys):
    path = stub(monkeypatch, splits, script, tmp_path)
    (tmp_path / busy).mkdir()
    (tmp_path / busy / "old").write_text("x")
    assert rehearse_run(tmp_path, path) == 2
    assert "is not empty" in capsys.readouterr().err


def test_the_rehearsal_refuses_a_matplotlib_that_does_not_load(tmp_path, splits, script,  # noqa: F811
                                                              monkeypatch, capsys):
    """The figures come last: a matplotlib that does not load stops the
    rehearsal at its start, with what to install, not hours later."""
    path = stub(monkeypatch, splits, script, tmp_path)
    monkeypatch.setitem(sys.modules, "matplotlib", None)
    assert rehearse_run(tmp_path, path) == 2
    assert "pip install -e .[figures]" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


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
@pytest.fixture
def committed(rehearsed, monkeypatch):
    """The synthetic script stands in for the committed one."""
    monkeypatch.setattr(rr, "SCRIPT_PATH", rehearsed.script)
    return rehearsed


def check(passed, machine=None, **overrides):
    machine = machine or passed  # this machine's Python and software: the rehearsal's
    here = {"commit": "a" * 40, "python": machine["python"], "software": machine["software"],
            "model_id": MODEL, "revision": "b" * 40, "tokenizer": TOKENIZER,
            "protocol": rr.protocol_sha256(provenance.plain(PROTOCOL)), "workers": 2,
            "script": machine["script_sha256"]}
    return rr.check_run_rehearsal(passed, **{**here, **overrides})


def test_a_passed_rehearsal_covers_only_its_own_machine(rehearsed, script):
    passed = json.loads((rehearsed.out / rr.RUN_REHEARSAL_PASSED).read_text())
    assert passed["script_sha256"] == rr.script_sha256(script)
    assert check(passed) == {}
    for key, value in (("commit", "c" * 40), ("python", "3.0.0"), ("model_id", "Org/Other"),
                       ("revision", "d" * 40), ("tokenizer", "other"), ("protocol", "e" * 64),
                       ("workers", 16), ("software", {"vllm": "0.0"}), ("script", "f" * 64)):
        differ = check(passed, **{key: value})
        assert list(differ) == [{"protocol": "protocol_sha256",
                                 "script": "script_sha256"}.get(key, key)], key
    # only a whole passed record covers anything
    for record in ({**passed, "passed": False}, {**passed, "kind": "pilot"},
                   {**passed, "checks": {}}, {**passed, "checks": {**passed["checks"], "x": False}},
                   {k: v for k, v in passed.items() if k != "script_sha256"},
                   {**passed, "model": "m"}, ["a", "list"], "text",
                   # only True passes: no truthy stand-in, no list of names
                   {**passed, "checks": {**passed["checks"], "x": 1}},
                   {**passed, "passed": "true"}, {**passed, "checks": ["a"]}):
        assert list(check(record, passed)) == ["record"], record
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
        committed, tmp_path, monkeypatch, capsys):
    stub_run(monkeypatch, tmp_path)
    record = committed.out / rr.RUN_REHEARSAL_PASSED
    # the rehearsal ran with 2 workers; the run asks for 16
    assert cli.main(run_argv(tmp_path, "--rehearsal-passed", str(record))) == 2
    err = capsys.readouterr().err
    assert "no passed run rehearsal covers this run" in err and '"workers"' in err
    assert not (tmp_path / "work").exists()
    monkeypatch.setattr(cli, "git_state", lambda repo: ("a" * 40, True))
    assert cli.main(run_argv(tmp_path, "--rehearsal-passed", str(record), "--workers", "2")) == 2
    assert "(dirty)" in capsys.readouterr().err
    monkeypatch.setattr(cli, "git_state", lambda repo: ("a" * 40, False))
    (tmp_path / "torn.json").write_text("{")
    assert cli.main(run_argv(tmp_path, "--rehearsal-passed", str(tmp_path / "torn.json"))) == 2
    assert "not a run rehearsal's record" in capsys.readouterr().err
    (tmp_path / "list.json").write_text("[]")
    assert cli.main(run_argv(tmp_path, "--rehearsal-passed", str(tmp_path / "list.json"),
                             "--workers", "2")) == 2
    assert '"record"' in capsys.readouterr().err
    # a rehearsal of another script than the committed one covers nothing
    monkeypatch.setattr(rr, "SCRIPT_PATH", tmp_path / "torn.json")
    assert cli.main(run_argv(tmp_path, "--rehearsal-passed", str(record), "--workers", "2")) == 2
    assert "committed script cannot be read" in capsys.readouterr().err
    other = tmp_path / "other.json"
    other.write_text(json.dumps({**json.loads(committed.script.read_text()), "extra": 1}))
    monkeypatch.setattr(rr, "SCRIPT_PATH", other)
    assert cli.main(run_argv(tmp_path, "--rehearsal-passed", str(record), "--workers", "2")) == 2
    assert '"script_sha256"' in capsys.readouterr().err
    assert not (tmp_path / "work").exists()


def test_run_starts_with_a_rehearsal_that_covers_it(committed, tmp_path, monkeypatch):
    stub_run(monkeypatch, tmp_path)
    monkeypatch.setattr(cli, "_require_recycled_workers", lambda workers: None)  # Python 3.10
    started = []
    monkeypatch.setattr(cli, "_run", lambda args, **kwargs: started.append(kwargs) or 0)
    assert cli.main(run_argv(tmp_path, "--rehearsal-passed",
                             str(committed.out / rr.RUN_REHEARSAL_PASSED), "--workers", "2",
                             "--stop-after-hours", "1.5")) == 0
    # the stop-after-hours clock starts inside _run, right before the run
    assert len(started) == 1 and started[0]["stop_after_hours"] == 1.5


def test_the_rehearsal_needs_disk_for_what_its_processes_keep(monkeypatch):
    import shutil
    usage = shutil.disk_usage("/")
    monkeypatch.setattr(shutil, "disk_usage", lambda path: usage._replace(free=20 * 2 ** 30))
    cli._require_rehearsal_disk(4)  # 4 x 50 games x 33 MB + 2 GB
    with pytest.raises(cli.CliError, match="needs about 28 GB of free disk"):
        cli._require_rehearsal_disk(16)


def test_the_rehearsal_measures_the_out_dir_where_it_is_another_disk(tmp_path, monkeypatch):
    """The games' copies go under the temporary directory; the run's 2 GB
    under --out-dir, which may be another disk."""
    import tempfile
    out = tmp_path / "out"
    out.mkdir()
    temp = tmp_path / "temp"
    temp.mkdir()
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(temp))
    real_stat = os.stat
    monkeypatch.setattr(os, "stat", lambda path, *a, **k: SimpleNamespace(st_dev=-1)
                        if Path(path) == out else real_stat(path, *a, **k))
    usage = shutil.disk_usage("/")
    free = {str(temp): 26 * 2 ** 30, str(out): 1 * 2 ** 30}
    monkeypatch.setattr(shutil, "disk_usage", lambda path: usage._replace(free=free[str(path)]))
    with pytest.raises(cli.CliError, match=f"needs about 2 GB of free disk under {out}"):
        cli._require_rehearsal_disk(16, out_dir=out)
    free[str(out)] = 3 * 2 ** 30
    cli._require_rehearsal_disk(16, out_dir=out)  # 16 x 50 games x 33 MB under the temp dir
    with pytest.raises(cli.CliError, match=f"needs about 28 GB of free disk under {temp}"):
        cli._require_rehearsal_disk(16, out_dir=temp / "out")  # one disk: both, 28 GB


@pytest.mark.parametrize("workers, version", [(1, (3, 12, 0)), (16, (3, 10, 20))])
def test_each_command_refuses_workers_that_are_never_replaced_before_any_write(
        tmp_path, monkeypatch, capsys, workers, version):
    """rehearse-run, rehearsal-check --kind run and run each make the check
    before anything is written."""
    monkeypatch.setattr(sys, "version_info", version)
    assert rehearse_run(tmp_path, tmp_path / "script.json", workers=workers) == 2
    assert "replaced every 50 games" in capsys.readouterr().err
    assert not (tmp_path / "out").exists() and not (tmp_path / "mirror").exists()
    assert cli.main(["rehearsal-check", "--kind", "run", "--passed", str(tmp_path / "x.json"),
                     "--model", MODEL, "--revision", "b" * 40, "--workers", str(workers)]) == 2
    assert "replaced every 50 games" in capsys.readouterr().err
    stub_run(monkeypatch, tmp_path)
    monkeypatch.setattr(cli, "_require_run_rehearsal", lambda args, prepared: None)
    assert cli.main(run_argv(tmp_path, "--rehearsal-passed", "x", "--workers",
                             str(workers))) == 2
    assert "replaced every 50 games" in capsys.readouterr().err
    assert not (tmp_path / "work").exists()


def test_the_rehearsal_refuses_a_small_disk_before_any_write(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "_require_recycled_workers", lambda workers: None)  # Python 3.10
    usage = shutil.disk_usage("/")
    monkeypatch.setattr(shutil, "disk_usage", lambda path: usage._replace(free=2 ** 30))
    assert rehearse_run(tmp_path, tmp_path / "script.json", workers=16) == 2
    assert "of free disk under" in capsys.readouterr().err
    assert not (tmp_path / "out").exists() and not (tmp_path / "mirror").exists()


@pytest.mark.parametrize("workers, version", [(1, (3, 12, 0)), (16, (3, 10, 20))])
def test_run_needs_workers_that_are_replaced(monkeypatch, workers, version):
    """Each opened game keeps 32 MB of disk until its process ends: a run
    needs worker processes replaced every 50 games (Python 3.11+)."""
    monkeypatch.setattr(sys, "version_info", version)
    with pytest.raises(cli.CliError, match="replaced every 50 games"):
        cli._require_recycled_workers(workers)
    monkeypatch.setattr(sys, "version_info", (3, 12, 0))
    cli._require_recycled_workers(2)


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
    monkeypatch.setattr(cli, "_require_recycled_workers", lambda workers: None)  # Python 3.10
    stand_in = SimpleNamespace(metadata=lambda: SimpleNamespace(
        backend_version=rehearsal.STAND_IN_VERSION))
    monkeypatch.setattr(cli, "_adapter", lambda args, prepared: stand_in)

    def run(args, *, adapter, **kwargs):
        return adapter()

    monkeypatch.setattr(cli, "_run", run)
    assert cli.main(run_argv(tmp_path, "--rehearsal-passed", "x")) == 2
    assert "a rehearsal's stand-in server, not vLLM" in capsys.readouterr().err


def test_rehearsal_check_reads_a_run_record_as_run_does(committed, tmp_path, monkeypatch,
                                                         capsys):
    monkeypatch.setattr(cli, "git_state", lambda repo: ("a" * 40, False))
    monkeypatch.setattr(cli, "_require_recycled_workers", lambda workers: None)  # Python 3.10
    record = str(committed.out / rr.RUN_REHEARSAL_PASSED)
    argv = ["rehearsal-check", "--kind", "run", "--passed", record, "--model", MODEL]
    assert cli.main(argv + ["--revision", "b" * 40, "--workers", "2"]) == 0
    assert cli.main(argv + ["--revision", "b" * 40, "--workers", "16"]) == 2
    assert cli.main(argv + ["--revision", "d" * 40, "--workers", "2"]) == 2
    capsys.readouterr()
    # what run checks, it needs
    assert cli.main(argv + ["--workers", "2"]) == 2
    assert cli.main(argv + ["--revision", "b" * 40]) == 2
    assert "--kind run needs --revision and --workers" in capsys.readouterr().err
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
