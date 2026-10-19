"""The EXP-0001 run end to end on scripted components (issue #13 Phase A):
a synthetic split manifest with the locked pool sizes, scripted ALFWorld
games, a scripted agent model and a scripted teacher. No GPU, no ALFWorld,
no model inference. The outcomes below are the scripted world's, not
EXP-0001 results."""

import copy
import dataclasses
import hashlib
import json
from types import SimpleNamespace

import pytest

from evaluation.benchmarks.alfworld.splits import (
    ALFWORLD_TASK_TYPES,
    build_split_manifest,
    collect_games,
)
from evaluation.exp0001.backends import ClosingEnvironment
from evaluation.exp0001.allocations import (
    AllocationError,
    even_split,
    out_of_scope_allocation,
    pool_tasks,
    replay_allocation,
    stage2_allocation,
)
from evaluation.exp0001.benefit_bank import build_benefit_bank
from evaluation.exp0001.episodes import EpisodeExecutor
from evaluation.exp0001.gate_runner import (
    GatePass,
    GateQueueError,
    PrefetchStop,
    key_round,
    queue_order,
)
from evaluation.exp0001.ledger import RunLedger
from evaluation.exp0001.pilot import build_pilot_manifest
from evaluation.exp0001.pipeline import Exp0001Run, RunError, _record_store_issues, cig_id
from evaluation.exp0001.protocol import CAPS, FINAL_EVALUATION_RESERVE, MINING_SEEDS
from evaluation.exp0001.provenance import (
    ProvenanceError,
    build_run_start_record,
    canonical_sha256,
)
from genome.validation.registry import TraitRegistry
from inheritance.runner import RunnerContractError
from runtime.loop import Budgets
from runtime.model_adapters import ScriptedAdapter
from runtime.model_adapters.base import GenerationResult
from runtime.seed_g0 import build_g0_seed_genome, g0_runtime_config
from runtime.tool_adapters.alfworld import AlfworldEnvironment
from tests.conftest import registry  # noqa: F401 (fixture)
from tests.evaluation.test_alfworld_splits import (
    UNSEEN_PER_FAMILY,
    full_plan,
    make_split,
    write_game,
)
from tests.evaluation.exp0001_workers import ClosingGameFactory
from tests.evaluation.test_exp0001_lifetime import Clock, Game
from tests.runtime.conftest import react_action, react_final

RUN_ID = "EXP-0001-scripted"
GOOD, BAD = "Check the receptacle first.", "Always give up early."
F = ALFWORLD_TASK_TYPES


def proposal(name, principle, families, generality):
    return {"name": name, "principle": principle, "when_to_apply": "household tasks",
            "applicability": {"task_families": list(families)},
            "estimated_generality": generality}


PROPOSALS = [proposal("check receptacle", GOOD, [F[0]], 0.9),
             proposal("give up", BAD, [F[1], F[2]], 0.4)]


class Agent(ScriptedAdapter):
    """Lifetime (temperature 0.7): wins when the sampling seed is even.
    Evaluation (temperature 0): wins only with the good skill and without
    the bad one in the prompt. ``fail_at`` raises at that call."""

    def __init__(self, fail_at=None):
        super().__init__()
        self.calls, self.fail_at, self.settings = 0, fail_at, []

    def generate(self, messages, settings):
        self.calls += 1
        self.settings.append(settings)
        if self.calls == self.fail_at:
            raise RuntimeError("backend died")
        text = "\n".join(m.content for m in messages)
        # evaluation-only metadata never reaches the agent (manifest runtime_input_policy)
        assert "task_families" not in text and "estimated_generality" not in text
        if settings.temperature > 0:
            win = settings.seed % 2 == 0
        else:
            win = GOOD in text and BAD not in text
        return GenerationResult(text=react_action("act", {"command": "win"}) if win
                                else react_final("Giving up."), finish_reason="stop")


class Teacher(ScriptedAdapter):
    def __init__(self, proposals=PROPOSALS):
        super().__init__()
        self.proposals, self.calls, self.settings = proposals, 0, []

    def structured_generate(self, messages, settings, schema):
        self.calls += 1
        self.settings.append(settings)
        text = json.dumps({"proposals": self.proposals}, sort_keys=True)
        self._validate_structured(text, schema)
        return GenerationResult(text=text, finish_reason="stop")


PILOTS = {}  # split manifest hash -> its synthetic pilot manifest
BANKS = {}  # split manifest hash -> its synthetic benefit bank (ADR-0012 D1)
TRAIN_PER_FAMILY = 230  # pools take at most 30 per family and the pilot 5: 195 left for the bank


@pytest.fixture(scope="module")
def splits(tmp_path_factory):
    root = tmp_path_factory.mktemp("alfworld")
    train = collect_games(make_split(root, "train", {f: TRAIN_PER_FAMILY for f in F}), "train")
    unseen_dir = root / "valid_unseen"
    for family, n in UNSEEN_PER_FAMILY.items():  # unseen trials never share a train id
        for i in range(n):
            write_game(unseen_dir, family, 500 + i)
    unseen = collect_games(unseen_dir, "valid_unseen")
    manifest = build_split_manifest(train, unseen, full_plan())
    pilot = PILOTS[manifest["manifest_sha256"]] = build_pilot_manifest(train, manifest)
    BANKS[manifest["manifest_sha256"]] = build_benefit_bank(train, manifest, pilot)
    return manifest


def bank_of(splits):
    return BANKS[splits["manifest_sha256"]]


class Words:
    identity = "scripted-word-counter-v1"

    def count_tokens(self, text):
        return len(text.split())


SCRIPTED = {"model_id": "scripted-test-model", "revision": "test-revision",
            "backend": "scripted", "backend_version": None}


def run_start(tmp_path, splits, caps=None, model=None):
    record = build_run_start_record(
        code_commit="a" * 40, code_dirty=False,
        model=model or {**SCRIPTED, "tokenizer": Words.identity}, teacher=SCRIPTED,
        weights={"sha256": "b" * 64}, split_manifest=splits,
        pilot_manifest=PILOTS[splits["manifest_sha256"]], bank_manifest=bank_of(splits),
        g0_genome=build_g0_seed_genome(TraitRegistry(tmp_path / "g0-registry")),
        run_id=RUN_ID, started_at="2026-10-20T09:00:00Z",
        expected_split_sha256=splits["manifest_sha256"],
        expected_pilot_sha256=PILOTS[splits["manifest_sha256"]]["pilot_sha256"],
        expected_bank_sha256=bank_of(splits)["bank_sha256"])
    if caps is not None:  # a test-only record with other caps, re-hashed like a real one
        record["protocol"]["caps"] = dict(caps)
        record["record_sha256"] = canonical_sha256(
            {k: v for k, v in record.items() if k != "record_sha256"})
    return record


class CountingGame(Game):
    closed = 0

    def close(self):
        CountingGame.closed += 1


def make_run(run_dir, splits, *, agent=None, teacher=None, caps=None, counter=None, record=None,
             workers=1, on_ledger_append=None, lock=None):
    # worker processes pickle the game factory: a module-level class then
    env_factory = (ClosingGameFactory() if workers > 1 else
                   (lambda split, game_id: ClosingEnvironment(CountingGame(game_id), game_id)))
    return Exp0001Run(
        run_dir=run_dir, run_start=record or run_start(run_dir.parent, splits, caps),
        **manifests(splits), model=agent or Agent(), teacher=teacher or Teacher(),
        env_factory=env_factory,
        token_counter=counter or Words(), ledger_clock=Clock(), episode_clock=Clock(),
        sleep=lambda _: None, workers=workers, sleep_between_retries=False,
        on_ledger_append=on_ledger_append, lock=lock)


def manifests(splits, **overrides):
    """The three pre-run manifests and the hashes the synthetic run binds to."""
    pilot, bank = PILOTS[splits["manifest_sha256"]], bank_of(splits)
    return {"split_manifest": splits, "pilot_manifest": pilot, "benefit_bank": bank,
            "expected_split_sha256": splits["manifest_sha256"],
            "expected_pilot_sha256": pilot["pilot_sha256"],
            "expected_bank_sha256": bank["bank_sha256"], **overrides}


def reserve_checks(run_dir):
    return RunLedger(run_dir / "ledger.jsonl", run_id=RUN_ID, clock=Clock(),
                     caps=json.loads((run_dir / "run-start.json").read_text())["protocol"][
                         "caps"]).reserve_checks


def ledger_keys(run_dir):
    lines = (run_dir / "ledger.jsonl").read_text(encoding="utf-8").splitlines()
    return [r["key"] for r in map(json.loads, lines) if r.get("type") == "episode"]


@pytest.fixture(scope="module")
def clean(tmp_path_factory, splits):
    run_dir = tmp_path_factory.mktemp("clean") / "run"
    agent, teacher = Agent(), Teacher()
    before = CountingGame.closed
    report = make_run(run_dir, splits, agent=agent, teacher=teacher).run()
    return run_dir, report, agent, teacher, CountingGame.closed - before


# per seed, the scripted gate (ADR-0012 D2): R1 plays both candidates' source tasks and
# stage-2 look 1 (48 with-trait games each against 48 cells each; the good candidate passes
# for efficacy, the bad one stops for futility); only the good one met alpha, so R5 plays its
# 24 out-of-scope games and the 40 T_reg tasks, each against its R5 cell; R6 has no pair
R1 = 2 + 2 * 48 + 2 * 48
R5 = 24 + 24 + 40 + 40
GATE_PER_SEED = R1 + R5
FINAL_PER_GENOME = 134 + 40 + 20  # T_eval, T_reg and the transfer subset (ADR-0009 D2)


def test_a_scripted_run_goes_end_to_end(clean, splits):
    run_dir, report, agent, teacher, closed = clean
    assert report["status"] == "complete" and report["aborted_attempts"] == 0
    # the scripted teacher proposes the same skills for every seed, so B and C are one
    # genome each
    assert report["episodes"] == 540 + 3 * GATE_PER_SEED + 3 * FINAL_PER_GENOME \
        == agent.calls == closed
    g0 = report_g0(run_dir)
    assert [(g["genome_id"] == g0, g["arms"]) for g in report["final_genomes"]] == [
        (True, ["A/once"]), (False, ["B/11", "B/23", "B/41"]), (False, ["C/11", "C/23", "C/41"])]
    assert report["model_calls"] == agent.calls + teacher.calls
    ledger = RunLedger(run_dir / "ledger.jsonl", run_id=RUN_ID, clock=Clock())
    assert ledger.state == "complete"
    for seed in MINING_SEEDS:
        gate = json.loads((run_dir / f"seed-{seed}" / "gate.json").read_text())
        good, bad = sorted(gate["verdicts"], key=lambda g: not g.startswith("check_receptacle"))
        assert gate["verdicts"] == {good: "promote", bad: "reject"}
        assert gate["eliminated_by_interaction"] == {good: None, bad: None}  # one eligible
        assert gate["aggregate_ids"] == {good: cig_id(seed, 0), bad: cig_id(seed, 1)}
        children = gate["children"]
        # ADR-0012 D3: the bad skill's source episode failed, so stage 1 passed
        assert children[f"{cig_id(seed, 1)}/S{seed}-replay"]["measurements"]["stage_pass"] is True
        bad_stage2 = children[f"{cig_id(seed, 1)}/S{seed}-ablation"]["measurements"]
        assert (bad_stage2["stop"]["decision"], bad_stage2["stop"]["reason"],
                bad_stage2["alpha_met"]) == ("fail", "futility", False)
        good_stage2 = children[f"{cig_id(seed, 0)}/S{seed}-ablation"]["measurements"]
        assert good_stage2["stop"] == {"look": 1, "round": 1, "games": 48, "decision": "pass",
                                       "reason": "efficacy"}
        assert good_stage2["contribution"]["exact"] == "1/1"
        # ADR-0012 D7: no out-of-scope check, 4A or 4C for the candidate that never met alpha
        for stage in ("generalization_out_of_scope", "interaction_regression"):
            assert children[f"{cig_id(seed, 1)}/S{seed}-{stage}"]["measurements"]["not_run"] \
                is True
        assert gate["rounds"] == {
            "R1": {"with_trait": 2 + 96, "cells": 96,
                   "cell_readers": {task: 1 for task in gate["rounds"]["R1"]["cell_readers"]}},
            "R5": {"with_trait": 64, "cells": 64,
                   "cell_readers": {task: 1 for task in gate["rounds"]["R5"]["cell_readers"]}}}
        # ADR-0012 D9: the reserve checks are ledger records, kept out of the gate artifact
        assert [(c["label"], c["round_episodes"], c["passed"]) for c in reserve_checks(run_dir)
                if c["label"].startswith(f"gate/{seed}/")] \
            == [(f"gate/{seed}/R1", R1, True), (f"gate/{seed}/R5", R5, True)]
        assert "reserve_checks" not in gate
        arms = json.loads((run_dir / f"seed-{seed}" / "arms.json").read_text())
        assert [s["gene_id"] for s in arms["B"]["genes"]["skills"]] == [good, bad]
        assert [s["gene_id"] for s in arms["C"]["genes"]["skills"]] == [good]
        for skill in arms["B"]["genes"]["skills"]:
            assert skill["origin"] == "assimilation"
            assert skill["provenance"]["cig_record"] == gate["aggregate_ids"][skill["gene_id"]]
        assert arms["C"]["generation"] == 1 and arms["C"]["parent"] == report_g0(run_dir)
        assert all(s["provenance"]["source_trajectory"].startswith("T-")
                   for s in arms["B"]["genes"]["skills"])
        candidates = json.loads((run_dir / f"seed-{seed}" / "candidates.json").read_text())
        assert {r["mining_seed"] for r in candidates["mining_records"]} == {seed}
    stats = report["statistics"]
    assert stats["status"] == "complete"
    assert stats["t_eval"]["contrasts"]["C-B"]["point_estimate"] == {"exact": "1/1", "float": 1.0}
    assert stats["h2"]["rule_met"] is True and stats["h1"]["rule_met"] is True
    assert report["genome_sizes"]["per_arm"]["A"]["runs"]["once"]["skill_genes"] == 0
    assert report["adaptation_cost"]["per_arm"]["C"]["pooled"]["solved"] == 60
    # ADR-0009 D5 and ADR-0012 D4, D7: what each gate decided, and the scope of the H3 preview
    assert report["gates"] == {str(seed): {
        "candidates": 2, "promoted": 1, "all_family_candidates": 0,
        "stage2_stops": {"fail:futility@look1": 1, "pass:efficacy@look1": 1},
        "reached_round_5": 1} for seed in MINING_SEEDS}
    assert "selected no trait" in report["adaptation_cost"]["scope"]
    # ADR-0012 D8: descriptive cell disagreement; every G0 cell of this world fails
    noise = report["cell_disagreement"]
    assert noise["descriptive_only"] is True and noise["cells"] == 3 * (96 + 64)
    assert noise["cross_seed"]["pairs"] > 0 and noise["cross_seed"]["disagreements"] == 0
    assert noise["cross_round"]["disagreements"] == 0
    # the pre-planned sensitivity grid, frozen in each gate artifact (analysis only)
    for seed in MINING_SEEDS:
        grids = report["sensitivity"][str(seed)]
        gate = json.loads((run_dir / f"seed-{seed}" / "gate.json").read_text())
        assert grids == gate["sensitivity"] and set(grids) == set(gate["verdicts"])
        for grid in grids.values():
            assert grid["analysis_only"] is True and grid["runner_calls"] == 0
            assert len(grid["cells"]) == 9


def test_each_gate_round_runs_as_one_queue_in_hash_order(clean):
    """ADR-0012 D2: a round's episodes (with-trait executions and cells) run
    as one queue in ascending sha256 of their ledger keys, after the
    previous round's and before the next round's."""
    keys = [k for k in ledger_keys(clean[0]) if k.startswith("gate/")]
    for seed in MINING_SEEDS:
        ran = [k for k in keys if k.startswith(f"gate/{seed}/")]
        rounds = [key_round(k) for k in ran]
        assert rounds == sorted(rounds) and set(rounds) == {1, 5}
        for gate_round in (1, 5):
            assert_one_hash_ordered_queue([k for k in ran if key_round(k) == gate_round])
    # the seeds' gates run one after another (each after its seed's mining)
    assert [int(k.split("/")[1]) for k in keys] == sorted(int(k.split("/")[1]) for k in keys)


def assert_one_hash_ordered_queue(queue):
    """ascending sha256 of the ledger key, computed here, not with the
    code's queue_order; cells and with-trait executions interleave rather
    than run as two blocks (ADR-0012 D2)."""
    assert queue == sorted(queue, key=lambda k: hashlib.sha256(k.encode("utf-8")).hexdigest())
    assert [queue_order(SimpleNamespace(key=k)) for k in queue] == \
        [hashlib.sha256(k.encode("utf-8")).hexdigest() for k in queue]
    is_cell = ["/cell/" in k for k in queue]
    assert any(is_cell) and not all(is_cell)
    assert is_cell != sorted(is_cell) and is_cell != sorted(is_cell, reverse=True)


def test_every_cell_is_read_only_in_its_round(clean):
    """Every comparison of a seed's gate reads the G0 cell of its own round:
    the cells are keyed by (seed, round, game) and the pipeline re-checks the
    stored trajectory ids on resume (ADR-0012 D2)."""
    run_dir = clean[0]
    keys = set(ledger_keys(run_dir))
    for seed in MINING_SEEDS:
        gate = json.loads((run_dir / f"seed-{seed}" / "gate.json").read_text())
        for child in gate["children"].values():
            for outcome in child["measurements"].get("paired_outcomes", ()):
                cell = outcome["cell"]
                assert cell["round"] == outcome["round"] and cell["seed"] == seed
                assert any(k.startswith(f"gate/{seed}/R{cell['round']}/cell/{outcome['task_id']}/")
                           for k in keys)


def test_the_generation_settings_are_the_protocols(clean):
    """ADR-0007 E1/E8: lifetime at 0.7 with the attempt seeds, gate at 0 with
    the mining seed, final evaluation at 0 with no seed; the miner at 0.7
    with the mining seed."""
    _, _, agent, teacher, _ = clean
    lifetime = [s for s in agent.settings if s.temperature == 0.7]
    gate = [s for s in agent.settings if s.temperature == 0.0 and s.seed is not None]
    final = [s for s in agent.settings if s.temperature == 0.0 and s.seed is None]
    assert len(lifetime) == 540 and len(gate) == 3 * GATE_PER_SEED
    assert len(final) == 3 * FINAL_PER_GENOME
    assert len(lifetime) + len(gate) + len(final) == len(agent.settings)
    assert {s.seed for s in gate} == set(MINING_SEEDS)
    assert all(s.stop == ("\nObservation:",) and s.max_tokens == 512 for s in agent.settings)
    assert {(s.temperature, s.seed) for s in teacher.settings} == {(0.7, seed)
                                                                 for seed in MINING_SEEDS}


def report_g0(run_dir):
    return json.loads((run_dir / "run-start.json").read_text())["g0"]["genome_id"]


def test_t_eval_is_touched_only_by_the_final_evaluation(clean, splits):
    run_dir = clean[0]
    t_eval = {t.task_id for t in pool_tasks(splits, "t_eval")}
    for key in ledger_keys(run_dir):
        if any(f"/{task_id}" in key for task_id in t_eval):
            assert key.startswith("final/") and "/t_eval/" in key


def test_a_finished_run_is_read_back_not_rerun(clean, splits):
    run_dir, report, _, _, _ = clean
    agent, teacher = Agent(), Teacher()
    assert make_run(run_dir, splits, agent=agent, teacher=teacher).run() == report
    assert agent.calls == teacher.calls == 0


def test_a_crashed_run_resumes_to_the_same_report_from_a_moved_directory(tmp_path, clean,
                                                                           splits):
    run_dir = tmp_path / "run"
    with pytest.raises(RuntimeError, match="backend died"):
        make_run(run_dir, splits, agent=Agent(fail_at=700)).run()  # in seed 11's gate
    assert (run_dir / "seed-11" / "candidates.json").exists()
    assert not (run_dir / "seed-11" / "gate.json").exists()
    seed11_pairs = len(json.loads((run_dir / "seed-11" / "candidates.json").read_text())[
        "mining_records"])  # one teacher call per pair
    moved = tmp_path / "elsewhere" / "run"  # the ledger holds no absolute path
    moved.parent.mkdir()
    run_dir.rename(moved)
    teacher = Teacher()
    report = make_run(moved, splits, teacher=teacher).run()
    assert teacher.calls == clean[3].calls - seed11_pairs  # seed 11 is not mined again
    # the episode the crash aborted stays charged its worst case (60 calls)
    assert report["aborted_attempts"] == 1
    assert report["model_calls"] == clean[1]["model_calls"] + 60
    assert {**report, "aborted_attempts": 0, "model_calls": 0} == \
        {**clean[1], "aborted_attempts": 0, "model_calls": 0}


def test_a_crash_between_the_gate_artifact_and_the_journal_is_finished_on_resume(
        tmp_path, clean, splits, monkeypatch):
    run_dir = tmp_path / "run"

    def crash(self, seed, candidates, gate):
        raise RuntimeError("killed after gate.json")

    with monkeypatch.context() as patch:
        patch.setattr(Exp0001Run, "_apply_decisions", crash)
        with pytest.raises(RuntimeError, match="killed after gate.json"):
            make_run(run_dir, splits).run()
    assert (run_dir / "seed-11" / "gate.json").exists()
    assert make_run(run_dir, splits).run() == clean[1]


def test_a_tampered_gate_artifact_fails_closed(tmp_path, splits, monkeypatch):
    run_dir = tmp_path / "run"
    with monkeypatch.context() as patch:
        patch.setattr(Exp0001Run, "_apply_decisions",
                      lambda self, *a: (_ for _ in ()).throw(RuntimeError("stop")))
        with pytest.raises(RuntimeError, match="stop"):
            make_run(run_dir, splits).run()
    path = run_dir / "seed-11" / "gate.json"
    gate = json.loads(path.read_text())
    bad = next(g for g, v in gate["verdicts"].items() if v == "reject")
    gate["verdicts"][bad] = "promote"  # arm C would take a rejected candidate
    path.write_text(json.dumps(gate))
    with pytest.raises(RunError, match="not its aggregate record's"):
        make_run(run_dir, splits).run()


def test_a_tampered_candidate_set_fails_closed(tmp_path, splits, monkeypatch):
    run_dir = tmp_path / "run"
    with monkeypatch.context() as patch:
        patch.setattr(Exp0001Run, "_gate",
                      lambda self, *a: (_ for _ in ()).throw(RuntimeError("stop")))
        with pytest.raises(RuntimeError, match="stop"):
            make_run(run_dir, splits).run()
    path = run_dir / "seed-11" / "candidates.json"
    frozen = json.loads(path.read_text())
    frozen["candidates"] = frozen["candidates"][:1]
    path.write_text(json.dumps(frozen))
    with pytest.raises(RunError, match="does not match its digest"):
        make_run(run_dir, splits).run()


def test_the_run_binds_its_run_start_record(tmp_path, splits):
    with pytest.raises(ProvenanceError, match="model adapter"):
        make_run(tmp_path / "a", splits, record=run_start(
            tmp_path, splits, model={**SCRIPTED, "model_id": "Qwen/Qwen2.5-7B-Instruct",
                                     "tokenizer": Words.identity}))

    class Other(Words):
        identity = "another-tokenizer"

    with pytest.raises(ProvenanceError, match="token counter"):
        make_run(tmp_path / "b", splits, counter=Other())
    tampered = copy.deepcopy(splits)
    tampered["pools"]["t_mine"]["games"][0], tampered["pools"]["t_eval"]["games"][0] = \
        tampered["pools"]["t_eval"]["games"][0], tampered["pools"]["t_mine"]["games"][0]
    with pytest.raises(ProvenanceError, match="does not verify"):
        Exp0001Run(run_dir=tmp_path / "c", run_start=run_start(tmp_path, splits),
                   **manifests(splits, split_manifest=tampered), model=Agent(),
                   teacher=Teacher(), env_factory=None, token_counter=Words())
    bank = copy.deepcopy(bank_of(splits))
    bank["families"][F[0]][0], bank["families"][F[1]][0] = \
        bank["families"][F[1]][0], bank["families"][F[0]][0]
    with pytest.raises(ProvenanceError, match="does not verify"):
        Exp0001Run(run_dir=tmp_path / "e", run_start=run_start(tmp_path, splits),
                   **manifests(splits, benefit_bank=bank), model=Agent(), teacher=Teacher(),
                   env_factory=None, token_counter=Words())
    for name in ("split", "pilot", "bank"):  # the default: the committed hash
        defaults = manifests(splits)
        defaults.pop(f"expected_{name}_sha256")
        with pytest.raises(ProvenanceError, match="not the pre-registered one"):
            Exp0001Run(run_dir=tmp_path / "d", run_start=run_start(tmp_path, splits),
                       **defaults, model=Agent(), teacher=Teacher(), env_factory=None,
                       token_counter=Words())
    assert not any((tmp_path / name).exists() for name in "abcde")  # nothing was written


def test_one_writer_per_run_directory(tmp_path, splits):
    first = make_run(tmp_path / "run", splits)
    with pytest.raises(RunError, match="in use"):
        make_run(tmp_path / "run", splits)
    first.close()
    make_run(tmp_path / "run", splits).close()


def test_a_caller_held_lock_outlives_the_run(tmp_path, splits):
    """The cli holds the directory through its last snapshot, after run():
    closing the run never releases the caller's lock."""
    from evaluation.exp0001.pipeline import lock_run_dir
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    held = lock_run_dir(run_dir)
    run = make_run(run_dir, splits, lock=held)
    run.close()
    with pytest.raises(RunError, match="in use"):
        lock_run_dir(run_dir)
    held.close()
    lock_run_dir(run_dir).close()


def test_a_long_skill_name_does_not_break_trajectory_files(tmp_path, splits):
    long_name = "check the receptacle " * 12  # a 252-character skill name
    # room for round R1 (the source task and look 1: 1 + 48 + 48) and the final
    # evaluation's reserve, not for R5
    caps = {**CAPS, "episodes_max": 540 + 97 + FINAL_EVALUATION_RESERVE}
    report = make_run(tmp_path / "run", splits, caps=caps,
                      teacher=Teacher([proposal(long_name, GOOD, [F[0]], 0.9)])).run()
    assert report["status"] == "incomplete" and report["stage"] == "gate/11"
    assert report["episodes"] == 540 + 97
    assert "reserve: before gate/11/R5" in report["reason"] and "episodes_max" in report["reason"]


def test_a_hard_cap_stops_the_run_incomplete(tmp_path, splits):
    run_dir = tmp_path / "run"
    caps = {**CAPS, "episodes_max": 100}
    report = make_run(run_dir, splits, caps=caps).run()
    assert report["status"] == "incomplete" and report["stage"] == "lifetime"
    assert "episodes_max 100" in report["reason"] and report["statistics"] is None
    ledger = RunLedger(run_dir / "ledger.jsonl", run_id=RUN_ID, clock=Clock(), caps=caps)
    assert ledger.state == "incomplete" and ledger.episodes == 100
    assert make_run(run_dir, splits, caps=caps).run() == report  # stays stopped


@pytest.mark.parametrize("spare,stage", [(0, "done"), (-1, "gate/41")])
def test_the_reserve_keeps_room_for_the_final_evaluation(tmp_path, splits, spare, stage):
    """ADR-0012 D9: before each gate round the run needs room for the round
    and the final evaluation's largest count. The last round is seed 41's R5;
    a cap one episode short stops the run there, before the round."""
    last_check = 540 + 2 * GATE_PER_SEED + R1 + R5 + FINAL_EVALUATION_RESERVE
    run_dir = tmp_path / "run"
    caps = {**CAPS, "episodes_max": last_check + spare}
    report = make_run(run_dir, splits, caps=caps).run()
    assert report["stage"] == stage
    checks = RunLedger(run_dir / "ledger.jsonl", run_id=RUN_ID, clock=Clock(),
                       caps=caps).reserve_checks
    assert [c["label"] for c in checks] == [f"gate/{seed}/R{r}" for seed in MINING_SEEDS
                                            for r in (1, 5)]
    assert [c["passed"] for c in checks] == [True] * 5 + [spare == 0]
    assert checks[-1]["final_reserve"] == FINAL_EVALUATION_RESERVE == 7 * 194
    if spare:
        assert report["status"] == "incomplete" and report["statistics"] is None
        assert "reserve: before gate/41/R5" in report["reason"]
        assert report["episodes"] == 540 + 2 * GATE_PER_SEED + R1  # R5 never started
    else:
        assert report["status"] == "complete"


def test_a_stop_in_the_final_evaluation_gives_no_verdict(tmp_path, splits, monkeypatch):
    """A stop the reserve cannot rule out (a call or wall-clock cap) inside
    the final evaluation: emulated by a run without the reserve."""
    import evaluation.exp0001.pipeline as pipeline
    monkeypatch.setattr(pipeline, "FINAL_EVALUATION_RESERVE", 0)
    report = make_run(tmp_path / "run", splits,
                      caps={**CAPS, "episodes_max": 540 + 3 * GATE_PER_SEED + 50}).run()
    assert report["status"] == "incomplete" and report["stage"] == "final"
    stats = report["statistics"]
    assert stats["status"] == "incomplete" and stats["h2"] is None and stats["h1"] is None


def test_a_candidate_for_every_family_gets_a_vacuous_out_of_scope_stage(tmp_path, splits):
    """ADR-0007 E6 (owner): no out-of-scope family exists, so that stage is
    vacuous; stage 2's 48 games cover every family (ADR-0012 D4, D5)."""
    universal = [proposal("everywhere", GOOD, F, 0.9)]
    run_dir = tmp_path / "run"
    agent = Agent()
    report = make_run(run_dir, splits, agent=agent, teacher=Teacher(universal)).run()
    assert report["status"] == "complete"
    # per seed: R1 the source task and look 1 (1 + 48 + 48), R5 the 40 T_reg tasks against
    # their cells; the final evaluation runs G0 and the one G1 genome every B and C arm shares
    assert report["episodes"] == 540 + 3 * (97 + 80) + 2 * FINAL_PER_GENOME == agent.calls
    assert {g["all_family_candidates"] for g in report["gates"].values()} == {1}
    for seed in MINING_SEEDS:
        gate = json.loads((run_dir / f"seed-{seed}" / "gate.json").read_text())
        (gene,) = gate["verdicts"]
        assert gate["verdicts"][gene] == "promote"
        out = gate["children"][f"{cig_id(seed, 0)}/S{seed}-generalization_out_of_scope"]
        assert out["measurements"]["vacuous"] is True and out["measurements"]["stage_pass"]
        assert out["measurements"]["locked_threshold"]["min_delta"] is None
        stage = gate["aggregates"][cig_id(seed, 0)]["stages"]["generalization_out_of_scope"]
        assert stage["pass"] is True and stage["threshold"].startswith("vacuous")
        inside = gate["children"][f"{cig_id(seed, 0)}/S{seed}-generalization_in_scope"]
        assert inside["measurements"]["by_reference"]["games"] == 48
        stage2 = gate["children"][f"{cig_id(seed, 0)}/S{seed}-ablation"]["measurements"]
        families = [o["family"] for o in stage2["paired_outcomes"]]
        assert families == [sorted(F)[j % 6] for j in range(48)]
    assert report["statistics"]["h2"]["rule_met"] is False  # B equals C in this world


def test_the_stage_2_and_out_of_scope_draws(tmp_path, splits):
    """ADR-0012 D4, D5: the bank's games in bank order, round-robin over the
    families in name order; every candidate needing n games of a family gets
    the same first n."""
    bank = bank_of(splits)
    assert even_split(10, ["b", "a", "c"]) == {"a": 4, "b": 3, "c": 3}
    assert even_split(24, F[:1]) == {F[0]: 24}
    store = TraitRegistry(tmp_path / "store")

    def envelope(name, families):
        uri = store.put("skill", name, 1, {**PROPOSALS[1],
                                            "applicability": {"task_families": list(families)}})
        return {"candidate": {"gene_id": name, "version": 1, "type": "skill",
                              "origin": "acquired", "artifact": uri,
                              "provenance": {"born_generation": 0,
                                             "source_trajectory": "T-000000000000"}},
                "validation": {"state": "candidate"}}

    two = envelope("give_up", [F[2], F[1]])
    stage2 = stage2_allocation(bank, two, store)
    first, second = sorted([F[1], F[2]])
    assert stage2.families == (first, second) and stage2.bank_sha256 == bank["bank_sha256"]
    assert [t.task_id for t in stage2.tasks[0::2]] == \
        [g["game_id"] for g in bank["families"][first][:96]]
    assert [t.task_id for t in stage2.tasks[1::2]] == \
        [g["game_id"] for g in bank["families"][second][:96]]
    assert {t.bank_id for t in stage2.tasks} == {f"benefit_bank[{first}]",
                                                f"benefit_bank[{second}]"}
    assert all(t.payload["split"] == "train" and t.payload["pool"] == "benefit_bank"
               for t in stage2.tasks)
    assert stage2_allocation(bank, two, store) == stage2  # a fixed draw
    out = out_of_scope_allocation(bank, two, store)
    omitted = sorted(f for f in F if f not in (F[1], F[2]))
    assert out.omitted_families == tuple(omitted) and len(out.tasks) == 24
    for index, family in enumerate(omitted):
        assert [t.task_id for t in out.tasks[index::4]] == \
            [g["game_id"] for g in bank["families"][family][:6]]
    # cross-trait reuse: another candidate needing the first games of a family gets them too
    one = stage2_allocation(bank, envelope("one", [F[1]]), store)
    assert [t.task_id for t in one.tasks[:96]] == [t.task_id for t in stage2.tasks
                                                   if t.family == F[1]]
    everywhere = envelope("everywhere", F)
    assert out_of_scope_allocation(bank, everywhere, store).vacuous
    assert out_of_scope_allocation(bank, everywhere, store).tasks == ()
    elsewhere = envelope("elsewhere", ["moon_landing"])
    with pytest.raises(AllocationError, match="outside ALFWorld"):
        stage2_allocation(bank, elsewhere, store)
    short = copy.deepcopy(bank)
    short["families"][F[1]] = short["families"][F[1]][:95]
    with pytest.raises(AllocationError, match="holds 95 games, 96 are needed"):
        stage2_allocation(short, two, store)


def test_replay_is_the_source_task_alone(splits):
    """ADR-0012 D3: stage 1 replays the source task with the trait, once."""
    source = pool_tasks(splits, "t_mine")[0]
    allocation = replay_allocation(source)
    assert allocation.source_task == source and allocation.tasks == (source,)


def gate_pass(tmp_path, registry, *, mode, collect_round=None, agent=None):  # noqa: F811
    from evaluation.exp0001.episodes import protocol_budgets
    ledger = RunLedger(tmp_path / "ledger.jsonl", run_id=RUN_ID, clock=Clock())
    executor = EpisodeExecutor(model=agent or Agent(), env_factory=lambda s, g: AlfworldEnvironment(
        Game(g), game_id=g), ledger=ledger, trajectories_dir=tmp_path / "t", run_id=RUN_ID,
        code_commit="a" * 40, clock=Clock(), sleep=lambda _: None)
    g0 = g0_runtime_config(registry)
    return GatePass(executor, seed=11, g0_config=g0, budgets=protocol_budgets(), mode=mode,
                    collect_round=collect_round), executor, ledger, g0


def test_the_gate_pass_keys_cells_and_contract(tmp_path, registry):  # noqa: F811
    """ADR-0012 D2: a dry pass collects its round's requests and stops at
    another round's; a game's cell is requested once per (round, pass) and
    read by every without-trait and neither request of the round; the read
    pass runs nothing."""
    from evaluation.exp0001.episodes import protocol_budgets
    from inheritance.overlay import evaluation_settings
    from inheritance.runner import EvaluationTask
    budgets, settings = protocol_budgets(), evaluation_settings(seed=11)
    task = EvaluationTask("g1", F[0], payload={"split": "train"})
    dry, executor, ledger, g0 = gate_pass(tmp_path, registry, mode="dry", collect_round=1)
    r1 = dry.runner(1, "a/ablation")
    first = r1.run_episode(task, g0, budgets, "without_trait", settings)
    again = dry.runner(1, "b/ablation").run_episode(task, g0, budgets, "neither", settings)
    assert first.status == again.status == "prefetch-placeholder" and again.condition == "neither"
    assert [s.key for s in dry.collected] == [f"gate/11/R1/cell/g1/{dry._g0_fingerprint}"]
    assert dry.cell_reads == {(1, "g1"): 2}
    with pytest.raises(RunnerContractError, match="ran at"):  # one cell, one set of settings
        dry.runner(1, "c/ablation").run_episode(task, g0, budgets, "without_trait",
                                                evaluation_settings(max_tokens=100, seed=11))
    r1.run_episode(task, g0, budgets, "with_trait", settings)
    assert dry.collected[-1].key.startswith("gate/11/R1/a/ablation/g1/with_trait/")
    with pytest.raises(RunnerContractError, match="requested twice"):
        r1.run_episode(task, g0, budgets, "with_trait", settings)
    with pytest.raises(PrefetchStop):
        dry.runner(2, "a/ablation").run_episode(task, g0, budgets, "without_trait", settings)
    assert ledger.episodes == 0  # a dry pass runs nothing
    executor.run_many(sorted(dry.collected, key=queue_order))
    assert ledger.episodes == 2
    reader = GatePass(executor, seed=11, g0_config=g0, budgets=budgets, mode="read")
    cell = reader.runner(1, "a/ablation").run_episode(task, g0, budgets, "without_trait",
                                                      settings)
    assert cell.status != "prefetch-placeholder" and cell.effective_settings == settings
    with pytest.raises(GateQueueError, match="outside its round"):
        reader.runner(2, "a/ablation").run_episode(task, g0, budgets, "without_trait", settings)
    assert ledger.episodes == 2 and len(reader.requested) == 1  # the recorded cell
    # the contract, before anything is requested
    other = dry.runner(1, "c/ablation")
    with pytest.raises(RunnerContractError, match="budgets"):
        other.run_episode(task, g0, Budgets(max_steps=4, max_total_retries=1), "with_trait",
                          settings)
    with pytest.raises(RunnerContractError, match="temperature 0"):
        other.run_episode(task, g0, budgets, "with_trait",
                          type(settings)(temperature=0.7, seed=11))
    with pytest.raises(RunnerContractError, match="seed 11"):
        other.run_episode(task, g0, budgets, "with_trait", evaluation_settings(seed=23))
    with pytest.raises(RunnerContractError, match="split"):
        other.run_episode(EvaluationTask("g2", F[0]), g0, budgets, "with_trait", settings)
    for payload in ({"split": "valid_unseen"}, {"split": "train", "pool": "t_eval"}):
        with pytest.raises(RunnerContractError, match="T_eval never enters the gate"):
            other.run_episode(EvaluationTask("g3", F[0], payload=payload), g0, budgets,
                              "with_trait", settings)
    skilled = dataclasses.replace(g0, genome_id="G-other")
    with pytest.raises(RunnerContractError, match="G0's exact config"):
        other.run_episode(EvaluationTask("g4", F[0], payload={"split": "train"}), skilled,
                          budgets, "without_trait", settings)
    with pytest.raises(ValueError, match="exactly one round"):
        GatePass(executor, seed=11, g0_config=g0, budgets=budgets, mode="dry")
    assert key_round("gate/11/R5/cell/g1/abc") == 5 and key_round("final/G-1/t_eval/g1") is None


def test_an_arm_that_assimilates_nothing_is_g0_itself(clean):
    from evaluation.exp0001.arms import ArmError, arm_genomes
    run_dir = clean[0]
    store = TraitRegistry(run_dir / "registry")
    g0 = build_g0_seed_genome(store)
    candidates = json.loads((run_dir / "seed-11" / "candidates.json").read_text())["candidates"]
    gate = json.loads((run_dir / "seed-11" / "gate.json").read_text())
    rejected = {gene: "reject" for gene in gate["verdicts"]}
    arms = arm_genomes(g0, candidates, rejected, gate["aggregate_ids"], store, 11)
    assert arms["C"] == g0 and arms["C"]["parent"] is None
    assert len(arms["B"]["genes"]["skills"]) == 2 and arms["B"]["parent"] == g0["genome_id"]
    assert arm_genomes(g0, [], {}, {}, store, 11) == {"B": g0, "C": g0}
    with pytest.raises(ArmError, match="exactly one verdict"):
        arm_genomes(g0, candidates, {}, gate["aggregate_ids"], store, 11)
    with pytest.raises(ArmError, match="canonical G0"):
        arm_genomes({**g0, "generation": 1}, candidates, rejected, gate["aggregate_ids"],
                    store, 11)



# --- review round 2: crash windows, tampering and bindings -----------------


def reopened_copy(clean, tmp_path):
    """The clean run as it stood just before its last step: the report not
    yet written and the ledger not yet complete."""
    import shutil
    run_dir = tmp_path / "run"
    shutil.copytree(clean[0], run_dir)
    (run_dir / "report.json").unlink()
    ledger = run_dir / "ledger.jsonl"
    lines = ledger.read_text(encoding="utf-8").splitlines(keepends=True)
    assert json.loads(lines[-1])["type"] == "complete"
    ledger.write_text("".join(lines[:-1]), encoding="utf-8")
    return run_dir


def edit(path, change):
    document = json.loads(path.read_text())
    change(document)
    path.write_text(json.dumps(document))


def test_a_crash_between_the_report_and_completion_resumes_to_the_report(tmp_path, clean, splits):
    run_dir = reopened_copy(clean, tmp_path)
    (run_dir / "report.json").write_text(json.dumps(clean[1]))  # the report made it to disk
    agent, teacher = Agent(), Teacher()
    assert make_run(run_dir, splits, agent=agent, teacher=teacher).run() == clean[1]
    assert agent.calls == teacher.calls == 0
    assert RunLedger(run_dir / "ledger.jsonl", run_id=RUN_ID, clock=Clock()).state == "complete"


def test_a_resume_recomputes_t_reg_and_the_arms_against_their_artifacts(tmp_path, clean, splits):
    run_dir = reopened_copy(clean, tmp_path)
    assert make_run(run_dir, splits).run() == clean[1]  # an honest resume
    for name, change in (("t_reg.json", lambda d: d["tasks"].pop()),
                         ("seed-23/arms.json", lambda d: d["C"]["genes"]["skills"].clear())):
        broken = reopened_copy(clean, tmp_path / name.replace("/", "-"))
        edit(broken / name, change)
        with pytest.raises(ProvenanceError, match="frozen with other content"):
            make_run(broken, splits).run()


def one_copy_of_a_pair(gate, gene):
    """A well-formed pair evidence in ``gene``'s stage-4 child only: its
    partner's child does not carry the second copy (ADR-0010 B4)."""
    from trajectory.storage.canonical import canonical_json, sha256_hex
    refs = sorted((a["artifacts"]["candidate"] for a in gate["aggregates"].values()),
                  key=lambda ref: ref["gene_id"])
    canonical = {"kind": "heritagent-interaction-pair",
                 "gene_ids": [ref["gene_id"] for ref in refs], "artifacts": refs}
    entry = {"evidence_sha256": sha256_hex(canonical_json(canonical).encode("utf-8")),
             "canonical_evidence": canonical,
             "other_gene_id": next(r["gene_id"] for r in refs if r["gene_id"] != gene)}
    child = gate["children"][gate["aggregate_ids"][gene] + "/S11-interaction_regression"]
    child["measurements"]["interaction"]["pair_evidence"] = [entry]


@pytest.mark.parametrize("tamper,message", [
    (lambda g, gene: g["verdicts"].update({"extra_gene": "reject"}), "exactly the frozen"),
    (lambda g, gene: g["aggregates"][g["aggregate_ids"][gene]].update(
        {"candidate_gene_id": "someone_else"}), "not its aggregate record's"),
    (lambda g, gene: g["aggregates"][g["aggregate_ids"][gene]]["artifacts"]["candidate"].update(
        {"artifact": "registry://skills/x@1/sha256:" + "0" * 64}), "not gated on its frozen"),
    (lambda g, gene: g["aggregates"][g["aggregate_ids"][gene]]["artifacts"]["candidate"].update(
        {"version": 2}), "not gated on its frozen"),
    (lambda g, gene: g["aggregates"][g["aggregate_ids"][gene]]["artifacts"]["candidate"].update(
        {"gene_id": "someone_else"}), "not gated on its frozen"),
    (lambda g, gene: g["aggregates"][g["aggregate_ids"][gene]].pop("artifacts"),
     "not gated on its frozen"),
    (lambda g, gene: g["aggregate_ids"].update({"extra_gene": "CIG-1199"}), "exactly the frozen"),
    (lambda g, gene: g["aggregates"].pop(g["aggregate_ids"][gene]), "not its aggregate record's"),
    (lambda g, gene: next(c for c in g["children"].values() if c["stage"] == "ablation")[
        "measurements"]["bootstrap"].update({"lower_bound": 0.99}),
     "does not recompute"),  # ADR-0005; ADR-0009 D5
    (lambda g, gene: g["sensitivity"].pop(gene), "sensitivity grids are not the frozen"),
    (lambda g, gene: g["sensitivity"].update({"extra_gene": g["sensitivity"][gene]}),
     "sensitivity grids are not the frozen"),
    (lambda g, gene: g["sensitivity"][gene].update({"analysis_only": False}),
     "not the pre-planned analysis-only grid"),
    (lambda g, gene: g["sensitivity"][gene].update({"runner_calls": 40}),
     "not the pre-planned analysis-only grid"),
    (lambda g, gene: g["sensitivity"][gene].update({"runner_calls": False}),
     "not the pre-planned analysis-only grid"),
    (lambda g, gene: g["sensitivity"][gene].update({"tau_r_grid": [0.02, 0.03, 0.04]}),
     "not the pre-planned analysis-only grid"),
    (lambda g, gene: g["sensitivity"][gene].update({"tau_c_grid": [0.05]}),
     "not the pre-planned analysis-only grid"),
    (lambda g, gene: g["sensitivity"][gene].update({"gene_id": "someone_else"}),
     "not the pre-planned analysis-only grid"),
    (lambda g, gene: g["sensitivity"][gene]["cells"].pop("tau_c=0.05|tau_r=0.03"),
     "not the pre-planned analysis-only grid"),
    (lambda g, gene: g["aggregates"].update(  # ADR-0010 B4 on stored state
        {"CIG-9999": g["aggregates"][g["aggregate_ids"][gene]]}), "is stored as"),
    (lambda g, gene: g["children"].pop(next(iter(
        g["aggregates"][g["aggregate_ids"][gene]]["child_records"]))),
     "a declared record is missing or malformed"),
    (lambda g, gene: (g["verdicts"].update({gene: "reject"}),
                      g["aggregates"][g["aggregate_ids"][gene]].update({"verdict": "reject"}),
                      g["aggregates"][g["aggregate_ids"][gene]]["stages"]["replay"].update(
                          {"pass": False}),
                      g["children"][g["aggregate_ids"][gene] + "/S11-replay"][
                          "measurements"].update({"stage_pass": False}),
                      g["children"][g["aggregate_ids"][gene] + "/S11-replay"][
                          "measurements"]["episodes"][0].update({"expressed_gene_ids": []})),
     "in the somatic journal"),  # a coherent record the journal contradicts
    # ADR-0012 D3 recomputed on resume: a stage-1 pass off the expression rule
    (lambda g, gene: g["children"][g["aggregate_ids"][gene] + "/S11-replay"][
        "measurements"]["episodes"][0].update({"expressed_gene_ids": []}), "expression rule"),
    # ADR-0012 D2: every comparison read its own round's cell
    (lambda g, gene: g["children"][g["aggregate_ids"][gene] + "/S11-ablation"]["measurements"][
        "paired_outcomes"][0]["cell"].update({"trajectory_id": "T-000000000000"}),
     "did not read its round-R1 cell"),
    (lambda g, gene: g["children"][g["aggregate_ids"][gene] + "/S11-generalization_out_of_scope"][
        "measurements"]["paired_outcomes"][0]["cell"].update({"trajectory_id": "T-000000000000"}),
     "did not read its R5 cell"),
    (lambda g, gene: g["children"][g["aggregate_ids"][gene] + "/S11-interaction_regression"][
        "measurements"]["regression"]["paired_g0_run"]["episodes"][0].update(
        {"trajectory_id": "T-000000000000"}), "was not paired with its R5 cell"),
    (lambda g, gene: g["children"][g["aggregate_ids"][gene] + "/S11-generalization_out_of_scope"][
        "measurements"]["paired_outcomes"][0].pop("cell"), "a record is malformed"),
    # ADR-0012 D5: the in-scope child is stage 2's record by reference
    (lambda g, gene: g["children"][g["aggregate_ids"][gene] + "/S11-generalization_in_scope"][
        "measurements"]["by_reference"].update({"paired_outcomes_sha256": "0" * 64}),
     "by reference"),
    # ADR-0012 D7: the out-of-scope check and 4A ran iff stage 2 met alpha
    (lambda g, gene: g["children"][g["aggregate_ids"][gene] + "/S11-ablation"][
        "measurements"].update({"alpha_met": False}), "does not recompute"),
    # the stored records re-verify as a record store (ADR-0010 B4)
    (lambda g, gene: g["children"][g["aggregate_ids"][gene] + "/S11-replay"][
        "measurements"].update({"candidate_version": 2}), "candidate_version 2 != aggregate"),
    (lambda g, gene: g["aggregates"][g["aggregate_ids"][gene]]["artifacts"].update(
        {"base_genome": {"genome_id": "G-99999999"}}), "base_genome.genome_id 'G-99999999'"),
    (lambda g, gene: one_copy_of_a_pair(g, gene), "names partner"),
    (lambda g, gene: g["children"].update({"CIG-9999/S11-replay": {}}), "undeclared child"),
])
def test_an_inconsistent_gate_artifact_fails_closed(tmp_path, clean, splits, tamper, message):
    run_dir = reopened_copy(clean, tmp_path)
    path = run_dir / "seed-11" / "gate.json"
    gate = json.loads(path.read_text())
    promoted = next(g for g, v in gate["verdicts"].items() if v == "promote")
    tamper(gate, promoted)
    path.write_text(json.dumps(gate))
    with pytest.raises(RunError, match=message):
        make_run(run_dir, splits).run()


def eleven_candidates(document):
    from trajectory.storage.canonical import canonical_json, sha256_hex
    document["candidates"] = (document["candidates"] * 6)[:11]
    document["candidate_set_sha256"] = sha256_hex(
        canonical_json(document["candidates"]).encode("utf-8"))


@pytest.mark.parametrize("change,message", [
    (lambda d: d.update({"mining_seed": 23}), "not seed 11's frozen candidate set"),
    (lambda d: d.update({"candidates": d["candidates"][:1]}), "does not match its digest"),
    (eleven_candidates, "at most 10"),
])
def test_a_tampered_candidate_set_fails_closed_on_resume(tmp_path, clean, splits, change, message):
    run_dir = reopened_copy(clean, tmp_path)
    edit(run_dir / "seed-11" / "candidates.json", change)
    with pytest.raises(RunError, match=message):
        make_run(run_dir, splits).run()


def test_a_charged_mining_seed_without_its_set_fails_closed(tmp_path, clean, splits):
    run_dir = reopened_copy(clean, tmp_path)
    (run_dir / "seed-11" / "candidates.json").unlink()
    teacher = Teacher()
    with pytest.raises(RunError, match="charged but its frozen candidate set is missing"):
        make_run(run_dir, splits, teacher=teacher).run()
    assert teacher.calls == 0


class NeverWins(Agent):
    def generate(self, messages, settings):
        if settings.temperature > 0:
            settings = type(settings)(temperature=settings.temperature,
                                      max_tokens=settings.max_tokens, seed=1, stop=settings.stop)
        return super().generate(messages, settings)


def test_a_t_reg_shortfall_stops_the_run_and_a_crash_before_the_stop_is_finished(tmp_path, splits,
                                                                                  monkeypatch):
    run_dir = tmp_path / "run"

    def crash(self, reason):
        raise RuntimeError("killed before the stop record")

    with monkeypatch.context() as patch:
        patch.setattr(RunLedger, "abandon", crash)
        with pytest.raises(RuntimeError, match="killed before the stop record"):
            make_run(run_dir, splits, agent=NeverWins()).run()
    report = json.loads((run_dir / "report.json").read_text())
    assert report["status"] == "incomplete" and report["stage"] == "t_reg"
    assert "new experiment revision" in report["reason"]
    assert RunLedger(run_dir / "ledger.jsonl", run_id=RUN_ID, clock=Clock()).state == "running"
    assert make_run(run_dir, splits, agent=NeverWins()).run() == report
    ledger = RunLedger(run_dir / "ledger.jsonl", run_id=RUN_ID, clock=Clock())
    assert ledger.state == "incomplete" and ledger.reason == report["reason"]


def test_a_cap_stop_whose_report_was_lost_gets_a_rebuilt_report(tmp_path, splits):
    run_dir = tmp_path / "run"
    caps = {**CAPS, "episodes_max": 100}
    first = make_run(run_dir, splits, caps=caps).run()
    (run_dir / "report.json").unlink()  # the crash came after the stop record
    rebuilt = make_run(run_dir, splits, caps=caps).run()
    assert rebuilt["status"] == "incomplete" and rebuilt["reason"] == first["reason"]
    assert rebuilt["stage"].startswith("unknown") and rebuilt["statistics"] is None


def test_a_crash_while_freezing_the_gate_artifact_resumes_to_the_same_report(tmp_path, clean,
                                                                             splits, monkeypatch):
    import evaluation.exp0001.pipeline as pipeline
    real = pipeline.freeze_json
    crashed = []

    def crash_once(path, document):
        if path.name == "gate.json" and not crashed:
            crashed.append(path)
            raise OSError("disk full")
        return real(path, document)

    run_dir = tmp_path / "run"
    with monkeypatch.context() as patch:
        patch.setattr(pipeline, "freeze_json", crash_once)
        with pytest.raises(OSError, match="disk full"):
            make_run(run_dir, splits).run()
    assert not (run_dir / "seed-11" / "gate.json").exists()
    assert make_run(run_dir, splits).run() == clean[1]


def test_a_teacher_crash_reruns_the_batch_and_keeps_its_reservation(tmp_path, splits):
    class DyingTeacher(Teacher):
        def structured_generate(self, messages, settings, schema):
            if self.calls == 1:
                raise RuntimeError("teacher died")
            return super().structured_generate(messages, settings, schema)

    run_dir, caps = tmp_path / "run", {**CAPS, "episodes_max": 540 + 10}
    with pytest.raises(RuntimeError, match="teacher died"):
        make_run(run_dir, splits, teacher=DyingTeacher(), caps=caps).run()
    report = make_run(run_dir, splits, caps=caps).run()  # the reserve stops it before R1
    pairs = len(json.loads((run_dir / "seed-11" / "candidates.json").read_text())["mining_records"])
    assert report["stage"] == "gate/11" and report["aborted_attempts"] == 1
    assert "reserve: before gate/11/R1" in report["reason"] and report["episodes"] == 540
    # the aborted batch stays charged its reservation (two calls per pair, ADR-0009 D8);
    # the rerun is charged its calls (every pair fit in its full form)
    assert report["model_calls"] == 540 + 2 * pairs + pairs


def test_a_pair_sent_in_both_evidence_forms_is_charged_two_calls():
    from evaluation.exp0001.pipeline import teacher_calls
    assert teacher_calls([{"evidence_form": "full"},
                          {"evidence_form": "without-admissible-commands"}, {}]) == 1 + 2 + 1


@pytest.mark.parametrize("change,message", [
    (lambda r: r.update({"record_sha256": "0" * 64}), "record_sha256"),
    (lambda r: r["g0"].update({"genome_id": "G-0000000000000000"}), "another G0"),
    (lambda r: r["splits"].update({"manifest_sha256": "0" * 64}), "another split manifest"),
    # ADR-0012 D1, D10: a record of another schema or threshold version is another run
    (lambda r: r.update({"schema": "heritagent/exp0001-run-start/0.1"}), "not a heritagent"),
    (lambda r: r["gate"].update({"threshold_version": "EXP-0001-CIG-M0-v1"}), "not a heritagent"),
    (lambda r: r.pop("benefit_bank"), "with a benefit bank"),
    (lambda r: r["benefit_bank"].update({"bank_sha256": "0" * 64}), "another benefit bank"),
])
def test_the_run_binds_the_records_identity(tmp_path, splits, change, message):
    record = run_start(tmp_path, splits)
    change(record)
    if message != "record_sha256":
        record["record_sha256"] = canonical_sha256(
            {k: v for k, v in record.items() if k != "record_sha256"})
    with pytest.raises(ProvenanceError, match=message) as failed:
        make_run(tmp_path / "run", splits, record=record)
    # while the failure (and its half-built object) is still alive, the
    # directory is free: the constructor released its lock itself
    make_run(tmp_path / "run", splits).close()
    del failed


@pytest.mark.parametrize("name,field", [("model", "revision"), ("model", "backend"),
                                        ("teacher", "model_id"), ("teacher", "revision")])
def test_every_adapter_field_is_bound(tmp_path, splits, name, field):
    record = run_start(tmp_path, splits)
    record[name][field] = "something-else"
    record["record_sha256"] = canonical_sha256(
        {k: v for k, v in record.items() if k != "record_sha256"})
    with pytest.raises(ProvenanceError, match=f"the {name} adapter"):
        make_run(tmp_path / "run", splits, record=record)


def test_a_run_object_is_single_use(tmp_path, splits):
    first = make_run(tmp_path / "run", splits, caps={**CAPS, "episodes_max": 1})
    report = first.run()  # stops at the cap and releases the directory
    with pytest.raises(RunError, match="construct a new Exp0001Run"):
        first.run()  # its ledger view is stale: a resume needs a new object
    assert make_run(tmp_path / "run", splits, caps={**CAPS, "episodes_max": 1}).run() == report


def test_a_crash_between_the_report_and_the_complete_record_is_finished(tmp_path, clean, splits,
                                                                         monkeypatch):
    run_dir = reopened_copy(clean, tmp_path)

    def crash(self):
        raise RuntimeError("killed before the complete record")

    with monkeypatch.context() as patch:
        patch.setattr(RunLedger, "complete", crash)
        with pytest.raises(RuntimeError, match="complete record"):
            make_run(run_dir, splits).run()
    assert json.loads((run_dir / "report.json").read_text()) == clean[1]  # frozen first
    assert RunLedger(run_dir / "ledger.jsonl", run_id=RUN_ID, clock=Clock()).state == "running"
    agent = Agent()
    assert make_run(run_dir, splits, agent=agent).run() == clean[1] and agent.calls == 0
    assert RunLedger(run_dir / "ledger.jsonl", run_id=RUN_ID, clock=Clock()).state == "complete"


def test_a_crash_between_the_frozen_set_and_its_charge_is_charged_once(tmp_path, splits,
                                                                        monkeypatch):
    run_dir, caps = tmp_path / "run", {**CAPS, "episodes_max": 540 + 10}

    def crash(self, key, *, model_calls):
        raise RuntimeError("killed before the calls record")

    with monkeypatch.context() as patch:
        patch.setattr(RunLedger, "record_calls", crash)
        with pytest.raises(RuntimeError, match="calls record"):
            make_run(run_dir, splits, caps=caps).run()
    assert (run_dir / "seed-11" / "candidates.json").exists()  # frozen before the charge
    teacher = Teacher()
    report = make_run(run_dir, splits, teacher=teacher, caps=caps).run()
    pairs = len(json.loads((run_dir / "seed-11" / "candidates.json").read_text())["mining_records"])
    assert teacher.calls == 0 and report["aborted_attempts"] == 0
    assert report["model_calls"] == 540 + pairs  # the reservation became the charge


# -- parallel episodes (ADR-0008) ------------------------------------------------------
GOOD_TOO = GOOD + " Then look again."  # a second good skill: both are eligible, one pair
THREE = PROPOSALS[:1] + [proposal("check twice", GOOD_TOO, [F[3]], 0.6)] + PROPOSALS[1:]


def test_a_parallel_run_gives_the_sequential_report(tmp_path, clean, splits):
    run_dir = tmp_path / "run"
    report = make_run(run_dir, splits, workers=2).run()
    assert report == clean[1]
    assert sorted(ledger_keys(run_dir)) == sorted(ledger_keys(clean[0]))
    sessions = [json.loads(line) for line in (run_dir / "ledger.jsonl").read_text().splitlines()
                if json.loads(line)["type"] == "session"]
    assert [s["info"]["workers"] for s in sessions] == [2]


def test_the_gate_rounds_cover_interaction_pairs_and_match_sequential(tmp_path, splits):
    """Two candidates meet alpha and pass: R6 plays their pair's 10 both-trait
    games against the seed's R6 cells (ADR-0012 D6), and a parallel run
    gives the sequential one's report and ledger keys."""
    sequential = make_run(tmp_path / "seq", splits, teacher=Teacher(THREE)).run()
    parallel = make_run(tmp_path / "par", splits, teacher=Teacher(THREE), workers=2).run()
    assert parallel == sequential and sequential["status"] == "complete"
    keys = ledger_keys(tmp_path / "par")
    assert sorted(keys) == sorted(ledger_keys(tmp_path / "seq"))
    assert len([k for k in keys if "/R6/pairs/" in k and "/both/" in k]) == 3 * 10
    assert len([k for k in keys if "/R6/cell/" in k]) == 3 * 10
    for seed in MINING_SEEDS:
        gate = json.loads((tmp_path / "par" / f"seed-{seed}" / "gate.json").read_text())
        assert sorted(gate["verdicts"].values()) == ["promote", "promote", "reject"]
        r6 = gate["rounds"]["R6"]
        assert (r6["with_trait"], r6["cells"]) == (10, 10)
        assert set(r6["cell_readers"].values()) == {1}  # one pair reads each cell once
        assert [c["label"] for c in reserve_checks(tmp_path / "par")
                if c["label"].startswith(f"gate/{seed}/")][-1] == f"gate/{seed}/R6"


MARK_A, MARK_B = " Mark alpha.", " Mark beta."
PAIRED = PROPOSALS[:1] + [proposal("check alpha", GOOD + MARK_A, [F[3]], 0.6),
                          proposal("check beta", GOOD + MARK_B, [F[4]], 0.5)] + PROPOSALS[1:]


class PairAgent(Agent):
    """As Agent, except on the interaction games at temperature 0: G0 wins
    them, and the two marked skills together lose them. So the gate sees one
    harmful pair and two that are not."""

    def __init__(self, interaction_games):
        super().__init__()
        self.interaction_games = tuple(interaction_games)

    def generate(self, messages, settings):
        text = "\n".join(m.content for m in messages)
        if settings.temperature == 0 and any(f"solve {game}." in text
                                             for game in self.interaction_games):
            self.calls += 1
            self.settings.append(settings)
            win = not (MARK_A in text and MARK_B in text)
            return GenerationResult(text=react_action("act", {"command": "win"}) if win
                                    else react_final("Giving up."), finish_reason="stop")
        return super().generate(messages, settings)


def test_a_harmful_and_a_harmless_pair_bind_their_artifacts_end_to_end(tmp_path, splits):
    """ADR-0010 with real pair evidence (the Critic's merge condition): three
    eligible candidates per seed give one harmful pair and two harmless ones;
    every stored record verifies as a record store, a resume passes, and a
    tampered copy of the harmful pair's evidence stops the resumed run."""
    from evaluation.exp0001.allocations import interaction_allocation
    games = [t.task_id for t in interaction_allocation(splits).tasks]
    run_dir = tmp_path / "run"
    report = make_run(run_dir, splits, agent=PairAgent(games), teacher=Teacher(PAIRED)).run()
    assert report["status"] == "complete"
    for seed in MINING_SEEDS:
        gate = json.loads((run_dir / f"seed-{seed}" / "gate.json").read_text())
        assert sorted(gate["verdicts"].values()) == ["promote", "promote", "reject", "reject"]
        eliminated = {g: by for g, by in gate["eliminated_by_interaction"].items() if by}
        assert len(eliminated) == 1 and gate["verdicts"][next(iter(eliminated))] == "reject"
        pairs = {}
        for child in gate["children"].values():
            if child["stage"] == "interaction_regression":
                for pair in child["measurements"]["interaction"]["pair_evidence"]:
                    pairs.setdefault(pair["evidence_sha256"], []).append(pair)
                    gene_ids = pair["canonical_evidence"]["gene_ids"]
                    assert [a["gene_id"] for a in pair["canonical_evidence"]["artifacts"]] \
                        == gene_ids
        assert len(pairs) == 3 and all(len(copies) == 2 for copies in pairs.values())
        assert sorted(copies[0]["harmful"] for copies in pairs.values()) == [False, False, True]
        assert _record_store_issues(gate["aggregates"], gate["children"]) == []
    honest = reopened_copy((run_dir,), tmp_path / "honest")
    assert make_run(honest, splits, agent=PairAgent(games), teacher=Teacher(PAIRED)).run() \
        == report

    def harm_less(gate):
        for child in gate["children"].values():
            for pair in child["measurements"].get("interaction", {}).get("pair_evidence", []):
                if pair["harmful"]:
                    pair["canonical_evidence"]["delta_pair"] = 0.0
                    return

    tampered = reopened_copy((run_dir,), tmp_path / "tampered")
    edit(tampered / "seed-11" / "gate.json", harm_less)
    with pytest.raises(RunError, match="gate.json's CIG records do not verify"):
        make_run(tampered, splits, agent=PairAgent(games), teacher=Teacher(PAIRED)).run()


@pytest.mark.parametrize("kind", ["cell", "with_trait"])
def test_a_recorded_gate_episode_the_gate_never_requests_fails_closed_on_resume(
        tmp_path, splits, monkeypatch, kind):
    """A gate episode no request asked for (a defect a dry pass could cause),
    a cell or a with-trait execution, stops the run, and the check reads the
    ledger, so the resumed run stops too."""
    original = EpisodeExecutor.run_many
    leaked = []

    def leaky(self, specs):
        if not leaked and specs and specs[0].key.startswith("gate/"):
            spec = next(s for s in specs if ("/cell/" in s.key) == (kind == "cell"))
            leaked.append(dataclasses.replace(spec, key=spec.key.rsplit("/", 1)[0] + "/spec-0"))
            specs = list(specs) + leaked
        return original(self, specs)

    run_dir = tmp_path / "run"
    with monkeypatch.context() as patch:
        patch.setattr(EpisodeExecutor, "run_many", leaky)
        with pytest.raises(RunError, match="not requested by the gate"):
            make_run(run_dir, splits, workers=2).run()
    assert leaked[0].key.startswith("gate/11/R1/") and leaked[0].key in ledger_keys(run_dir)
    with pytest.raises(RunError, match="not requested by the gate"):
        make_run(run_dir, splits, workers=2).run()


def test_a_round_that_needs_more_episodes_after_its_queue_fails_closed(tmp_path, splits,
                                                                       monkeypatch):
    """ADR-0012 D2: a round is one queue. If the episodes the dry pass
    collected leave the round incomplete, the run stops for a human."""
    original = EpisodeExecutor.run_many

    def short(self, specs):
        if specs and specs[0].key.startswith("gate/11/R1/"):
            specs = list(specs)[:-1]  # one collected episode never runs
        return original(self, specs)

    monkeypatch.setattr(EpisodeExecutor, "run_many", short)
    with pytest.raises(RunError, match="round R1 needed 1 more episodes after its queue ran"):
        make_run(tmp_path / "run", splits).run()


def test_the_miner_stamps_the_ledger_after_every_teacher_call(clean):
    run_dir, _, _, teacher, _ = clean
    lines = [json.loads(line) for line in (run_dir / "ledger.jsonl").read_text().splitlines()]
    assert sum(r["type"] == "heartbeat" for r in lines) == teacher.calls > 0


def test_a_resume_on_another_serving_build_fails_closed(tmp_path, splits):
    other_build = {**SCRIPTED, "backend_version": "0.29.0+cu130", "tokenizer": Words.identity}
    with pytest.raises(ProvenanceError, match="model adapter"):
        make_run(tmp_path / "run", splits, record=run_start(tmp_path, splits, model=other_build))


# -- stage 2 beyond look 1, through the real rounds (ADR-0012 D2, D4, D7) ----------------
MARK2, MARK3, MARK4 = " Pass at look two.", " Pass at look three.", " Alpha at look four."
LOOKS_PROPOSALS = [proposal("look two", GOOD + MARK2, [F[0]], 0.9),
                   proposal("look three", GOOD + MARK3, [F[3]], 0.7),
                   proposal("look four", GOOD + MARK4, [F[4]], 0.6), PROPOSALS[1]]


def looks_wins(splits):
    """Per marked skill, the stage-2 games it wins, by allocation index (a
    one-family candidate's game j is its family's j-th bank game): 10 + 10
    pass at look 2 ((20, 0)); 10 pass at look 3 ((10, 0), point 10/144); 9
    meet alpha at look 4 ((9, 0), point 9/192 < tau_c: the stage fails)."""
    bank = bank_of(splits)["families"]
    return {MARK2: {bank[F[0]][j]["game_id"] for j in [*range(10), *range(48, 58)]},
            MARK3: {bank[F[3]][j]["game_id"] for j in range(10)},
            MARK4: {bank[F[4]][j]["game_id"] for j in range(9)}}


class WinSetAgent(Agent):
    """As Agent at temperature 0.7. At temperature 0 a marked skill wins only
    its chosen stage-2 games; G0, the bad skill and every other game lose."""

    def __init__(self, wins, fail_at=None):
        super().__init__(fail_at=fail_at)
        self.wins = wins

    def generate(self, messages, settings):
        if settings.temperature > 0:
            return super().generate(messages, settings)
        self.calls += 1
        self.settings.append(settings)
        if self.calls == self.fail_at:
            raise RuntimeError("backend died")
        text = "\n".join(m.content for m in messages)
        win = BAD not in text and any(
            marker in text and any(f"solve {game}." in text for game in games)
            for marker, games in self.wins.items())
        return GenerationResult(text=react_action("act", {"command": "win"}) if win
                                else react_final("Giving up."), finish_reason="stop")


@pytest.fixture(scope="module")
def looks(tmp_path_factory, splits):
    run_dir = tmp_path_factory.mktemp("looks") / "run"
    report = make_run(run_dir, splits, agent=WinSetAgent(looks_wins(splits)),
                      teacher=Teacher(LOOKS_PROPOSALS)).run()
    return run_dir, report


def by_marker(gate, run_dir):
    """gene id -> its marker (or 'bad'), read from the frozen candidate set."""
    seed = gate["mining_seed"]
    candidates = json.loads((run_dir / f"seed-{seed}" / "candidates.json").read_text())
    store = TraitRegistry(run_dir / "registry")
    names = {}
    for envelope in candidates["candidates"]:
        principle = json.loads(store.resolve(envelope["candidate"]["artifact"]))["principle"]
        names[envelope["candidate"]["gene_id"]] = next(
            (m for m in (MARK2, MARK3, MARK4) if m in principle), "bad")
    return names


def test_stage_2_plays_looks_two_to_four_in_their_rounds(looks):
    run_dir, report = looks
    assert report["status"] == "complete"
    keys = [k for k in ledger_keys(run_dir) if k.startswith("gate/")]
    for seed in MINING_SEEDS:
        gate = json.loads((run_dir / f"seed-{seed}" / "gate.json").read_text())
        marker = by_marker(gate, run_dir)
        stops, rounds_played = {}, {}
        for child in gate["children"].values():
            if child["stage"] != "ablation":
                continue
            m = child["measurements"]
            stops[marker[m["candidate_gene_id"]]] = m["stop"]
            # every game reads the cell of its own look's round, R1-R4
            assert [o["round"] for o in m["paired_outcomes"]] == \
                [o["look"] for o in m["paired_outcomes"]] == \
                [o["cell"]["round"] for o in m["paired_outcomes"]] == \
                [k for k, n in zip((1, 2, 3, 4), (48, 48, 48, 48)) for _ in range(n)][
                    :len(m["paired_outcomes"])]
            rounds_played[marker[m["candidate_gene_id"]]] = [look["round"] for look in m["looks"]]
        assert stops == {
            MARK2: {"look": 2, "round": 2, "games": 96, "decision": "pass", "reason": "efficacy"},
            MARK3: {"look": 3, "round": 3, "games": 144, "decision": "pass", "reason": "efficacy"},
            MARK4: {"look": 4, "round": 4, "games": 192, "decision": "fail",
                    "reason": "no efficacy at the last look"},
            "bad": {"look": 1, "round": 1, "games": 48, "decision": "fail", "reason": "futility"}}
        assert rounds_played == {MARK2: [1, 2], MARK3: [1, 2, 3], MARK4: [1, 2, 3, 4], "bad": [1]}
        assert sorted(gate["rounds"]) == ["R1", "R2", "R3", "R4", "R5", "R6"]
        assert (gate["rounds"]["R1"]["with_trait"], gate["rounds"]["R1"]["cells"]) == (196, 192)
        for name, games in (("R2", 144), ("R3", 96), ("R4", 48)):
            assert (gate["rounds"][name]["with_trait"], gate["rounds"][name]["cells"]) == \
                (games, games)
        assert (gate["rounds"]["R6"]["with_trait"], gate["rounds"]["R6"]["cells"]) == (10, 10)
        verdicts = {marker[g]: v for g, v in gate["verdicts"].items()}
        assert verdicts == {MARK2: "promote", MARK3: "promote", MARK4: "reject", "bad": "reject"}
        ran = [k for k in keys if k.startswith(f"gate/{seed}/")]
        rounds = [key_round(k) for k in ran]
        assert rounds == sorted(rounds) and set(rounds) == {1, 2, 3, 4, 5, 6}
        for gate_round in range(1, 7):
            assert_one_hash_ordered_queue([k for k in ran if key_round(k) == gate_round])
        assert [c["label"] for c in reserve_checks(run_dir)
                if c["label"].startswith(f"gate/{seed}/")] == \
            [f"gate/{seed}/R{r}" for r in range(1, 7)]
    assert report["gates"]["11"]["stage2_stops"] == {
        "fail:futility@look1": 1, "fail:no efficacy at the last look@look4": 1,
        "pass:efficacy@look2": 1, "pass:efficacy@look3": 1}


def test_alpha_met_without_a_pass_reaches_round_5(looks):
    """ADR-0012 D7: (9, 0) at look 4 meets alpha and fails the point rule;
    its out-of-scope check and 4A run in R5, its stage 2 fails it."""
    run_dir, report = looks
    for seed in MINING_SEEDS:
        gate = json.loads((run_dir / f"seed-{seed}" / "gate.json").read_text())
        marker = by_marker(gate, run_dir)
        gene = next(g for g, m in marker.items() if m == MARK4)
        children = {c["stage"]: c["measurements"] for c in gate["children"].values()
                    if c["measurements"].get("candidate_gene_id") == gene}
        assert children["ablation"]["alpha_met"] is True
        assert children["ablation"]["stage_pass"] is False
        assert children["generalization_out_of_scope"].get("not_run") is not True
        assert children["generalization_out_of_scope"]["episodes_run"] == 24
        assert children["interaction_regression"].get("not_run") is not True
        assert gate["verdicts"][gene] == "reject"
        assert report["gates"][str(seed)]["reached_round_5"] == 3
        assert report["gates"][str(seed)]["promoted"] == 2


def test_a_stricter_tau_c_that_would_continue_is_not_identifiable_end_to_end(looks):
    """ADR-0012 D7 in the frozen grid: the look-3 pass (point 10/144) would
    have continued under tau_c 0.08; under 0.03 the look-4 candidate becomes
    eligible and its pairs were never played."""
    run_dir, _ = looks
    gate = json.loads((run_dir / "seed-11" / "gate.json").read_text())
    marker = by_marker(gate, run_dir)
    look3 = next(g for g, m in marker.items() if m == MARK3)
    grid = gate["sensitivity"][look3]
    assert grid["ablation"]["tau_c=0.08"]["identifiable"] is False
    for tau_r in (0.02, 0.03, 0.05):
        strict = grid["cells"][f"tau_c=0.08|tau_r={tau_r}"]
        assert strict["identifiable"] is False and strict["unidentified_stage2"] == [look3]
        primary = grid["cells"][f"tau_c=0.05|tau_r={tau_r}"]
        assert primary["identifiable"] is True
        assert primary["hypothetical_promote"] == sorted(
            g for g, m in marker.items() if m in (MARK2, MARK3))
        loose = grid["cells"][f"tau_c=0.03|tau_r={tau_r}"]
        assert loose["identifiable"] is False and loose["missing_interaction_pairs"]


def test_a_crash_inside_r3_resumes_to_the_same_gate_artifacts(tmp_path, looks, splits):
    """A crash in seed 11's R3 queue: the resume re-checks R3's remaining
    queue (one more reserve record) and gives the same report and the same
    gate artifacts, byte for byte."""
    run_dir = tmp_path / "run"
    keys = ledger_keys(looks[0])
    fail_at = next(i for i, k in enumerate(keys, start=1) if k.startswith("gate/11/R3/")) + 4
    with pytest.raises(RuntimeError, match="backend died"):
        make_run(run_dir, splits, agent=WinSetAgent(looks_wins(splits), fail_at=fail_at),
                 teacher=Teacher(LOOKS_PROPOSALS)).run()
    assert not any(k.startswith("gate/11/R4/") for k in ledger_keys(run_dir))
    report = make_run(run_dir, splits, agent=WinSetAgent(looks_wins(splits)),
                      teacher=Teacher(LOOKS_PROPOSALS)).run()
    assert report["aborted_attempts"] == 1
    assert {**report, "aborted_attempts": 0, "model_calls": 0} == \
        {**looks[1], "aborted_attempts": 0, "model_calls": 0}
    for seed in MINING_SEEDS:
        assert (run_dir / f"seed-{seed}" / "gate.json").read_bytes() == \
            (looks[0] / f"seed-{seed}" / "gate.json").read_bytes()
    assert [c["label"] for c in reserve_checks(run_dir) if c["label"].startswith("gate/11/")] \
        == ["gate/11/R1", "gate/11/R2", "gate/11/R3", "gate/11/R3", "gate/11/R4", "gate/11/R5",
            "gate/11/R6"]


def _rehash_pairs(gate, change):
    """Apply ``change`` to every copy of every pair's canonical evidence and
    re-derive its evidence hash, so only the pipeline's own checks can object."""
    from trajectory.storage.canonical import canonical_json, sha256_hex
    for child in gate["children"].values():
        for pair in (child["measurements"].get("interaction") or {}).get("pair_evidence", ()):
            change(pair["canonical_evidence"])
            pair["evidence_sha256"] = sha256_hex(
                canonical_json(pair["canonical_evidence"]).encode("utf-8"))


def _neither_off_its_cell(canonical):
    for outcome in canonical["paired_outcomes"]:
        if outcome["condition"] == "neither":
            outcome["trajectory_id"] = "T-000000000000"
            return


def _over_another_family(gate, run_dir):
    """MARK2's stage 2 rewritten consistently onto another family, its
    in-scope reference re-derived: only the frozen applicability objects."""
    from inheritance.generalization import paired_outcomes_sha256
    stage2 = _child_of(gate, run_dir, MARK2, "ablation")
    stage2["allocation"]["families"] = [F[1]]
    for outcome in stage2["paired_outcomes"]:
        outcome["family"] = F[1]
    _child_of(gate, run_dir, MARK2, "generalization_in_scope")["by_reference"][
        "paired_outcomes_sha256"] = paired_outcomes_sha256(stage2["paired_outcomes"])


def _cell_of_another_seed(gate, run_dir):
    """A stage-2 cell relabelled to another seed, its in-scope reference
    re-derived: only the seed binding of the recompute objects."""
    from inheritance.generalization import paired_outcomes_sha256
    stage2 = _child_of(gate, run_dir, MARK2, "ablation")
    stage2["paired_outcomes"][0]["cell"]["seed"] = 99
    _child_of(gate, run_dir, MARK2, "generalization_in_scope")["by_reference"][
        "paired_outcomes_sha256"] = paired_outcomes_sha256(stage2["paired_outcomes"])


def _child_of(gate, run_dir, mark, stage):
    marker = by_marker(gate, run_dir)
    gene = next(g for g, m in marker.items() if m == mark)
    return gate["children"][f"{gate['aggregate_ids'][gene]}/S11-{stage}"]["measurements"]


@pytest.mark.parametrize("tamper,message", [
    (lambda g, d: _rehash_pairs(g, _neither_off_its_cell), "is not its R6 cell"),
    (lambda g, d: _child_of(g, d, MARK4, "generalization_out_of_scope").update(not_run=True),
     "must run iff stage 2 met alpha"),
    (lambda g, d: _child_of(g, d, MARK4, "interaction_regression").update(not_run=True),
     "must run iff stage 2 met alpha"),
    (lambda g, d: _child_of(g, d, "bad", "generalization_out_of_scope").pop("not_run"),
     "must run iff stage 2 met alpha"),
    (lambda g, d: _child_of(g, d, MARK2, "generalization_in_scope")["by_reference"].update(
        child_id="CIG-1104/S11-ablation"), "by reference"),
    (lambda g, d: _child_of(g, d, MARK2, "ablation")["allocation"].update(
        bank_sha256="0" * 64), "benefit bank"),
    (lambda g, d: _over_another_family(g, d), "frozen applicability"),
    (lambda g, d: _cell_of_another_seed(g, d), "a cell of seed 99"),
    (lambda g, d: _child_of(g, d, MARK4, "generalization_out_of_scope")["allocation"].update(
        bank_sha256="0" * 64), "benefit bank"),
])
def test_a_tampered_looks_gate_artifact_fails_closed_on_resume(tmp_path, looks, splits, tamper,
                                                              message):
    run_dir = reopened_copy(looks, tmp_path)
    edit(run_dir / "seed-11" / "gate.json", lambda gate: tamper(gate, run_dir))
    with pytest.raises(RunError, match=message):
        make_run(run_dir, splits, agent=WinSetAgent(looks_wins(splits)),
                 teacher=Teacher(LOOKS_PROPOSALS)).run()


def test_an_honest_resume_of_the_looks_world_gives_its_report(tmp_path, looks, splits):
    run_dir = reopened_copy(looks, tmp_path)
    agent = WinSetAgent(looks_wins(splits))
    assert make_run(run_dir, splits, agent=agent, teacher=Teacher(LOOKS_PROPOSALS)).run() \
        == looks[1]
    assert agent.calls == 0  # every gate and final episode is read back


def test_an_evaluator_that_cannot_be_built_stops_the_gate_before_its_first_episode(
        tmp_path, splits, monkeypatch):
    """Every evaluator is built before the first request, and an error with
    no placeholder answered is real: the run fails before any gate episode."""
    import evaluation.exp0001.pipeline as pipeline
    from inheritance.overlay import EvaluationSetupError
    real = pipeline.AblationEvaluator

    def refuses_the_bad_skill(**kwargs):
        if kwargs["candidate_envelope"]["candidate"]["gene_id"].startswith("give_up"):
            raise EvaluationSetupError("cannot build this candidate's stage 2")
        return real(**kwargs)

    monkeypatch.setattr(pipeline, "AblationEvaluator", refuses_the_bad_skill)
    run_dir = tmp_path / "run"
    with pytest.raises(EvaluationSetupError, match="cannot build"):
        make_run(run_dir, splits).run()
    assert not [k for k in ledger_keys(run_dir) if k.startswith("gate/")]
