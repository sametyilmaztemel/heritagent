"""The EXP-0001 run end to end on scripted components (issue #13 Phase A):
a synthetic split manifest with the locked pool sizes, scripted ALFWorld
games, a scripted agent model and a scripted teacher. No GPU, no ALFWorld,
no model inference. The outcomes below are the scripted world's, not
EXP-0001 results."""

import json

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
    generalization_allocation,
    generalization_rank,
    pool_tasks,
    replay_allocation,
)
from evaluation.exp0001.episodes import EpisodeExecutor
from evaluation.exp0001.gate_runner import LedgerGateRunner
from evaluation.exp0001.ledger import RunLedger
from evaluation.exp0001.pilot import build_pilot_manifest
from evaluation.exp0001.pipeline import Exp0001Run, RunError, cig_id
from evaluation.exp0001.protocol import CAPS, MINING_SEEDS
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


@pytest.fixture(scope="module")
def splits(tmp_path_factory):
    root = tmp_path_factory.mktemp("alfworld")
    train = collect_games(make_split(root, "train", {f: 40 for f in F}), "train")
    unseen_dir = root / "valid_unseen"
    for family, n in UNSEEN_PER_FAMILY.items():  # unseen trials never share a train id
        for i in range(n):
            write_game(unseen_dir, family, 500 + i)
    unseen = collect_games(unseen_dir, "valid_unseen")
    manifest = build_split_manifest(train, unseen, full_plan())
    PILOTS[manifest["manifest_sha256"]] = build_pilot_manifest(train, manifest)
    return manifest


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
        pilot_manifest=PILOTS[splits["manifest_sha256"]],
        g0_genome=build_g0_seed_genome(TraitRegistry(tmp_path / "g0-registry")),
        run_id=RUN_ID, started_at="2026-10-20T09:00:00Z",
        expected_split_sha256=splits["manifest_sha256"],
        expected_pilot_sha256=PILOTS[splits["manifest_sha256"]]["pilot_sha256"])
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
        split_manifest=splits, model=agent or Agent(), teacher=teacher or Teacher(),
        env_factory=env_factory,
        token_counter=counter or Words(), ledger_clock=Clock(), episode_clock=Clock(),
        sleep=lambda _: None, expected_split_sha256=splits["manifest_sha256"],
        workers=workers, sleep_between_retries=False, on_ledger_append=on_ledger_append,
        lock=lock)


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


def test_a_scripted_run_goes_end_to_end(clean, splits):
    run_dir, report, agent, teacher, closed = clean
    assert report["status"] == "complete" and report["aborted_attempts"] == 0
    # 540 lifetime + per seed 2 candidates x 130 gate episodes and the shared G0 run on
    # T_reg (40, ADR-0009 D1) + 194 final episodes per distinct genome (ADR-0009 D2): the
    # scripted teacher proposes the same skills for every seed, so B and C are one genome each
    assert report["episodes"] == 540 + 3 * (260 + 40) + 3 * (134 + 40 + 20) == agent.calls \
        == closed
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
        bad_aggregate = gate["aggregates"][cig_id(seed, 1)]
        assert gate["children"][f"{cig_id(seed, 1)}/S{seed}-replay"]["measurements"][
            "stage_pass"] is False
        assert bad_aggregate["cig_id"] == cig_id(seed, 1)
        assert gate["aggregate_ids"] == {good: cig_id(seed, 0), bad: cig_id(seed, 1)}
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
    # ADR-0009 D5: what each gate decided, and the scope of the H3 preview
    assert report["gates"] == {str(seed): {"candidates": 2, "promoted": 1,
                                           "all_family_candidates": 0} for seed in MINING_SEEDS}
    assert "stage-2 ablation pool" in report["adaptation_cost"]["scope"]
    # the pre-planned sensitivity grid, frozen in each gate artifact (analysis only)
    for seed in MINING_SEEDS:
        grids = report["sensitivity"][str(seed)]
        gate = json.loads((run_dir / f"seed-{seed}" / "gate.json").read_text())
        assert grids == gate["sensitivity"] and set(grids) == set(gate["verdicts"])
        for grid in grids.values():
            assert grid["analysis_only"] is True and grid["runner_calls"] == 0
            assert len(grid["cells"]) == 9


def test_the_generation_settings_are_the_protocols(clean):
    """ADR-0007 E1/E8: lifetime at 0.7 with the attempt seeds, gate at 0 with
    the mining seed, final evaluation at 0 with no seed; the miner at 0.7
    with the mining seed."""
    _, _, agent, teacher, _ = clean
    lifetime = [s for s in agent.settings if s.temperature == 0.7]
    gate = [s for s in agent.settings if s.temperature == 0.0 and s.seed is not None]
    final = [s for s in agent.settings if s.temperature == 0.0 and s.seed is None]
    assert len(lifetime) == 540 and len(gate) == 3 * 300 and len(final) == 3 * 194
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
    import copy
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
                   split_manifest=tampered, model=Agent(), teacher=Teacher(),
                   env_factory=None, token_counter=Words(),
                   expected_split_sha256=splits["manifest_sha256"])
    with pytest.raises(ProvenanceError, match="not the pre-registered one"):
        Exp0001Run(run_dir=tmp_path / "d", run_start=run_start(tmp_path, splits),
                   split_manifest=splits, model=Agent(), teacher=Teacher(),
                   env_factory=None, token_counter=Words())  # the default: the committed hash
    assert not any((tmp_path / name).exists() for name in "abcd")  # nothing was written


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
    report = make_run(tmp_path / "run", splits, teacher=Teacher([proposal(long_name, GOOD, [F[0]],
                                                                          0.9)]),
                      caps={**CAPS, "episodes_max": 540 + 30}).run()
    assert report["status"] == "incomplete" and report["stage"] == "gate/11"
    assert "episodes_max" in report["reason"]


def test_a_hard_cap_stops_the_run_incomplete(tmp_path, splits):
    run_dir = tmp_path / "run"
    caps = {**CAPS, "episodes_max": 100}
    report = make_run(run_dir, splits, caps=caps).run()
    assert report["status"] == "incomplete" and report["stage"] == "lifetime"
    assert "episodes_max 100" in report["reason"] and report["statistics"] is None
    ledger = RunLedger(run_dir / "ledger.jsonl", run_id=RUN_ID, clock=Clock(), caps=caps)
    assert ledger.state == "incomplete" and ledger.episodes == 100
    assert make_run(run_dir, splits, caps=caps).run() == report  # stays stopped


def test_a_cap_in_the_final_evaluation_gives_no_verdict(tmp_path, splits):
    report = make_run(tmp_path / "run", splits,
                      caps={**CAPS, "episodes_max": 540 + 3 * 300 + 50}).run()
    assert report["status"] == "incomplete" and report["stage"] == "final"
    stats = report["statistics"]
    assert stats["status"] == "incomplete" and stats["h2"] is None and stats["h1"] is None


def test_a_candidate_for_every_family_gets_a_vacuous_out_of_scope_stage(tmp_path, splits):
    """ADR-0007 E6 (owner): no out-of-scope family exists, so that stage is
    vacuous; the 10 in-scope games cover every family."""
    universal = [proposal("everywhere", GOOD, F, 0.9)]
    run_dir = tmp_path / "run"
    agent = Agent()
    report = make_run(run_dir, splits, agent=agent, teacher=Teacher(universal)).run()
    assert report["status"] == "complete"
    # per seed: replay 10 + ablation 40 + in-scope 20 + regression 40 + the shared G0 run
    # 40; the final evaluation runs G0 and the one G1 genome every B and C arm shares
    assert report["episodes"] == 540 + 3 * 150 + 2 * (134 + 40 + 20) == agent.calls
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
        plan = inside["measurements"]["allocation"]["in_scope_family_plan"]
        assert plan == even_split(10, F) and set(plan) == set(F)
    assert report["statistics"]["h2"]["rule_met"] is False  # B equals C in this world


def test_the_generalization_split_and_draw(tmp_path, splits):
    assert even_split(10, ["b", "a", "c"]) == {"a": 4, "b": 3, "c": 3}
    assert even_split(10, F[:1]) == {F[0]: 10}
    store = TraitRegistry(tmp_path / "store")
    payload = {**PROPOSALS[1], "applicability": {"task_families": [F[1], F[2]]}}
    uri = store.put("skill", "give_up", 1, payload)
    envelope = {"candidate": {"gene_id": "give_up", "version": 1, "type": "skill",
                              "origin": "acquired", "artifact": uri,
                              "provenance": {"born_generation": 0,
                                             "source_trajectory": "T-000000000000"}},
                "validation": {"state": "candidate"}}
    allocation = generalization_allocation(splits, envelope, store)
    assert dict(allocation.frozen_in_scope_plan) == {F[1]: 5, F[2]: 5}
    assert dict(allocation.frozen_out_scope_plan) == even_split(10, [F[0], *F[3:]])
    bank = [g for g in splits["pools"]["generalization_bank"]["games"] if g["family"] == F[1]]
    first5 = sorted(bank, key=lambda g: generalization_rank(F[1], g["game_id"]))[:5]
    assert [t.task_id for t in allocation.in_scope_tasks if t.family == F[1]] == \
        [g["game_id"] for g in first5]
    assert generalization_allocation(splits, envelope, store) == allocation  # fixed draw
    uri = store.put("skill", "everywhere", 1, {**payload, "applicability": {"task_families": list(F)}})
    vacuous = generalization_allocation(splits, {**envelope, "candidate": {
        **envelope["candidate"], "gene_id": "everywhere", "artifact": uri}}, store)
    assert vacuous.out_scope_vacuous and dict(vacuous.frozen_out_scope_plan) == {}
    assert dict(vacuous.frozen_in_scope_plan) == even_split(10, F)
    with pytest.raises(AllocationError, match="outside ALFWorld"):
        uri = store.put("skill", "elsewhere", 1, {**payload, "applicability": {
            "task_families": ["moon_landing"]}})
        generalization_allocation(splits, {**envelope, "candidate": {
            **envelope["candidate"], "gene_id": "elsewhere", "artifact": uri}}, store)


def test_replay_takes_the_source_familys_bank(splits):
    source = pool_tasks(splits, "t_mine")[0]
    allocation = replay_allocation(splits, source)
    assert allocation.source_task == source
    assert {t.family for t in allocation.bank_variants} == {source.family}
    assert len(allocation.bank_variants) == 4
    assert allocation.replay_bank_id == f"replay_bank[{source.family}]"


def test_the_gate_runner_keys_and_contract(tmp_path, registry):  # noqa: F811
    from evaluation.exp0001.episodes import protocol_budgets
    from inheritance.overlay import evaluation_settings
    budgets = protocol_budgets()
    ledger = RunLedger(tmp_path / "ledger.jsonl", run_id=RUN_ID, clock=Clock())
    executor = EpisodeExecutor(model=Agent(), env_factory=lambda s, g: AlfworldEnvironment(
        Game(g), game_id=g), ledger=ledger, trajectories_dir=tmp_path / "t", run_id=RUN_ID,
        code_commit="a" * 40, clock=Clock(), sleep=lambda _: None)
    runner = LedgerGateRunner(executor, "gate/11/x/replay", budgets)
    from inheritance.runner import EvaluationTask
    task = EvaluationTask("g1", F[0], payload={"split": "train"})
    config, settings = g0_runtime_config(registry), evaluation_settings(seed=11)
    first = runner.run_episode(task, config, budgets, "without_trait", settings)
    second = runner.run_episode(task, config, budgets, "without_trait", settings)
    assert first.trajectory_id != second.trajectory_id and ledger.episodes == 2
    assert first.effective_settings == settings and first.success is False
    with pytest.raises(RunnerContractError, match="budgets"):
        runner.run_episode(task, config, Budgets(max_steps=4, max_total_retries=1), "with_trait",
                           settings)
    with pytest.raises(RunnerContractError, match="temperature 0"):
        runner.run_episode(task, config, budgets, "with_trait",
                           evaluation_settings(seed=11).__class__(temperature=0.7))
    with pytest.raises(RunnerContractError, match="split"):
        runner.run_episode(EvaluationTask("g2", F[0]), config, budgets, "with_trait", settings)
    for payload in ({"split": "valid_unseen"}, {"split": "train", "pool": "t_eval"}):
        with pytest.raises(RunnerContractError, match="T_eval never enters the gate"):
            runner.run_episode(EvaluationTask("g3", F[0], payload=payload), config, budgets,
                               "with_trait", settings)
    assert ledger.episodes == 2


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
    (lambda g, gene: (g["verdicts"].update({gene: "reject"}),
                      g["aggregates"][g["aggregate_ids"][gene]].update({"verdict": "reject"})),
     "in the somatic journal"),
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
    report = make_run(run_dir, splits, caps=caps).run()  # stops at the cap in seed 11's gate
    pairs = len(json.loads((run_dir / "seed-11" / "candidates.json").read_text())["mining_records"])
    assert report["stage"] == "gate/11" and report["aborted_attempts"] == 1
    # the aborted batch stays charged its reservation (two calls per pair, ADR-0009 D8);
    # the rerun is charged its calls (every pair fit in its full form)
    assert report["model_calls"] == 540 + 10 + 2 * pairs + pairs


def test_a_pair_sent_in_both_evidence_forms_is_charged_two_calls():
    from evaluation.exp0001.pipeline import teacher_calls
    assert teacher_calls([{"evidence_form": "full"},
                          {"evidence_form": "without-admissible-commands"}, {}]) == 1 + 2 + 1


@pytest.mark.parametrize("change,message", [
    (lambda r: r.update({"record_sha256": "0" * 64}), "record_sha256"),
    (lambda r: r["g0"].update({"genome_id": "G-0000000000000000"}), "another G0"),
    (lambda r: r["splits"].update({"manifest_sha256": "0" * 64}), "another split manifest"),
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
    assert report["model_calls"] == 540 + 10 + pairs  # the reservation became the charge


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


def test_the_gate_prefetch_covers_interaction_pairs_and_matches_sequential(tmp_path, splits,
                                                                          monkeypatch):
    sequential = make_run(tmp_path / "seq", splits, teacher=Teacher(THREE)).run()
    one_at_a_time = []  # what the real gate pass finds: every episode already recorded
    original = EpisodeExecutor.run

    def run(self, spec):
        result = original(self, spec)
        one_at_a_time.append(result.resumed)
        return result

    monkeypatch.setattr(EpisodeExecutor, "run", run)
    parallel = make_run(tmp_path / "par", splits, teacher=Teacher(THREE), workers=2).run()
    assert parallel == sequential and sequential["status"] == "complete"
    assert len(one_at_a_time) == 3 * (3 * 130 + 20 + 40) and all(one_at_a_time)
    keys = ledger_keys(tmp_path / "par")
    assert sorted(keys) == sorted(ledger_keys(tmp_path / "seq"))
    pair_keys = [k for k in keys if "/both/" in k or "/neither/" in k]
    assert len(pair_keys) == 3 * 20  # one eligible pair per seed, 10 tasks x both/neither
    gate = json.loads((tmp_path / "par" / "seed-11" / "gate.json").read_text())
    assert sorted(gate["verdicts"].values()) == ["promote", "promote", "reject"]


def test_a_recorded_gate_episode_the_gate_never_requests_fails_closed_on_resume(
        tmp_path, splits, monkeypatch):
    """A speculative gate episode (a defect a dry pass could cause) stops the
    run, and the check reads the ledger, so the resumed run stops too."""
    import dataclasses
    original = Exp0001Run._prefetch
    leaked = []

    def leaky(self, dry_pass, collect):
        if not leaked:
            collected = []
            try:
                dry_pass((collected, collect))
            except Exception:  # noqa: BLE001
                pass
            spec = collected[0]
            leaked.append(dataclasses.replace(spec, key=spec.key.rsplit("/", 1)[0] + "/spec-0"))
            self.executor.run_many(leaked)
        original(self, dry_pass, collect)

    run_dir = tmp_path / "run"
    with monkeypatch.context() as patch:
        patch.setattr(Exp0001Run, "_prefetch", leaky)
        with pytest.raises(RunError, match="not requested by the gate"):
            make_run(run_dir, splits, workers=2).run()
    assert leaked[0].key.startswith("gate/11/") and leaked[0].key in ledger_keys(run_dir)
    with pytest.raises(RunError, match="not requested by the gate"):
        make_run(run_dir, splits, workers=2).run()


def test_the_miner_stamps_the_ledger_after_every_teacher_call(clean):
    run_dir, _, _, teacher, _ = clean
    lines = [json.loads(line) for line in (run_dir / "ledger.jsonl").read_text().splitlines()]
    assert sum(r["type"] == "heartbeat" for r in lines) == teacher.calls > 0


def test_a_resume_on_another_serving_build_fails_closed(tmp_path, splits):
    other_build = {**SCRIPTED, "backend_version": "0.29.0+cu130", "tokenizer": Words.identity}
    with pytest.raises(ProvenanceError, match="model adapter"):
        make_run(tmp_path / "run", splits, record=run_start(tmp_path, splits, model=other_build))
