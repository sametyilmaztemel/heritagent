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
from evaluation.exp0001.pipeline import Exp0001Run, cig_id
from evaluation.exp0001.protocol import CAPS, MINING_SEEDS
from evaluation.exp0001.provenance import build_run_start_record
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
from tests.evaluation.test_exp0001_artifacts import load
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
        self.calls, self.fail_at = 0, fail_at

    def generate(self, messages, settings):
        self.calls += 1
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
        self.proposals, self.calls = proposals, 0

    def structured_generate(self, messages, settings, schema):
        self.calls += 1
        text = json.dumps({"proposals": self.proposals}, sort_keys=True)
        self._validate_structured(text, schema)
        return GenerationResult(text=text, finish_reason="stop")


class Words:
    identity = "scripted-word-counter-v1"

    def count_tokens(self, text):
        return len(text.split())


@pytest.fixture(scope="module")
def splits(tmp_path_factory):
    root = tmp_path_factory.mktemp("alfworld")
    train = collect_games(make_split(root, "train", {f: 40 for f in F}), "train")
    unseen_dir = root / "valid_unseen"
    for family, n in UNSEEN_PER_FAMILY.items():  # unseen trials never share a train id
        for i in range(n):
            write_game(unseen_dir, family, 500 + i)
    unseen = collect_games(unseen_dir, "valid_unseen")
    return build_split_manifest(train, unseen, full_plan())


def run_start(tmp_path, splits):
    return build_run_start_record(
        code_commit="a" * 40, code_dirty=False,
        model={"model_id": "scripted", "revision": "r", "backend": "scripted",
               "backend_version": None, "tokenizer": "scripted-words"},
        weights={"sha256": "b" * 64}, split_manifest=splits,
        pilot_manifest=load("EXP-0001-pilot.json"),
        g0_genome=build_g0_seed_genome(TraitRegistry(tmp_path / "g0-registry")),
        run_id=RUN_ID, started_at="2026-10-20T09:00:00Z")


def make_run(run_dir, splits, *, agent=None, teacher=None, caps=CAPS):
    return Exp0001Run(
        run_dir=run_dir, run_start=run_start(run_dir.parent, splits), split_manifest=splits,
        model=agent or Agent(), teacher=teacher or Teacher(),
        env_factory=lambda split, game_id: AlfworldEnvironment(Game(game_id), game_id=game_id),
        token_counter=Words(), ledger_clock=Clock(), episode_clock=Clock(),
        sleep=lambda _: None, caps=caps)


def ledger_keys(run_dir):
    lines = (run_dir / "ledger.jsonl").read_text(encoding="utf-8").splitlines()
    return [r["key"] for r in map(json.loads, lines) if r.get("type") == "episode"]


@pytest.fixture(scope="module")
def clean(tmp_path_factory, splits):
    run_dir = tmp_path_factory.mktemp("clean") / "run"
    agent, teacher = Agent(), Teacher()
    report = make_run(run_dir, splits, agent=agent, teacher=teacher).run()
    return run_dir, report, agent, teacher


def test_a_scripted_run_goes_end_to_end(clean, splits):
    run_dir, report, agent, teacher = clean
    assert report["status"] == "complete"
    # 540 lifetime + per seed 2 candidates x 130 gate episodes + 7 x 194 final
    assert report["episodes"] == 540 + 3 * 260 + 7 * (134 + 40 + 20) == agent.calls
    assert report["model_calls"] == agent.calls + teacher.calls
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
    stats = report["statistics"]
    assert stats["status"] == "complete"
    assert stats["t_eval"]["contrasts"]["C-B"]["point_estimate"] == {"exact": "1/1", "float": 1.0}
    assert stats["h2"]["rule_met"] is True and stats["h1"]["rule_met"] is True
    assert report["genome_sizes"]["per_arm"]["A"]["runs"]["once"]["skill_genes"] == 0
    assert report["adaptation_cost"]["per_arm"]["C"]["pooled"]["solved"] == 60


def report_g0(run_dir):
    return json.loads((run_dir / "run-start.json").read_text())["g0"]["genome_id"]


def test_t_eval_is_touched_only_by_the_final_evaluation(clean, splits):
    run_dir = clean[0]
    t_eval = {t.task_id for t in pool_tasks(splits, "t_eval")}
    for key in ledger_keys(run_dir):
        if any(f"/{task_id}" in key for task_id in t_eval):
            assert key.startswith("final/") and "/t_eval/" in key


def test_a_finished_run_is_read_back_not_rerun(clean, splits):
    run_dir, report, _, _ = clean
    agent, teacher = Agent(), Teacher()
    assert make_run(run_dir, splits, agent=agent, teacher=teacher).run() == report
    assert agent.calls == teacher.calls == 0


def test_a_crashed_run_resumes_to_the_same_report(tmp_path, clean, splits):
    run_dir = tmp_path / "run"
    with pytest.raises(RuntimeError, match="backend died"):
        make_run(run_dir, splits, agent=Agent(fail_at=700)).run()  # in seed 11's gate
    assert (run_dir / "seed-11" / "candidates.json").exists()
    assert not (run_dir / "seed-11" / "gate.json").exists()
    seed11_pairs = len(json.loads((run_dir / "seed-11" / "candidates.json").read_text())[
        "mining_records"])  # one teacher call per pair
    teacher = Teacher()
    report = make_run(run_dir, splits, teacher=teacher).run()
    assert teacher.calls == clean[3].calls - seed11_pairs  # seed 11 is not mined again
    assert report == clean[1]


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
                      caps={**CAPS, "episodes_max": 540 + 3 * 260 + 50}).run()
    assert report["status"] == "incomplete" and report["stage"] == "final"
    stats = report["statistics"]
    assert stats["status"] == "incomplete" and stats["h2"] is None and stats["h1"] is None


def test_a_candidate_for_every_family_fails_closed_before_the_gate(tmp_path, splits):
    universal = [proposal("everywhere", GOOD, F, 0.9)]
    run_dir = tmp_path / "run"
    report = make_run(run_dir, splits, teacher=Teacher(universal)).run()
    assert report["status"] == "incomplete" and report["stage"] == "gate/11"
    assert "no out-of-scope family" in report["reason"]
    assert not [k for k in ledger_keys(run_dir) if k.startswith("gate/")]


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
    with pytest.raises(AllocationError, match="no out-of-scope family"):
        uri = store.put("skill", "everywhere", 1, {**payload,
                                                      "applicability": {"task_families": list(F)}})
        generalization_allocation(splits, {**envelope, "candidate": {
            **envelope["candidate"], "gene_id": "everywhere", "artifact": uri}}, store)


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

