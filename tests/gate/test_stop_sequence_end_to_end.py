"""ADR-0003 A2 end to end (own review of #62). The ReAct protocol stop must
be the same in every stage: what each evaluator hands its runner, what
each child record stores, and what the gate accepts as stage 1–2
evidence. The first version of A2 changed the evaluators but left the
gate requiring stop == [] and regression running without the stop; the
suite stayed green because the gate tests hand-build their children.
These tests feed real evaluator output through the gate."""

from inheritance.ablation import AblationEvaluator
from inheritance.gate import CigGateOrchestrator, GateCandidateInput
from inheritance.overlay import evaluation_settings
from inheritance.regression import RegressionEvaluator
from inheritance.replay import ReplayEvaluator
from runtime.loop import run_agent
from runtime.model_adapters import ScriptedAdapter
from runtime.policies.planner_react import STOP_SEQUENCES
from runtime.seed_g0 import g0_runtime_config
from tests.gate.helpers import BUDGETS
from tests.gate.test_gate_orchestrator import build_world
from tests.inheritance.conftest import MINING_SEED, PARENT_CIG
from tests.inheritance.conftest import base_config, envelope, registry  # noqa: F401 (fixtures)
from tests.inheritance.scripted_runner import ScriptedRunner
from tests.inheritance.test_ablation import make_allocation as ablation_allocation
from tests.inheritance.test_ablation import with_gene
from tests.inheritance.test_replay import make_allocation as replay_allocation
from tests.inheritance.test_replay import outcomes_for
from tests.runtime.conftest import ScriptedEnv, react_final

LOCKED = evaluation_settings(max_tokens=BUDGETS.max_tokens_per_request, seed=MINING_SEED)


def test_real_replay_and_ablation_children_pass_the_gate_check(tmp_path, registry,
                                                              base_config, envelope):
    replay_evaluator = ReplayEvaluator(
        candidate_envelope=envelope, registry=registry, base_config=base_config,
        runner=ScriptedRunner(), budgets=BUDGETS, parent_cig_id=PARENT_CIG,
        mining_seed=MINING_SEED)
    replay_evaluator._runner.outcomes = outcomes_for([True] * 5, [False] * 5)
    replay = replay_evaluator.evaluate(replay_allocation(tmp_path, registry))
    ablation_evaluator = AblationEvaluator(
        candidate_envelope=envelope, registry=registry, base_config=base_config,
        runner=ScriptedRunner(), budgets=BUDGETS, parent_cig_id=PARENT_CIG,
        mining_seed=MINING_SEED)
    ablation_evaluator._runner.outcomes = {
        (f"abl-{i}", condition): (condition == "with_trait" or i % 2 == 0,
                                  with_gene(condition == "with_trait"))
        for i in range(20) for condition in ("with_trait", "without_trait")}
    ablation = ablation_evaluator.evaluate(ablation_allocation(tmp_path, registry))
    for evaluation in (replay, ablation):
        recorded = evaluation.child_record["measurements"]["eval_generation_settings"]
        assert recorded["stop"] == list(STOP_SEQUENCES)

    gate = CigGateOrchestrator(registry=registry, base_config=base_config,
                               runner=ScriptedRunner(), budgets=BUDGETS,
                               parent_cig_id=PARENT_CIG, mining_seed=MINING_SEED,
                               token_counter=None)
    gate._consume_stage12_evidence(GateCandidateInput(
        candidate_envelope=envelope, frozen_rank=0, replay_evaluation=replay,
        ablation_evaluation=ablation, generalization_allocation=None,
        regression_allocation=None, parent_cig_id=PARENT_CIG))  # must not raise


def _settings_records(node, found):
    """Every recorded generation-settings dict in a CIG record."""
    if isinstance(node, dict):
        for key, value in node.items():
            if key in ("eval_generation_settings", "locked_settings") and isinstance(value, dict):
                found.append(value)
            _settings_records(value, found)
    elif isinstance(node, list):
        for item in node:
            _settings_records(item, found)


def test_every_record_of_a_gate_run_stores_the_protocol_stop(tmp_path):
    orchestrator, _, cig_store, _, inputs, _ = build_world(tmp_path, candidates=("a", "b"))
    orchestrator.run(inputs)
    found: list[dict] = []
    for record in list(cig_store.children.values()) + list(cig_store.aggregates.values()):
        _settings_records(record, found)
    stages = {child["stage"] for child in cig_store.children.values()}
    assert stages == {"replay", "ablation", "generalization_in_scope",
                      "generalization_out_of_scope", "interaction_regression"}
    assert len(found) >= 10  # two candidates x five children, plus pair evidence
    assert all(record["stop"] == list(STOP_SEQUENCES) for record in found)


def test_the_regression_stage_runs_with_settings_run_agent_accepts(tmp_path, registry,
                                                                  base_config, envelope):
    regression = RegressionEvaluator(
        candidate_envelope=envelope, registry=registry, base_config=base_config,
        runner=ScriptedRunner(), budgets=BUDGETS, parent_cig_id=PARENT_CIG,
        mining_seed=MINING_SEED)
    assert regression._settings == LOCKED
    result = run_agent(config=g0_runtime_config(registry),
                       model=ScriptedAdapter([react_final("done")]), env=ScriptedEnv({}),
                       task="t", budgets=BUDGETS, settings=regression._settings)
    assert result.status == "finished"
