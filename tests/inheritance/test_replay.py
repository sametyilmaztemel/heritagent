"""Stage-1 evaluator tests (ADR-0012 D3): one with-trait episode on the
source task, passing iff the target gene is expressed or the episode fails."""

import pytest

from inheritance.replay import REPLAY_ROUND, ReplayAllocation, ReplayEvaluator
from inheritance.runner import EvaluationTask, RunnerContractError
from runtime.loop import Budgets
from tests.inheritance.conftest import GENE_ID, MINING_SEED, PARENT_CIG
from tests.inheritance.scripted_runner import ScriptedRunner

SUCCESS_GENES = (GENE_ID,)  # tuple of gene-id strings
NO_GENES = ()
SOURCE = "src-1"


def make_allocation(tmp_path=None, registry=None) -> ReplayAllocation:
    """The source task alone (ADR-0012 D3)."""
    return ReplayAllocation(source_task=EvaluationTask(
        task_id=SOURCE, family="heat_and_place", payload={"goal": "heat"}, bank_id="t_mine"))


def make_evaluator(tmp_path, registry, base_config, envelope, budgets=None):
    return ReplayEvaluator(
        candidate_envelope=envelope, registry=registry, base_config=base_config,
        runner=ScriptedRunner(),
        budgets=budgets or Budgets(max_steps=4, max_total_retries=10,
                                   max_tokens_per_request=64),
        parent_cig_id=PARENT_CIG, mining_seed=MINING_SEED)


def outcomes_for(success: bool, expressed: bool = True):
    """The source task's with-trait outcome, with a fake T-... trajectory."""
    return {(SOURCE, "with_trait"): (success, SUCCESS_GENES if expressed else NO_GENES,
                                     "T-replay00w")}


@pytest.mark.parametrize("success, expressed, passed", [
    (True, True, True),     # a successful replay expresses its gene
    (True, False, False),   # a successful replay that never expressed it fails
    (False, False, True),   # a failed episode passes: nothing is compared (D3)
    (False, True, True),
])
def test_the_expression_rule(tmp_path, registry, base_config, envelope, success, expressed,
                             passed):
    evaluator = make_evaluator(tmp_path, registry, base_config, envelope)
    evaluator._runner.outcomes = outcomes_for(success, expressed)
    result = evaluator.evaluate(make_allocation())
    assert result.passed is passed
    assert (result.success, result.expressed) == (success, expressed)
    m = result.child_record["measurements"]
    assert m["stage_pass"] is passed
    assert m["expression_check"] == {"expressed": expressed, "success": success,
                                     "passed": passed, "rule": m["thresholds"]["rule"]}


def test_exactly_one_with_trait_call_on_the_source_task(tmp_path, registry, base_config,
                                                        envelope):
    evaluator = make_evaluator(tmp_path, registry, base_config, envelope)
    evaluator._runner.outcomes = outcomes_for(True)
    evaluator.evaluate(make_allocation())
    assert evaluator._runner.calls == [(SOURCE, "with_trait")]  # no without-trait execution


def test_the_allocation_is_the_source_task_alone():
    allocation = make_allocation()
    assert allocation.tasks == (allocation.source_task,)
    assert allocation.allocation_metadata() == {
        "source_task_id": SOURCE, "source_family": "heat_and_place", "bank_id": "t_mine",
        "instance_count": 1}
    with pytest.raises(ValueError, match="EvaluationTask"):
        ReplayAllocation(source_task="src-1")


def test_child_record_schema_valid_and_coherent(tmp_path, registry, base_config, envelope):
    from genome.validation.loader import load_cig
    evaluator = make_evaluator(tmp_path, registry, base_config, envelope)
    evaluator._runner.outcomes = outcomes_for(True)
    child = evaluator.evaluate(make_allocation()).child_record
    load_cig(child)  # schema-valid
    assert child["cig_id"] == f"{PARENT_CIG}/S{MINING_SEED}-replay"
    assert (child["parent_cig_id"], child["seed"], child["stage"]) == \
        (PARENT_CIG, MINING_SEED, "replay")
    m = child["measurements"]
    assert m["stage_pass"] is True
    assert m["deterministic_execution_policy"] == {
        "runs_per_task_condition": 1, "eval_temperature": 0.0, "paired_conditions": ["with_trait"]}
    [episode] = m["episodes"]
    assert (episode["task_id"], episode["condition"], episode["round"]) == \
        (SOURCE, "with_trait", REPLAY_ROUND)
    assert m["allocation"]["source_family"] == "heat_and_place"
    for key in ("candidate_version", "candidate_artifact", "base_genome_id"):
        assert key in m  # ADR-0010 B1


def test_runner_contract_violations_fail_closed(tmp_path, registry, base_config, envelope):
    evaluator = make_evaluator(tmp_path, registry, base_config, envelope)
    evaluator._runner.outcomes = outcomes_for(True)
    from inheritance.runner import EpisodeOutcome
    real_run = evaluator._runner.run_episode

    def mismatched(task, config, budgets, condition, settings):
        outcome = real_run(task, config, budgets, condition, settings)
        return EpisodeOutcome(task_id="wrong-id", condition=condition,
                              success=outcome.success,
                              expressed_gene_ids=outcome.expressed_gene_ids,
                              status=outcome.status, effective_settings=settings)
    evaluator._runner.run_episode = mismatched
    with pytest.raises(RunnerContractError, match="task_id"):
        evaluator.evaluate(make_allocation())


def test_no_somatic_decision_no_germline_mutation(tmp_path, registry, base_config, envelope):
    """Stage 1 never writes a somatic terminal decision or a germline ref."""
    from somatic.store import SomaticStore
    store = SomaticStore.create(tmp_path / "somatic.jsonl", registry)
    store.add_candidate(envelope)
    store.close()
    evaluator = make_evaluator(tmp_path, registry, base_config, envelope)
    evaluator._runner.outcomes = outcomes_for(True)
    assert evaluator.evaluate(make_allocation()).passed is True
    assert store.state("look_before_heat_v1", 1) == "candidate"
    assert "cig_record" not in store.get("look_before_heat_v1", 1)["candidate"]["provenance"]
    store.close()


def test_preflight_invalid_parent_zero_runner_calls(tmp_path, registry, base_config,
                                                    envelope):
    """Invalid parent CIG id => typed error BEFORE any episode."""
    evaluator = make_evaluator(tmp_path, registry, base_config, envelope)
    with pytest.raises(RunnerContractError, match="parent CIG id"):
        evaluator.__init__(
            candidate_envelope=envelope, registry=registry,
            base_config=base_config, runner=evaluator._runner,
            budgets=Budgets(max_steps=4, max_total_retries=10, max_tokens_per_request=64),
            parent_cig_id="GATE-1", mining_seed=MINING_SEED)
    assert evaluator._runner.calls == []


def test_preflight_invalid_seed_zero_runner_calls(tmp_path, registry, base_config,
                                                  envelope):
    evaluator = make_evaluator(tmp_path, registry, base_config, envelope)
    with pytest.raises(RunnerContractError, match="non-negative integer"):
        evaluator.__init__(
            candidate_envelope=envelope, registry=registry,
            base_config=base_config, runner=evaluator._runner,
            budgets=Budgets(max_steps=4, max_total_retries=10, max_tokens_per_request=64),
            parent_cig_id=PARENT_CIG, mining_seed=-1)
    assert evaluator._runner.calls == []


def test_missing_trajectory_rejected_no_child(tmp_path, registry, base_config, envelope):
    from inheritance.runner import EpisodeOutcome
    evaluator = make_evaluator(tmp_path, registry, base_config, envelope)
    evaluator._runner.outcomes = outcomes_for(True)
    real_run = evaluator._runner.run_episode

    def no_trajectory(task, config, budgets, condition, settings=None):
        outcome = real_run(task, config, budgets, condition, settings)
        return EpisodeOutcome(
            task_id=outcome.task_id, condition=outcome.condition, success=outcome.success,
            expressed_gene_ids=outcome.expressed_gene_ids, status=outcome.status,
            trajectory_id=None, effective_settings=outcome.effective_settings)
    evaluator._runner.run_episode = no_trajectory
    with pytest.raises(RunnerContractError, match="trajectory_id"):
        evaluator.evaluate(make_allocation())


def test_runner_temperature_violation_rejected_no_child(tmp_path, registry, base_config,
                                                        envelope):
    """A runner reporting temperature 0.7 is rejected before any child."""
    evaluator = make_evaluator(tmp_path, registry, base_config, envelope)
    evaluator._runner.outcomes = outcomes_for(True)
    evaluator._runner.acknowledge_settings = False
    evaluator._runner.reported_temperature = 0.7
    with pytest.raises(RunnerContractError, match="temperature"):
        evaluator.evaluate(make_allocation())


def test_runner_acks_temp0_but_changes_max_tokens_rejected(tmp_path, registry,
                                                           base_config, envelope):
    """Acknowledging temperature 0 while changing max_tokens/seed/stop is
    rejected: exact settings divergence, no completed child."""
    from inheritance.runner import EpisodeOutcome
    from runtime.model_adapters import GenerationSettings
    evaluator = make_evaluator(tmp_path, registry, base_config, envelope)
    evaluator._runner.outcomes = outcomes_for(True)
    real_run = evaluator._runner.run_episode

    def diverging(task, config, budgets, condition, settings):
        outcome = real_run(task, config, budgets, condition, settings)
        return EpisodeOutcome(
            task_id=outcome.task_id, condition=outcome.condition, success=outcome.success,
            expressed_gene_ids=outcome.expressed_gene_ids, status=outcome.status,
            trajectory_id=outcome.trajectory_id,
            effective_settings=GenerationSettings(temperature=0.0, max_tokens=9999,
                                                  seed=12345, stop=("END",)))
    evaluator._runner.run_episode = diverging
    with pytest.raises(RunnerContractError, match="effective settings differ"):
        evaluator.evaluate(make_allocation())


def test_locked_settings_derived_from_budgets(tmp_path, registry, base_config, envelope):
    budgets = Budgets(max_steps=4, max_total_retries=10, max_tokens_per_request=64)
    evaluator = make_evaluator(tmp_path, registry, base_config, envelope, budgets=budgets)
    assert evaluator._locked_settings.temperature == 0.0
    assert evaluator._locked_settings.max_tokens == budgets.max_tokens_per_request
    assert evaluator._locked_settings.seed == MINING_SEED  # mining seed as fixed seed
    assert evaluator._locked_settings.stop == ("\nObservation:",)  # ADR-0003 A2
