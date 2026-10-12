"""Stage 4C interaction evaluator unit tests (issue #12; ADR-0002
Amendment A1) — canonical gene order, frozen target regulation, durable
canonical pair evidence, exact-base control arm."""

import pytest

from runtime.regulation import Condition, Predicate
from tests.gate.helpers import ScriptedGateRunner, interaction_tasks
from tests.inheritance.conftest import MINING_SEED, PARENT_CIG  # noqa: F401
from tests.somatic.conftest import acquired_envelope

from inheritance.interaction import (
    InteractionAllocation,
    InteractionEvaluator,
)
from inheritance.overlay import EvaluationSetupError, evaluation_settings
from inheritance.runner import EvaluationIntegrityError

BUDGETS = None  # set in fixture


@pytest.fixture
def world(tmp_path, registry, base_config, budgets):
    """Two candidates whose gene ids sort REVERSE caller order ('z_pair'
    vs 'a_pair') plus a scripted runner covering the pair conditions."""
    env_a = acquired_envelope(registry, gene_id="a_pair")
    env_z = acquired_envelope(registry, gene_id="z_pair")
    runner = ScriptedGateRunner()
    for i in range(10):
        tid = f"ix-{i}"
        runner.set(tid, "both", True, ("a_pair", "z_pair"))
        runner.set(tid, "neither", True, ())
    allocation = InteractionAllocation(tasks=interaction_tasks(10))
    evaluator = InteractionEvaluator(
        registry=registry, base_config=base_config, runner=runner,
        budgets=budgets)
    return {"registry": registry, "base": base_config, "runner": runner,
             "allocation": allocation, "evaluator": evaluator,
             "env_a": env_a, "env_z": env_z}


def test_gene_ids_canonical_regardless_of_caller_order(world):
    locked = evaluation_settings(max_tokens=64, seed=MINING_SEED)
    ev1 = world["evaluator"].evaluate_pair(world["env_z"], world["env_a"],
                                            world["allocation"], locked)
    ev2 = world["evaluator"].evaluate_pair(world["env_a"], world["env_z"],
                                            world["allocation"], locked)
    # canonical (sorted) gene-id order, not caller order
    assert ev1.gene_ids == ("a_pair", "z_pair")
    assert ev2.gene_ids == ("a_pair", "z_pair")
    # same canonical order -> identical evidence hash from either caller order
    assert ev1.evidence_sha256 == ev2.evidence_sha256
    # the both-arm inserts skills in canonical order
    config_a, _, _ = world["evaluator"]._build_pair_configs(
        world["env_z"], world["env_a"])
    tail = [s.gene_id for s in config_a.skills[-2:]]
    assert tail == ["a_pair", "z_pair"]


def test_target_regulation_preserved_in_both_arm(world):
    """A candidate with a frozen express_when condition keeps it in the
    both-trait overlay (never made constitutive); the neither arm carries
    no target regulation."""
    cond = Condition(all_of=(Predicate("consecutive_failures", ">=", 2),))
    regulated = InteractionEvaluator(
        registry=world["registry"], base_config=world["base"],
        runner=world["runner"], budgets=world["evaluator"]._budgets,
        target_regulation_by_gene={"z_pair": cond})
    both, neither, gene_ids = regulated._build_pair_configs(
        world["env_a"], world["env_z"])
    assert gene_ids == ("a_pair", "z_pair")
    # z_pair keeps its frozen regulation in the both arm
    assert both.regulation.get("z_pair") is cond
    # a_pair (no supplied regulation) stays constitutive (absent)
    assert "a_pair" not in both.regulation
    # neither arm = exact base: no target regulation at all
    assert "z_pair" not in neither.regulation
    assert "a_pair" not in neither.regulation
    assert neither.skills == world["base"].skills
    assert neither.regulation == world["base"].regulation
    # the regulated candidate's skill object carries the condition
    z_skill = [s for s in both.skills if s.gene_id == "z_pair"][0]
    assert z_skill.condition is cond


def test_canonical_evidence_is_durable_and_hash_domain(world):
    from trajectory.storage.canonical import canonical_json, sha256_hex
    locked = evaluation_settings(max_tokens=64, seed=MINING_SEED)
    evidence = world["evaluator"].evaluate_pair(
        world["env_a"], world["env_z"], world["allocation"], locked)
    record = evidence.canonical_evidence
    # the record IS the hash domain
    assert sha256_hex(canonical_json(record).encode("utf-8")) == \
        evidence.evidence_sha256
    # compact + reconstructable: outcomes, allocation, delta, settings
    assert record["kind"] == "heritagent-interaction-pair"
    assert record["gene_ids"] == ["a_pair", "z_pair"]
    assert len(record["paired_outcomes"]) == 20
    assert record["delta_pair"] == evidence.delta_pair
    assert record["harmful"] == evidence.harmful
    assert record["locked_settings"]["seed"] == MINING_SEED
    # frozen: mutating the returned copy does not corrupt future reads
    record["delta_pair"] = 99.0
    assert evidence.canonical_evidence["delta_pair"] == evidence.delta_pair


def test_neither_arm_expression_aborts(world):
    for i in range(10):
        world["runner"].set(f"ix-{i}", "neither", True, ("a_pair",))
    locked = evaluation_settings(max_tokens=64, seed=MINING_SEED)
    with pytest.raises(EvaluationIntegrityError, match="contaminated"):
        world["evaluator"].evaluate_pair(world["env_a"], world["env_z"],
                                          world["allocation"], locked)


def test_shared_gene_id_pair_rejected(world):
    from copy import deepcopy
    env = world["env_a"]
    with pytest.raises(EvaluationSetupError, match="share gene_id"):
        world["evaluator"].evaluate_pair(env, deepcopy(env),
                                          world["allocation"],
                                          evaluation_settings(max_tokens=64))
