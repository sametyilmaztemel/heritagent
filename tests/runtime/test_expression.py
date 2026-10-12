"""EAG -> runtime config compilation tests (issue #7 criteria 4, 9)."""

import json

import pytest

from genome.validation.errors import GenomeValidationError
from runtime.expression import compile_runtime_config
from tests.runtime.conftest import config_with_skill, seed_with_skill


def test_g0_seed_compiles_to_typed_policies(g0_config):
    assert g0_config.genome_id == "G-0f1e2d3c"
    planner = g0_config.policies["cognition.planner"]
    retry = g0_config.policies["execution.retry_policy"]
    assert planner.config == {"style": "react", "max_plan_steps": 8}
    assert retry.config == {"max_retries": 3, "backoff": "exponential"}
    assert g0_config.skills == ()
    assert g0_config.regulation == {}


def test_evaluation_only_metadata_never_enters_runtime_config(registry):
    config = config_with_skill(registry)  # skill payload carries applicability.task_families
    assert len(config.skills) == 1
    serialized = json.dumps(config.skills[0].__dict__, default=str)
    for forbidden in ("applicability", "task_families", "cig_record", "gate_reports", "provenance"):
        assert forbidden not in serialized, f"{forbidden} leaked into runtime config"
    assert config.skills[0].procedure[0]["id"] == "S1"  # stable ids preserved


def test_malformed_planner_payload_rejected(registry):
    genome = seed_with_skill(registry)
    # rebind planner payload to an invalid style via a new version
    uri = registry.put("policy", "planner_react_v1", 2, {"style": "cot", "max_plan_steps": 4})
    genome["genes"]["cognition"]["planner"]["version"] = 2
    genome["genes"]["cognition"]["planner"]["artifact"] = uri
    with pytest.raises(GenomeValidationError, match="cognition.planner"):
        compile_runtime_config(genome, registry)


def test_malformed_retry_payload_rejected(registry):
    genome = seed_with_skill(registry)
    uri = registry.put("policy", "retry_backoff_v1", 2, {"max_retries": 3, "backoff": "fibonacci"})
    genome["genes"]["execution"]["retry_policy"]["version"] = 2
    genome["genes"]["execution"]["retry_policy"]["artifact"] = uri
    with pytest.raises(GenomeValidationError, match="execution.retry_policy"):
        compile_runtime_config(genome, registry)


def test_skill_procedure_ids_must_be_unique(registry):
    genome = seed_with_skill(registry)
    genome["genes"]["skills"][0]["version"] = 2
    genome["genes"]["skills"][0]["artifact"] = registry.put("skill", "look_before_heat_v1", 2, {
        "name": "look_before_heat_v1",
        "principle": "p", "when_to_apply": "w",
        "procedure": [{"id": "S1", "text": "a"}, {"id": "S1", "text": "b"}],
    })
    with pytest.raises(GenomeValidationError, match="unique"):
        compile_runtime_config(genome, registry)


def test_skill_procedure_id_pattern_enforced(registry):
    genome = seed_with_skill(registry)
    genome["genes"]["skills"][0]["version"] = 2
    genome["genes"]["skills"][0]["artifact"] = registry.put("skill", "look_before_heat_v1", 2, {
        "name": "look_before_heat_v1",
        "principle": "p", "when_to_apply": "w",
        "procedure": [{"id": "step-one", "text": "a"}],
    })
    with pytest.raises(GenomeValidationError,
                        match="does not match"):
        compile_runtime_config(genome, registry)


def test_regulated_gene_must_exist_in_runtime_config(registry):
    genome = seed_with_skill(registry)
    genome["regulation"] = {"ghost_skill_v1": {
        "express_when": {"all_of": [{"metric": "consecutive_failures", "op": ">=", "value": 2}]}}}
    with pytest.raises(GenomeValidationError, match="unknown gene"):
        compile_runtime_config(genome, registry)


def test_compiled_skill_regulation_wiring(registry):
    config = config_with_skill(
        registry, express_when={"all_of": [{"metric": "consecutive_failures", "op": ">=", "value": 2}]})
    assert config.skills[0].condition is not None
    assert not config.is_active("look_before_heat_v1", {"consecutive_failures": 0})
    assert config.is_active("look_before_heat_v1", {"consecutive_failures": 3})
    # constitutive genes have no condition
    assert config.is_active("planner_react_v1", {})
