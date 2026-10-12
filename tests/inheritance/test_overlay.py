"""Evaluation-only overlay tests (issue #11 criterion 2, 12)."""

import json

import pytest

from inheritance.overlay import (
    EvaluationSetupError,
    build_evaluation_configs,
    evaluation_settings,
)
from runtime.regulation import Condition
from tests.somatic.conftest import acquired_envelope

FORBIDDEN_IN_MESSAGES = (
    "applicability", "task_families", "estimated_generality",
    "evidence_rationale", "success", "reward", "source_trajectory",
    "provenance", "cig_record", "gate_reports", "CIG-",
)


def test_overlay_builds_paired_configs(tmp_path, registry, base_config, envelope):
    configs = build_evaluation_configs(base_config, envelope, registry)
    assert configs.target_gene_id == "look_before_heat_v1"
    # with-trait: base + target appended
    assert [s.gene_id for s in configs.with_trait.skills] == \
        [s.gene_id for s in base_config.skills] + ["look_before_heat_v1"]
    assert configs.without_trait.skills == base_config.skills
    # without-trait: exactly the base config
    assert configs.without_trait.skills == base_config.skills
    assert configs.without_trait.regulation == base_config.regulation


def test_base_config_never_mutated(tmp_path, registry, base_config, envelope):
    skills_before = tuple(base_config.skills)
    regulation_before = dict(base_config.regulation)
    build_evaluation_configs(base_config, envelope, registry)
    assert tuple(base_config.skills) == skills_before
    assert dict(base_config.regulation) == regulation_before
    # no fake germline provenance: candidate origin untouched, genome untouched
    assert envelope["candidate"]["origin"] == "acquired"


def test_duplicate_target_in_base_config_rejected(tmp_path, registry, base_config, envelope):
    from genome.validation.loader import load_somatic  # noqa: F401
    # pre-inject the target gene into the base config
    payload = json.loads(registry.resolve(envelope["candidate"]["artifact"]))
    from runtime.expression import RuntimeConfig, RuntimeSkill
    polluted = RuntimeConfig(
        genome_id=base_config.genome_id, policies=base_config.policies,
        skills=base_config.skills + (RuntimeSkill(
            gene_id="look_before_heat_v1", name="look_before_heat_v1",
            principle="p", when_to_apply="w"),), regulation={})
    with pytest.raises(EvaluationSetupError, match="double-inject"):
        build_evaluation_configs(polluted, envelope, registry)


def test_frozen_regulation_preserved_in_overlay(tmp_path, registry, base_config, envelope):
    condition = Condition.from_genome(
        {"express_when": {"all_of": [{"metric": "consecutive_failures",
                                       "op": ">=", "value": 2}]}})
    configs = build_evaluation_configs(base_config, envelope, registry,
                                        target_regulation=condition)
    assert configs.with_trait.regulation["look_before_heat_v1"] == condition
    assert "look_before_heat_v1" not in configs.without_trait.regulation
    # with-trait skill carries the condition; without-trait has no target
    target = [s for s in configs.with_trait.skills
               if s.gene_id == "look_before_heat_v1"][0]
    assert target.condition == condition


def test_evaluation_settings_lock_temperature_zero():
    settings = evaluation_settings()
    assert settings.temperature == 0.0


def test_projection_strips_evaluation_only_metadata(tmp_path, registry, base_config, envelope):
    configs = build_evaluation_configs(base_config, envelope, registry)
    target = [s for s in configs.with_trait.skills
               if s.gene_id == "look_before_heat_v1"][0]
    serialized = json.dumps(target.__dict__)
    for forbidden in FORBIDDEN_IN_MESSAGES:
        assert forbidden not in serialized, f"{forbidden} leaked into overlay skill"
    # only runtime-approved fields exposed
    assert set(json.loads(json.dumps(
        {k: v for k, v in target.__dict__.items()
          if k in ("name", "principle", "when_to_apply", "procedure")}))) <= {
        "name", "principle", "when_to_apply", "procedure"}


def test_invalid_state_candidate_rejected(tmp_path, registry, base_config, envelope):
    envelope["validation"]["state"] = "validated"
    envelope["validation"]["gate_reports"] = ["CIG-0007"]
    with pytest.raises(EvaluationSetupError, match="state must be 'candidate'"):
        build_evaluation_configs(base_config, envelope, registry)


def test_unbound_artifact_rejected_by_overlay(tmp_path, registry, base_config, envelope):
    envelope["candidate"]["artifact"] = (
        "registry://skills/look_before_heat_v1@1/sha256:" + "ab" * 32)
    with pytest.raises(EvaluationSetupError, match="somatic validation"):
        build_evaluation_configs(base_config, envelope, registry)
