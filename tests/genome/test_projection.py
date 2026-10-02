"""Runtime-input projection hygiene (criterion 5): no evaluation-only metadata in any arm."""

import json

import pytest

from genome.validation.errors import GenomeValidationError
from genome.validation.projection import (
    project_for_runtime,
    project_skill,
    project_skillrl,
)
from tests.conftest import registered_genome

EVALUATION_ONLY_KEYS = ("applicability", "task_families", "evaluation", "fitness", "gate_reports", "cig_record", "provenance")


def skill_payload(**extra) -> dict:
    payload = {
        "name": "look_before_heat_v1",
        "principle": "Inspect affordances before heating.",
        "when_to_apply": "Heat tasks.",
        "procedure": [{"id": "S1", "text": "find"}],
    }
    payload.update(extra)
    return payload


def test_runtime_projection_strips_evaluation_only_fields():
    projected = project_skill(skill_payload(
        applicability={"task_families": ["heat_and_place"]},
        fitness=0.83,
        cig_record="CIG-0007",
    ))
    assert set(projected) == {"name", "principle", "when_to_apply", "procedure"}
    for key in EVALUATION_ONLY_KEYS:
        assert key not in projected


def test_runtime_projection_requires_core_fields():
    with pytest.raises(GenomeValidationError, match="missing required field"):
        project_skill({"name": "x", "when_to_apply": "y"})


def test_skillrl_projection_strips_procedure():
    projected = project_skillrl(skill_payload(applicability={"task_families": ["x"]}))
    assert projected == {
        "name": "look_before_heat_v1",
        "principle": "Inspect affordances before heating.",
        "when_to_apply": "Heat tasks.",
    }
    assert "procedure" not in projected


def test_policy_payload_projection_strips_evaluation_only():
    from genome.validation.projection import _project_policy
    projected = _project_policy({"top_k": 6, "applicability": {"task_families": ["x"]}, "fitness": 0.5})
    assert projected == {"top_k": 6}


def test_whole_genome_projection_is_clean(registry):
    genome = registered_genome(registry)  # skill payload carries applicability + procedure
    projection = project_for_runtime(genome, registry)
    text = json.dumps(projection)
    for key in EVALUATION_ONLY_KEYS:
        assert f'"{key}"' not in text, f"evaluation-only key {key} leaked into runtime projection"
    assert projection["genome_id"] == "G-a1b2c3d4"
    assert projection["genes"]["skills"][0]["procedure"][0]["id"] == "S1"
    assert projection["genes"]["cognition"]["planner"] == {"style": "react", "max_plan_steps": 8}
    assert "regulation" in projection  # regulation is runtime logic, not evaluation metadata


def test_projection_leak_guard_is_defense_in_depth(registry, monkeypatch):
    # if a future edit reintroduces an evaluation-only key past the allowlist,
    # the final scan must still raise
    from genome.validation import projection as projection_module
    genome = registered_genome(registry)
    monkeypatch.setattr(projection_module, "RUNTIME_SKILL_FIELDS",
                        ("name", "principle", "when_to_apply", "procedure", "applicability"))
    with pytest.raises(GenomeValidationError, match="leaked evaluation-only"):
        project_for_runtime(genome, registry)
