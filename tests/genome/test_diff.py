"""Deterministic parent->child genome diff (criterion 6, golden test)."""

import pytest

from genome.validation.diff import diff_genomes
from genome.validation.loader import load_genome
from tests.conftest import registered_genome, registered_parent_genome


@pytest.fixture
def parent_and_child(registry):
    parent = registered_parent_genome(registry)
    child = registered_genome(registry)
    assert child["parent"] == parent["genome_id"]
    return parent, child


def test_golden_diff(registry, parent_and_child):
    parent, child = parent_and_child
    load_genome(parent, registry)
    load_genome(child, registry)
    edge = diff_genomes(parent, child, fitness_delta=0.084)
    assert edge == {
        "parent": "G-0f1e2d3c",
        "child": "G-a1b2c3d4",
        "inheritance": ["planner_react_v1", "retry_backoff_v1"],
        "assimilated_traits": ["look_before_heat_v1"],
        "mutations": ["regulation.look_before_heat_v1: added"],
        "fitness_delta": 0.084,
    }


def test_diff_is_deterministic(parent_and_child):
    parent, child = parent_and_child
    assert diff_genomes(parent, child) == diff_genomes(parent, child)


def test_diff_of_cloned_child_has_no_mutations(registry, parent_and_child):
    import copy
    parent, _ = parent_and_child
    child = copy.deepcopy(parent)
    child["genome_id"] = "G-11111111"
    child["generation"] = 1
    child["parent"] = parent["genome_id"]
    edge = diff_genomes(parent, child)
    assert edge["inheritance"] == ["planner_react_v1", "retry_backoff_v1"]
    assert edge["mutations"] == []
    assert edge["assimilated_traits"] == []


def test_diff_detects_slot_mutation_and_skill_removal(registry):
    import copy
    parent = registered_genome(registry)  # has the assimilated skill + regulation
    child = copy.deepcopy(parent)
    child["genome_id"] = "G-99999999"
    child["generation"] = 2
    child["parent"] = parent["genome_id"]
    # child mutates the retry policy, drops the assimilated skill and its regulation
    child["genes"]["execution"]["retry_policy"] = {
        "gene_id": "retry_hypothesis_shift_v4", "version": 4, "type": "policy",
        "origin": "mutation",
        "artifact": "registry://policies/retry_hypothesis_shift_v4@4/sha256:" + "be" * 32,
        "provenance": {"born_generation": 1},
    }
    child["genes"]["skills"] = []
    del child["regulation"]["look_before_heat_v1"]
    edge = diff_genomes(parent, child)
    assert "execution.retry_policy: retry_backoff_v1@1 -> retry_hypothesis_shift_v4@4" in edge["mutations"]
    assert "skills-: look_before_heat_v1@1" in edge["mutations"]
    assert "regulation.look_before_heat_v1: removed" in edge["mutations"]
    assert edge["assimilated_traits"] == []


def test_diff_rejects_self_diff(parent_and_child):
    parent, _ = parent_and_child
    with pytest.raises(ValueError, match="must differ"):
        diff_genomes(parent, parent)


def test_new_mutation_skill_is_not_assimilation(registry, parent_and_child):
    parent, child = parent_and_child
    child["genes"]["skills"].append({
        "gene_id": "recovery_skill_17", "version": 1, "type": "skill",
        "origin": "recombination",
        "artifact": "registry://skills/recovery_skill_17@1/sha256:" + "ca" * 32,
        "provenance": {"born_generation": 1},
    })
    edge = diff_genomes(parent, child)
    assert "skills+: recovery_skill_17@1" in edge["mutations"]
    assert edge["assimilated_traits"] == ["look_before_heat_v1"]
