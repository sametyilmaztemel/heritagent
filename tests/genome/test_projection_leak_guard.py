"""The runtime leak guard checks keys, not text (review of #58, 2026-10-13).

``project_for_runtime`` ends with a defense-in-depth check that no
evaluation-only key survives the projection. It used to scan the JSON
text for each quoted key word, so a string value equal to one, such as a
skill named "evaluation", a principle "provenance", or a gene id
"fitness" used as a regulation map key, was rejected although nothing
had leaked. The guard now walks object keys at any depth, skipping only
the regulation map's keys, which are gene ids."""

import pytest

from genome.validation.errors import GenomeValidationError
from genome.validation.projection import (
    EVALUATION_ONLY_KEYS,
    _assert_no_evaluation_only,
    project_for_runtime,
)
from runtime.expression import compile_runtime_config
from runtime.seed_g0 import build_g0_seed_genome

CONDITION = {"all_of": [{"metric": "consecutive_failures", "op": ">=", "value": 2}]}


def finalize(genome: dict) -> dict:
    """The canonical content-hash genome_id once ADR-0001 A2 (PR #29) is on
    main; until then the seed's id stands."""
    try:
        from genome.validation.identity import compute_genome_id
    except ImportError:
        return genome
    genome["genome_id"] = compute_genome_id(genome)
    return genome


def skill_payload(name: str = "probe_heater", **overrides) -> dict:
    payload = {"name": name,
               "principle": "Probe the heater before heating.",
               "when_to_apply": "Heat tasks.",
               "procedure": [{"id": "S1", "text": "probe the heater"}],
               "applicability": {"task_families": ["heat_and_place"]}}  # stripped by projection
    payload.update(overrides)
    return payload


def genome_with_skill(registry, payload: dict, *, gene_id: str = "probe_heater",
                      regulated: bool = False) -> dict:
    genome = build_g0_seed_genome(registry)
    uri = registry.put("skill", gene_id, 1, payload)
    genome["genes"]["skills"].append({"gene_id": gene_id, "version": 1, "type": "skill",
                                      "origin": "seed", "artifact": uri,
                                      "provenance": {"born_generation": 0}})
    if regulated:
        genome["regulation"] = {gene_id: {"express_when": CONDITION}}
    return finalize(genome)


@pytest.mark.parametrize("word", EVALUATION_ONLY_KEYS)
@pytest.mark.parametrize("field", ["name", "principle", "when_to_apply", "step"])
def test_a_skill_whose_text_is_a_key_word_compiles(registry, word, field):
    if field == "step":
        payload = skill_payload(procedure=[{"id": "S1", "text": word}])
    else:
        payload = skill_payload(**{field: word})
    genome = genome_with_skill(registry, payload)
    projection = project_for_runtime(genome, registry)
    assert word in [projection["genes"]["skills"][0].get(field),
                    projection["genes"]["skills"][0]["procedure"][0]["text"]]
    config = compile_runtime_config(genome, registry)
    assert [s.gene_id for s in config.skills] == ["probe_heater"]


@pytest.mark.parametrize("word", ["evaluation", "fitness", "provenance", "applicability"])
def test_a_regulated_gene_whose_id_is_a_key_word_compiles(registry, word):
    genome = genome_with_skill(registry, skill_payload(name=word), gene_id=word, regulated=True)
    assert list(project_for_runtime(genome, registry)["regulation"]) == [word]
    assert [s.gene_id for s in compile_runtime_config(genome, registry).skills] == [word]


@pytest.mark.parametrize("key", EVALUATION_ONLY_KEYS)
def test_a_nested_key_in_a_procedure_step_is_still_rejected(registry, key):
    # in the second of two steps: every list item is walked (review of #61)
    payload = skill_payload(procedure=[{"id": "S1", "text": "find"},
                                       {"id": "S2", "text": "probe", key: {"x": 1}}])
    with pytest.raises(GenomeValidationError, match=rf"leaked evaluation-only keys: \['{key}'\]"):
        project_for_runtime(genome_with_skill(registry, payload), registry)


@pytest.mark.parametrize("key", EVALUATION_ONLY_KEYS)
def test_a_nested_key_in_a_policy_payload_is_still_rejected(registry, key):
    genome = build_g0_seed_genome(registry)
    planner = genome["genes"]["cognition"]["planner"]
    planner["artifact"] = registry.put("policy", planner["gene_id"], 2, {
        "style": "react", "max_plan_steps": 8, "notes": [{"origin": {key: "x"}}]})
    planner["version"] = 2
    with pytest.raises(GenomeValidationError, match=rf"leaked evaluation-only keys: \['{key}'\]"):
        project_for_runtime(finalize(genome), registry)


def test_every_leaked_key_is_named_in_the_locked_order(registry):
    # locked order (EVALUATION_ONLY_KEYS), which differs from both the
    # alphabetical and the insertion order here
    payload = skill_payload(procedure=[{"id": "S1", "text": "probe",
                                        "provenance": 1, "cig_record": 2, "gate_reports": 3}])
    with pytest.raises(GenomeValidationError,
                       match=r"\['gate_reports', 'cig_record', 'provenance'\]"):
        project_for_runtime(genome_with_skill(registry, payload), registry)


def test_a_deeply_nested_key_gets_the_typed_error():
    """Review of #61: a key buried 5000 levels deep is reported as a leak,
    not as a RecursionError (the walk is iterative)."""
    deep = {"fitness": 1}
    for _ in range(5000):
        deep = {"meta": [deep]}
    projection = {"genome_id": "G-00000000", "schema_version": "0.1",
                  "genes": {"skills": [{"name": "x", "notes": deep}]}, "regulation": {}}
    with pytest.raises(GenomeValidationError, match=r"\['fitness'\]"):
        _assert_no_evaluation_only(projection)
