"""Loader/validator: structural invariants + semantic checks (issue #4 criterion 2, 7)."""

import hashlib

import pytest

from genome.validation.errors import GenomeValidationError, RegistryIntegrityError
from genome.validation.loader import load_genome
from tests.conftest import fake_uri, minimal_genome, registered_genome


def skill_ref(doc_genome: dict) -> dict:
    return doc_genome["genes"]["skills"][0]


def test_missing_artifact_rejected(doc_genome):
    del skill_ref(doc_genome)["artifact"]
    with pytest.raises(GenomeValidationError, match="artifact"):
        load_genome(doc_genome)


def test_missing_provenance_rejected(doc_genome):
    del skill_ref(doc_genome)["provenance"]
    with pytest.raises(GenomeValidationError, match="provenance"):
        load_genome(doc_genome)


def test_born_generation_required_for_seed_origin():
    genome = minimal_genome()
    del genome["genes"]["cognition"]["planner"]["provenance"]["born_generation"]
    with pytest.raises(GenomeValidationError, match="born_generation"):
        load_genome(genome)


def test_born_generation_required_for_mutation_origin():
    genome = minimal_genome()
    ref = genome["genes"]["cognition"]["planner"]
    ref["origin"] = "mutation"
    ref["provenance"] = {"notes": "mutated without generation stamp"}
    with pytest.raises(GenomeValidationError, match="born_generation"):
        load_genome(genome)


@pytest.mark.parametrize("missing", ["source_trajectory", "cig_record"])
def test_assimilation_provenance_required(missing, doc_genome):
    del skill_ref(doc_genome)["provenance"][missing]
    with pytest.raises(GenomeValidationError, match=missing):
        load_genome(doc_genome)


def test_opaque_artifact_uri_rejected(doc_genome):
    skill_ref(doc_genome)["artifact"] = "foo"
    with pytest.raises(GenomeValidationError, match="does not match"):
        load_genome(doc_genome)


def test_short_digest_rejected(doc_genome):
    skill_ref(doc_genome)["artifact"] = fake_uri("skills", "look_before_heat_v1", 1, digest="9f2c81aa")
    with pytest.raises(GenomeValidationError, match="does not match"):
        load_genome(doc_genome)


def test_uppercase_digest_rejected(doc_genome):
    skill_ref(doc_genome)["artifact"] = fake_uri("skills", "look_before_heat_v1", 1, digest="0123456789ABCDEF" * 4)
    with pytest.raises(GenomeValidationError, match="does not match"):
        load_genome(doc_genome)


def test_uri_name_must_equal_gene_id():
    genome = minimal_genome()
    genome["genes"]["cognition"]["planner"]["artifact"] = fake_uri("policies", "planner_other_v1", 1)
    with pytest.raises(GenomeValidationError, match="gene_id"):
        load_genome(genome)


def test_uri_version_must_equal_gene_version():
    genome = minimal_genome()
    genome["genes"]["cognition"]["planner"]["artifact"] = fake_uri("policies", "planner_react_v1", 2)
    with pytest.raises(GenomeValidationError, match="version"):
        load_genome(genome)


def test_uri_kind_must_match_gene_type():
    genome = minimal_genome()
    genome["genes"]["cognition"]["planner"]["artifact"] = fake_uri("skills", "planner_react_v1", 1)
    with pytest.raises(GenomeValidationError, match="kind"):
        load_genome(genome)


def test_regulation_must_reference_known_genes():
    genome = minimal_genome()
    genome["regulation"] = {
        "nonexistent_skill_v1": {
            "express_when": {"all_of": [{"metric": "consecutive_failures", "op": ">=", "value": 2}]}
        }
    }
    with pytest.raises(GenomeValidationError, match="unknown gene"):
        load_genome(genome)


def test_semantic_checks_pass_with_registry_content(registry):
    genome = registered_genome(registry)
    assert load_genome(genome, registry) == genome


def test_missing_registry_artifact_rejected(registry):
    genome = registered_genome(registry)
    genome["genes"]["skills"][0]["artifact"] = fake_uri("skills", "look_before_heat_v1", 1, digest="ab" * 32)
    with pytest.raises(GenomeValidationError, match="not found in registry"):
        load_genome(genome, registry)


def test_digest_mismatch_against_registry_bytes_rejected(registry):
    genome = registered_genome(registry)
    skill_ref_dict = genome["genes"]["skills"][0]
    uri = skill_ref_dict["artifact"]
    digest = uri.split("sha256:")[1]
    # tamper with the stored artifact in place: bytes no longer hash to the URI digest
    path = registry.root / "skills" / digest
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(RegistryIntegrityError, match="tampered"):
        load_genome(genome, registry)


def test_digest_verified_against_actual_bytes(registry):
    genome = registered_genome(registry)
    # success path: every URI digest equals the sha256 of the stored artifact bytes
    for ref in [genome["genes"]["cognition"]["planner"], genome["genes"]["execution"]["retry_policy"],
                *genome["genes"]["skills"]]:
        stored = registry.resolve(ref["artifact"])
        assert hashlib.sha256(stored).hexdigest() == ref["artifact"].split("sha256:")[1]
