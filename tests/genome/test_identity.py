"""Canonical genome identity contract tests (ADR-0001 Amendment A2,
issue #22): identity domain, deterministic canonicalization, hash
contract, claimed-id validation, lineage-metadata exclusion, ordering
semantics, security fail-closed behaviour, and the G0 production path."""

import copy
import json

import pytest

from genome.validation.errors import GenomeValidationError
from genome.validation.identity import (
    GENOME_ID_HEX_CHARS,
    GENOME_ID_PATTERN,
    canonical_genome_identity_payload,
    compute_genome_id,
    genome_identity_hash_hex,
    validate_genome_id,
)
from genome.validation.loader import load_genome
from tests.conftest import finalize_genome, minimal_genome
from trajectory.storage.canonical import canonical_json

DIGEST = "ab" * 32
DIGEST2 = "cd" * 32


KIND_BY_GENE_TYPE = {"policy": "policies", "skill": "skills"}


def gene_ref(kind: str, gene_id: str, version: int, digest: str = DIGEST) -> dict:
    return {"gene_id": gene_id, "version": version,
             "type": kind,
             "origin": "seed",
             "artifact": (f"registry://{KIND_BY_GENE_TYPE[kind]}/{gene_id}"
                           f"@{version}/sha256:{digest}"),
             "provenance": {"born_generation": 0}}


def base_genome() -> dict:
    """Fully valid genome (registry-grammar URIs) with canonical id set."""
    return finalize_genome({
        "schema_version": "0.1",
        "generation": 1,
        "lineage_id": "L-alpha",
        "parent": "G-0000000000000000",
        "genes": {
            "cognition": {"planner": gene_ref("policy", "planner_react_v1", 1)},
            "execution": {"retry_policy": gene_ref("policy", "retry_backoff_v1", 1)},
            "skills": [gene_ref("skill", "look_before_heat_v1", 1),
                        gene_ref("skill", "check_heat_state_v1", 1, DIGEST2)],
        },
        "regulation": {
            "look_before_heat_v1": {"express_when": {"all_of": [
                {"metric": "consecutive_failures", "op": ">=", "value": 2}]}}
        },
    })


# -- identity domain: heritable functional changes => NEW id -----------------


@pytest.mark.parametrize("mutation", [
    pytest.param(lambda g: g["genes"]["skills"][0].update(
        version=2, artifact=f"registry://skills/look_before_heat_v1@2/sha256:{DIGEST2}"),
        id="skill artifact version change"),
    pytest.param(lambda g: g["genes"]["skills"].append(
        gene_ref("skill", "recovery_skill_v1", 1, DIGEST2)),
        id="skill added"),
    pytest.param(lambda g: g["genes"]["skills"].pop(0), id="skill removed"),
    pytest.param(lambda g: g["genes"]["cognition"]["planner"].update(
        artifact=f"registry://policies/planner_react_v1@1/sha256:{DIGEST2}"),
        id="policy artifact change"),
    pytest.param(lambda g: g["genes"]["cognition"].update(
        planner=gene_ref("policy", "planner_chain_of_note_v2", 2, DIGEST2)),
        id="policy reference change"),
    pytest.param(lambda g: g["regulation"]["look_before_heat_v1"]["express_when"]
        ["all_of"][0].update(op=">"), id="regulation predicate change"),
    pytest.param(lambda g: g["regulation"].pop("look_before_heat_v1"),
        id="regulation removed"),
    pytest.param(lambda g: g["genes"]["skills"].reverse(),
        id="expression-relevant skill reorder"),
    pytest.param(lambda g: g.update(schema_version="0.2"),
        id="schema version change"),
])
def test_heritable_functional_change_produces_new_genome_id(mutation):
    genome = base_genome()
    original_id = genome["genome_id"]
    mutation(genome)
    assert compute_genome_id(genome) != original_id


def test_regulation_added_is_heritable():
    genome = base_genome()
    original_id = genome["genome_id"]
    genome["regulation"]["retry_backoff_v1"] = {"express_when": {"all_of": [
        {"metric": "steps_without_progress", "op": ">=", "value": 3}]}}
    assert compute_genome_id(genome) != original_id


# -- identity domain: lineage/audit metadata => SAME id ----------------------


@pytest.mark.parametrize("mutation", [
    pytest.param(lambda g: g.update(parent="G-9999999999999999"),
        id="different parent"),
    pytest.param(lambda g: g.update(parent=None), id="null parent (founder)"),
    pytest.param(lambda g: g.update(generation=7), id="different generation"),
    pytest.param(lambda g: g.update(lineage_id="L-beta"),
        id="different lineage id"),
    pytest.param(lambda g: g["genes"]["skills"][0].update(origin="assimilation"),
        id="gene origin change"),
    pytest.param(lambda g: g["genes"]["skills"][0]["provenance"].update(
        born_generation=42), id="born_generation change"),
    pytest.param(lambda g: g["genes"]["skills"][0]["provenance"].update(
        source_trajectory="T-other"), id="source_trajectory change"),
    pytest.param(lambda g: g["genes"]["skills"][0]["provenance"].update(
        cig_record="CIG-9999"), id="cig_record change"),
    pytest.param(lambda g: g["genes"]["skills"][0]["provenance"].update(
        notes="auditor was here"), id="audit notes"),
])
def test_non_heritable_metadata_keeps_same_genome_id(mutation):
    genome = base_genome()
    original_id = genome["genome_id"]
    mutation(genome)
    assert compute_genome_id(genome) == original_id


def test_parent_and_generation_never_fork_content_identity():
    """G(content=C, parent=P1, generation=1) and G(content=C, parent=P2,
    generation=7) MUST share one genome_id — lineage bookkeeping is not
    content identity (issue #22 §11; critical for #13 descendant
    lineages)."""
    g1 = base_genome()
    g1.update(parent="G-1111111111111111", generation=1)
    g2 = copy.deepcopy(g1)
    g2.update(parent="G-2222222222222222", generation=7,
               lineage_id="L-descendant-branch")
    assert compute_genome_id(g1) == compute_genome_id(g2) == g1["genome_id"]


def test_claimed_genome_id_is_never_an_input_to_its_own_hash():
    genome = base_genome()
    expected = compute_genome_id(genome)
    for fake in ("G-ffffffffffffffff", "G-deadbeef", "totally-arbitrary"):
        genome["genome_id"] = fake
        assert compute_genome_id(genome) == expected


# -- ordering / canonicalization determinism ---------------------------------


def test_dict_insertion_order_independence():
    """The same semantic genome built with different Python dict insertion
    orders produces the SAME id (canonical JSON sorts map keys)."""
    skills_first = {
        "genes": {"skills": [gene_ref("skill", "look_before_heat_v1", 1)],
                   "execution": {"retry_policy": gene_ref("policy", "retry_backoff_v1", 1)},
                   "cognition": {"planner": gene_ref("policy", "planner_react_v1", 1)}},
        "regulation": {"look_before_heat_v1": {"express_when": {"all_of": [
            {"metric": "consecutive_failures", "op": ">=", "value": 2}]}}},
        "lineage_id": "L-alpha", "parent": None, "generation": 0,
        "schema_version": "0.1",
    }
    genes_first = {
        "schema_version": "0.1", "generation": 0, "lineage_id": "L-alpha",
        "parent": None,
        "genes": {"cognition": {"planner": gene_ref("policy", "planner_react_v1", 1)},
                   "execution": {"retry_policy": gene_ref("policy", "retry_backoff_v1", 1)},
                   "skills": [gene_ref("skill", "look_before_heat_v1", 1)]},
        "regulation": {"look_before_heat_v1": {"express_when": {"all_of": [
            {"value": 2, "op": ">=", "metric": "consecutive_failures"}]}}},
    }
    id_a = compute_genome_id(skills_first)
    id_b = compute_genome_id(genes_first)
    assert id_a == id_b
    # and byte-identical canonical payloads
    assert (canonical_genome_identity_payload(skills_first)
             == canonical_genome_identity_payload(genes_first))


def test_skill_array_order_is_semantically_ordered():
    """ADR decision: skills[] order is runtime-functional (block/prompt
    rendering order) — reversing it MUST change the id (never sorted)."""
    genome = base_genome()
    reversed_genome = copy.deepcopy(genome)
    reversed_genome["genes"]["skills"].reverse()
    assert compute_genome_id(reversed_genome) != genome["genome_id"]


def test_regulation_predicate_arrays_are_order_insensitive():
    """ADR decision: all_of/any_of are conjunctions/disjunctions —
    reordering predicates must NOT change the id."""
    genome = base_genome()
    genome["regulation"]["look_before_heat_v1"]["express_when"]["all_of"] = [
        {"metric": "consecutive_failures", "op": ">=", "value": 2},
        {"metric": "failing_tools", "op": ">=", "value": 1},
    ]
    reordered = copy.deepcopy(genome)
    reordered["regulation"]["look_before_heat_v1"]["express_when"]["all_of"].reverse()
    assert compute_genome_id(genome) == compute_genome_id(reordered)


def test_absent_and_empty_policy_sections_are_equivalent():
    genome = base_genome()
    del genome["genes"]["execution"]
    empty = copy.deepcopy(genome)
    empty["genes"]["execution"] = {}
    assert compute_genome_id(genome) == compute_genome_id(empty)


def test_absent_and_empty_regulation_are_equivalent():
    genome = base_genome()
    del genome["regulation"]
    empty = copy.deepcopy(genome)
    empty["regulation"] = {}
    assert compute_genome_id(genome) == compute_genome_id(empty)


# -- constitutive regulation canonical equivalence (A2) ----------------------


def test_constitutive_regulation_representations_share_identity():
    """No rule, `{}` and `{"express_when": {}}` are runtime-equivalent
    (constitutive) — all three MUST produce the SAME identity payload and
    genome_id; adding a real predicate must change it."""
    stripped = base_genome()
    del stripped["regulation"]["look_before_heat_v1"]
    empty_rule = copy.deepcopy(stripped)
    empty_rule["regulation"]["look_before_heat_v1"] = {}
    empty_express = copy.deepcopy(stripped)
    empty_express["regulation"]["look_before_heat_v1"] = {"express_when": {}}
    empty_all_of = copy.deepcopy(stripped)
    empty_all_of["regulation"]["look_before_heat_v1"] = {
        "express_when": {"all_of": []}}
    ids = {compute_genome_id(g) for g in
            (stripped, empty_rule, empty_express, empty_all_of)}
    assert len(ids) == 1
    payloads = {canonical_json(canonical_genome_identity_payload(g))
                 for g in (stripped, empty_rule, empty_express, empty_all_of)}
    assert len(payloads) == 1
    # an EFFECTIVE predicate stays in the identity domain
    effective = copy.deepcopy(stripped)
    effective["regulation"]["look_before_heat_v1"] = {"express_when": {
        "all_of": [{"metric": "consecutive_failures", "op": ">=", "value": 2}]}}
    assert compute_genome_id(effective) != ids.pop()
    # any_of-only rule also stays in the identity domain
    any_of_only = copy.deepcopy(stripped)
    any_of_only["regulation"]["look_before_heat_v1"] = {"express_when": {
        "any_of": [{"metric": "failing_tools", "op": ">=", "value": 1}]}}
    assert compute_genome_id(any_of_only) != stripped["genome_id"]


def test_constitutive_forms_are_runtime_equivalent():
    """The three constitutive representations really do behave identically
    at runtime (always active) before they are canonicalized together."""
    from runtime.regulation import Condition, constitutive
    gene = "look_before_heat_v1"
    # absent from the regulation map
    assert constitutive(gene, {}) is True
    # empty condition object
    assert Condition.from_genome({}).is_constitutive() is True
    # empty express_when
    assert Condition.from_genome({"express_when": {}}).is_constitutive() is True
    config = {"genes": {}, "regulation": {gene: {}}}
    assert Condition.from_genome(
        config["regulation"][gene]).evaluate({}) is True


# -- strict container typing: PRESENT malformed fields fail closed -----------


@pytest.mark.parametrize("mutation", [
    pytest.param(lambda g: g.update(genes=[]), id="genes list"),
    pytest.param(lambda g: g["genes"].update(skills={}), id="skills dict"),
    pytest.param(lambda g: g.update(regulation=[]), id="regulation list"),
    pytest.param(lambda g: g["genes"].update(cognition=[]), id="policy section list"),
    pytest.param(lambda g: g["genes"].update(memory=[]), id="memory section list"),
    pytest.param(lambda g: g["regulation"].update(look_before_heat_v1=[]),
                  id="condition list"),
    pytest.param(lambda g: g["regulation"]["look_before_heat_v1"]
        .update(express_when=[]), id="express_when list"),
    pytest.param(lambda g: g["regulation"]["look_before_heat_v1"]["express_when"]
        .update(all_of={}), id="all_of dict"),
    pytest.param(lambda g: g["regulation"]["look_before_heat_v1"]["express_when"]
        .update(any_of="metric>=1"), id="any_of string"),
])
def test_present_malformed_containers_fail_closed(mutation):
    genome = base_genome()
    mutation(genome)
    with pytest.raises(GenomeValidationError, match="when present"):
        compute_genome_id(genome)
    with pytest.raises(GenomeValidationError):
        genome_identity_hash_hex(genome)


def test_absent_field_normalization_still_holds():
    """ABSENT-field normalization is unchanged: absent policy section ==
    {}, absent skills == [], absent regulation == {}."""
    genome = base_genome()
    del genome["genes"]["execution"]
    del genome["genes"]["skills"]
    del genome["regulation"]
    populated = copy.deepcopy(genome)
    populated["genes"]["execution"] = {}
    populated["genes"]["skills"] = []
    populated["regulation"] = {}
    assert compute_genome_id(genome) == compute_genome_id(populated)


# -- hash contract -----------------------------------------------------------


def test_hash_contract_domain_prefix_and_length():
    from trajectory.storage.canonical import canonical_json, sha256_hex
    genome = base_genome()
    payload_json = canonical_json(canonical_genome_identity_payload(genome))
    # exact locked domain separation (ADR-0001 A2)
    expected = sha256_hex(
        f"heritagent:eag-genome-id:v1\n{payload_json}".encode("utf-8"))
    assert genome_identity_hash_hex(genome) == expected
    # NOT the bare payload hash (namespace separation)
    assert genome_identity_hash_hex(genome) != sha256_hex(payload_json.encode("utf-8"))
    genome_id = compute_genome_id(genome)
    assert GENOME_ID_PATTERN.match(genome_id)
    assert len(genome_id) == 2 + GENOME_ID_HEX_CHARS
    assert genome_id == "G-" + expected[:GENOME_ID_HEX_CHARS]
    # full digest is reproducible and the prefix is its head
    assert genome_identity_hash_hex(genome).startswith(genome_id[2:])


def test_recomputation_is_deterministic():
    genome = base_genome()
    ids = {compute_genome_id(genome) for _ in range(8)}
    assert len(ids) == 1
    hashes = {genome_identity_hash_hex(genome) for _ in range(8)}
    assert len(hashes) == 1


def test_input_is_never_mutated():
    genome = base_genome()
    snapshot = copy.deepcopy(genome)
    compute_genome_id(genome)
    canonical_genome_identity_payload(genome)
    validate_genome_id(genome)
    assert genome == snapshot


def test_payload_is_isolated_from_later_mutation():
    genome = base_genome()
    expected = compute_genome_id(genome)
    payload = canonical_genome_identity_payload(genome)
    payload["genes"]["skills"].append(gene_ref("skill", "ghost_v1", 9))
    payload["genes"]["cognition"]["planner"]["artifact"] = "tampered"
    assert compute_genome_id(genome) == expected


# -- security / fail closed --------------------------------------------------


def test_nan_value_fails_closed():
    genome = base_genome()
    genome["regulation"]["look_before_heat_v1"]["express_when"]["all_of"][0][
        "value"] = float("nan")
    with pytest.raises(GenomeValidationError, match="canonicalizable|NaN"):
        compute_genome_id(genome)


def test_infinity_value_fails_closed():
    genome = base_genome()
    genome["regulation"]["look_before_heat_v1"]["express_when"]["all_of"][0][
        "value"] = float("inf")
    with pytest.raises(GenomeValidationError, match="canonicalizable|Infinity"):
        genome_identity_hash_hex(genome)


def test_non_json_value_fails_closed():
    genome = base_genome()
    genome["genes"]["cognition"]["planner"]["artifact"] = {"nested": {1, 2}}
    with pytest.raises(GenomeValidationError, match="canonicalizable"):
        compute_genome_id(genome)


def test_missing_required_fields_fail_closed():
    with pytest.raises(GenomeValidationError, match="schema_version"):
        compute_genome_id({"genes": {}})
    with pytest.raises(GenomeValidationError, match="'genes'"):
        compute_genome_id({"schema_version": "0.1"})


def test_gene_reference_missing_functional_keys_fails_closed():
    with pytest.raises(GenomeValidationError, match="functional identity keys"):
        compute_genome_id({"schema_version": "0.1",
                            "genes": {"skills": [{"gene_id": "x_v1"}]}})


def test_whitespace_and_key_order_cannot_change_the_hash():
    """Attacker-controlled ordering/whitespace in the SOURCE document cannot
    move the hash: canonicalization normalizes before hashing."""
    genome = base_genome()
    scrambled = json.loads(json.dumps(genome))
    # rebuild every dict in reverse insertion order
    def reverse_keys(obj):
        if isinstance(obj, dict):
            return {k: reverse_keys(obj[k]) for k in reversed(list(obj))}
        if isinstance(obj, list):
            return [reverse_keys(v) for v in obj]
        return obj
    scrambled = reverse_keys(scrambled)
    assert compute_genome_id(scrambled) == compute_genome_id(genome)


# -- claimed-ID validation ----------------------------------------------------


def test_valid_computed_id_passes_end_to_end(registry):
    genome = base_genome()
    assert load_genome(genome) == genome  # canonical id accepted


def test_single_hex_flip_is_rejected(registry):
    genome = base_genome()
    valid_id = genome["genome_id"]
    flipped = ("0" if valid_id[-1] != "0" else "1")
    genome["genome_id"] = valid_id[:-1] + flipped
    with pytest.raises(GenomeValidationError, match="canonical content hash"):
        load_genome(genome)


def test_stale_id_after_functional_mutation_rejected(registry):
    """An id computed from OLD content no longer validates after the
    content changed — mutation must recompute (fail closed on forgery)."""
    genome = base_genome()
    genome["genes"]["skills"].append(gene_ref("skill", "extra_v1", 1, DIGEST2))
    with pytest.raises(GenomeValidationError, match="canonical content hash"):
        load_genome(genome)


@pytest.mark.parametrize("bad_id,stage", [
    ("G-deadbeef", "semantic"),                    # too short (8 chars, legacy placeholder shape)
    ("G-0123456789abcdef0", "semantic"),           # too long (17 hex)
    ("0123456789abcdef", "structural"),            # missing G- prefix
    ("X-0123456789abcdef", "structural"),          # wrong prefix letter
    ("G-0123456789ABCDEF", "structural"),          # uppercase hex
    ("G-0123456789abcdeg", "structural"),          # non-hex char
    ("G-", "structural"),                          # empty hash
    (None, "structural"),                          # missing
    (123, "structural"),                           # wrong type
])
def test_malformed_claimed_ids_rejected(bad_id, stage, registry):
    genome = base_genome()
    genome["genome_id"] = bad_id
    if stage == "structural":
        with pytest.raises(GenomeValidationError):
            load_genome(genome)
    else:
        with pytest.raises(GenomeValidationError,
                            match="canonical genome-id contract"):
            load_genome(genome)
    with pytest.raises(GenomeValidationError):
        validate_genome_id(genome)


def test_validate_genome_id_diagnostic_contains_expected_and_actual():
    genome = base_genome()
    genome["genome_id"] = "G-" + "0" * 16
    with pytest.raises(GenomeValidationError) as excinfo:
        validate_genome_id(genome)
    message = json.dumps(excinfo.value.errors)
    assert "G-" + "0" * 16 in message                       # claimed
    assert compute_genome_id(genome) in message             # expected


# -- fixture helper contract (issue #22 §15) ---------------------------------


def test_finalize_genome_helper_sets_valid_id(registry):
    genome = minimal_genome()
    assert GENOME_ID_PATTERN.match(genome["genome_id"])
    validate_genome_id(genome)
    load_genome(genome)


# -- G0 production path -------------------------------------------------------


def test_g0_seed_uses_canonical_content_hash_id(registry):
    from runtime.seed_g0 import build_g0_seed_genome
    genome = build_g0_seed_genome(registry)
    # validator accepts the production seed
    assert load_genome(genome, registry) == genome
    # id is exactly the recomputed content hash (not hard-coded)
    assert genome["genome_id"] == compute_genome_id(genome)
    assert GENOME_ID_PATTERN.match(genome["genome_id"])


def test_g0_id_deterministic_across_registry_locations(tmp_path):
    """The same seed content yields the same id regardless of registry
    storage location (artifact digests are content-addressed)."""
    from genome.validation.registry import TraitRegistry
    from runtime.seed_g0 import build_g0_seed_genome
    first = build_g0_seed_genome(TraitRegistry(tmp_path / "a"))
    second = build_g0_seed_genome(TraitRegistry(tmp_path / "b"))
    assert first["genome_id"] == second["genome_id"]


def test_g0_compiled_runtime_config_carries_canonical_id(registry):
    from runtime.expression import compile_runtime_config
    from runtime.seed_g0 import build_g0_seed_genome
    genome = build_g0_seed_genome(registry)
    config = compile_runtime_config(genome, registry)
    assert config.genome_id == genome["genome_id"]
