"""Content-addressed trait registry: determinism, immutability, collisions (criterion 3)."""

import hashlib

import pytest

from genome.validation.errors import GenomeValidationError, RegistryIntegrityError
from genome.validation.loader import load_genome
from genome.validation.registry import canonical_bytes, parse_uri
from tests.conftest import fake_uri, registered_genome


def test_put_resolve_roundtrip(registry):
    payload = {"b": 2, "a": 1}
    uri = registry.put("policy", "planner_react_v1", 1, payload)
    parts = parse_uri(uri)
    assert parts["kind"] == "policies"
    assert parts["name"] == "planner_react_v1"
    assert parts["version"] == "1"
    assert len(parts["digest"]) == 64
    import json
    assert json.loads(registry.resolve(uri)) == payload


def test_deterministic_canonical_hashing(registry):
    # key insertion order must not affect the digest
    uri_a = registry.put("policy", "p_v1", 1, {"alpha": 1, "beta": [1, 2], "gamma": {"x": 1}})
    uri_b = registry.put("policy", "p_v1", 1, {"gamma": {"x": 1}, "beta": [1, 2], "alpha": 1})
    assert uri_a == uri_b


def test_canonical_bytes_spec():
    assert canonical_bytes({"a": 1, "b": 2}) == b'{"a":1,"b":2}'
    raw = b"\x00\x01binary"
    assert canonical_bytes(raw) is raw


def test_put_is_idempotent_for_identical_content(registry):
    uri1 = registry.put("skill", "s_v1", 1, {"name": "s"})
    uri2 = registry.put("skill", "s_v1", 1, {"name": "s"})
    assert uri1 == uri2


def test_unknown_gene_type_rejected(registry):
    with pytest.raises(GenomeValidationError, match="unknown gene type"):
        registry.put("behaviour", "b_v1", 1, {})


def test_rebinding_identity_to_different_content_rejected(registry):
    registry.put("policy", "p_v1", 1, {"v": 1})
    with pytest.raises(RegistryIntegrityError, match="already bound"):
        registry.put("policy", "p_v1", 1, {"v": 2})


def test_binding_persists_across_registry_instances(tmp_path):
    from genome.validation.registry import TraitRegistry
    registry_a = TraitRegistry(tmp_path / "shared")
    uri_a = registry_a.put("policy", "p_v1", 1, {"v": 1})
    registry_b = TraitRegistry(tmp_path / "shared")
    with pytest.raises(RegistryIntegrityError, match="already bound"):
        registry_b.put("policy", "p_v1", 1, {"v": 2})
    assert registry_b.resolve(uri_a)


def test_content_sharing_across_identities_allowed(registry):
    uri_a = registry.put("policy", "a_v1", 1, {"shared": True})
    uri_b = registry.put("policy", "b_v1", 1, {"shared": True})
    assert uri_a != uri_b
    assert registry.resolve(uri_a) == registry.resolve(uri_b)


def test_verify_binding_catches_forged_uri(registry):
    registry.put("policy", "p_v1", 1, {"v": 1})
    other_uri = registry.put("policy", "q_v1", 1, {"v": 2})
    other_digest = parse_uri(other_uri)["digest"]
    forged = f"registry://policies/p_v1@1/sha256:{other_digest}"
    with pytest.raises(RegistryIntegrityError, match="is bound to"):
        registry.verify_binding(forged)


def test_unbound_identity_rejected_even_if_digest_file_exists(registry):
    # identity A's content is on disk; a forged URI for a never-stored
    # identity B pointing at that digest must be rejected as unbound
    uri_a = registry.put("policy", "p_v1", 1, {"v": 1})
    digest_a = parse_uri(uri_a)["digest"]
    forged = f"registry://policies/z_v9@1/sha256:{digest_a}"
    registry.resolve(forged)  # content-addressed lookup alone still works
    with pytest.raises(RegistryIntegrityError, match="not bound"):
        registry.verify_binding(forged)


def test_loader_rejects_unbound_identity(registry):
    genome = registered_genome(registry)
    planner_digest = genome["genes"]["cognition"]["planner"]["artifact"].split("sha256:")[1]
    ref = genome["genes"]["execution"]["retry_policy"]
    ref["gene_id"] = "brand_new_v1"  # identity never stored via put()
    ref["artifact"] = fake_uri("policies", "brand_new_v1", 1, digest=planner_digest)
    with pytest.raises(RegistryIntegrityError, match="not bound"):
        load_genome(genome, registry)


def test_binding_check_ref_catches_unbound(registry):
    uri_a = registry.put("policy", "p_v1", 1, {"v": 1})
    digest_a = parse_uri(uri_a)["digest"]
    forged_ref = {"gene_id": "z_v9", "version": 1, "type": "policy",
                  "artifact": f"registry://policies/z_v9@1/sha256:{digest_a}"}
    with pytest.raises(RegistryIntegrityError, match="not bound"):
        registry.check_ref(forged_ref)


def test_digest_collision_detected(registry):
    uri = registry.put("policy", "p_v1", 1, {"v": 1})
    digest = parse_uri(uri)["digest"]
    target = registry.root / "policies" / digest
    target.write_bytes(b'{"v": 999}')  # tamper: same digest path, different bytes
    with pytest.raises(RegistryIntegrityError, match="collision"):
        registry.put("policy", "p_v1", 1, {"v": 1})


def test_tampered_artifact_fails_resolve(registry):
    uri = registry.put("policy", "p_v1", 1, {"v": 1})
    digest = parse_uri(uri)["digest"]
    (registry.root / "policies" / digest).write_bytes(b'{"v": 42}')
    with pytest.raises(RegistryIntegrityError, match="tampered"):
        registry.resolve(uri)


def test_resolve_missing_artifact(registry):
    uri = "registry://policies/missing_v1@1/sha256:" + "cd" * 32
    with pytest.raises(GenomeValidationError, match="not found"):
        registry.resolve(uri)


@pytest.mark.parametrize("bad_uri", [
    "foo",
    "registry://policies/p_v1@1/sha256:short",
    "registry://policies/p_v1@0/sha256:" + "ab" * 32,
    "registry://widgets/p_v1@1/sha256:" + "ab" * 32,
    "registry://policies/P_V1@1/sha256:" + "ab" * 32,
    "registry://policies/p_v1@1/sha256:" + "AB" * 32,
])
def test_uri_grammar_enforced(bad_uri):
    with pytest.raises(GenomeValidationError, match="canonical grammar"):
        parse_uri(bad_uri)


def test_check_ref_semantics(registry):
    payload = {"style": "react"}
    uri = registry.put("policy", "planner_react_v1", 1, payload)
    ref = {"gene_id": "planner_react_v1", "version": 1, "type": "policy", "artifact": uri}
    assert registry.check_ref(ref)["digest"] == hashlib.sha256(canonical_bytes(payload)).hexdigest()
