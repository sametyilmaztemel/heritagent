"""Content-addressed trait registry: determinism, immutability, collisions (criterion 3)."""

import hashlib

import pytest

from genome.validation.errors import GenomeValidationError, RegistryIntegrityError
from genome.validation.registry import canonical_bytes, parse_uri


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
