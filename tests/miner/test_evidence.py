"""Deterministic evidence renderer tests (issue #9 criterion 3)."""

import json

import pytest

from trajectory.storage.canonical import canonical_json, sha256_hex
from traits.miner.evidence import render_pair_evidence
from tests.miner.conftest import make_pair


def test_render_is_canonical_and_deterministic(tmp_path, registry):
    pair = make_pair(tmp_path, registry)
    text_a, digest_a = render_pair_evidence(pair)
    text_b, digest_b = render_pair_evidence(pair)  # same evidence rendered twice
    assert text_a == text_b
    assert digest_a == digest_b
    assert digest_a == sha256_hex(text_a.encode("utf-8"))
    assert text_a == canonical_json(json.loads(text_a))  # canonical serialization


def test_different_runs_yield_different_evidence(tmp_path, registry):
    """Independent runs have distinct opaque trajectory ids, so their
    evidence digests differ — identity is per-run, not content-derived."""
    pair_a = make_pair(tmp_path / "a", registry)
    pair_b = make_pair(tmp_path / "b", registry)
    _, digest_a = render_pair_evidence(pair_a)
    _, digest_b = render_pair_evidence(pair_b)
    assert digest_a != digest_b


def test_evidence_preserves_failures(tmp_path, registry):
    pair = make_pair(tmp_path, registry)
    text, _ = render_pair_evidence(pair)
    document = json.loads(text)
    failed = document["failed_trajectory"]
    assert failed["label"] == "failure"
    assert failed["outcome"]["success"] is False
    assert failed["steps"][0]["tool_calls"][0]["ok"] is False
    assert failed["steps"][0]["tool_calls"][0]["result_content"] == "heater offline"
    assert document["successful_trajectory"]["label"] == "success"


def test_evidence_excludes_provenance(tmp_path, registry):
    pair = make_pair(tmp_path, registry)
    text, _ = render_pair_evidence(pair)
    for forbidden in ("code_commit", "backend_version", "benchmark", "split",
                       "R-success-ep", "R-failure-ep"):
        assert forbidden not in text, f"{forbidden} leaked into teacher evidence"
    # trajectory ids ARE included (the evidence must be referenceable)
    assert pair.success.trajectory_id in text
    assert pair.failure.trajectory_id in text


def test_evidence_includes_task_family_and_outcomes(tmp_path, registry):
    pair = make_pair(tmp_path / "fam", registry, family="clean_surface")
    text, _ = render_pair_evidence(pair)
    document = json.loads(text)
    assert document["task_family"] == "clean_surface"
    assert document["kind"] == "heritagent-differential-evidence"
