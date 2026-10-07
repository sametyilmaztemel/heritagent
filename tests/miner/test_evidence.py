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


def test_identical_behavior_identical_evidence_across_ids(tmp_path, registry):
    """Identical behavioral evidence under different opaque trajectory ids
    yields IDENTICAL evidence text and hash — ids are provenance, not
    teacher input."""
    pair_a = make_pair(tmp_path / "a", registry,
                        success_trajectory_id="T-aaaa1111aaaa",
                        failure_trajectory_id="T-ffff5555ffff")
    pair_b = make_pair(tmp_path / "b", registry,
                        success_trajectory_id="T-bbbb2222bbbb",
                        failure_trajectory_id="T-eeee3333eeee")
    text_a, digest_a = render_pair_evidence(pair_a)
    text_b, digest_b = render_pair_evidence(pair_b)
    assert text_a == text_b
    assert digest_a == digest_b


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


def test_evidence_excludes_provenance_and_trajectory_ids(tmp_path, registry):
    """Opaque trajectory ids carry no skill-abstraction information and stay
    in provenance/mining records — never in the teacher prompt."""
    pair = make_pair(tmp_path, registry)
    text, _ = render_pair_evidence(pair)
    for forbidden in ("code_commit", "backend_version", "benchmark", "split",
                       "evaluator_id", "evaluator_version",
                       pair.success.trajectory_id, pair.failure.trajectory_id,
                       "R-success-ep", "R-failure-ep"):
        assert forbidden not in text, f"{forbidden} leaked into teacher evidence"
    # behavioral evidence and outcome labels ARE included
    assert pair.success.task in text
    assert '"success":true' in text and '"success":false' in text


def test_evidence_includes_task_family_and_outcomes(tmp_path, registry):
    pair = make_pair(tmp_path / "fam", registry, family="clean_surface")
    text, _ = render_pair_evidence(pair)
    document = json.loads(text)
    assert document["task_family"] == "clean_surface"
    assert document["kind"] == "heritagent-differential-evidence"
