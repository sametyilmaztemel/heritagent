"""EXP-0001 run-start provenance and the cap ledger (issue #13 Phase A;
design section 8 hard caps; ADR-0007)."""

import json

import pytest

from evaluation.exp0001.ledger import CapExceeded, LedgerError, RunLedger
from evaluation.exp0001.protocol import (
    CAPS,
    PROTOCOL,
    attempt_sampling_seed,
    max_model_calls_per_episode,
)
from evaluation.exp0001.provenance import (
    ProvenanceError,
    build_run_start_record,
    canonical_sha256,
    freeze_run_start_record,
    sha256_files,
    verify_run_start_record,
)
from runtime.seed_g0 import build_g0_seed_genome
from tests.conftest import registry  # noqa: F401 (fixture)
from tests.evaluation.test_exp0001_artifacts import load

COMMIT = "a" * 40


class Clock:
    def __init__(self):
        self.now = 1_000.0

    def __call__(self):
        return self.now


def make_record(registry, **overrides):  # noqa: F811
    args = dict(code_commit=COMMIT, code_dirty=False,
                model={"model_id": "Qwen/Qwen2.5-7B-Instruct", "revision": "abc", "backend": "vllm",
                       "backend_version": "0.6", "tokenizer": "Qwen2.5 tokenizer@abc"},
                weights={"sha256": "b" * 64, "files": {"model.safetensors": "c" * 64}},
                split_manifest=load("EXP-0001-splits.json"),
                pilot_manifest=load("EXP-0001-pilot.json"),
                g0_genome=build_g0_seed_genome(registry), run_id="EXP-0001-run-1",
                started_at="2026-10-20T09:00:00Z")
    args.update(overrides)
    return build_run_start_record(**args)


def test_the_record_names_everything_a_result_must_carry(registry):  # noqa: F811
    record = make_record(registry)
    verify_run_start_record(record)
    assert record["code"] == {"commit": COMMIT, "dirty": False}
    assert record["splits"]["manifest_sha256"] == load("EXP-0001-splits.json")["manifest_sha256"]
    assert set(record["splits"]["pools"]) == {"t_mine", "ablation", "interaction", "replay_bank",
                                              "generalization_bank", "t_eval"}
    assert record["g0"]["genome_id"].startswith("G-")
    assert record["generation"]["stop"] == ["\nObservation:"]
    assert record["generation"]["episode_max_steps"] == 50
    assert record["gate"]["ablation_bootstrap"] == {"seed": 0, "resamples": 10_000,
                                                    "cluster_order": "task_id"}
    assert record["final_statistics"]["t_reg"] == "one-set-shared-across-seeds"
    assert record["protocol"]["lifetime_attempts_per_task"] == 3
    assert record["protocol"]["caps"]["model_calls_max"] == 225_000


@pytest.mark.parametrize("override,message", [
    ({"code_commit": "abc"}, "full 40-hex commit"),
    ({"code_dirty": True}, "clean checkout"),
    ({"model": {"model_id": "m", "backend": "vllm"}}, "model.tokenizer"),
    ({"weights": {"sha256": "short"}}, "sha256 of the served weights"),
    ({"run_id": " "}, "run_id"),
])
def test_an_incomplete_record_fails_closed(registry, override, message):  # noqa: F811
    with pytest.raises(ProvenanceError, match=message):
        make_record(registry, **override)


def test_a_non_canonical_g0_fails_closed(registry):  # noqa: F811
    genome = build_g0_seed_genome(registry)
    genome["genome_id"] = "G-" + "0" * 16
    with pytest.raises(ProvenanceError, match="canonical content id"):
        make_record(registry, g0_genome=genome)


def test_the_record_is_written_once(tmp_path, registry):  # noqa: F811
    record = make_record(registry)
    path = tmp_path / "run" / "run_start.json"
    freeze_run_start_record(path, record)
    assert freeze_run_start_record(path, record) == record  # a resumed run: no-op
    other = make_record(registry, started_at="2026-10-21T09:00:00Z")
    with pytest.raises(ProvenanceError, match="already holds another run-start record"):
        freeze_run_start_record(path, other)
    tampered = {**record, "run_id": "x"}
    with pytest.raises(ProvenanceError, match="record_sha256"):
        freeze_run_start_record(tmp_path / "other.json", tampered)


def test_weights_are_hashed_file_by_file(tmp_path):
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    (tmp_path / "config.json").write_text("{}")
    first = sha256_files(tmp_path)
    assert set(first["files"]) == {"model.safetensors", "config.json"}
    (tmp_path / "model.safetensors").write_bytes(b"other weights")
    assert sha256_files(tmp_path)["sha256"] != first["sha256"]
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ProvenanceError, match="no model files"):
        sha256_files(empty)
    with pytest.raises(ProvenanceError, match="not found"):
        sha256_files(tmp_path / "missing")


def test_attempt_seeds_are_fixed_and_distinct():
    seeds = {attempt_sampling_seed(s, f"task-{t}", a)
             for s in PROTOCOL["mining_seeds"] for t in range(60) for a in range(3)}
    assert len(seeds) == 540
    assert attempt_sampling_seed(11, "task-0", 0) == attempt_sampling_seed(11, "task-0", 0)
    assert all(0 <= s < 2 ** 31 for s in seeds)


def test_the_ledger_resumes_and_never_runs_an_episode_twice(tmp_path):
    clock = Clock()
    ledger = RunLedger(tmp_path / "ledger.jsonl", run_id="r", clock=clock)
    assert ledger.admit("lifetime/11/t0/0") is True
    ledger.record("lifetime/11/t0/0", model_calls=12, outcome={"success": True})
    again = RunLedger(tmp_path / "ledger.jsonl", run_id="r", clock=clock)
    assert again.admit("lifetime/11/t0/0") is False and again.model_calls == 12
    assert again.done_record("lifetime/11/t0/0")["outcome"] == {"success": True}
    with pytest.raises(LedgerError, match="already recorded"):
        again.record("lifetime/11/t0/0", model_calls=1, outcome={})
    with pytest.raises(LedgerError, match="belongs to run"):
        RunLedger(tmp_path / "ledger.jsonl", run_id="other", clock=clock)


@pytest.mark.parametrize("caps,used,message", [
    ({**CAPS, "episodes_max": 2}, 2, "episodes_max"),
    ({**CAPS, "model_calls_max": 100}, 1, "model_calls_max"),
])
def test_a_cap_that_would_be_exceeded_stops_the_run(tmp_path, caps, used, message):
    ledger = RunLedger(tmp_path / "ledger.jsonl", run_id="r", clock=Clock(), caps=caps)
    for i in range(used):
        ledger.admit(f"e{i}")
        ledger.record(f"e{i}", model_calls=max_model_calls_per_episode() if message ==
                      "model_calls_max" else 1, outcome={})
    with pytest.raises(CapExceeded, match=message):
        ledger.admit("next")
    assert ledger.state == "incomplete" and message in ledger.reason
    with pytest.raises(LedgerError, match="incomplete"):
        ledger.admit("later")
    reopened = RunLedger(tmp_path / "ledger.jsonl", run_id="r", clock=Clock(), caps=caps)
    assert reopened.state == "incomplete"


def test_the_wall_clock_cap_stops_the_run(tmp_path):
    clock = Clock()
    ledger = RunLedger(tmp_path / "ledger.jsonl", run_id="r", clock=clock)
    clock.now += 96 * 3600 + 1
    with pytest.raises(CapExceeded, match="wall_clock_hours_max"):
        ledger.admit("e0")


def test_a_torn_or_corrupt_ledger_fails_closed(tmp_path):
    path = tmp_path / "ledger.jsonl"
    ledger = RunLedger(path, run_id="r", clock=Clock())
    ledger.admit("e0")
    ledger.record("e0", model_calls=3, outcome={})
    data = path.read_bytes()
    path.write_bytes(data[:-5])
    with pytest.raises(LedgerError, match="torn write"):
        RunLedger(path, run_id="r", clock=Clock())
    path.write_bytes(data + b'{"type": "episode", "key": "e0", "model_calls": 1}\n')
    with pytest.raises(LedgerError, match="duplicate episode key"):
        RunLedger(path, run_id="r", clock=Clock())


def test_a_complete_run_admits_nothing_more(tmp_path):
    ledger = RunLedger(tmp_path / "ledger.jsonl", run_id="r", clock=Clock())
    ledger.complete()
    with pytest.raises(LedgerError, match="complete"):
        ledger.admit("e0")
    assert json.loads((tmp_path / "ledger.jsonl").read_text().splitlines()[-1])["type"] == "complete"
    assert canonical_sha256({"a": 1}) == canonical_sha256({"a": 1})
