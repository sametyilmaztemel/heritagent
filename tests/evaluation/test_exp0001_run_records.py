"""EXP-0001 run-start provenance and the cap ledger (issue #13 Phase A;
design section 8 hard caps; ADR-0007)."""

import json

import pytest

from evaluation.exp0001.ledger import (
    CapExceeded,
    LedgerError,
    RunLedger,
    append_note,
    repair_torn_tail,
)
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
                teacher={"model_id": "Qwen/Qwen2.5-7B-Instruct", "revision": "abc",
                         "backend": "vllm", "backend_version": "0.6"},
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
    assert record["environment"] is None  # the CLI names it; see the next test


def test_the_record_names_the_software_and_the_data_it_ran_on(registry):  # noqa: F811
    environment = {"software": {"vllm": "0.29.0+cu129", "transformers": "5.19.0"},
                   "data": {"alfworld_release": "0.2.2", "archives": {"a.zip": "e" * 64},
                            "games_verified": 338}}
    record = make_record(registry, environment=environment)
    verify_run_start_record(record)
    assert record["environment"] == environment
    assert record["record_sha256"] != make_record(registry)["record_sha256"]


@pytest.mark.parametrize("override,message", [
    ({"code_commit": "abc"}, "full 40-hex commit"),
    ({"code_dirty": True}, "clean checkout"),
    ({"model": {"model_id": "m", "revision": "r", "backend": "vllm"}}, "model.tokenizer"),
    ({"model": {"model_id": "m", "backend": "vllm", "tokenizer": "t"}}, "model.revision"),
    ({"teacher": {"model_id": "m", "revision": " ", "backend": "vllm"}}, "teacher.revision"),
    ({"weights": {"sha256": "short"}}, "sha256 .64 hex. of the served weights"),
    ({"weights": {"sha256": "zz" * 32}}, "sha256 .64 hex. of the served weights"),
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


def test_teacher_calls_count_against_the_cap_but_are_not_episodes(tmp_path):
    caps = {**CAPS, "model_calls_max": max_model_calls_per_episode() + 40}
    ledger = RunLedger(tmp_path / "ledger.jsonl", run_id="r", clock=Clock(), caps=caps)
    assert ledger.admit_calls("mining/11", 40) is True
    ledger.record_calls("mining/11", model_calls=40)
    assert (ledger.episodes, ledger.model_calls) == (0, 40)
    reopened = RunLedger(tmp_path / "ledger.jsonl", run_id="r", clock=Clock(), caps=caps)
    assert reopened.admit_calls("mining/11", 40) is False and reopened.model_calls == 40
    assert reopened.is_charged("mining/11") and not reopened.is_done("mining/11")
    with pytest.raises(LedgerError, match="already recorded"):
        reopened.record_calls("mining/11", model_calls=1)
    assert reopened.admit("e0") is True  # 40 + one episode's worst case fits
    reopened.record("e0", model_calls=1, outcome={})
    with pytest.raises(CapExceeded, match="model_calls_max"):
        reopened.admit_calls("mining/23", max_model_calls_per_episode())


def test_a_protocol_stop_marks_the_run_incomplete_with_its_reason(tmp_path):
    ledger = RunLedger(tmp_path / "ledger.jsonl", run_id="r", clock=Clock())
    with pytest.raises(LedgerError, match="reason"):
        ledger.abandon(" ")
    ledger.abandon("t_reg: only 39 T_mine tasks were solved")
    assert ledger.state == "incomplete" and ledger.reason.startswith("t_reg")
    with pytest.raises(LedgerError, match="incomplete"):
        ledger.admit("e0")
    with pytest.raises(LedgerError, match="already incomplete"):
        ledger.abandon("again")
    reopened = RunLedger(tmp_path / "ledger.jsonl", run_id="r", clock=Clock())
    assert reopened.state == "incomplete" and reopened.reason.startswith("t_reg")


def test_the_record_binds_the_pre_registered_artifacts(registry):  # noqa: F811
    import copy
    splits, pilot = load("EXP-0001-splits.json"), load("EXP-0001-pilot.json")
    tampered = copy.deepcopy(splits)  # a T_eval game swapped into T_mine, hashes left as they were
    tampered["pools"]["t_mine"]["games"][0], tampered["pools"]["t_eval"]["games"][0] = \
        tampered["pools"]["t_eval"]["games"][0], tampered["pools"]["t_mine"]["games"][0]
    with pytest.raises(ProvenanceError, match="does not verify"):
        make_record(registry, split_manifest=tampered)
    stale_pilot = copy.deepcopy(pilot)
    stale_pilot["games"] = stale_pilot["games"][1:]
    with pytest.raises(ProvenanceError, match="does not verify"):
        make_record(registry, pilot_manifest=stale_pilot)
    with pytest.raises(ProvenanceError, match="not the pre-registered one"):
        make_record(registry, expected_split_sha256="0" * 64)
    with pytest.raises(ProvenanceError, match="not the pre-registered one"):
        make_record(registry, expected_pilot_sha256="0" * 64)
    record = make_record(registry)
    assert record["teacher"]["model_id"] == "Qwen/Qwen2.5-7B-Instruct"
    assert record["model"]["revision"] == "abc"


def test_the_record_carries_every_frozen_input(registry):  # noqa: F811
    record = make_record(registry)
    assert set(record) == {"schema", "experiment", "run_id", "started_at", "code", "model",
                           "teacher", "weights", "splits", "pilot", "miner", "g0", "candidate_set",
                           "gate", "final_statistics", "generation", "protocol",
                           "required_at_run_time", "environment", "record_sha256"}
    assert record["pilot"]["pilot_sha256"] == load("EXP-0001-pilot.json")["pilot_sha256"]
    assert len(record["miner"]["prompt_sha256"]) == 64 == len(record["miner"]["output_schema_sha256"])
    assert record["candidate_set"] == {"cap": 10, "ranking": ["miner-estimated-generality",
                                                             "discovery-order"]}
    assert record["gate"]["threshold_version"] == "EXP-0001-CIG-M0-v1"
    assert record["splits"]["plan"]["seed"] == 0
    assert record["weights"]["sha256"] == "b" * 64
    assert record["protocol"]["budgets"] == {"max_steps": 50, "max_total_retries": 10,
                                             "max_tokens_per_request": 512}


def test_a_frozen_artifact_is_never_replaced(tmp_path):
    from evaluation.exp0001.provenance import freeze_json
    path = tmp_path / "gate.json"
    assert freeze_json(path, {"a": (1, 2)}) == {"a": [1, 2]}
    assert freeze_json(path, {"a": [1, 2]}) == {"a": [1, 2]}  # identical: a no-op
    with pytest.raises(ProvenanceError, match="frozen with other content"):
        freeze_json(path, {"a": [1, 3]})


def test_the_keyed_hash_rules_match_the_adr_literally():
    """Golden values computed with hashlib from the ADR-0007 formulas, not
    with the production functions (E1, E3, E6)."""
    from evaluation.exp0001.allocations import generalization_rank
    from evaluation.exp0001.lifetime import t_reg_rank
    game = "pick_and_place_simple-Box-None-ArmChair-212/trial_T20190908_214946_567644"
    assert attempt_sampling_seed(11, game, 0) == 1009155567
    assert attempt_sampling_seed(41, "x", 2) == 1584960060
    assert t_reg_rank("x") == "6aea84556f329decfc95454ae28e8dee42de8490eb7b51cd383673e094221847"
    assert generalization_rank("pick_and_place_simple", "x") == \
        "fc7c49b643c2ae28764d98b3ad381f09f9394b29555f219bddaece3505e405ce"


def test_the_protocol_constants_are_the_manifests():
    """pyyaml is not a test dependency: the locked lines are read with patterns."""
    import re
    from evaluation.exp0001.protocol import (
        CANDIDATE_CAP,
        LOCKED_PILOT_SHA256,
        LOCKED_SPLIT_MANIFEST_SHA256,
        MINING_SEEDS,
    )
    from tests.evaluation.test_exp0001_artifacts import MANIFESTS
    text = (MANIFESTS / "EXP-0001-minimal-cig.yaml").read_text(encoding="utf-8")
    for name in ("episodes_max", "model_calls_max", "wall_clock_hours_max", "accelerator_nodes_max"):
        assert int(re.search(rf"^\s+{name}: (\d+)", text, re.MULTILINE).group(1)) == CAPS[name]
    assert re.search(r"^\s+mining_seeds: \[11, 23, 41\]", text, re.MULTILINE)
    assert MINING_SEEDS == (11, 23, 41)
    assert int(re.search(r"^\s+cap: (\d+)\s+# K", text, re.MULTILINE).group(1)) == CANDIDATE_CAP
    assert f"sha256: {LOCKED_SPLIT_MANIFEST_SHA256}" in text and \
        f"sha256: {LOCKED_PILOT_SHA256}" in text
    assert load("EXP-0001-splits.json")["manifest_sha256"] == LOCKED_SPLIT_MANIFEST_SHA256
    assert load("EXP-0001-pilot.json")["pilot_sha256"] == LOCKED_PILOT_SHA256


def test_an_aborted_attempt_is_charged_its_reservation(tmp_path):
    path = tmp_path / "ledger.jsonl"
    ledger = RunLedger(path, run_id="r", clock=Clock())
    assert ledger.admit("e0")  # the process dies during e0
    reopened = RunLedger(path, run_id="r", clock=Clock())
    worst = max_model_calls_per_episode()
    assert (reopened.aborted_attempts, reopened.model_calls, reopened.episodes) == (1, worst, 0)
    assert reopened.admit("e0")  # run again
    reopened.record("e0", model_calls=3, outcome={})
    assert (reopened.aborted_attempts, reopened.model_calls) == (1, worst + 3)
    again = RunLedger(path, run_id="r", clock=Clock())
    assert (again.aborted_attempts, again.model_calls, again.episodes) == (1, worst + 3, 1)


def test_frozen_teacher_calls_without_a_charge_finish_the_aborted_attempt(tmp_path):
    """A crash after the candidate set was frozen but before its calls were
    recorded: the resumed run records them, and the aborted reservation is
    replaced, as a reload reads it."""
    path = tmp_path / "ledger.jsonl"
    RunLedger(path, run_id="r", clock=Clock()).admit_calls("mining/11", 40)
    resumed = RunLedger(path, run_id="r", clock=Clock())
    assert (resumed.aborted_attempts, resumed.model_calls) == (1, 40)
    resumed.record_calls("mining/11", model_calls=38)
    assert (resumed.aborted_attempts, resumed.model_calls) == (0, 38)
    again = RunLedger(path, run_id="r", clock=Clock())
    assert (again.aborted_attempts, again.model_calls) == (0, 38)


def test_a_reopened_ledger_keeps_its_caps_and_counts_active_run_time_only(tmp_path):
    """ADR-0008 (owner decision): the wall-clock cap counts the sessions'
    spans, not the time between them."""
    path, clock = tmp_path / "ledger.jsonl", Clock()
    first = RunLedger(path, run_id="r", clock=clock)
    with pytest.raises(LedgerError, match="other caps"):
        RunLedger(path, run_id="r", clock=clock, caps={**CAPS, "episodes_max": 10 ** 6})
    first.admit("e0")
    clock.now += 50 * 3600
    first.record("e0", model_calls=1, outcome={})  # session 1: 50 active hours
    clock.now += 1000 * 3600  # the notebook was disconnected: not run time
    second = RunLedger(path, run_id="r", clock=clock)
    assert second.active_hours() == pytest.approx(50)
    clock.now += 46 * 3600  # 96 active hours: exactly the cap, not exceeded
    assert not second.admission_blocked("e1") and second.admit("e1")
    clock.now += 1
    assert second.admission_blocked("e2") and second.state == "running"  # asking changes nothing
    with pytest.raises(CapExceeded, match=r"wall_clock_hours_max 96 exceeded \(96.00 active"):
        second.admit("e2")
    reader = RunLedger(path, run_id="r", clock=clock, readonly=True)
    assert [s["index"] for s in reader.sessions] == [1, 2] and reader.state == "incomplete"
    assert reader.active_hours() == pytest.approx(96 + 1 / 3600)


def test_a_crashed_session_counts_up_to_its_last_record(tmp_path):
    path, clock = tmp_path / "ledger.jsonl", Clock()
    ledger = RunLedger(path, run_id="r", clock=clock)
    ledger.admit("e0")
    clock.now += 3600
    ledger.record("e0", model_calls=1, outcome={})
    clock.now += 7200  # the process dies two hours after its last record
    assert RunLedger(path, run_id="r", clock=clock, readonly=True).active_hours() == \
        pytest.approx(1)


def test_a_session_records_its_info_and_a_reader_writes_nothing(tmp_path):
    path = tmp_path / "ledger.jsonl"
    gpu = {"name": "NVIDIA A100-SXM4-80GB", "memory_mib": 81920}
    RunLedger(path, run_id="r", clock=Clock(), session_info={"accelerator": gpu})
    size = path.stat().st_size
    reader = RunLedger(path, run_id="r", clock=Clock(), readonly=True)
    assert path.stat().st_size == size
    assert reader.sessions[0]["info"] == {"accelerator": gpu}
    with pytest.raises(LedgerError, match="no session|read-only"):
        reader.admit("e0")
    with pytest.raises(LedgerError, match="does not exist"):
        RunLedger(tmp_path / "missing.jsonl", run_id="r", clock=Clock(), readonly=True)
    RunLedger(path, run_id="r", clock=Clock()).complete()
    size = path.stat().st_size
    RunLedger(path, run_id="r", clock=Clock())  # a finished run opens no session
    assert path.stat().st_size == size


def test_the_caps_count_every_attempt_in_flight(tmp_path):
    """ADR-0008: parallel episodes never pass a cap that the same episodes
    run one at a time would respect."""
    worst = max_model_calls_per_episode()
    ledger = RunLedger(tmp_path / "a.jsonl", run_id="r", clock=Clock(),
                       caps={**CAPS, "episodes_max": 3})
    assert ledger.admit("e0") and ledger.admit("e1") and ledger.in_flight == 2
    assert not ledger.admission_blocked("e2") and ledger.admit("e2")
    assert ledger.admission_blocked("e3")  # 0 recorded + 3 in flight + 1 > 3
    ledger.record("e0", model_calls=1, outcome={})
    assert ledger.admission_blocked("e3")  # 1 recorded + 2 in flight + 1 > 3
    ledger.abort("e1")  # e1 raised: charged its worst case, no longer in flight
    assert ledger.in_flight == 1 and ledger.model_calls == 1 + worst
    assert not ledger.admission_blocked("e3") and ledger.admit("e3")
    with pytest.raises(CapExceeded, match="episodes_max 3"):
        ledger.admit("e4")

    ledger = RunLedger(tmp_path / "b.jsonl", run_id="r", clock=Clock(),
                       caps={**CAPS, "model_calls_max": 2 * worst + 5})
    ledger.admit("e0")
    ledger.admit("e1")
    assert ledger.admission_blocked("e2")  # 2 reservations in flight + the next worst case
    ledger.record("e0", model_calls=5, outcome={})
    assert not ledger.admission_blocked("e2")  # 5 used + 1 in flight + the next: exactly the cap
    assert ledger.admission_blocked("e0") is False  # a recorded key is resumed, never blocked
    ledger.admit("e2")
    with pytest.raises(CapExceeded, match=r"used 5, reserved in flight 120"):
        ledger.admit_calls("mining/11", 1)


def test_a_torn_tail_is_cut_off_on_purpose_and_kept_on_record(tmp_path):
    path = tmp_path / "ledger.jsonl"
    ledger = RunLedger(path, run_id="r", clock=Clock())
    ledger.admit("e0")
    ledger.record("e0", model_calls=3, outcome={})
    ledger.admit("e1")
    torn = b'{"type":"episode","key":"e1"'  # the crash interrupted e1's record
    path.write_bytes(path.read_bytes() + torn)
    with pytest.raises(LedgerError, match="torn write"):
        RunLedger(path, run_id="r", clock=Clock())
    note = repair_torn_tail(path)
    assert note["removed_bytes"] == len(torn)
    assert (tmp_path / note["saved_as"]).read_bytes() == torn
    assert repair_torn_tail(path) is None  # nothing left to repair
    again = RunLedger(path, run_id="r", clock=Clock())
    assert [n["type"] for n in again.notes] == ["repair"]
    worst = max_model_calls_per_episode()
    assert (again.episodes, again.aborted_attempts, again.model_calls) == (1, 1, 3 + worst)
    header_only = tmp_path / "torn-header.jsonl"
    header_only.write_bytes(b'{"type":"hea')
    with pytest.raises(LedgerError, match="header itself is torn"):
        repair_torn_tail(header_only)


def test_notes_change_no_count_and_only_notes_are_appended(tmp_path):
    path = tmp_path / "ledger.jsonl"
    ledger = RunLedger(path, run_id="r", clock=Clock())
    ledger.admit("e0")
    ledger.record("e0", model_calls=3, outcome={})
    append_note(path, {"type": "restore", "snapshot_seq": 7})
    again = RunLedger(path, run_id="r", clock=Clock())
    assert (again.episodes, again.model_calls, again.notes[0]["snapshot_seq"]) == (1, 3, 7)
    with pytest.raises(LedgerError, match="a note is one of"):
        append_note(path, {"type": "episode", "key": "e9", "model_calls": 0})


@pytest.mark.parametrize("line,message", [
    (b'{"type":"attempt","key":"e9","reserved":60}\n', "no valid clock"),
    (b'{"type":"attempt","key":"e9","reserved":60,"clock":"now"}\n', "no valid clock"),
    (b'{"type":"session","index":5,"clock":1.0}\n', "out of order"),
])
def test_records_without_a_clock_or_session_order_fail_closed(tmp_path, line, message):
    path = tmp_path / "ledger.jsonl"
    RunLedger(path, run_id="r", clock=Clock())
    path.write_bytes(path.read_bytes() + line)
    with pytest.raises(LedgerError, match=message):
        RunLedger(path, run_id="r", clock=Clock())


def test_a_record_outside_a_session_fails_closed(tmp_path):
    path = tmp_path / "ledger.jsonl"
    RunLedger(path, run_id="r", clock=Clock())
    header = path.read_bytes().splitlines()[0]
    path.write_bytes(header + b'\n{"type":"attempt","key":"e0","reserved":60,"clock":1.0}\n')
    with pytest.raises(LedgerError, match="outside a session"):
        RunLedger(path, run_id="r", clock=Clock())


def test_nothing_follows_a_terminal_record(tmp_path):
    path = tmp_path / "ledger.jsonl"
    ledger = RunLedger(path, run_id="r", clock=Clock())
    ledger.complete()
    path.write_bytes(path.read_bytes() + b'{"type": "attempt", "key": "e0", "reserved": 60}\n')
    with pytest.raises(LedgerError, match="follows the terminal state"):
        RunLedger(path, run_id="r", clock=Clock())


@pytest.mark.parametrize("line,message", [
    (b'{"type": "attempt", "key": "e9", "reserved": -1}\n', "bad attempt record"),
    (b'{"type": "attempt", "key": "e9", "reserved": 1.5}\n', "bad attempt record"),
    (b'{"type": "attempt", "key": "e9", "reserved": true}\n', "bad attempt record"),
    (b'{"type": "attempt", "key": 7, "reserved": 60}\n', "bad attempt record"),
    (b'{"type": "attempt", "key": "e0", "reserved": 60}\n', "attempt after"),
    (b'{"type": "attempt", "key": "mining/11", "reserved": 40}\n', "attempt after"),
])
def test_bad_attempt_records_fail_closed(tmp_path, line, message):
    path = tmp_path / "ledger.jsonl"
    ledger = RunLedger(path, run_id="r", clock=Clock())
    ledger.admit("e0")
    ledger.record("e0", model_calls=1, outcome={})
    ledger.admit_calls("mining/11", 40)
    ledger.record_calls("mining/11", model_calls=40)
    path.write_bytes(path.read_bytes() + line)
    with pytest.raises(LedgerError, match=message):
        RunLedger(path, run_id="r", clock=Clock())


def test_notes_may_follow_a_terminal_record(tmp_path):
    """A finished run restored from its mirror gets a restore note after its
    terminal record; it must still load (review of #69)."""
    for finish in ("complete", "abandon"):
        path = tmp_path / f"{finish}.jsonl"
        ledger = RunLedger(path, run_id="r", clock=Clock())
        ledger.admit("e0")
        ledger.record("e0", model_calls=2, outcome={})
        ledger.complete() if finish == "complete" else ledger.abandon("t_reg: short")
        append_note(path, {"type": "restore", "snapshot_seq": 3})
        path.write_bytes(path.read_bytes() + b'{"type":"repa')
        assert repair_torn_tail(path)["type"] == "repair"
        for readonly in (True, False):
            again = RunLedger(path, run_id="r", clock=Clock(), readonly=readonly)
            assert again.state == ("complete" if finish == "complete" else "incomplete")
            assert [n["type"] for n in again.notes] == ["restore", "repair"]
            assert again.episodes == 1


def test_a_heartbeat_keeps_the_session_span_current(tmp_path):
    path, clock = tmp_path / "ledger.jsonl", Clock()
    ledger = RunLedger(path, run_id="r", clock=clock)
    ledger.admit_calls("mining/11", 40)
    for _ in range(3):
        clock.now += 600
        ledger.heartbeat()  # one per teacher call
    reader = RunLedger(path, run_id="r", clock=clock, readonly=True)
    assert reader.active_hours() == pytest.approx(0.5) and reader.model_calls == 40


def test_a_deferred_admission_writes_nothing(tmp_path):
    from evaluation.exp0001.ledger import AdmissionDeferred
    path = tmp_path / "ledger.jsonl"
    ledger = RunLedger(path, run_id="r", clock=Clock(), caps={**CAPS, "episodes_max": 1})
    ledger.admit("e0")
    size = path.stat().st_size
    with pytest.raises(AdmissionDeferred, match="episodes_max 1"):
        ledger.admit("e1", defer=True)
    assert path.stat().st_size == size and ledger.state == "running" and ledger.in_flight == 1
    ledger.record("e0", model_calls=1, outcome={})
    with pytest.raises(CapExceeded):
        ledger.admit("e1")  # nothing in flight: the stop is written
    assert ledger.done_keys("e") == ("e0",)
