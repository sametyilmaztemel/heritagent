"""The Colab execution pieces of EXP-0001 (issue #13 Phase B; ADR-0008):
the run mirror, the vLLM server adapter (against a local HTTP server), the
A100 80 GB check, the calibration pilot and the command line. No GPU, no
vLLM, no ALFWorld."""

import json
import pickle
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

import evaluation.exp0001.mirror as mirror_module
from evaluation.exp0001 import cli
from evaluation.exp0001.backends import (
    AlfworldEnvFactory,
    BackendError,
    gpu_inventory,
    require_a100_80gb,
)
from evaluation.exp0001.episodes import EpisodeExecutor
from evaluation.exp0001.ledger import RunLedger
from evaluation.exp0001.mirror import MirrorError, RunMirror, latest_snapshot, restore
from evaluation.exp0001.pilot import PILOT_MEASURES
from evaluation.exp0001.pilot_run import (
    GATE_EPISODES_PER_CANDIDATE,
    SMOKE_GENE_ID,
    pilot_specs,
    project_exp0001,
    run_pilot,
)
from evaluation.exp0001.protocol import CAPS
from runtime.model_adapters.base import GenerationSettings, ModelMessage, ToolSpec
from runtime.model_adapters.errors import BackendUnavailableError, ModelResponseError
from runtime.model_adapters.openai_server import (
    NEUTRAL_SAMPLING,
    ServerRequestError,
    VLLMServerAdapter,
)
from runtime.seed_g0 import g0_runtime_config
from tests.conftest import registry  # noqa: F401 (fixture)
from tests.evaluation.exp0001_workers import GameFactory, WinsOnGames
from tests.evaluation.test_exp0001_lifetime import Clock
from tests.evaluation.test_exp0001_pipeline import (  # noqa: F401 (fixtures)
    PILOTS,
    clean,
    make_run,
    splits,
)


# -- the run mirror ---------------------------------------------------------------------
def files(root):
    return {p.relative_to(root).as_posix(): p.read_bytes()
            for p in sorted(root.rglob("*")) if p.is_file()}


def test_a_snapshot_restores_byte_for_byte_and_leaves_a_restore_note(tmp_path):
    local, drive = tmp_path / "local", tmp_path / "drive"
    ledger = RunLedger(local / "ledger.jsonl", run_id="r", clock=Clock())
    ledger.admit("e0")
    ledger.record("e0", model_calls=2, outcome={})
    (local / "trajectories").mkdir()
    (local / "trajectories" / "T-1.jsonl").write_text("one\n")
    (local / ".lock").write_text("")
    mirror = RunMirror(local, drive)
    assert mirror.snapshot() == 1
    (local / "trajectories" / "T-2.jsonl").write_text("two\n")
    assert mirror.snapshot() == 2
    expected = {k: v for k, v in files(local).items() if k != ".lock"}
    doc = restore(drive, tmp_path / "restored")
    assert doc["seq"] == 2
    restored = files(tmp_path / "restored")
    ledger_bytes = restored.pop("ledger.jsonl")
    assert ledger_bytes.startswith(expected.pop("ledger.jsonl")) and restored == expected
    reopened = RunLedger(tmp_path / "restored" / "ledger.jsonl", run_id="r", clock=Clock())
    assert reopened.notes[0]["type"] == "restore" and reopened.notes[0]["snapshot_seq"] == 2
    assert reopened.episodes == 1
    with pytest.raises(MirrorError, match="not empty"):
        restore(drive, tmp_path / "restored")


def test_a_crash_during_a_snapshot_leaves_the_previous_one_whole(tmp_path, monkeypatch):
    local, drive = tmp_path / "local", tmp_path / "drive"
    local.mkdir()
    (local / "a.json").write_text("1")
    warnings = []
    mirror = RunMirror(local, drive, warn=warnings.append)
    mirror.snapshot()
    (local / "a.json").write_text("22")
    real = mirror_module._write_atomic

    def dies_at_the_manifest(path, data):
        if path.name.startswith("snapshot-"):
            raise OSError("Drive went away")
        real(path, data)

    monkeypatch.setattr(mirror_module, "_write_atomic", dies_at_the_manifest)
    assert mirror.snapshot() is None and mirror.failures == 1 and "Drive went away" in warnings[0]
    monkeypatch.setattr(mirror_module, "_write_atomic", real)
    assert latest_snapshot(drive)["seq"] == 1
    restore(drive, tmp_path / "restored")
    assert (tmp_path / "restored" / "a.json").read_text() == "1"
    assert mirror.snapshot() == 2  # the next snapshot retries


def test_a_corrupt_blob_or_a_second_writer_fails_closed(tmp_path):
    local, drive = tmp_path / "local", tmp_path / "drive"
    local.mkdir()
    (local / "a.json").write_text("1")
    first = RunMirror(local, drive)
    first.snapshot()
    blob = next((drive / "blobs").glob("*/*"))
    blob.write_text("tampered")
    with pytest.raises(MirrorError, match="does not match its hash"):
        restore(drive, tmp_path / "restored")
    RunMirror(local, drive)  # another process takes over the mirror
    with pytest.raises(MirrorError, match="another process took over"):
        first.snapshot()


def test_snapshots_are_throttled_and_old_blobs_collected(tmp_path, monkeypatch):
    local, drive = tmp_path / "local", tmp_path / "drive"
    local.mkdir()
    now = [0.0]
    mirror = RunMirror(local, drive, min_interval_seconds=60, clock=lambda: now[0])
    (local / "a.json").write_text("1")
    mirror.on_append({"type": "attempt"})
    mirror.on_append({"type": "attempt"})  # within the interval: no snapshot
    assert mirror.seq == 1
    now[0] = 61.0
    mirror.on_append({"type": "episode"})
    assert mirror.seq == 2
    mirror.on_append({"type": "complete"})  # a terminal record always snapshots
    assert mirror.seq == 3
    monkeypatch.setattr(mirror_module, "GC_EVERY", 1)
    for content in ("2", "3", "4"):
        (local / "a.json").write_text(content)
        mirror.snapshot()
    assert len(list(drive.glob("snapshot-*.json"))) == 2
    blobs = {b.read_text() for b in (drive / "blobs").glob("*/*")}
    assert blobs == {"3", "4"}  # only what the two kept snapshots reference


def test_a_run_restored_from_its_mirror_resumes_to_the_same_report(tmp_path, clean, splits):  # noqa: F811
    local, drive = tmp_path / "local", tmp_path / "drive"
    mirror = RunMirror(local, drive, min_interval_seconds=0)
    from tests.evaluation.test_exp0001_pipeline import Agent
    with pytest.raises(RuntimeError, match="backend died"):
        make_run(local, splits, agent=Agent(fail_at=40),
                 on_ledger_append=mirror.on_append).run()
    restore(drive, tmp_path / "vm2")  # a new VM: the local directory is gone
    report = make_run(tmp_path / "vm2", splits).run()
    # the restored snapshot was taken before the failed attempt's reservation
    # could be written, or after: either way the outcome is the clean run's
    assert {**report, "aborted_attempts": 0, "model_calls": 0} == \
        {**clean[1], "aborted_attempts": 0, "model_calls": 0}
    ledger = RunLedger(tmp_path / "vm2" / "ledger.jsonl", run_id=report_run_id(tmp_path / "vm2"),
                       clock=Clock(), readonly=True, caps=CAPS)
    assert [n["type"] for n in ledger.notes] == ["restore"]


def report_run_id(run_dir):
    return json.loads((run_dir / "run-start.json").read_text())["run_id"]


# -- the vLLM server adapter -----------------------------------------------------------------
class Server:
    """A local chat-completions server: ``replies`` are (status, body) in order."""

    def __init__(self, replies):
        self.replies, self.requests = list(replies), []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def _reply(self):
                length = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(length)) if length else None
                outer.requests.append((self.command, self.path, body))
                status, payload = outer.replies.pop(0)
                data = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            do_GET = do_POST = _reply

            def log_message(self, *args):
                pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()


def completion(text, prompt=11, completion_tokens=7):
    return (200, {"choices": [{"message": {"role": "assistant", "content": text},
                               "finish_reason": "stop"}],
                  "usage": {"prompt_tokens": prompt, "completion_tokens": completion_tokens}})


def adapter(server, **kwargs):
    return VLLMServerAdapter(server.url, "Qwen/Qwen2.5-7B-Instruct", "a" * 40,
                             backend_version="0.29.0", sleep=lambda _: None, **kwargs)


MESSAGES = [ModelMessage("system", "s"), ModelMessage("user", "u")]
SETTINGS = GenerationSettings(temperature=0.0, max_tokens=512, seed=11,
                              stop=("\nObservation:",))


def test_a_generation_sends_explicit_and_neutral_settings():
    server = Server([completion("Thought: ok\nFinal Answer: x")])
    try:
        result = adapter(server).generate(MESSAGES, SETTINGS)
    finally:
        server.close()
    assert (result.text, result.prompt_tokens, result.completion_tokens) == \
        ("Thought: ok\nFinal Answer: x", 11, 7)
    method, path, body = server.requests[0]
    assert (method, path) == ("POST", "/v1/chat/completions")
    assert body["messages"] == [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    assert (body["model"], body["temperature"], body["max_tokens"], body["seed"], body["stop"],
            body["n"]) == ("Qwen/Qwen2.5-7B-Instruct", 0.0, 512, 11, ["\nObservation:"], 1)
    assert {k: body[k] for k in NEUTRAL_SAMPLING} == NEUTRAL_SAMPLING
    assert body["repetition_penalty"] == 1.0 and body["top_k"] == 0  # never the model's defaults


def test_structured_output_and_tool_calls_are_constrained_and_validated():
    schema = {"type": "object", "required": ["a"], "properties": {"a": {"type": "integer"}}}
    tools = [ToolSpec("act", "act", {"type": "object", "required": ["command"],
                                     "properties": {"command": {"type": "string"}}})]
    server = Server([completion('{"a": 1}'), completion('{"a": "x"}'),
                     completion('{"name": "act", "arguments": {"command": "look"}}')])
    try:
        model = adapter(server)
        assert json.loads(model.structured_generate(MESSAGES, SETTINGS, schema).text) == {"a": 1}
        with pytest.raises(ModelResponseError):
            model.structured_generate(MESSAGES, SETTINGS, schema)
        call = model.tool_call(MESSAGES, SETTINGS, tools)
    finally:
        server.close()
    assert (call.name, call.arguments) == ("act", {"command": "look"})
    response_format = server.requests[0][2]["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["schema"] == schema


def test_transport_failures_retry_then_fail_closed_and_rejections_do_not_retry():
    server = Server([(503, {"error": "busy"}), (429, {"error": "slow down"}),
                     completion("Thought: ok\nFinal Answer: x"), (400, {"error": "too long"})])
    try:
        model = adapter(server)
        assert model.generate(MESSAGES, SETTINGS).text.endswith("x")
        with pytest.raises(ServerRequestError, match="rejected \\(400\\)"):
            model.generate(MESSAGES, SETTINGS)
    finally:
        server.close()
    assert len(server.requests) == 4
    gone = VLLMServerAdapter("http://127.0.0.1:9", "m", "a" * 40, transport_retries=2,
                             sleep=lambda _: None, timeout_seconds=2)
    with pytest.raises(BackendUnavailableError, match="after 3 attempts"):
        gone.generate(MESSAGES, SETTINGS)


def test_the_adapter_checks_the_served_model_and_pickles():
    server = Server([(200, {"data": [{"id": "Qwen/Qwen2.5-7B-Instruct"}]}),
                     (200, {"data": [{"id": "another"}]})])
    try:
        model = adapter(server)
        model.require_served()
        with pytest.raises(BackendUnavailableError, match="not 'Qwen"):
            model.require_served()
    finally:
        server.close()
    copy = pickle.loads(pickle.dumps(VLLMServerAdapter(server.url, "Qwen/Qwen2.5-7B-Instruct",
                                                       "a" * 40, backend_version="0.29.0")))
    metadata = copy.metadata()
    assert (metadata.model_id, metadata.revision, metadata.backend, metadata.backend_version) == \
        ("Qwen/Qwen2.5-7B-Instruct", "a" * 40, "vllm-server", "0.29.0")
    with pytest.raises(ValueError, match="pinned"):
        VLLMServerAdapter(server.url, "m", "")


# -- the accelerator ---------------------------------------------------------------------
def smi(output=None, error=None):
    def run(*args, **kwargs):
        if error:
            raise error
        return subprocess.CompletedProcess(args, 0, stdout=output, stderr="")
    return run


@pytest.mark.parametrize("output,ok", [
    ("NVIDIA A100-SXM4-80GB, 81920, 550.54\n", True),
    ("NVIDIA A100 80GB PCIe, 81920, 550.54\n", True),
    ("NVIDIA A100-SXM4-40GB, 40960, 550.54\n", False),
    ("NVIDIA L4, 23034, 550.54\n", False),
    ("NVIDIA H100 80GB HBM3, 81559, 550.54\n", False),
    ("NVIDIA A100-SXM4-80GB, 81920, 550.54\nNVIDIA A100-SXM4-80GB, 81920, 550.54\n", False),
])
def test_only_one_a100_80gb_is_admitted(output, ok):
    gpus = gpu_inventory(run=smi(output))
    if ok:
        assert require_a100_80gb(gpus)["memory_mib"] == 81920
    else:
        with pytest.raises(BackendError, match="A100 80 GB|exactly one GPU"):
            require_a100_80gb(gpus)


def test_no_gpu_fails_closed():
    with pytest.raises(BackendError, match="no GPU found"):
        gpu_inventory(run=smi(error=FileNotFoundError("nvidia-smi")))
    with pytest.raises(BackendError, match="exactly one GPU"):
        require_a100_80gb(gpu_inventory(run=smi("")))


def test_the_game_factory_pickles_with_its_frozen_index(splits):  # noqa: F811
    factory = pickle.loads(pickle.dumps(AlfworldEnvFactory("/nowhere", splits)))
    game = splits["pools"]["t_mine"]["games"][0]
    with pytest.raises(BackendError, match="cannot read"):
        factory("train", game["game_id"])
    with pytest.raises(BackendError, match="not a game of the frozen split manifest"):
        factory("train", "unknown/trial")


# -- the calibration pilot -------------------------------------------------------------------
@pytest.mark.parametrize("workers", [1, 2])
def test_the_pilot_measures_what_its_manifest_froze(tmp_path, registry, splits, workers):  # noqa: F811
    pilot = PILOTS[splits["manifest_sha256"]]
    games = sorted(g["game_id"] for g in pilot["games"])
    ledger = RunLedger(tmp_path / "ledger.jsonl", run_id="pilot", clock=Clock())
    executor = EpisodeExecutor(model=WinsOnGames(games[::2]), env_factory=GameFactory(),
                               ledger=ledger, trajectories_dir=tmp_path / "trajectories",
                               run_id="pilot", code_commit="a" * 40, workers=workers,
                               sleep_between_retries=False)
    try:
        measured = run_pilot(executor, pilot, g0_runtime_config(registry))
    finally:
        executor.close()
    measures = measured["measures"]
    assert tuple(measures) == PILOT_MEASURES
    assert measures["g0_success_rate_p0"] == {"numerator": 15, "denominator": 30, "value": 0.5}
    assert measures["temperature_0_repeat_determinism"]["all_identical"] is True
    assert len(measures["temperature_0_repeat_determinism"]["games"]) == 6
    assert measures["arm_b_rendering_smoke_test"]["passed"] is True
    assert measures["throughput_episodes_per_hour"]["workers"] == workers
    assert measures["model_calls_per_episode"]["max"] == 1
    assert ledger.episodes == 30 + 6 + 1
    specs = pilot_specs(pilot, g0_runtime_config(registry))
    assert [s.config.skills[-1].gene_id for s in specs["smoke"]] == [SMOKE_GENE_ID]
    assert all(s.settings.temperature == 0.0 and s.settings.seed is None
               for kind in specs.values() for s in kind)


def test_the_projection_counts_the_protocols_episodes():
    measured = {"measures": {"model_calls_per_episode": {"mean": 20.0},
                             "throughput_episodes_per_hour": {"value": 500.0}}}
    projection = project_exp0001(measured, t_mine_size=60, t_eval_size=134, transfer_size=20,
                                 candidates_per_seed=(2, 10))
    assert projection["fixed_episodes"] == {"lifetime": 540, "final_evaluation": 1358}
    assert GATE_EPISODES_PER_CANDIDATE == 130
    two = [s for s in projection["scenarios"] if s["candidates_per_seed"] == 2]
    assert [s["episodes"] for s in two] == [2678, 2678 + 3 * 20]  # the scripted run's 2678
    worst = projection["scenarios"][-1]
    assert worst["episodes"] == 540 + 1358 + 3 * (10 * 130 + 45 * 20)
    assert worst["within_caps"]["episodes"] is False  # 8498 > 7000
    assert two[0]["model_calls_at_pilot_mean"] == 2678 * 20 + 180
    assert projection["status"].startswith("M ")


# -- the command line --------------------------------------------------------------------------
def test_status_reads_a_run_without_writing(tmp_path, clean, capsys):  # noqa: F811
    run_dir = clean[0]
    before = (run_dir / "ledger.jsonl").read_bytes()
    assert cli.main(["status", "--run-dir", str(run_dir)]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["state"] == "complete" and summary["report.json"]["status"] == "complete"
    assert (run_dir / "ledger.jsonl").read_bytes() == before


def test_repair_ledger_and_gpu_check_from_the_command_line(tmp_path, capsys, monkeypatch):
    ledger = RunLedger(tmp_path / "ledger.jsonl", run_id="r", clock=Clock())
    ledger.admit("e0")
    path = tmp_path / "ledger.jsonl"
    path.write_bytes(path.read_bytes() + b'{"type":"epi')
    assert cli.main(["repair-ledger", "--run-dir", str(tmp_path)]) == 0
    assert json.loads(capsys.readouterr().out)["type"] == "repair"
    monkeypatch.setattr(cli, "gpu_inventory",
                        lambda: [{"name": "Tesla T4", "memory_mib": 15360, "driver": "1"}])
    assert cli.main(["gpu-check"]) == 2
    assert "A100 80 GB" in capsys.readouterr().err


def test_git_state_names_the_commit_and_any_change(tmp_path):
    def git(*args):
        subprocess.run(["git", "-C", str(tmp_path), *args], check=True, capture_output=True)
    git("init", "-q")
    git("config", "user.email", "t@example.com")
    git("config", "user.name", "t")
    (tmp_path / "a.txt").write_text("1")
    git("add", "a.txt")
    git("commit", "-qm", "a")
    commit, dirty = cli.git_state(tmp_path)
    assert len(commit) == 40 and dirty is False
    (tmp_path / "new.py").write_text("x = 1")
    assert cli.git_state(tmp_path) == (commit, True)


def test_a_resumed_run_must_use_the_code_weights_and_software_it_started_with(tmp_path):
    local = tmp_path / "run"
    local.mkdir()
    software = {"vllm": "0.29.0+cu129", "transformers": "5.19.0", "xgrammar": "0.2.8"}
    (local / "run-start.json").write_text(json.dumps(
        {"code": {"commit": "a" * 40}, "weights": {"sha256": "b" * 64},
         "environment": {"software": software, "data": {}}}))
    args = type("Args", (), {"run_id": None})()
    prepared = {"weights": {"sha256": "b" * 64}}
    env = {"software": dict(software), "data": {}}
    with pytest.raises(cli.CliError, match="Check out that commit"):
        cli._run_start(args, local, prepared, None, None, None, "c" * 40, False, "tok", env)
    with pytest.raises(cli.CliError, match="dirty"):
        cli._run_start(args, local, prepared, None, None, None, "a" * 40, True, "tok", env)
    with pytest.raises(cli.CliError, match="weights differ"):
        cli._run_start(args, local, {"weights": {"sha256": "d" * 64}}, None, None, None,
                       "a" * 40, False, "tok", env)
    other = {"software": {**software, "transformers": "5.20.0"}, "data": {}}
    with pytest.raises(cli.CliError, match=r"installed software differs.*'transformers': "
                                           r"\('5.19.0', '5.20.0'\)"):
        cli._run_start(args, local, prepared, None, None, None, "a" * 40, False, "tok", other)
    assert cli._run_start(args, local, prepared, None, None, None, "a" * 40, False,
                          "tok", env)["code"]["commit"] == "a" * 40
    with pytest.raises(cli.CliError, match="needs --run-id"):
        cli._run_start(args, tmp_path / "fresh", prepared, None, None, None, "a" * 40, False,
                       "tok", env)


def test_the_software_stack_names_every_pinned_package():
    from importlib.metadata import PackageNotFoundError

    from evaluation.exp0001.serving import SOFTWARE_PACKAGES, software_stack

    def version(name):
        if name == "llguidance":
            raise PackageNotFoundError(name)
        return f"{name}-v"

    stack = software_stack(version=version)
    assert list(stack) == list(SOFTWARE_PACKAGES)
    assert stack["llguidance"] is None and stack["vllm"] == "vllm-v"
    assert software_stack()["vllm"] is None or isinstance(software_stack()["vllm"], str)


def test_preflight_checks_the_react_reply_and_the_teachers_schema(tmp_path, capsys):
    from evaluation.exp0001.serving import PREPARED_SCHEMA
    prepared = tmp_path / "prepared.json"
    prepared.write_text(json.dumps({"schema": PREPARED_SCHEMA, "model_id": "Qwen/Qwen2.5-7B-Instruct",
                                    "revision": "a" * 40, "path": "/nowhere",
                                    "weights": {"sha256": "b" * 64}}))
    served = (200, {"data": [{"id": "Qwen/Qwen2.5-7B-Instruct"}]})
    server = Server([(200, {"version": "0.29.0"}), served, completion("Thought: ready"),
                     completion('{"proposals": []}'),
                     (200, {"version": "0.29.0"}), served,
                     (200, {"choices": [{"message": {"content": "Thought: ready"}}]})])
    try:
        argv = ["preflight", "--prepared", str(prepared), "--server-url", server.url]
        assert cli.main(argv) == 0
        assert json.loads(capsys.readouterr().out)["teacher_reply"] == {"proposals": []}
        assert cli.main(argv) == 2  # no usage block: the run needs exact token counts
        assert "token usage" in capsys.readouterr().err
    finally:
        server.close()
    teacher = server.requests[3][2]
    assert teacher["response_format"]["json_schema"]["schema"]["required"] == ["proposals"]
    assert teacher["temperature"] == 0.7 and server.requests[2][2]["stop"] == ["\nObservation:"]


# -- review of #69: the data release, the server's grammar, a finished run restored -------
def test_the_server_never_gets_unique_items_and_the_full_schema_still_validates():
    from traits.miner.miner import output_schema
    duplicate = {"proposals": [{"name": "n", "principle": "p", "when_to_apply": "w",
                                "applicability": {"task_families": ["a", "a"]},
                                "estimated_generality": 0.5}]}
    server = Server([completion('{"proposals": []}'), completion(json.dumps(duplicate))])
    try:
        model = adapter(server)
        schema = output_schema()
        assert json.loads(model.structured_generate(MESSAGES, SETTINGS, schema).text) == \
            {"proposals": []}
        with pytest.raises(ModelResponseError):  # the miner's malformed-output path
            model.structured_generate(MESSAGES, SETTINGS, schema)
    finally:
        server.close()
    sent = json.dumps(server.requests[0][2]["response_format"])
    assert "uniqueItems" not in sent and "uniqueItems" in json.dumps(output_schema())


def test_a_finished_run_restored_from_its_mirror_loads(tmp_path):
    local, drive = tmp_path / "local", tmp_path / "drive"
    ledger_path = local / "ledger.jsonl"
    mirror = RunMirror(local, drive)
    ledger = RunLedger(ledger_path, run_id="r", clock=Clock(), on_append=mirror.on_append)
    ledger.admit("e0")
    ledger.record("e0", model_calls=1, outcome={})
    ledger.complete()  # a terminal record always snapshots
    restore(drive, tmp_path / "vm2")
    reopened = RunLedger(tmp_path / "vm2" / "ledger.jsonl", run_id="r", clock=Clock())
    assert reopened.state == "complete" and reopened.notes[0]["type"] == "restore"


def test_one_file_of_the_newest_snapshot_and_the_mirror_check(tmp_path):
    from evaluation.exp0001.mirror import check_mirror_dir, snapshot_file
    local, drive = tmp_path / "local", tmp_path / "drive"
    local.mkdir()
    (local / "run-start.json").write_text(json.dumps({"code": {"commit": "c" * 40}}))
    RunMirror(local, drive).snapshot()
    assert json.loads(snapshot_file(drive, "run-start.json"))["code"]["commit"] == "c" * 40
    assert snapshot_file(drive, "missing.json") is None
    assert snapshot_file(tmp_path / "empty", "run-start.json") is None
    assert check_mirror_dir(drive)["seconds_per_small_file"] >= 0
    assert not (drive / ".mirror-check").exists()
    (tmp_path / "a-file").write_text("x")
    with pytest.raises(MirrorError, match="cannot be used"):
        RunMirror(local, tmp_path / "a-file")


def data_tree(tmp_path):
    from evaluation.benchmarks.alfworld.splits import (
        ALFWORLD_TASK_TYPES,
        build_split_manifest,
        collect_games,
    )
    from evaluation.exp0001.pilot import build_pilot_manifest
    from tests.evaluation.test_alfworld_splits import (
        UNSEEN_PER_FAMILY,
        full_plan,
        make_split,
        write_game,
    )
    root = tmp_path / "json_2.1.1"
    train = collect_games(make_split(root, "train", {f: 40 for f in ALFWORLD_TASK_TYPES}), "train")
    for family, n in UNSEEN_PER_FAMILY.items():
        for i in range(n):
            write_game(root / "valid_unseen", family, 500 + i)
    unseen = collect_games(root / "valid_unseen", "valid_unseen")
    manifest = build_split_manifest(train, unseen, full_plan())
    return root, manifest, build_pilot_manifest(train, manifest)


def test_verify_data_hashes_every_manifest_and_pilot_game(tmp_path):
    from evaluation.exp0001.data import verify_data
    root, manifest, pilot = data_tree(tmp_path)
    report = verify_data(root, manifest, pilot)
    assert report["games"] == 174 + 134 + 30 and report["release"] == "0.2.2"
    assert set(report["archives"]) == {"json_2.1.1_json.zip", "json_2.1.1_tw-pddl.zip"}
    game = pilot["games"][0]["game_id"]
    (root / "train" / game / "game.tw-pddl").write_text('{"changed": true}')
    with pytest.raises(BackendError, match="1 differ"):
        verify_data(root, manifest, pilot)
    (root / "train" / game / "game.tw-pddl").unlink()
    with pytest.raises(BackendError, match="1 manifest games are missing"):
        verify_data(root, manifest, pilot)


def test_fetch_data_checks_every_archive_and_its_paths(tmp_path, monkeypatch):
    import hashlib
    import zipfile

    import evaluation.exp0001.data as data
    good = tmp_path / "src" / "good.zip"
    good.parent.mkdir()
    with zipfile.ZipFile(good, "w") as bundle:
        bundle.writestr("json_2.1.1/train/g/traj_data.json", "{}")
    bad = tmp_path / "src" / "bad.zip"
    with zipfile.ZipFile(bad, "w") as bundle:
        bundle.writestr("elsewhere/x.json", "{}")
    sha = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (good, bad)}
    downloads = []

    def download(source):
        def fetch(url, path):
            downloads.append(url)
            path.write_bytes(source.read_bytes())
        return fetch

    monkeypatch.setattr(data, "ARCHIVES", {"a.zip": sha["good.zip"]})
    root = data.fetch_data(tmp_path / "data", download=download(good))
    assert (root / "train" / "g" / "traj_data.json").exists()
    assert downloads == ["https://github.com/alfworld/alfworld/releases/download/0.2.2/a.zip"]
    data.fetch_data(tmp_path / "data", download=download(good))
    assert len(downloads) == 1  # a verified archive is reused
    with pytest.raises(BackendError, match="not the pinned"):
        data.fetch_data(tmp_path / "other", download=download(bad))
    monkeypatch.setattr(data, "ARCHIVES", {"b.zip": sha["bad.zip"]})
    with pytest.raises(BackendError, match="unexpected path"):
        data.fetch_data(tmp_path / "third", download=download(bad))


def test_the_pinned_archives_are_release_0_2_2():
    from evaluation.exp0001.data import ALFWORLD_RELEASE, ARCHIVES
    assert ALFWORLD_RELEASE == "0.2.2"
    assert ARCHIVES == {
        "json_2.1.1_json.zip": "25171f16e20ad7b048c47275c45b0babf3aa1cbab29cec97387922350a9844bc",
        "json_2.1.1_tw-pddl.zip":
            "d376c4bbb097c068528a2a9805e358e4cd14bc75c4d68a24e5141aebd234ef67"}


def test_run_commit_reads_the_runs_own_commit_from_the_mirror(tmp_path, capsys):
    local, drive = tmp_path / "local", tmp_path / "drive"
    assert cli.main(["run-commit", "--mirror-dir", str(drive)]) == 0
    assert capsys.readouterr().out.strip() == ""  # a new run
    local.mkdir()
    (local / "run-start.json").write_text(json.dumps({"code": {"commit": "d" * 40}}))
    RunMirror(local, drive).snapshot()
    assert cli.main(["run-commit", "--mirror-dir", str(drive)]) == 0
    assert capsys.readouterr().out.strip() == "d" * 40


def _a_run_dir(tmp_path):
    local = tmp_path / "local"
    local.mkdir()
    (local / "ledger.jsonl").write_text('{"type": "x"}\n')
    return local


def test_a_failed_last_snapshot_leaves_a_marker_and_exit_code_3(tmp_path, capsys):
    """ADR-0008 D5: nobody may release the machine holding the only copy."""
    local, mirror_dir = _a_run_dir(tmp_path), tmp_path / "mirror"
    mirror = RunMirror(local, mirror_dir)
    blobs = mirror_dir / "blobs"
    if blobs.exists():
        import shutil
        shutil.rmtree(blobs)
    blobs.write_text("not a directory")  # Drive stops accepting writes
    assert cli._final_snapshot(mirror, local) is False
    marker = json.loads((local / cli.UNMIRRORED).read_text())
    assert marker["failures"] == 1 and marker["last_error"]
    assert "UNMIRRORED" in capsys.readouterr().err
    args = cli.parser().parse_args(["snapshot", "--run-dir", str(local),
                                    "--mirror-dir", str(mirror_dir)])
    assert args.func(args) == cli.EXIT_UNMIRRORED  # still failing: the marker stays
    assert (local / cli.UNMIRRORED).exists()
    blobs.unlink()  # Drive is back
    assert cli.main(["snapshot", "--run-dir", str(local), "--mirror-dir", str(mirror_dir)]) == 0
    assert not (local / cli.UNMIRRORED).exists() and latest_snapshot(mirror_dir) is not None
    # the marker describes one machine: never mirrored, and dropped from a restore
    (local / cli.UNMIRRORED).write_text("{}")
    assert RunMirror(local, mirror_dir).snapshot() is not None
    assert cli.UNMIRRORED not in latest_snapshot(mirror_dir)["files"]
    fresh = tmp_path / "fresh"
    cli._restore_if_missing(fresh, mirror_dir)
    assert (fresh / "ledger.jsonl").exists() and not (fresh / cli.UNMIRRORED).exists()
    assert cli._final_snapshot(None, local) is True  # no mirror, nothing to keep
    with pytest.raises(cli.CliError, match="no run or pilot ledger"):
        cli.cmd_snapshot(type("Args", (), {"run_dir": tmp_path / "empty",
                                           "mirror_dir": mirror_dir})())


def test_serve_reuses_its_own_earlier_server_and_refuses_any_other(tmp_path, monkeypatch, capsys):
    import os

    from evaluation.exp0001 import serving

    class Models(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"data": [{"id": "Qwen/Qwen2.5-7B-Instruct"}]}')

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Models)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    port = server.server_address[1]
    try:
        assert serving.port_answers(port) is True
        prepared = tmp_path / "prepared.json"
        prepared.write_text(json.dumps({"model_id": "Qwen/Qwen2.5-7B-Instruct", "revision": "a" * 40,
                                        "path": str(tmp_path), "weights": {"sha256": "b" * 64}}))
        monkeypatch.setattr(cli, "load_prepared", lambda path: json.loads(Path(path).read_text()))
        started = []
        monkeypatch.setattr(cli, "start_server", lambda *a, **k: started.append(a))
        info = tmp_path / "server.json"
        argv = ["serve", "--prepared", str(prepared), "--port", str(port), "--log",
                str(tmp_path / "log"), "--info", str(info), "--timeout", "5"]
        args = cli.parser().parse_args(argv)
        command = cli.server_command(json.loads(prepared.read_text()), port=port,
                                     max_model_len=args.max_model_len,
                                     gpu_memory_utilization=args.gpu_memory_utilization)
        with pytest.raises(cli.CliError, match="already answers.*pgrep -f 'vllm serve'"):
            args.func(args)  # no record of who started it
        info.write_text(json.dumps({"command": command, "environment": cli.server_environment(),
                                    "port": port, "pid": os.getpid()}))
        assert args.func(args) == 0 and "reusing it" in capsys.readouterr().out
        other = cli.parser().parse_args(argv + ["--no-batch-invariant"])
        with pytest.raises(cli.CliError, match=f"already answers.*pid {os.getpid()}"):
            other.func(other)  # other settings
        info.write_text(json.dumps({"command": command, "environment": cli.server_environment(),
                                    "port": port, "pid": 2 ** 22 + 12345}))
        with pytest.raises(cli.CliError, match="already answers"):
            args.func(args)  # the recorded server is gone
        assert started == []
    finally:
        server.shutdown()
        server.server_close()
    assert serving.port_answers(port) is False
    # an interrupted serve: its info file names a live server that does not answer yet
    waited = []
    monkeypatch.setattr(cli, "wait_until_ready", lambda url, model, process, **k:
                        waited.append(process.poll()))
    info.write_text(json.dumps({"command": command, "environment": cli.server_environment(),
                                "port": port, "pid": os.getpid()}))
    assert args.func(args) == 0 and waited == [None] and started == []
    # a new server: its info is written before the wait
    seen = []
    monkeypatch.setattr(cli, "start_server",
                        lambda *a, **k: type("P", (), {"pid": 2 ** 22 + 7, "poll": lambda s: None})())
    monkeypatch.setattr(cli, "wait_until_ready", lambda *a, **k: seen.append(
        json.loads(info.read_text())["pid"]))
    info.unlink()
    assert args.func(args) == 0 and seen == [2 ** 22 + 7]
    assert serving.process_alive(os.getpid()) and not serving.process_alive(None)


def test_a_stop_or_a_run_error_ends_without_a_traceback(monkeypatch, capsys):
    from evaluation.exp0001.pipeline import RunError

    def stopped(args):
        raise KeyboardInterrupt

    def locked(args):
        raise RunError("/content/exp0001-run is in use by another run")

    for func, code, text in ((stopped, 130, "run the same command again"),
                             (locked, 2, "a previous `run` may still be running")):
        monkeypatch.setattr(cli, "cmd_gpu_check", func)
        assert cli.main(["gpu-check"]) == code
        assert text in capsys.readouterr().err

    def other(args):
        raise RuntimeError("a bug")

    monkeypatch.setattr(cli, "cmd_gpu_check", other)
    with pytest.raises(RuntimeError, match="a bug"):
        cli.main(["gpu-check"])


def test_a_command_stops_cleanly_when_its_parent_dies(tmp_path):
    """The kernel restarts or dies: the command gets one SIGINT, as at the stop
    button, instead of running on as an orphan that holds the run directory."""
    import sys
    import time as _time
    out = tmp_path / "child.out"
    child = (
        "import os, sys, time\n"
        "from evaluation.exp0001 import cli\n"
        "try:\n"
        "    cli._stop_with_parent(int(os.environ['PARENT']), poll_seconds=0.05)\n"
        "    time.sleep(30)\n"
        "except KeyboardInterrupt:\n"
        "    print('stopped with the parent', flush=True)\n")
    parent = (
        "import os, subprocess, sys\n"
        f"subprocess.Popen([sys.executable, '-c', {child!r}], stdout=open({str(out)!r}, 'w'), "
        "start_new_session=True, env={**os.environ, 'PARENT': str(os.getpid())})\n")
    # the parent exits at once, perhaps before the child starts watching
    subprocess.run([sys.executable, "-c", parent], check=True, cwd=Path(cli.__file__).parents[2])
    deadline = _time.monotonic() + 20
    while _time.monotonic() < deadline and "stopped" not in (out.read_text() if out.exists() else ""):
        _time.sleep(0.1)
    assert out.read_text().strip() == "stopped with the parent"
