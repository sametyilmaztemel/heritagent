"""ADR-0008's serving, accelerator, session and worker decisions at the
places the command line wires them: the server's exact command (D4), the
A100 check of `run` and `pilot` before anything is read or written (D1),
the session record (D1, D4) and the worker pool's recycling (D3). No GPU,
no vLLM and no worker process is needed."""

import json
import sys
from concurrent import futures
from types import SimpleNamespace

import pytest

from evaluation.exp0001 import cli
from evaluation.exp0001.episodes import WORKER_TASKS, EpisodeExecutor
from evaluation.exp0001.serving import server_command

T4 = {"name": "Tesla T4", "memory_mib": 15360, "driver": "550.54.15"}
A100 = {"name": "NVIDIA A100-SXM4-80GB", "memory_mib": 81920, "driver": "550.54.15"}


def test_the_server_command_for_a_prepared_model():
    """D4: vLLM's own generation defaults (not the model's
    generation_config.json), engine seed 0, the pre-registered window
    (ADR-0009 D4), the language model only and thinking off (ADR-0011)."""
    prepared = {"path": "/content/models/Qwen--Qwen3.8-27B", "model_id": "Qwen/Qwen3.8-27B",
                "revision": "a" * 40}
    assert server_command(prepared) == [
        "vllm", "serve", "/content/models/Qwen--Qwen3.8-27B",
        "--served-model-name", "Qwen/Qwen3.8-27B",
        "--host", "127.0.0.1", "--port", "8000",
        "--max-model-len", "32768",
        "--gpu-memory-utilization", "0.9",
        "--generation-config", "vllm", "--seed", "0", "--language-model-only",
        "--default-chat-template-kwargs", '{"enable_thinking": false}']


def command_line(command, tmp_path):
    out = "--local-dir" if command == "run" else "--out-dir"
    argv = [command, "--prepared", str(tmp_path / "prepared.json"),
            "--data-root", str(tmp_path / "json_2.1.1"), "--repo", str(tmp_path / "repo"),
            "--server-info", str(tmp_path / "vllm-server.json"), "--workers", "16",
            out, str(tmp_path / "out"), "--mirror-dir", str(tmp_path / "mirror")]
    if command == "run":
        argv += ["--run-id", "EXP-0001-run-1",
                 "--rehearsal-passed", str(tmp_path / "run-passed.json")]
    return argv


@pytest.mark.parametrize("command", ["run", "pilot"])
@pytest.mark.parametrize("gpus", [[T4], [A100, A100],
                                  [{**A100, "name": "NVIDIA A100-SXM4-40GB",
                                    "memory_mib": 40960}]])
def test_run_and_pilot_stop_on_another_gpu_before_anything_else(command, gpus, tmp_path,
                                                                monkeypatch, capsys):
    """D1: the GPU check comes first; the prepared model is not even read,
    and nothing is written."""
    monkeypatch.delenv(cli.STOP_WITH_PARENT, raising=False)
    monkeypatch.setattr(cli, "gpu_inventory", lambda: [dict(gpu) for gpu in gpus])
    later = []

    def load_prepared(path):
        later.append(path)
        raise AssertionError("read past the GPU check")

    monkeypatch.setattr(cli, "load_prepared", load_prepared)
    before = sorted(tmp_path.rglob("*"))
    assert cli.main(command_line(command, tmp_path)) == 2
    assert later == []
    assert sorted(tmp_path.rglob("*")) == before == []
    err = capsys.readouterr().err
    assert "A100 80 GB" in err or "exactly one GPU" in err


@pytest.mark.parametrize("command", ["run", "pilot"])
def test_run_and_pilot_read_on_past_an_a100_80gb(command, tmp_path, monkeypatch):
    """The other side of the check: an A100 80 GB reaches the next step."""
    monkeypatch.delenv(cli.STOP_WITH_PARENT, raising=False)
    monkeypatch.setattr(cli, "gpu_inventory", lambda: [dict(A100)])
    later = []

    def load_prepared(path):
        later.append(path)
        raise cli.CliError("stop here")

    monkeypatch.setattr(cli, "load_prepared", load_prepared)
    assert cli.main(command_line(command, tmp_path)) == 2
    assert later == [tmp_path / "prepared.json"]


def test_the_session_record_names_the_gpu_the_software_and_the_server(tmp_path, monkeypatch):
    """D1: the ledger's session record names the GPU; D4: the installed
    software and the server's version and settings, so a resumed session
    on others is visible."""
    software = {"vllm": "0.29.0+cu129", "torch": "2.13.0+cu129"}
    monkeypatch.setattr(cli, "software_stack", lambda: dict(software))
    metadata = SimpleNamespace(model_id="Qwen/Qwen3.8-27B", revision="a" * 40, backend="vllm",
                               backend_version="0.29.0",
                               capabilities={"server": "http://127.0.0.1:8000"})
    adapter = SimpleNamespace(metadata=lambda: metadata)
    settings = {"command": ["vllm", "serve"], "environment": {"VLLM_BATCH_INVARIANT": "0"}}
    info = tmp_path / "vllm-server.json"
    info.write_text(json.dumps(settings))
    session = cli._session_info(dict(A100), adapter, 16, info)
    assert set(session) == {"accelerator", "server", "vllm", "software", "server_settings",
                            "workers", "opened_at"}
    assert session["accelerator"] == A100
    assert set(session["accelerator"]) == {"name", "memory_mib", "driver"}
    assert session["software"] == software
    assert session["vllm"] == "0.29.0" and session["server"] == "http://127.0.0.1:8000"
    assert session["server_settings"] == settings and session["workers"] == 16
    assert cli._session_info(dict(A100), adapter, 16, tmp_path / "missing.json")[
        "server_settings"] is None


@pytest.mark.skipif(sys.version_info < (3, 11),
                    reason="max_tasks_per_child needs Python 3.11; `run` refuses older ones")
def test_the_worker_pool_replaces_a_worker_after_worker_tasks_games(monkeypatch):
    """D3: a worker is replaced after WORKER_TASKS games (TextWorld's caches
    grow with every game a process opens)."""
    built = []

    class Pool:
        def __init__(self, **kwargs):
            built.append(kwargs)

    monkeypatch.setattr(futures, "ProcessPoolExecutor", Pool)
    executor = EpisodeExecutor(model=None, env_factory=None, ledger=None,
                               trajectories_dir="unused", run_id="r", code_commit="c" * 40,
                               workers=4)
    executor._get_pool()
    (kwargs,) = built
    assert kwargs["max_tasks_per_child"] == WORKER_TASKS == 50
    assert kwargs["max_workers"] == 4
