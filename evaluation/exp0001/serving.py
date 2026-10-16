"""Serving the EXP-0001 model on one GPU node (issue #13 Phase B; ADR-0008).

- ``prepare_model``: download exactly one pinned revision of the model
  (any given revision is resolved to its commit; none means the hub's
  current one) and hash its files with ``sha256_files``. The weights the
  run-start record names are the files the server loads.
- ``server_command`` / ``start_server``: ``vllm serve`` on that local
  snapshot under the model's hub id, with vLLM's neutral sampling defaults
  (``--generation-config vllm``), a fixed engine seed and batch-invariant
  kernels (``VLLM_BATCH_INVARIANT=1``: a reply does not depend on what else
  is in the batch; vLLM lists Qwen2.5-7B-Instruct among the validated
  models and needs compute capability 8.0, which the A100 has).
- ``wait_until_ready``: poll the server until it serves the model, or fail
  with the tail of its log."""

from __future__ import annotations

import json
import os
import subprocess
import time
from importlib import metadata as importlib_metadata
from collections.abc import Callable, Mapping
from pathlib import Path

from evaluation.exp0001.protocol import CONTEXT_WINDOW
from evaluation.exp0001.provenance import ProvenanceError, sha256_files
from runtime.model_adapters.openai_server import VLLMServerAdapter

PREPARED_SCHEMA = "heritagent/exp0001-prepared-model/0.1"
DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"  # manifest model.name
DEFAULT_PORT = 8000
DEFAULT_MAX_MODEL_LEN = CONTEXT_WINDOW  # pre-registered (ADR-0009 D4)


# the packages that serve, render, tokenize and constrain a request, and play
# an episode: the notebooks pin each to one version (ADR-0008 D4)
SOFTWARE_PACKAGES = ("vllm", "torch", "transformers", "tokenizers", "huggingface-hub", "xgrammar",
                     "llguidance", "jinja2", "textworld", "alfworld")


def software_stack(packages=SOFTWARE_PACKAGES,
                   version: Callable[[str], str] = importlib_metadata.version) -> dict:
    """The installed version of each package (None when it is not installed).
    The server runs from the same environment as this process."""
    stack = {}
    for name in packages:
        try:
            stack[name] = version(name)
        except importlib_metadata.PackageNotFoundError:
            stack[name] = None
    return stack


def _is_commit(revision) -> bool:
    return isinstance(revision, str) and len(revision) == 40 and \
        all(c in "0123456789abcdef" for c in revision)


def prepare_model(model_id: str = DEFAULT_MODEL, *, revision: str | None = None,
                  cache_dir: Path | None = None) -> dict:
    from huggingface_hub import HfApi, snapshot_download
    commit = HfApi().model_info(model_id, revision=revision).sha  # a branch or tag -> its commit
    if not _is_commit(commit):
        raise ProvenanceError(f"{model_id}@{revision} did not resolve to a commit ({commit!r})")
    path = Path(snapshot_download(model_id, revision=commit,
                                  cache_dir=None if cache_dir is None else str(cache_dir)))
    return {"schema": PREPARED_SCHEMA, "model_id": model_id, "revision": commit,
            "path": str(path), "weights": sha256_files(path)}


def load_prepared(path: Path) -> dict:
    prepared = json.loads(Path(path).read_text(encoding="utf-8"))
    if prepared.get("schema") != PREPARED_SCHEMA or not _is_commit(prepared.get("revision")):
        raise ProvenanceError(f"{path} is not a prepared-model record with a pinned commit")
    return prepared


def server_command(prepared: Mapping, *, port: int = DEFAULT_PORT,
                   max_model_len: int = DEFAULT_MAX_MODEL_LEN,
                   gpu_memory_utilization: float = 0.90) -> list[str]:
    return ["vllm", "serve", prepared["path"], "--served-model-name", prepared["model_id"],
            "--host", "127.0.0.1", "--port", str(port),
            "--max-model-len", str(max_model_len),
            "--gpu-memory-utilization", str(gpu_memory_utilization),
            "--generation-config", "vllm", "--seed", "0"]


def server_environment(*, batch_invariant: bool = True) -> dict:
    return {"VLLM_BATCH_INVARIANT": "1" if batch_invariant else "0"}


def start_server(command: list[str], log_path: Path, *, environment: Mapping | None = None,
                 popen: Callable = subprocess.Popen) -> subprocess.Popen:
    """Start the server detached from this process (it outlives a notebook
    cell), with its output in ``log_path``."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open("ab")
    return popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                 env={**os.environ, **dict(environment or server_environment())})


def port_answers(port: int, *, host: str = "127.0.0.1", timeout_seconds: float = 2.0) -> bool:
    """Whether an HTTP server answers ``GET /v1/models`` on the port."""
    import urllib.error
    import urllib.request
    try:
        with urllib.request.urlopen(f"http://{host}:{port}/v1/models", timeout=timeout_seconds):
            return True
    except urllib.error.HTTPError:
        return True  # something answers, even if not with 200
    except OSError:
        return False


def process_alive(pid) -> bool:
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _tail(path: Path | None, lines: int = 40) -> str:
    if path is None or not Path(path).exists():
        return ""
    return "\n".join(Path(path).read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])


def wait_until_ready(base_url: str, model_id: str, *, process=None,
                     log_path: Path | None = None, timeout_seconds: float = 1800.0,
                     poll_seconds: float = 5.0, clock: Callable[[], float] = time.monotonic,
                     sleep: Callable[[float], None] = time.sleep) -> None:
    deadline = clock() + timeout_seconds
    probe = VLLMServerAdapter(base_url, model_id, "readiness-probe", transport_retries=0,
                              timeout_seconds=30.0)
    while True:
        if process is not None and process.poll() is not None:
            raise RuntimeError(f"the model server exited with code {process.returncode}:\n"
                               f"{_tail(log_path)}")
        try:
            probe.require_served()
            return
        except Exception as exc:  # noqa: BLE001 - not up yet
            if clock() > deadline:
                raise RuntimeError(f"the model server was not ready after {timeout_seconds:.0f} s "
                                   f"({exc}):\n{_tail(log_path)}") from exc
        sleep(poll_seconds)
