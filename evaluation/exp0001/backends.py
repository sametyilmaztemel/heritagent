"""The real backends of the EXP-0001 run (issue #13 Phase A wiring).

- ALFWorld games: a game is opened only if its files still hash to the
  content hash the frozen split manifest records, so drifted data can never
  enter a run. The game session is closed after every episode.
- The model: ``VLLMAdapter`` with a pinned revision, or, for parallel
  episodes, ``VLLMServerAdapter`` on one ``vllm serve`` process (ADR-0008).
- The token counter of the overhead check (tau_k) and of the genome-size
  outputs: the model's own tokenizer, never the chars/4 budgeting estimate.
- The accelerator: an A100 80 GB, the only GPU the owner admitted for the
  run (ADR-0008); anything else fails closed before any episode.

The offline tests exercise these through fakes; Phase B runs them."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path

from evaluation.benchmarks.alfworld.splits import _content_sha256
from runtime.model_adapters.vllm_backend import VLLMAdapter
from runtime.seed_g0 import EPISODE_MAX_STEPS
from runtime.tool_adapters.alfworld import AlfworldEnvironment, open_game

GAME_FILE = "game.tw-pddl"
TRAJ_FILE = "traj_data.json"


class BackendError(RuntimeError):
    """A backend does not match what the run froze."""


def manifest_content_index(split_manifest: Mapping) -> dict[tuple[str, str], str]:
    """(split, game_id) -> content hash, over every pool of the manifest."""
    from evaluation.exp0001.allocations import POOL_SPLITS
    index: dict[tuple[str, str], str] = {}
    for pool, entry in split_manifest["pools"].items():
        for game in entry["games"]:
            index[(POOL_SPLITS[pool], game["game_id"])] = game["content_sha256"]
    return index


class ClosingEnvironment:
    """An ``AlfworldEnvironment`` that closes its game session afterwards."""

    def __init__(self, session, game_id: str):
        self._session = session
        self._env = AlfworldEnvironment(session, game_id=game_id)

    def start(self):
        return self._env.start()

    def list_tools(self):
        return self._env.list_tools()

    def call(self, name, arguments):
        return self._env.call(name, arguments)

    def record(self):
        return self._env.record()

    def close(self) -> None:
        close = getattr(self._session, "close", None)
        if callable(close):
            close()


class AlfworldEnvFactory:
    """``env_factory(split, game_id)`` over ``<data_root>/<split>/<game_id>``
    (``data_root`` is ALFWorld's ``json_2.1.1`` directory). A plain object,
    so it pickles into worker processes; ``extra_games`` admits games of
    another frozen manifest (the calibration pilot's) by their content hash."""

    def __init__(self, data_root: Path, split_manifest: Mapping, *,
                 opener: Callable = open_game, max_episode_steps: int = EPISODE_MAX_STEPS,
                 extra_games: Mapping[tuple[str, str], str] | None = None):
        self._root = Path(data_root)
        self._index = {**manifest_content_index(split_manifest), **dict(extra_games or {})}
        self._opener = opener
        self._max_episode_steps = max_episode_steps

    def __call__(self, split: str, game_id: str) -> ClosingEnvironment:
        expected = self._index.get((split, game_id))
        if expected is None:
            raise BackendError(f"{split}/{game_id} is not a game of the frozen split manifest")
        trial = self._root / split / game_id
        try:
            actual = _content_sha256((trial / TRAJ_FILE).read_bytes(),
                                     (trial / GAME_FILE).read_bytes())
        except OSError as exc:
            raise BackendError(f"cannot read {trial}: {exc}") from exc
        if actual != expected:
            raise BackendError(f"{split}/{game_id} no longer matches its frozen content hash")
        return ClosingEnvironment(self._opener(str(trial / GAME_FILE),
                                               max_episode_steps=self._max_episode_steps),
                                  game_id)


def alfworld_env_factory(data_root: Path, split_manifest: Mapping, *,
                         opener: Callable = open_game,
                         max_episode_steps: int = EPISODE_MAX_STEPS) -> AlfworldEnvFactory:
    return AlfworldEnvFactory(data_root, split_manifest, opener=opener,
                              max_episode_steps=max_episode_steps)


class TokenizerCounter:
    """``TokenCounter`` over a tokenizer's ``encode`` (special tokens off)."""

    def __init__(self, tokenizer, identity: str):
        if not isinstance(identity, str) or not identity.strip():
            raise BackendError("a token counter needs a stable identity")
        self._tokenizer = tokenizer
        self._identity = identity

    @property
    def identity(self) -> str:
        return self._identity

    def count_tokens(self, text: str) -> int:
        return len(self._tokenizer.encode(text, add_special_tokens=False))


def hf_token_counter(model_id: str, revision: str, path: Path | None = None) -> TokenizerCounter:
    """The model's tokenizer; ``path`` is the local snapshot of exactly that
    revision (the one the server loads), else the hub is asked for it."""
    from transformers import AutoTokenizer  # installed with vLLM
    tokenizer = (AutoTokenizer.from_pretrained(str(path)) if path is not None
                 else AutoTokenizer.from_pretrained(model_id, revision=revision))
    return TokenizerCounter(tokenizer, f"hf-tokenizer:{model_id}@{revision}")


def pinned_vllm_model(model_id: str, revision: str, **kwargs) -> VLLMAdapter:
    from evaluation.exp0001.protocol import CONTEXT_WINDOW
    if not isinstance(revision, str) or not revision.strip():
        raise BackendError("the model revision must be pinned (manifest: pin-at-run-time)")
    return VLLMAdapter(model=model_id, revision=revision,
                       **{"max_model_len": CONTEXT_WINDOW, **kwargs})


# -- the accelerator (ADR-0008: an A100 80 GB only) ------------------------------
A100_80GB_MIN_MIB = 76_000  # an 80 GB A100 reports 81920 MiB; a 40 GB one 40960


def gpu_inventory(run: Callable | None = None) -> list[dict]:
    """The GPUs ``nvidia-smi`` reports, as name / memory MiB / driver."""
    import subprocess
    run = run or subprocess.run
    try:
        out = run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                   "--format=csv,noheader,nounits"], capture_output=True, text=True,
                  check=True, timeout=60).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        raise BackendError(f"no GPU found (nvidia-smi failed: {exc})") from exc
    gpus = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 3:
            raise BackendError(f"unexpected nvidia-smi line {line!r}")
        try:
            memory = int(float(parts[1]))
        except ValueError as exc:
            raise BackendError(f"unexpected nvidia-smi memory {parts[1]!r}") from exc
        gpus.append({"name": parts[0], "memory_mib": memory, "driver": parts[2]})
    return gpus


def require_a100_80gb(gpus: list[dict]) -> dict:
    """Exactly one GPU, an A100 with 80 GB; anything else fails closed
    (ADR-0008, owner decision). Returns the GPU for the session record."""
    if len(gpus) != 1:
        raise BackendError(f"the run uses exactly one GPU, found {len(gpus)}: {gpus}")
    gpu = gpus[0]
    if "A100" not in gpu["name"] or gpu["memory_mib"] < A100_80GB_MIN_MIB:
        raise BackendError(f"the GPU is {gpu['name']} with {gpu['memory_mib']} MiB; the run "
                           f"admits only an A100 80 GB (ADR-0008). In Colab: Runtime > Change "
                           f"runtime type > A100, and check the memory with nvidia-smi")
    return dict(gpu)
