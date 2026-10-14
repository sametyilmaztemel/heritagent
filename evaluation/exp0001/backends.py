"""The real backends of the EXP-0001 run (issue #13 Phase A wiring).

- ALFWorld games: a game is opened only if its files still hash to the
  content hash the frozen split manifest records, so drifted data can never
  enter a run. The game session is closed after every episode.
- The model: ``VLLMAdapter`` with a pinned revision.
- The token counter of the overhead check (tau_k) and of the genome-size
  outputs: the model's own tokenizer, never the chars/4 budgeting estimate.

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


def alfworld_env_factory(data_root: Path, split_manifest: Mapping, *,
                         opener: Callable = open_game,
                         max_episode_steps: int = EPISODE_MAX_STEPS):
    """``env_factory(split, game_id)`` over ``<data_root>/<split>/<game_id>``
    (``data_root`` is ALFWorld's ``json_2.1.1`` directory)."""
    root = Path(data_root)
    index = manifest_content_index(split_manifest)

    def factory(split: str, game_id: str) -> ClosingEnvironment:
        expected = index.get((split, game_id))
        if expected is None:
            raise BackendError(f"{split}/{game_id} is not a game of the frozen split manifest")
        trial = root / split / game_id
        try:
            actual = _content_sha256((trial / TRAJ_FILE).read_bytes(),
                                     (trial / GAME_FILE).read_bytes())
        except OSError as exc:
            raise BackendError(f"cannot read {trial}: {exc}") from exc
        if actual != expected:
            raise BackendError(f"{split}/{game_id} no longer matches its frozen content hash")
        return ClosingEnvironment(opener(str(trial / GAME_FILE),
                                         max_episode_steps=max_episode_steps), game_id)

    return factory


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


def hf_token_counter(model_id: str, revision: str) -> TokenizerCounter:
    from transformers import AutoTokenizer  # installed with vLLM
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    return TokenizerCounter(tokenizer, f"hf-tokenizer:{model_id}@{revision}")


def pinned_vllm_model(model_id: str, revision: str, **kwargs) -> VLLMAdapter:
    if not isinstance(revision, str) or not revision.strip():
        raise BackendError("the model revision must be pinned (manifest: pin-at-run-time)")
    return VLLMAdapter(model=model_id, revision=revision, **kwargs)
