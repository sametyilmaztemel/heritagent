"""The real EXP-0001 backends (issue #13 Phase A wiring) through fakes: no
ALFWorld install, no vLLM, no tokenizer download."""

import pytest

from evaluation.benchmarks.alfworld.splits import (
    ALFWORLD_TASK_TYPES,
    build_split_manifest,
    collect_games,
)
from evaluation.exp0001.backends import (
    AlfworldEnvFactory,
    BackendError,
    TokenizerCounter,
    alfworld_env_factory,
    pinned_vllm_model,
)
from inheritance.regression import validate_token_counter
from tests.evaluation.test_alfworld_splits import (
    UNSEEN_PER_FAMILY,
    full_plan,
    make_split,
    write_game,
)
from tests.evaluation.test_exp0001_lifetime import Game

F = ALFWORLD_TASK_TYPES


class Session(Game):
    closed = 0

    def close(self):
        Session.closed += 1


@pytest.fixture
def tree(tmp_path):
    root = tmp_path / "json_2.1.1"
    train = collect_games(make_split(root, "train", {f: 40 for f in F}), "train")
    for family, n in UNSEEN_PER_FAMILY.items():
        for i in range(n):
            write_game(root / "valid_unseen", family, 500 + i)
    unseen = collect_games(root / "valid_unseen", "valid_unseen")
    return root, build_split_manifest(train, unseen, full_plan())


def test_a_frozen_game_opens_and_its_session_is_closed(tree):
    root, manifest = tree
    opened = []

    def opener(gamefile, *, max_episode_steps):
        opened.append((gamefile, max_episode_steps))
        return Session(gamefile)

    factory = alfworld_env_factory(root, manifest, opener=opener)
    game = manifest["pools"]["t_eval"]["games"][0]
    env = factory("valid_unseen", game["game_id"])
    assert opened == [(str(root / "valid_unseen" / game["game_id"] / "game.tw-pddl"), 50)]
    assert "Your task is to" in env.start()
    env.close()
    assert Session.closed == 1


def test_the_factory_the_run_builds_gives_the_game_the_protocols_step_limit(tree):
    """The pilot, the rehearsals and the run construct the class directly
    (cli.py), not through ``alfworld_env_factory``: its own defaults are what
    the game receives. The game's own limit is the protocol's 50 steps
    (ADR-0003 D4)."""
    import inspect

    from evaluation.exp0001.episodes import protocol_budgets
    from runtime.tool_adapters.alfworld import open_game

    root, manifest = tree
    opened = []

    def opener(gamefile, **kwargs):
        opened.append(kwargs)
        return Session(gamefile)

    factory = AlfworldEnvFactory(root, manifest, opener=opener, extra_games={})
    game = manifest["pools"]["t_eval"]["games"][0]
    factory("valid_unseen", game["game_id"]).close()
    assert opened == [{"max_episode_steps": protocol_budgets().max_steps}]
    assert protocol_budgets().max_steps == 50
    # and the opener cli.py leaves to its default is the real game's
    assert inspect.signature(AlfworldEnvFactory).parameters["opener"].default is open_game


def test_drifted_or_unknown_games_fail_closed(tree):
    root, manifest = tree
    factory = alfworld_env_factory(root, manifest, opener=lambda g, **_: Session(g))
    game = manifest["pools"]["t_mine"]["games"][0]
    with pytest.raises(BackendError, match="not a game of the frozen split manifest"):
        factory("valid_unseen", game["game_id"])  # a train game under the wrong split
    (root / "train" / game["game_id"] / "game.tw-pddl").write_text('{"solvable": true, "x": 1}')
    with pytest.raises(BackendError, match="no longer matches"):
        factory("train", game["game_id"])
    (root / "train" / game["game_id"] / "game.tw-pddl").unlink()
    with pytest.raises(BackendError, match="cannot read"):
        factory("train", game["game_id"])


def test_the_token_counter_uses_the_tokenizer():
    class Tok:
        def encode(self, text, add_special_tokens=True):
            assert add_special_tokens is False
            return text.split()

    counter = TokenizerCounter(Tok(), "hf-tokenizer:qwen@abc")
    validate_token_counter(counter)
    assert counter.count_tokens("a b c") == 3 and counter.identity == "hf-tokenizer:qwen@abc"
    with pytest.raises(BackendError, match="identity"):
        TokenizerCounter(Tok(), " ")


def test_the_model_revision_must_be_pinned():
    with pytest.raises(BackendError, match="pinned"):
        pinned_vllm_model("Qwen/Qwen3.8-27B", "")
    adapter = pinned_vllm_model("Qwen/Qwen3.8-27B", "abc123")
    assert adapter.metadata().revision == "abc123"  # no engine is created


def test_the_pinned_offline_model_uses_the_protocols_window_and_thinking_off(monkeypatch):
    """ADR-0009 D4 and ADR-0011 D3 reach the engine and every chat call."""
    import sys
    import types

    from evaluation.exp0001.protocol import CONTEXT_WINDOW
    from runtime.model_adapters import GenerationSettings, ModelMessage

    seen = {}

    class LLM:
        def __init__(self, **kwargs):
            seen["engine"] = kwargs

        def chat(self, messages, params, **kwargs):
            seen["chat"] = kwargs
            output = types.SimpleNamespace(text="Final Answer: x", finish_reason="stop",
                                           token_ids=[1, 2])
            return [types.SimpleNamespace(outputs=[output], prompt_token_ids=[1])]

    module = types.ModuleType("vllm")
    module.LLM = LLM
    module.SamplingParams = lambda **kwargs: types.SimpleNamespace(**kwargs)
    monkeypatch.setitem(sys.modules, "vllm", module)
    adapter = pinned_vllm_model("Qwen/Qwen3.8-27B", "abc123")
    adapter.generate([ModelMessage("user", "u")], GenerationSettings())
    assert (seen["engine"]["model"], seen["engine"]["revision"],
            seen["engine"]["max_model_len"]) == ("Qwen/Qwen3.8-27B", "abc123", CONTEXT_WINDOW)
    assert seen["chat"] == {"chat_template_kwargs": {"enable_thinking": False}}
