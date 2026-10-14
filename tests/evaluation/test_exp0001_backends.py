"""The real EXP-0001 backends (issue #13 Phase A wiring) through fakes: no
ALFWorld install, no vLLM, no tokenizer download."""

import pytest

from evaluation.benchmarks.alfworld.splits import (
    ALFWORLD_TASK_TYPES,
    build_split_manifest,
    collect_games,
)
from evaluation.exp0001.backends import (
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
        pinned_vllm_model("Qwen/Qwen2.5-7B-Instruct", "")
    adapter = pinned_vllm_model("Qwen/Qwen2.5-7B-Instruct", "abc123")
    assert adapter.metadata().revision == "abc123"  # no engine is created
