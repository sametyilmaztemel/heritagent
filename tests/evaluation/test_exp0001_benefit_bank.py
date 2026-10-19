"""The EXP-0001 benefit bank (ADR-0012 D1): the committed manifest, built
from the real ALFWorld 0.2.2 train split, verifies from its own hashes and
against the committed split and pilot manifests, so no ALFWorld data is
needed here. The builder is also tested on a synthetic tree."""

import copy
import json
from pathlib import Path

import pytest

from evaluation.benchmarks.alfworld.splits import (
    ALFWORLD_TASK_TYPES,
    POOL_ORDER,
    SplitError,
    _sha256,
    build_split_manifest,
    collect_games,
)
from evaluation.exp0001.benefit_bank import (
    BANK_PER_FAMILY,
    bank_rank,
    build_benefit_bank,
    verify_benefit_bank,
)
from evaluation.exp0001.pilot import build_pilot_manifest
from evaluation.exp0001.protocol import LOCKED_PILOT_SHA256, LOCKED_SPLIT_MANIFEST_SHA256
from tests.evaluation.test_alfworld_splits import (
    UNSEEN_PER_FAMILY,
    full_plan,
    make_split,
    write_game,
)

ROOT = Path(__file__).resolve().parents[2]
MANIFESTS = ROOT / "experiments" / "manifests"
# the committed bank, rebuilt byte for byte from ALFWorld 0.2.2 by an independent review
COMMITTED_BANK_SHA256 = "fd44bb361e7e0176f281f14b75e7381f4bbf68a0b7b73eee33d50b44323ca545"


def load(name):
    return json.loads((MANIFESTS / name).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def committed():
    return load("EXP-0001-benefit-bank.json"), load("EXP-0001-splits.json"), \
        load("EXP-0001-pilot.json")


def make_unseen(root: Path) -> Path:
    """A synthetic valid_unseen split whose game ids no train game shares, as
    in ALFWorld 0.2.2 (the bank fails closed otherwise)."""
    split_dir = root / "valid_unseen"
    for family, n in UNSEEN_PER_FAMILY.items():
        for i in range(n):
            write_game(split_dir, family, 5000 + i)
    return split_dir


def _rehash(bank):
    bank["games_sha256"] = _sha256(bank["families"])
    bank["bank_sha256"] = _sha256({k: v for k, v in bank.items() if k != "bank_sha256"})
    return bank


def _swap_in(bank, game):
    """Put ``game`` in place of a bank game of its family, keeping hash order."""
    family = game["family"]
    games = bank["families"][family][1:] + [dict(game)]
    bank["families"][family] = sorted(games, key=lambda g: bank_rank(family, g["game_id"]))


def test_the_committed_bank_verifies_against_the_locked_manifests(committed):
    bank, splits, pilot = committed
    verify_benefit_bank(bank, splits, pilot)
    assert bank["bank_sha256"] == COMMITTED_BANK_SHA256
    assert bank["split_manifest_sha256"] == LOCKED_SPLIT_MANIFEST_SHA256
    assert bank["pilot_sha256"] == LOCKED_PILOT_SHA256
    assert {f: len(games) for f, games in bank["families"].items()} == \
        {f: BANK_PER_FAMILY for f in ALFWORLD_TASK_TYPES}
    assert bank["status"] == "frozen before any EXP-0001 episode"


def test_the_committed_bank_records_the_availability_adr_0012_states(committed):
    """ADR-0012 D1: games left per family after the pools and the pilot,
    re-read from the manifests by two evaluations."""
    bank, splits, pilot = committed
    assert bank["available_by_family"] == {
        "look_at_obj_in_light": 277, "pick_and_place_simple": 754,
        "pick_clean_then_place_in_recep": 615, "pick_cool_then_place_in_recep": 499,
        "pick_heat_then_place_in_recep": 427, "pick_two_obj_and_place": 777}
    accepted = splits["collections"]["train"]["accepted_by_family"]
    used = {f: 0 for f in ALFWORLD_TASK_TYPES}
    for pool in POOL_ORDER:
        for game in splits["pools"][pool]["games"]:
            used[game["family"]] += 1
    for game in pilot["games"]:
        used[game["family"]] += 1
    assert bank["available_by_family"] == {f: accepted[f] - used[f] for f in ALFWORLD_TASK_TYPES}


@pytest.mark.parametrize("tamper, message", [
    (lambda b, s, p: b["families"]["look_at_obj_in_light"].reverse(), "bank_sha256"),
    (lambda b, s, p: b.__setitem__("split_manifest_sha256", "0" * 64), "bank_sha256"),
])
def test_an_edited_bank_fails_its_own_hash(committed, tamper, message):
    bank, splits, pilot = (copy.deepcopy(x) for x in committed)
    tamper(bank, splits, pilot)
    with pytest.raises(SplitError, match=message):
        verify_benefit_bank(bank, splits, pilot)


def _relabel(bank):
    """A train game of another family, labelled and filed as look."""
    game = dict(bank["families"]["pick_two_obj_and_place"][0], family="look_at_obj_in_light")
    _swap_in(bank, game)


def _cross_family_duplicate(bank):
    """The same game id in two families, each list otherwise well formed."""
    game = dict(bank["families"]["look_at_obj_in_light"][0], family="pick_two_obj_and_place")
    _swap_in(bank, game)


@pytest.mark.parametrize("tamper, message", [
    # T_eval's valid_unseen ids are in a pool of the split manifest too
    (lambda b, s, p: _swap_in(b, s["pools"]["t_eval"]["games"][0]), "T_eval included"),
    (lambda b, s, p: _swap_in(b, s["pools"]["ablation"]["games"][0]), "T_eval included"),
    (lambda b, s, p: _swap_in(b, s["pools"]["interaction"]["games"][0]), "T_eval included"),
    (lambda b, s, p: _swap_in(b, s["pools"]["replay_bank"]["games"][0]), "T_eval included"),
    (lambda b, s, p: _relabel(b), "another family"),
    (lambda b, s, p: b["families"]["look_at_obj_in_light"][0].__setitem__(
        "content_sha256", "zz"), "malformed game record"),
    (lambda b, s, p: b["families"]["look_at_obj_in_light"][0].__setitem__("extra", 1),
     "malformed game record"),
    (lambda b, s, p: b["available_by_family"].__setitem__("look_at_obj_in_light", 9999),
     "available_by_family"),
    (lambda b, s, p: b.__setitem__("experiment", "EXP-0002"), "experiment"),
    (lambda b, s, p: b["selection"].__setitem__("per_family", 192.0), "selection"),
    (lambda b, s, p: b["selection"].__setitem__("seed", False), "selection"),
    (lambda b, s, p: b["selection"].__setitem__("rule", "any"), "selection"),
    (lambda b, s, p: b.__setitem__("extra", True), "fields"),
])
def test_a_rehashed_wrong_committed_bank_fails_closed(committed, tamper, message):
    bank, splits, pilot = (copy.deepcopy(x) for x in committed)
    tamper(bank, splits, pilot)
    with pytest.raises(SplitError, match=message):
        verify_benefit_bank(_rehash(bank), splits, pilot)


def test_a_bank_against_tampered_split_or_pilot_manifests_fails_closed(committed):
    bank, splits, pilot = (copy.deepcopy(x) for x in committed)
    splits["pools"]["t_mine"]["games"].pop()  # recorded hashes left as they were
    with pytest.raises(SplitError, match="do not match"):
        verify_benefit_bank(bank, splits, pilot)
    bank, splits, pilot = (copy.deepcopy(x) for x in committed)
    pilot["games"].pop()
    with pytest.raises(SplitError, match="does not match the pilot manifest content"):
        verify_benefit_bank(bank, splits, pilot)


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    # pools take at most 10 + 4 + 2 + 4 + 10 = 30 per family and the pilot 5,
    # so 230 leave at least 195 for the bank; read-only for the tests
    tmp_path = tmp_path_factory.mktemp("bank")
    train = collect_games(make_split(tmp_path, "train", {f: 230 for f in ALFWORLD_TASK_TYPES}),
                          "train")
    unseen = collect_games(make_unseen(tmp_path), "valid_unseen")
    splits = build_split_manifest(train, unseen, full_plan())
    return train, splits, build_pilot_manifest(train, splits)


def test_the_bank_takes_the_smallest_hashes_of_the_unused_games(synthetic):
    train, splits, pilot = synthetic
    bank = build_benefit_bank(train, splits, pilot)
    verify_benefit_bank(bank, splits, pilot)
    assert build_benefit_bank(train, splits, pilot) == bank  # deterministic
    excluded = {g["game_id"] for pool in POOL_ORDER for g in splits["pools"][pool]["games"]}
    excluded |= {g["game_id"] for g in pilot["games"]}
    for family, games in train.by_family().items():
        left = sorted((g for g in games if g.game_id not in excluded),
                      key=lambda g: bank_rank(family, g.game_id))
        assert [g["game_id"] for g in bank["families"][family]] == \
            [g.game_id for g in left[:BANK_PER_FAMILY]]
        assert bank["available_by_family"][family] == len(left)


@pytest.mark.parametrize("left, fails", [(191, True), (192, False)])
def test_fewer_than_192_games_left_fail_closed_and_192_suffice(synthetic, tmp_path, left, fails):
    """The family keeps exactly ``left`` games outside the pools and the pilot."""
    _, splits, _ = synthetic
    family = "look_at_obj_in_light"
    used = sum(1 for pool in POOL_ORDER for g in splits["pools"][pool]["games"]
               if g["family"] == family) + 5  # the pilot takes 5 per family
    counts = {f: 230 for f in ALFWORLD_TASK_TYPES}
    counts[family] = used + left
    train = collect_games(make_split(tmp_path, "train", counts), "train")
    unseen = collect_games(make_unseen(tmp_path), "valid_unseen")
    splits = build_split_manifest(train, unseen, full_plan())
    pilot = build_pilot_manifest(train, splits)
    assert sum(1 for pool in POOL_ORDER for g in splits["pools"][pool]["games"]
               if g["family"] == family) + 5 == used  # the same plan uses as many
    if fails:
        with pytest.raises(SplitError, match=rf"benefit_bank\[{family}\]: needs 192 games, 191"):
            build_benefit_bank(train, splits, pilot)
    else:
        bank = build_benefit_bank(train, splits, pilot)
        assert bank["available_by_family"][family] == 192
        verify_benefit_bank(bank, splits, pilot)


def test_the_bank_needs_the_collection_the_splits_were_built_from(tmp_path, synthetic):
    _, splits, pilot = synthetic
    other = collect_games(make_split(tmp_path / "other", "train",
                                     {f: 2 for f in ALFWORLD_TASK_TYPES}), "train")
    with pytest.raises(SplitError, match="differs from the one the split manifest"):
        build_benefit_bank(other, splits, pilot)


def test_t_eval_ids_shared_with_train_games_fail_closed(tmp_path):
    train = collect_games(make_split(tmp_path, "train", {f: 230 for f in ALFWORLD_TASK_TYPES}),
                          "train")
    unseen = collect_games(make_split(tmp_path, "valid_unseen", UNSEEN_PER_FAMILY), "valid_unseen")
    splits = build_split_manifest(train, unseen, full_plan())  # same ids in both splits
    with pytest.raises(SplitError, match="T_eval game ids are also train game ids"):
        build_benefit_bank(train, splits, build_pilot_manifest(train, splits))


def test_the_bank_needs_the_pool_games_as_the_manifests_record_them(synthetic):
    """Equal counts are not enough: every pool and pilot game must be in the
    collection with its recorded content."""
    train, splits, _ = synthetic
    edited = copy.deepcopy(splits)
    edited["pools"]["t_mine"]["games"][0]["content_sha256"] = "0" * 64
    edited["pools"]["t_mine"]["sha256"] = _sha256(edited["pools"]["t_mine"]["games"])
    edited["manifest_sha256"] = _sha256({k: v for k, v in edited.items()
                                         if k != "manifest_sha256"})
    pilot = build_pilot_manifest(train, edited)
    with pytest.raises(SplitError, match="not in the train collection as the split"):
        build_benefit_bank(train, edited, pilot)


@pytest.mark.parametrize("tamper, message", [
    (lambda b, s, p: _swap_in(b, s["pools"]["t_mine"]["games"][0]), "T_eval included"),
    (lambda b, s, p: _swap_in(b, s["pools"]["generalization_bank"]["games"][-1]),
     "T_eval included"),
    (lambda b, s, p: _swap_in(b, p["games"][0]), "T_eval included"),
    (lambda b, s, p: b["families"]["pick_two_obj_and_place"].reverse(), "hash order"),
    (lambda b, s, p: b["families"]["pick_two_obj_and_place"].pop(), "needs 192 distinct"),
    (lambda b, s, p: b["families"]["look_at_obj_in_light"].__setitem__(
        1, b["families"]["look_at_obj_in_light"][0]), "needs 192 distinct"),
    (lambda b, s, p: b["families"]["look_at_obj_in_light"][0].__setitem__(
        "family", "pick_and_place_simple"), "another family"),
    # a game id names its family, so a game cannot sit in two families' lists
    (lambda b, s, p: _cross_family_duplicate(b), "another family"),
    (lambda b, s, p: b["families"].pop("pick_cool_then_place_in_recep"), "six ALFWorld families"),
    (lambda b, s, p: b["selection"].__setitem__("per_family", 191), "selection"),
    (lambda b, s, p: b.__setitem__("pilot_sha256", "0" * 64), "another pilot manifest"),
    (lambda b, s, p: b.__setitem__("split_manifest_sha256", "0" * 64), "another split manifest"),
    (lambda b, s, p: b.__setitem__("families", 7), "malformed"),
])
def test_a_rehashed_but_wrong_bank_fails_closed(synthetic, tamper, message):
    """Content checks hold even when a forger recomputes both hashes."""
    train, splits, pilot = synthetic
    bank = build_benefit_bank(train, splits, pilot)
    tamper(bank, splits, pilot)
    with pytest.raises(SplitError, match=message):
        verify_benefit_bank(_rehash(bank), splits, pilot)


def test_the_split_builder_never_draws_the_bank(committed):
    """The bank sits outside the split builder's POOL_ORDER and the committed
    split manifest's pool order (ADR-0012 D1)."""
    _, splits, _ = committed
    assert "benefit_bank" not in POOL_ORDER
    assert "benefit_bank" not in splits["pool_order"] and "benefit_bank" not in splits["pools"]
