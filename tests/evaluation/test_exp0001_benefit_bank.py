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
from tests.evaluation.test_alfworld_splits import UNSEEN_PER_FAMILY, full_plan, make_split

ROOT = Path(__file__).resolve().parents[2]
MANIFESTS = ROOT / "experiments" / "manifests"
LOCKED_SPLITS_SHA256 = "2f74d188"  # ADR-0012 D1 names the split manifest by these digits
LOCKED_PILOT_SHA256 = "4fc2b6fe"


def load(name):
    return json.loads((MANIFESTS / name).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def committed():
    return load("EXP-0001-benefit-bank.json"), load("EXP-0001-splits.json"), \
        load("EXP-0001-pilot.json")


def test_the_committed_bank_verifies_against_the_locked_manifests(committed):
    bank, splits, pilot = committed
    verify_benefit_bank(bank, splits, pilot)
    assert bank["split_manifest_sha256"].startswith(LOCKED_SPLITS_SHA256)
    assert bank["pilot_sha256"].startswith(LOCKED_PILOT_SHA256)
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


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    # pools take at most 10 + 4 + 2 + 4 + 10 = 30 per family and the pilot 5,
    # so 230 leave at least 195 for the bank; read-only for the tests
    tmp_path = tmp_path_factory.mktemp("bank")
    train = collect_games(make_split(tmp_path, "train", {f: 230 for f in ALFWORLD_TASK_TYPES}),
                          "train")
    unseen = collect_games(make_split(tmp_path, "valid_unseen", UNSEEN_PER_FAMILY), "valid_unseen")
    splits = build_split_manifest(train, unseen, full_plan())
    return train, splits, build_pilot_manifest(train, splits)


def _rehash(bank):
    from evaluation.benchmarks.alfworld.splits import _sha256
    bank["games_sha256"] = _sha256(bank["families"])
    bank["bank_sha256"] = _sha256({k: v for k, v in bank.items() if k != "bank_sha256"})
    return bank


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


def test_too_few_games_left_for_the_bank_fail_closed(tmp_path):
    # 230 per family except one family with 220: at most 30 + 5 used leaves 185 < 192
    counts = {f: 230 for f in ALFWORLD_TASK_TYPES}
    counts["look_at_obj_in_light"] = 220
    train = collect_games(make_split(tmp_path, "train", counts), "train")
    unseen = collect_games(make_split(tmp_path, "valid_unseen", UNSEEN_PER_FAMILY), "valid_unseen")
    splits = build_split_manifest(train, unseen, full_plan())
    with pytest.raises(SplitError, match=r"benefit_bank\[look_at_obj_in_light\]: needs 192"):
        build_benefit_bank(train, splits, build_pilot_manifest(train, splits))


def test_the_bank_needs_the_collection_the_splits_were_built_from(tmp_path, synthetic):
    _, splits, pilot = synthetic
    other = collect_games(make_split(tmp_path / "other", "train",
                                     {f: 2 for f in ALFWORLD_TASK_TYPES}), "train")
    with pytest.raises(SplitError, match="differs from the one the split manifest"):
        build_benefit_bank(other, splits, pilot)


def _swap_in(bank, game):
    """Put ``game`` in place of a bank game of its family, keeping hash order."""
    family = game["family"]
    games = bank["families"][family][1:] + [game]
    bank["families"][family] = sorted(games, key=lambda g: bank_rank(family, g["game_id"]))


@pytest.mark.parametrize("tamper, message", [
    (lambda b, s, p: _swap_in(b, s["pools"]["t_mine"]["games"][0]), "in an EXP-0001 pool or the pilot"),
    (lambda b, s, p: _swap_in(b, s["pools"]["generalization_bank"]["games"][-1]),
     "in an EXP-0001 pool or the pilot"),
    (lambda b, s, p: _swap_in(b, p["games"][0]), "in an EXP-0001 pool or the pilot"),
    (lambda b, s, p: b["families"]["pick_two_obj_and_place"].reverse(), "hash order"),
    (lambda b, s, p: b["families"]["pick_two_obj_and_place"].pop(), "needs 192 distinct"),
    (lambda b, s, p: b["families"]["look_at_obj_in_light"].__setitem__(
        1, b["families"]["look_at_obj_in_light"][0]), "needs 192 distinct"),
    (lambda b, s, p: b["families"]["look_at_obj_in_light"][0].__setitem__(
        "family", "pick_and_place_simple"), "another family"),
    (lambda b, s, p: b["families"].pop("pick_cool_then_place_in_recep"), "six ALFWorld families"),
    (lambda b, s, p: b["selection"].__setitem__("per_family", 191), "selection"),
    (lambda b, s, p: b.__setitem__("pilot_sha256", "0" * 64), "another pilot manifest"),
    (lambda b, s, p: b.__setitem__("split_manifest_sha256", "0" * 64), "another split manifest"),
])
def test_a_rehashed_but_wrong_bank_fails_closed(synthetic, tamper, message):
    """Content checks hold even when a forger recomputes both hashes."""
    train, splits, pilot = synthetic
    bank = build_benefit_bank(train, splits, pilot)
    tamper(bank, splits, pilot)
    with pytest.raises(SplitError, match=message):
        verify_benefit_bank(_rehash(bank), splits, pilot)


def test_the_split_builder_never_draws_the_bank():
    """The bank sits outside POOL_ORDER, so neither the split manifest nor
    the pilot's draw (which reads POOL_ORDER) changes (ADR-0012 D1)."""
    assert "benefit_bank" not in POOL_ORDER
