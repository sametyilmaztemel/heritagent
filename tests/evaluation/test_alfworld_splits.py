"""ALFWorld split-manifest builder (EXP-0001 splits; no ALFWorld data needed:
the tests build a synthetic tree in the ALFWorld json_2.1.1 layout)."""

import copy
import hashlib
import json
from pathlib import Path

import pytest

from evaluation.benchmarks.alfworld.splits import (
    ALFWORLD_TASK_TYPES,
    POOL_ORDER,
    SplitError,
    SplitPlan,
    build_split_manifest,
    collect_games,
    main,
    proportional_counts,
    validate_plan,
    verify_manifest,
)

F = ALFWORLD_TASK_TYPES


def write_game(split_dir: Path, family: str, index: int, *, solvable=True,
               with_game=True, solvable_key=True, prefix=None) -> str:
    task_dir = prefix or f"{family}-Obj{index}-None-Recep-{index % 30}"
    trial = split_dir / task_dir / f"trial_T{index:04d}"
    trial.mkdir(parents=True)
    (trial / "traj_data.json").write_text(json.dumps({"task_type": family, "i": index}))
    if with_game:
        game = {"solvable": solvable, "i": index} if solvable_key else {"i": index}
        (trial / "game.tw-pddl").write_text(json.dumps(game))
    return trial.relative_to(split_dir).as_posix()


def make_split(root: Path, split: str, per_family: dict[str, int], *, with_rejects=False) -> Path:
    split_dir = root / split
    for family, n in per_family.items():
        for i in range(n):
            write_game(split_dir, family, i)
    if with_rejects:
        write_game(split_dir, F[0], 900, prefix="pick_and_place_simple-movable-x")
        write_game(split_dir, F[0], 901, prefix="pick_and_place_simple-Sliced-x")
        write_game(split_dir, "pick_and_place_with_movable_recep", 902, prefix="other-task")
        write_game(split_dir, F[1], 903, with_game=False)
        write_game(split_dir, F[1], 904, solvable_key=False)
        write_game(split_dir, F[2], 905, solvable=False)
    return split_dir


UNSEEN_PER_FAMILY = dict(zip(F, (24, 18, 31, 23, 21, 17)))  # 134 in total


def full_plan(**overrides) -> SplitPlan:
    counts = {
        "t_mine": {f: 10 for f in F},
        "ablation": dict(zip(F, (4, 4, 3, 3, 3, 3))),
        "interaction": dict(zip(F, (2, 2, 2, 2, 1, 1))),
        "replay_bank": {f: 4 for f in F},
        "generalization_bank": {f: 10 for f in F},
    }
    for pool, value in overrides.pop("counts", {}).items():
        counts[pool] = value
    return SplitPlan(seed=overrides.pop("seed", 20261013), **counts, **overrides)


@pytest.fixture
def data(tmp_path):
    train = make_split(tmp_path, "train", {f: 35 for f in F}, with_rejects=True)
    unseen = make_split(tmp_path, "valid_unseen", UNSEEN_PER_FAMILY)
    return tmp_path, collect_games(train, "train"), collect_games(unseen, "valid_unseen")


# -- collection mirrors the alfworld loader ---------------------------------------

def test_collector_applies_the_alfworld_filters(data):
    _, train, unseen = data
    assert len(train.accepted) == 35 * 6
    assert dict(train.skipped) == {"movable_or_sliced": 2, "task_type_excluded": 1,
                                   "missing_game_file": 1, "missing_solvable_key": 1,
                                   "unsolvable": 1}
    assert len(unseen.accepted) == 134
    ids = [g.game_id for g in train.accepted]
    assert ids == sorted(ids)


def test_collection_is_independent_of_creation_order(tmp_path):
    a = make_split(tmp_path / "a", "train", {F[0]: 5, F[3]: 5})
    b_dir = tmp_path / "b" / "train"
    for family, i in reversed([(f, i) for f in (F[0], F[3]) for i in range(5)]):
        write_game(b_dir, family, i)
    ra, rb = collect_games(a, "train"), collect_games(b_dir, "train")
    assert [g.record() for g in ra.accepted] == [g.record() for g in rb.accepted]


def test_games_are_ordered_by_game_id_not_by_path(tmp_path):
    """Path order compares components: '<task>-x' sorts before '<task>-x-y',
    while the game-id strings sort the other way ('-' < '/')."""
    split = tmp_path / "train"
    short = write_game(split, F[0], 1, prefix=f"{F[0]}-x")
    long = write_game(split, F[0], 2, prefix=f"{F[0]}-x-y")
    assert sorted([split / short, split / long]) == [split / short, split / long]
    ids = [g.game_id for g in collect_games(split, "train").accepted]
    assert ids == [long, short] == sorted(ids)


def test_content_hash_tracks_the_game_files(tmp_path):
    split = make_split(tmp_path, "train", {F[0]: 2})
    before = collect_games(split, "train").accepted[0]
    game_file = split / before.game_id / "game.tw-pddl"
    game_file.write_text(json.dumps({"solvable": True, "i": 0, "edited": True}))
    after = collect_games(split, "train").accepted[0]
    assert after.game_id == before.game_id and after.content_sha256 != before.content_sha256


def test_content_hash_framing_is_pinned(tmp_path):
    """Known answer: both files' raw bytes, labelled and length-prefixed (in
    bytes), in this order. Non-canonical JSON, CRLF and a non-ASCII
    character catch re-serialisation, newline and character-count changes."""
    traj = b'{"task_type":"pick_and_place_simple","n":"\xc3\xa9"}\r\n'
    game = b'{ "solvable" : true }\r\n'
    trial = tmp_path / "train" / "pick_and_place_simple-A-None-B-1" / "trial_T1"
    trial.mkdir(parents=True)
    (trial / "traj_data.json").write_bytes(traj)
    (trial / "game.tw-pddl").write_bytes(game)
    expected = hashlib.sha256(
        b"traj_data.json\0" + str(len(traj)).encode() + b"\0" + traj
        + b"game.tw-pddl\0" + str(len(game)).encode() + b"\0" + game).hexdigest()
    assert expected == "0d5c79fa5cebc0eebffd72dbef75bea3c8efce9820874a7a4f2f49f7d251d26c"
    [instance] = collect_games(tmp_path / "train", "train").accepted
    assert instance.content_sha256 == expected


@pytest.mark.parametrize("edited", ["traj_data.json", "game.tw-pddl"])
def test_an_edited_game_file_changes_the_manifest(tmp_path, data, edited):
    """Review of PR #34: the hash must cover both runtime inputs, and it must
    reach the manifest (pool hash and manifest hash), not only GameInstance."""
    root, train, unseen = data
    before = build_split_manifest(train, unseen, full_plan())
    for pool in ("t_mine", "t_eval"):
        game_id = before["pools"][pool]["games"][0]["game_id"]
        split = root / ("train" if pool == "t_mine" else "valid_unseen")
        path = split / game_id / edited
        path.write_text(json.dumps({**json.loads(path.read_text()), "edited": True}))
    after = build_split_manifest(collect_games(root / "train", "train"),
                                 collect_games(root / "valid_unseen", "valid_unseen"), full_plan())
    for pool in ("t_mine", "t_eval"):
        old, new = before["pools"][pool], after["pools"][pool]
        assert [g["game_id"] for g in old["games"]] == [g["game_id"] for g in new["games"]]
        assert old["games"][0]["content_sha256"] != new["games"][0]["content_sha256"]
        assert old["games"][1:] == new["games"][1:]
        assert old["sha256"] != new["sha256"]
    assert before["manifest_sha256"] != after["manifest_sha256"]


def test_manifest_records_carry_each_games_content_hash(data):
    _, train, unseen = data
    manifest = build_split_manifest(train, unseen, full_plan())
    by_id = {g.game_id: g for g in (*train.accepted, *unseen.accepted)}
    for pool in (*POOL_ORDER, "t_eval"):
        for record in manifest["pools"][pool]["games"]:
            game = by_id[record["game_id"]]
            assert record == {"game_id": game.game_id, "family": game.family,
                              "content_sha256": game.content_sha256}


# -- plan validation enforces the locked counts -----------------------------------

@pytest.mark.parametrize("pool,counts,match", [
    ("t_mine", {f: 9 for f in F}, "locked total is 60"),
    ("t_mine", dict(zip(F, (7, 13, 10, 10, 10, 10))), "below the locked per-family minimum 8"),
    ("ablation", {f: 3 for f in F}, "locked total is 20"),
    ("interaction", {F[0]: 11}, "locked total is 10"),
    ("replay_bank", {**{f: 4 for f in F}, F[2]: 3}, "minimum 4"),
    ("generalization_bank", {**{f: 10 for f in F}, F[5]: 9}, "minimum 10"),
    ("t_mine", {**{f: 10 for f in F[:5]}, "made_up_family": 10}, "unknown families"),
    ("ablation", {**dict(zip(F, (4, 4, 3, 3, 3, 2))), F[5]: True}, "non-negative int"),
])
def test_plan_violations_fail_closed(pool, counts, match):
    with pytest.raises(SplitError, match=match):
        validate_plan(full_plan(counts={pool: counts}))


def test_fallback_is_bank_only_and_must_match_the_count():
    with pytest.raises(SplitError, match="only allowed"):
        validate_plan(full_plan(counts={"t_mine": dict(zip(F, (6, 14, 10, 10, 10, 10)))},
                                feasibility_fallback={"t_mine": {F[0]: 6}}))
    with pytest.raises(SplitError, match="must equal the pre-registered fallback maximum 7"):
        validate_plan(full_plan(counts={"generalization_bank": {**{f: 10 for f in F}, F[1]: 8}},
                                feasibility_fallback={"generalization_bank": {F[1]: 7}}))
    validate_plan(full_plan(counts={"generalization_bank": {**{f: 10 for f in F}, F[1]: 7}},
                            feasibility_fallback={"generalization_bank": {F[1]: 7}}))


@pytest.mark.parametrize("seed", [True, 1.0, "1", None])
def test_seed_must_be_an_int(seed):
    with pytest.raises(SplitError, match="seed"):
        full_plan(seed=seed)


def test_plan_is_immutable():
    plan = full_plan()
    with pytest.raises(TypeError):
        plan.t_mine[F[0]] = 99


# -- building the manifest ---------------------------------------------------------

def test_pools_have_the_planned_sizes_and_are_disjoint(data):
    _, train, unseen = data
    plan = full_plan()
    manifest = build_split_manifest(train, unseen, plan)
    seen = set()
    for pool in POOL_ORDER:
        games = manifest["pools"][pool]["games"]
        assert manifest["pools"][pool]["family_counts"] == {
            f: n for f, n in plan.counts(pool).items() if n}
        ids = {g["game_id"] for g in games}
        assert len(ids) == len(games) and not ids & seen
        seen |= ids
    assert len(manifest["pools"]["t_mine"]["games"]) == 60
    assert manifest["t_reg"]["status"].startswith("pending")


def test_t_eval_is_the_whole_unseen_split_and_never_sampled(data):
    _, train, unseen = data
    manifest = build_split_manifest(train, unseen, full_plan())
    t_eval = manifest["pools"]["t_eval"]
    assert t_eval["sampled"] is False and t_eval["source"] == "valid_unseen"
    assert [g["game_id"] for g in t_eval["games"]] == [g.game_id for g in unseen.accepted]
    train_side = {g["game_id"] for p in POOL_ORDER for g in manifest["pools"][p]["games"]}
    assert all(g.source == "train" for g in train.accepted)
    assert len(train_side) == 60 + 20 + 10 + 6 * 4 + 6 * 10


def test_unseen_split_must_have_134_games(tmp_path, data):
    _, train, _ = data
    short = collect_games(make_split(tmp_path / "s", "valid_unseen", {F[0]: 133}), "valid_unseen")
    with pytest.raises(SplitError, match="134"):
        build_split_manifest(train, short, full_plan())


def test_build_is_deterministic_and_seed_sensitive(data):
    _, train, unseen = data
    a = build_split_manifest(train, unseen, full_plan())
    b = build_split_manifest(train, unseen, full_plan())
    c = build_split_manifest(train, unseen, full_plan(seed=7))
    assert a == b
    assert a["manifest_sha256"] != c["manifest_sha256"]
    assert a["pools"]["t_mine"]["games"] != c["pools"]["t_mine"]["games"]


def test_selection_rule_is_recomputable_by_hand(data):
    _, train, unseen = data
    plan = full_plan()
    manifest = build_split_manifest(train, unseen, plan)
    family = F[3]
    candidates = [g.game_id for g in train.accepted if g.family == family]
    ranked = sorted(candidates, key=lambda gid: hashlib.sha256(
        f"heritagent-split|{plan.seed}|t_mine|{family}|{gid}".encode()).hexdigest())
    expected = sorted(ranked[:plan.t_mine[family]])
    got = [g["game_id"] for g in manifest["pools"]["t_mine"]["games"] if g["family"] == family]
    assert got == expected


def test_selection_rule_is_recomputable_for_every_pool(data):
    """Each pool ranks the games still remaining after the earlier pools,
    with the pool name in the key."""
    _, train, unseen = data
    plan = full_plan()
    manifest = build_split_manifest(train, unseen, plan)
    for family in F:
        remaining = [g.game_id for g in train.accepted if g.family == family]
        for pool in POOL_ORDER:
            k = plan.counts(pool).get(family, 0)
            ranked = sorted(remaining, key=lambda gid: hashlib.sha256(
                f"heritagent-split|{plan.seed}|{pool}|{family}|{gid}".encode()).hexdigest())
            expected = sorted(ranked[:k])
            got = [g["game_id"] for g in manifest["pools"][pool]["games"] if g["family"] == family]
            assert got == expected, (pool, family)
            remaining = [gid for gid in remaining if gid not in set(expected)]


def test_not_enough_games_fails_closed(tmp_path, data):
    _, _, unseen = data
    small = collect_games(make_split(tmp_path / "small", "train", {f: 20 for f in F}), "train")
    with pytest.raises(SplitError, match="remain after earlier pools"):
        build_split_manifest(small, unseen, full_plan())


def test_feasibility_fallback_builds_with_the_declared_maximum(tmp_path, data):
    """21 games: below the floors through the generalization bank (8 + 4 + 10),
    so no plan could give it 10. With only its floors earlier (t_mine 8,
    replay 4), the bank takes the 9 games left (ADR-0004)."""
    _, _, unseen = data
    sizes = {f: 35 for f in F}
    sizes[F[1]] = 21
    train = collect_games(make_split(tmp_path / "fb", "train", sizes), "train")
    plan = scarce_plan(counts={"generalization_bank": {**{f: 10 for f in F}, F[1]: 9}},
                       feasibility_fallback={"generalization_bank": {F[1]: 9}})
    manifest = build_split_manifest(train, unseen, plan)
    assert manifest["pools"]["generalization_bank"]["family_counts"][F[1]] == 9
    assert manifest["plan"]["feasibility_fallback"] == {"generalization_bank": {F[1]: 9}}
    assert manifest["pools"]["generalization_bank"]["feasibility_fallback"] == {
        F[1]: {"maximum": 9, "available_at_draw": 9, "accepted_in_train": 21,
               "locked_floor_through_bank": 22, "locked_minimum": 10}}
    assert "feasibility_fallback" not in manifest["pools"]["replay_bank"]


# -- review of PR #34 / #42: a fallback only where availability prevents the minimum --

def _spread(total: int, keys) -> dict:
    keys = list(keys)
    base, extra = divmod(total, len(keys))
    return {k: base + (i < extra) for i, k in enumerate(keys)}


def scarce_plan(*families, **overrides) -> SplitPlan:
    """The given families (default F[1]) get only their locked floors before
    the banks: t_mine 8, nothing in ablation or interaction. The other
    families share the rest of each locked total."""
    families = families or (F[1],)
    others = [f for f in F if f not in families]
    counts = {
        "t_mine": {**{f: 8 for f in families}, **_spread(60 - 8 * len(families), others)},
        "ablation": {**{f: 0 for f in families}, **_spread(20, others)},
        "interaction": {**{f: 0 for f in families}, **_spread(10, others)},
        **overrides.pop("counts", {}),
    }
    return full_plan(counts=counts, **overrides)


@pytest.mark.parametrize("counts", [
    {"generalization_bank": {**{f: 10 for f in F}, F[1]: 3}},
    {"generalization_bank": {**{f: 10 for f in F}, F[1]: 0}},
    # review of #42: the plan's own replay count must not manufacture the shortfall
    {"replay_bank": {**{f: 4 for f in F}, F[1]: 12},
     "generalization_bank": {**{f: 10 for f in F}, F[1]: 7}},
])
def test_fallback_without_an_availability_shortfall_fails_closed(data, counts):
    _, train, unseen = data
    maximum = counts["generalization_bank"][F[1]]
    plan = full_plan(counts=counts, feasibility_fallback={"generalization_bank": {F[1]: maximum}})
    with pytest.raises(SplitError, match=r"35 accepted train games, enough for the locked "
                                         r"per-family floors through this bank \(22\).*minimum 10"):
        build_split_manifest(train, unseen, plan)


@pytest.mark.parametrize("pool,size,floor,minimum,maximum", [
    ("generalization_bank", 22, 22, 10, 2),   # boundary: 8 + 4 + 10 fit exactly
    ("replay_bank", 12, 12, 4, 3),            # boundary: 8 + 4 fit exactly
    ("replay_bank", 35, 12, 4, 2),
])
def test_fallback_at_the_locked_floor_fails_closed(tmp_path, data, pool, size, floor,
                                                   minimum, maximum):
    _, _, unseen = data
    sizes = {f: 35 for f in F}
    sizes[F[1]] = size
    train = collect_games(make_split(tmp_path / "b", "train", sizes), "train")
    base = {"replay_bank": 4, "generalization_bank": 10}[pool]
    plan = scarce_plan(counts={pool: {**{f: base for f in F}, F[1]: maximum}},
                       feasibility_fallback={pool: {F[1]: maximum}})
    with pytest.raises(SplitError, match=rf"{size} accepted train games.*through this bank "
                                         rf"\({floor}\).*locked minimum {minimum}"):
        build_split_manifest(train, unseen, plan)


@pytest.mark.parametrize("pool,size,floor,maximum", [
    ("generalization_bank", 22, 22, 2),
    ("replay_bank", 12, 12, 3),
])
def test_fallback_at_the_floor_fails_closed_whatever_the_plan_draws_first(
        tmp_path, data, pool, size, floor, maximum):
    """Review of #43: with full_plan the family spends more than its floors in
    t_mine (10), ablation (4) and interaction (2), so a check relative to the
    plan's allocation would see a shortfall. The family could still meet the
    minimum, so the fallback must fail closed."""
    _, _, unseen = data
    sizes = {f: 35 for f in F}
    sizes[F[1]] = size
    train = collect_games(make_split(tmp_path / "p", "train", sizes), "train")
    base = {"replay_bank": 4, "generalization_bank": 10}[pool]
    plan = full_plan(counts={pool: {**{f: base for f in F}, F[1]: maximum}},
                     feasibility_fallback={pool: {F[1]: maximum}})
    with pytest.raises(SplitError, match=rf"{size} accepted train games.*\({floor}\)"):
        build_split_manifest(train, unseen, plan)


@pytest.mark.parametrize("valid,invalid", [(F[1], F[3]), (F[5], F[0])])
def test_every_fallback_family_is_checked(tmp_path, data, valid, invalid):
    """One family is genuinely short (21 games, given only its floors), the
    other is not (35); the invalid one sorts after, then before, the valid
    one."""
    _, _, unseen = data
    sizes = {f: 35 for f in F}
    sizes[valid] = 21
    train = collect_games(make_split(tmp_path / "f", "train", sizes), "train")
    plan = scarce_plan(valid,
                       counts={"generalization_bank": {**{f: 10 for f in F}, valid: 9, invalid: 3}},
                       feasibility_fallback={"generalization_bank": {valid: 9, invalid: 3}})
    with pytest.raises(SplitError, match=rf"generalization_bank\[{invalid}\].*35 accepted"):
        build_split_manifest(train, unseen, plan)


@pytest.mark.parametrize("invalid_pool", ["replay_bank", "generalization_bank"])
def test_every_fallback_pool_is_checked(tmp_path, data, invalid_pool):
    """F[1] (11 games) is short in both banks; F[2] (35) is not. The invalid
    fallback sits in the later, then the earlier, pool of the plan."""
    _, _, unseen = data
    sizes = {f: 35 for f in F}
    sizes[F[1]] = 11
    train = collect_games(make_split(tmp_path / "q", "train", sizes), "train")
    counts = {"replay_bank": {**{f: 4 for f in F}, F[1]: 3},
              "generalization_bank": {**{f: 10 for f in F}, F[1]: 0}}
    fallback = {"replay_bank": {F[1]: 3}, "generalization_bank": {F[1]: 0}}
    bad = {"replay_bank": 2, "generalization_bank": 3}[invalid_pool]
    counts[invalid_pool] = {**counts[invalid_pool], F[2]: bad}
    fallback[invalid_pool] = {**fallback[invalid_pool], F[2]: bad}
    plan = scarce_plan(counts=counts, feasibility_fallback=fallback)
    with pytest.raises(SplitError, match=rf"{invalid_pool}\[{F[2]}\].*35 accepted"):
        build_split_manifest(train, unseen, plan)


def test_fallback_one_below_the_locked_floor_builds(tmp_path, data):
    """11 games for F[1]: t_mine 8 leaves 3 for the replay bank (minimum 4)
    and none for the generalization bank."""
    _, _, unseen = data
    sizes = {f: 35 for f in F}
    sizes[F[1]] = 11
    train = collect_games(make_split(tmp_path / "s", "train", sizes), "train")
    plan = scarce_plan(counts={"replay_bank": {**{f: 4 for f in F}, F[1]: 3},
                               "generalization_bank": {**{f: 10 for f in F}, F[1]: 0}},
                       feasibility_fallback={"replay_bank": {F[1]: 3},
                                             "generalization_bank": {F[1]: 0}})
    manifest = build_split_manifest(train, unseen, plan)
    assert manifest["pools"]["replay_bank"]["family_counts"][F[1]] == 3
    assert manifest["pools"]["replay_bank"]["feasibility_fallback"] == {
        F[1]: {"maximum": 3, "available_at_draw": 3, "accepted_in_train": 11,
               "locked_floor_through_bank": 12, "locked_minimum": 4}}
    assert manifest["pools"]["generalization_bank"]["feasibility_fallback"] == {
        F[1]: {"maximum": 0, "available_at_draw": 0, "accepted_in_train": 11,
               "locked_floor_through_bank": 22, "locked_minimum": 10}}


@pytest.mark.parametrize("pool,size,maximum,target", [
    ("generalization_bank", 21, 8, 9),   # fewer than the data leaves
    ("generalization_bank", 21, 0, 9),
    ("generalization_bank", 18, 7, 6),   # more than the data leaves (below the minimum 10)
    ("replay_bank", 11, 2, 3),
    ("replay_bank", 11, 0, 3),
])
def test_fallback_maximum_must_equal_what_the_data_leaves(tmp_path, data, pool, size,
                                                          maximum, target):
    """ADR-0004: a fallback bank takes every game the family has left after
    its locked floors in the earlier pools; nothing fewer."""
    _, _, unseen = data
    sizes = {f: 35 for f in F}
    sizes[F[1]] = size
    train = collect_games(make_split(tmp_path / "eq", "train", sizes), "train")
    counts = {pool: {**{f: {"replay_bank": 4, "generalization_bank": 10}[pool] for f in F},
                     F[1]: maximum}}
    fallback = {pool: {F[1]: maximum}}
    if pool == "replay_bank":  # nothing is left for the generalization bank
        counts["generalization_bank"] = {**{f: 10 for f in F}, F[1]: 0}
        fallback["generalization_bank"] = {F[1]: 0}
    plan = scarce_plan(counts=counts, feasibility_fallback=fallback)
    with pytest.raises(SplitError, match=rf"{pool}\[{F[1]}\]: the pre-registered fallback "
                                         rf"maximum {maximum} must equal the {target} games"):
        build_split_manifest(train, unseen, plan)


@pytest.mark.parametrize("pool,value", [
    ("t_mine", 9), ("ablation", 1), ("interaction", 1), ("replay_bank", 5)])
def test_fallback_family_takes_only_its_floors_from_earlier_pools(tmp_path, data, pool, value):
    """Review of #48 (owner decision): a plan must not shrink a fallback bank
    by giving the family more than its floors earlier. F[0] absorbs the
    difference so every locked total still holds."""
    _, _, unseen = data
    sizes = {f: 35 for f in F}
    sizes[F[1]] = 21
    train = collect_games(make_split(tmp_path / "pin", "train", sizes), "train")
    base = scarce_plan()
    counts = {pool: {**dict(base.counts(pool)), F[1]: value}}
    if pool != "replay_bank":
        counts[pool][F[0]] -= value - dict(base.counts(pool))[F[1]]
    plan = scarce_plan(counts={**counts,
                               "generalization_bank": {**{f: 10 for f in F}, F[1]: 9}},
                       feasibility_fallback={"generalization_bank": {F[1]: 9}})
    need = {"t_mine": 8, "ablation": 0, "interaction": 0, "replay_bank": 4}[pool]
    with pytest.raises(SplitError, match=rf"so {pool}\[{F[1]}\] must be {need}; "
                                         rf"the plan has {value}"):
        build_split_manifest(train, unseen, plan)


def test_full_plan_allocation_cannot_shrink_a_fallback_bank(tmp_path, data):
    """The review's example: under full_plan (t_mine 10, ablation 4,
    interaction 2) a 21-game family would leave 1 game for the bank instead
    of 9."""
    _, _, unseen = data
    sizes = {f: 35 for f in F}
    sizes[F[1]] = 21
    train = collect_games(make_split(tmp_path / "fp", "train", sizes), "train")
    plan = full_plan(counts={"generalization_bank": {**{f: 10 for f in F}, F[1]: 1}},
                     feasibility_fallback={"generalization_bank": {F[1]: 1}})
    with pytest.raises(SplitError, match=rf"t_mine\[{F[1]}\] must be 8; the plan has 10"):
        build_split_manifest(train, unseen, plan)


def test_two_fallback_families_in_one_bank(tmp_path, data):
    """Both families get their data-determined size; a wrong maximum for
    either one, first or second, fails closed."""
    _, _, unseen = data
    sizes = {f: 35 for f in F}
    sizes[F[1]], sizes[F[4]] = 21, 18
    train = collect_games(make_split(tmp_path / "two", "train", sizes), "train")

    def plan_with(a, b):
        return scarce_plan(F[1], F[4],
                           counts={"generalization_bank": {**{f: 10 for f in F}, F[1]: a, F[4]: b}},
                           feasibility_fallback={"generalization_bank": {F[1]: a, F[4]: b}})
    manifest = build_split_manifest(train, unseen, plan_with(9, 6))
    evidence = manifest["pools"]["generalization_bank"]["feasibility_fallback"]
    assert {f: e["maximum"] for f, e in evidence.items()} == {F[1]: 9, F[4]: 6}
    with pytest.raises(SplitError, match=rf"\[{F[4]}\].*maximum 5 must equal the 6 games"):
        build_split_manifest(train, unseen, plan_with(9, 5))
    with pytest.raises(SplitError, match=rf"\[{F[1]}\].*maximum 8 must equal the 9 games"):
        build_split_manifest(train, unseen, plan_with(8, 6))


def test_omitted_family_with_games_left_fails_closed(tmp_path, data):
    """A zero fallback for a family left out of the bank counts is only
    valid when the data leaves it nothing (review of #48)."""
    _, _, unseen = data
    sizes = {f: 35 for f in F}
    sizes[F[1]] = 21
    train = collect_games(make_split(tmp_path / "om", "train", sizes), "train")
    plan = scarce_plan(counts={"generalization_bank": {f: 10 for f in F if f != F[1]}},
                       feasibility_fallback={"generalization_bank": {F[1]: 0}})
    validate_plan(plan)
    with pytest.raises(SplitError, match="maximum 0 must equal the 9 games"):
        build_split_manifest(train, unseen, plan)


def test_family_below_the_t_mine_minimum_fails_closed(tmp_path, data):
    """T_mine has no fallback: a family with 5 games cannot be built."""
    _, _, unseen = data
    sizes = {f: 35 for f in F}
    sizes[F[1]] = 5
    train = collect_games(make_split(tmp_path / "t5", "train", sizes), "train")
    plan = scarce_plan(counts={"replay_bank": {**{f: 4 for f in F}, F[1]: 0},
                               "generalization_bank": {**{f: 10 for f in F}, F[1]: 0}},
                       feasibility_fallback={"replay_bank": {F[1]: 0},
                                             "generalization_bank": {F[1]: 0}})
    with pytest.raises(SplitError, match="5 accepted train games cannot meet the locked "
                                         "t_mine minimum 8"):
        build_split_manifest(train, unseen, plan)


def test_zero_fallback_for_a_family_omitted_from_the_bank_counts(tmp_path, data):
    """8 games for F[1], all in T_mine. The banks leave F[1] out of their
    counts (as proportional_counts does for an empty family); a fallback of
    0 is the declared maximum."""
    _, _, unseen = data
    sizes = {f: 35 for f in F}
    sizes[F[1]] = 8
    train = collect_games(make_split(tmp_path / "z", "train", sizes), "train")
    without = {pool: {f: n for f, n in {"replay_bank": {f: 4 for f in F},
                                        "generalization_bank": {f: 10 for f in F}}[pool].items()
                      if f != F[1]}
               for pool in ("replay_bank", "generalization_bank")}
    plan = scarce_plan(counts=without, feasibility_fallback={"replay_bank": {F[1]: 0},
                                                             "generalization_bank": {F[1]: 0}})
    validate_plan(plan)
    manifest = build_split_manifest(train, unseen, plan)
    for pool, floor, minimum in (("replay_bank", 12, 4), ("generalization_bank", 22, 10)):
        assert F[1] not in manifest["pools"][pool]["family_counts"]
        assert manifest["pools"][pool]["feasibility_fallback"] == {
            F[1]: {"maximum": 0, "available_at_draw": 0, "accepted_in_train": 8,
                   "locked_floor_through_bank": floor, "locked_minimum": minimum}}


@pytest.mark.parametrize("pool,maximum", [
    ("generalization_bank", 10),   # not below the minimum
    ("generalization_bank", 12),
    ("replay_bank", 4),
    ("replay_bank", -1),
    ("replay_bank", True),
    ("replay_bank", 2.0),
])
def test_fallback_maximum_must_be_an_int_below_the_minimum(pool, maximum):
    base = {"replay_bank": 4, "generalization_bank": 10}[pool]
    plan = full_plan(counts={pool: {**{f: base for f in F}, F[2]: maximum}},
                     feasibility_fallback={pool: {F[2]: maximum}})
    with pytest.raises(SplitError, match="fallback maximum must be an int"):
        validate_plan(plan)


@pytest.mark.parametrize("pool,maximum", [("replay_bank", 3), ("generalization_bank", 9)])
def test_fallback_maximum_just_below_the_minimum_is_accepted(pool, maximum):
    base = {"replay_bank": 4, "generalization_bank": 10}[pool]
    validate_plan(full_plan(counts={pool: {**{f: base for f in F}, F[2]: maximum}},
                            feasibility_fallback={pool: {F[2]: maximum}}))


def test_count_below_the_declared_maximum_fails_closed():
    with pytest.raises(SplitError, match="must equal the pre-registered fallback maximum 3"):
        validate_plan(full_plan(counts={"replay_bank": {**{f: 4 for f in F}, F[2]: 2}},
                                feasibility_fallback={"replay_bank": {F[2]: 3}}))


def test_fallback_for_an_unknown_family_fails_closed():
    with pytest.raises(SplitError, match="unknown family"):
        validate_plan(full_plan(feasibility_fallback={"replay_bank": {"made_up_family": 0}}))


def test_verify_manifest_detects_tampering(data):
    _, train, unseen = data
    manifest = build_split_manifest(train, unseen, full_plan())
    verify_manifest(manifest)
    swapped = copy.deepcopy(manifest)
    swapped["pools"]["t_mine"]["games"][0]["game_id"] = "pick_and_place_simple-X/trial_T9999"
    with pytest.raises(SplitError, match="t_mine"):
        verify_manifest(swapped)
    relabelled = copy.deepcopy(manifest)
    relabelled["plan"]["seed"] = 1
    with pytest.raises(SplitError, match="manifest_sha256"):
        verify_manifest(relabelled)


def test_cli_writes_a_verifiable_manifest(data):
    root, _, _ = data
    plan_path = root / "plan.json"
    plan_path.write_text(json.dumps(full_plan().record()))
    out = root / "manifest.json"
    assert main(["--data-root", str(root), "--plan", str(plan_path), "--out", str(out)]) == 0
    manifest = json.loads(out.read_text())
    verify_manifest(manifest)
    assert SplitPlan.from_json(manifest["plan"]).record() == full_plan().record()


# -- proposal helper -------------------------------------------------------------

def test_proportional_counts_respects_floor_and_total():
    availability = dict(zip(F, (790, 300, 620, 460, 520, 750)))
    counts = proportional_counts(availability, 60, per_family_min=8)
    assert sum(counts.values()) == 60 and min(counts.values()) >= 8
    assert counts == proportional_counts(dict(reversed(list(availability.items()))), 60, 8)
    assert counts[F[0]] >= counts[F[1]]


def test_proportional_counts_breaks_exact_ties_by_family_name():
    """Shares 4/3, 1/3, 1/3: equal remainders. In floats 4/3 - 1 is the
    smallest, so float shares would give the extra game to the wrong family."""
    availability = {F[1]: 4, F[0]: 1, F[2]: 1}  # F[1] sorts first by name
    assert proportional_counts(availability, 2) == {F[1]: 2, F[0]: 0, F[2]: 0}


def test_proportional_counts_infeasible():
    with pytest.raises(SplitError):
        proportional_counts({F[0]: 3, F[1]: 3}, 10)
    with pytest.raises(SplitError):
        proportional_counts({f: 100 for f in F}, 40, per_family_min=8)
