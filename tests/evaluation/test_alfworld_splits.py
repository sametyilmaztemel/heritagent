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


def test_content_hash_tracks_the_game_files(tmp_path):
    split = make_split(tmp_path, "train", {F[0]: 2})
    before = collect_games(split, "train").accepted[0]
    game_file = split / before.game_id / "game.tw-pddl"
    game_file.write_text(json.dumps({"solvable": True, "i": 0, "edited": True}))
    after = collect_games(split, "train").accepted[0]
    assert after.game_id == before.game_id and after.content_sha256 != before.content_sha256


def test_content_hash_framing_is_pinned(tmp_path):
    """Known answer: both files, labelled and length-prefixed, in this order."""
    trial = tmp_path / "train" / "pick_and_place_simple-A-None-B-1" / "trial_T1"
    trial.mkdir(parents=True)
    (trial / "traj_data.json").write_bytes(b'{"task_type": "pick_and_place_simple"}')
    (trial / "game.tw-pddl").write_bytes(b'{"solvable": true}')
    expected = hashlib.sha256(
        b"traj_data.json\0" + b"38\0" + b'{"task_type": "pick_and_place_simple"}'
        + b"game.tw-pddl\0" + b"18\0" + b'{"solvable": true}').hexdigest()
    [game] = collect_games(tmp_path / "train", "train").accepted
    assert game.content_sha256 == expected


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
    _, _, unseen = data
    sizes = {f: 35 for f in F}
    sizes[F[1]] = 23  # 10 + 4 + 2 + 4 = 20 drawn first, 3 left for the bank
    train = collect_games(make_split(tmp_path / "fb", "train", sizes), "train")
    plan = full_plan(counts={"generalization_bank": {**{f: 10 for f in F}, F[1]: 3}},
                     feasibility_fallback={"generalization_bank": {F[1]: 3}})
    manifest = build_split_manifest(train, unseen, plan)
    assert manifest["pools"]["generalization_bank"]["family_counts"][F[1]] == 3
    assert manifest["plan"]["feasibility_fallback"] == {"generalization_bank": {F[1]: 3}}
    assert manifest["pools"]["generalization_bank"]["feasibility_fallback"] == {
        F[1]: {"maximum": 3, "available_at_draw": 3, "locked_minimum": 10}}
    assert "feasibility_fallback" not in manifest["pools"]["replay_bank"]


# -- review of PR #34: a fallback only where availability prevents the minimum -----

@pytest.mark.parametrize("maximum", [3, 0])
def test_fallback_without_an_availability_shortfall_fails_closed(data, maximum):
    """35 games per family: 15 remain for the generalization bank, so the
    locked minimum of 10 is available and no fallback may waive it."""
    _, train, unseen = data
    plan = full_plan(counts={"generalization_bank": {**{f: 10 for f in F}, F[1]: maximum}},
                     feasibility_fallback={"generalization_bank": {F[1]: maximum}})
    with pytest.raises(SplitError, match="15 games remain.*does not prevent the locked minimum 10"):
        build_split_manifest(train, unseen, plan)


def test_zero_fallback_builds_when_nothing_remains(tmp_path, data):
    _, _, unseen = data
    sizes = {f: 35 for f in F}
    sizes[F[1]] = 16  # t_mine 10 + ablation 4 + interaction 2: nothing left for the banks
    train = collect_games(make_split(tmp_path / "z", "train", sizes), "train")
    plan = full_plan(counts={"replay_bank": {**{f: 4 for f in F}, F[1]: 0},
                             "generalization_bank": {**{f: 10 for f in F}, F[1]: 0}},
                     feasibility_fallback={"replay_bank": {F[1]: 0},
                                           "generalization_bank": {F[1]: 0}})
    manifest = build_split_manifest(train, unseen, plan)
    for pool, minimum in (("replay_bank", 4), ("generalization_bank", 10)):
        assert F[1] not in manifest["pools"][pool]["family_counts"]
        assert manifest["pools"][pool]["feasibility_fallback"] == {
            F[1]: {"maximum": 0, "available_at_draw": 0, "locked_minimum": minimum}}


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


def test_proportional_counts_infeasible():
    with pytest.raises(SplitError):
        proportional_counts({F[0]: 3, F[1]: 3}, 10)
    with pytest.raises(SplitError):
        proportional_counts({f: 100 for f in F}, 40, per_family_min=8)
