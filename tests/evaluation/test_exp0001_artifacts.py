"""The committed EXP-0001 pre-run artifacts (issue #13 Phase A; ADR-0007 E2):
the split plan, the split manifest built from the real ALFWorld data, and
the calibration-pilot manifest. They verify from their own hashes, so no
ALFWorld data is needed here. The pilot builder is also tested on a
synthetic tree."""

import copy
import json
from collections import Counter
from pathlib import Path

import pytest

from evaluation.benchmarks.alfworld.splits import (
    ALFWORLD_TASK_TYPES,
    POOL_ORDER,
    SplitError,
    SplitPlan,
    build_split_manifest,
    collect_games,
    proportional_counts,
    verify_manifest,
)
from evaluation.exp0001.pilot import (
    PILOT_PER_FAMILY,
    PILOT_SEED,
    build_pilot_manifest,
    verify_pilot_manifest,
)
from tests.evaluation.test_alfworld_splits import UNSEEN_PER_FAMILY, full_plan, make_split

ROOT = Path(__file__).resolve().parents[2]
MANIFESTS = ROOT / "experiments" / "manifests"


def load(name):
    return json.loads((MANIFESTS / name).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def splits():
    return load("EXP-0001-splits.json")


def test_the_committed_split_manifest_verifies(splits):
    verify_manifest(splits)
    assert splits["plan"] == SplitPlan.from_json(load("EXP-0001-split-plan.json")).record()
    assert splits["plan"]["seed"] == 0  # ADR-0007 E2
    assert splits["t_reg"]["status"].startswith("pending")


def test_the_pools_have_the_locked_sizes_and_are_disjoint(splits):
    pools = splits["pools"]
    sizes = {name: len(pool["games"]) for name, pool in pools.items()}
    assert sizes == {"t_mine": 60, "ablation": 20, "interaction": 10, "replay_bank": 24,
                     "generalization_bank": 60, "t_eval": 134}
    assert all(pools["t_mine"]["family_counts"][f] >= 8 for f in ALFWORLD_TASK_TYPES)
    assert pools["replay_bank"]["family_counts"] == {f: 4 for f in ALFWORLD_TASK_TYPES}
    assert pools["generalization_bank"]["family_counts"] == {f: 10 for f in ALFWORLD_TASK_TYPES}
    ids = [g["game_id"] for pool in pools.values() for g in pool["games"]]
    assert len(ids) == len(set(ids))
    assert pools["t_eval"]["source"] == "valid_unseen" and pools["t_eval"]["sampled"] is False
    assert all(not pool.get("feasibility_fallback") for pool in pools.values())


def test_the_plan_counts_follow_from_the_recorded_availability(splits):
    """The counts are the proportional proposal for the recorded train
    availability, so they can be recomputed from the manifest alone."""
    available = splits["collections"]["train"]["accepted_by_family"]
    counts = splits["plan"]["counts"]
    assert counts["t_mine"] == proportional_counts(available, 60, 8)
    assert counts["ablation"] == proportional_counts(available, 20)
    assert counts["interaction"] == proportional_counts(available, 10)
    assert splits["collections"]["valid_unseen"]["accepted"] == 134


def test_the_committed_pilot_is_disjoint_and_verifies(splits):
    pilot = load("EXP-0001-pilot.json")
    verify_pilot_manifest(pilot, splits)
    assert Counter(g["family"] for g in pilot["games"]) == {f: PILOT_PER_FAMILY
                                                            for f in ALFWORLD_TASK_TYPES}
    assert pilot["selection"]["seed"] == PILOT_SEED == 0
    assert "model_calls_per_episode" in pilot["measures"]
    assert pilot["status"] == "frozen before any pilot episode"


def test_a_tampered_pilot_fails_closed(splits):
    pilot = load("EXP-0001-pilot.json")
    swapped = copy.deepcopy(pilot)
    swapped["games"][0] = splits["pools"]["t_mine"]["games"][0]
    with pytest.raises(SplitError):
        verify_pilot_manifest(swapped, splits)
    stale = copy.deepcopy(pilot)
    stale["split_manifest_sha256"] = "0" * 64
    with pytest.raises(SplitError):
        verify_pilot_manifest(stale, splits)


@pytest.fixture
def synthetic(tmp_path):
    train = collect_games(make_split(tmp_path, "train", {f: 40 for f in ALFWORLD_TASK_TYPES}), "train")
    unseen = collect_games(make_split(tmp_path, "valid_unseen", UNSEEN_PER_FAMILY), "valid_unseen")
    return train, build_split_manifest(train, unseen, full_plan())


def test_the_pilot_draws_from_the_games_no_pool_uses(synthetic):
    train, manifest = synthetic
    pilot = build_pilot_manifest(train, manifest)
    verify_pilot_manifest(pilot, manifest)
    used = {g["game_id"] for pool in POOL_ORDER for g in manifest["pools"][pool]["games"]}
    assert not used & {g["game_id"] for g in pilot["games"]}
    assert Counter(g["family"] for g in pilot["games"]) == {f: 5 for f in ALFWORLD_TASK_TYPES}
    assert build_pilot_manifest(train, manifest) == pilot  # deterministic


def test_the_pilot_needs_the_collection_the_splits_were_built_from(tmp_path, synthetic):
    _, manifest = synthetic
    other = collect_games(make_split(tmp_path / "other", "train",
                                     {f: 41 for f in ALFWORLD_TASK_TYPES}), "train")
    with pytest.raises(SplitError, match="differs from the one the split manifest"):
        build_pilot_manifest(other, manifest)


def test_too_few_games_left_for_the_pilot_fail_closed(tmp_path):
    # 10 + 4 + 3 + 4 + 10 = 31 per family at most go to the pools; 34 leave 3 < 5
    train = collect_games(make_split(tmp_path, "train", {f: 34 for f in ALFWORLD_TASK_TYPES}), "train")
    unseen = collect_games(make_split(tmp_path, "valid_unseen", UNSEEN_PER_FAMILY), "valid_unseen")
    with pytest.raises(SplitError, match="needs 5 games"):
        build_pilot_manifest(train, build_split_manifest(train, unseen, full_plan()))


def test_the_manifest_points_at_the_committed_artifacts(splits):
    """pyyaml is not a test dependency: run_protocol is read with patterns."""
    import re
    text = (MANIFESTS / "EXP-0001-minimal-cig.yaml").read_text(encoding="utf-8")
    block = re.search(r"^run_protocol:[^\n]*\n((?:  .*\n)+)", text, re.MULTILINE)
    assert block, "run_protocol is missing"
    body = block.group(1)
    assert re.search(rf"split_manifest: \{{file: EXP-0001-splits.json, sha256: "
                     rf"{splits['manifest_sha256']}\}}", body)
    pilot = load("EXP-0001-pilot.json")
    assert re.search(rf"calibration_pilot: \{{file: EXP-0001-pilot.json, sha256: "
                     rf"{pilot['pilot_sha256']},", body)
    assert re.search(r"lifetime: \{attempts_per_task: 3, temperature: 0.7,", body)
    assert re.search(r"t_reg: \{eligible: solved-in-any-lifetime-episode, size: 40, seed: 0,", body)
    assert re.search(r"budgets: \{max_steps: 50, max_total_retries: 10, max_tokens_per_request: 512\}",
                     body)
    assert re.search(r"generalization_allocation: \{split: even-over-families, .*"
                     r"all_family_applicability: vacuous-out-of-scope\}", body)
    assert "pending" not in block.group(0)  # E1-E8 are owner decisions
