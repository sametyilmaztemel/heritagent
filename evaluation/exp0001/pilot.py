"""The EXP-0001 calibration pilot manifest (ADR-0003 "Experiment needed";
ADR-0007 E2, owner decision 2026-10-14).

The pilot runs on train games disjoint from every EXP-0001 pool: the games
the split manifest left, 5 per family, drawn with seed 0 by the split
builder's keyed-hash rule under the pool name ``calibration_pilot``. Its
selection and its measurement plan are frozen here, before any pilot
episode. The pilot measures; it tunes nothing (a protocol change after the
pilot is a pre-execution amendment or a new experiment id)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from evaluation.benchmarks.alfworld.splits import (
    ALFWORLD_TASK_TYPES,
    POOL_ORDER,
    CollectionReport,
    SplitError,
    _draw,
    _sha256,
    collect_games,
    verify_manifest,
)

PILOT_SCHEMA = "heritagent/alfworld-pilot-manifest/0.1"
PILOT_POOL = "calibration_pilot"
PILOT_SEED = 0
PILOT_PER_FAMILY = 5
# what the pilot reports (ADR-0003 "Experiment needed"; #13 Phase A handoff;
# the compute-cap decision, takeover note section 5.12)
PILOT_MEASURES = (
    "g0_success_rate_p0",
    "invalid_command_rate",
    "parse_failure_rate",
    "retry_rate",
    "steps_to_success",
    "model_calls_per_episode",
    "throughput_episodes_per_hour",
    "arm_b_rendering_smoke_test",
    "temperature_0_repeat_determinism",
)
PILOT_EXECUTION = {
    "genome": "G0 (planner_react_v2, 50 steps; ADR-0003 D4)",
    "settings": "evaluation settings: temperature 0.0, the ReAct stop (ADR-0003 A2)",
    "runs": "every pilot game once",
    "determinism_repeat": "the first game of each family (game_id order) a second time; "
                          "trajectories must be identical",
    "arm_b_rendering_smoke_test": "one fixed synthetic skill block rendered into G0's prompt; "
                                  "checks the ReAct format survives, measures nothing else",
    "no_tuning": "the pilot measures only; no protocol, prompt or threshold changes inside it",
}


def build_pilot_manifest(train: CollectionReport, split_manifest: dict) -> dict:
    """Draw the pilot from the train games no EXP-0001 pool uses."""
    verify_manifest(split_manifest)
    if train.source != "train":
        raise SplitError("the pilot is drawn from the train collection")
    if train.summary() != split_manifest["collections"]["train"]:
        raise SplitError("the train collection differs from the one the split manifest "
                         "was built from")
    used = {record["game_id"] for pool in POOL_ORDER
            for record in split_manifest["pools"][pool]["games"]}
    remaining = {family: [g for g in games if g.game_id not in used]
                 for family, games in train.by_family().items()}
    games = []
    for family in ALFWORLD_TASK_TYPES:
        games.extend(_draw(remaining[family], PILOT_PER_FAMILY, PILOT_SEED, PILOT_POOL, family))
    records = [g.record() for g in games]
    pilot = {
        "schema": PILOT_SCHEMA,
        "experiment": "EXP-0001",
        "split_manifest_sha256": split_manifest["manifest_sha256"],
        "selection": {"pool": PILOT_POOL, "seed": PILOT_SEED, "per_family": PILOT_PER_FAMILY,
                      "rule": split_manifest["selection_rule"],
                      "drawn_from": "train games in no EXP-0001 pool"},
        "games": records,
        "games_sha256": _sha256(records),
        "measures": list(PILOT_MEASURES),
        "execution": dict(PILOT_EXECUTION),
        "status": "frozen before any pilot episode",
    }
    pilot["pilot_sha256"] = _sha256(pilot)
    return pilot


def verify_pilot_manifest(pilot: dict, split_manifest: dict) -> None:
    """Fail closed on drift, on a pilot built for another split manifest,
    and on any overlap with an EXP-0001 pool."""
    verify_manifest(split_manifest)
    body = {k: v for k, v in pilot.items() if k != "pilot_sha256"}
    if _sha256(body) != pilot.get("pilot_sha256"):
        raise SplitError("pilot_sha256 does not match the pilot manifest content")
    if _sha256(pilot["games"]) != pilot["games_sha256"]:
        raise SplitError("pilot games do not match the recorded sha256")
    if pilot["split_manifest_sha256"] != split_manifest["manifest_sha256"]:
        raise SplitError("the pilot was drawn for another split manifest")
    # the pilot is train-side; a game is (source split, game_id), so only the
    # train-side pools can overlap it (T_eval is valid_unseen)
    pooled = {record["game_id"] for pool in POOL_ORDER
              for record in split_manifest["pools"][pool]["games"]}
    overlap = sorted(pooled & {record["game_id"] for record in pilot["games"]})
    if overlap:
        raise SplitError(f"pilot games are in EXP-0001 pools: {overlap[:5]}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the EXP-0001 calibration pilot manifest.")
    parser.add_argument("--data-root", type=Path, required=True,
                        help="directory holding the train/ split (e.g. $ALFWORLD_DATA/json_2.1.1)")
    parser.add_argument("--splits", type=Path, required=True, help="the EXP-0001 split manifest")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    split_manifest = json.loads(args.splits.read_text(encoding="utf-8"))
    pilot = build_pilot_manifest(collect_games(args.data_root / "train", "train"), split_manifest)
    args.out.write_text(json.dumps(pilot, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.out} (pilot_sha256 {pilot['pilot_sha256']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
