"""The EXP-0001 benefit bank (ADR-0012 D1; owner, 2026-10-19).

The gate's stage 2 (ADR-0012 D4) and its out-of-scope check (D5) play
games from this bank. It holds, per family, the 192 loader-accepted train
games of ALFWorld 0.2.2 that are in no pool of the split manifest and not
among the calibration pilot's 30, taken by the smallest
``sha256('heritagent-split|0|benefit_bank|<family>|<game_id>')`` (the split
builder's keyed-hash rule). Within a family the bank's order is ascending
by that hash; allocations read the bank in that order.

The bank has its own manifest, so neither the split manifest nor the
pilot's draw changes: ``benefit_bank`` is not in the split builder's
``POOL_ORDER``. Building it makes no model call. A family with fewer than
192 such games fails closed; ADR-0004's fallback is a split-level rule and
does not cover this bank."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping
from pathlib import Path

from evaluation.benchmarks.alfworld.splits import (
    ALFWORLD_TASK_TYPES,
    POOL_ORDER,
    CollectionReport,
    SplitError,
    _sha256,
    collect_games,
    verify_manifest,
)
from evaluation.exp0001.pilot import verify_pilot_manifest

BANK_SCHEMA = "heritagent/alfworld-benefit-bank/0.1"
BANK_POOL = "benefit_bank"
BANK_SEED = 0
BANK_PER_FAMILY = 192
BANK_RULE = ("per family: the 192 train games in no pool of the split manifest and not in "
             "the calibration pilot with the smallest "
             "sha256('heritagent-split|0|benefit_bank|<family>|<game_id>')")
BANK_ORDER = "within a family, ascending by the selection hash"


def bank_rank(family: str, game_id: str) -> str:
    """A game's selection hash; the bank keeps the smallest, in this order."""
    key = f"heritagent-split|{BANK_SEED}|{BANK_POOL}|{family}|{game_id}"
    return hashlib.sha256(key.encode()).hexdigest()


def _excluded(split_manifest: Mapping, pilot_manifest: Mapping) -> set[str]:
    """Train game ids the bank may not take: every train-side pool and the pilot."""
    used = {record["game_id"] for pool in POOL_ORDER
            for record in split_manifest["pools"][pool]["games"]}
    return used | {record["game_id"] for record in pilot_manifest["games"]}


def build_benefit_bank(train: CollectionReport, split_manifest: dict,
                       pilot_manifest: dict) -> dict:
    """Draw the bank from the train games no pool and no pilot game uses."""
    verify_pilot_manifest(pilot_manifest, split_manifest)  # verifies the split manifest too
    if train.source != "train":
        raise SplitError("the benefit bank is drawn from the train collection")
    if train.summary() != split_manifest["collections"]["train"]:
        raise SplitError("the train collection differs from the one the split manifest "
                         "was built from")
    excluded = _excluded(split_manifest, pilot_manifest)
    available, families = {}, {}
    for family, games in train.by_family().items():
        left = [g for g in games if g.game_id not in excluded]
        available[family] = len(left)
        if len(left) < BANK_PER_FAMILY:
            raise SplitError(f"{BANK_POOL}[{family}]: needs {BANK_PER_FAMILY} games, "
                             f"{len(left)} are in no pool and not in the pilot; the bank "
                             f"fails closed (ADR-0012 D1; ADR-0004's fallback does not "
                             f"cover it)")
        ranked = sorted(left, key=lambda g: bank_rank(family, g.game_id))
        families[family] = [g.record() for g in ranked[:BANK_PER_FAMILY]]
    bank = {
        "schema": BANK_SCHEMA,
        "experiment": "EXP-0001",
        "decision": "ADR-0012 D1",
        "split_manifest_sha256": split_manifest["manifest_sha256"],
        "pilot_sha256": pilot_manifest["pilot_sha256"],
        "selection": {"pool": BANK_POOL, "seed": BANK_SEED, "per_family": BANK_PER_FAMILY,
                      "rule": BANK_RULE, "order": BANK_ORDER,
                      "drawn_from": "train games in no EXP-0001 pool and not in the "
                                    "calibration pilot"},
        "available_by_family": available,
        "families": families,
        "games_sha256": _sha256(families),
        "status": "frozen before any EXP-0001 episode",
    }
    bank["bank_sha256"] = _sha256(bank)
    return bank


def verify_benefit_bank(bank: Mapping, split_manifest: dict, pilot_manifest: dict) -> None:
    """Fail closed on drift, on a bank built for another split or pilot
    manifest, on a family that is not 192 distinct games of its own family
    in hash order, and on any game a pool or the pilot uses. Which games
    were left to draw from needs the data: ``build_benefit_bank`` from the
    same collection must give this manifest."""
    verify_pilot_manifest(pilot_manifest, split_manifest)
    body = {k: v for k, v in bank.items() if k != "bank_sha256"}
    if _sha256(body) != bank.get("bank_sha256"):
        raise SplitError("bank_sha256 does not match the benefit bank's content")
    if bank.get("schema") != BANK_SCHEMA:
        raise SplitError(f"the benefit bank's schema is {bank.get('schema')!r}, not {BANK_SCHEMA}")
    if _sha256(bank["families"]) != bank["games_sha256"]:
        raise SplitError("the benefit bank's games do not match the recorded sha256")
    if bank["split_manifest_sha256"] != split_manifest["manifest_sha256"]:
        raise SplitError("the benefit bank was drawn for another split manifest")
    if bank["pilot_sha256"] != pilot_manifest["pilot_sha256"]:
        raise SplitError("the benefit bank was drawn for another pilot manifest")
    selection = bank["selection"]
    if (selection["pool"], selection["seed"], selection["per_family"]) != \
            (BANK_POOL, BANK_SEED, BANK_PER_FAMILY):
        raise SplitError("the benefit bank's selection is not ADR-0012 D1's")
    if sorted(bank["families"]) != sorted(ALFWORLD_TASK_TYPES):
        raise SplitError("the benefit bank must hold exactly the six ALFWorld families")
    excluded = _excluded(split_manifest, pilot_manifest)
    seen: set[str] = set()
    for family, records in bank["families"].items():
        ids = [record["game_id"] for record in records]
        if len(ids) != BANK_PER_FAMILY or len(set(ids)) != len(ids):
            raise SplitError(f"{BANK_POOL}[{family}]: needs {BANK_PER_FAMILY} distinct games")
        if any(record["family"] != family for record in records):
            raise SplitError(f"{BANK_POOL}[{family}]: holds a game of another family")
        ranks = [bank_rank(family, game_id) for game_id in ids]
        if ranks != sorted(ranks):
            raise SplitError(f"{BANK_POOL}[{family}]: the games are not in hash order")
        overlap = sorted(excluded & set(ids))
        if overlap:
            raise SplitError(f"{BANK_POOL}[{family}]: games in an EXP-0001 pool or the "
                             f"pilot: {overlap[:5]}")
        seen |= set(ids)
    if len(seen) != BANK_PER_FAMILY * len(ALFWORLD_TASK_TYPES):
        raise SplitError("a game is in two families of the benefit bank")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the EXP-0001 benefit bank (ADR-0012 D1).")
    parser.add_argument("--data-root", type=Path, required=True,
                        help="directory holding the train/ split (e.g. $ALFWORLD_DATA/json_2.1.1)")
    parser.add_argument("--splits", type=Path, required=True, help="the EXP-0001 split manifest")
    parser.add_argument("--pilot", type=Path, required=True, help="the calibration pilot manifest")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    split_manifest = json.loads(args.splits.read_text(encoding="utf-8"))
    pilot_manifest = json.loads(args.pilot.read_text(encoding="utf-8"))
    bank = build_benefit_bank(collect_games(args.data_root / "train", "train"),
                              split_manifest, pilot_manifest)
    args.out.write_text(json.dumps(bank, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.out} (bank_sha256 {bank['bank_sha256']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
