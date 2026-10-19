"""The EXP-0001 benefit bank (ADR-0012 D1; owner, 2026-10-19).

Under ADR-0012 the gate's stage 2 (D4) and its out-of-scope check (D5)
draw their games from this bank (the gate PR wires them). It holds, per family, the 192 loader-accepted train
games of ALFWorld 0.2.2 that are in no pool of the split manifest and not
among the calibration pilot's 30, taken by the smallest
``sha256('heritagent-split|0|benefit_bank|<family>|<game_id>')`` (the split
builder's keyed-hash rule). Within a family the bank's order is ascending
by that hash; allocations are to read the bank in that order.

The bank has its own manifest, so neither the split manifest nor the
pilot's draw changes: ``benefit_bank`` is not in the split builder's
``POOL_ORDER``. Building it makes no model call. A family with fewer than
192 such games fails closed; ADR-0004's fallback is a split-level rule and
does not cover this bank."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections.abc import Mapping
from pathlib import Path

from evaluation.benchmarks.alfworld.splits import (
    ALFWORLD_TASK_TYPES,
    CollectionReport,
    SplitError,
    _sha256,
    collect_games,
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
BANK_DRAWN_FROM = "train games in no EXP-0001 pool and not in the calibration pilot"
BANK_STATUS = "frozen before any EXP-0001 episode"
RECORD_KEYS = {"game_id", "family", "content_sha256"}
# ALFWorld's game ids: <task directory>/trial_T<digits and underscores> (every accepted
# train and valid_unseen game of release 0.2.2); one spelling per game, no aliases
GAME_ID = re.compile(r"[A-Za-z0-9_.\-]+/trial_T[0-9_]+")


def bank_rank(family: str, game_id: str) -> str:
    """A game's selection hash; the bank keeps the smallest, in this order."""
    key = f"heritagent-split|{BANK_SEED}|{BANK_POOL}|{family}|{game_id}"
    return hashlib.sha256(key.encode()).hexdigest()


def _excluded(split_manifest: Mapping, pilot_manifest: Mapping) -> dict[str, Mapping]:
    """The game records the bank may not take, by game id: every pool of
    the split manifest (T_eval's valid_unseen ids included, though no train
    game shares one) and the pilot's games."""
    records = {record["game_id"]: record for pool in split_manifest["pools"].values()
               for record in pool["games"]}
    records.update({record["game_id"]: record for record in pilot_manifest["games"]})
    return records


def _available(split_manifest: Mapping, pilot_manifest: Mapping) -> dict[str, int]:
    """Per family, the loader-accepted train games left after the train-side
    pools and the pilot, from the manifests alone."""
    accepted = split_manifest["collections"]["train"]["accepted_by_family"]
    used = {family: 0 for family in ALFWORLD_TASK_TYPES}
    for pool in split_manifest["pools"].values():
        if pool["source"] == "train":
            for record in pool["games"]:
                used[record["family"]] += 1
    for record in pilot_manifest["games"]:
        used[record["family"]] += 1
    return {family: accepted[family] - used[family] for family in ALFWORLD_TASK_TYPES}


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
    collected = {g.game_id: g for g in train.accepted}
    shared = sorted(r["game_id"] for pool in split_manifest["pools"].values()
                    if pool["source"] != "train" for r in pool["games"] if r["game_id"] in collected)
    if shared:  # never in ALFWorld 0.2.2; the bank could not tell the two games apart
        raise SplitError(f"T_eval game ids are also train game ids ({shared[:3]}); the bank "
                         f"fails closed")
    train_side = [record for pool in split_manifest["pools"].values() if pool["source"] == "train"
                  for record in pool["games"]] + list(pilot_manifest["games"])
    for record in train_side:
        game = collected.get(record["game_id"])
        if game is None or (game.family, game.content_sha256) != \
                (record["family"], record["content_sha256"]):
            raise SplitError(f"{record['game_id']!r} is not in the train collection as the split "
                             f"or pilot manifest records it; the bank must be drawn from the "
                             f"data the manifests were built from")
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
                      "rule": BANK_RULE, "order": BANK_ORDER, "drawn_from": BANK_DRAWN_FROM},
        "available_by_family": available,
        "families": families,
        "games_sha256": _sha256(families),
        "status": BANK_STATUS,
    }
    bank["bank_sha256"] = _sha256(bank)
    return bank


def verify_benefit_bank(bank: Mapping, split_manifest: dict, pilot_manifest: dict) -> None:
    """Fail closed on drift, on a bank built for another split or pilot
    manifest, on any field off ADR-0012 D1's selection, on availability
    figures the manifests do not imply, on a family that is not 192
    distinct well-formed games of its own family in hash order, and on any
    game a pool (T_eval included) or the pilot uses. Which games were left
    to draw from needs the data: ``build_benefit_bank`` from the same
    collection must give this manifest."""
    verify_pilot_manifest(pilot_manifest, split_manifest)  # verifies the split manifest too
    try:
        _verify_bank(bank, split_manifest, pilot_manifest)
    except (KeyError, TypeError, AttributeError, ValueError) as exc:
        if isinstance(exc, SplitError):
            raise
        raise SplitError(f"the benefit bank is malformed ({exc!r})") from exc


def _verify_bank(bank: Mapping, split_manifest: dict, pilot_manifest: dict) -> None:
    body = {k: v for k, v in bank.items() if k != "bank_sha256"}
    if _sha256(body) != bank.get("bank_sha256"):
        raise SplitError("bank_sha256 does not match the benefit bank's content")
    expected_keys = {"schema", "experiment", "decision", "split_manifest_sha256", "pilot_sha256",
                     "selection", "available_by_family", "families", "games_sha256", "status",
                     "bank_sha256"}
    if set(bank) != expected_keys:
        raise SplitError(f"the benefit bank's fields are {sorted(bank)}, not {sorted(expected_keys)}")
    if (bank["schema"], bank["experiment"], bank["decision"], bank["status"]) != \
            (BANK_SCHEMA, "EXP-0001", "ADR-0012 D1", BANK_STATUS):
        raise SplitError("the benefit bank's schema, experiment, decision or status is not "
                         "ADR-0012 D1's")
    if _sha256(bank["families"]) != bank["games_sha256"]:
        raise SplitError("the benefit bank's games do not match the recorded sha256")
    if bank["split_manifest_sha256"] != split_manifest["manifest_sha256"]:
        raise SplitError("the benefit bank was drawn for another split manifest")
    if bank["pilot_sha256"] != pilot_manifest["pilot_sha256"]:
        raise SplitError("the benefit bank was drawn for another pilot manifest")
    selection = bank["selection"]
    expected_selection = {"pool": BANK_POOL, "seed": BANK_SEED, "per_family": BANK_PER_FAMILY,
                          "rule": BANK_RULE, "order": BANK_ORDER, "drawn_from": BANK_DRAWN_FROM}
    if set(selection) != set(expected_selection) or any(
            type(selection[k]) is not type(v) or selection[k] != v
            for k, v in expected_selection.items()):
        raise SplitError("the benefit bank's selection is not ADR-0012 D1's")
    if sorted(bank["families"]) != sorted(ALFWORLD_TASK_TYPES):
        raise SplitError("the benefit bank must hold exactly the six ALFWorld families")
    if any(type(v) is not int for v in bank["available_by_family"].values()) \
            or bank["available_by_family"] != _available(split_manifest, pilot_manifest):
        raise SplitError("the benefit bank's available_by_family is not what the split and "
                         "pilot manifests leave")
    excluded = _excluded(split_manifest, pilot_manifest)
    # a game id begins with its family's name, so no game sits in two families
    for family, records in bank["families"].items():
        for record in records:
            if set(record) != RECORD_KEYS or not isinstance(record["game_id"], str) \
                    or not isinstance(record["content_sha256"], str) \
                    or len(record["content_sha256"]) != 64 \
                    or any(c not in "0123456789abcdef" for c in record["content_sha256"]):
                raise SplitError(f"{BANK_POOL}[{family}]: a malformed game record {record!r}")
            if not GAME_ID.fullmatch(record["game_id"]):
                raise SplitError(f"{BANK_POOL}[{family}]: {record['game_id']!r} is not a "
                                 f"canonical ALFWorld game id")
            if record["family"] != family or not record["game_id"].startswith(family + "-"):
                raise SplitError(f"{BANK_POOL}[{family}]: holds a game of another family "
                                 f"({record['game_id']!r})")
            if "movable" in record["game_id"] or "Sliced" in record["game_id"]:
                raise SplitError(f"{BANK_POOL}[{family}]: {record['game_id']!r} is a game the "
                                 f"loader skips")
        ids = [record["game_id"] for record in records]
        if len(ids) != BANK_PER_FAMILY or len(set(ids)) != len(ids):
            raise SplitError(f"{BANK_POOL}[{family}]: needs {BANK_PER_FAMILY} distinct games")
        ranks = [bank_rank(family, game_id) for game_id in ids]
        if ranks != sorted(ranks):
            raise SplitError(f"{BANK_POOL}[{family}]: the games are not in hash order")
        overlap = sorted(set(excluded) & set(ids))
        if overlap:
            raise SplitError(f"{BANK_POOL}[{family}]: games in an EXP-0001 pool (T_eval "
                             f"included) or the pilot: {overlap[:5]}")


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
    verify_benefit_bank(bank, split_manifest, pilot_manifest)
    args.out.write_text(json.dumps(bank, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.out} (bank_sha256 {bank['bank_sha256']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
