"""ALFWorld split-manifest builder for EXP-0001 (prior audit P0-8; SPEC §24).

Materializes the pre-registered EXP-0001 splits (manifest
``environment.splits``; design §2) from the ALFWorld game files, and records
them with content hashes so a run can prove which instances it used.

What this module decides and what it does not:

- It applies the *locked* constraints (manifest + design §2):
  - T_mine: 60 train games, at least 8 per family;
  - T_gate pools: ablation 20, interaction 10;
  - family-indexed banks: replay at least 4 and generalization at least 10
    per family, or a family-specific maximum pre-registered as a
    feasibility fallback where availability prevents the minimum;
  - every train-side pool disjoint from every other;
  - T_eval = the whole official unseen split (134), never sampled.
- It does **not** choose the per-family counts or the sampling seed. Those
  are pre-registration inputs (issue #6) and come in through
  :class:`SplitPlan`. :func:`proportional_counts` is a helper for proposing
  counts; the counts that are used are the ones frozen in the plan.
- T_reg (40 games of T_mine that G0 solved) is chosen after the G0
  lifetime, so it is not built here. The manifest records it as pending.

Game collection mirrors ``AlfredTWEnv.collect_game_files`` of alfworld 0.4.2:
- skip trial directories whose path (relative to the split directory; alfworld
  tests the absolute path) contains ``movable`` or ``Sliced``;
- skip task types outside the six ALFWorld types;
- skip a trial with no ``game.tw-pddl``, or one whose ``game.tw-pddl`` has no
  ``solvable`` key or ``solvable: false``.

alfworld itself lists games in ``os.walk`` order, which depends on the file
system, and truncates by position. Here every list is sorted by a stable
game id instead.

Family ids are ALFWorld's own ``task_type`` labels; none are invented.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

ALFWORLD_TASK_TYPES = (
    "pick_and_place_simple",
    "look_at_obj_in_light",
    "pick_clean_then_place_in_recep",
    "pick_heat_then_place_in_recep",
    "pick_cool_then_place_in_recep",
    "pick_two_obj_and_place",
)
COLLECTOR_REFERENCE = "alfworld 0.4.2 AlfredTWEnv.collect_game_files (sorted, no truncation)"
MANIFEST_SCHEMA = "heritagent/alfworld-split-manifest/0.1"

# Locked EXP-0001 constraints (experiments/manifests/EXP-0001-minimal-cig.yaml,
# environment.splits; EXP-0001-design.md section 2). Pools are drawn in this
# order; changing the order changes the selection, so it is part of the
# builder's identity (recorded in the manifest).
POOL_ORDER = ("t_mine", "ablation", "interaction", "replay_bank", "generalization_bank")
POOL_TOTALS = {"t_mine": 60, "ablation": 20, "interaction": 10}
POOL_PER_FAMILY_MIN = {"t_mine": 8, "replay_bank": 4, "generalization_bank": 10}
FALLBACK_POOLS = ("replay_bank", "generalization_bank")
T_EVAL_SIZE = 134


class SplitError(ValueError):
    """The data, the plan or the selection violates a locked split rule."""


@dataclass(frozen=True)
class GameInstance:
    """One loader-accepted ALFWorld game.

    ``game_id`` is the trial directory relative to the split directory
    (POSIX), e.g. ``pick_heat_then_place_in_recep-Egg-None-Fridge-13/trial_T1``.
    ``content_sha256`` covers the bytes of ``traj_data.json`` and
    ``game.tw-pddl``, so a changed file of any game in the manifest changes
    its pool hash and the manifest hash."""

    game_id: str
    family: str
    source: str
    content_sha256: str

    def record(self) -> dict:
        return {"game_id": self.game_id, "family": self.family,
                "content_sha256": self.content_sha256}


@dataclass(frozen=True)
class CollectionReport:
    source: str
    accepted: tuple[GameInstance, ...]
    skipped: Mapping[str, int]

    def by_family(self) -> dict[str, list[GameInstance]]:
        out: dict[str, list[GameInstance]] = {f: [] for f in ALFWORLD_TASK_TYPES}
        for game in self.accepted:
            out[game.family].append(game)
        return out

    def summary(self) -> dict:
        counts = {f: len(g) for f, g in self.by_family().items()}
        return {"source": self.source, "accepted": len(self.accepted),
                "accepted_by_family": counts, "skipped": dict(sorted(self.skipped.items()))}


def _content_sha256(traj_bytes: bytes, game_bytes: bytes) -> str:
    digest = hashlib.sha256()
    for label, data in ((b"traj_data.json", traj_bytes), (b"game.tw-pddl", game_bytes)):
        digest.update(label + b"\0" + str(len(data)).encode() + b"\0" + data)
    return digest.hexdigest()


def collect_games(split_dir: Path, source: str) -> CollectionReport:
    """Loader-accepted games under one ALFWorld split directory (e.g.
    ``$ALFWORLD_DATA/json_2.1.1/train``), sorted by game id."""
    split_dir = Path(split_dir)
    if not split_dir.is_dir():
        raise SplitError(f"split directory not found: {split_dir}")
    skipped = {"movable_or_sliced": 0, "task_type_excluded": 0,
               "missing_game_file": 0, "missing_solvable_key": 0, "unsolvable": 0}
    accepted = []
    for traj_path in sorted(split_dir.rglob("traj_data.json")):
        trial = traj_path.parent
        game_id = trial.relative_to(split_dir).as_posix()
        if "movable" in game_id or "Sliced" in game_id:
            skipped["movable_or_sliced"] += 1
            continue
        traj_bytes = traj_path.read_bytes()
        family = json.loads(traj_bytes).get("task_type")
        if family not in ALFWORLD_TASK_TYPES:
            skipped["task_type_excluded"] += 1
            continue
        game_path = trial / "game.tw-pddl"
        if not game_path.is_file():
            skipped["missing_game_file"] += 1
            continue
        game_bytes = game_path.read_bytes()
        game_data = json.loads(game_bytes)
        if "solvable" not in game_data:
            skipped["missing_solvable_key"] += 1
            continue
        if not game_data["solvable"]:
            skipped["unsolvable"] += 1
            continue
        accepted.append(GameInstance(game_id, family, source,
                                     _content_sha256(traj_bytes, game_bytes)))
    # Path order compares components (and ignores case on Windows), which can
    # differ from game-id order; the T_eval record order feeds its pool hash.
    accepted.sort(key=lambda g: g.game_id)
    return CollectionReport(source, tuple(accepted), MappingProxyType(skipped))


def _frozen_counts(counts: Mapping[str, int]) -> Mapping[str, int]:
    return MappingProxyType({f: counts[f] for f in sorted(counts)})


@dataclass(frozen=True)
class SplitPlan:
    """Pre-registered split inputs: per-pool, per-family counts and the seed.

    ``feasibility_fallback[pool][family]`` declares a family-specific maximum
    below the per-family minimum of a bank (design §2). It is allowed only for
    the replay and generalization banks, and the count used for that family
    must equal the declared maximum. The build accepts it only where
    availability prevents the minimum: the family has fewer accepted train
    games than the locked per-family floors of every pool up to and
    including that bank, so no plan could meet the minimum."""

    seed: int
    t_mine: Mapping[str, int]
    ablation: Mapping[str, int]
    interaction: Mapping[str, int]
    replay_bank: Mapping[str, int]
    generalization_bank: Mapping[str, int]
    feasibility_fallback: Mapping[str, Mapping[str, int]] = field(default_factory=dict)

    def __post_init__(self):
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise SplitError(f"seed must be an int, got {self.seed!r}")
        for pool in POOL_ORDER:
            object.__setattr__(self, pool, _frozen_counts(getattr(self, pool)))
        object.__setattr__(self, "feasibility_fallback", MappingProxyType(
            {pool: _frozen_counts(v) for pool, v in sorted(self.feasibility_fallback.items())}))

    def counts(self, pool: str) -> Mapping[str, int]:
        return getattr(self, pool)

    def record(self) -> dict:
        return {"seed": self.seed,
                "counts": {pool: dict(self.counts(pool)) for pool in POOL_ORDER},
                "feasibility_fallback": {p: dict(v) for p, v in self.feasibility_fallback.items()}}

    @classmethod
    def from_json(cls, data: dict) -> "SplitPlan":
        counts = data["counts"]
        unknown = set(counts) - set(POOL_ORDER)
        if unknown:
            raise SplitError(f"unknown pools in plan: {sorted(unknown)}")
        return cls(seed=data["seed"], **{pool: counts[pool] for pool in POOL_ORDER},
                   feasibility_fallback=data.get("feasibility_fallback", {}))


def validate_plan(plan: SplitPlan) -> None:
    """Fail closed unless the plan satisfies every locked count rule."""
    for pool, fallback in plan.feasibility_fallback.items():
        if pool not in FALLBACK_POOLS:
            raise SplitError(f"feasibility fallback is only allowed for {FALLBACK_POOLS}, not {pool!r}")
        minimum = POOL_PER_FAMILY_MIN[pool]
        for family, maximum in fallback.items():
            if family not in ALFWORLD_TASK_TYPES:
                raise SplitError(f"{pool}: fallback for unknown family {family!r}")
            if isinstance(maximum, bool) or not isinstance(maximum, int) \
                    or not 0 <= maximum < minimum:
                raise SplitError(f"{pool}[{family}]: fallback maximum must be an int in "
                                 f"[0, {minimum}), below the locked minimum, got {maximum!r}")
            if plan.counts(pool).get(family, 0) != maximum:
                raise SplitError(f"{pool}[{family}]: count must equal the pre-registered "
                                 f"fallback maximum {maximum}")
    for pool in POOL_ORDER:
        counts = plan.counts(pool)
        unknown = set(counts) - set(ALFWORLD_TASK_TYPES)
        if unknown:
            raise SplitError(f"{pool}: unknown families {sorted(unknown)}")
        for family, n in counts.items():
            if isinstance(n, bool) or not isinstance(n, int) or n < 0:
                raise SplitError(f"{pool}[{family}]: count must be a non-negative int, got {n!r}")
        if pool in POOL_TOTALS and sum(counts.values()) != POOL_TOTALS[pool]:
            raise SplitError(f"{pool}: counts sum to {sum(counts.values())}, "
                             f"locked total is {POOL_TOTALS[pool]}")
        minimum = POOL_PER_FAMILY_MIN.get(pool)
        if minimum is not None:
            fallback = plan.feasibility_fallback.get(pool, {})
            for family in ALFWORLD_TASK_TYPES:
                n = counts.get(family, 0)
                if n < minimum and family not in fallback:
                    raise SplitError(f"{pool}[{family}]: {n} is below the locked per-family "
                                     f"minimum {minimum} and no fallback is pre-registered")


def proportional_counts(availability: Mapping[str, int], total: int,
                        per_family_min: int = 0) -> dict[str, int]:
    """Propose counts proportional to availability with a per-family floor
    (largest-remainder rounding, ties broken by family name).

    A proposal only: the counts EXP-0001 uses are the ones frozen in the plan.
    The rounding rule is not pre-registered anywhere."""
    families = sorted(f for f, a in availability.items() if a > 0)
    if per_family_min * len(families) > total:
        raise SplitError("per-family minimum exceeds the total")
    counts = {f: min(per_family_min, availability[f]) for f in families}
    remaining = total - sum(counts.values())
    spare = {f: availability[f] - counts[f] for f in families}
    if remaining > sum(spare.values()):
        raise SplitError("not enough games for the requested total")
    pool = sum(spare.values())
    shares = {f: Fraction(remaining * spare[f], pool) if pool else Fraction(0) for f in families}
    extra = {f: min(int(shares[f]), spare[f]) for f in families}
    order = sorted(families, key=lambda f: (-(shares[f] - int(shares[f])), f))
    left = remaining - sum(extra.values())
    while left > 0:
        progressed = False
        for f in order:
            if left and extra[f] < spare[f]:
                extra[f] += 1
                left -= 1
                progressed = True
        if not progressed:
            raise SplitError("not enough games for the requested total")
    return {f: counts[f] + extra[f] for f in families}


def _canonical(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def _sha256(obj) -> str:
    return hashlib.sha256(_canonical(obj)).hexdigest()


SELECTION_RULE = ("per (pool, family), in pool_order: the k remaining games with the "
                  "smallest sha256('heritagent-split|<seed>|<pool>|<family>|<game_id>')")


def _draw(available: list[GameInstance], k: int, seed: int, pool: str, family: str) -> list[GameInstance]:
    """Deterministic draw without replacement by keyed-hash rank.

    Unlike ``random.sample``, whose algorithm Python does not promise to keep
    across versions, a SHA-256 rank gives the same selection on every
    interpreter, and any game's rank can be recomputed by hand."""
    if k > len(available):
        raise SplitError(f"{pool}[{family}]: needs {k} games, {len(available)} remain "
                         f"after earlier pools")
    def rank(game: GameInstance) -> str:
        key = f"heritagent-split|{seed}|{pool}|{family}|{game.game_id}"
        return hashlib.sha256(key.encode()).hexdigest()
    return sorted(sorted(available, key=rank)[:k], key=lambda g: g.game_id)


def _locked_floor_through(pool: str) -> int:
    """Sum of the locked per-family minima of every pool drawn up to and
    including ``pool`` (t_mine 8, replay 4, generalization 10). A family with
    fewer accepted games cannot meet this bank's minimum under any plan;
    with at least this many, some plan meets it, so a shortfall at the draw
    would be the plan's own allocation, not availability.

    Deliberately narrow and fail-closed: the rule is per family and ignores
    the fixed pool totals (T_mine 60, ablation 20, interaction 10). On a
    train set too small for every locked count at once (fewer than 174
    accepted games), it can reject a plan whose shortfall is real; that case
    is the owner's decision, not an allocation for the builder to search.
    ALFWorld's train split is far larger (about 3,300 loader-accepted games;
    the smallest family has about 300 before filtering), so the case is not
    expected there; the manifest records the loader's per-family counts."""
    return sum(POOL_PER_FAMILY_MIN.get(p, 0) for p in POOL_ORDER[:POOL_ORDER.index(pool) + 1])


def build_split_manifest(train: CollectionReport, unseen: CollectionReport,
                         plan: SplitPlan) -> dict:
    """Draw every train-side pool from ``train`` and take all of ``unseen``
    as T_eval; return the manifest (JSON-serializable, content-hashed)."""
    validate_plan(plan)
    if train.source != "train" or unseen.source != "valid_unseen":
        raise SplitError("expected the train and valid_unseen collections")
    if len(unseen.accepted) != T_EVAL_SIZE:
        raise SplitError(f"T_eval must be the full official unseen split ({T_EVAL_SIZE} "
                         f"games), the loader accepted {len(unseen.accepted)}")
    remaining = train.by_family()
    accepted = {f: len(g) for f, g in remaining.items()}
    for pool, fallback in plan.feasibility_fallback.items():
        floor = _locked_floor_through(pool)
        for family in fallback:
            if accepted[family] >= floor:
                raise SplitError(
                    f"{pool}[{family}]: a fallback is pre-registered, but the family has "
                    f"{accepted[family]} accepted train games, enough for the locked per-family "
                    f"floors through this bank ({floor}), so availability does not prevent "
                    f"the locked minimum {POOL_PER_FAMILY_MIN[pool]}")
    pools = {}
    for pool in POOL_ORDER:
        games = []
        fallback_evidence = {}
        for family in ALFWORLD_TASK_TYPES:
            k = plan.counts(pool).get(family, 0)
            if family in plan.feasibility_fallback.get(pool, {}):
                fallback_evidence[family] = {
                    "maximum": k, "available_at_draw": len(remaining[family]),
                    "accepted_in_train": accepted[family],
                    "locked_floor_through_bank": _locked_floor_through(pool),
                    "locked_minimum": POOL_PER_FAMILY_MIN[pool]}
            if not k:
                continue
            drawn = _draw(remaining[family], k, plan.seed, pool, family)
            taken = {g.game_id for g in drawn}
            remaining[family] = [g for g in remaining[family] if g.game_id not in taken]
            games.extend(drawn)
        records = [g.record() for g in games]
        pools[pool] = {"source": "train",
                       "family_counts": {f: sum(g.family == f for g in games)
                                         for f in ALFWORLD_TASK_TYPES if any(g.family == f for g in games)},
                       "games": records, "sha256": _sha256(records)}
        if fallback_evidence:
            pools[pool]["feasibility_fallback"] = fallback_evidence
    eval_records = [g.record() for g in unseen.accepted]
    pools["t_eval"] = {"source": "valid_unseen", "sampled": False,
                       "family_counts": {f: len(g) for f, g in unseen.by_family().items() if g},
                       "games": eval_records, "sha256": _sha256(eval_records)}
    seen: dict[str, str] = {}
    for pool in POOL_ORDER:
        for record in pools[pool]["games"]:
            if record["game_id"] in seen:
                raise SplitError(f"{record['game_id']} is in both {seen[record['game_id']]} and {pool}")
            seen[record["game_id"]] = pool
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "experiment": "EXP-0001",
        "benchmark": "ALFWorld",
        "collector": COLLECTOR_REFERENCE,
        "families": list(ALFWORLD_TASK_TYPES),
        "pool_order": list(POOL_ORDER),
        "selection_rule": SELECTION_RULE,
        "plan": plan.record(),
        "collections": {"train": train.summary(), "valid_unseen": unseen.summary()},
        "pools": pools,
        "t_reg": {"status": "pending: selected from t_mine after the G0 lifetime",
                  "size": 40, "source_pool": "t_mine"},
        "contamination_rule": "T_eval never visible to miner, gate, or threshold tuning",
    }
    manifest["manifest_sha256"] = _sha256(manifest)
    return manifest


def verify_manifest(manifest: dict) -> None:
    """Recompute every pool hash and the manifest hash; fail closed on drift."""
    body = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    for name, pool in manifest["pools"].items():
        if _sha256(pool["games"]) != pool["sha256"]:
            raise SplitError(f"pool {name}: games do not match the recorded sha256")
    if _sha256(body) != manifest.get("manifest_sha256"):
        raise SplitError("manifest_sha256 does not match the manifest content")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the EXP-0001 ALFWorld split manifest.")
    parser.add_argument("--data-root", type=Path, required=True,
                        help="directory holding the train/ and valid_unseen/ split directories "
                             "(e.g. $ALFWORLD_DATA/json_2.1.1)")
    parser.add_argument("--plan", type=Path, required=True, help="pre-registered SplitPlan JSON")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    plan = SplitPlan.from_json(json.loads(args.plan.read_text(encoding="utf-8")))
    manifest = build_split_manifest(collect_games(args.data_root / "train", "train"),
                                    collect_games(args.data_root / "valid_unseen", "valid_unseen"),
                                    plan)
    args.out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.out} (manifest_sha256 {manifest['manifest_sha256']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
