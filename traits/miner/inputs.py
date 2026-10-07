"""Differential mining input contract (issue #9 criterion 2).

A `MiningPair` binds ONE complete verified successful trajectory and ONE
complete verified failed trajectory of the SAME task, labelled by the caller
with a single task_family. Success/failure comes exclusively from the #8
generic outcome annotation — the miner never invents benchmark success from
runtime `finished`. A `MiningBatch` carries multiple pairs plus the frozen
task-family universe used later to validate applicability.
"""

from __future__ import annotations

from dataclasses import dataclass

from trajectory.normalization import NormalizedTrajectory


class MinerInputError(ValueError):
    """Malformed mining input (pair/batch contract violation)."""


def _require_complete_verified(nt: NormalizedTrajectory, role: str) -> None:
    if nt.status == "incomplete":
        raise MinerInputError(
            f"{role} trajectory {nt.trajectory_id!r} is incomplete; mining requires "
            f"complete verified trajectories")
    if nt.outcome is None or nt.outcome.get("success") is None:
        raise MinerInputError(
            f"{role} trajectory {nt.trajectory_id!r} has no generic outcome "
            f"annotation with a non-null success field")


def _require_matching_context(pair_members: tuple[NormalizedTrajectory, NormalizedTrajectory]) -> None:
    """Same relevant environment/benchmark context: benchmark/split/
    environment must match EXACTLY — one-sided absence cannot be verified
    and is therefore rejected as well."""
    a, b = pair_members
    for field in ("benchmark", "split", "environment"):
        va, vb = a.provenance.get(field), b.provenance.get(field)
        if va != vb:
            raise MinerInputError(
                f"pair members disagree on {field}: {va!r} != {vb!r}")


@dataclass(frozen=True)
class MiningPair:
    """One success/failure differential pair for a single task family."""

    success: NormalizedTrajectory
    failure: NormalizedTrajectory
    task_family: str

    def __post_init__(self):
        if not self.task_family or not isinstance(self.task_family, str):
            raise MinerInputError("task_family must be a non-empty string")
        _require_complete_verified(self.success, "success")
        _require_complete_verified(self.failure, "failure")
        if self.success.trajectory_id == self.failure.trajectory_id:
            raise MinerInputError(
                f"both pair members are trajectory {self.success.trajectory_id!r}; "
                f"pair members must be distinct trajectories")
        if self.success.task != self.failure.task:
            raise MinerInputError(
                f"pair members are different tasks: {self.success.task!r} vs "
                f"{self.failure.task!r}")
        if self.success.outcome.get("success") is not True:
            raise MinerInputError(
                f"success member {self.success.trajectory_id!r} must have "
                f"outcome.success is True (got {self.success.outcome.get('success')!r}); "
                f"runtime status alone is not benchmark success")
        if self.failure.outcome.get("success") is not False:
            raise MinerInputError(
                f"failure member {self.failure.trajectory_id!r} must have "
                f"outcome.success is False (got {self.failure.outcome.get('success')!r})")
        _require_matching_context((self.success, self.failure))

    @property
    def members(self) -> tuple[NormalizedTrajectory, NormalizedTrajectory]:
        return self.success, self.failure


@dataclass(frozen=True)
class MiningBatch:
    """Frozen mining batch: differential pairs across one or more task
    families, plus the frozen family universe for applicability validation.
    Pairing policy is experiment-runner responsibility (#13), not the miner's."""

    pairs: tuple[MiningPair, ...]
    family_universe: tuple[str, ...]

    def __post_init__(self):
        if not self.pairs:
            raise MinerInputError("mining batch requires at least one pair")
        if not self.family_universe:
            raise MinerInputError("family universe must be non-empty")
        if len(set(self.family_universe)) != len(self.family_universe):
            raise MinerInputError("family universe contains duplicates")
        # canonical order: the universe is interpolated into the teacher
        # prompt, so its order is part of the teacher-call identity
        object.__setattr__(self, "family_universe", tuple(sorted(self.family_universe)))
        for pair in self.pairs:
            if pair.task_family not in self.family_universe:
                raise MinerInputError(
                    f"pair task_family {pair.task_family!r} is outside the frozen "
                    f"family universe {sorted(self.family_universe)}")

    @property
    def task_families(self) -> tuple[str, ...]:
        """Deterministic ordered set of families referenced by the pairs."""
        seen: list[str] = []
        for pair in self.pairs:
            if pair.task_family not in seen:
                seen.append(pair.task_family)
        return tuple(seen)
