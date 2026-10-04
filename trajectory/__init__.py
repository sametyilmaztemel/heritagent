"""HeritAgent trajectory layer (issue #8): full functional phenotype capture,
append-only JSONL persistence with integrity digests, and a deterministic
normalizer for #9. Infrastructure only — not a research contribution."""

from trajectory.recorder.context import (
    BudgetSnapshot,
    ModelProvenance,
    OutcomeAnnotation,
    TrajectoryContext,
)
from trajectory.recorder.recorder import FinalizedTrajectory, TrajectoryRecorder
from trajectory.recorder.errors import (
    MalformedTrajectoryError,
    RecorderError,
    TrajectoryIntegrityError,
)
from trajectory.storage.canonical import canonical_json, sha256_hex
from trajectory.verification import (
    validate_event_stream,
    verify_document,
)
from trajectory.normalization import (
    NormalizedTrajectory,
    load_normalized,
    normalize_trajectory,
)
from trajectory.storage.jsonl import load_trajectory

__all__ = [
    "BudgetSnapshot",
    "FinalizedTrajectory",
    "MalformedTrajectoryError",
    "ModelProvenance",
    "NormalizedTrajectory",
    "OutcomeAnnotation",
    "RecorderError",
    "TrajectoryContext",
    "TrajectoryIntegrityError",
    "TrajectoryRecorder",
    "canonical_json",
    "load_normalized",
    "load_trajectory",
    "normalize_trajectory",
    "sha256_hex",
    "validate_event_stream",
    "verify_document",
]
