"""Trajectory recorder subpackage (context, hook-based capture, finalization)."""

from trajectory.recorder.context import (
    BudgetSnapshot,
    ModelProvenance,
    OutcomeAnnotation,
    TrajectoryContext,
)
from trajectory.recorder.errors import (
    MalformedTrajectoryError,
    RecorderError,
    StreamInvariantError,
    TrajectoryIntegrityError,
)
from trajectory.recorder.recorder import FinalizedTrajectory, TrajectoryRecorder

__all__ = [
    "BudgetSnapshot",
    "FinalizedTrajectory",
    "MalformedTrajectoryError",
    "ModelProvenance",
    "OutcomeAnnotation",
    "RecorderError",
    "StreamInvariantError",
    "TrajectoryContext",
    "TrajectoryIntegrityError",
    "TrajectoryRecorder",
]
