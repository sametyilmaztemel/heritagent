"""Trajectory-layer errors (fail closed)."""


class RecorderError(Exception):
    """Base class for trajectory recorder failures."""


class StreamInvariantError(RecorderError):
    """The captured event stream violates a structural invariant."""


class MalformedTrajectoryError(RecorderError):
    """Persisted JSONL is malformed/truncated and was loaded without the
    explicit `allow_incomplete=True` recovery path."""


class TrajectoryIntegrityError(RecorderError):
    """Persisted evidence does not match its recorded integrity digest."""
