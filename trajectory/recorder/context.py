"""Explicit reproducibility context (issue #8 criterion 2).

Everything that #13 will later require under the SPEC §24 policy (code
commit, model revision, genome id, split, seed) is an EXPLICIT input here —
the recorder never reads git state, environment variables, or ambient
configuration on its own. Arbitrary tags/metadata must be JSON-safe and are
validated fail-closed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from runtime.model_adapters import ModelMetadata


def _require_json_safe(name: str, value) -> None:
    try:
        round_trip = json.loads(json.dumps(value))
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be JSON-safe: {exc}") from None
    if round_trip != value:
        raise TypeError(f"{name} must round-trip through JSON unchanged")


@dataclass(frozen=True)
class ModelProvenance:
    """Model identity metadata supplied by the caller (typically from
    ModelAdapter.metadata()); None fields are allowed at #8 and enforced by
    #13's reporting policy."""

    model_id: str | None = None
    revision: str | None = None
    backend: str | None = None
    backend_version: str | None = None

    @staticmethod
    def from_metadata(metadata: ModelMetadata) -> "ModelProvenance":
        return ModelProvenance(model_id=metadata.model_id, revision=metadata.revision,
                               backend=metadata.backend,
                               backend_version=metadata.backend_version)


@dataclass(frozen=True)
class BudgetSnapshot:
    max_steps: int | None = None
    max_total_retries: int | None = None
    max_tokens_per_request: int | None = None
    max_total_tokens: int | None = None


@dataclass(frozen=True)
class TrajectoryContext:
    """Run identity / reproducibility boundary for one trajectory."""

    genome_id: str
    run_id: str | None = None
    experiment_id: str | None = None          # optional at #8; required later (#13)
    model: ModelProvenance | None = None
    seed: int | None = None
    code_commit: str | None = None            # optional at #8; required later (#13)
    environment: str | None = None            # generic environment/benchmark identifiers
    benchmark: str | None = None
    split: str | None = None
    budget: BudgetSnapshot | None = None
    tags: dict = field(default_factory=dict)      # arbitrary JSON-safe tags
    metadata: dict = field(default_factory=dict)  # arbitrary JSON-safe extensions

    def __post_init__(self):
        if not self.genome_id:
            raise ValueError("TrajectoryContext requires a genome_id")
        for name in ("tags", "metadata"):
            value = getattr(self, name)
            if not isinstance(value, dict):
                raise TypeError(f"{name} must be a dict")
            _require_json_safe(name, value)

    def to_dict(self) -> dict:
        return {
            "genome_id": self.genome_id,
            "run_id": self.run_id,
            "experiment_id": self.experiment_id,
            "model": vars(self.model).copy() if self.model else None,
            "seed": self.seed,
            "code_commit": self.code_commit,
            "environment": self.environment,
            "benchmark": self.benchmark,
            "split": self.split,
            "budget": vars(self.budget).copy() if self.budget else None,
            "tags": json.loads(json.dumps(self.tags)),
            "metadata": json.loads(json.dumps(self.metadata)),
        }


@dataclass(frozen=True)
class OutcomeAnnotation:
    """Generic, benchmark-agnostic post-run outcome (issue #8 criterion 9).
    Runtime status (finished / budget_exhausted) is NOT benchmark success;
    evaluators attach this annotation at finalization without mutating any
    historical event."""

    success: bool | None = None
    reward: float | None = None
    evaluator_id: str | None = None
    evaluator_version: str | None = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        _require_json_safe("outcome.metadata", self.metadata)

    def to_dict(self) -> dict:
        return {"success": self.success, "reward": self.reward,
                "evaluator_id": self.evaluator_id,
                "evaluator_version": self.evaluator_version,
                "metadata": json.loads(json.dumps(self.metadata))}
