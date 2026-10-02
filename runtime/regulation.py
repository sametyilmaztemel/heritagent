"""v0 regulation evaluator (issue #7 criterion 5).

Pure, deterministic evaluation of EAG `express_when` conditions over
agent-observable runtime counters. No environment or task-family labels
exist here by design (ADR-0001 D5). A gene with no regulation rule is
constitutively expressed; a missing counter is treated as an unsatisfied
predicate (deterministic, fail-closed) rather than an error.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from genome.validation.errors import GenomeValidationError

_ALLOWED_METRICS = frozenset({
    "consecutive_failures", "failing_tools", "repeated_tool_error", "steps_without_progress",
})
_OPS = {
    ">=": lambda a, b: a >= b,
    ">": lambda a, b: a > b,
    "==": lambda a, b: a == b,
    "<": lambda a, b: a < b,
    "<=": lambda a, b: a <= b,
}


@dataclass(frozen=True)
class Predicate:
    metric: str
    op: str
    value: float

    def evaluate(self, counters: dict) -> bool:
        if self.metric not in _ALLOWED_METRICS:
            raise GenomeValidationError(f"regulation predicate uses non-observable metric {self.metric!r}")
        if self.op not in _OPS:
            raise GenomeValidationError(f"regulation predicate uses unknown operator {self.op!r}")
        current = counters.get(self.metric)
        if current is None:
            return False  # missing counter = predicate unsatisfied (deterministic)
        return _OPS[self.op](float(current), float(self.value))


@dataclass(frozen=True)
class Condition:
    """`express_when` rule: `all_of` must all hold and, if `any_of` is
    non-empty, at least one of `any_of` must hold. Empty condition = always
    true (constitutive)."""

    all_of: tuple[Predicate, ...] = ()
    any_of: tuple[Predicate, ...] = ()

    @staticmethod
    def from_genome(rule: dict) -> "Condition":
        express = (rule or {}).get("express_when") or {}
        try:
            all_of = tuple(Predicate(**p) for p in express.get("all_of") or ())
            any_of = tuple(Predicate(**p) for p in express.get("any_of") or ())
        except TypeError as exc:
            raise GenomeValidationError(f"malformed regulation predicate: {exc}") from None
        return Condition(all_of=all_of, any_of=any_of)

    def evaluate(self, counters: dict) -> bool:
        if not all(p.evaluate(counters) for p in self.all_of):
            return False
        if self.any_of and not any(p.evaluate(counters) for p in self.any_of):
            return False
        return True

    def is_constitutive(self) -> bool:
        return not self.all_of and not self.any_of


def constitutive(gene_id: str, regulation: dict) -> bool:
    """True when `gene_id` has no regulation rule (always expressed)."""
    return gene_id not in regulation


__all__ = ["Condition", "Predicate", "constitutive"]
