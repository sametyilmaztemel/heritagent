"""Typed step decisions produced by planner policies."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ToolAction:
    name: str
    arguments: dict


@dataclass(frozen=True)
class StepDecision:
    thought: str
    action: ToolAction | None
    final_answer: str | None


__all__ = ["StepDecision", "ToolAction"]
