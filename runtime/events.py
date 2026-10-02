"""Runtime event log with ordered hooks (issue #7 criterion 7).

The execution loop emits structured events instead of persisting anything:
trajectory persistence (#8) subscribes as a hook and records the full
functional phenotype without rewriting the loop. Events are appended and
then handed to hooks in registration order, so the log and every hook see
the same total order.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


@dataclass(frozen=True)
class RuntimeEvent:
    """One observable runtime occurrence (e.g. step_started, tool_result)."""

    seq: int
    kind: str
    step: int | None
    data: dict


class EventLog:
    def __init__(self, hooks: tuple[Callable[[RuntimeEvent], None], ...] = ()):
        self.events: list[RuntimeEvent] = []
        self._hooks = tuple(hooks)

    def emit(self, kind: str, *, step: int | None = None, **data) -> RuntimeEvent:
        event = RuntimeEvent(seq=len(self.events) + 1, kind=kind, step=step, data=dict(data))
        self.events.append(event)
        for hook in self._hooks:
            hook(event)
        return event

    def kinds(self) -> list[str]:
        return [event.kind for event in self.events]

    def of_kind(self, kind: str) -> list[RuntimeEvent]:
        return [event for event in self.events if event.kind == kind]
