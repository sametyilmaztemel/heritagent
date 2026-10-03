"""Runtime event log with ordered hooks (issue #7 criterion 7; hardened per critic review).

The execution loop emits structured events instead of persisting anything:
trajectory persistence (#8) subscribes as a hook and reconstructs the full
functional phenotype without rewriting the loop.

Immutability contract: `emit` deep-copies the payload into the stored event,
hands each hook its own deep copy, and all public accessors return deep
copies — neither callers nor hooks can mutate historical evidence. Sequence
numbers are deterministic; every event carries a `monotonic_s` timestamp
from the same injectable monotonic clock that drives latency measurement,
so the whole stream shares one clock domain.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class RuntimeEvent:
    """One observable runtime occurrence. `data` must be treated as
    immutable; EventLog hands out deep copies. `monotonic_s` comes from the
    run's injectable monotonic clock (same domain as latency measurement)."""

    seq: int
    kind: str
    step: int | None
    monotonic_s: float
    data: dict


class EventLog:
    def __init__(self, hooks: tuple[Callable[[RuntimeEvent], None], ...] = (),
                 clock: Callable[[], float] = time.monotonic):
        self._events: list[RuntimeEvent] = []
        self._hooks = tuple(hooks)
        self._clock = clock

    def emit(self, kind: str, *, step: int | None = None, **data) -> RuntimeEvent:
        stored = RuntimeEvent(seq=len(self._events) + 1, kind=kind, step=step,
                              monotonic_s=self._clock(), data=copy.deepcopy(data))
        self._events.append(stored)
        for hook in self._hooks:
            hook(RuntimeEvent(seq=stored.seq, kind=stored.kind, step=stored.step,
                              monotonic_s=stored.monotonic_s, data=copy.deepcopy(stored.data)))
        return RuntimeEvent(seq=stored.seq, kind=stored.kind, step=stored.step,
                            monotonic_s=stored.monotonic_s, data=copy.deepcopy(stored.data))

    @property
    def events(self) -> tuple[RuntimeEvent, ...]:
        return tuple(RuntimeEvent(seq=e.seq, kind=e.kind, step=e.step,
                                  monotonic_s=e.monotonic_s,
                                  data=copy.deepcopy(e.data)) for e in self._events)

    def kinds(self) -> list[str]:
        return [event.kind for event in self._events]

    def of_kind(self, kind: str) -> tuple[RuntimeEvent, ...]:
        return tuple(e for e in self.events if e.kind == kind)

    def __len__(self) -> int:
        return len(self._events)
