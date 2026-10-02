"""Somatic envelopes and two-level CIG record stores.

CIG evidence is two-level (ADR-0001 section 4): one aggregate record per
candidate carrying the verdict, plus immutable per-seed/per-run child
records. ``CigRecordStore`` enforces aggregate<->child consistency.
``SomaticStore`` enforces the candidate -> validated/rejected lifecycle with
immutable terminal decisions (SPEC section 9 fail path keeps rejected traits
as somatic memory).
"""

from __future__ import annotations

from genome.validation.errors import GenomeValidationError, RecordConsistencyError
from genome.validation.loader import load_cig, load_somatic


class CigRecordStore:
    """In-memory store of aggregate + child CIG records with consistency checks."""

    def __init__(self):
        self._aggregates: dict[str, dict] = {}
        self._children: dict[str, dict] = {}

    def add_aggregate(self, record: dict) -> None:
        load_cig(record)
        cig_id = record["cig_id"]
        if cig_id in self._aggregates:
            raise RecordConsistencyError(f"aggregate CIG record {cig_id!r} already exists")
        self._aggregates[cig_id] = record

    def add_child(self, record: dict) -> None:
        load_cig(record)
        cig_id, parent_id = record["cig_id"], record["parent_cig_id"]
        if parent_id not in self._aggregates:
            raise RecordConsistencyError(f"child record {cig_id!r} references unknown aggregate {parent_id!r}")
        if not cig_id.startswith(parent_id + "/S"):
            raise RecordConsistencyError(
                f"child record id {cig_id!r} is not a child of {parent_id!r} (expected prefix {parent_id + '/S'!r})")
        if cig_id in self._children:
            raise RecordConsistencyError(f"child CIG record {cig_id!r} already exists (append-only)")
        self._children[cig_id] = record

    def verify(self) -> list[str]:
        """Return consistency issues; empty list means the store is consistent."""
        issues = []
        for cig_id, aggregate in self._aggregates.items():
            for ref in aggregate["child_records"]:
                child = self._children.get(ref)
                if child is None:
                    issues.append(f"aggregate {cig_id!r} references missing child record {ref!r}")
                    continue
                if child["parent_cig_id"] != cig_id:
                    issues.append(f"child record {ref!r} points at {child['parent_cig_id']!r}, expected {cig_id!r}")
                if child["stage"] not in aggregate["stages"]:
                    issues.append(f"child record {ref!r} has stage {child['stage']!r} "
                                  f"not present in aggregate {cig_id!r} stages")
        return issues

    def require_consistent(self) -> None:
        issues = self.verify()
        if issues:
            raise RecordConsistencyError(sorted(issues))

    @property
    def aggregates(self) -> dict[str, dict]:
        return dict(self._aggregates)

    @property
    def children(self) -> dict[str, dict]:
        return dict(self._children)


class SomaticStore:
    """Lifecycle store for somatic candidate traits (SPEC sections 5, 9-10)."""

    TERMINAL_STATES = ("validated", "rejected")

    def __init__(self, registry=None):
        self._envelopes: dict[str, dict] = {}
        self._registry = registry

    def add(self, envelope: dict) -> None:
        load_somatic(envelope, self._registry)
        gene_id = envelope["candidate"]["gene_id"]
        state = envelope["validation"]["state"]
        existing = self._envelopes.get(gene_id)
        if existing is not None:
            if existing["validation"]["state"] in self.TERMINAL_STATES:
                raise RecordConsistencyError(
                    f"somatic candidate {gene_id!r} already reached terminal state "
                    f"{existing['validation']['state']!r}; decisions are immutable")
        self._envelopes[gene_id] = envelope

    def state(self, gene_id: str) -> str | None:
        envelope = self._envelopes.get(gene_id)
        return envelope["validation"]["state"] if envelope else None

    def all(self) -> dict[str, dict]:
        return dict(self._envelopes)
