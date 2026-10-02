"""Somatic envelopes and two-level CIG record stores.

CIG evidence is two-level (ADR-0001 section 4): one aggregate record per
candidate carrying the verdict, plus immutable per-seed/per-run child
records. ``CigRecordStore`` enforces aggregate<->child consistency:
children must be declared in the aggregate's ``child_records`` (no
undeclared/orphan evidence), child ids must carry the seed they claim, and
``verify()`` flags missing, orphan, and incoherent records. ``SomaticStore``
enforces the candidate -> validated/rejected lifecycle with immutable
terminal decisions (SPEC section 9 fail path keeps rejected traits as
somatic memory).
"""

from __future__ import annotations

import copy
import re

from genome.validation.errors import GenomeValidationError, RecordConsistencyError
from genome.validation.loader import load_cig, load_somatic

_SEED_IN_ID = re.compile(r"/S(\d+)")


class CigRecordStore:
    """In-memory store of aggregate + child CIG records with consistency checks.

    Records are deep-copied on insertion and accessors return deep copies:
    stored evidence is immutable from the caller's side (append-only,
    externally unmutable). White-box corruption of ``_children`` /
    ``_aggregates`` remains possible deliberately for ``verify()``
    defense-in-depth tests.
    """

    def __init__(self):
        self._aggregates: dict[str, dict] = {}
        self._children: dict[str, dict] = {}

    def add_aggregate(self, record: dict) -> None:
        load_cig(record)
        cig_id = record["cig_id"]
        if cig_id in self._aggregates:
            raise RecordConsistencyError(f"aggregate CIG record {cig_id!r} already exists")
        self._aggregates[cig_id] = copy.deepcopy(record)

    def add_child(self, record: dict) -> None:
        load_cig(record)
        cig_id, parent_id = record["cig_id"], record["parent_cig_id"]
        if parent_id not in self._aggregates:
            raise RecordConsistencyError(f"child record {cig_id!r} references unknown aggregate {parent_id!r}")
        if not cig_id.startswith(parent_id + "/S"):
            raise RecordConsistencyError(
                f"child record id {cig_id!r} is not a child of {parent_id!r} (expected prefix {parent_id + '/S'!r})")
        parent = self._aggregates[parent_id]
        if cig_id not in parent["child_records"]:
            raise RecordConsistencyError(
                f"child record {cig_id!r} is not declared in aggregate {parent_id!r} child_records")
        issue = self._seed_issue(record)
        if issue:
            raise RecordConsistencyError(issue)
        if cig_id in self._children:
            raise RecordConsistencyError(f"child CIG record {cig_id!r} already exists (append-only)")
        self._children[cig_id] = copy.deepcopy(record)

    @staticmethod
    def _seed_issue(record: dict) -> str | None:
        """Child id `/S<seed>...` must agree with the record's ``seed`` field."""
        match = _SEED_IN_ID.search(record["cig_id"])
        if match is None:
            return f"child record {record['cig_id']!r} carries no seed digits after /S"
        seed_in_id = int(match.group(1))
        seed = record["seed"]
        if seed != seed_in_id and str(seed) != str(seed_in_id):
            return (f"child record {record['cig_id']!r} encodes seed {seed_in_id} "
                    f"but declares seed {seed!r}")
        return None

    def verify(self) -> list[str]:
        """Return consistency issues; empty list means the store is consistent.

        Detects: declared-but-missing children, children pointing at the
        wrong aggregate, child stages absent from the aggregate, orphan /
        undeclared children, and child-id/seed incoherence.
        """
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
        declared = {ref for agg in self._aggregates.values() for ref in agg["child_records"]}
        for child_id, child in self._children.items():
            if child_id not in declared:
                issues.append(f"orphan child record {child_id!r}: not declared by its aggregate "
                              f"{child['parent_cig_id']!r}")
            issue = self._seed_issue(child)
            if issue:
                issues.append(issue)
        return issues

    def require_consistent(self) -> None:
        issues = self.verify()
        if issues:
            raise RecordConsistencyError(sorted(issues))

    @property
    def aggregates(self) -> dict[str, dict]:
        return copy.deepcopy(self._aggregates)

    @property
    def children(self) -> dict[str, dict]:
        return copy.deepcopy(self._children)


class SomaticStore:
    """Lifecycle store for somatic candidate traits (SPEC sections 5, 9-10).

    Envelopes are deep-copied on insertion and ``all()`` returns deep
    copies: candidate state and terminal decisions are immutable from the
    caller's side.
    """

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
        self._envelopes[gene_id] = copy.deepcopy(envelope)

    def state(self, gene_id: str) -> str | None:
        envelope = self._envelopes.get(gene_id)
        return envelope["validation"]["state"] if envelope else None

    def all(self) -> dict[str, dict]:
        return copy.deepcopy(self._envelopes)
