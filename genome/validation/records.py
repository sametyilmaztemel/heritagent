"""CIG record store plus the persistent somatic trait store re-export.

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
from genome.validation.loader import load_cig
from somatic.store import SomaticStore as _PersistentSomaticStore

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


class SomaticStore(_PersistentSomaticStore):
    """Compatibility re-export of the persistent somatic store (issue #10).

    ``SomaticStore()`` keeps the historical in-memory usage
    (``path=None``); persistent journals are created/opened through the
    same classmethods as the base implementation (``SomaticStore.create /
    SomaticStore.open``) — logical identity ``(gene_id, version)`` is
    always preserved. The old ``add()``-driven decision API is retired:
    candidates are inserted with :meth:`add_candidate` and terminal
    decisions use :meth:`decide` — a single lifecycle implementation now
    backs both entry points."""

    def __init__(self, registry=None, path=None, mode="create"):
        super().__init__(path, registry=registry, mode=mode)

    @classmethod
    def create(cls, path=None, registry=None) -> "SomaticStore":
        """Create a new persistent journal; returns the COMPAT class instance
        so version-less convenience APIs and the add() alias survive on the
        persistent path."""
        return cls(path=path, registry=registry, mode="create")

    @classmethod
    def open(cls, path, registry=None) -> "SomaticStore":
        """Open + verify an existing journal; returns the COMPAT class
        instance (see create)."""
        return cls(path=path, registry=registry, mode="open")

    def add(self, envelope: dict) -> dict:
        """Compat alias accepting only fresh candidates (transitions go
        through :meth:`decide`)."""
        return self.add_candidate(envelope)

    def get(self, gene_id: str, version: int | None = None):  # noqa: F811
        """Compat: version optional. 0 versions -> None; exactly 1 ->
        convenience lookup; >1 -> typed ambiguity error (logical identity
        ``(gene_id, version)`` never collapses)."""
        if version is not None:
            return super().get(gene_id, version)
        versions = sorted(key[1] for key in self._envelopes if key[0] == gene_id)
        if not versions:
            return None
        if len(versions) > 1:
            raise RecordConsistencyError(
                f"ambiguous gene_id {gene_id!r}: versions {versions} exist; "
                f"address the trait as (gene_id, version)")
        return super().get(gene_id, versions[0])

    def state(self, gene_id: str, version: int | None = None):  # noqa: F811
        """Compat: version optional, same ambiguity contract as get()."""
        if version is not None:
            return super().state(gene_id, version)
        versions = sorted(key[1] for key in self._envelopes if key[0] == gene_id)
        if not versions:
            return None
        if len(versions) > 1:
            raise RecordConsistencyError(
                f"ambiguous gene_id {gene_id!r}: versions {versions} exist; "
                f"address the trait as (gene_id, version)")
        return super().state(gene_id, versions[0])


# re-export the journal pieces so existing ``genome.validation`` users keep
# a stable import surface
__all__ = ["CigRecordStore", "SomaticStore"]
