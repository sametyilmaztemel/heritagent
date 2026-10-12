"""CIG record store plus the persistent somatic trait store re-export.

CIG evidence is two-level (ADR-0001 section 4): one aggregate record per
candidate carrying the verdict, plus immutable per-seed/per-run child
records. ``CigRecordStore`` enforces aggregate<->child consistency:
children must be declared in the aggregate's ``child_records`` (no
undeclared/orphan evidence), child ids must carry the seed they claim, and
``verify()`` flags missing, orphan, and incoherent records.

An M0 gate bundle (``add_bundle``) is strictly one CIG run: exactly the
five M0 stages, exactly one child per stage, exact ``<parent>/S<seed>-<stage>``
child ids, one coherent seed across the bundle, and ``verdict = promote``
iff every stage pass is true. ``verify()`` re-checks the same invariants
for every M0-shaped aggregate (defense in depth against white-box
corruption). ``SomaticStore`` enforces the candidate -> validated/rejected
lifecycle with immutable terminal decisions (SPEC section 9 fail path keeps
rejected traits as somatic memory).
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

    def snapshot(self) -> "CigRecordStore":
        """Deep-copy clone of the CURRENT live store state, for staged
        (prevalidated) commits: ``add_bundle`` on the clone never touches
        the live store, so a successful staged add proves the live add will
        succeed against exactly this state."""
        clone = CigRecordStore()
        clone._aggregates = copy.deepcopy(self._aggregates)
        clone._children = copy.deepcopy(self._children)
        return clone

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
                f"child record {cig_id!r} is not a child of {parent_id!r} (expected prefix {parent_id + '/S'!r})")
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

    M0_STAGE_KEYS = ("replay", "ablation", "generalization_in_scope",
                     "generalization_out_of_scope", "interaction_regression")

    @classmethod
    def _exact_m0_child_id(cls, aggregate_id: str, record: dict) -> str:
        return f"{aggregate_id}/S{record['seed']}-{record['stage']}"

    @classmethod
    def _verdict_issue(cls, aggregate: dict) -> str | None:
        """``verdict`` must be promote iff every declared stage pass is true."""
        all_pass = all(stage["pass"] for stage in aggregate["stages"].values())
        expected = "promote" if all_pass else "reject"
        if aggregate["verdict"] != expected:
            return (f"aggregate {aggregate['cig_id']!r} verdict "
                    f"{aggregate['verdict']!r} != coherent verdict "
                    f"{expected!r} (promote iff every stage pass is true)")
        return None

    @staticmethod
    def _stage_pass_issue(aggregate: dict, child: dict, *,
                          require: bool = False) -> str | None:
        """A child's ``measurements.stage_pass`` must agree with the
        aggregate's declared pass for that stage; for M0 bundle children
        (``require=True``) the field is MANDATORY and its absence is itself
        an issue."""
        measurements = child.get("measurements") or {}
        stage = child["stage"]
        if stage not in aggregate["stages"]:
            return None  # already flagged as absent-stage elsewhere
        if "stage_pass" not in measurements:
            if require:
                return (f"child record {child['cig_id']!r} measurements "
                        f"missing required stage_pass for stage {stage!r}")
            return None
        declared = aggregate["stages"][stage]["pass"]
        if bool(measurements["stage_pass"]) != bool(declared):
            return (f"child record {child['cig_id']!r} stage_pass "
                    f"{measurements['stage_pass']!r} disagrees with aggregate "
                    f"{aggregate['cig_id']!r} stage {stage!r} pass {declared!r}")
        return None

    @staticmethod
    def _candidate_binding_issue(aggregate: dict, child: dict) -> str | None:
        """The child must carry the aggregate's ``candidate_gene_id``
        (borrowed evidence from another candidate is corruption)."""
        m = child.get("measurements") or {}
        if m.get("candidate_gene_id") != aggregate["candidate_gene_id"]:
            return (f"child record {child['cig_id']!r} candidate_gene_id "
                    f"{m.get('candidate_gene_id')!r} != aggregate "
                    f"{aggregate['cig_id']!r} candidate_gene_id "
                    f"{aggregate['candidate_gene_id']!r}")
        return None

    def add_bundle(self, aggregate: dict, children: list[dict]) -> None:
        """ALL-OR-NOTHING atomic bundle commit (issue #12 criterion 12):
        validates deep copies of the aggregate and every child BEFORE
        mutating the live store, then inserts everything. Any validation
        failure leaves the live store byte-identical.

        An M0 bundle is exactly one CIG run: the declared child set must
        equal the provided set (unique ids, none borrowed), the aggregate
        must declare exactly the five M0 stages with EXACTLY ONE child per
        stage under the EXACT id ``<parent>/S<seed>-<stage>``, every child
        must share one coherent seed, child ``candidate_gene_id`` must equal
        the aggregate's, child ``stage_pass`` must agree with the aggregate
        stage pass, and the verdict must be promote iff every stage pass is
        true."""
        staged_agg = copy.deepcopy(aggregate)
        staged_children = copy.deepcopy(children)
        issues: list[str] = []
        load_cig(staged_agg)
        if staged_agg["cig_id"] in self._aggregates:
            issues.append(f"aggregate {staged_agg['cig_id']!r} already exists")
        for child in staged_children:
            load_cig(child)
        child_ids = [c["cig_id"] for c in staged_children]
        declared = list(staged_agg["child_records"])
        if sorted(child_ids) != sorted(declared):
            issues.append(f"child records {sorted(child_ids)} do not match the "
                           f"declared set {sorted(declared)}")
        if len(set(child_ids)) != len(child_ids):
            issues.append("duplicate child ids in bundle")
        m0_stages = set(self.M0_STAGE_KEYS)
        stage_keys = set(staged_agg["stages"])
        if stage_keys != m0_stages:
            issues.append(f"aggregate stages {sorted(stage_keys)} != the five "
                           f"M0 stage keys {sorted(m0_stages)}")
        verdict_issue = self._verdict_issue(staged_agg)
        if verdict_issue:
            issues.append(verdict_issue)
        if issues:
            raise RecordConsistencyError(sorted(issues))
        # per-child checks (staged; store untouched)
        stage_counts: dict[str, int] = {}
        bundle_seeds: set[str] = set()
        for child in staged_children:
            stage_counts[child["stage"]] = stage_counts.get(child["stage"], 0) + 1
            bundle_seeds.add(str(child["seed"]))
            if child["parent_cig_id"] != staged_agg["cig_id"]:
                issues.append(f"child {child['cig_id']!r} parent mismatch")
            if not child["cig_id"].startswith(child["parent_cig_id"] + "/S"):
                issues.append(f"child {child['cig_id']!r} is not a child of "
                               f"{child['parent_cig_id']!r}")
            expected_id = self._exact_m0_child_id(staged_agg["cig_id"], child)
            if child["cig_id"] != expected_id:
                issues.append(f"child {child['cig_id']!r} id != exact M0 form "
                               f"{expected_id!r}")
            seed_issue = self._seed_issue(child)
            if seed_issue:
                issues.append(seed_issue)
            if child["cig_id"] in self._children:
                issues.append(f"child {child['cig_id']!r} already exists "
                               f"(append-only)")
            stage = child["stage"]
            if stage not in staged_agg["stages"]:
                issues.append(f"child {child['cig_id']!r} stage {stage!r} not "
                               f"present in aggregate stages")
                continue
            m = child["measurements"]
            binding_issue = self._candidate_binding_issue(staged_agg, child)
            if binding_issue:
                issues.append(binding_issue)
            pass_issue = self._stage_pass_issue(staged_agg, child, require=True)
            if pass_issue:
                issues.append(pass_issue)
        for stage in self.M0_STAGE_KEYS:
            count = stage_counts.get(stage, 0)
            if count != 1:
                issues.append(f"stage {stage!r} must have exactly one child in "
                               f"the bundle, got {count}")
        if len(bundle_seeds) != 1:
            issues.append(f"bundle children declare incoherent seeds "
                           f"{sorted(bundle_seeds)}; one CIG run = one seed")
        if issues:
            raise RecordConsistencyError(sorted(issues))
        # all-or-nothing commit
        self._aggregates[staged_agg["cig_id"]] = staged_agg
        for child in staged_children:
            self._children[child["cig_id"]] = child

    def verify(self) -> list[str]:
        """Return consistency issues; empty list means the store is consistent.

        Detects: declared-but-missing children, children pointing at the
        wrong aggregate, child stages absent from the aggregate, orphan /
        undeclared children, child-id/seed incoherence, child ``stage_pass``
        disagreeing with the aggregate stage pass, verdicts incoherent with
        the stage passes, and — for M0-shaped aggregates (exactly the five
        M0 stage keys) — duplicate/missing per-stage children, non-exact
        ``<parent>/S<seed>-<stage>`` child ids, and incoherent bundle seeds.
        """
        issues = []
        for cig_id, aggregate in self._aggregates.items():
            m0_shape = set(aggregate["stages"]) == set(self.M0_STAGE_KEYS)
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
                pass_issue = self._stage_pass_issue(aggregate, child,
                                                     require=m0_shape)
                if pass_issue:
                    issues.append(pass_issue)
                if m0_shape:
                    binding_issue = self._candidate_binding_issue(aggregate,
                                                                   child)
                    if binding_issue:
                        issues.append(binding_issue)
            verdict_issue = self._verdict_issue(aggregate)
            if verdict_issue:
                issues.append(verdict_issue)
            if m0_shape:
                issues.extend(self._m0_shape_issues(cig_id, aggregate))
        declared = {ref for agg in self._aggregates.values() for ref in agg["child_records"]}
        for child_id, child in self._children.items():
            if child_id not in declared:
                issues.append(f"orphan child record {child_id!r}: not declared by its aggregate "
                              f"{child['parent_cig_id']!r}")
            issue = self._seed_issue(child)
            if issue:
                issues.append(issue)
        return issues

    def _m0_shape_issues(self, cig_id: str, aggregate: dict) -> list[str]:
        """M0 bundle invariants re-checked over stored state: exactly one
        child per stage, exact child ids, coherent bundle seed."""
        issues: list[str] = []
        stage_refs: dict[str, list[str]] = {}
        seeds: set[str] = set()
        for ref in aggregate["child_records"]:
            child = self._children.get(ref)
            if child is None:
                continue  # missing-child already flagged
            stage_refs.setdefault(child["stage"], []).append(ref)
            seeds.add(str(child["seed"]))
            expected_id = self._exact_m0_child_id(cig_id, child)
            if ref != expected_id:
                issues.append(f"child record {ref!r} id != exact M0 form "
                               f"{expected_id!r} in aggregate {cig_id!r}")
        for stage in self.M0_STAGE_KEYS:
            count = len(stage_refs.get(stage, []))
            if count != 1:
                issues.append(f"aggregate {cig_id!r} stage {stage!r} has {count} "
                               f"children, expected exactly 1")
        if len(seeds) > 1:
            issues.append(f"aggregate {cig_id!r} bundle seeds {sorted(seeds)} "
                           f"are incoherent; one CIG run = one seed")
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
