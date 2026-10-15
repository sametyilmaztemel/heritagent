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
iff every stage pass is true. Each M0 bundle also names the artifact and
the base genome it measured, and every child and pair evidence must agree
with them (ADR-0010 B4). ``verify()`` re-checks the same invariants for
every M0-shaped aggregate (defense in depth against white-box corruption),
and links the two copies of each pair evidence across bundles. ``SomaticStore`` enforces the candidate -> validated/rejected
lifecycle with immutable terminal decisions (SPEC section 9 fail path keeps
rejected traits as somatic memory).
"""

from __future__ import annotations

import copy
import re

from genome.validation.errors import GenomeValidationError, RecordConsistencyError
from genome.validation.loader import load_cig
from genome.validation.registry import parse_uri
from somatic.store import SomaticStore as _PersistentSomaticStore
from trajectory.storage.canonical import canonical_json, sha256_hex

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
        aggregate's declared pass for that stage via DIRECT boolean
        equality (no ``bool()`` coercion); for M0 bundle children
        (``require=True``) the field is MANDATORY and must be a STRICT
        boolean (``1``, ``"true"``, ``{}``, ... are rejected)."""
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
        value = measurements["stage_pass"]
        if require and not isinstance(value, bool):
            return (f"child record {child['cig_id']!r} stage_pass "
                    f"{value!r} must be a strict boolean for M0 stage "
                    f"{stage!r}")
        if value != declared:
            return (f"child record {child['cig_id']!r} stage_pass "
                    f"{value!r} disagrees with aggregate "
                    f"{aggregate['cig_id']!r} stage {stage!r} pass {declared!r}")
        return None

    # the child fields that name what a stage measured (ADR-0010 B1)
    BINDING_KEYS = ("candidate_version", "candidate_artifact", "base_genome_id")

    @staticmethod
    def _reference_issue(ref, where: str) -> str | None:
        """A ``{gene_id, version, artifact}`` reference is coherent when its
        version is a strict int and its registry URI parses and names that
        gene id and version (ADR-0010 B4; the digest is not recomputed)."""
        if not isinstance(ref, dict):
            return f"{where} {ref!r} is not a {{gene_id, version, artifact}} reference"
        gene_id, version, uri = ref.get("gene_id"), ref.get("version"), ref.get("artifact")
        if type(version) is not int:
            return f"{where} version {version!r} is not a strict int"
        try:
            parsed = parse_uri(uri) if isinstance(uri, str) else None
        except GenomeValidationError:
            parsed = None
        if parsed is None:
            return f"{where} artifact {uri!r} is not a registry URI"
        if parsed["name"] != gene_id or int(parsed["version"]) != version:
            return (f"{where} artifact {uri!r} does not name gene {gene_id!r} "
                    f"version {version!r}")
        return None

    @staticmethod
    def _references(aggregate: dict) -> tuple:
        """The aggregate's ``artifacts.candidate`` and base genome id (either
        may be missing or malformed; ``_m0_reference_issues`` says so)."""
        artifacts = aggregate.get("artifacts")
        artifacts = artifacts if isinstance(artifacts, dict) else {}
        base = artifacts.get("base_genome")
        return (artifacts.get("candidate"),
                base.get("genome_id") if isinstance(base, dict) else None)

    @classmethod
    def _m0_reference_issues(cls, aggregate: dict) -> list[str]:
        """An M0 aggregate names a complete, coherent ``artifacts.candidate``
        for its ``candidate_gene_id`` and, in ``artifacts.base_genome``, its
        own ``genome_id`` (ADR-0010 B4)."""
        where = f"aggregate {aggregate['cig_id']!r}"
        candidate, genome_id = cls._references(aggregate)
        issues = []
        issue = cls._reference_issue(candidate, f"{where} artifacts.candidate")
        if issue:
            issues.append(issue)
        elif candidate["gene_id"] != aggregate["candidate_gene_id"]:
            issues.append(f"{where} artifacts.candidate gene_id {candidate['gene_id']!r} != "
                          f"its candidate_gene_id {aggregate['candidate_gene_id']!r}")
        if genome_id != aggregate["genome_id"]:
            issues.append(f"{where} artifacts.base_genome.genome_id {genome_id!r} != its "
                          f"genome_id {aggregate['genome_id']!r}")
        return issues

    @classmethod
    def _candidate_binding_issues(cls, aggregate: dict, child: dict) -> list[str]:
        """The child must carry the aggregate's ``candidate_gene_id``
        (borrowed evidence from another candidate is corruption), and the
        version, artifact and base genome the aggregate names, compared
        with their types (ADR-0010 B4: a missing field is a mismatch)."""
        m = child.get("measurements") or {}
        if m.get("candidate_gene_id") != aggregate["candidate_gene_id"]:
            return [f"child record {child['cig_id']!r} candidate_gene_id "
                    f"{m.get('candidate_gene_id')!r} != aggregate "
                    f"{aggregate['cig_id']!r} candidate_gene_id "
                    f"{aggregate['candidate_gene_id']!r}"]
        candidate, genome_id = cls._references(aggregate)
        candidate = candidate if isinstance(candidate, dict) else {}
        expected = dict(zip(cls.BINDING_KEYS, (candidate.get("version"),
                                                candidate.get("artifact"), genome_id)))
        issues = []
        for key, value in expected.items():
            got = m.get(key)
            if type(got) is not type(value) or got != value:
                issues.append(f"child record {child['cig_id']!r} {key} {got!r} != aggregate "
                              f"{aggregate['cig_id']!r} {value!r} (ADR-0010)")
        issues.extend(cls._pair_evidence_issues(aggregate, child))
        return issues

    @staticmethod
    def _pair_entries(child: dict):
        """The pair evidence list of an interaction_regression child, or a
        string naming why it is malformed (absent counts as none)."""
        if child["stage"] != "interaction_regression":
            return []
        interaction = (child.get("measurements") or {}).get("interaction", {})
        if not isinstance(interaction, dict):
            return f"interaction {interaction!r} is not an object"
        entries = interaction.get("pair_evidence", [])
        if not isinstance(entries, list):
            return f"pair_evidence {entries!r} is not a list"
        return entries

    @staticmethod
    def _pair_genes(entry) -> list | None:
        """The two distinct gene ids a pair evidence entry names, or None."""
        canonical = entry.get("canonical_evidence") if isinstance(entry, dict) else None
        gene_ids = canonical.get("gene_ids") if isinstance(canonical, dict) else None
        if not isinstance(gene_ids, list) or len(gene_ids) != 2 or \
                not all(isinstance(g, str) for g in gene_ids) or gene_ids[0] == gene_ids[1]:
            return None
        return gene_ids

    @classmethod
    def _pair_evidence_issues(cls, aggregate: dict, child: dict) -> list[str]:
        """Each pair evidence of an interaction_regression child matches its
        own digest, names the aggregate's candidate as one of its two genes
        and the other as ``other_gene_id``, and carries one coherent
        ``{gene_id, version, artifact}`` per gene in ``gene_ids`` order, the
        candidate's equal to ``artifacts.candidate`` (ADR-0010 B2, B4)."""
        where = f"child record {child['cig_id']!r}"
        entries = cls._pair_entries(child)
        if isinstance(entries, str):
            return [f"{where} {entries}"]
        candidate, _ = cls._references(aggregate)
        gene = aggregate["candidate_gene_id"]
        issues = []
        for index, entry in enumerate(entries):
            at = f"{where} pair evidence {index}"
            canonical = entry.get("canonical_evidence") if isinstance(entry, dict) else None
            if not isinstance(canonical, dict):
                issues.append(f"{at} carries no canonical_evidence object")
                continue
            try:
                digest = sha256_hex(canonical_json(canonical).encode("utf-8"))
            except (TypeError, ValueError) as exc:
                issues.append(f"{at} canonical_evidence is not canonical JSON ({exc})")
                continue
            if entry.get("evidence_sha256") != digest:
                issues.append(f"{at} evidence_sha256 {entry.get('evidence_sha256')!r} != the "
                              f"digest of its canonical_evidence {digest!r}")
            gene_ids = cls._pair_genes(entry)
            if gene_ids is None or gene not in gene_ids:
                issues.append(f"{at} gene_ids {canonical.get('gene_ids')!r} are not two "
                              f"distinct genes including {gene!r}")
                continue
            partner = gene_ids[1] if gene_ids[0] == gene else gene_ids[0]
            if entry.get("other_gene_id") != partner:
                issues.append(f"{at} other_gene_id {entry.get('other_gene_id')!r} != the "
                              f"partner {partner!r} its gene_ids name")
            artifacts = canonical.get("artifacts")
            if not isinstance(artifacts, list) or len(artifacts) != len(gene_ids):
                issues.append(f"{at} artifacts {artifacts!r} are not one reference per gene")
                continue
            for gene_id, ref in zip(gene_ids, artifacts):
                issue = cls._reference_issue(ref, f"{at} artifact for {gene_id!r}")
                if issue:
                    issues.append(issue)
                elif ref["gene_id"] != gene_id:
                    issues.append(f"{at} artifacts name {ref['gene_id']!r} where gene_ids "
                                  f"order names {gene_id!r}")
                elif gene_id == gene and ref != candidate:
                    issues.append(f"{at} entry for the candidate {ref!r} != aggregate "
                                  f"{aggregate['cig_id']!r} artifacts.candidate {candidate!r}")
        return issues

    def _partner_link_issues(self) -> list[str]:
        """Both copies of a pair evidence exist: every well-formed pair
        evidence in an M0 interaction_regression child of candidate g that
        names partner p is carried, with the same ``evidence_sha256``, by an
        interaction_regression child of an M0 aggregate for p (ADR-0010 B4;
        malformed entries are flagged per child)."""
        carried: dict[str, set] = {}
        named: list[tuple[str, str, str, object]] = []
        for cig_id, aggregate in self._aggregates.items():
            if set(aggregate["stages"]) != set(self.M0_STAGE_KEYS):
                continue
            gene = aggregate["candidate_gene_id"]
            for ref in aggregate["child_records"]:
                child = self._children.get(ref)
                if child is None or child["parent_cig_id"] != cig_id:
                    continue
                entries = self._pair_entries(child)
                for entry in entries if isinstance(entries, list) else []:
                    gene_ids = self._pair_genes(entry)
                    if gene_ids is None or gene not in gene_ids:
                        continue
                    digest = entry.get("evidence_sha256")
                    carried.setdefault(gene, set()).add(digest)
                    partner = gene_ids[1] if gene_ids[0] == gene else gene_ids[0]
                    named.append((ref, gene, partner, digest))
        return [f"pair evidence {digest!r} in child record {ref!r} of {gene!r} names partner "
                f"{partner!r}, but no M0 interaction_regression child of {partner!r} carries "
                f"it (ADR-0010)"
                for ref, gene, partner, digest in named
                if digest not in carried.get(partner, set())]

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
        true. The aggregate must name the evaluated artifact and base genome,
        and every child and pair evidence must agree with them (ADR-0010 B4);
        the link between the two copies of a pair evidence spans bundles, so
        only ``verify()`` checks it."""
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
        issues.extend(self._m0_reference_issues(staged_agg))
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
            issues.extend(self._candidate_binding_issues(staged_agg, child))
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
        ``<parent>/S<seed>-<stage>`` child ids, incoherent bundle seeds, and
        the ADR-0010 references, child bindings, pair evidence and the
        cross-bundle link between the two copies of each pair evidence.
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
                    issues.extend(self._candidate_binding_issues(aggregate, child))
            verdict_issue = self._verdict_issue(aggregate)
            if verdict_issue:
                issues.append(verdict_issue)
            if m0_shape:
                issues.extend(self._m0_shape_issues(cig_id, aggregate))
                issues.extend(self._m0_reference_issues(aggregate))
        issues.extend(self._partner_link_issues())
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
