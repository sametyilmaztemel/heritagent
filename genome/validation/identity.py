"""Canonical genome identity (ADR-0001 Amendment A2; issue #22).

``genome_id`` is the identity of the HERITABLE FUNCTIONAL CONTENT of a
genome — not a run/lineage bookkeeping id. This module is the single
authoritative implementation of the contract:

    genome_id = "G-" + sha256("heritagent:eag-genome-id:v1\\n"
                              + canonical_identity_json)[:16]   (lowercase hex)

Identity domain (what IS hashed):
- ``schema_version`` (the manifest-format version that fixes functional
  interpretation);
- the policy slot structure (cognition/memory/execution) with each
  policyRef's functional identity: ``gene_id``, ``version``, ``type``,
  ``artifact`` (the content-addressed URI — artifact content is pinned by
  its SHA-256 digest);
- ``skills[]`` IN ORDER — skill order is runtime-functional (prompt
  rendering / block order), so reordering is a heritable change;
- ``regulation`` — every gene's ``express_when`` condition; the regulation
  map is unordered (canonical JSON sorts keys) and ``all_of``/``any_of``
  predicate arrays are conjunctions/disjunctions (order-insensitive), so
  they are canonically sorted;
- normalization: an absent policy section equals an empty one, and an
  absent regulation equals ``{}`` — both are functionally identical.

Excluded from identity (lineage/audit bookkeeping, never hashed):
- ``genome_id`` itself (the claimed value is never an input);
- ``parent``, ``generation``, ``lineage_id``;
- per-gene ``origin`` and ``provenance`` (``born_generation``,
  ``source_trajectory``, ``cig_record``, ``notes``) — history/audit, not
  function.

Consequences (normative in the ADR): identical heritable functional
content under different lineage metadata yields the SAME genome_id; any
heritable functional change (artifact URI/version, add/remove skill,
policy ref change, regulation change, expression-relevant skill order)
yields a NEW genome_id.

Hash input is domain-separated so genome identity bytes can never collide
with another artifact type's hash namespace. The 16-hex prefix (64 bits)
is a convenience content identifier; the FULL digest is reproducible via
:func:`genome_identity_hash_hex`. ``validate_genome_id()`` guarantees ONLY
per-document integrity (claimed prefix == this genome's own full-digest
prefix); M0 keeps NO global collision registry, so a genuine prefix
collision (two distinct full digests sharing the first 16 hex chars) is
NOT detectable by single-document validation. A future lineage/genome
store that observes different full digests under one ``genome_id`` must
fail closed; resolving such a collision requires a longer-prefix /
new-ADR migration.

All serialization reuses ``trajectory.storage.canonical`` (sorted keys,
compact separators, UTF-8, non-finite floats and non-JSON values rejected)
— no duplicate canonical-JSON implementation.
"""

from __future__ import annotations

import copy
import re

from genome.validation.errors import GenomeValidationError
from trajectory.storage.canonical import canonical_json, sha256_hex

GENOME_ID_DOMAIN = "heritagent:eag-genome-id:v1"
GENOME_ID_HEX_CHARS = 16
GENOME_ID_PATTERN = re.compile(r"^G-[0-9a-f]{16}$")
# the only EAG manifest version this identity contract can hash; anything
# else fails closed (a future version requires its own contract/ADR)
_SUPPORTED_SCHEMA_VERSION = "0.1"

# the functional identity of one gene reference (provenance/origin are
# audit metadata and are deliberately NOT part of the identity domain)
_FUNCTIONAL_REF_KEYS = ("gene_id", "version", "type", "artifact")
_POLICY_SECTIONS = ("cognition", "memory", "execution")


def _plain(obj):
    """Convert frozen views (mappingproxy/tuple) into plain JSON structures
    so canonicalization sees ordinary JSON values."""
    import types
    if isinstance(obj, (dict, types.MappingProxyType)):
        return {k: _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    return obj


def _functional_gene_ref(ref, path: str) -> dict:
    if not isinstance(ref, dict):
        raise GenomeValidationError(
            [f"identity payload {path}: gene reference must be an object, "
             f"got {type(ref).__name__}"])
    missing = [key for key in _FUNCTIONAL_REF_KEYS if key not in ref]
    if missing:
        raise GenomeValidationError(
            [f"identity payload {path}: gene reference is missing the "
             f"functional identity keys {sorted(missing)}"])
    # Self-contained artifact-ref integrity (fail closed, A2): grammar +
    # name/version/kind coherence are checked with the AUTHORITATIVE
    # registry helpers — no URI parsing duplicated here. Registry
    # RESOLUTION is deliberately out of scope for identity canonicalization
    # (no bytes lookup, no digest existence check, no I/O); ref-level
    # integrity must already hold when an id is minted, even on
    # construction paths that run before load_genome().
    from genome.validation.registry import KIND_BY_TYPE, parse_uri
    try:
        parts = parse_uri(ref["artifact"])
    except GenomeValidationError as exc:
        raise GenomeValidationError(
            [f"identity payload {path}: {e}" for e in exc.errors]) from None
    errors: list[str] = []
    if parts["name"] != ref["gene_id"]:
        errors.append(f"artifact URI name {parts['name']!r} != gene_id "
                       f"{ref['gene_id']!r}")
    if int(parts["version"]) != ref["version"]:
        errors.append(f"artifact URI version {parts['version']} != gene "
                       f"version {ref['version']}")
    expected_kind = KIND_BY_TYPE.get(ref["type"])
    if parts["kind"] != expected_kind:
        errors.append(f"artifact URI kind {parts['kind']!r} does not match "
                       f"gene type {ref['type']!r} (expected "
                       f"{expected_kind!r})")
    if errors:
        raise GenomeValidationError(
            [f"identity payload {path}: {e}" for e in errors])
    return {key: _plain(ref[key]) for key in _FUNCTIONAL_REF_KEYS}


def _canonical_condition(condition, path: str) -> dict:
    """Canonical form of one regulation condition.

    ``all_of``/``any_of`` are order-insensitive conjunctions/disjunctions,
    so their predicate arrays are sorted by canonical JSON (normative ADR
    decision). A rule carrying NO effective predicate — no or empty
    ``all_of`` AND no or empty ``any_of`` — is runtime-equivalent to
    constitutive expression, so the whole entry is DROPPED from the
    identity payload (absent rule == ``{}`` == ``{"express_when": {}}``).

    Container types are STRICT when a field is PRESENT: only ABSENT fields
    are normalized (malformed containers fail closed).
    """
    if not isinstance(condition, dict):
        raise GenomeValidationError(
            [f"identity payload {path}: regulation condition must be an "
             f"object when present, got {type(condition).__name__}"])
    if "express_when" not in condition:
        return {}  # no rule at all -> constitutive -> dropped
    express = condition["express_when"]
    if not isinstance(express, dict):
        raise GenomeValidationError(
            [f"identity payload {path}: express_when must be an object "
             f"when present, got {type(express).__name__}"])
    effective: dict = {}
    for key in ("all_of", "any_of"):
        if key not in express:
            continue
        predicates = express[key]
        if not isinstance(predicates, list):
            raise GenomeValidationError(
                [f"identity payload {path}: express_when.{key} must be an "
                 f"array when present, got {type(predicates).__name__}"])
        if predicates:
            effective[key] = sorted(
                predicates,
                key=lambda predicate: canonical_json(_plain(predicate)))
    if not effective:
        return {}  # no effective predicate -> constitutive -> dropped
    return {"express_when": effective}


def canonical_genome_identity_payload(genome: dict) -> dict:
    """Deterministic functional-identity payload of a genome (ADR-0001
    Amendment A2).

    The input is deep-copied first: caller mutation of the genome (during
    or after the call) cannot influence the payload. Excluded fields
    (``genome_id``/``parent``/``generation``/``lineage_id``/gene
    provenance) never enter the payload — in particular the claimed
    ``genome_id`` is not an input to its own computation. Map key order is
    normalized by canonical JSON; ``skills[]`` order is PRESERVED
    (runtime-functional); ``all_of``/``any_of`` are canonically sorted
    (order-insensitive). Non-JSON values and NaN/Infinity fail closed.
    """
    source = _plain(copy.deepcopy(genome))
    errors: list[str] = []
    if not isinstance(source, dict):
        raise GenomeValidationError(
            ["identity payload: genome must be an object"])
    if "schema_version" not in source:
        errors.append("identity payload: genome is missing 'schema_version'")
    elif source["schema_version"] != _SUPPORTED_SCHEMA_VERSION:
        # fail closed on inconsistent schema version: this implementation
        # hashes EAG '0.1' genomes under the 'heritagent:eag-genome-id:v1'
        # contract only; a future EAG version needs a new contract (and,
        # if functional interpretation changes, a new domain separator +
        # ADR) — it must never silently mint ids under this one
        errors.append(
            f"identity payload: unsupported schema_version "
            f"{source['schema_version']!r}; this identity contract hashes "
            f"EAG {_SUPPORTED_SCHEMA_VERSION!r} genomes only")
    if "genes" not in source:
        errors.append("identity payload: genome is missing 'genes'")
    if errors:
        raise GenomeValidationError(sorted(errors))
    # STRICT container types for PRESENT fields — only ABSENT fields are
    # normalized (A2: absent policy section == empty; absent skills ==
    # empty array; absent regulation == {}). A falsy malformed container
    # ([] where a dict is required, {} where an array is required, ...)
    # must NEVER be silently normalized into empty valid content.
    genes_in = source["genes"]
    if not isinstance(genes_in, dict):
        raise GenomeValidationError(
            [f"identity payload: 'genes' must be an object when present, "
             f"got {type(genes_in).__name__}"])
    genes_out: dict = {}
    for section in _POLICY_SECTIONS:
        if section not in genes_in:
            continue  # ABSENT policy section == empty (A2 normalization)
        slots = genes_in[section]
        if not isinstance(slots, dict):
            raise GenomeValidationError(
                [f"identity payload: 'genes.{section}' must be an object "
                 f"when present, got {type(slots).__name__}"])
        if slots:
            genes_out[section] = {
                slot: _functional_gene_ref(ref, f"genes.{section}.{slot}")
                for slot, ref in slots.items()}
    if "skills" in genes_in:
        skills_in = genes_in["skills"]
        if not isinstance(skills_in, list):
            raise GenomeValidationError(
                [f"identity payload: 'genes.skills' must be an array when "
                 f"present, got {type(skills_in).__name__}"])
    else:
        skills_in = []  # ABSENT skills == empty array (A2 normalization)
    # skills[] is SEMANTICALLY ORDERED (runtime block order): the order is
    # part of the identity domain and must NOT be sorted here
    genes_out["skills"] = [_functional_gene_ref(ref, f"genes.skills[{i}]")
                            for i, ref in enumerate(skills_in)]
    if "regulation" in source:
        regulation_in = source["regulation"]
        if not isinstance(regulation_in, dict):
            raise GenomeValidationError(
                [f"identity payload: 'regulation' must be an object when "
                 f"present, got {type(regulation_in).__name__}"])
    else:
        regulation_in = {}  # ABSENT regulation == {} (A2 normalization)
    regulation_out: dict = {}
    for gene_id, condition in regulation_in.items():
        canonical_condition = _canonical_condition(condition,
                                                    f"regulation.{gene_id}")
        if canonical_condition:
            # only rules carrying an EFFECTIVE predicate stay in the
            # identity domain; predicate-free rules are constitutive and
            # equivalent to absence (A2 canonical equivalence)
            regulation_out[gene_id] = canonical_condition
    return {"schema_version": source["schema_version"],
             "genes": genes_out,
             "regulation": regulation_out}


def genome_identity_hash_hex(genome: dict) -> str:
    """Full lowercase-hex SHA-256 over the domain-separated canonical
    identity payload (audit-grade digest; the genome_id prefix is its
    first :data:`GENOME_ID_HEX_CHARS` characters). Non-finite floats and
    non-JSON values fail closed as typed validation errors."""
    try:
        payload_json = canonical_json(canonical_genome_identity_payload(genome))
    except (TypeError, ValueError) as exc:
        raise GenomeValidationError(
            [f"genome identity payload is not canonicalizable: {exc}"]) from None
    domain_input = f"{GENOME_ID_DOMAIN}\n{payload_json}".encode("utf-8")
    return sha256_hex(domain_input)


def compute_genome_id(genome: dict) -> str:
    """Canonical content-hash genome id:
    ``"G-" + sha256(domain\\n + canonical_payload)[:16]`` (lowercase hex).

    Deterministic; the claimed ``genome_id`` in the input is ignored; the
    input is never mutated."""
    return "G-" + genome_identity_hash_hex(genome)[:GENOME_ID_HEX_CHARS]


def validate_genome_id(genome: dict) -> None:
    """Fail closed unless the genome's claimed ``genome_id`` is exactly the
    canonical content hash of its heritable functional content.

    - missing/malclaimed pattern (must be ``G-`` + 16 lowercase hex) is
      rejected with the contract pattern in the diagnostic;
    - claimed != recomputed is rejected with BOTH values in the
      diagnostic.
    """
    claimed = genome.get("genome_id")
    if not isinstance(claimed, str) or not GENOME_ID_PATTERN.match(claimed):
        raise GenomeValidationError(
            [f"genome_id {claimed!r} does not match the canonical genome-id "
             f"contract {GENOME_ID_PATTERN.pattern!r} "
             f"(G- + {GENOME_ID_HEX_CHARS} lowercase hex chars of the "
             f"content hash)"])
    expected = compute_genome_id(genome)
    if claimed != expected:
        raise GenomeValidationError(
            [f"genome_id {claimed!r} != canonical content hash "
             f"{expected!r} recomputed from the genome's heritable "
             f"functional content (claimed id does not identify this "
             f"content)"])
