"""Loader/validator for EAG v0.1 documents.

Validation boundary:
- structural: JSON Schema draft 2020-12 (schema artifacts in genome/schema/),
  including the geneRef invariants (artifact + provenance required,
  assimilation if/then, canonical URI grammar pattern);
- semantic (this module): URI name == gene_id, URI version == gene version,
  URI kind == gene type, SHA-256 digest == registry artifact bytes (when a
  registry is supplied), regulation keys reference existing genes.
"""

from __future__ import annotations

import json
from pathlib import Path

from jsonschema import Draft202012Validator

from genome.validation.errors import GenomeValidationError
from genome.validation.registry import TraitRegistry

SCHEMA_DIR = Path(__file__).resolve().parents[1] / "schema"

_SCHEMA_CACHE: dict[str, dict] = {}


def _schema(name: str) -> dict:
    if name not in _SCHEMA_CACHE:
        _SCHEMA_CACHE[name] = json.loads((SCHEMA_DIR / name).read_text(encoding="utf-8"))
    return _SCHEMA_CACHE[name]


def _validate_structural(document, schema_name: str, label: str) -> None:
    schema = _schema(schema_name)
    validator = Draft202012Validator(schema)
    errors = [f"{label}: {e.message} (path: {list(e.absolute_path)})" for e in validator.iter_errors(document)]
    if errors:
        raise GenomeValidationError(sorted(errors))


def _slot_gene_refs(genes: dict):
    """Yield (path, geneRef) for every gene reference in a genome."""
    for section in ("cognition", "memory", "execution"):
        for slot, ref in (genes.get(section) or {}).items():
            yield f"{section}.{slot}", ref
    for i, ref in enumerate(genes.get("skills") or []):
        yield f"skills[{i}]", ref


def _check_gene_ref(ref: dict, path: str, registry: TraitRegistry | None) -> None:
    # semantic checks that hold with or without a registry: URI grammar
    # (defense in depth beyond the schema pattern), name == gene_id,
    # version == gene version, kind == gene type.
    from genome.validation.registry import KIND_BY_TYPE, parse_uri

    try:
        parts = parse_uri(ref.get("artifact", ""))
    except GenomeValidationError as exc:
        raise GenomeValidationError([f"{path}: {e}" for e in exc.errors]) from None
    errors = []
    if parts["name"] != ref.get("gene_id"):
        errors.append(f"{path}: artifact URI name {parts['name']!r} != gene_id {ref.get('gene_id')!r}")
    if int(parts["version"]) != ref.get("version"):
        errors.append(f"{path}: artifact URI version {parts['version']} != gene version {ref.get('version')}")
    expected_kind = KIND_BY_TYPE.get(ref.get("type"))
    if parts["kind"] != expected_kind:
        errors.append(f"{path}: artifact URI kind {parts['kind']!r} does not match gene type "
                      f"{ref.get('type')!r} (expected {expected_kind!r})")
    if errors:
        raise GenomeValidationError(errors)
    if registry is not None:
        registry.resolve(ref["artifact"])  # digest == stored bytes; integrity errors propagate as-is


def _check_regulation(genome: dict) -> None:
    known = {ref["gene_id"] for _, ref in _slot_gene_refs(genome.get("genes", {}))}
    for gene_id in genome.get("regulation", {}):
        if gene_id not in known:
            raise GenomeValidationError(f"regulation references unknown gene {gene_id!r}")


def load_genome(document: dict, registry: TraitRegistry | None = None) -> dict:
    """Validate a genome document structurally and semantically; return it unchanged."""
    _validate_structural(document, "eag-0.1.genome.schema.json", "genome")
    for path, ref in _slot_gene_refs(document.get("genes", {})):
        _check_gene_ref(ref, path, registry)
    _check_regulation(document)
    return document


def load_somatic(document: dict, registry: TraitRegistry | None = None) -> dict:
    """Validate a somatic trait envelope structurally and semantically."""
    _validate_structural(document, "eag-0.1.somatic.schema.json", "somatic")
    _check_gene_ref(document["candidate"], "candidate", registry)
    return document


def load_cig(document: dict) -> dict:
    """Validate a single CIG record (aggregate or child) structurally."""
    _validate_structural(document, "eag-0.1.cig.schema.json", "cig")
    return document
