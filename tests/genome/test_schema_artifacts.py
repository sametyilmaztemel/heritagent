"""Schema artifacts compile, carry the ADR invariants, and accept the doc example."""

import pytest
from jsonschema import Draft202012Validator

from genome.validation.errors import GenomeValidationError
from genome.validation.loader import SCHEMA_DIR, load_genome

SCHEMA_FILES = [
    "eag-0.1.genome.schema.json",
    "eag-0.1.somatic.schema.json",
    "eag-0.1.cig.schema.json",
]


@pytest.mark.parametrize("name", SCHEMA_FILES)
def test_schema_artifacts_compile(name):
    schema = load_schema(name)
    Draft202012Validator.check_schema(schema)


def load_schema(name):
    import json
    return json.loads((SCHEMA_DIR / name).read_text(encoding="utf-8"))


def test_genome_schema_enforces_adopted_invariants():
    gene_ref = load_schema("eag-0.1.genome.schema.json")["$defs"]["geneRef"]
    assert "artifact" in gene_ref["required"]
    assert "provenance" in gene_ref["required"]
    assert "sha256:[0-9a-f]{64}" in gene_ref["properties"]["artifact"]["pattern"]
    # assimilation if/then requiring source trajectory + CIG record
    assert gene_ref["if"]["properties"]["origin"] == {"const": "assimilation"}
    then_required = gene_ref["then"]["properties"]["provenance"]["required"]
    assert {"source_trajectory", "cig_record", "born_generation"} <= set(then_required)
    # provenance requires born_generation for all origins
    assert "born_generation" in gene_ref["properties"]["provenance"]["required"]
    # no oracle environment labels in the runtime regulation vocabulary
    predicate = load_schema("eag-0.1.genome.schema.json")["$defs"]["predicate"]
    assert "task_type" not in predicate["properties"]["metric"]["enum"]


def test_doc_example_genome_validates(doc_genome):
    assert load_genome(doc_genome) == doc_genome


def test_genome_rejects_evaluation_metadata(doc_genome):
    # D7: evaluation metadata is never stored in the EAG itself
    polluted = dict(doc_genome)
    polluted["fitness"] = {"task_success": 0.8}
    with pytest.raises(GenomeValidationError, match="fitness"):
        load_genome(polluted)
