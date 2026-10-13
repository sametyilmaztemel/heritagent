"""Shared fixtures/helpers for genome schema-layer tests."""

import json
import re
from pathlib import Path

import pytest

from genome.validation.registry import TraitRegistry

REPO_ROOT = Path(__file__).resolve().parents[1]


def doc_example_genome() -> dict:
    """Extract the example genome instance embedded in architecture/eag-v0-schema.md."""
    text = (REPO_ROOT / "architecture" / "eag-v0-schema.md").read_text(encoding="utf-8")
    for block in re.findall(r"```json\n(.*?)```", text, re.S):
        data = json.loads(block)
        if "genome_id" in data and "genes" in data:
            return data
    raise AssertionError("example genome instance not found in architecture/eag-v0-schema.md")


@pytest.fixture
def doc_genome():
    return doc_example_genome()


HEX64 = "0123456789abcdef" * 4  # 64 hex chars, grammar-valid placeholder


def fake_uri(kind: str, name: str, version: int, digest: str = HEX64) -> str:
    return f"registry://{kind}/{name}@{version}/sha256:{digest}"


def finalize_genome(genome: dict) -> dict:
    """Set the canonical content-hash ``genome_id`` on a fully built genome
    (ADR-0001 Amendment A2 / issue #22): compute from the heritable
    functional content, never hard-code."""
    from genome.validation.identity import compute_genome_id
    genome["genome_id"] = compute_genome_id(genome)
    return genome


def minimal_genome(**overrides) -> dict:
    """Grammar-valid genome with placeholder digests (structural tests).

    The canonical content-hash id is computed over the BASE content unless
    the caller overrides ``genome_id`` explicitly (rejection tests).
    Callers that mutate functional CONTENT must recompute via
    :func:`finalize_genome` before a positive ``load_genome``."""
    genome = {
        "genome_id": "G-deadbeef",
        "schema_version": "0.1",
        "generation": 0,
        "lineage_id": "L-test",
        "parent": None,
        "genes": {
            "cognition": {
                "planner": {
                    "gene_id": "planner_react_v1", "version": 1, "type": "policy",
                    "origin": "seed",
                    "artifact": fake_uri("policies", "planner_react_v1", 1),
                    "provenance": {"born_generation": 0},
                }
            },
            "execution": {
                "retry_policy": {
                    "gene_id": "retry_backoff_v1", "version": 1, "type": "policy",
                    "origin": "seed",
                    "artifact": fake_uri("policies", "retry_backoff_v1", 1),
                    "provenance": {"born_generation": 0},
                }
            },
            "skills": [],
        },
    }
    genome.update(overrides)
    if "genome_id" not in overrides:
        finalize_genome(genome)
    return genome


def registered_genome(registry: TraitRegistry) -> dict:
    """Genome whose artifact URIs point at real registry content."""
    planner_uri = registry.put("policy", "planner_react_v1", 1,
                               {"style": "react", "max_plan_steps": 8})
    retry_uri = registry.put("policy", "retry_backoff_v1", 1,
                             {"max_retries": 3, "backoff": "exponential"})
    skill_uri = registry.put("skill", "look_before_heat_v1", 1, {
        "name": "look_before_heat_v1",
        "principle": "Inspect object affordances before heating.",
        "when_to_apply": "Heat tasks where the target object must be warmed.",
        "applicability": {"task_families": ["heat_and_place"]},
        "procedure": [
            {"id": "S1", "text": "find the object"},
            {"id": "S2", "text": "examine affordances"},
        ],
    })
    return finalize_genome({
        "genome_id": "G-a1b2c3d4",
        "schema_version": "0.1",
        "generation": 1,
        "lineage_id": "L-alpha",
        "parent": "G-0f1e2d3c",
        "genes": {
            "cognition": {
                "planner": {
                    "gene_id": "planner_react_v1", "version": 1, "type": "policy",
                    "origin": "seed", "artifact": planner_uri,
                    "provenance": {"born_generation": 0},
                }
            },
            "execution": {
                "retry_policy": {
                    "gene_id": "retry_backoff_v1", "version": 1, "type": "policy",
                    "origin": "seed", "artifact": retry_uri,
                    "provenance": {"born_generation": 0},
                }
            },
            "skills": [{
                "gene_id": "look_before_heat_v1", "version": 1, "type": "skill",
                "origin": "assimilation", "artifact": skill_uri,
                "provenance": {"source_trajectory": "T-00184", "cig_record": "CIG-0007",
                                "born_generation": 1},
            }],
        },
        "regulation": {
            "look_before_heat_v1": {
                "express_when": {"all_of": [{"metric": "consecutive_failures", "op": ">=", "value": 2}]}
            }
        },
    })


def registered_parent_genome(registry: TraitRegistry) -> dict:
    """G0 founder used as the diff-golden parent."""
    planner_uri = registry.put("policy", "planner_react_v1", 1,
                               {"style": "react", "max_plan_steps": 8})
    retry_uri = registry.put("policy", "retry_backoff_v1", 1,
                             {"max_retries": 3, "backoff": "exponential"})
    return finalize_genome({
        "genome_id": "G-0f1e2d3c",
        "schema_version": "0.1",
        "generation": 0,
        "lineage_id": "L-alpha",
        "parent": None,
        "genes": {
            "cognition": {
                "planner": {
                    "gene_id": "planner_react_v1", "version": 1, "type": "policy",
                    "origin": "seed", "artifact": planner_uri,
                    "provenance": {"born_generation": 0},
                }
            },
            "execution": {
                "retry_policy": {
                    "gene_id": "retry_backoff_v1", "version": 1, "type": "policy",
                    "origin": "seed", "artifact": retry_uri,
                    "provenance": {"born_generation": 0},
                }
            },
            "skills": [],
        },
    })


@pytest.fixture
def registry(tmp_path):
    return TraitRegistry(tmp_path / "registry")
