"""Shared fixtures for somatic store tests."""

import pytest

from genome.validation.registry import TraitRegistry
from somatic.store import SomaticStore
from tests.trajectory.conftest import run_captured  # noqa: F401 — used in miner integration
from trajectory.normalization import load_normalized
from trajectory.recorder.context import OutcomeAnnotation
from trajectory.recorder.recorder import TrajectoryRecorder


@pytest.fixture
def registry(tmp_path):
    return TraitRegistry(tmp_path / "registry")


def acquired_envelope(registry: TraitRegistry, gene_id: str = "look_before_heat_v1",
                       version: int = 1) -> dict:
    """A schema-valid acquired candidate whose artifact exists in the registry."""
    payload = {
        "name": gene_id,
        "principle": "Inspect object affordances before acting.",
        "when_to_apply": "Heat tasks requiring pre-inspection.",
        "applicability": {"task_families": ["heat_and_place"]},
        "procedure": [{"id": "S1", "text": "find the object"}],
    }
    uri = registry.put("skill", gene_id, version, payload)
    return {
        "candidate": {
            "gene_id": gene_id, "version": version, "type": "skill",
            "origin": "acquired", "artifact": uri,
            "provenance": {"born_generation": 1,
                            "source_trajectory": "T-aaaa1111aaaa"},
        },
        "validation": {"state": "candidate"},
    }


@pytest.fixture
def envelope(registry):
    return acquired_envelope(registry)
