"""Somatic origin="acquired" schema amendment tests (issue #9 criterion 1)
plus germline rejection and runtime-projection compatibility."""

import pytest

from genome.validation.errors import GenomeValidationError
from genome.validation.loader import load_genome, load_somatic
from tests.miner.conftest import TASK, make_pair, run_episode


def acquired_envelope(registry, tmp_path, *, with_source_trajectory=True,
                       with_born_generation=True):
    from tests.runtime.conftest import ToolObservation, react_action, react_final
    import json as _json
    success = run_episode(tmp_path, registry, name="s", success=True, task=TASK)
    artifact = registry.put("skill", "look_before_heat_v1", 1, {
        "name": "look_before_heat_v1",
        "principle": "Inspect affordances first.",
        "when_to_apply": "Heat tasks.",
    })
    provenance = {}
    if with_source_trajectory:
        provenance["source_trajectory"] = success.trajectory_id
    if with_born_generation:
        provenance["born_generation"] = 1
    return {
        "candidate": {
            "gene_id": "look_before_heat_v1", "version": 1, "type": "skill",
            "origin": "acquired", "artifact": artifact,
            "provenance": provenance,
        },
        "validation": {"state": "candidate"},
    }


def test_valid_acquired_somatic_candidate_validates(registry, tmp_path):
    envelope = acquired_envelope(registry, tmp_path)
    assert load_somatic(envelope, registry) == envelope


def test_acquired_states_lifecycle_valid(registry, tmp_path):
    envelope = acquired_envelope(registry, tmp_path)
    for state in ("candidate", "rejected", "validated"):
        envelope["validation"] = {"state": state, "gate_reports": ["CIG-0001"]} \
            if state != "candidate" else {"state": state}
        assert load_somatic(envelope, registry)["validation"]["state"] == state


def test_acquired_without_source_trajectory_rejected(registry, tmp_path):
    envelope = acquired_envelope(registry, tmp_path, with_source_trajectory=False)
    with pytest.raises(GenomeValidationError, match="source_trajectory"):
        load_somatic(envelope, registry)


def test_acquired_without_born_generation_rejected(registry, tmp_path):
    envelope = acquired_envelope(registry, tmp_path, with_born_generation=False)
    with pytest.raises(GenomeValidationError, match="born_generation"):
        load_somatic(envelope, registry)


def test_acquired_origin_in_germline_genome_rejected(tmp_path, registry):
    """The germline genome/0.1 schema still rejects `acquired` — only CIG
    assimilation creates germline refs (origin=assimilation)."""
    from tests.trajectory.conftest import run_captured as _rc  # noqa: F401
    genome = {
        "genome_id": "G-a1b2c3d4",
        "schema_version": "0.1",
        "generation": 1,
        "lineage_id": "L-test",
        "parent": None,
        "genes": {
            "skills": [{
                "gene_id": "look_before_heat_v1", "version": 1, "type": "skill",
                "origin": "acquired",
                "artifact": f"registry://skills/look_before_heat_v1@1/sha256/{'ab' * 32}"
                             .replace("/sha256/", "/sha256:"),
                "provenance": {"source_trajectory": "T-1", "born_generation": 1},
            }],
        },
    }
    with pytest.raises(GenomeValidationError, match="is not one of"):
        load_genome(genome)


def test_acquired_with_cig_record_is_contradictory_but_schema_valid(registry, tmp_path):
    """cig_record belongs to assimilation, not acquisition; the somatic schema
    does not forbid the extra provenance key (documented semantics: miners
    never emit it — asserted here via the produced candidates in test_miner)."""
    envelope = acquired_envelope(registry, tmp_path)
    envelope["candidate"]["provenance"]["cig_record"] = "CIG-9999"
    assert load_somatic(envelope, registry) == envelope
