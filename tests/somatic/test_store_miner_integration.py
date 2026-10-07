"""#9 miner → #10 store persistence integration (issue #10 criterion 12).

A real miner-produced acquired candidate is persisted, reloaded, decided,
and reloaded again through the actual production surfaces — no live model."""

import json

import pytest

from runtime.loop import Budgets, ToolObservation, run_agent
from runtime.model_adapters import ScriptedAdapter
from traits.miner.inputs import MiningBatch
from somatic.store import SomaticStore
from traits.miner.miner import MiningSettings, TraitMiner
from trajectory.recorder.recorder import TrajectoryRecorder
from tests.miner.conftest import make_batch
from tests.runtime.conftest import react_final
from tests.somatic.conftest import acquired_envelope


@pytest.fixture
def miner_candidate(tmp_path, registry):
    """Mine one candidate through the real #9 pipeline (scripted teacher)."""
    batch, _ = make_batch(tmp_path, registry)
    miner = TraitMiner(model=ScriptedAdapter([{"proposals": [
        {"name": "check_heater_before_use",
         "principle": "Verify the heater responds before committing.",
         "when_to_apply": "Any task that requires heating an object.",
         "applicability": {"task_families": ["heat_and_place"]},
         "estimated_generality": 0.5},
    ]}]), registry=registry, settings=MiningSettings(seed=11), born_generation=1)
    frozen = miner.mine(batch)
    assert len(frozen.candidates) == 1
    return frozen.candidates[0]  # validated somatic envelope, origin=acquired


def test_miner_candidate_persists_and_reloads(tmp_path, registry, miner_candidate):
    path = tmp_path / "journal.jsonl"
    store = SomaticStore.create(path, registry)
    store.add_candidate(miner_candidate)
    store.close()

    gene_id = miner_candidate["candidate"]["gene_id"]
    reopened = SomaticStore.open(path, registry)
    stored = reopened.get(gene_id, 1)
    assert stored == miner_candidate
    assert stored["validation"]["state"] == "candidate"
    assert stored["candidate"]["origin"] == "acquired"
    # frozen applicability stays in the stored artifact (evaluation-only)
    stored_artifact = json.loads(registry.resolve(stored["candidate"]["artifact"]))
    assert stored_artifact["applicability"]["task_families"] == ["heat_and_place"]
    reopened.close()


def test_miner_candidate_rejection_preserves_artifact_uri(tmp_path, registry,
                                                            miner_candidate):
    path = tmp_path / "journal.jsonl"
    store = SomaticStore.create(path, registry)
    store.add_candidate(miner_candidate)
    gene_id = miner_candidate["candidate"]["gene_id"]
    original_uri = miner_candidate["candidate"]["artifact"]
    store.decide(gene_id, 1, "rejected", ["CIG-0042"])
    store.close()

    reopened = SomaticStore.open(path, registry)
    rejected = reopened.by_state("rejected")
    assert (gene_id, 1) in rejected
    # exact original artifact URI preserved through the rejection
    assert rejected[(gene_id, 1)]["candidate"]["artifact"] == original_uri
    history = reopened.history(gene_id, 1)
    assert [e["validation"]["state"] for e in history] == ["candidate", "rejected"]
    assert history[1]["validation"]["gate_reports"] == ["CIG-0042"]
    reopened.close()


def test_miner_candidate_validation_does_not_assimilate(tmp_path, registry,
                                                          miner_candidate):
    path = tmp_path / "journal.jsonl"
    store = SomaticStore.create(path, registry)
    store.add_candidate(miner_candidate)
    gene_id = miner_candidate["candidate"]["gene_id"]
    terminal = store.decide(gene_id, 1, "validated", ["CIG-0007"])
    assert terminal["candidate"]["origin"] == "acquired"  # NOT assimilation
    store.close()
    reopened = SomaticStore.open(path, registry)
    assert reopened.state(gene_id, 1) == "validated"
    assert reopened.get(gene_id, 1)["candidate"]["origin"] == "acquired"
    reopened.close()


def test_persisted_miner_candidate_runtime_projection_unchanged(tmp_path, registry,
                                                                  miner_candidate):
    """Persistence round trip changes nothing about the runtime projection:
    evaluation-only metadata still stripped after reload."""
    from runtime.expression import compile_runtime_config
    from runtime.seed_g0 import build_g0_seed_genome
    path = tmp_path / "journal.jsonl"
    store = SomaticStore.create(path, registry)
    store.add_candidate(miner_candidate)
    store.close()
    reopened = SomaticStore.open(path, registry)
    stored = reopened.get(miner_candidate["candidate"]["gene_id"], 1)

    genome = build_g0_seed_genome(registry)
    genome["genes"]["skills"].append({
        "gene_id": stored["candidate"]["gene_id"], "version": 1, "type": "skill",
        "origin": "seed", "artifact": stored["candidate"]["artifact"],
        "provenance": {"born_generation": 1},
    })
    config = compile_runtime_config(genome, registry)
    runtime_payload = json.dumps(vars(config.skills[0]))
    for forbidden in ("applicability", "task_families"):
        assert f'"{forbidden}"' not in runtime_payload
    reopened.close()
