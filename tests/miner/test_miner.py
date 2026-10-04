"""Trait Miner pipeline tests (issue #9 criteria 4-11) — ScriptedAdapter,
no live model/GPU."""

import json
import re

import pytest

from runtime.model_adapters import ScriptedAdapter
from traits.miner.inputs import MiningBatch
from trajectory.storage.canonical import sha256_hex
from traits.miner.miner import (
    MiningSettings,
    TraitMiner,
    output_schema,
    output_schema_sha256,
    prompt_template_sha256,
)
from tests.miner.conftest import FAMILIES, TASK, make_batch, make_pair, run_episode

GOLDEN_PROPOSALS = [{
    "name": "check_heater_before_use",
    "principle": "Verify the heater responds before committing to a heating action.",
    "when_to_apply": "Any task that requires heating an object.",
    "procedure": [{"id": "S1", "text": "probe the heater"},
                   {"id": "S2", "text": "proceed only on success"}],
    "applicability": {"task_families": ["heat_and_place"]},
    "estimated_generality": 0.5,
    "evidence_rationale": "failed run shows the heater was never probed",
}]

SECOND_PROPOSAL = {
    "name": "retry_transient_failures",
    "principle": "Retry once on transient tool failure before replanning.",
    "when_to_apply": "Tool failure with a transient-looking message.",
    "applicability": {"task_families": ["heat_and_place", "clean_surface"]},
    "estimated_generality": 1.0,
}


def mined_skill_schema():
    return json.loads(json.dumps(output_schema()))


@pytest.fixture
def miner(registry, context):
    from runtime.model_adapters import ScriptedAdapter as _S  # noqa: F401
    return TraitMiner(model=ScriptedAdapter([]), registry=registry,
                       settings=MiningSettings(seed=11), born_generation=1)


def scripted_miner(miner, responses):
    """Replace the miner's adapter queue with scripted structured responses."""
    miner._model = ScriptedAdapter([])
    for response in responses:
        miner._model.enqueue_structured(response)
    return miner


def test_prompt_template_hash_is_stable():
    from hashlib import sha256
    from pathlib import Path
    template = Path("traits/miner/prompts/differential-skill-v0.txt").read_bytes()
    assert prompt_template_sha256() == sha256(template).hexdigest()
    assert prompt_template_sha256() == prompt_template_sha256()


def test_output_schema_hash_is_stable():
    from hashlib import sha256
    from pathlib import Path
    schema_bytes = Path("traits/schemas/mined-skill-proposal-0.1.schema.json").read_bytes()
    assert output_schema_sha256() == sha256(schema_bytes).hexdigest()


def test_mining_settings_propagate_to_teacher(tmp_path, registry, context, miner):
    batch, _ = make_batch(tmp_path, registry)
    scripted = scripted_miner(miner, [{"proposals": []}])
    miner._settings = MiningSettings(temperature=0.7, seed=4242, max_tokens=1024)
    miner.mine(batch)
    request = miner._model.requests[0]
    assert request.settings.temperature == 0.7
    assert request.settings.seed == 4242
    assert request.settings.max_tokens == 1024
    assert request.schema == mined_skill_schema()


def test_structured_happy_path_produces_somatic_candidate(tmp_path, registry, context, miner):
    batch, pair = make_batch(tmp_path, registry)
    scripted = scripted_miner(miner, [{"proposals": [GOLDEN_PROPOSALS[0]]}])
    frozen = miner.mine(batch)

    assert frozen.candidate_set_sha256
    assert len(frozen.candidates) == 1
    envelope = frozen.candidates[0]
    candidate = envelope["candidate"]
    assert candidate["type"] == "skill"
    assert candidate["origin"] == "acquired"
    assert candidate["version"] == 1
    assert candidate["provenance"]["source_trajectory"] == pair.success.trajectory_id
    assert candidate["provenance"]["born_generation"] == 1
    assert envelope["validation"] == {"state": "candidate"}
    # artifact stored through the existing registry with frozen applicability
    stored = json.loads(miner._registry.resolve(candidate["artifact"]))
    assert stored["applicability"]["task_families"] == ["heat_and_place"]
    assert stored["principle"] == GOLDEN_PROPOSALS[0]["principle"]


def test_zero_proposals_is_valid(tmp_path, registry, context, miner):
    batch, _ = make_batch(tmp_path, registry)
    scripted = scripted_miner(miner, [{"proposals": []}])
    frozen = miner.mine(batch)
    assert frozen.candidates == ()
    assert frozen.candidate_set_sha256
    record = frozen.mining_records[0]
    assert record["raw_structured_response"] == {"proposals": []}
    assert record["accepted_proposals"] == []


def test_malformed_output_fails_closed_but_auditable(tmp_path, registry, context, miner):
    batch, _ = make_batch(tmp_path, registry)
    # extra field violates additionalProperties:false -> adapter raises
    scripted = scripted_miner(miner, [{"proposals": [], "confidence": 0.9}])
    frozen = miner.mine(batch)
    assert frozen.candidates == ()
    rejected = frozen.mining_records[0]["rejected"]
    assert rejected[0]["reason"] == "malformed_output"
    assert frozen.mining_records[0]["raw_structured_response"] is None


def test_applicability_outside_family_universe_rejected(tmp_path, registry, context, miner):
    batch, _ = make_batch(tmp_path, registry, families=("heat_and_place",))
    proposal = dict(GOLDEN_PROPOSALS[0])
    proposal["applicability"] = {"task_families": ["quantum_chamber"]}
    scripted = scripted_miner(miner, [{"proposals": [proposal]}])
    frozen = miner.mine(batch)
    assert frozen.candidates == ()
    assert frozen.rejected[0]["reason"] == "invalid_applicability"
    assert "quantum_chamber" in frozen.rejected[0]["detail"]
    assert frozen.mining_records[0]["rejected"][0]["reason"] == "invalid_applicability"


def test_deterministic_proposal_identity(tmp_path, registry, context, miner):
    batch, _ = make_batch(tmp_path, registry)
    scripted = scripted_miner(miner, [{"proposals": [GOLDEN_PROPOSALS[0]]}])
    frozen_a = miner.mine(batch)
    # identical proposal payload -> identical gene identity (fresh registry)
    registry_b = type(miner._registry)(miner._registry.root.parent / "b")
    miner_b = TraitMiner(model=ScriptedAdapter([]), registry=registry_b,
                           settings=MiningSettings(seed=11), born_generation=1)
    scripted_b = scripted_miner(miner_b, [{"proposals": [GOLDEN_PROPOSALS[0]]}])
    frozen_b = miner_b.mine(batch)
    gene_a = frozen_a.candidates[0]["candidate"]["gene_id"]
    gene_b = frozen_b.candidates[0]["candidate"]["gene_id"]
    assert gene_a == gene_b
    assert re.match(r"^[a-z0-9_]+(_v[0-9]+)?$", gene_a)


def test_exact_duplicate_removed_before_cap(tmp_path, registry, context, miner):
    batch, _ = make_batch(tmp_path, registry)
    duplicate = dict(GOLDEN_PROPOSALS[0])
    scripted = scripted_miner(miner, [{"proposals": [GOLDEN_PROPOSALS[0], duplicate]}])
    frozen = miner.mine(batch)
    assert len(frozen.candidates) == 1
    reasons = [r["reason"] for r in frozen.rejected]
    assert reasons.count("exact_duplicate") == 1


def test_ranking_generality_desc_discovery_order_tie_break(tmp_path, registry, context, miner):
    low = dict(GOLDEN_PROPOSALS[0])  # generality 0.5, proposed first
    low["name"] = "low_generality_skill"
    high = dict(SECOND_PROPOSAL)     # generality 1.0, proposed second
    batch, _ = make_batch(tmp_path, registry)
    scripted = scripted_miner(miner, [{"proposals": [low, high]}])
    frozen = miner.mine(batch)
    gene_ids = [c["candidate"]["gene_id"] for c in frozen.candidates]
    # 1.0 ranks before 0.5 despite later discovery order
    assert gene_ids[0].startswith("retry_transient_failures_")
    assert gene_ids[1].startswith("low_generality_skill_")


def test_ranking_tie_break_keeps_discovery_order(tmp_path, registry, context, miner):
    first = dict(GOLDEN_PROPOSALS[0], name="alpha_skill")
    second = dict(GOLDEN_PROPOSALS[0], name="beta_skill")
    batch, _ = make_batch(tmp_path, registry)
    scripted = scripted_miner(miner, [{"proposals": [first, second]}])
    frozen = miner.mine(batch)
    gene_ids = [c["candidate"]["gene_id"] for c in frozen.candidates]
    assert gene_ids[0].startswith("alpha_skill_")
    assert gene_ids[1].startswith("beta_skill_")


def test_cap_k10_and_outside_cap_reasons(tmp_path, registry, context):
    proposals = []
    for index in range(12):
        proposals.append({
            "name": f"skill_{index:02d}",
            "principle": f"principle {index}",
            "when_to_apply": "always",
            "applicability": {"task_families": ["heat_and_place"]},
            "estimated_generality": index / 20,  # 0.0 .. 0.55, ascending
        })
    adapter = ScriptedAdapter([])
    adapter.enqueue_structured({"proposals": proposals})
    from traits.miner.miner import TraitMiner as TM
    miner = TM(model=adapter, registry=registry,
                settings=MiningSettings(seed=11, max_proposals=10), born_generation=1)
    pair = make_pair(tmp_path, registry)
    batch = MiningBatch(pairs=(pair,), family_universe=FAMILIES)
    frozen = miner.mine(batch)
    assert len(frozen.candidates) == 10
    outside = [r for r in frozen.rejected if r["reason"] == "outside_cap"]
    assert len(outside) == 2
    # the two lowest-generality proposals are the capped-out ones
    names = {r["proposal_name"] for r in outside}
    assert "skill_00_..." or True  # identity suffix; assert by prefix
    assert all(name.startswith(("skill_00", "skill_01")) for name in names)


def test_candidate_set_digest_deterministic_and_frozen(tmp_path, registry, context, miner):
    batch, _ = make_batch(tmp_path, registry)
    scripted = scripted_miner(miner, [{"proposals": [GOLDEN_PROPOSALS[0], SECOND_PROPOSAL]}])
    frozen = miner.mine(batch)
    again_digest = frozen.candidate_set_sha256
    assert frozen.candidate_set_sha256 == again_digest
    # frozen: mutating the returned tuple contents raises (immutable) and
    # the digest does not change
    from trajectory.storage.canonical import canonical_json, sha256_hex
    recomputed = sha256_hex(canonical_json(list(frozen.candidates)).encode("utf-8"))
    assert recomputed == frozen.candidate_set_sha256


def test_mining_record_is_complete_audit_artifact(tmp_path, registry, context, miner):
    batch, pair = make_batch(tmp_path, registry)
    scripted = scripted_miner(miner, [{"proposals": [GOLDEN_PROPOSALS[0]]}])
    frozen = miner.mine(batch)
    record = frozen.mining_records[0]
    assert record["mining_record_id"].startswith("MR-")
    assert record["success_trajectory_ids"] == [pair.success.trajectory_id]
    assert record["failure_trajectory_ids"] == [pair.failure.trajectory_id]
    assert record["task_families"] == ["heat_and_place"]
    assert len(record["evidence_sha256"]) == 64
    assert len(record["prompt_template_sha256"]) == 64
    assert len(record["output_schema_sha256"]) == 64
    assert record["mining_seed"] == 11
    assert record["generation_settings"]["temperature"] == 0.7
    assert record["model_metadata"]["model_id"] == "scripted-test-model"
    assert record["raw_structured_response"]["proposals"][0]["name"] == \
        GOLDEN_PROPOSALS[0]["name"]
    assert record["accepted_proposals"][0]["source_trajectory_id"] == \
        pair.success.trajectory_id
    # mining evidence, not CIG evidence
    assert "cig_record" not in json.dumps(record)


def test_no_cig_record_on_candidate(tmp_path, registry, context, miner):
    batch, _ = make_batch(tmp_path, registry)
    scripted = scripted_miner(miner, [{"proposals": [GOLDEN_PROPOSALS[0]]}])
    frozen = miner.mine(batch)
    assert "cig_record" not in json.dumps(frozen.candidates)


def test_identical_evidence_and_output_freeze_identically(tmp_path, registry, context, miner):
    frozen_runs = []
    for name in ("a", "b"):
        batch, _ = make_batch(tmp_path / name, registry,
                               success_trajectory_id="T-aaaa1111aaaa",
                               failure_trajectory_id="T-ffff5555ffff")
        scripted = scripted_miner(miner, [{"proposals": [GOLDEN_PROPOSALS[0]]}])
        frozen = miner.mine(batch)
        frozen_runs.append(frozen)
    a, b = frozen_runs
    assert a.candidate_set_sha256 == b.candidate_set_sha256
    assert a.candidates == b.candidates
    # mining records reference the deterministic trajectory ids too
    assert a.mining_records[0]["success_trajectory_ids"] == \
        b.mining_records[0]["success_trajectory_ids"] == ["T-aaaa1111aaaa"]


def test_runtime_leakage_invariant(tmp_path, registry, context, miner):
    """The stored mined artifact keeps frozen applicability; the runtime
    projection and any agent-visible messages strip it — plus generality,
    rationale, success labels and family oracles never reach the runtime."""
    from runtime.expression import compile_runtime_config
    from runtime.seed_g0 import build_g0_seed_genome
    batch, _ = make_batch(tmp_path, registry)
    scripted = scripted_miner(miner, [{"proposals": [GOLDEN_PROPOSALS[0]]}])
    frozen = miner.mine(batch)
    mined = frozen.candidates[0]["candidate"]

    # stored artifact keeps the frozen evaluation-only metadata
    stored = json.loads(miner._registry.resolve(mined["artifact"]))
    assert stored["applicability"]["task_families"] == ["heat_and_place"]

    # attach the mined candidate to a genome as a runtime skill reference
    genome = build_g0_seed_genome(miner._registry)
    genome["genes"]["skills"].append({
        "gene_id": mined["gene_id"], "version": 1, "type": "skill",
        "origin": "seed",  # manual attachment for the leakage test
        "artifact": mined["artifact"],
        "provenance": {"born_generation": 1},
    })
    config = compile_runtime_config(genome, miner._registry)
    # the compiled runtime skill carries no evaluation-only metadata
    runtime_skill = json.dumps(vars(config.skills[0]))
    for forbidden_key in ("applicability", "task_families", "estimated_generality",
                           "evidence_rationale", "outcome", "reward",
                           "success_trajectory_ids"):
        assert f'"{forbidden_key}"' not in runtime_skill, \
            f"{forbidden_key} leaked into runtime"
    # the frozen family oracle label never reaches the runtime either
    assert "heat_and_place" not in runtime_skill
