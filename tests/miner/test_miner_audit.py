"""Miner contract checks from the module audit (2026-10-13).

Fixes: a proposal the runtime skill boundary rejects is never frozen,
malformed teacher output is listed in the frozen set's rejected list, and
a MiningBatch keeps its own tuple of pairs. Test gaps closed (each
behaviour existed, but deleting it failed no test): discovery-order
tie-break, sorted applicability families, one-sided pair context, a
different-task pair, and failure evidence (parse failures, rejected calls,
retries) reaching the teacher prompt."""

import copy
import dataclasses
import json

import pytest

from runtime.loop import Budgets, ToolObservation, run_agent
from runtime.model_adapters import ScriptedAdapter
from runtime.seed_g0 import g0_runtime_config
from tests.miner.conftest import FAMILIES, TASK, make_batch, make_pair, react, run_episode
from tests.runtime.conftest import ScriptedEnv
from traits.miner.evidence import render_pair_evidence
from traits.miner.inputs import MinerInputError, MiningBatch, MiningPair
from traits.miner.miner import MiningSettings, TraitMiner
from trajectory.normalization import load_normalized
from trajectory.recorder.context import OutcomeAnnotation, TrajectoryContext
from trajectory.recorder.recorder import TrajectoryRecorder

VALID = {
    "name": "check_heater_before_use",
    "principle": "Verify the heater responds before committing to a heating action.",
    "when_to_apply": "Any task that requires heating an object.",
    "procedure": [{"id": "S1", "text": "probe the heater"},
                  {"id": "S2", "text": "proceed only on success"}],
    "applicability": {"task_families": ["heat_and_place"]},
    "estimated_generality": 0.5,
}
REPEATED_STEP_ID = {**copy.deepcopy(VALID), "name": "repeat_the_probe",
                    "procedure": [{"id": "S1", "text": "probe the heater"},
                                  {"id": "S1", "text": "probe it again"}],
                    "estimated_generality": 0.9}


def miner_with(registry, responses, *, max_proposals=10):
    model = ScriptedAdapter([])
    for response in responses:
        model.enqueue_structured(response)
    return TraitMiner(model=model, registry=registry,
                      settings=MiningSettings(seed=11, max_proposals=max_proposals),
                      born_generation=1)


def test_a_proposal_with_repeated_step_ids_is_rejected_before_the_cap(tmp_path, registry):
    batch, _ = make_batch(tmp_path, registry)
    miner = miner_with(registry, [{"proposals": [REPEATED_STEP_ID, VALID]}], max_proposals=1)
    frozen = miner.mine(batch)
    # the higher-ranked invalid proposal takes no K slot; the valid one is frozen
    assert [c["candidate"]["gene_id"].startswith("check_heater_before_use")
            for c in frozen.candidates] == [True]
    rejected = [e for e in frozen.rejected if e["reason"] == "invalid_runtime_payload"]
    assert len(rejected) == 1 and rejected[0]["proposal_name"] == "repeat_the_probe"
    assert "procedure step ids must be unique" in rejected[0]["detail"]
    assert not any(e["reason"] == "outside_cap" for e in frozen.rejected)
    record = frozen.mining_records[0]
    assert [e["reason"] for e in record["rejected"]] == ["invalid_runtime_payload"]
    assert not any(key.startswith(f"skills/{rejected[0]['gene_id']}@")
                   for key in registry._index())  # never bound to the registry


def test_malformed_teacher_output_is_in_the_frozen_rejected_list(tmp_path, registry):
    batch, _ = make_batch(tmp_path, registry)
    miner = miner_with(registry, [{"proposals": [], "confidence": 0.9}])  # schema-invalid
    frozen = miner.mine(batch)
    assert frozen.candidates == ()
    assert [e["reason"] for e in frozen.rejected] == ["malformed_output"]
    assert frozen.rejected[0] == frozen.mining_records[0]["rejected"][0]


def test_a_mining_batch_keeps_its_own_tuple_of_pairs(tmp_path, registry):
    pair = make_pair(tmp_path, registry)
    pairs = [pair]
    batch = MiningBatch(pairs=pairs, family_universe=("heat_and_place",))
    pairs.clear()
    assert batch.pairs == (pair,)


@pytest.mark.parametrize("empty", [[], ()])
def test_an_empty_mining_batch_is_rejected(empty):
    with pytest.raises(MinerInputError, match="at least one pair"):
        MiningBatch(pairs=empty, family_universe=("heat_and_place",))


# -- test gaps -------------------------------------------------------------------

def gene_names(frozen):
    return [c["candidate"]["gene_id"].rsplit("_", 1)[0] for c in frozen.candidates]


def test_equal_generality_ranks_by_discovery_order_not_by_name(tmp_path, registry):
    zeta = {**copy.deepcopy(VALID), "name": "zeta_skill"}
    alpha = {**copy.deepcopy(VALID), "name": "alpha_skill"}
    batch, _ = make_batch(tmp_path, registry)
    frozen = miner_with(registry, [{"proposals": [zeta, alpha]}]).mine(batch)
    assert gene_names(frozen) == ["zeta_skill", "alpha_skill"]


def test_applicability_family_order_does_not_change_identity(tmp_path, registry):
    forward = {**copy.deepcopy(VALID),
               "applicability": {"task_families": ["heat_and_place", "clean_surface"]}}
    reverse = {**copy.deepcopy(VALID),
               "applicability": {"task_families": ["clean_surface", "heat_and_place"]}}
    batch, _ = make_batch(tmp_path, registry, families=FAMILIES)
    frozen = miner_with(registry, [{"proposals": [forward, reverse]}]).mine(batch)
    assert len(frozen.candidates) == 1
    assert [e["reason"] for e in frozen.rejected] == ["exact_duplicate"]
    assert frozen.rejected[0]["gene_id"] == frozen.candidates[0]["candidate"]["gene_id"]


@pytest.mark.parametrize("field", ["benchmark", "split", "environment"])
@pytest.mark.parametrize("one_sided", [True, False])
def test_pair_context_must_match_exactly(tmp_path, registry, field, one_sided):
    success = run_episode(tmp_path, registry, name="s", success=True)
    failure = run_episode(tmp_path, registry, name="f", success=False)
    success = dataclasses.replace(success, provenance={**success.provenance, field: "train"})
    failure = dataclasses.replace(failure, provenance={
        **failure.provenance, field: None if one_sided else "valid_seen"})
    with pytest.raises(MinerInputError, match=f"pair members disagree on {field}"):
        MiningPair(success=success, failure=failure, task_family="heat_and_place")


def test_a_pair_of_different_tasks_is_rejected(tmp_path, registry):
    with pytest.raises(MinerInputError, match="pair members are different tasks"):
        make_pair(tmp_path, registry, failure_task="Clean the counter.")


def failure_with_every_failure_kind(tmp_path, registry):
    """A failed episode with a plan parse failure, a rejected tool call and
    a retried tool failure, recorded and normalized like any other."""
    adapter = ScriptedAdapter(["I am not sure what to do.",
                               react("teleport", {"to": "kitchen"}),
                               react("heat_object", {"object": "plate"})])
    env = ScriptedEnv({"heat_object": [
        ToolObservation(ok=False, content="heater busy"),
        ToolObservation(ok=False, content="heater offline", retryable=False)]})
    path = tmp_path / "every-failure.jsonl"
    recorder = TrajectoryRecorder(TrajectoryContext(genome_id="G-0f1e2d3c",
                                                     run_id="R-every-failure"), path)
    result = run_agent(config=g0_runtime_config(registry), model=adapter, env=env, task=TASK,
                       budgets=Budgets(max_steps=2, max_total_retries=5,
                                       max_tokens_per_request=64),
                       hooks=(recorder,), sleep=lambda _delay: None)
    assert result.status == "budget_exhausted"
    recorder.attach_outcome(OutcomeAnnotation(success=False, reward=0.0,
                                              evaluator_id="scripted-evaluator",
                                              evaluator_version="1.0"))
    recorder.finalize()
    return load_normalized(path)


def test_failure_evidence_keeps_parse_failures_rejections_and_retries(tmp_path, registry):
    success = run_episode(tmp_path, registry, name="s", success=True)
    pair = MiningPair(success=success, failure=failure_with_every_failure_kind(tmp_path, registry),
                      task_family="heat_and_place")
    steps = json.loads(render_pair_evidence(pair)[0])["failed_trajectory"]["steps"]
    calls = [c for s in steps for c in s["model_calls"]]
    assert [c["parse_ok"] for c in calls][:1] == [False]
    assert calls[0]["parse_error"] and calls[0]["raw_response"] == "I am not sure what to do."
    tools = [t for s in steps for t in s["tool_calls"]]
    assert [(t["tool"], t["rejected"], t["reject_reason"]) for t in tools][0] == (
        "teleport", True, "unknown_tool")
    assert [t["result_content"] for t in tools if t["tool"] == "heat_object"][-1] == "heater offline"
    assert [r["reason"] for s in steps for r in s["retries"]] == ["plan_parse", "tool_failure"]
