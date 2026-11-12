"""The gate refuses stage-1 and stage-2 evidence that does not belong to
the candidate, its CIG run or its stage, before any episode (issue #12;
ADR-0012 D3)."""

import copy
import dataclasses

import pytest

from inheritance.gate import EvidenceCoherenceError
from tests.gate.test_gate_orchestrator import build_world


def _set(path, value):
    def change(child):
        *parents, key = path
        node = child
        for parent in parents:
            node = node[parent]
        node[key] = value
    return change


def _drop_settings(child):
    del child["measurements"]["eval_generation_settings"]


@pytest.mark.parametrize("stage,change,message", [
    ("replay", _set(("measurements", "candidate_gene_id"), "zz"),
     "replay child candidate_gene_id 'zz' != 'a'"),
    ("ablation", _set(("measurements", "candidate_gene_id"), "zz"),
     "ablation child candidate_gene_id 'zz' != 'a'"),
    ("replay", _set(("parent_cig_id",), "CIG-0099"),
     "replay child parent_cig_id 'CIG-0099' != 'CIG-0007'"),
    ("ablation", _set(("stage",), "replay"), "ablation child has stage 'replay'"),
    ("replay", _drop_settings, "replay child lacks eval_generation_settings provenance"),
    ("replay", _set(("measurements", "episodes"), []),
     r"replay child holds \[\] episodes, not exactly one \(ADR-0012 D3\)"),
    ("replay", _set(("measurements", "episodes", 0, "success"), 1),
     "replay child's episode lacks a boolean success and expressed genes"),
    ("replay", _set(("measurements", "episodes", 0, "expressed_gene_ids"), "a"),
     "replay child's episode lacks a boolean success and expressed genes"),
])
def test_a_child_that_is_not_the_candidates_is_refused_before_any_episode(
        tmp_path, stage, change, message):
    orch, _, _, runner, inputs, _ = build_world(tmp_path, candidates=("a",))
    evaluation = getattr(inputs[0], f"{stage}_evaluation")
    child = copy.deepcopy(evaluation.child_record)
    change(child)
    bad = dataclasses.replace(inputs[0], **{f"{stage}_evaluation": dataclasses.replace(
        evaluation, _child_record=child)})
    with pytest.raises(EvidenceCoherenceError, match=message):
        orch.run([bad])
    assert runner.calls == []


def test_a_child_id_used_twice_is_refused_before_any_episode(tmp_path):
    orch, _, _, runner, inputs, _ = build_world(tmp_path, candidates=("a",))
    ablation = inputs[0].ablation_evaluation
    child = copy.deepcopy(ablation.child_record)
    child["cig_id"] = inputs[0].replay_evaluation.child_record["cig_id"]
    bad = dataclasses.replace(inputs[0], ablation_evaluation=dataclasses.replace(
        ablation, _child_record=child))
    with pytest.raises(EvidenceCoherenceError, match="duplicate child id 'CIG-0007/S11-replay'"):
        orch.run([bad])
    assert runner.calls == []
