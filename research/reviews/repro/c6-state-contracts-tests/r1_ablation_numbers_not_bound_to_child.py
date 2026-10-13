"""R1: _consume_stage12_evidence checks ablation `passed` vs child
stage_pass, ids and settings, but NOT the numbers the gate actually USES:
`evaluation.delta_success` (harmful-pair contribution ranking, sensitivity)
and `evaluation.bootstrap['lower_bound']` (sensitivity tau_c recomposition)
are never cross-checked against the durable child
(`measurements.point_estimate_delta_success`, `measurements.bootstrap.lower_bound`).

Scenario: a (durable child delta 0.9) and b (durable child delta 0.1) form a
harmful pair. b's in-memory AblationEvaluation carries delta_success 0.95
(incoherent with its own child). The gate eliminates a — the candidate whose
DURABLE evidence has the higher contribution — and commits that to the live
CIG store + somatic journal. The committed bundle cannot reproduce the
elimination decision from its own children.

Run: cd pr28c && PYTHONPATH=. ../venv314/bin/python <this file>
"""
import sys
import tempfile
from pathlib import Path

from inheritance.ablation import AblationEvaluation
from inheritance.gate import GateCandidateInput
from tests.gate.test_gate_orchestrator import build_world

tmp = Path(tempfile.mkdtemp())
orch, somatic, cig, runner, inputs, registry = build_world(
    tmp, candidates=("a", "b"), ablation_deltas={"a": 0.9, "b": 0.1},
    interaction={"both": (False, ()), "neither": (True, ())})

b = inputs[1]
abl = b.ablation_evaluation
assert abl.child_record["measurements"]["point_estimate_delta_success"] == 0.1
forged = AblationEvaluation(
    _passed=abl.passed, _deltas=tuple(abl.deltas), _delta_success=0.95,
    _bootstrap=dict(abl.bootstrap, lower_bound=0.9),
    _episodes=(), _child_record=abl.child_record,
    _allocation=abl.allocation)
inputs[1] = GateCandidateInput(
    candidate_envelope=b.candidate_envelope, frozen_rank=b.frozen_rank,
    replay_evaluation=b.replay_evaluation, ablation_evaluation=forged,
    generalization_allocation=b.generalization_allocation,
    regression_allocation=b.regression_allocation,
    parent_cig_id=b.parent_cig_id)

outcome = orch.run(inputs)  # expected: EvidenceCoherenceError before stage 3
print("run() accepted incoherent ablation evidence; committed =", outcome.committed)
print("verdicts:", outcome.verdicts)
for r in outcome.results:
    print(f"  {r.gene_id}: eliminated_by={r.eliminated_by}")
durable = {gid: cig.children[f"{agg}/S11-ablation"]["measurements"]
           ["point_estimate_delta_success"]
           for gid, agg in (("a", "CIG-0007"), ("b", "CIG-0008"))}
print("durable child contributions in LIVE store:", durable)
print("somatic:", somatic.state("a", 1), somatic.state("b", 1))
bad = outcome.verdicts == {"a": "reject", "b": "promote"} and durable["a"] > durable["b"]
print("DEFECT REPRODUCED" if bad else "not reproduced")
sys.exit(0 if bad else 1)
