"""R3: harmful-pair resolution (and sensitivity) use the in-memory
AblationEvaluation.delta_success / bootstrap['lower_bound'], which the #11
preflight never cross-checks against the durable ablation child
(measurements.point_estimate_delta_success / bootstrap.lower_bound). The
stage-4 child does not persist the contributions/ranks it used either.
Run from the PR checkout with PYTHONPATH=."""
import tempfile, pathlib
from dataclasses import replace
from tests.gate.test_gate_orchestrator import build_world
from inheritance.gate import GateCandidateInput

tmp = pathlib.Path(tempfile.mkdtemp())
orch, somatic, cig, runner, inputs, reg = build_world(
    tmp, candidates=("a", "b"), ablation_deltas={"a": 0.5, "b": 0.1},
    interaction={"both": (False, ()), "neither": (True, ())})

# b's in-memory contribution says 0.95; its durable child still says 0.1
b_inp = inputs[1]
abl = b_inp.ablation_evaluation
forged = type(abl)(_passed=abl.passed, _deltas=tuple(abl.deltas),
                   _delta_success=0.95,
                   _bootstrap=dict(abl.bootstrap, lower_bound=0.9),
                   _episodes=(), _child_record=abl.child_record,
                   _allocation=abl.allocation)
inputs[1] = replace(b_inp, ablation_evaluation=forged)
print("b durable child point_estimate:",
      forged.child_record["measurements"]["point_estimate_delta_success"],
      "| in-memory delta_success:", forged.delta_success)
outcome = orch.run(inputs)
print("verdicts:", outcome.verdicts)
print("committed:", outcome.committed)
s4 = [r for r in outcome.results if r.gene_id == "a"][0].stage4_child
print("a stage-4 interaction keys:", sorted(s4["measurements"]["interaction"]))
# durable evidence in the live store: a's ablation child 0.5 > b's 0.1,
# yet a was eliminated
for gid in ("a", "b"):
    agg = [r for r in outcome.results if r.gene_id == gid][0].aggregate
    ab_id = [c for c in agg["child_records"] if c.endswith("-ablation")][0]
    print(gid, "live-store ablation point estimate:",
          cig.children[ab_id]["measurements"]["point_estimate_delta_success"],
          "verdict:", outcome.verdicts[gid])
print("store consistent:", cig.verify() == [])
assert outcome.verdicts == {"a": "reject", "b": "promote"}
print("REPRODUCED: committed durable evidence says a (0.5) > b (0.1) yet a "
      "was eliminated; no check, no persisted contribution in stage-4.")
