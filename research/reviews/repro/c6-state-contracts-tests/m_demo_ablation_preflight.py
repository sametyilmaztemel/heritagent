"""Pinned tree rejects a corrupted ABLATION #11 child (temperature 0.7)
before any episode; the no_ablation_preflight
mutant accepts it and runs stage 3/4 -- and the suite never notices."""
import tempfile
from pathlib import Path
from inheritance.ablation import AblationEvaluation
from inheritance.gate import GateCandidateInput
from tests.gate.test_gate_orchestrator import build_world

orch, somatic, cig, runner, inputs, registry = build_world(
    Path(tempfile.mkdtemp()), candidates=("a",))
abl = inputs[0].ablation_evaluation
child = abl.child_record
child["measurements"]["eval_generation_settings"]["temperature"] = 0.7
# (foreign candidate_gene_id would only be caught later, at commit, by add_bundle)
bad = AblationEvaluation(_passed=abl.passed, _deltas=tuple(abl.deltas),
    _delta_success=abl.delta_success, _bootstrap=abl.bootstrap, _episodes=(),
    _child_record=child, _allocation=abl.allocation)
inp = GateCandidateInput(candidate_envelope=inputs[0].candidate_envelope,
    frozen_rank=0, replay_evaluation=inputs[0].replay_evaluation,
    ablation_evaluation=bad,
    generalization_allocation=inputs[0].generalization_allocation,
    regression_allocation=inputs[0].regression_allocation)
try:
    out = orch.run([inp])
    print("ACCEPTED corrupted ablation child; runner calls:", len(runner.calls),
          "verdicts:", out.verdicts)
except Exception as exc:
    print("rejected:", type(exc).__name__, "runner calls:", len(runner.calls))
