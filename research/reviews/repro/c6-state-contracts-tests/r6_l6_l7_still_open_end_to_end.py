"""R6: critic items L6/L7 are still open at c6ce4c1, end to end through
CigGateOrchestrator.run() (not just the store):
  L7: #11 settings check is `settings.get(key) != expected` for 4 keys, so
      temperature=False, max_tokens=64.0, seed=11.0 and extra keys pass;
  L6: `evaluation.passed != m.get('stage_pass')` is value equality, so a #11
      child with stage_pass=1 passes preflight, and add_bundle()'s
      bool()-coercing _stage_pass_issue commits it to the LIVE store.

Run: cd pr28c && PYTHONPATH=. <venv314>/bin/python <this file>
"""
import sys
import tempfile
from pathlib import Path

from inheritance.gate import GateCandidateInput
from tests.gate.test_gate_orchestrator import _replay_with_child, build_world


def run_with(mutate):
    tmp = Path(tempfile.mkdtemp())
    orch, somatic, cig, runner, inputs, registry = build_world(
        tmp, candidates=("a",))
    replay = inputs[0].replay_evaluation
    child = replay.child_record
    mutate(child["measurements"])
    bad = GateCandidateInput(
        candidate_envelope=inputs[0].candidate_envelope, frozen_rank=0,
        replay_evaluation=_replay_with_child(replay, child),
        ablation_evaluation=inputs[0].ablation_evaluation,
        generalization_allocation=inputs[0].generalization_allocation,
        regression_allocation=inputs[0].regression_allocation)
    try:
        out = orch.run([bad])
        stored = cig.children["CIG-0007/S11-replay"]["measurements"]
        return f"ACCEPTED committed={out.committed} stored={ {k: stored[k] for k in ('stage_pass', 'eval_generation_settings')} } stage3/4 calls={len(runner.calls)}"
    except Exception as exc:  # noqa: BLE001
        return f"rejected: {type(exc).__name__}"


def settings(**kw):
    def m(meas):
        meas["eval_generation_settings"].update(kw)
    return m


cases = {
    "L7 temperature=False": settings(temperature=False),
    "L7 max_tokens=64.0": settings(max_tokens=64.0),
    "L7 seed=11.0": settings(seed=11.0),
    "L7 extra key top_p=0.9": settings(top_p=0.9),
    "L6 stage_pass=1": lambda meas: meas.__setitem__("stage_pass", 1),
}
accepted = 0
for name, fn in cases.items():
    res = run_with(fn)
    accepted += res.startswith("ACCEPTED")
    print(f"{name:26s} -> {res}")
print("DEFECT REPRODUCED" if accepted else "not reproduced")
sys.exit(0 if accepted else 1)
