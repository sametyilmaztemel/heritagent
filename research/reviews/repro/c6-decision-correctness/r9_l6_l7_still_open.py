"""R9 (overlap L6/L7): at c6ce4c1 the #11 settings check is value-only
(temperature=False, max_tokens=64.0, seed=11.0, extra keys accepted) and
the M0 child stage_pass is truthy-coerced (stage_pass=1 / 'true' accepted).
Run from the PR checkout with PYTHONPATH=."""
import copy, tempfile, pathlib
from dataclasses import replace
from tests.gate.test_gate_orchestrator import build_world, _replay_with_child

tmp = pathlib.Path(tempfile.mkdtemp())
orch, somatic, cig, runner, inputs, reg = build_world(tmp, candidates=("a",),
                                                      commit_stores=False)
rp = inputs[0].replay_evaluation
child = copy.deepcopy(rp.child_record)
child["measurements"]["eval_generation_settings"] = {
    "temperature": False, "max_tokens": 64.0, "seed": 11.0, "stop": [],
    "top_p": 0.9}
inp = replace(inputs[0], replay_evaluation=_replay_with_child(rp, child))
out = orch.run([inp])
print("L7: aliased/extra-key settings accepted ->", out.verdicts)

# L6: stage_pass truthy coercion in add_bundle
from genome.validation.records import CigRecordStore
r = out.results[0]
children = orch._bundle_children(r)
for c in children:
    if c["stage"] == "generalization_in_scope":
        c["measurements"]["stage_pass"] = 1
store = CigRecordStore()
try:
    store.add_bundle(r.aggregate, children)
    print("L6: add_bundle accepted stage_pass=1; verify():", store.verify())
except Exception as e:
    print("L6 rejected:", e)
