"""Adjacent: child-level seed=11.0 passes #11 preflight (gate.py:573 uses ==) and fails only later at commit."""
import copy, dataclasses, pathlib, tempfile
from tests.gate.test_gate_orchestrator import build_world
with tempfile.TemporaryDirectory() as d:
    orch, somatic, cig_store, runner, inputs, registry = build_world(pathlib.Path(d), candidates=("a",))
    inp = inputs[0]
    child = copy.deepcopy(inp.replay_evaluation.child_record); child["seed"] = 11.0
    ev = dataclasses.replace(inp.replay_evaluation, _child_record=child)
    before = len(runner.calls)
    try:
        res = orch.run([dataclasses.replace(inp, replay_evaluation=ev)])
        print("ACCEPTED; committed=", getattr(res, "committed", None), "calls=", len(runner.calls) - before)
    except Exception as e:
        print(type(e).__name__, "after new runner calls =", len(runner.calls) - before, "::", str(e)[:200])
