import copy, dataclasses, tempfile, pathlib
from tests.gate.test_gate_orchestrator import build_world
SP = pathlib.Path("<scratch>/out/repro/claims-2/tmp"); SP.mkdir(parents=True, exist_ok=True)
orch, som, cig, runner, inputs, reg = build_world(pathlib.Path(tempfile.mkdtemp(dir=SP)), candidates=("a",))
ev = inputs[0].replay_evaluation; c = copy.deepcopy(ev.child_record); c["cig_id"] = "CIG-0007/S99-replay"
inp = dataclasses.replace(inputs[0], replay_evaluation=dataclasses.replace(ev, _child_record=c))
out = orch.run([inp])
print("aggregate child_records:", out.results[0].aggregate["child_records"])
print("somatic state:", som.state("a", 1), "gate_reports:", som.get("a", 1)["validation"]["gate_reports"])
print("live cig aggregates:", list(cig._aggregates), "runner calls:", len(runner.calls))
