"""V2 at 6014bc7: three candidates; c fails replay (not preliminary-eligible);
a and b are eligible (>=2) -> the a/b pair is evaluated. c's stage-4 child
still records reason 'fewer than two preliminary-eligible candidates'."""
import tempfile, json
from pathlib import Path
from tests.gate.test_gate_orchestrator import build_world
orch, somatic, cig, runner, inputs, registry = build_world(
    Path(tempfile.mkdtemp()), candidates=("a", "b", "c"),
    replay_pass_for={"c": False})
out = orch.run(inputs)
print("eligible:", orch._last.eligible, "pairs:", sorted(orch._last.pair_evidence))
for r in out.results:
    ix = r.stage4_child["measurements"]["interaction"]
    print(r.gene_id, "verdict", r.verdict, "vacuous", ix["vacuous"], "pair_count", ix["pair_count"], "reason", json.dumps(ix["reason"]))
print("committed:", out.committed)
