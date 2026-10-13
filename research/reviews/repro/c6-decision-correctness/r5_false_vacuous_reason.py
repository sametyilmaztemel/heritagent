"""R5: with >=2 preliminary-eligible candidates (a, b), a NON-eligible
candidate c gets a stage-4 child claiming
  vacuous=True, reason='vacuous: fewer than two preliminary-eligible
  candidates ...'
which is factually false (2 were eligible and a pair WAS evaluated). The
child never records whether the candidate was preliminary-eligible.
Run from the PR checkout with PYTHONPATH=."""
import tempfile, pathlib
from tests.gate.test_gate_orchestrator import build_world

tmp = pathlib.Path(tempfile.mkdtemp())
orch, somatic, cig, runner, inputs, reg = build_world(
    tmp, candidates=("a", "b", "c"),
    ablation_deltas={"a": 0.5, "b": 0.5, "c": 0.0})
out = orch.run(inputs)
ix = [c for c in runner.calls if c[0].startswith("ix-")]
print("interaction calls:", len(ix), "(one pair a-b evaluated)")
for r in out.results:
    inter = r.stage4_child["measurements"]["interaction"]
    print(r.gene_id, out.verdicts[r.gene_id], "vacuous:", inter["vacuous"],
          "pairs:", inter["pair_count"], "reason:", inter["reason"])
c_inter = [r for r in out.results if r.gene_id == "c"][0].stage4_child[
    "measurements"]["interaction"]
assert c_inter["vacuous"] is True and "fewer than two" in c_inter["reason"]
assert len(ix) == 20
print("REPRODUCED: committed child for c states a false reason")
