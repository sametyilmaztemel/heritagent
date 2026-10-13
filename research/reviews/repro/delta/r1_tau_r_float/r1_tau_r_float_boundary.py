"""R1: sensitivity tau_r=0.05 cell rejects 38/40 (exact drop 2/40 = 0.05)
because regression_drop = 1.0 - 38/40 = 0.050000000000000044 in float.
Run from the PR checkout with PYTHONPATH=. """
import tempfile, pathlib
from tests.gate.test_gate_orchestrator import build_world

tmp = pathlib.Path(tempfile.mkdtemp())
orch, somatic, cig, runner, inputs, reg = build_world(
    tmp, candidates=("a",), treg_success_count=38,
    interaction={"both": (True, ("a",)), "neither": (True, ())})
out = orch.run(inputs)
r = out.results[0]
print("primary verdict:", r.verdict)
drop = r.stage4_child["measurements"]["regression"]["regression_drop"]
print("persisted regression_drop:", repr(drop), " exact = 2/40 = 0.05")
grid = orch.sensitivity_recomputation("a")
print("grid regression flags:", grid["regression"])
for key in ("tau_c=0.05|tau_r=0.03", "tau_c=0.05|tau_r=0.05"):
    c = grid["cells"][key]
    print(key, "hyp_eligible:", c["hypothetical_eligible"],
          "promote:", c.get("hypothetical_promote"))
from fractions import Fraction
print("exact rational check 1 - 38/40 <= 0.05:",
      1 - Fraction(38, 40) <= Fraction(5, 100))
assert grid["regression"]["tau_r=0.05"] is False, "bug not reproduced"
print("REPRODUCED: tau_r=0.05 cell is identical to tau_r=0.03 for every "
      "possible 40-task outcome (38/40 wrongly fails)")
