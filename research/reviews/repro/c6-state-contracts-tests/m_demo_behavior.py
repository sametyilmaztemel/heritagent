"""Behavioral demo for the surviving mutations: run the SAME scenario in the
pinned tree and in a mutated copy; the verdicts differ, yet the mutated
copy's full test suite still passes (see run_mutations.sh).

Usage (run from the tree under test, PYTHONPATH=.):
  python m_demo_behavior.py rank      # harmful pair, higher contribution has LATER rank
  python m_demo_behavior.py overhead  # 600-token expression block (> tau_k 512)
  python m_demo_behavior.py gen       # a fails in-scope generalization, harmful pair with b
"""
import sys
import tempfile
from pathlib import Path

from tests.gate.test_gate_orchestrator import build_world

mode = sys.argv[1]
tmp = Path(tempfile.mkdtemp())
if mode == "rank":
    world = build_world(tmp, candidates=("a", "b"), ranks={"a": 1, "b": 0},
                        ablation_deltas={"a": 0.9, "b": 0.1},
                        interaction={"both": (False, ()), "neither": (True, ())})
elif mode == "overhead":
    world = build_world(tmp, candidates=("a",))

    class _Big:
        identity = "fixed-600"

        def count_tokens(self, text):
            return 0 if text == "" else 600
    world[0]._token_counter = _Big()
elif mode == "gen":
    world = build_world(tmp, candidates=("a", "b"),
                        ablation_deltas={"a": 0.9, "b": 0.1},
                        interaction={"both": (False, ()), "neither": (True, ())})
    runner = world[3]
    for i in range(10):  # a's in-scope generalization: with-trait fails
        runner.set(f"in-a-{i}", "with_trait", False, ("a",))
else:
    raise SystemExit(mode)
orch, somatic, cig, runner, inputs, registry = world
out = orch.run(inputs)
print(mode, "verdicts:", out.verdicts,
      "eliminated_by:", {r.gene_id: r.eliminated_by for r in out.results},
      "stage4 pass:", {r.gene_id: r.aggregate["stages"]["interaction_regression"]["pass"]
                       for r in out.results})
