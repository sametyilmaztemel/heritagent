"""R8: sensitivity_recomputation() is documented as recomputation "over the
immutable primary stage measurements", but it reads `self._last`, which
  (a) stores the CALLER's `inputs` list by reference
      (`_LastRun(inputs=inputs, gene_ids=gene_ids, ...)`): reordering the
      caller's list after run() silently re-pairs gene ids with the wrong
      #11 evidence in `zip(run.inputs, run.gene_ids)`;
  (b) is not cleared when a later run() aborts (e.g. contaminated
      interaction), so sensitivity silently answers for the PREVIOUS run;
      and it is set BEFORE the commit, so a run whose commit raised is also
      reported as if it were a primary run. The grid carries no run/outcome
      identity, so a caller cannot tell.

Run: cd pr28c && PYTHONPATH=. <venv314>/bin/python <this file>
"""
import sys
import tempfile
from pathlib import Path

from inheritance.runner import EvaluationIntegrityError
from tests.gate.test_gate_orchestrator import build_world

# (a) aliasing: a passes replay, b fails replay
orch, somatic, cig, runner, inputs, registry = build_world(
    Path(tempfile.mkdtemp()), candidates=("a", "b"),
    replay_pass_for={"a": True, "b": False})
outcome = orch.run(inputs)
before = orch.sensitivity_recomputation("a")["replay"]["hypothetical_pass"]
inputs.reverse()   # caller reuses / reorders its own list after the run
after = orch.sensitivity_recomputation("a")["replay"]["hypothetical_pass"]
print(f"(a) primary verdict a={outcome.verdicts['a']}; sensitivity replay "
      f"pass for a: before={before} after caller list reorder={after}")
alias_bug = before != after

# (b) stale run: a second run on the same orchestrator aborts
orch2, somatic2, cig2, runner2, inputs2, registry2 = build_world(
    Path(tempfile.mkdtemp()), candidates=("a", "b"), commit_stores=False)
first = orch2.run(inputs2)
for i in range(10):  # contaminate the neither arm for the second run
    runner2.set(f"ix-{i}", "neither", True, ("a",))
try:
    orch2.run(inputs2)
except EvaluationIntegrityError as exc:
    print("(b) second run aborted:", str(exc)[:70], "...")
grid = orch2.sensitivity_recomputation("a")
print("(b) sensitivity after the aborted run still answers:",
      grid["primary_eligible"], "(from the FIRST run, no staleness signal)")
stale_bug = grid["primary_eligible"] == ["a", "b"]
print("DEFECT REPRODUCED" if (alias_bug and stale_bug) else
      f"partial: alias={alias_bug} stale={stale_bug}")
sys.exit(0 if (alias_bug or stale_bug) else 1)
