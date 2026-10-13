"""R4: setup errors that are knowable before any episode are raised only
after stage-3/4 episodes have been spent (GateInputError docstring: 'fail
closed before any episode').
 (i) token_counter=None -> all stage-3 + first regression episodes run;
 (ii) interaction_allocation=None with 2 candidates -> all stage-3 + stage-4A
      episodes run before the error;
 (iii) live CIG store already holds aggregate CIG-0007 -> every episode
      (incl. interaction) runs, then commit prevalidation fails.
Run from the PR checkout with PYTHONPATH=."""
import tempfile, pathlib
from tests.gate.test_gate_orchestrator import build_world
from inheritance.gate import CigGateOrchestrator, GateInputError

def fresh(**kw):
    tmp = pathlib.Path(tempfile.mkdtemp())
    return build_world(tmp, candidates=("a", "b"), **kw)

# (i)
orch, somatic, cig, runner, inputs, reg = fresh()
orch._token_counter = None   # same as constructing with token_counter=None
try:
    orch.run(inputs)
except GateInputError as e:
    print("(i) GateInputError after", len(runner.calls), "runner calls:", e)

# (ii)
orch, somatic, cig, runner, inputs, reg = fresh()
orch._interaction_allocation = None
try:
    orch.run(inputs)
except GateInputError as e:
    print("(ii) GateInputError after", len(runner.calls), "runner calls:", e)

# (iii) pre-existing aggregate id in the live CIG store
orch, somatic, cig, runner, inputs, reg = fresh()
first = orch.run(inputs[:1])          # commits CIG-0007 for candidate a
print("(iii) first run committed:", first.committed, "calls:", len(runner.calls))
calls0 = len(runner.calls)
# a second, different candidate set re-using parent CIG-0007 (b with a's id)
from dataclasses import replace
again = [replace(inputs[1], parent_cig_id="CIG-0007")]
try:
    # #11 children of b say CIG-0008 -> rebuild coherent ones for CIG-0007
    from tests.gate.helpers import coherent_stage11_children
    rp, ab = coherent_stage11_children("b", parent="CIG-0007",
                                       replay_pass=True, ablation_pass=True)
    again = [replace(again[0], replay_evaluation=rp, ablation_evaluation=ab)]
    orch.run(again)
except Exception as e:
    print("(iii)", type(e).__name__, "after", len(runner.calls) - calls0,
          "NEW runner calls:", str(e)[:160])
