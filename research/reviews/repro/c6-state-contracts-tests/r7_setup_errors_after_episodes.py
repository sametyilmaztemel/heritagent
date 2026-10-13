"""R7: GateInputError is documented as "Invalid orchestrator input (fail
closed before any episode)", but two input/setup errors that are knowable
up front are raised only AFTER live episodes have been spent:
  * token_counter=None -> raised inside the step-4 loop, i.e. after ALL
    candidates' stage-3 generalization episodes (40 calls per candidate,
    80 for two);
  * interaction_allocation=None with >=2 candidates -> raised after every
    candidate's stage-3 + stage-4A episodes.
Both could be checked in run() before step 2 (the counter unconditionally;
the allocation whenever len(inputs) >= 2, or at least before stage 3).

Run: cd pr28c && PYTHONPATH=. <venv314>/bin/python <this file>
"""
import sys
import tempfile
from pathlib import Path

from inheritance.gate import CigGateOrchestrator, GateInputError
from tests.gate.test_gate_orchestrator import build_world


def rebuild(orch, **overrides):
    kw = dict(registry=orch._registry, base_config=orch._base_config,
              runner=orch._runner, budgets=orch._budgets,
              parent_cig_id="CIG-0007", mining_seed=11,
              token_counter=orch._token_counter,
              interaction_allocation=orch._interaction_allocation)
    kw.update(overrides)
    return CigGateOrchestrator(**kw)


hits = 0
for label, overrides in (("token_counter=None", {"token_counter": None}),
                         ("interaction_allocation=None",
                          {"interaction_allocation": None})):
    orch, somatic, cig, runner, inputs, registry = build_world(
        Path(tempfile.mkdtemp()), candidates=("a", "b"), commit_stores=False)
    o2 = rebuild(orch, **overrides)
    try:
        o2.run(inputs)
        print(f"{label}: no error")
    except GateInputError as exc:
        print(f"{label}: GateInputError after {len(runner.calls)} runner "
              f"calls: {exc}")
        hits += len(runner.calls) > 0
print("DEFECT REPRODUCED" if hits else "not reproduced")
sys.exit(0 if hits else 1)
