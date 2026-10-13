"""R2: _commit_results() is documented as a fail-atomic BATCH commit ("Any
failure leaves both live stores untouched"), but the only somatic
prevalidation is `state(...) == 'candidate'`. The live loop is
`for each result: somatic.decide(...); cig.add_bundle(...)` with no rollback,
so any failure of the k-th decide() (journal append I/O error, stale-handle
guard, schema re-validation) leaves candidates 1..k-1 committed in BOTH live
stores and run() raises without returning a GateOutcome.

Fault injected: the persistent somatic journal's 2nd append raises OSError
(e.g. ENOSPC). Expected (per docstring): both live stores unchanged.

Run: cd pr28c && PYTHONPATH=. <venv314>/bin/python <this file>
"""
import sys
import tempfile
from pathlib import Path

from tests.gate.test_gate_orchestrator import build_world

tmp = Path(tempfile.mkdtemp())
orch, somatic, cig, runner, inputs, registry = build_world(
    tmp, candidates=("a", "b"))

writer = somatic._writer
real_append = writer.append
calls = {"n": 0}


def flaky_append(record):
    calls["n"] += 1
    # patched after build_world's add_candidate()s: the 2nd append seen
    # here is candidate b's decide()
    if record["op"] == "decision_recorded" and calls["n"] == 2:
        raise OSError(28, "No space left on device")
    return real_append(record)


writer.append = flaky_append
try:
    orch.run(inputs)
    print("run() returned normally")
except OSError as exc:
    print("run() raised:", exc)

print("live CIG aggregates:", sorted(cig.aggregates))
print("live CIG children  :", len(cig.children))
print("somatic a:", somatic.state("a", 1), " b:", somatic.state("b", 1))
partial = (sorted(cig.aggregates) == ["CIG-0007"]
           and somatic.state("a", 1) == "validated"
           and somatic.state("b", 1) == "candidate")
print("DEFECT REPRODUCED (partial batch commit)" if partial else "not reproduced")
_partial = partial

# recovery: the fault is gone, the caller retries the same batch
writer.append = real_append
calls_before = len(runner.calls)
try:
    orch.run(inputs)
    print("retry succeeded")
except Exception as exc:  # noqa: BLE001
    print(f"retry re-ran {len(runner.calls) - calls_before} episodes, then "
          f"failed: {type(exc).__name__}: {str(exc)[:90]}")
print("after retry: somatic b:", somatic.state("b", 1),
      "| live aggregates:", sorted(cig.aggregates))
sys.exit(0 if _partial else 1)
