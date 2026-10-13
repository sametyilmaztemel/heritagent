"""Groups 1+2 repro against PR #28 @4ca5517 (read-only; tmp dirs only)."""
import tempfile, copy, json
from pathlib import Path
from tests.gate.test_gate_orchestrator import build_world
from tests.gate.helpers import mint_envelope, coherent_stage11_children
from genome.validation.records import CigRecordStore

print("== G1/G2: two candidates a,b, stores supplied ==")
tmp = Path(tempfile.mkdtemp())
orch, somatic, cig_store, runner, inputs, registry = build_world(tmp, candidates=("a", "b"))
calls_before = len(getattr(runner, "calls", []) or [])
out = orch.run(inputs)
print("committed:", out.committed)
print("verdicts:", out.verdicts)
print("aggregate ids per result:", [r.aggregate_cig_id for r in out.results])
print("parent ids per result:", [r.parent_cig_id for r in out.results])
print("live cig_store aggregates:", list(cig_store.aggregates), "children:", list(cig_store.children))
print("live verify():", cig_store.verify())
for g in ("a", "b"):
    print(f"somatic {g}@1:", somatic.state(g, 1), somatic.get(g, 1)["validation"])
# stage-4 / child ids collide across candidates
for r in out.results:
    print(r.gene_id, "child_records:", r.aggregate["child_records"])
# would live exposure collide?
live = CigRecordStore()
r0, r1 = out.results
live.add_bundle(r0.aggregate, orch._bundle_children(r0))
try:
    live.add_bundle(r1.aggregate, orch._bundle_children(r1))
    print("second live add_bundle: ACCEPTED")
except Exception as e:
    print("second live add_bundle:", type(e).__name__, str(e)[:160])

print("\n== G2: explicit duplicate per-input parent ids -> any preflight rejection? ==")
tmp = Path(tempfile.mkdtemp())
orch, somatic, cig_store, runner, inputs, registry = build_world(tmp, candidates=("a", "b"))
import dataclasses
inputs = [dataclasses.replace(i, parent_cig_id="CIG-0007") for i in inputs]
n0 = len(runner.calls) if hasattr(runner, "calls") else None
try:
    out = orch.run(inputs)
    print("run with duplicate explicit parent ids: ACCEPTED; aggregates:",
          [r.aggregate_cig_id for r in out.results],
          "runner calls:", (len(runner.calls) if hasattr(runner, "calls") else "n/a"))
except Exception as e:
    print("rejected:", type(e).__name__, e)

print("\n== G1: version-2 candidate (version hard-coded to 1 in _commit_results) ==")
from tests.gate.helpers import proposal
from inheritance.gate import GateCandidateInput
tmp = Path(tempfile.mkdtemp())
orch, somatic, cig_store, runner, inputs, registry = build_world(tmp, candidates=("a",))
# replace candidate with a v2 envelope of the same gene; register v2 in somatic store
payload = proposal("a", "heat_and_place")
uri2 = registry.put("skill", "a", 2, payload)
env2 = copy.deepcopy(inputs[0].candidate_envelope)
env2["candidate"]["version"] = 2
env2["candidate"]["artifact"] = uri2
try:
    somatic.add_candidate(env2)
    print("somatic add v2 ok; states:", somatic.state("a", 1), somatic.state("a", 2))
except Exception as e:
    print("add v2 failed:", type(e).__name__, e)
# make v1 terminal so only v2 is 'candidate'
somatic.decide("a", 1, "rejected", ["CIG-0001"])
inp2 = dataclasses.replace(inputs[0], candidate_envelope=env2)
try:
    out = orch.run([inp2])
    print("v2 run committed:", out.committed, "states v1/v2:", somatic.state("a", 1), somatic.state("a", 2))
except Exception as e:
    print("v2 run raised:", type(e).__name__, str(e)[:200],
          "| states v1/v2:", somatic.state("a", 1), somatic.state("a", 2))
