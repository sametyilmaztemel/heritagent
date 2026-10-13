"""G1 extra: (i) v2 evaluated while v1 still candidate -> which version is decided?
(ii) batch partial commit when 2nd candidate's somatic preflight fails."""
import tempfile, copy, dataclasses
from pathlib import Path
from tests.gate.test_gate_orchestrator import build_world
from tests.gate.helpers import proposal

tmp = Path(tempfile.mkdtemp())
orch, somatic, cig_store, runner, inputs, registry = build_world(tmp, candidates=("a",))
uri2 = registry.put("skill", "a", 2, proposal("a", "heat_and_place"))
env2 = copy.deepcopy(inputs[0].candidate_envelope)
env2["candidate"]["version"] = 2; env2["candidate"]["artifact"] = uri2
somatic.add_candidate(env2)
out = orch.run([dataclasses.replace(inputs[0], candidate_envelope=env2)])
print("(i) evaluated v2; committed:", out.committed,
      "| state a@1:", somatic.state("a", 1), "| state a@2:", somatic.state("a", 2))

tmp = Path(tempfile.mkdtemp())
orch, somatic, cig_store, runner, inputs, registry = build_world(tmp, candidates=("a", "b"))
somatic.decide("b", 1, "rejected", ["CIG-0099"])   # b no longer candidate
try:
    orch.run(inputs)
    print("(ii) run ok")
except Exception as e:
    print("(ii) run raised:", type(e).__name__, e)
print("(ii) after failure: state a@1:", somatic.state("a", 1), "| b@1:", somatic.state("b", 1),
      "| live cig aggregates:", list(cig_store.aggregates))
