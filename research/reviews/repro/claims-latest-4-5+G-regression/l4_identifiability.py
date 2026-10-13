"""L4 repro: per-cell required-pair identifiability."""
import tempfile, json
from pathlib import Path
from tests.gate.test_gate_orchestrator import build_world

def run_case(label, **kw):
    tmp = Path(tempfile.mkdtemp())
    post = kw.pop("post", None)
    world = build_world(tmp, **kw)
    orch, somatic, cig_store, runner, inputs, registry = world
    if post: post(runner)
    outcome = orch.run(inputs)
    n = len(runner.calls)
    grid = orch.sensitivity_recomputation(inputs[0].candidate_envelope["candidate"]["gene_id"])
    assert len(runner.calls) == n
    print(f"== {label}: verdicts={outcome.verdicts} primary_eligible={grid['primary_eligible']}")
    for k, c in grid["cells"].items():
        print(f"  {k}: elig={c['hypothetical_eligible']} ident={c['identifiable']} "
              f"missing={c.get('missing_interaction_pairs')} promote={c.get('hypothetical_promote')}")

# Case A: discriminating case old 'newly' rule would mark not_identifiable:
# a primary-eligible with regression drop 0.025 (passes tau_r=0.03, fails 0.02);
# b fails primary ablation (0.04) but has drop 0.0. Cell tau_c=0.03|tau_r=0.02:
# hyp eligible = [b] only (newly) -> should be identifiable.
def fix_b(runner):
    for i in range(40):
        runner.set(f"treg-b-{i}", "with_trait", True, ("b",))
run_case("A newly-eligible-alone-while-primary-drops-out",
         candidates=("a", "b"), ablation_deltas={"a": 0.9, "b": 0.04},
         treg_success_count=39, post=fix_b)

# Case B: three candidates, a,b primary-eligible (pair collected), c newly at tau_c=0.03
run_case("B three-candidates",
         candidates=("a", "b", "c"), ablation_deltas={"a": 0.9, "b": 0.8, "c": 0.04})
