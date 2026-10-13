"""R6: interaction pair episodes use the BATCH default seed
(self._mining_seed), not the candidates' resolved per-candidate mining
seeds, while every other #12 stage uses the per-candidate seed. Pair
evidence embedded in <parent>/S11-interaction_regression was generated at
seed 0. Run from the PR checkout with PYTHONPATH=."""
import tempfile, pathlib
from dataclasses import replace
from tests.gate.test_gate_orchestrator import build_world
from inheritance.gate import CigGateOrchestrator

tmp = pathlib.Path(tempfile.mkdtemp())
orch, somatic, cig, runner, inputs, reg = build_world(tmp, candidates=("a", "b"))
# candidates carry their own seed 11 (their #11 children are S11); the
# orchestrator's batch default is 0
inputs = [replace(i, mining_seed=11) for i in inputs]
orch0 = CigGateOrchestrator(
    registry=reg, base_config=orch._base_config, runner=runner,
    budgets=orch._budgets, parent_cig_id="CIG-0100", mining_seed=0,
    token_counter=orch._token_counter,
    interaction_allocation=orch._interaction_allocation,
    somatic_store=somatic, cig_record_store=cig)
out = orch0.run(inputs)
for r in out.results:
    s4 = r.stage4_child
    pe = s4["measurements"]["interaction"]["pair_evidence"][0]
    print(r.gene_id, s4["cig_id"], "child seed:", s4["seed"],
          "| regression seed:", s4["measurements"]["regression"]["eval_generation_settings"]["seed"],
          "| pair-evidence seed:", pe["canonical_evidence"]["locked_settings"]["seed"])
print("GateOutcome.mining_seed:", out.mining_seed)
