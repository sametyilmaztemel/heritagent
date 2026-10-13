"""R7: target regulation is an orchestrator-level side input
(target_regulation_by_gene). It is not bound to the #11 evidence, not
cross-checked with the regulation the #11 replay/ablation ran under, and
not persisted in ANY #12 child. Here #11 evidence (constitutive) is
combined with regulated stage-3/4 runs; nothing in the committed records
reveals that the phenotype differed across stages.
Run from the PR checkout with PYTHONPATH=."""
import json, tempfile, pathlib
from tests.gate.test_gate_orchestrator import build_world
from inheritance.gate import CigGateOrchestrator
from runtime.regulation import Condition, Predicate

tmp = pathlib.Path(tempfile.mkdtemp())
orch, somatic, cig, runner, inputs, reg = build_world(tmp, candidates=("a",))
cond = Condition(all_of=(Predicate("consecutive_failures", ">=", 2),))
seen = []
real = runner.run_episode
def spy(task, config, budgets, condition, settings=None):
    seen.append((task.task_id, condition, repr(config.regulation.get("a"))))
    return real(task, config, budgets, condition, settings)
runner.run_episode = spy
orch_reg = CigGateOrchestrator(
    registry=reg, base_config=orch._base_config, runner=runner,
    budgets=orch._budgets, parent_cig_id="CIG-0007", mining_seed=11,
    token_counter=orch._token_counter,
    interaction_allocation=orch._interaction_allocation,
    somatic_store=somatic, cig_record_store=cig,
    target_regulation_by_gene={"a": cond})
out = orch_reg.run(inputs)
with_trait = [s for s in seen if s[1] == "with_trait"]
print("stage-3/4 with_trait regulation for a:", {s[2] for s in with_trait})
blob = json.dumps({cid: c for cid, c in cig.children.items()}, default=str)
print("verdict:", out.verdicts, "| committed:", out.committed)
print("'consecutive_failures' anywhere in committed children:",
      "consecutive_failures" in blob,
      "| 'regulation' anywhere:", "regulation" in blob)
