"""T2 demo: orchestrator wired with a frozen target regulation for 'a';
record whether stage-3 / stage-4A / stage-4C with-trait configs carry it."""
import tempfile
from pathlib import Path
from runtime.regulation import Condition, Predicate
from tests.gate.test_gate_orchestrator import build_world
orch, somatic, cig, runner, inputs, registry = build_world(
    Path(tempfile.mkdtemp()), candidates=("a", "b"), commit_stores=False)
gid = inputs[0].candidate_envelope["candidate"]["gene_id"]
cond = Condition(all_of=(Predicate("consecutive_failures", ">=", 2),))
orch._target_regulation_by_gene = {gid: cond}
seen = {}
orig = runner.run_episode
def spy(task, config, budgets, condition, settings):
    stage = task.task_id.split("-")[0]
    if condition in ("with_trait", "both") and (stage in ("ix",) or f"-{gid}-" in task.task_id):
        seen.setdefault((stage, condition), set()).add(config.regulation.get(gid) is cond)
    return orig(task, config, budgets, condition, settings)
runner.run_episode = spy
orch.run(inputs)
for k in sorted(seen): print(k, "regulation preserved:", seen[k])
