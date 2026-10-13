"""G6/G7/G9 repro at c6ce4c1."""
import copy, tempfile
from pathlib import Path
from tests.gate.test_gate_orchestrator import build_world
from runtime.regulation import Condition, Predicate
from inheritance.interaction import InteractionEvaluator
from inheritance.gate import CigGateOrchestrator, VACUOUS_INTERACTION_REASON
from trajectory.storage.canonical import canonical_json, sha256_hex
def T(): return Path(tempfile.mkdtemp())

print("== G6: gate-wired pair overlay: regulation preserved + canonical insertion order")
orch, somatic, cig, runner, inputs, registry = build_world(T(), candidates=("z", "a"))
cond = Condition(all_of=(Predicate(metric="retry_count", op=">=", value=1.0),))
seen = []
orig = runner.run_episode
def spy(task, config, budgets, condition, settings=None):
    if condition == "both": seen.append(config)
    return orig(task, config, budgets, condition, settings)
runner.run_episode = spy
orch._target_regulation_by_gene = {"z": cond}
# mark both expressed in both arm
for i in range(10): runner.set(f"ix-{i}", "both", True, ("a","z"))
out = orch.run(inputs)
cfg = seen[0]
print("  both-arm tail gene order", [s.gene_id for s in cfg.skills[-2:]])
print("  regulation z preserved:", cfg.regulation.get("z") == cond, "| skill.condition z:", [s.condition for s in cfg.skills if s.gene_id=="z"][0] == cond, "| a constitutive:", "a" not in cfg.regulation)

print("== G7: canonical evidence identical in both children + hash recomputable")
ch = {r.gene_id: r.stage4_child for r in out.results}
ea = ch["a"]["measurements"]["interaction"]["pair_evidence"][0]
ez = ch["z"]["measurements"]["interaction"]["pair_evidence"][0]
print("  identical canonical_evidence:", canonical_json(ea["canonical_evidence"]) == canonical_json(ez["canonical_evidence"]))
print("  recomputed sha == stored:", sha256_hex(canonical_json(ea["canonical_evidence"]).encode()) == ea["evidence_sha256"] == ez["evidence_sha256"])
ce = ea["canonical_evidence"]
print("  keys", sorted(ce), "| gene_ids", ce["gene_ids"], "| settings", ce["locked_settings"])
print("  task[0]", ce["allocation"]["tasks"][0], "| outcome[0]", ce["paired_outcomes"][0])
# persisted in live store too?
live = cig.children
print("  live store children carry same record:", all(
    canonical_json(live[c["cig_id"]]["measurements"]["interaction"]["pair_evidence"][0]["canonical_evidence"]) == canonical_json(ce)
    for c in ch.values()))

print("== G9: validate_token_counter called at construction; vacuous reason")
class Bad: identity = "bad"; count_tokens = lambda self, t: -1
try:
    CigGateOrchestrator(registry=registry, base_config=orch._base_config, runner=runner, budgets=orch._budgets, token_counter=Bad())
    print("  NOT REJECTED")
except Exception as e: print("  bad counter rejected at construction:", type(e).__name__)
orch, somatic, cig, runner, inputs, registry = build_world(T(), candidates=("a",))
out = orch.run(inputs)
print("  vacuous reason:", out.results[0].stage4_child["measurements"]["interaction"]["reason"] == VACUOUS_INTERACTION_REASON)
