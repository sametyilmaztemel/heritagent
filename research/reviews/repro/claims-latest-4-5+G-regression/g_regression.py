"""G1-G9 regression repro at c6ce4c1."""
import copy, tempfile, dataclasses
from pathlib import Path
from tests.gate.test_gate_orchestrator import build_world, _WordCounter
from tests.gate.helpers import coherent_stage11_children, proposal, MINING_SEED, BUDGETS, ScriptedGateRunner, interaction_tasks
from inheritance.gate import GateCandidateInput, GateInputError, EvidenceCoherenceError, CigGateOrchestrator
from genome.validation.loader import load_cig
from trajectory.storage.canonical import canonical_json, sha256_hex

def T(): return Path(tempfile.mkdtemp())
def rebuild(inp, **kw):
    d = {f.name: getattr(inp, f.name) for f in dataclasses.fields(inp)}
    d.update(kw); return GateCandidateInput(**d)

print("== G1: live add_bundle + version derived from envelope (v2 candidate)")
orch, somatic, cig, runner, inputs, registry = build_world(T(), candidates=("a",))
env = copy.deepcopy(inputs[0].candidate_envelope)
env["candidate"]["version"] = 2
env["candidate"]["artifact"] = registry.put("skill", "a", 2, proposal("a"))
somatic.add_candidate(env)
inp2 = rebuild(inputs[0], candidate_envelope=env)
out = orch.run([inp2])
r = out.results[0]
print("  candidate_version", r.candidate_version, "state v1", somatic.state("a",1), "state v2", somatic.state("a",2))
print("  live aggregates", list(cig.aggregates), "children", len(cig.children)); cig.require_consistent(); print("  require_consistent OK")

print("== G2: duplicate parent via batch fallback rejected pre-episode")
orch, somatic, cig, runner, inputs, registry = build_world(T(), candidates=("a","b"))
fb = [rebuild(i, parent_cig_id=None) for i in inputs]
n = len(runner.calls)
try: orch.run(fb); print("  NOT REJECTED")
except GateInputError as e: print("  rejected:", str(e)[:90], "| new calls", len(runner.calls)-n)

print("== G4: preflight - wrong ablation child id / seed mismatch / load_cig fail -> zero calls")
from tests.gate.test_gate_orchestrator import _replay_with_child
for label, mut in [("ablation id", lambda c: c.__setitem__("cig_id", c["cig_id"]+"x")),
                   ("seed", lambda c: c.__setitem__("seed", 99)),
                   ("load_cig", lambda c: c.pop("measurements"))]:
    orch, somatic, cig, runner, inputs, registry = build_world(T(), candidates=("a",))
    ab = inputs[0].ablation_evaluation
    ch = copy.deepcopy(ab.child_record); mut(ch)
    d = {f.name: getattr(ab, f.name) for f in dataclasses.fields(ab)}; d["_child_record"] = ch
    bad = rebuild(inputs[0], ablation_evaluation=type(ab)(**d))
    n = len(runner.calls)
    try: orch.run([bad]); print("  ", label, "NOT REJECTED")
    except EvidenceCoherenceError as e: print("  ", label, "rejected; new calls", len(runner.calls)-n, "|", str(e)[:80])

print("== G5: no-store run -> stage4 load_cig valid + settings persisted; per-candidate seed 5")
orch, somatic, cig, runner, inputs, registry = build_world(T(), candidates=("a","b"), commit_stores=False)
new = []
for inp, seed in zip(inputs, (5, 6)):
    gid = inp.candidate_envelope["candidate"]["gene_id"]
    parent = inp.parent_cig_id
    rp, ab = coherent_stage11_children(gid, parent=parent, replay_pass=True, ablation_pass=True, seed=seed)
    new.append(rebuild(inp, replay_evaluation=rp, ablation_evaluation=ab, mining_seed=seed))
out = orch.run(new)
for r in out.results:
    s4 = r.stage4_child; load_cig(copy.deepcopy(s4))
    pe = s4["measurements"]["interaction"]["pair_evidence"]
    print("  ", r.gene_id, s4["cig_id"], "reg settings", s4["measurements"]["regression"]["eval_generation_settings"],
          "| pair locked seed", pe[0]["canonical_evidence"]["locked_settings"]["seed"] if pe else None, "committed", out.committed)

print("== G6: pair overlay keeps regulation, canonical order")
from runtime.regulation import Condition, Predicate
from inheritance.interaction import InteractionEvaluator
orch, somatic, cig, runner, inputs, registry = build_world(T(), candidates=("z","a"))
cond = Condition()
import inspect; print("  Predicate sig", inspect.signature(Predicate))
