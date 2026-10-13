"""Independent repro for PR #28 @4ca5517 residual groups 4/5/6. Read-only; tmp dirs only."""
import copy, tempfile, pathlib, dataclasses
from tests.gate.test_gate_orchestrator import build_world
import inheritance.gate as gate_mod
from genome.validation.loader import load_cig
from genome.validation.errors import GenomeValidationError

SP = pathlib.Path("<scratch>/out/repro/claims-2/tmp")
SP.mkdir(parents=True, exist_ok=True)

def world(**kw):
    d = pathlib.Path(tempfile.mkdtemp(dir=SP))
    return build_world(d, **kw)

def strip_stores(orch):
    orch._somatic_store = None; orch._cig_record_store = None

def mutate_child(inp, stage, fn):
    ev = inp.replay_evaluation if stage == "replay" else inp.ablation_evaluation
    child = copy.deepcopy(ev.child_record); fn(child)
    new = dataclasses.replace(ev, _child_record=child)
    field = "replay_evaluation" if stage == "replay" else "ablation_evaluation"
    return dataclasses.replace(inp, **{field: new})

print("=== GROUP 4: #11 evidence preflight ===")
cases = {
  "schema_invalid_extra_key": lambda c: c.__setitem__("bogus_extra", 1),
  "wrong_suffix_id": lambda c: c.__setitem__("cig_id", c["cig_id"] + "X"),
  "wrong_seed_in_id": lambda c: c.__setitem__("cig_id", c["cig_id"].replace("/S11-", "/S99-")),
}
for name, fn in cases.items():
    for with_stores in (False, True):
        orch, som, cig, runner, inputs, reg = world(candidates=("a",))
        if not with_stores: strip_stores(orch)
        inputs = [mutate_child(inputs[0], "replay", fn)]
        try:
            schema_ok = True; load_cig(copy.deepcopy(inputs[0].replay_evaluation.child_record))
        except GenomeValidationError: schema_ok = False
        try:
            out = orch.run(inputs)
            res = f"RUN OK verdict={out.results[0].verdict} committed={out.committed} live_children={sorted(cig._children) if with_stores else '-'}"
        except Exception as e:
            res = f"RAISED {type(e).__name__}: {str(e)[:110]}"
        print(f"{name:26s} stores={with_stores!s:5s} child_schema_valid={schema_ok!s:5s} runner_calls={len(runner.calls):3d} -> {res}")

print("\n=== GROUP 5: stage-4 persistence / load_cig ===")
orch, som, cig, runner, inputs, reg = world(candidates=("a",))
strip_stores(orch)
seen = []
orig = gate_mod.load_cig
gate_mod.load_cig = lambda d: (seen.append(d.get("cig_id")), orig(d))[1]
out = orch.run(inputs)
gate_mod.load_cig = orig
s4 = out.results[0].stage4_child
print("load_cig calls inside gate.py (no stores):", seen)
print("stage4 cig_id:", s4["cig_id"], "validated_by_gate_load_cig:", s4["cig_id"] in seen)
print("stage4 measurement keys:", sorted(s4["measurements"]))
print("stage4 regression keys:", sorted(s4["measurements"]["regression"]))
print("'eval_generation_settings' anywhere in stage4:", "eval_generation_settings" in repr(s4))
reg_eval = orch._last.reg_results["a"]
print("RegressionEvaluation fields:", [f.name for f in dataclasses.fields(reg_eval)])
print("gen in-scope child has settings:", "eval_generation_settings" in out.results[0].stage4_child["measurements"] or "eval_generation_settings" in repr(orch._last.gen_results["a"].in_scope_child))
# A schema-invalid stage-4 child is returned without error when no stores supplied:
orch2, som2, cig2, runner2, inputs2, reg2 = world(candidates=("a",))
strip_stores(orch2)
orig_b = orch2._build_stage4_child
def bad(*a, **k):
    c = orig_b(*a, **k); c["bogus_extra"] = True; return c
orch2._build_stage4_child = bad
try:
    o2 = orch2.run(inputs2); r = "RETURNED result with schema-invalid stage4 child; verdict=" + o2.results[0].verdict
except Exception as e:
    r = f"RAISED {type(e).__name__}"
print("injected schema-invalid stage4, no stores:", r)
orch3, som3, cig3, runner3, inputs3, reg3 = world(candidates=("a",))
orig_c = orch3._build_stage4_child
orch3._build_stage4_child = lambda *a, **k: {**orig_c(*a, **k), "bogus_extra": True}
try:
    orch3.run(inputs3); r = "RETURNED"
except Exception as e:
    r = f"RAISED {type(e).__name__} (after {len(runner3.calls)} runner calls)"
print("injected schema-invalid stage4, WITH stores:", r)

print("\n=== GROUP 6: interaction regulation / canonical order ===")
from inheritance.interaction import InteractionEvaluator, InteractionAllocation
from runtime.regulation import Condition, Predicate
from tests.gate.conftest import interaction_tasks
from tests.gate.helpers import BUDGETS
from inheritance.overlay import evaluation_settings
import inspect
orch, som, cig, runner, inputs, reg = world(candidates=("a", "b"))
print("InteractionEvaluator.__init__ params:", list(inspect.signature(InteractionEvaluator.__init__).parameters))
cond = Condition(all_of=(Predicate("consecutive_failures", ">=", 2),))
ie = InteractionEvaluator(registry=reg, base_config=orch._base_config, runner=runner, budgets=BUDGETS)
ea, eb = inputs[0].candidate_envelope, inputs[1].candidate_envelope
both, neither, gids = ie._build_pair_configs(ea, eb)
for g in gids:
    sk = [s for s in both.skills if s.gene_id == g][0]
    print(f"both-arm gene {g}: skill.condition={sk.condition} in_regulation={g in both.regulation} is_active(counters={{}})={both.is_active(g, {})}")
# compare with the stage-3/4A path which DOES carry regulation
from inheritance.overlay import build_evaluation_configs
cfg = build_evaluation_configs(orch._base_config, ea, reg, target_regulation=cond)
print("stage-3/4A overlay with target_regulation: is_active('a',{}) =", cfg.with_trait.is_active("a", {}))
# orchestrator path: supply regulation map; interaction still ignores it
src = inspect.getsource(gate_mod.CigGateOrchestrator.run)
print("orchestrator passes target_regulation to InteractionEvaluator:", "target_regulation" in src.split("InteractionEvaluator(")[1].split(")")[0])
# order
alloc = InteractionAllocation(tasks=interaction_tasks(10))
locked = evaluation_settings(max_tokens=BUDGETS.max_tokens_per_request, seed=11)
for gid in ("a", "b"):
    for i in range(10):
        runner.set(f"ix-{i}", "both", True, ("a", "b")); runner.set(f"ix-{i}", "neither", True, ())
e_ab = ie.evaluate_pair(ea, eb, alloc, locked)
e_ba = ie.evaluate_pair(eb, ea, alloc, locked)
b_ab, _, _ = ie._build_pair_configs(ea, eb); b_ba, _, _ = ie._build_pair_configs(eb, ea)
print("gene_ids (a,b):", e_ab.gene_ids, " (b,a):", e_ba.gene_ids)
print("evidence_sha256 equal across caller order:", e_ab.evidence_sha256 == e_ba.evidence_sha256)
n = len(orch._base_config.skills)
print("both-arm appended skill order (a,b):", [s.gene_id for s in b_ab.skills[n:]], " (b,a):", [s.gene_id for s in b_ba.skills[n:]])
# orchestrator: rank order drives argument order
orch_r, *_rest = world(candidates=("a", "b"), ranks={"a": 1, "b": 0})
strip_stores(orch_r)
out_r = orch_r.run(_rest[3])
for k, ev in orch_r._last.pair_evidence.items():
    print("orchestrator ranks a=1,b=0 -> pair key", k, "evidence.gene_ids", ev.gene_ids)
