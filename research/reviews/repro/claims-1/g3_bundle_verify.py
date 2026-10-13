"""G3 repro: CigRecordStore.add_bundle()/verify() coherence gaps @4ca5517."""
import tempfile, copy
from pathlib import Path
from tests.gate.test_gate_orchestrator import build_world
from genome.validation.records import CigRecordStore

tmp = Path(tempfile.mkdtemp())
orch, somatic, cig_store, runner, inputs, registry = build_world(tmp, candidates=("a",))
out = orch.run(inputs)
R = out.results[0]
AGG, KIDS = R.aggregate, orch._bundle_children(R)
print("baseline verdict:", AGG["verdict"], "ids:", AGG["child_records"])

def attempt(name, agg, kids):
    s = CigRecordStore()
    try:
        s.add_bundle(agg, kids)
        print(f"[{name}] add_bundle ACCEPTED; verify() -> {s.verify()}")
    except Exception as e:
        print(f"[{name}] add_bundle REJECTED: {type(e).__name__}: {str(e)[:140]}")

attempt("baseline", AGG, KIDS)

# (a) five children all stage 'replay' (duplicate stage replaces others)
agg = copy.deepcopy(AGG); kids = copy.deepcopy(KIDS)
for k in kids:
    k["stage"] = "replay"; k["measurements"]["stage_pass"] = agg["stages"]["replay"]["pass"]
attempt("a: 5x replay children", agg, kids)

# (b) one child id/seed mismatch (id S11, seed 99) -- schema-valid
agg = copy.deepcopy(AGG); kids = copy.deepcopy(KIDS)
kids[0]["seed"] = 99
attempt("b: child seed!=id seed", agg, kids)

# (c) self-consistent but bundle-incoherent seeds (S11 and S12 in one bundle)
agg = copy.deepcopy(AGG); kids = copy.deepcopy(KIDS)
old = kids[1]["cig_id"]; new = old.replace("/S11-", "/S12-")
kids[1]["cig_id"] = new; kids[1]["seed"] = 12
agg["child_records"] = [new if c == old else c for c in agg["child_records"]]
attempt("c: mixed bundle seeds", agg, kids)

# (d) promote with one failed stage (children made coherent with the failed stage)
agg = copy.deepcopy(AGG); kids = copy.deepcopy(KIDS)
agg["stages"]["ablation"]["pass"] = False
for k in kids:
    if k["stage"] == "ablation": k["measurements"]["stage_pass"] = False
assert agg["verdict"] == "promote"
attempt("d: promote with failed stage", agg, kids)

# (e) reject with all stages passing
agg = copy.deepcopy(AGG); kids = copy.deepcopy(KIDS); agg["verdict"] = "reject"
attempt("e: reject with all pass", agg, kids)

# (f) wrong id<->stage mapping: swap ids of replay and ablation children
agg = copy.deepcopy(AGG); kids = copy.deepcopy(KIDS)
r = next(k for k in kids if k["stage"] == "replay"); a = next(k for k in kids if k["stage"] == "ablation")
r["cig_id"], a["cig_id"] = a["cig_id"], r["cig_id"]
attempt("f: replay child carries -ablation id", agg, kids)

# (g) arbitrary non-<parent>/S<seed>-<stage> ids, no seed digits
agg = copy.deepcopy(AGG); kids = copy.deepcopy(KIDS)
newids = []
for i, k in enumerate(kids):
    k["cig_id"] = f"{AGG['cig_id']}/Sx{i}"; newids.append(k["cig_id"])
agg["child_records"] = newids
attempt("g: ids without seed digits / stage suffix", agg, kids)

# (h) missing stage_pass on a child whose aggregate stage failed
agg = copy.deepcopy(AGG); kids = copy.deepcopy(KIDS)
agg["stages"]["replay"]["pass"] = False; agg["verdict"] = "reject"
for k in kids:
    if k["stage"] == "replay": del k["measurements"]["stage_pass"]
attempt("h: missing child stage_pass (agg stage False)", agg, kids)

# (i) verify() parity after white-box corruption of a consistent store
print("\n-- verify() white-box corruption --")
def corrupt(name, fn):
    s = CigRecordStore(); s.add_bundle(copy.deepcopy(AGG), copy.deepcopy(KIDS)); fn(s)
    print(f"[{name}] verify() -> {s.verify()}")
cid = AGG["cig_id"]
first = AGG["child_records"][0]
corrupt("verdict->reject w/ all pass", lambda s: s._aggregates[cid].__setitem__("verdict", "reject"))
corrupt("stage pass flipped (agg)", lambda s: s._aggregates[cid]["stages"]["replay"].__setitem__("pass", False))
corrupt("child candidate_gene_id", lambda s: s._children[first]["measurements"].__setitem__("candidate_gene_id", "zzz"))
corrupt("child stage_pass flipped", lambda s: s._children[first]["measurements"].__setitem__("stage_pass", False))
corrupt("child stage -> duplicate stage", lambda s: s._children[AGG["child_records"][1]].__setitem__("stage", "replay"))
corrupt("child seed != id seed", lambda s: s._children[first].__setitem__("seed", 99))
