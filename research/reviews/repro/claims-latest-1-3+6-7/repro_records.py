"""L1 + L6 repro against pinned c6ce4c1 CigRecordStore."""
import copy
from genome.validation.errors import RecordConsistencyError
from genome.validation.records import CigRecordStore
from tests.genome.test_records import aggregate_record, m0_children, full_store

def try_bundle(label, mutate):
    store = CigRecordStore()
    agg = aggregate_record(); ch = m0_children()
    mutate(agg, ch)
    try:
        store.add_bundle(agg, ch)
        print(f"[ACCEPTED] add_bundle {label}; verify()={store.verify()}")
    except RecordConsistencyError as e:
        print(f"[rejected] add_bundle {label}: {str(e)[:160]}")

def try_verify(label, mutate):
    store = full_store()
    mutate(store._children["CIG-0042/S11-replay"]["measurements"])
    print(f"verify {label}: {'FLAGGED' if store.verify() else 'CLEAN (accepted)'} {store.verify()[:1]}")

print("== L1 ==")
try_bundle("baseline valid", lambda a, c: None)
try_bundle("missing stage_pass", lambda a, c: c[2]["measurements"].pop("stage_pass"))
try_bundle("stage_pass=False vs agg True", lambda a, c: c[2]["measurements"].__setitem__("stage_pass", False))
try_bundle("wrong candidate_gene_id", lambda a, c: c[1]["measurements"].__setitem__("candidate_gene_id", "x"))
try_bundle("missing candidate_gene_id", lambda a, c: c[1]["measurements"].pop("candidate_gene_id"))
try_verify("missing stage_pass", lambda m: m.pop("stage_pass"))
try_verify("stage_pass False", lambda m: m.__setitem__("stage_pass", False))
try_verify("wrong candidate_gene_id", lambda m: m.__setitem__("candidate_gene_id", "x"))
try_verify("missing candidate_gene_id", lambda m: m.pop("candidate_gene_id"))

print("== L6 ==")
for bad in (1, "true", "false", [0], 0.5, {"x": 1}):
    try_bundle(f"stage_pass={bad!r} (agg pass True)", lambda a, c, bad=bad: c[2]["measurements"].__setitem__("stage_pass", bad))
    try_verify(f"stage_pass={bad!r}", lambda m, bad=bad: m.__setitem__("stage_pass", bad))
# falsy non-bool with aggregate failing stage
def fail_stage(a, c, bad):
    a["stages"]["ablation"]["pass"] = False; a["verdict"] = "reject"
    c[1]["measurements"]["stage_pass"] = bad
for bad in (0, "", None, []):
    try_bundle(f"stage_pass={bad!r} (agg ablation pass False)", lambda a, c, bad=bad: fail_stage(a, c, bad))
