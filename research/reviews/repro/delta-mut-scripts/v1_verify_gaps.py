"""V1 at 6014bc7: CigRecordStore.verify() key-vs-cig_id, no load_cig, truthiness."""
import copy
from genome.validation.loader import load_cig
from genome.validation.records import CigRecordStore
from tests.genome.test_records import full_store

def schema(rec):
    try: load_cig(copy.deepcopy(rec)); return "load_cig accepts"
    except Exception as e: return f"load_cig REJECTS"

cases = {}
s = full_store(); s._children["CIG-0042/S11-replay"]["cig_id"] = "CIG-0042/S11-ablation"
cases["a1 child internal cig_id != dict key"] = (s.verify(), schema(s._children["CIG-0042/S11-replay"]))
s = full_store(); s._aggregates["CIG-0042"]["cig_id"] = "CIG-0099"
cases["a2 aggregate internal cig_id != dict key"] = (s.verify(), schema(s._aggregates["CIG-0042"]))
# truthiness: aggregate stage pass int 1 + child stage_pass True (1 == True)
s = full_store(); s._aggregates["CIG-0042"]["stages"]["replay"]["pass"] = 1
cases["b1 aggregate stage pass = 1 (int)"] = (s.verify(), schema(s._aggregates["CIG-0042"]))
# schema-invalid stored child (missing required field) -> no load_cig rerun
s = full_store(); c = s._children["CIG-0042/S11-replay"]; del c["measurements"]["candidate_gene_id"]; c["measurements"]["candidate_gene_id"]=s._aggregates["CIG-0042"]["candidate_gene_id"]
s._children["CIG-0042/S11-replay"]["stage"] = "replay"; s._children["CIG-0042/S11-replay"]["extra_unknown_field"] = {"x": 1}
cases["b2 child with unknown top-level field"] = (s.verify(), schema(s._children["CIG-0042/S11-replay"]))
s = full_store(); s._aggregates["CIG-0042"]["verdict"] = "promote"; s._aggregates["CIG-0042"]["genome_id"] = 12345
cases["b3 aggregate genome_id = int"] = (s.verify(), schema(s._aggregates["CIG-0042"]))
# non-M0 shaped aggregate: stage pass "false" string (truthy) + verdict promote
s = full_store(); agg = s._aggregates["CIG-0042"]
print("full_store verdict:", agg["verdict"], {k: v["pass"] for k, v in agg["stages"].items()})
for k in agg["stages"]: agg["stages"][k]["pass"] = True
agg["verdict"] = "promote"
agg["stages"]["replay"]["pass"] = "false"
for ch in s._children.values():
    if ch["parent_cig_id"] == "CIG-0042" and ch["stage"] == "replay":
        ch["measurements"]["stage_pass"] = "false"
    elif ch["parent_cig_id"] == "CIG-0042": ch["measurements"]["stage_pass"] = True
cases["b4 M0 agg pass='false' + child stage_pass='false' + verdict promote"] = (s.verify(), schema(agg))
for k, (issues, sch) in cases.items():
    print(f"{k:70s} [{sch}] verify() -> {issues}")
