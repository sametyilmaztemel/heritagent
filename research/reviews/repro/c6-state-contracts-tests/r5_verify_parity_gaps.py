"""R5: CigRecordStore.verify() is documented to re-check "the same
invariants" add_bundle() enforces (defense in depth against white-box
corruption), but:
  (a) verify() keys every check on the DICT KEY, never on the record's own
      `cig_id`: a stored child/aggregate whose internal cig_id no longer
      matches its key passes verify() (add_bundle checks record['cig_id']);
  (b) verify() never re-runs load_cig(): schema-invalid stored state (e.g. a
      stage pass corrupted to the string "false", which is truthy) passes,
      because _verdict_issue/_stage_pass_issue use truthiness/bool();
  (c) [still-open L6] add_bundle() accepts M0 child stage_pass=1 / "true".

Run: cd pr28c && PYTHONPATH=. <venv314>/bin/python <this file>
"""
import sys

from genome.validation.loader import load_cig
from genome.validation.records import CigRecordStore
from tests.genome.test_records import aggregate_record, full_store, m0_children

results = {}

# (a1) child internal id diverges from its storage key
s = full_store()
s._children["CIG-0042/S11-replay"]["cig_id"] = "CIG-0042/S11-ablation"
results["child cig_id != key"] = s.verify()
# (a2) aggregate internal id diverges from its storage key
s = full_store()
s._aggregates["CIG-0042"]["cig_id"] = "CIG-0099"
results["aggregate cig_id != key"] = s.verify()
# (b) schema-invalid stage pass (truthy string) with promote verdict
s = full_store()
s._aggregates["CIG-0042"]["stages"]["replay"]["pass"] = "false"
try:
    load_cig(s._aggregates["CIG-0042"])
    schema = "load_cig accepts"
except Exception as exc:  # noqa: BLE001
    schema = f"load_cig REJECTS ({type(exc).__name__})"
results[f"stage pass='false' [{schema}]"] = s.verify()
# (b2) child stage_pass corrupted to the string "false" (aggregate pass True)
s = full_store()
s._children["CIG-0042/S11-replay"]["measurements"]["stage_pass"] = "false"
results["child stage_pass='false'"] = s.verify()

# (c) L6 still open at c6ce4c1: add_bundle accepts non-boolean stage_pass
accepted = []
for bad in (1, "true"):
    store = CigRecordStore()
    children = m0_children()
    children[0]["measurements"]["stage_pass"] = bad
    try:
        store.add_bundle(aggregate_record(), children)
        accepted.append(bad)
    except Exception:  # noqa: BLE001
        pass

for name, issues in results.items():
    print(f"{name:45s} verify() -> {issues}")
print("add_bundle accepted non-bool stage_pass values:", accepted)
bad = all(v == [] for v in results.values()) or accepted
print("DEFECT REPRODUCED" if bad else "not reproduced")
sys.exit(0 if bad else 1)
