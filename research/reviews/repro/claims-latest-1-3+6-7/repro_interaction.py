"""L3 repro: durable interaction evidence in BOTH stage-4 children of a live gate run."""
import json, pathlib, tempfile
from trajectory.storage.canonical import canonical_json, sha256_hex
from tests.gate.test_gate_orchestrator import build_world
from inheritance.interaction import InteractionAllocation
from inheritance.runner import EvaluationTask
with tempfile.TemporaryDirectory() as d:
    orch, somatic, cig_store, runner, inputs, registry = build_world(pathlib.Path(d), candidates=("a", "b"))
    orch.run(inputs)
    kids = cig_store.children
    s4 = {k: v for k, v in kids.items() if k.endswith("interaction_regression")}
    pes = [v["measurements"]["interaction"]["pair_evidence"] for v in s4.values()]
    print("stage4 children:", sorted(s4), "pair counts:", [len(p) for p in pes])
    recs = [p[0]["canonical_evidence"] for p in pes]
    print("canonical records identical across children:", canonical_json(recs[0]) == canonical_json(recs[1]))
    print("hash == sha256(canonical_json(record)) for each:", [p[0]["evidence_sha256"] == sha256_hex(canonical_json(p[0]["canonical_evidence"]).encode()) for p in pes])
    r = recs[0]
    print("record keys:", sorted(r))
    print("allocation keys:", sorted(r["allocation"]), "task[0]:", r["allocation"]["tasks"][0])
    conds = {}
    for o in r["paired_outcomes"]:
        conds.setdefault(o["condition"], []).append(o["expressed_gene_ids"])
    print("outcome keys:", sorted(r["paired_outcomes"][0]), "n=", len(r["paired_outcomes"]))
    print("expressed both[0]/neither[0]:", conds["both"][0], conds["neither"][0])
    print("locked_settings:", r["locked_settings"])
# payload digest sensitivity
mk = lambda goal: InteractionAllocation(tasks=tuple(EvaluationTask(task_id=f"ix-{i}", family="f", payload={"goal": goal, "n": [1, {"x": 2}]}, bank_id="bank") for i in range(10)))
a, b = mk("g1").allocation_metadata(), mk("g2").allocation_metadata()
print("payload digest differs when payload differs:", a["tasks"][0]["payload_sha256"] != b["tasks"][0]["payload_sha256"])
print("payload digest deterministic:", a["tasks"][0]["payload_sha256"] == mk("g1").allocation_metadata()["tasks"][0]["payload_sha256"])
