"""Third-reviewer repro for PR #28 @ 4ca5517, groups 7/8/9. Read-only."""
import json, tempfile, pathlib
from tests.gate.test_gate_orchestrator import build_world, _WordCounter
from inheritance.gate import CigGateOrchestrator

def tmp():
    return pathlib.Path(tempfile.mkdtemp())

print("=== GROUP 7: durable interaction evidence ===")
orch, somatic, cig, runner, inputs, reg = build_world(tmp(), candidates=("a", "b"))
out = orch.run(inputs)
c = [r.stage4_child for r in out.results]
ix = [x["measurements"]["interaction"] for x in c]
print("stage4 child interaction block (a):", json.dumps(ix[0], sort_keys=True))
pe = ix[0]["pair_evidence"][0]
print("persisted pair-evidence keys:", sorted(pe))
for f in ("gene_ids","allocation","task_ids","families","bank_id","paired_outcomes",
          "trajectory_id","expressed_gene_ids","locked_settings","seed","stop"):
    print(f"  field {f!r} present in persisted child JSON:", f in json.dumps(c[0]))
print("hashes identical across children:", ix[0]["pair_evidence"][0]["evidence_sha256"] == ix[1]["pair_evidence"][0]["evidence_sha256"])
print("children byte-identical pair blocks (excluding other_gene_id/retained):",
      {k: v for k, v in ix[0]["pair_evidence"][0].items() if k not in ("other_gene_id","retained")}
      == {k: v for k, v in ix[1]["pair_evidence"][0].items() if k not in ("other_gene_id","retained")})
ev = list(orch._last.pair_evidence.values())[0]
print("full evidence lives only in orch._last (in-memory _LastRun):", type(orch._last).__name__,
      "episodes:", len(ev.episodes))
print("GateOutcome public attrs:", [a for a in dir(out) if not a.startswith("_")])
print("allocation_metadata (hash domain):", ev.allocation)
print("hash-domain drops expressed_gene_ids although episodes record them:",
      "expressed_gene_ids" in ev.episodes[0])
# caller order -> gene_ids order
orch2, *_rest = build_world(tmp(), candidates=("b", "a"))
out2 = orch2.run(_rest[3])
ev2 = list(orch2._last.pair_evidence.values())[0]
print("gene_ids order with candidates=(a,b):", ev.gene_ids, " with (b,a):", ev2.gene_ids,
      " hash equal:", ev.evidence_sha256 == ev2.evidence_sha256)
print("live CIG store children after commit:", len(cig.children), "aggregates:", len(cig.aggregates))

print("\n=== GROUP 8: sensitivity composition ===")
# b: ablation 0.04 -> fails primary tau_c=0.05 but would pass tau_c=0.03
orch, somatic, cig, runner, inputs, reg = build_world(tmp(), candidates=("a", "b"),
                                                       ablation_deltas={"a": 0.5, "b": 0.04})
out = orch.run(inputs)
print("primary verdicts:", out.verdicts, " pair evidence keys:", list(orch._last.pair_evidence))
g = orch.sensitivity_recomputation("b")
print("grid keys:", sorted(g))
print("ablation flags (b):", g["ablation"])
print("regression flags (b):", g["regression"])
print("has per-cell (tau_c,tau_r) entries:", any("," in k or "cell" in k for k in g))
print("any key with 'eligible'/'composition'/'not_identifiable':",
      [k for k in json.dumps(g).split('"') if any(s in k for s in ("eligib","composition","not_identifiable","identifiable"))][:5])
print("interaction_note is static string:", isinstance(g["interaction_note"], str))

print("\n=== GROUP 9: hardening/tests ===")
class BadCounter:
    identity = "   "
    def count_tokens(self, text): return 3.5
orch, somatic, cig, runner, inputs, reg = build_world(tmp(), candidates=("a",))
orch._token_counter = BadCounter()
try:
    out = orch.run(inputs)
    oh = out.results[0].stage4_child["measurements"]["expression_overhead"]
    print("blank identity + float count ACCEPTED; persisted:", oh)
except Exception as e:
    print("rejected:", type(e).__name__, e)
# also construct directly with bad counter
try:
    CigGateOrchestrator(registry=reg, base_config=orch._base_config, runner=runner,
                        budgets=orch._budgets, token_counter=BadCounter())
    print("constructor accepts BadCounter (validate_token_counter not called)")
except Exception as e:
    print("constructor rejected:", e)

orch, somatic, cig, runner, inputs, reg = build_world(tmp(), candidates=("a",))
out = orch.run(inputs)
print("vacuous single-candidate interaction block:",
      out.results[0].stage4_child["measurements"]["interaction"])

orch, somatic, cig, runner, inputs, reg = build_world(tmp(), candidates=("a", "b"),
    ablation_deltas={"a": 0.9, "b": 0.1},
    interaction={"both": (False, ()), "neither": (True, ())})
out = orch.run(inputs)
print("harmful-pair actual verdicts:", out.verdicts, "a stages:", out.results[0]._stage_results if hasattr(out.results[0], "_stage_results") else None)

orch, somatic, cig, runner, inputs, reg = build_world(tmp(), candidates=("a",))
out = orch.run(inputs)
print("committed:", out.committed, "somatic state:", somatic.state("a", 1),
      "gate_reports:", somatic.get("a", 1)["validation"]["gate_reports"])
print("live CIG aggregates:", list(cig.aggregates), "children:", list(cig.children))
cig.require_consistent()
print("require_consistent() passes vacuously on EMPTY live store")
