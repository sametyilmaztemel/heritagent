"""R3: the commit path binds the somatic decision to the evaluated candidate
ONLY by (gene_id, version): `_commit_results` checks
`somatic_store.state(gene_id, version) == 'candidate'` and then decide()s
whatever envelope the store holds under that key. Nothing compares the
stored candidate ref (artifact URI/sha256, type, provenance) with
`inp.candidate_envelope`, and the CIG aggregate itself records neither the
candidate version nor the artifact digest. So a somatic store whose
(gene_id, version) entry points at a DIFFERENT artifact than the one the
gate evaluated gets that never-evaluated artifact marked `validated`, with
gate_reports pointing at evidence produced on another payload.

Scenario: gate registry R1 holds skill a@1 = payload P1 (evaluated). The
somatic store is persistent but bound to registry R2 where a@1 = payload P2
(e.g. a second mining run / another machine's registry). Both stores are
individually valid.

Run: cd pr28c && PYTHONPATH=. <venv314>/bin/python <this file>
"""
import sys
import tempfile
from pathlib import Path

from genome.validation.registry import TraitRegistry
from inheritance.gate import CigGateOrchestrator
from somatic.store import SomaticStore
from tests.gate.test_gate_orchestrator import build_world

tmp = Path(tempfile.mkdtemp())
orch, _somatic, cig, runner, inputs, registry = build_world(
    tmp, candidates=("a",), commit_stores=False)
evaluated_artifact = inputs[0].candidate_envelope["candidate"]["artifact"]

# a second, independently valid registry where a@1 is a DIFFERENT payload
r2 = TraitRegistry(tmp / "registry2")
other_uri = r2.put("skill", "a", 1, {
    "name": "a", "principle": "a totally different procedure",
    "when_to_apply": "always", "applicability": {"task_families": ["clean_surface"]},
    "estimated_generality": 0.9})
other_env ={"candidate": dict(inputs[0].candidate_envelope["candidate"],
                                artifact=other_uri),
             "validation": {"state": "candidate"}}
somatic = SomaticStore.create(tmp / "somatic2.jsonl", r2)
somatic.add_candidate(other_env)
assert other_uri != evaluated_artifact

orch2 = CigGateOrchestrator(
    registry=registry, base_config=orch._base_config, runner=runner,
    budgets=orch._budgets, parent_cig_id="CIG-0007", mining_seed=11,
    token_counter=orch._token_counter,
    interaction_allocation=orch._interaction_allocation,
    somatic_store=somatic, cig_record_store=cig)
outcome = orch2.run(inputs)
stored = somatic.get("a", 1)
print("verdict:", outcome.verdicts, "committed:", outcome.committed)
print("evaluated artifact :", evaluated_artifact)
print("validated artifact :", stored["candidate"]["artifact"],
      stored["validation"])
agg = cig.aggregates["CIG-0007"]
print("aggregate keys (no version/artifact binding):", sorted(agg))
bad = (stored["validation"]["state"] == "validated"
       and stored["candidate"]["artifact"] != evaluated_artifact)
print("DEFECT REPRODUCED" if bad else "not reproduced")
sys.exit(0 if bad else 1)
