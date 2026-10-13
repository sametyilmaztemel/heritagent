"""R4: per-candidate CIG run identity (G2) is not honored by stage 4C.
`run()` builds the pair settings with
`evaluation_settings(max_tokens=..., seed=self._mining_seed)` — the BATCH
default seed — even when every candidate carries its own
GateCandidateInput.mining_seed. The stage-4 child of a candidate whose run
seed is 23 (child `seed: 23`, id `.../S23-interaction_regression`, its
regression `eval_generation_settings.seed: 23`) then embeds interaction
evidence whose `locked_settings.seed` is 11, i.e. one child = two seeds;
add_bundle()/verify() accept it. GateOutcome.mining_seed / parent_cig_id
also report only the batch defaults. No test sets
GateCandidateInput.mining_seed at all.

Run: cd pr28c && PYTHONPATH=. <venv314>/bin/python <this file>
"""
import sys
import tempfile
from pathlib import Path

from inheritance.gate import GateCandidateInput
from tests.gate.helpers import coherent_stage11_children
from tests.gate.test_gate_orchestrator import build_world

tmp = Path(tempfile.mkdtemp())
orch, somatic, cig, runner, inputs, registry = build_world(
    tmp, candidates=("a", "b"), interaction={"both": (True, ("a", "b")),
                                              "neither": (True, ())})
seeds = {"a": 23, "b": 41}
new_inputs = []
for inp in inputs:
    gid = inp.candidate_envelope["candidate"]["gene_id"]
    replay, ablation = coherent_stage11_children(
        gid, replay_pass=True, ablation_pass=True, parent=inp.parent_cig_id,
        seed=seeds[gid])
    new_inputs.append(GateCandidateInput(
        candidate_envelope=inp.candidate_envelope, frozen_rank=inp.frozen_rank,
        replay_evaluation=replay, ablation_evaluation=ablation,
        generalization_allocation=inp.generalization_allocation,
        regression_allocation=inp.regression_allocation,
        parent_cig_id=inp.parent_cig_id, mining_seed=seeds[gid]))

outcome = orch.run(new_inputs)
print("committed:", outcome.committed, "verdicts:", outcome.verdicts)
print("GateOutcome.mining_seed:", outcome.mining_seed,
      "parent_cig_id:", outcome.parent_cig_id)
bad = False
for r in outcome.results:
    child = r.stage4_child
    m = child["measurements"]
    pair_seed = m["interaction"]["pair_evidence"][0]["canonical_evidence"][
        "locked_settings"]["seed"]
    print(f"{r.gene_id}: child id={child['cig_id']} child.seed={child['seed']} "
          f"regression.seed={m['regression']['eval_generation_settings']['seed']} "
          f"interaction.locked_settings.seed={pair_seed}")
    bad |= pair_seed != child["seed"]
print("live verify():", cig.verify())
print("DEFECT REPRODUCED" if bad else "not reproduced")
sys.exit(0 if bad else 1)
