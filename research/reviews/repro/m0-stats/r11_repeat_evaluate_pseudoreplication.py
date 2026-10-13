"""R11 - "Exactly one execution per (task instance, condition)" is enforced
only WITHIN a single evaluate() call: `seen_pairs` is a local set
(replay.py:201, ablation.py:218). Because allocation ids are already
validated distinct (replay.py:77-79, ablation.py:70-71), the in-loop
duplicate guard (replay.py:207-210, ablation.py:231-234) is unreachable,
while the realistic duplication path - calling evaluate() again on the same
evaluator/candidate - is unguarded. A second call re-executes all episodes
and emits a second schema-valid child with the SAME cig_id; if the backend is
not perfectly deterministic the two children can disagree (best-of-k).

Run from /home/user/heritagent:
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. <venv314>/bin/python <this file>
"""
import tempfile
from pathlib import Path

from genome.validation.loader import load_cig
from genome.validation.registry import TraitRegistry
from inheritance.ablation import AblationAllocation, AblationEvaluator
from inheritance.replay import ReplayAllocation, ReplayEvaluator
from inheritance.runner import EvaluationTask
from runtime.loop import Budgets
from runtime.seed_g0 import g0_runtime_config
from tests.inheritance.scripted_runner import ScriptedRunner
from tests.somatic.conftest import acquired_envelope

GENE = "look_before_heat_v1"
tmp = Path(tempfile.mkdtemp(prefix="r11-", dir=str(Path(__file__).parent)))
registry = TraitRegistry(tmp / "registry")
envelope = acquired_envelope(registry)
base = g0_runtime_config(registry)
budgets = Budgets(max_steps=4, max_total_retries=10, max_tokens_per_request=64)


class DriftingRunner(ScriptedRunner):
    """Simulates backend nondeterminism: the 2nd execution of a pair flips
    the with-trait outcome on one task."""
    def run_episode(self, task, config, budgets, condition, settings=None):
        o = super().run_episode(task, config, budgets, condition, settings)
        n_prev = sum(1 for c in self.calls if c == (task.task_id, condition))
        if n_prev >= 2 and condition == "with_trait" and task.task_id == self.flip_id:
            from inheritance.runner import EpisodeOutcome
            return EpisodeOutcome(task_id=o.task_id, condition=o.condition, success=True,
                                  expressed_gene_ids=(GENE,), status="finished",
                                  trajectory_id=o.trajectory_id + "b",
                                  effective_settings=o.effective_settings)
        return o


# ---- replay -----------------------------------------------------------------
BANK = "replay_bank[heat_and_place]"
ralloc = ReplayAllocation(
    source_task=EvaluationTask(task_id="src-1", family="heat_and_place", payload={}),
    bank_variants=tuple(EvaluationTask(task_id=f"bank-{i}", family="heat_and_place",
                                       payload={}, bank_id=BANK) for i in range(1, 5)),
    source_family="heat_and_place", replay_bank_id=BANK)
ids = [t.task_id for t in ralloc.tasks]
outcomes = {}
for k, tid in enumerate(ids):
    outcomes[(tid, "with_trait")] = (k < 2, (GENE,) if k < 2 else ())   # 2/5 with
    outcomes[(tid, "without_trait")] = (False, ())
runner = DriftingRunner(outcomes=outcomes)
runner.flip_id = "bank-2"
ev = ReplayEvaluator(candidate_envelope=envelope, registry=registry, base_config=base,
                     runner=runner, budgets=budgets, parent_cig_id="CIG-0007", mining_seed=11)
r1 = ev.evaluate(ralloc)
r2 = ev.evaluate(ralloc)
for r in (r1, r2):
    load_cig(r.child_record)
print("replay: two evaluate() calls on the same evaluator + allocation")
print(f"  runner calls: {len(runner.calls)} (one-per-pair contract would allow 10)")
print(f"  child ids: {r1.child_record['cig_id']} / {r2.child_record['cig_id']} (identical)")
print(f"  stage_pass: first={r1.passed} second={r2.passed}  (both children schema-valid)")

# ---- ablation ---------------------------------------------------------------
fam = ["heat_and_place"] * 10 + ["clean_surface"] * 6 + ["navigate"] * 4
aalloc = AblationAllocation(
    tasks=tuple(EvaluationTask(task_id=f"abl-{i}", family=f, payload={}, bank_id="ablation_bank")
                for i, f in enumerate(fam)),
    frozen_strata={"heat_and_place": 10, "clean_surface": 6, "navigate": 4})
outcomes = {(f"abl-{i}", c): (False, ()) for i in range(20) for c in ("with_trait", "without_trait")}
for i in range(3):
    outcomes[(f"abl-{i}", "with_trait")] = (True, (GENE,))
runner = DriftingRunner(outcomes=outcomes)
runner.flip_id = "abl-5"
ev = AblationEvaluator(candidate_envelope=envelope, registry=registry, base_config=base,
                       runner=runner, budgets=budgets, parent_cig_id="CIG-0007", mining_seed=11)
a1 = ev.evaluate(aalloc)
a2 = ev.evaluate(aalloc)
print("ablation: two evaluate() calls on the same evaluator + allocation")
print(f"  runner calls: {len(runner.calls)} (contract: 40)")
print(f"  child ids identical: {a1.child_record['cig_id'] == a2.child_record['cig_id']}")
print(f"  first: point={a1.delta_success} LB={a1.bootstrap['lower_bound']} pass={a1.passed}; "
      f"second: point={a2.delta_success} LB={a2.bootstrap['lower_bound']} pass={a2.passed}")
