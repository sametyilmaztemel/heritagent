"""R08 - Two fail-open guards around stage-2 evidence.

(a) AblationAllocation (inheritance/ablation.py:57-87) validates count,
    distinct ids and family strata, but NOT task provenance (bank_id). Tasks
    labelled bank_id="T_eval" or bank_id=None are accepted, whereas
    ReplayAllocation enforces bank provenance (replay.py:85-89). The
    contamination rule "T_eval never visible to miner, gate, or threshold
    tuning" (yaml:67) is therefore not enforced at the stage-2 boundary.
(b) validate_outcome (inheritance/runner.py:184-223) never inspects
    `status` (runner.py:164 is a free str); an infrastructure failure
    (e.g. status="adapter_error", success=False) in one arm is silently
    counted as a genuine paired outcome (d_i = -1 or +1).

Run from /home/user/heritagent:
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. <venv314>/bin/python <this file>
"""
import tempfile
from pathlib import Path

from genome.validation.registry import TraitRegistry
from inheritance.ablation import AblationAllocation, AblationEvaluator
from inheritance.runner import EpisodeOutcome, EvaluationTask
from runtime.loop import Budgets
from runtime.seed_g0 import g0_runtime_config
from tests.inheritance.scripted_runner import ScriptedRunner
from tests.somatic.conftest import acquired_envelope

fam = ["heat_and_place"] * 10 + ["clean_surface"] * 6 + ["navigate"] * 4
strata = {"heat_and_place": 10, "clean_surface": 6, "navigate": 4}
for label, bank in (("T_eval-labelled", "T_eval"), ("no provenance", None),
                    ("replay bank id", "replay_bank[heat_and_place]")):
    tasks = tuple(EvaluationTask(task_id=f"eval-{i}", family=f, payload={}, bank_id=bank)
                  for i, f in enumerate(fam))
    try:
        a = AblationAllocation(tasks=tasks, frozen_strata=strata)
        print(f"(a) AblationAllocation with {label:16s} bank_id={bank!r}: ACCEPTED "
              f"(metadata keys: {sorted(a.allocation_metadata())})")
    except ValueError as exc:
        print(f"(a) {label}: rejected: {exc}")

tmp = Path(tempfile.mkdtemp(prefix="r08-", dir=str(Path(__file__).parent)))
registry = TraitRegistry(tmp / "registry")
envelope = acquired_envelope(registry)
alloc = AblationAllocation(
    tasks=tuple(EvaluationTask(task_id=f"abl-{i}", family=f, payload={}, bank_id="ablation_bank")
                for i, f in enumerate(fam)), frozen_strata=strata)


class ErrorInjectingRunner(ScriptedRunner):
    def run_episode(self, task, config, budgets, condition, settings=None):
        o = super().run_episode(task, config, budgets, condition, settings)
        if condition == "without_trait" and task.task_id in {f"abl-{i}" for i in range(6)}:
            # infrastructure failure in the control arm, reported as a failed episode
            return EpisodeOutcome(task_id=o.task_id, condition=o.condition, success=False,
                                  expressed_gene_ids=(), status="adapter_error",
                                  trajectory_id=o.trajectory_id,
                                  effective_settings=o.effective_settings)
        return o


outcomes = {(f"abl-{i}", c): (True, ("look_before_heat_v1",) if c == "with_trait" else ())
            for i in range(20) for c in ("with_trait", "without_trait")}
ev = AblationEvaluator(candidate_envelope=envelope, registry=registry,
                       base_config=g0_runtime_config(registry),
                       runner=ErrorInjectingRunner(outcomes=outcomes),
                       budgets=Budgets(max_steps=4, max_total_retries=10, max_tokens_per_request=64),
                       parent_cig_id="CIG-0007", mining_seed=11)
res = ev.evaluate(alloc)
statuses = sorted({e["status"] for e in res.episodes})
print(f"\n(b) trait has NO effect (both arms succeed everywhere) but 6 control-arm episodes "
      f"hit status='adapter_error':")
print(f"    statuses accepted: {statuses}")
print(f"    deltas={res.deltas}")
print(f"    point={res.delta_success} LB={res.bootstrap['lower_bound']} passed={res.passed}")
