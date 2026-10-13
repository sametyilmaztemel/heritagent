"""R05 - AblationEvaluator(bootstrap_seed=None) is accepted (no type check at
inheritance/ablation.py:188/204; bootstrap.py:48 passes it to
random.Random(None) = OS entropy). The PRIMARY stage verdict then becomes
non-reproducible while the child record stays schema-valid (cig schema
'measurements' is a free object) and records "seed": null.

Run from /home/user/heritagent:
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. <venv314>/bin/python <this file>
"""
import tempfile
from collections import Counter
from pathlib import Path

from genome.validation.loader import load_cig
from genome.validation.registry import TraitRegistry
from inheritance.ablation import AblationAllocation, AblationEvaluator
from inheritance.runner import EvaluationTask
from runtime.loop import Budgets
from runtime.seed_g0 import g0_runtime_config
from tests.inheritance.scripted_runner import ScriptedRunner
from tests.somatic.conftest import acquired_envelope

GENE = "look_before_heat_v1"
POS, NEG = 10, 3   # boundary composition from R02 (exact P(mean<=0)=0.0258)
tmp = Path(tempfile.mkdtemp(prefix="r05-", dir=str(Path(__file__).parent)))
registry = TraitRegistry(tmp / "registry")
envelope = acquired_envelope(registry)
fam = ["heat_and_place"] * 10 + ["clean_surface"] * 6 + ["navigate"] * 4
alloc = AblationAllocation(
    tasks=tuple(EvaluationTask(task_id=f"abl-{i}", family=f, payload={}, bank_id="ablation_bank")
                for i, f in enumerate(fam)),
    frozen_strata={"heat_and_place": 10, "clean_surface": 6, "navigate": 4})
outcomes = {}
for i in range(20):
    w, wo = (True, False) if i < POS else ((False, True) if i < POS + NEG else (True, True))
    outcomes[(f"abl-{i}", "with_trait")] = (w, (GENE,))
    outcomes[(f"abl-{i}", "without_trait")] = (wo, ())

verdicts = Counter()
seeds_recorded = set()
for _ in range(12):
    ev = AblationEvaluator(candidate_envelope=envelope, registry=registry,
                           base_config=g0_runtime_config(registry),
                           runner=ScriptedRunner(outcomes=dict(outcomes)),
                           budgets=Budgets(max_steps=4, max_total_retries=10, max_tokens_per_request=64),
                           parent_cig_id="CIG-0007", mining_seed=11, bootstrap_seed=None)
    res = ev.evaluate(alloc)
    child = res.child_record
    load_cig(child)  # still schema-valid
    m = child["measurements"]
    verdicts[(m["stage_pass"], m["bootstrap"]["lower_bound"])] += 1
    seeds_recorded.add(repr(m["bootstrap"]["seed"]))
print("12 primary evaluations on IDENTICAL evidence with bootstrap_seed=None:")
for (passed, lb), n in sorted(verdicts.items()):
    print(f"  stage_pass={passed} LB={lb}: {n} runs")
print(f"  recorded bootstrap seed(s) in child records: {sorted(seeds_recorded)}")
print("  all 12 child records passed load_cig (schema-valid)")
