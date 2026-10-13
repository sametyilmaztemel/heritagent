"""R04 - Degenerate inputs for paired_cluster_bootstrap / stage_passes, and
evaluator-level all-zero evidence (which no merged test asserts as FAIL).

Run from /home/user/heritagent:
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. <venv314>/bin/python <this file>
"""
import math
import tempfile
from pathlib import Path

from inheritance.bootstrap import paired_cluster_bootstrap, stage_passes

cases = {
    "all zero (20)": [0.0] * 20,
    "all +1 (20)": [1.0] * 20,
    "all -1 (20)": [-1.0] * 20,
    "single +1 (n=1)": [1.0],
    "single 0 (n=1)": [0.0],
    "one +1, 19 zero (point == tau_c)": [1.0] + [0.0] * 19,
}
for name, d in cases.items():
    r = paired_cluster_bootstrap(d, seed=0)
    print(f"{name:34s} point={r.point_estimate:+.3f} CI=[{r.lower_bound:+.3f}, {r.upper_bound:+.3f}]"
          f" pass={stage_passes(r.point_estimate, r.lower_bound)}")

print("\npoint == 0.05 exactness for binary deltas (float sum of ints / 20):")
ok = all((k / 20 == sum([1.0] * (k + j) + [-1.0] * j + [0.0] * (20 - k - 2 * j)) / 20)
         for k in range(0, 21) for j in range(0, (20 - k) // 2 + 1))
print(f"  sum(int-valued floats)/20 == k/20 for all compositions: {ok}")
print(f"  1/20 == 0.05: {1/20 == 0.05};  stage_passes(1/20, 0.01) = {stage_passes(1/20, 0.01)}")

print("\nunvalidated inputs (no finiteness / range / type check):")
for name, d in {
    "NaN among deltas": [float('nan')] + [1.0] * 19,
    "out-of-range delta 5.0": [5.0] + [0.0] * 19,
    "bool deltas": [True] * 20,
}.items():
    try:
        r = paired_cluster_bootstrap(d, seed=0)
        print(f"  {name:24s} accepted: point={r.point_estimate} CI=[{r.lower_bound}, {r.upper_bound}]"
              f" pass={stage_passes(r.point_estimate, r.lower_bound)}")
    except Exception as exc:  # noqa: BLE001
        print(f"  {name:24s} rejected: {type(exc).__name__}: {exc}")

for conf in (0.0, 1.0, 0.95):
    try:
        r = paired_cluster_bootstrap([0.1] * 20, seed=0, confidence=conf)
        print(f"  confidence={conf}: ok")
    except ValueError as exc:
        print(f"  confidence={conf}: rejected ({exc})")
try:
    paired_cluster_bootstrap([0.1] * 20, seed=None)
    print("  seed=None accepted (random.Random(None) seeds from OS entropy -> non-deterministic)")
except Exception as exc:  # noqa: BLE001
    print(f"  seed=None rejected: {exc}")

# ---- evaluator-level all-zero evidence --------------------------------------
from genome.validation.registry import TraitRegistry
from inheritance.ablation import AblationAllocation, AblationEvaluator
from inheritance.runner import EvaluationTask
from runtime.loop import Budgets
from runtime.seed_g0 import g0_runtime_config
from tests.inheritance.scripted_runner import ScriptedRunner
from tests.somatic.conftest import acquired_envelope

tmp = Path(tempfile.mkdtemp(prefix="r04-", dir=str(Path(__file__).parent)))
registry = TraitRegistry(tmp / "registry")
envelope = acquired_envelope(registry)
fam = ["heat_and_place"] * 10 + ["clean_surface"] * 6 + ["navigate"] * 4
alloc = AblationAllocation(
    tasks=tuple(EvaluationTask(task_id=f"abl-{i}", family=f, payload={}, bank_id="ablation_bank")
                for i, f in enumerate(fam)),
    frozen_strata={"heat_and_place": 10, "clean_surface": 6, "navigate": 4})
for label, w, wo in (("both succeed everywhere", True, True), ("both fail everywhere", False, False)):
    outcomes = {}
    for i in range(20):
        outcomes[(f"abl-{i}", "with_trait")] = (w, ("look_before_heat_v1",))
        outcomes[(f"abl-{i}", "without_trait")] = (wo, ())
    ev = AblationEvaluator(candidate_envelope=envelope, registry=registry,
                           base_config=g0_runtime_config(registry),
                           runner=ScriptedRunner(outcomes=outcomes),
                           budgets=Budgets(max_steps=4, max_total_retries=10, max_tokens_per_request=64),
                           parent_cig_id="CIG-0007", mining_seed=11)
    res = ev.evaluate(alloc)
    b = res.bootstrap
    print(f"\nevaluator, {label}: point={res.delta_success} CI=[{b['lower_bound']}, {b['upper_bound']}] "
          f"passed={res.passed}")
