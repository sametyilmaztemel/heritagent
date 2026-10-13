"""R02 - The bootstrap seed is a free constructor parameter of the PRIMARY
ablation evaluator (inheritance/ablation.py:188, 204) and is not
pre-registered anywhere (EXP-0001 yaml/design, ADR-0002). tau_c and
n_resamples are locked (ablation.py:199-205), but at n=20 the seed alone can
flip the primary stage verdict for boundary compositions.

(a) find compositions whose exact P(resample mean <= 0) is near 0.025;
(b) count passes over bootstrap seeds 0..199 using the merged code;
(c) end-to-end: run AblationEvaluator twice on IDENTICAL scripted episode
    evidence with two different bootstrap_seed values -> two schema-valid
    child CIG records with opposite stage_pass.

Run from /home/user/heritagent:
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. <venv314>/bin/python <this file>
"""
import sys
import tempfile
from math import comb
from pathlib import Path

from inheritance.bootstrap import paired_cluster_bootstrap, stage_passes

N = 20


def exact_q(pos, neg, n=N):
    a, b = pos / n, neg / n
    c = 1.0 - a - b
    return sum(comb(n, x) * comb(n - x, y) * a**x * b**y * c**(n - x - y)
               for x in range(n + 1) for y in range(n + 1 - x) if x - y <= 0)


near = []
for pos in range(N + 1):
    for neg in range(N + 1 - pos):
        q = exact_q(pos, neg)
        if 0.018 <= q <= 0.032:
            near.append((abs(q - 0.025), pos, neg, q))
near.sort()
print("compositions with exact P(mean<=0) in [0.018, 0.032]:")
for _, pos, neg, q in near:
    print(f"  pos={pos} neg={neg} point={(pos-neg)/N:.2f} exactP={q:.4f}")

SEEDS = range(200)
print("\npass rate of the LOCKED rule over bootstrap seeds 0..199 (merged code):")
flip_example = None
for _, pos, neg, q in near:
    deltas = [1.0] * pos + [-1.0] * neg + [0.0] * (N - pos - neg)
    results = {}
    for s in SEEDS:
        r = paired_cluster_bootstrap(deltas, seed=s)
        results[s] = stage_passes(r.point_estimate, r.lower_bound)
    n_pass = sum(results.values())
    print(f"  pos={pos} neg={neg} point={(pos-neg)/N:.2f}: passes for {n_pass}/200 seeds;"
          f" seed0={'PASS' if results[0] else 'FAIL'}")
    if 0 < n_pass < 200 and flip_example is None:
        s_pass = next(s for s in SEEDS if results[s])
        s_fail = next(s for s in SEEDS if not results[s])
        flip_example = (pos, neg, s_pass, s_fail)

if flip_example is None:
    print("no flipping composition found")
    sys.exit(0)

pos, neg, s_pass, s_fail = flip_example
print(f"\nflip example: pos={pos} neg={neg}; seed {s_pass} -> PASS, seed {s_fail} -> FAIL")

# ---- (c) end-to-end through the primary AblationEvaluator -----------------
from genome.validation.loader import load_cig
from genome.validation.registry import TraitRegistry
from inheritance.ablation import AblationAllocation, AblationEvaluator
from inheritance.runner import EvaluationTask
from runtime.loop import Budgets
from runtime.seed_g0 import g0_runtime_config
from tests.inheritance.scripted_runner import ScriptedRunner
from tests.somatic.conftest import acquired_envelope

GENE = "look_before_heat_v1"
tmp = Path(tempfile.mkdtemp(prefix="r02-", dir=str(Path(__file__).parent)))
registry = TraitRegistry(tmp / "registry")
envelope = acquired_envelope(registry)
base = g0_runtime_config(registry)
families = ["heat_and_place"] * 10 + ["clean_surface"] * 6 + ["navigate"] * 4
tasks = tuple(EvaluationTask(task_id=f"abl-{i}", family=f, payload={}, bank_id="ablation_bank")
              for i, f in enumerate(families))
alloc = AblationAllocation(tasks=tasks, frozen_strata={"heat_and_place": 10,
                                                        "clean_surface": 6, "navigate": 4})
# scripted evidence: first `pos` tasks: with succeeds, without fails (+1);
# next `neg` tasks: with fails, without succeeds (-1); rest: both succeed (0)
outcomes = {}
for i in range(N):
    tid = f"abl-{i}"
    if i < pos:
        w, wo = True, False
    elif i < pos + neg:
        w, wo = False, True
    else:
        w, wo = True, True
    outcomes[(tid, "with_trait")] = (w, (GENE,))
    outcomes[(tid, "without_trait")] = (wo, ())


def run(seed):
    ev = AblationEvaluator(candidate_envelope=envelope, registry=registry, base_config=base,
                           runner=ScriptedRunner(outcomes=dict(outcomes)),
                           budgets=Budgets(max_steps=4, max_total_retries=10,
                                           max_tokens_per_request=64),
                           parent_cig_id="CIG-0007", mining_seed=11, bootstrap_seed=seed)
    res = ev.evaluate(alloc)
    child = res.child_record
    load_cig(child)
    m = child["measurements"]
    return res, m


for seed in (s_pass, s_fail):
    res, m = run(seed)
    print(f"  AblationEvaluator(bootstrap_seed={seed}): deltas sum={sum(res.deltas):+.0f} "
          f"point={m['point_estimate_delta_success']:.2f} LB={m['bootstrap']['lower_bound']:.2f} "
          f"tau_c={m['tau_c']} n={m['bootstrap']['n_resamples']} stage_pass={m['stage_pass']} "
          f"(child {res.child_record['cig_id']} schema-valid)")
print("=> identical episode evidence, identical locked tau_c/n_resamples, opposite primary verdicts")
