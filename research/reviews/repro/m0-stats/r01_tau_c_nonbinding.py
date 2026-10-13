"""R01 - Is tau_c = 0.05 ever binding for the locked stage-2 rule at n=20?

Stage 2 (EXP-0001-design.md:48, inheritance/ablation.py:268-271): 20 task
instances, one deterministic run per (task, condition), binary success ->
paired d_i in {-1, 0, +1}. Pass iff point >= tau_c AND percentile-bootstrap
95% LB > 0 (inheritance/bootstrap.py:66-70).

This script:
  (a) computes, for every (pos, neg) composition with pos+neg <= 20, the EXACT
      probability q = P(resample mean <= 0) under the nonparametric bootstrap
      (multinomial), which governs LB > 0 (LB > 0 iff fewer than 251 of the
      10,000 resample means are <= 0, see R03);
  (b) runs the merged code (seed 0, 10k) on every composition;
  (c) recomputes the gate decision under the pre-registered secondary
      tau_c grid {0.03, 0.05, 0.08} (design §5 line 71) and checks whether any
      decision changes.

Run from /home/user/heritagent:
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. <venv314>/bin/python <this file>
"""
from math import comb
import time

from inheritance.bootstrap import paired_cluster_bootstrap, stage_passes

N = 20


def exact_q(pos, neg, n=N):
    """P(sum of n iid draws from the empirical {+1:pos, -1:neg, 0:rest} <= 0)."""
    a, b = pos / n, neg / n
    c = 1.0 - a - b
    total = 0.0
    for x in range(n + 1):
        for y in range(n + 1 - x):
            if x - y <= 0:
                z = n - x - y
                total += comb(n, x) * comb(n - x, y) * a**x * b**y * c**z
    return total


def deltas_for(pos, neg, n=N):
    return [1.0] * pos + [-1.0] * neg + [0.0] * (n - pos - neg)


t0 = time.time()
rows = []
for pos in range(N + 1):
    for neg in range(N + 1 - pos):
        d = deltas_for(pos, neg)
        r = paired_cluster_bootstrap(d, seed=0)
        rows.append((pos, neg, r.point_estimate, r.lower_bound, exact_q(pos, neg)))
elapsed = time.time() - t0

print(f"compositions evaluated: {len(rows)} (code, seed=0, 10k)  [{elapsed:.1f}s]")

passing = [(p, n_, pt, lb, q) for p, n_, pt, lb, q in rows if stage_passes(pt, lb)]
min_pass_point = min(pt for _, _, pt, _, _ in passing)
print(f"compositions passing the locked rule (tau_c=0.05): {len(passing)}")
print(f"minimum point estimate among passing compositions: {min_pass_point:.2f}")
print("passing compositions with the smallest point estimate:")
for p, n_, pt, lb, q in sorted(passing, key=lambda r: (r[2], -r[0]))[:6]:
    print(f"  pos={p:2d} neg={n_:2d} point={pt:.2f} LB={lb:.2f} exact P(mean<=0)={q:.4f}")

print("\nLB>0 among compositions with point in (0, 0.20):")
small = [(p, n_, pt, lb, q) for p, n_, pt, lb, q in rows if 0 < pt < 0.20 - 1e-12]
print(f"  count with point in (0,0.20): {len(small)}; with LB>0: "
      f"{sum(1 for r in small if r[3] > 0)}")
print(f"  smallest exact P(mean<=0) among them: {min(r[4] for r in small):.4f} "
      f"(needs < ~0.025 for LB>0)")
for p, n_, pt, lb, q in sorted(small, key=lambda r: r[4])[:4]:
    print(f"    pos={p} neg={n_} point={pt:.2f} LB={lb:.2f} exactP={q:.4f}")

print("\nnet +1/20 (point == tau_c == 0.05) compositions: exact P(mean<=0)")
for p, n_, pt, lb, q in rows:
    if p - n_ == 1 and p <= 3:
        print(f"  pos={p} neg={n_} point={pt} (== 0.05: {pt == 0.05}) LB={lb} exactP={q:.4f}")

print("\nSecondary tau_c grid {0.03, 0.05, 0.08} (design §5:71):")
changes = 0
for p, n_, pt, lb, q in rows:
    decisions = {tau: stage_passes(pt, lb, tau_c=tau) for tau in (0.03, 0.05, 0.08)}
    if len(set(decisions.values())) > 1:
        changes += 1
        print("  decision changes:", p, n_, decisions)
print(f"  compositions whose decision differs across the grid: {changes} / {len(rows)}")

print("\nlargest tau_c that would still never bind (smallest passing point):",
      f"{min_pass_point:.2f}")
print("passing tau_c threshold sweep (how many compositions pass):")
for tau in (0.0, 0.05, 0.10, 0.15, 0.20, 0.25):
    print(f"  tau_c={tau:.2f}: {sum(1 for r in rows if stage_passes(r[2], r[3], tau_c=tau))}")
