"""R12 - Operating characteristic of the locked stage-2 rule (n=20, binary
outcomes, merged code with BOOTSTRAP_SEED=0): probability that a trait with
a given TRUE per-task paired effect distribution passes stage 2.

Per-task paired outcome d in {+1, 0, -1} with P(+1)=a, P(-1)=b (iid over the
20 instances, a simplification). P(pass) = sum over (pos, neg) of the
multinomial probability x the merged code's verdict for that composition.

Run from /home/user/heritagent:
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. <venv314>/bin/python <this file>
"""
from math import comb

from inheritance.bootstrap import paired_cluster_bootstrap, stage_passes

N = 20
verdict = {}
for pos in range(N + 1):
    for neg in range(N + 1 - pos):
        d = [1.0] * pos + [-1.0] * neg + [0.0] * (N - pos - neg)
        r = paired_cluster_bootstrap(d, seed=0)
        verdict[(pos, neg)] = stage_passes(r.point_estimate, r.lower_bound)


def p_pass(a, b):
    c = 1 - a - b
    return sum(comb(N, x) * comb(N - x, y) * a**x * b**y * c**(N - x - y)
               for (x, y), ok in verdict.items() if ok)


print("true effect = a - b; tau_c = 0.05")
for a, b in ((0.05, 0.0), (0.10, 0.0), (0.10, 0.05), (0.15, 0.05), (0.20, 0.0),
             (0.20, 0.10), (0.25, 0.05), (0.30, 0.0), (0.30, 0.10), (0.40, 0.10)):
    print(f"  P(+1)={a:.2f} P(-1)={b:.2f} true effect={a-b:+.2f}: P(pass stage 2) = {p_pass(a, b):.3f}")
print(f"  null (a=b=0.10): P(pass) = {p_pass(0.10, 0.10):.4f}; null (a=b=0.25): {p_pass(0.25, 0.25):.4f}")
