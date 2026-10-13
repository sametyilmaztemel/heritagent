"""R03 - Percentile-index definition (inheritance/bootstrap.py:57-59) and
Monte-Carlo sensitivity of the verdict to task ORDER.

(a) The code takes means[int(alpha/2*B)] and means[int((1-alpha/2)*B)]. For
    B=10,000 and conf=0.95 the indices are 250 and 9750: the 251st-smallest
    and the 250th-largest resample means, i.e. the two tails are cut
    asymmetrically (250 values below LB, 249 values above UB). A type-1
    (inverted-CDF) 2.5th percentile is the 250th smallest (index 249);
    numpy's default (type 7, linear) interpolates at position 249.975.
    The decision rule LB > 0 is therefore "count(means <= 0) <= 250" in the
    code vs "<= 249" for type 1. No quantile definition is pre-registered.
(b) The resample stream draws INDICES, so permuting the same 20 deltas
    (e.g. a different but equally valid allocation order) changes the LB and
    can flip the verdict at the same seed.

Run from /home/user/heritagent:
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. <venv314>/bin/python <this file>
"""
import random

from inheritance.bootstrap import paired_cluster_bootstrap, stage_passes

print("(a) index arithmetic used by bootstrap.py:57-59")
for conf, B in ((0.95, 10_000), (0.95, 1_000), (0.95, 500), (0.90, 10_000), (0.99, 10_000)):
    alpha = 1.0 - conf
    lo = max(0, int(alpha / 2 * B))
    hi = min(B - 1, int((1 - alpha / 2) * B))
    print(f"  conf={conf} B={B}: alpha/2*B={alpha/2*B!r} -> lower_index={lo} "
          f"({lo+1}th smallest); (1-alpha/2)*B={(1-alpha/2)*B!r} -> upper_index={hi} "
          f"({B-hi}th largest)")

print("\n    conf=0.95, B=10000: lower = means[250] (251st smallest), upper = means[9750]"
      " (250th largest) -> 250 resample means strictly inside-below LB vs 249 above UB")

# find a (deltas, seed) whose resample-mean count at <= 0 is exactly 250:
# merged code says LB > 0 (PASS); a type-1 2.5th percentile says LB == 0 (FAIL)
print("\n(b0) boundary where the index choice decides the verdict:")
deltas = [1.0] * 10 + [-1.0] * 3 + [0.0] * 7   # exact P(mean<=0)=0.0258 (see R02)
found = None
for seed in range(2000):
    rng = random.Random(seed)
    means = []
    for _ in range(10_000):
        means.append(sum(deltas[rng.randrange(20)] for _ in range(20)) / 20)
    count_le0 = sum(1 for m in means if m <= 0)
    if count_le0 == 250:
        means.sort()
        r = paired_cluster_bootstrap(deltas, seed=seed)
        type7 = means[249] + 0.975 * (means[250] - means[249])
        found = (seed, count_le0, means[249], means[250], r.lower_bound, type7)
        break
if found:
    seed, c, type1, code_lb, lb, type7 = found
    print(f"  seed={seed}: count(resample mean <= 0) = {c}")
    print(f"  type-1 2.5th percentile (means[249]) = {type1}  -> LB>0: {type1 > 0}")
    print(f"  merged code LB (means[250])          = {lb}  -> LB>0: {lb > 0}")
    print(f"  numpy-default type-7 (pos 249.975)   = {type7:.5f} -> LB>0: {type7 > 0}")
    print(f"  stage_passes(point=0.35, code LB) = {stage_passes(0.35, lb)}")
else:
    print("  no seed with exactly 250 found in range")

print("\n(b) same 20 deltas, same seed 0, different task order:")
base = [1.0] * 10 + [-1.0] * 3 + [0.0] * 7
orders = {
    "pos,neg,zero": base,
    "zero,neg,pos": [0.0] * 7 + [-1.0] * 3 + [1.0] * 10,
}
rng = random.Random(12345)
for k in range(6):
    perm = base[:]
    rng.shuffle(perm)
    orders[f"shuffle{k}"] = perm
for name, d in orders.items():
    r = paired_cluster_bootstrap(d, seed=0)
    print(f"  order={name:13s} point={r.point_estimate:.2f} LB={r.lower_bound:+.2f} "
          f"verdict={'PASS' if stage_passes(r.point_estimate, r.lower_bound) else 'FAIL'}")
