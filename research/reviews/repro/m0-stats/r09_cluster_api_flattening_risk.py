"""R09 - Forward risk: paired_cluster_bootstrap (inheritance/bootstrap.py:35)
takes a flat list of deltas and treats EACH element as a cluster. It has no
cluster-id argument. EXP-0001 §5 (design.md:63) requires "cluster bootstrap
over tasks, 10k resamples" for the T_eval arm contrasts, which pool 3 mining
seeds (design.md:59; yaml:97). If a caller passes 134 tasks x 3 seeds = 402
deltas, the same task is resampled as 3 independent clusters
(pseudo-replication) and the CI shrinks by ~sqrt(3).

Synthetic, deterministic data (random.Random(7)); no repo state touched.

Run from /home/user/heritagent:
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. <venv314>/bin/python <this file>
"""
import random

from inheritance.bootstrap import paired_cluster_bootstrap

T, S = 134, 3
rng = random.Random(7)
# task-level paired outcome C-B in {-1,0,+1}, strongly shared across seeds
# (same task, temp 0, deterministic env); each seed flips the task delta to 0
# with small probability to model mining-seed variation.
found = None
for trial in range(400):
    base = [rng.choice([1, 1, 0, 0, 0, 0, 0, -1, -1]) for _ in range(T)]
    per_seed = [[(d if rng.random() > 0.1 else 0) for d in base] for _ in range(S)]
    flat = [float(per_seed[s][t]) for s in range(S) for t in range(T)]
    clustered = [sum(per_seed[s][t] for s in range(S)) / S for t in range(T)]
    rf = paired_cluster_bootstrap(flat, seed=0)
    rc = paired_cluster_bootstrap(clustered, seed=0)
    if rf.lower_bound > 0 >= rc.lower_bound:
        found = (trial, rf, rc)
        break

if found:
    trial, rf, rc = found
    print(f"trial {trial}: identical data, two call shapes")
    print(f"  flat 402 deltas (task x seed as clusters): point={rf.point_estimate:+.4f} "
          f"CI=[{rf.lower_bound:+.4f}, {rf.upper_bound:+.4f}] width={rf.upper_bound-rf.lower_bound:.4f}"
          f" -> 'CI excludes 0': {rf.lower_bound > 0}")
    print(f"  134 task clusters (seed-averaged):         point={rc.point_estimate:+.4f} "
          f"CI=[{rc.lower_bound:+.4f}, {rc.upper_bound:+.4f}] width={rc.upper_bound-rc.lower_bound:.4f}"
          f" -> 'CI excludes 0': {rc.lower_bound > 0}")
    print(f"  width ratio cluster/flat = {(rc.upper_bound-rc.lower_bound)/(rf.upper_bound-rf.lower_bound):.2f}"
          f" (sqrt(3) = 1.73)")
else:
    print("no discordant trial found")
