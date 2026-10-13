"""EXP-0001 validity checks (read-only; imports repo code, writes nothing to the repo).

Run:  venv314/bin/python -I -B validity_checks.py /home/user/heritagent
Sections:
  1. effective integer thresholds per gate stage (n=5/20/10/40/10)
  2. float-boundary check of the PR #28 regression/sensitivity expression
  3. stage-2 pass region using the repo's own locked bootstrap
  4. model-based gate pass probabilities (assumption-laden; see docstring)
  5. budget recount (interaction per seed; arm-A pseudo-replication)
"""
from __future__ import annotations

import itertools
import math
import sys
from functools import lru_cache

sys.dont_write_bytecode = True
REPO = sys.argv[1] if len(sys.argv) > 1 else "/home/user/heritagent"
sys.path.insert(0, REPO)
from inheritance.bootstrap import paired_cluster_bootstrap, stage_passes  # noqa: E402

print("== 1. effective integer thresholds ==")
# stage 3: delta = (wins - losses)/10
for name, thr in (("in-scope  delta>=0.0", 0.0), ("out-scope delta>=-0.05", -0.05)):
    ok = [k for k in range(-10, 11) if k / 10 >= thr]
    print(f"  {name}: passes iff net (wins-losses) >= {min(ok)}")
# harmful pair: delta_pair < -0.05 with n=10
harm = [k for k in range(-10, 11) if k / 10 < -0.05]
print(f"  harmful pair (n=10, delta<-0.05): harmful iff net <= {max(harm)}  (one net loss)")
# replay
print("  replay (n=5): successes_with>=3 and net>=2 (integer rule in inheritance/replay.py:44-46)")

print("\n== 2. float boundary: PR #28 inheritance/regression.py:231 `drop = 1.0 - successes / len(tasks)` ==")
for tau in (0.02, 0.03, 0.05):
    allowed = [f for f in range(0, 41) if (1.0 - (40 - f) / 40) <= tau]
    exact = [f for f in range(0, 41) if f / 40 <= tau]
    print(f"  tau_r={tau}: failures allowed (code float) <= {max(allowed)} ; exact rational <= {max(exact)}"
          f"  drop(38/40)={1.0 - 38/40!r}")

print("\n== 3. stage-2 pass region (repo bootstrap, seed 0, 10k) ==")
N2 = 20
region: dict[tuple[int, int], bool] = {}
for b in range(0, N2 + 1):
    for c in range(0, N2 + 1 - b):
        deltas = [1.0] * b + [-1.0] * c + [0.0] * (N2 - b - c)
        if b - c <= 0:
            region[(b, c)] = False
            continue
        r = paired_cluster_bootstrap(deltas, seed=0)
        region[(b, c)] = stage_passes(r.point_estimate, r.lower_bound, tau_c=0.05)
min_pass = min((b - c) / N2 for (b, c), ok in region.items() if ok)
print(f"  minimum passing point estimate: {min_pass:.2f}")
for c in range(0, 4):
    bs = [b for b in range(0, N2 + 1 - c) if region.get((b, c))]
    print(f"  losses={c}: passes iff wins >= {min(bs) if bs else 'never'}")


# ---- 4. model-based pass probabilities -----------------------------------
DOC = """
Model (assumption-laden, NOT data): each task independently falls in one of four
paired cells: both succeed, both fail, win (with=1,without=0), loss (with=0,without=1).
G0 success p; discordance d = P(win)+P(loss); net effect e = P(win)-P(loss).
win=(d+e)/2, loss=(d-e)/2. Out-of-scope tasks: e=0, same d. T_reg tasks are G0
successes, so per-task with-trait failure prob f = loss/p. Stages treated as
independent (disjoint task sets). Interaction stage ignored (single eligible).
"""


def multinom_pmf(counts, probs):
    n = sum(counts)
    coef = math.factorial(n)
    for k in counts:
        coef //= math.factorial(k)
    pr = 1.0
    for k, q in zip(counts, probs):
        pr *= q ** k if k else 1.0
    return coef * pr


def p_replay(p, win, loss):
    both1 = p - loss
    both0 = 1 - p - win
    tot = 0.0
    for w in range(6):
        for l_ in range(6 - w):
            for b1 in range(6 - w - l_):
                b0 = 5 - w - l_ - b1
                succ_with = b1 + w
                if succ_with >= 3 and (w - l_) >= 2:
                    tot += multinom_pmf((w, l_, b1, b0), (win, loss, both1, both0))
    return tot


def p_stage2(win, loss):
    tie = 1 - win - loss
    tot = 0.0
    for (b, c), ok in region.items():
        if ok:
            tot += multinom_pmf((b, c, N2 - b - c), (win, loss, tie))
    return tot


def p_net_nonneg(n, win, loss):
    tie = 1 - win - loss
    tot = 0.0
    for b in range(n + 1):
        for c in range(n + 1 - b):
            if b >= c:
                tot += multinom_pmf((b, c, n - b - c), (win, loss, tie))
    return tot


def p_stage4(f, n=40, max_fail=1):
    return sum(math.comb(n, k) * f ** k * (1 - f) ** (n - k) for k in range(max_fail + 1))


print("\n== 4. model-based gate pass probabilities ==")
print(DOC.strip())
print("\n  P(T_reg pass, <=1 failure of 40) by per-task flip-to-fail prob f:")
for f in (0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20):
    print(f"    f={f:<6} P={p_stage4(f):.3f}")

def gate_table(p):
    print(f"\n  G0 success p={p}")
    print("  e      d     win   loss  f=loss/p | replay  stage2  in>=0  out>=0  T_reg | P(promote)  E[|C|] of K=10")
    for e in (0.0, 0.05, 0.10, 0.20, 0.30):
        for extra in (0.0, 0.10, 0.20, 0.30):
            d = e + extra
            if d == 0:
                continue
            win, loss = (d + e) / 2, (d - e) / 2
            if win > 1 - p or loss > p:
                continue
            f = loss / p
            pr = p_replay(p, win, loss)
            s2 = p_stage2(win, loss)
            pin = p_net_nonneg(10, win, loss)
            pout = p_net_nonneg(10, d / 2, d / 2)
            s4 = p_stage4(f)
            tot = pr * s2 * pin * pout * s4
            print(f"  {e:<5.2f} {d:<5.2f} {win:<5.3f} {loss:<5.3f} {f:<8.3f} | {pr:6.3f}  {s2:6.3f}  {pin:5.3f}  "
                  f"{pout:5.3f}  {s4:5.3f} | {tot:9.4f}  {10*tot:6.3f}")


for p in (0.40, 0.667):
    gate_table(p)
p = 0.40

print("\n  Null traits (e=0): per-candidate false promotion and P(>=1 of 10 null promoted)")
for d in (0.05, 0.10, 0.20, 0.30, 0.40):
    win = loss = d / 2
    tot = p_replay(p, win, loss) * p_stage2(win, loss) * p_net_nonneg(10, win, loss) ** 2 * p_stage4(loss / p)
    print(f"    d={d:<5} P(promote)={tot:.2e}  FWER(10)={1-(1-tot)**10:.2e}  stage2-only={p_stage2(win, loss):.4f}")

print("\n== 5. budget recount ==")
seeds, K = 3, 10
mining = seeds * 60
cig = seeds * K * (5 * 2 + 20 * 2 + 20 * 2 + 40)
final = 3 * (134 + 40 + 20) * seeds
pairs = K * (K - 1) // 2
inter_per_seed = pairs * 10 * 2
print(f"  mining={mining} cig={cig} final_eval={final} interaction_worst_per_seed={inter_per_seed}")
print(f"  total without interaction = {mining + cig + final}")
print(f"  total with interaction counted once (design) = {mining + cig + final + inter_per_seed}")
print(f"  total with interaction per seed (x3)        = {mining + cig + final + seeds * inter_per_seed}")
inter_dedup = pairs * 10 + 10
print(f"  interaction if the identical deterministic 'neither' arm ran once per seed: {inter_dedup} per seed")
armA_dup = 2 * (134 + 40 + 20)
print(f"  arm A deterministic duplicates across 3 seeds (same G0, temp 0): {armA_dup} episodes")
abl_dup = (K - 1) * 20
print(f"  stage-2 'without' arm = identical G0 on the same 20 tasks for all K candidates: {abl_dup} duplicate episodes/seed"
      " (usable as a free determinism audit)")
print(f"  T_reg feasibility: needs >=40 G0 successes in 60 T_mine tasks (>= {40/60:.1%})")
for pg in (0.30, 0.45, 0.60, 0.667, 0.75):
    pr = sum(math.comb(60, k) * pg ** k * (1 - pg) ** (60 - k) for k in range(40, 61))
    print(f"    G0 success {pg:.3f}: P(>=40/60 solved) = {pr:.4f}")
