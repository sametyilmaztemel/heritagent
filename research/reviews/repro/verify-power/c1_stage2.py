"""C1 independent check: stage-2 (ablation, n=20) decision as a function of the
observed composition (a rescues, b breaks).

Independent of exp0001_power_sim.py. Two methods:
 (1) exact bootstrap tail p_ab = P(resampled sum <= 0) under Multinomial(20;
     a/20, b/20, c/20), by direct enumeration of (A, B) counts;
 (2) the ACTUAL seed-0 random stream used by inheritance/bootstrap.py
     (random.Random(0).randrange(20), 10000 x 20 calls), turned into an index
     count matrix M (10000 x 20). For a delta vector d placed in task order,
     resample sums = M @ d; LB = sorted(sums/20)[250]. We sample many random
     task orderings per composition and report the empirical pass fraction.
     Validated against the repo bootstrap (loaded read-only by file path).
"""
import sys, math, random, importlib.util, json
sys.dont_write_bytecode = True
import numpy as np
from scipy.stats import binom

REPO = "/home/user/heritagent"
N = 20
NRES = 10_000
LOWER_INDEX = max(0, int((1.0 - 0.95) / 2 * NRES))  # same expression as bootstrap.py
assert LOWER_INDEX == 250, LOWER_INDEX

# load repo bootstrap read-only (no bytecode written)
spec = importlib.util.spec_from_file_location("repo_bootstrap", f"{REPO}/inheritance/bootstrap.py")
rb = importlib.util.module_from_spec(spec); sys.modules["repo_bootstrap"] = rb; spec.loader.exec_module(rb)


def exact_tail(a, b, n=N):
    """P(A' - B' <= 0), (A', B', C') ~ Multinomial(n; a/n, b/n, (n-a-b)/n)."""
    pa, pb = a / n, b / n
    if a == 0:
        return 1.0
    tot = 0.0
    for x in range(n + 1):  # A' = x
        px = binom.pmf(x, n, pa)
        if px == 0:
            continue
        if b == 0:
            tot += px * (1.0 if x == 0 else 0.0)
            continue
        q = min(1.0, pb / (1 - pa)) if pa < 1 else 0.0
        # need B' >= x, B' | A'=x ~ Bin(n-x, q)
        tot += px * (float(x == 0) if n - x == 0 else binom.sf(x - 1, n - x, q))
    return tot


# (2) index count matrix from the exact seed-0 stream
rng = random.Random(0)
M = np.zeros((NRES, N), dtype=np.int16)
for r in range(NRES):
    for _ in range(N):
        M[r, rng.randrange(N)] += 1


def lb_from_M(d):
    sums = M @ np.asarray(d, dtype=np.int64)
    s = np.sort(sums)
    return s[LOWER_INDEX] / N


# validate M-based LB against the repo bootstrap on random vectors
vrng = np.random.default_rng(1)
for _ in range(12):
    d = vrng.choice([-1.0, 0.0, 1.0], size=N, p=[0.1, 0.6, 0.3]).tolist()
    res = rb.paired_cluster_bootstrap(d, seed=0)
    mine = lb_from_M([int(x) for x in d])
    assert abs(res.lower_bound - mine) < 1e-12, (res.lower_bound, mine)
print("M-matrix LB matches repo bootstrap.py on 12 random vectors")

prng = np.random.default_rng(20261013)
K_ORDER = 4000
rows = []
for a in range(0, N + 1):
    for b in range(0, N + 1 - a):
        if a - b < 1:
            continue  # point < 0.05 -> fails point rule regardless
        p = exact_tail(a, b)
        binom_pass = binom.cdf(LOWER_INDEX, NRES, p)
        # empirical with the true seed-0 stream over random orderings
        base = np.array([1] * a + [-1] * b + [0] * (N - a - b), dtype=np.int64)
        D = np.empty((N, K_ORDER), dtype=np.int64)
        for k in range(K_ORDER):
            D[:, k] = prng.permutation(base)
        sums = M.astype(np.int64) @ D            # NRES x K
        cnt_le0 = (sums <= 0).sum(axis=0)        # number of resample means <= 0
        emp_pass = float(np.mean(cnt_le0 <= LOWER_INDEX))
        rows.append(dict(a=a, b=b, net=a - b, point=(a - b) / N, p_tail=p,
                         binom_pass=float(binom_pass), emp_pass=emp_pass,
                         cnt_min=int(cnt_le0.min()), cnt_max=int(cnt_le0.max())))

# spot-check a few orderings of boundary compositions directly with repo code
spot = {}
for (a, b) in [(6, 1), (8, 2), (10, 3), (4, 0), (3, 0), (5, 1)]:
    passes = 0
    for k in range(25):
        base = [1.0] * a + [-1.0] * b + [0.0] * (N - a - b)
        random.Random(1000 + k).shuffle(base)
        r = rb.paired_cluster_bootstrap(base, seed=0)
        passes += rb.stage_passes(r.point_estimate, r.lower_bound, tau_c=0.05)
    spot[f"{a},{b}"] = f"{passes}/25"

print("\nComposition table (only net>=1 shown where anything can pass):")
print(f"{'a':>3}{'b':>3}{'net':>4}{'p_tail':>9}{'binomApprox':>12}{'seed0_emp':>10}  cnt_range")
for r in rows:
    if r["emp_pass"] > 0 or r["binom_pass"] > 1e-4 or r["b"] <= 3 and r["a"] <= 12:
        print(f"{r['a']:>3}{r['b']:>3}{r['net']:>4}{r['p_tail']:>9.4f}{r['binom_pass']:>12.4f}{r['emp_pass']:>10.4f}  [{r['cnt_min']},{r['cnt_max']}]")

# derived rule
print("\nMinimal a passing (emp_pass==1) per b:")
for b in range(0, 9):
    sure = [r["a"] for r in rows if r["b"] == b and r["emp_pass"] == 1.0]
    any_ = [r["a"] for r in rows if r["b"] == b and r["emp_pass"] > 0]
    print(f"  b={b}: always-pass a>= {min(sure) if sure else None}; some-pass a in {any_[:3] if any_ else None}")

rng_dep = [(r["a"], r["b"], r["emp_pass"], r["binom_pass"]) for r in rows if 0 < r["emp_pass"] < 1 or 1e-3 < r["binom_pass"] < 0.999]
print("\nRNG/ordering-dependent compositions (0<emp<1 or binom in (0.001,0.999)):")
for t in rng_dep:
    print(f"  (a={t[0]}, b={t[1]}): seed0-empirical {t[2]:.4f}, binomial approx {t[3]:.4f}")

# monotonicity check: is pass monotone in a for fixed b?
nonmono = []
for b in range(0, 10):
    seq = [(r["a"], r["emp_pass"]) for r in rows if r["b"] == b]
    for (a1, e1), (a2, e2) in zip(seq, seq[1:]):
        if e2 < e1 - 1e-9:
            nonmono.append((b, a1, e1, a2, e2))
print("non-monotone (in a) cells:", nonmono)

# tau_c grid: does any composition with LB>0 (in any sampled ordering) fail point>=tau_c?
min_net_pass = min(r["net"] for r in rows if r["emp_pass"] > 0)
print("\nmin net among compositions that can pass LB>0:", min_net_pass)
for tau in (0.03, 0.05, 0.08):
    need = min(k for k in range(0, 21) if (k * 1.0) / 20 >= tau)
    # float exactly as code: sum(deltas)/n with deltas floats
    print(f"  tau_c={tau}: point rule needs net>={need}; binding? {need > min_net_pass}")
print("spot checks with repo bootstrap (25 orderings each):", spot)

json.dump(dict(rows=rows, spot=spot), open(__file__.replace('.py', '_out.json'), 'w'), indent=1)
