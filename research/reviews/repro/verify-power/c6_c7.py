"""C6: H2 regression component on 40 T_reg x 3 seeds; C7: seed-direction null.

C6 model (from the study's text, T4): per task per seed, arm B fails w.p. r_B, arm C
w.p. r_C, independent arms; rho=1 -> identical across seeds, rho=0 -> independent.
Per-task D = mean_s(FB - FC). Pass iff 10k percentile cluster-bootstrap LB of mean(D) > 0.
rho=1: EXACT enumeration over trinomial compositions with exact bootstrap tail.
rho=0: MC over datasets with EXACT bootstrap tail (FFT), 10k-RNG fuzz via Bin(10000,p)<=250.
"""
import math, json
import numpy as np
from scipy.stats import binom

HERE = "<scratch>/out/repro/verify-power"
N = 40; S = 3; NRES = 10_000; LIDX = 250


def lbp(p):
    return binom.cdf(LIDX, NRES, p)


def tail_tri(a, b, n):
    if a == 0:
        return 1.0
    pa, pb = a / n, b / n
    x = np.arange(n + 1)
    px = binom.pmf(x, n, pa)
    if b == 0:
        return float(px[0])
    q = min(1.0, pb / (1 - pa)) if pa < 1 else 0.0
    sf = binom.sf(x - 1, n - x, q)
    sf = np.where(n - x == 0, (x == 0).astype(float), sf)
    return float(np.sum(px * sf))


def rho1_exact(rB, rC, couple=0.0):
    """couple = P(B fails | C fails) mixing toward full dependence (C's genes subset of B's)."""
    # P(+1)=P(B fail, C ok), P(-1)=P(C fail, B ok)
    pCf = rC
    pBf_given_Cf = couple + (1 - couple) * rB
    # keep marginal P(B fail)=rB: P(B fail, C ok) = rB - rC*pBf_given_Cf
    pp = rB - rC * pBf_given_Cf
    pm = rC * (1 - pBf_given_Cf)
    if pp < 0:
        return float('nan')
    tot = 0.0
    for a in range(N + 1):
        for b in range(N + 1 - a):
            if a <= b:
                continue
            if (pm == 0 and b > 0) or (pp == 0 and a > 0):
                continue
            lw = (math.lgamma(N + 1) - math.lgamma(a + 1) - math.lgamma(b + 1) - math.lgamma(N - a - b + 1)
                  + (a * math.log(pp) if a else 0) + (b * math.log(pm) if b else 0)
                  + ((N - a - b) * math.log(1 - pp - pm) if N - a - b else 0))
            w = math.exp(lw)
            if w < 1e-15:
                continue
            tot += w * lbp(tail_tri(a, b, N))
    return tot


def rho0_mc(rB, rC, reps=20000, rng=None):
    FB = (rng.random((reps, N, S)) < rB).astype(np.int8)
    FC = (rng.random((reps, N, S)) < rC).astype(np.int8)
    Dk = (FB - FC).sum(axis=2)  # -3..3
    counts = np.stack([(Dk == k).sum(axis=1) for k in range(-3, 4)], axis=1) / N
    L = 512; ks = np.arange(-3, 4); w = 2 * np.pi * np.arange(L) / L
    phi = counts @ np.exp(-1j * np.outer(ks, w))
    pmf = np.real(np.fft.ifft(phi ** N, axis=1))
    ptail = np.clip(pmf[:, 0] + pmf[:, L - 3 * N:].sum(axis=1), 0, 1)
    point_pos = Dk.sum(axis=1) > 0
    return float(np.mean(lbp(ptail) * point_pos))


rng = np.random.default_rng(7)
print("=== C6: regression component power P[reg(B)-reg(C) > 0 with LB>0] ===")
print(f"{'r_B':>6}{'r_C':>7}{'diff':>7}{'rho=0':>9}{'rho=1':>9}   study(rho0/rho1)")
study = {(0.05, 0.0): (0.857, 0.135), (0.05, 0.025): (0.161, 0.041), (0.1, 0.0): (0.995, 0.601),
         (0.1, 0.025): (0.679, 0.263), (0.1, 0.05): (0.343, 0.120), (0.15, 0.05): (0.744, 0.291),
         (0.2, 0.05): (0.951, 0.535), (0.3, 0.05): (1.0, 0.872)}
res = {}
for rB in (0.05, 0.075, 0.10, 0.15, 0.20, 0.30):
    for rC in (0.0, 0.025, 0.05):
        if rB <= rC:
            continue
        p0 = rho0_mc(rB, rC, rng=rng); p1 = rho1_exact(rB, rC)
        res[f"{rB},{rC}"] = (p0, p1)
        s = study.get((rB, rC), ("", ""))
        print(f"{rB:>6}{rC:>7}{rB-rC:>7.3f}{p0:>9.3f}{p1:>9.3f}   {s}")

print("\nCells with BOTH arms <= 10pp and true diff in [5, 7.5]pp:")
sel = {k: v for k, v in res.items() if float(k.split(',')[0]) <= 0.10 and 0.05 - 1e-9 <= float(k.split(',')[0]) - float(k.split(',')[1]) <= 0.075 + 1e-9}
for k, v in sel.items():
    print(f"  (r_B,r_C)=({k}): rho0={v[0]:.3f} rho1={v[1]:.3f}")
lo0, hi0 = min(v[0] for v in sel.values()), max(v[0] for v in sel.values())
lo1, hi1 = min(v[1] for v in sel.values()), max(v[1] for v in sel.values())
print(f"  range rho=0: {lo0:.3f}-{hi0:.3f};  rho=1: {lo1:.3f}-{hi1:.3f}")

print("\nCoupling sensitivity (rho=1; P(B fails | C fails)):")
for (rB, rC) in ((0.1, 0.025), (0.15, 0.05), (0.2, 0.05)):
    print(f"  ({rB},{rC}):", ", ".join(f"c={c}: {rho1_exact(rB, rC, c):.3f}" for c in (0.0, 0.5, 1.0)))

print("\nSelection/circularity case: arm C empty (C=G0 reproduces the G0 baseline -> r_C=0) or one gene that")
print("already passed stage 4 on the SAME 40 T_reg tasks (<=1 failure, deterministic rerun -> r_C<=1/40):")
for rB in (0.025, 0.05, 0.075, 0.10):
    print(f"  r_B={rB}: r_C=0 -> rho0={rho0_mc(rB, 0.0, rng=rng):.3f} rho1={rho1_exact(rB, 0.0):.3f}"
          f" | closed form rho1 P(Bin(40,rB)>=4)={1-binom.cdf(3,40,rB):.3f}")
# with rho=1 and r_C=0 the decision is 'B has >= m failing tasks': find m
for m in range(1, 7):
    print(f"   m={m} failing B tasks, 0 C failures: tail=(1-m/40)^40={(1-m/40)**40:.4f} -> pass prob {lbp((1-m/40)**40):.3f}")

print("\n=== C7: P(direction C-B > 0 in >= 2 of 3 seeds) under the null ===")
print("continuous/no ties, independent symmetric seeds: P = 3*(1/2)^3 + (1/2)^3 =", 3 * 0.125 + 0.125)
for f in (0.1, 0.2, 0.3):
    q = f * (1 - f)   # P(+1) = P(-1) per task under the null (sym noise; independent of Y0)
    # P(per-seed sum == 0) over 134 tasks, exact
    n = 134
    tie = 0.0
    for k in range(0, n // 2 + 1):  # k pluses, k minuses
        lw = math.lgamma(n + 1) - 2 * math.lgamma(k + 1) - math.lgamma(n - 2 * k + 1) + 2 * k * math.log(q) + (n - 2 * k) * math.log(1 - 2 * q)
        tie += math.exp(lw)
    ppos = (1 - tie) / 2
    p23 = ppos ** 3 + 3 * ppos ** 2 * (1 - ppos)
    print(f"  f_arm={f}: P(tie)={tie:.4f}, P(seed>0)={ppos:.4f}, P(>=2/3, rho=0)={p23:.4f}, P(>=2/3, rho=1)={ppos:.4f}")
json.dump(res, open(f"{HERE}/c6_out.json", "w"), indent=1)
