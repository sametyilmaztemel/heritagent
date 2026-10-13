"""C5 independent check of the H2 T_eval component power / MDE.

Criterion (study's reading A6): per-task D_i = mean over 3 seeds of (C_is - B_is);
pass iff mean(D) >= 0.05 AND 10k percentile cluster-bootstrap LB > 0 AND
per-seed mean(C_s - B_s) > 0 in >= 2 of 3 seeds.

Arm model (re-implemented from the study's text, A2/A7/A8):
  Y0_i ~ Bern(p0=0.4) shared by all seeds (G0 at temp 0).
  B_is = Y0_i XOR FB_is ; C_is = Cpre_is XOR FC_is, Cpre = 1 if Y0=1 else Bern(delta/(1-p0)).
  noise 'sym': F ~ Bern(f) independent per arm. noise 'redraw': w.p. f the arm outcome
  is redrawn from Bern(p0) [B] or Bern(p0+delta) [C].
  rho=1: all per-seed draws shared across seeds; rho=0: flips independent per seed;
  rescue either independent per seed ('rho0_indepRescue') or shared ('rho0_sharedRescue').

Methods: (A) rho=1 EXACT enumeration over (a,b) compositions with exact bootstrap tail;
(B) MC over datasets with EXACT bootstrap tail via FFT (no resampling noise; 10k-resample
RNG fuzz added as P(Bin(10000,p)<=250)); (C) closed-form normal approximation.
"""
import json, math, sys
import numpy as np
from scipy.stats import binom, norm

HERE = "<scratch>/out/repro/verify-power"
N = 134; S = 3; P0 = 0.4
NRES = 10_000; LIDX = 250


def lb_pass_prob(ptail):
    return binom.cdf(LIDX, NRES, ptail)


def exact_tail_trinomial(a, b, n):
    """P(sum of n iid draws from empirical {+1:a, -1:b, 0:rest} <= 0)."""
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


def per_task_cell_probs(delta, f, noise):
    """Joint law of (Y0, Cpre/C, B) for one seed -> P(C-B = +1, -1)."""
    r = delta / (1 - P0)
    pp = pm = 0.0
    for y0, py0 in ((1, P0), (0, 1 - P0)):
        for cpre, pc in (((1, 1.0),) if y0 == 1 else ((1, r), (0, 1 - r))):
            if noise == "sym":
                pC1 = cpre * (1 - f) + (1 - cpre) * f
                pB1 = y0 * (1 - f) + (1 - y0) * f
            else:  # redraw
                pC1 = (1 - f) * cpre + f * (P0 + delta)
                pB1 = (1 - f) * y0 + f * P0
            w = py0 * pc
            pp += w * pC1 * (1 - pB1)
            pm += w * (1 - pC1) * pB1
    return pp, pm


def rho1_exact(delta, f, noise="sym"):
    pp, pm = per_task_cell_probs(delta, f, noise)
    tot_full = tot_lb = 0.0
    for a in range(N + 1):
        pa_marg = None
        for b in range(N + 1 - a):
            w = math.exp(math.lgamma(N + 1) - math.lgamma(a + 1) - math.lgamma(b + 1) - math.lgamma(N - a - b + 1)
                         + (a * math.log(pp) if a else 0) + (b * math.log(pm) if b else 0)
                         + ((N - a - b) * math.log(1 - pp - pm) if N - a - b else 0))
            if w < 1e-14:
                continue
            if a <= b:
                continue  # LB>0 and direction both impossible
            pl = lb_pass_prob(exact_tail_trinomial(a, b, N))
            tot_lb += w * pl
            if (a - b) / N >= 0.05:
                tot_full += w * pl
    mu = pp - pm
    return dict(full=tot_full, lb=tot_lb, realized=mu)


def simulate(delta, f, mode, noise="sym", reps=4000, rng=None):
    r = delta / (1 - P0)
    Y0 = (rng.random((reps, N, 1)) < P0).astype(np.int8)
    shape_seed = (reps, N, S)
    def draw(p, shared):
        if shared:
            u = rng.random((reps, N, 1)) < p
            return np.broadcast_to(u, shape_seed).astype(np.int8)
        return (rng.random(shape_seed) < p).astype(np.int8)
    shared_flip = mode == "rho1"
    shared_rescue = mode in ("rho1", "rho0_sharedRescue")
    resc = draw(r, shared_rescue)
    Cpre = np.maximum(np.broadcast_to(Y0, shape_seed), resc)
    Y0b = np.broadcast_to(Y0, shape_seed)
    if noise == "sym":
        FB = draw(f, shared_flip); FC = draw(f, shared_flip)
        B = Y0b ^ FB; C = Cpre ^ FC
    else:
        RB = draw(f, shared_flip); RC = draw(f, shared_flip)
        nb = draw(P0, shared_flip); nc = draw(P0 + delta, shared_flip)
        B = np.where(RB == 1, nb, Y0b); C = np.where(RC == 1, nc, Cpre)
    Dk = (C.astype(np.int16) - B.astype(np.int16)).sum(axis=2)  # in units of 1/3: -3..3
    point = Dk.sum(axis=1) / (3 * N)
    seed_means = (C.astype(np.int16) - B.astype(np.int16)).sum(axis=1)  # reps x S
    dir_ok = (seed_means > 0).sum(axis=1) >= 2
    # empirical distribution over k in -3..3
    counts = np.stack([(Dk == k).sum(axis=1) for k in range(-3, 4)], axis=1) / N  # reps x 7
    L = 1024
    ks = np.arange(-3, 4)
    w = 2 * np.pi * np.arange(L) / L
    phi = counts @ np.exp(-1j * np.outer(ks, w))           # reps x L
    pmf = np.real(np.fft.ifft(phi ** N, axis=1))           # index m <-> sum = m (mod L), sums in [-402, 402]
    # P(sum <= 0): indices 0 (sum 0) and L-402..L-1 (negative sums)
    ptail = pmf[:, 0] + pmf[:, L - 3 * N:].sum(axis=1)
    ptail = np.clip(ptail, 0, 1)
    plb = lb_pass_prob(ptail)
    full = np.mean(plb * (point >= 0.05) * dir_ok)
    lbonly = np.mean(plb * dir_ok)
    return dict(full=float(full), lb_dir=float(lbonly), lb=float(np.mean(plb)),
                dir=float(np.mean(dir_ok)), point=float(np.mean(point >= 0.05)),
                realized=float(point.mean()), sd=float(point.std()))


def normal_approx(delta, f, mode, noise="sym"):
    """Var of per-task seed-averaged D from exact one-task enumeration (MC-free)."""
    # enumerate one task: Y0, rescue per seed (shared or not), flips per seed (shared or not)
    rng = np.random.default_rng(5)
    # cheap exact-by-simulation of one task's law (1e6 draws) -- only to get mu, var
    res = simulate(delta, f, mode, noise, reps=1, rng=rng) if False else None
    M = 400_000
    r = delta / (1 - P0)
    Y0 = rng.random((M, 1)) < P0
    def draw(p, shared):
        return np.broadcast_to(rng.random((M, 1)) < p, (M, S)) if shared else (rng.random((M, S)) < p)
    sf = mode == "rho1"; sr = mode in ("rho1", "rho0_sharedRescue")
    Cpre = Y0 | draw(r, sr)
    if noise == "sym":
        B = Y0 ^ draw(f, sf); C = Cpre ^ draw(f, sf)
    else:
        B = np.where(draw(f, sf), draw(P0, sf), np.broadcast_to(Y0, (M, S)))
        C = np.where(draw(f, sf), draw(P0 + delta, sf), Cpre)
    D = (C.astype(float) - B.astype(float)).mean(axis=1)
    mu, sd = D.mean(), D.std()
    se = sd / math.sqrt(N)
    p_lb = norm.cdf(mu / se - 1.96)
    thr = max(0.05, 1.96 * se)
    p_full = norm.cdf((mu - thr) / se)
    return dict(mu=mu, sd=sd, se=se, lb=p_lb, full=p_full)


def mde(xs, ys, realized, target=0.8):
    for i in range(1, len(xs)):
        if ys[i - 1] < target <= ys[i]:
            t = (target - ys[i - 1]) / (ys[i] - ys[i - 1])
            return xs[i - 1] + t * (xs[i] - xs[i - 1]), realized[i - 1] + t * (realized[i] - realized[i - 1])
    return None, None


if __name__ == "__main__":
    rng = np.random.default_rng(20261013)
    grid = [round(x, 3) for x in np.arange(0.0, 0.62, 0.02)]
    results = {}
    for noise in ("sym", "redraw"):
        for f in (0.1, 0.2, 0.3):
            # rho=1 exact
            xs, yf, yl, rz = [], [], [], []
            for d in grid:
                if d > 1 - P0: break
                e = rho1_exact(d, f, noise)
                xs.append(d); yf.append(e["full"]); yl.append(e["lb"]); rz.append(e["realized"])
                if e["full"] > 0.95: break
            mn, mr = mde(xs, yf, rz)
            results[f"{noise}|rho1|f={f}"] = dict(delta=xs, full=yf, lb=yl, realized=rz, mde_nominal=mn, mde_realized=mr)
            print(f"[{noise}] rho=1 exact f={f}: MDE80 nominal={mn} realized={mr}; null full={yf[0]:.4f} lb={yl[0]:.4f}")
            for mode in ("rho0_indepRescue", "rho0_sharedRescue"):
                xs, yf, yl, rz, dirs = [], [], [], [], []
                for d in grid:
                    if d > 1 - P0: break
                    s = simulate(d, f, mode, noise, reps=3000, rng=rng)
                    xs.append(d); yf.append(s["full"]); yl.append(s["lb_dir"]); rz.append(s["realized"]); dirs.append(s["dir"])
                    if s["full"] > 0.95: break
                mn, mr = mde(xs, yf, rz)
                results[f"{noise}|{mode}|f={f}"] = dict(delta=xs, full=yf, lb_dir=yl, realized=rz, dir=dirs, mde_nominal=mn, mde_realized=mr)
                print(f"[{noise}] {mode} f={f}: MDE80 nominal={mn} realized={mr}; null full={yf[0]:.4f} dir(null)={dirs[0]:.3f}")
    # MC check of rho1 exact on a couple of cells
    for (d, f) in ((0.10, 0.1), (0.15, 0.1), (0.20, 0.2)):
        s = simulate(d, f, "rho1", "sym", reps=4000, rng=rng)
        e = rho1_exact(d, f, "sym")
        print(f"rho1 cross-check d={d} f={f}: MC full={s['full']:.4f} exact full={e['full']:.4f}; MC lb={s['lb']:.4f} exact lb={e['lb']:.4f}")
    # study spot cells
    print("\nStudy spot cells (sym):")
    for mode, d, f, study in (("rho0_indepRescue", 0.10, 0.1, 0.877), ("rho0_sharedRescue", 0.10, 0.1, 0.877),
                              ("rho0_indepRescue", 0.15, 0.2, 0.841), ("rho0_sharedRescue", 0.15, 0.2, 0.841),
                              ("rho1", 0.10, 0.1, 0.444), ("rho1", 0.15, 0.1, 0.741), ("rho1", 0.20, 0.2, 0.592)):
        if mode == "rho1":
            v = rho1_exact(d, f)["full"]
        else:
            v = simulate(d, f, mode, "sym", reps=6000, rng=rng)["full"]
        print(f"  {mode} nominal={d} f={f}: mine={v:.3f} study={study}")
    # Normal approximation MDE (realized) via root-finding on the delta grid
    print("\nNormal approximation (seed-averaged per-task variance; LB>0 power; full = also point>=5pp):")
    na = {}
    for noise in ("sym", "redraw"):
        for mode in ("rho1", "rho0_indepRescue", "rho0_sharedRescue"):
            for f in (0.1, 0.2, 0.3):
                xs, yl, yf, rz = [], [], [], []
                for d in np.arange(0.0, 0.601, 0.01):
                    o = normal_approx(float(d), f, mode, noise)
                    xs.append(float(d)); yl.append(o["lb"]); yf.append(o["full"]); rz.append(o["mu"])
                    if o["full"] > 0.9: break
                mn, mr = mde(xs, yf, rz)
                mnl, mrl = mde(xs, yl, rz)
                na[f"{noise}|{mode}|f={f}"] = dict(mde_full_nominal=mn, mde_full_realized=mr, mde_lb_nominal=mnl, mde_lb_realized=mrl,
                                                    sd_at_null=normal_approx(0.0, f, mode, noise)["sd"])
                print(f"  [{noise}] {mode:18s} f={f}: realized MDE full={mr if mr is None else round(mr,4)} lb={mrl if mrl is None else round(mrl,4)} (nominal {mn if mn is None else round(mn,3)})")
    # alternative estimator reading: pool 402 (task,seed) rows as independent clusters (anti-conservative if rho=1)
    print("\nAlt reading: 402 pooled rows as independent (normal approx, sym, LB only):")
    for f in (0.1, 0.2, 0.3):
        # per-row var = var of single-seed C-B
        def pw(d):
            pp, pm = per_task_cell_probs(d, f, "sym"); mu = pp - pm; var = pp + pm - mu ** 2
            return norm.cdf(mu / math.sqrt(var / 402) - 1.96), mu
        ds = np.arange(0, 0.6, 0.005); vals = [pw(d) for d in ds]
        for d, (p, mu) in zip(ds, vals):
            if p >= 0.8:
                print(f"  f={f}: realized MDE(LB) ~ {mu:.4f} (nominal {d:.3f})"); break
    json.dump(dict(results=results, normal=na), open(f"{HERE}/c5_out.json", "w"), indent=1, default=float)
