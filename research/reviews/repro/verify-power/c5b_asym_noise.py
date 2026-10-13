"""Adversarial C5: unequal arm noise (arm C has few genes, so f_C < f_B). Exact per-task law,
normal approximation of LB>0 power, sym noise, p0=0.4, rho in {0 (indep flips+rescue), 1}."""
import itertools, math
import numpy as np
from scipy.stats import norm
P0 = 0.4; N = 134; S = 3
def task_moments(delta, fB, fC, rho):
    r = delta / (1 - P0)
    # single-seed joint law of (C-B) given Y0 ; seeds iid given Y0 when rho=0, identical when rho=1
    mus, m2s = [], []
    out_mu = 0.0; out_var_terms = 0.0
    E = 0.0; E2 = 0.0
    for y0, py in ((1, P0), (0, 1 - P0)):
        # distribution of d = C-B for one seed given y0
        pd = {-1: 0.0, 0: 0.0, 1: 0.0}
        for cpre, pc in (((1, 1.0),) if y0 else ((1, r), (0, 1 - r))):
            pC1 = cpre * (1 - fC) + (1 - cpre) * fC
            pB1 = y0 * (1 - fB) + (1 - y0) * fB
            pd[1] += pc * pC1 * (1 - pB1); pd[-1] += pc * (1 - pC1) * pB1
        pd[0] = 1 - pd[1] - pd[-1]
        m = pd[1] - pd[-1]; v1 = pd[1] + pd[-1] - m * m
        # per-task D = mean of S seeds
        varD = v1 if rho == 1 else v1 / S
        E += py * m; E2 += py * (varD + m * m)
    return E, E2 - E * E
def mde(fB, fC, rho):
    for d in np.arange(0, 0.6, 0.001):
        mu, var = task_moments(d, fB, fC, rho)
        se = math.sqrt(var / N)
        if norm.cdf(mu / se - 1.96) >= 0.8 and mu >= 0.05:
            return round(d, 3), round(mu, 4)
    return None
for rho in (0, 1):
    for fB in (0.1, 0.2, 0.3):
        cells = [f"fC={fC}: {mde(fB, fC, rho)}" for fC in (fB, 0.05, 0.0)]
        print(f"rho={rho} fB={fB}: (nominal, realized) MDE80 ->", " | ".join(cells))
