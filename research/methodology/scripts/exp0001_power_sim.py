#!/usr/bin/env python
"""EXP-0001 operating-characteristics / power simulation (LOCKED design).

Reproduces, under explicit assumed effect models, the operating
characteristics of every locked EXP-0001 gate rule and of the H1/H2 support
conditions. It does NOT touch the repository, GitHub, or any model; it is a
pure numpy/scipy simulation with fixed seeds.

Sources of the locked rules (read-only, commit 0ce0a36 on main and PR #28
branch origin/issue-12-cig-gate head 7886852):
  experiments/manifests/EXP-0001-design.md  (sections 4-5)
  experiments/manifests/EXP-0001-minimal-cig.yaml
  research/decisions/ADR-0002-exp-0001-design.md (+ Amendment A1 on PR #28)
  inheritance/bootstrap.py  (percentile paired cluster bootstrap, 10k, seed 0)
  inheritance/replay.py     (>=3/5 with, net >=+2/5, expression check)
  inheritance/ablation.py   (n=20, point>=0.05 AND LB>0)
  inheritance/generalization.py (PR #28: n=10 in >=0, n=10 out >=-0.05)
  inheritance/regression.py     (PR #28: 40 T_reg, baseline=1, drop<=0.03)
  inheritance/interaction.py + gate.py (PR #28: both-vs-neither, n=10,
                                        harmful iff delta_pair < -0.05)

Paired-outcome generative model (per task instance, binary success):
  Y0 ~ Bernoulli(p0)                                   (without trait)
  effect step:  if delta>0 and Y0=0: rescue w.p. r = delta/(1-p0)
                if delta<0 and Y0=1: break  w.p. h = -delta/p0
  noise step (the deterministic-trajectory perturbation any added prompt
  text induces, even for a truly neutral trait):
     symmetric: Y1 = 1 - Y1'            w.p. f
     redraw:    Y1 ~ Bernoulli(p0+delta) w.p. f   (exactly mean-neutral)
  T_reg tasks have Y0 = 1 by construction (G0 solved them at temp 0).

Everything is parameterised through CLI flags; results are written as JSON
and as markdown tables. Runtime target: < 10 min on 4 cores at defaults.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import random
import sys
import time

import numpy as np
from scipy.stats import binom, multinomial

# ----------------------------------------------------------------------------
# Locked constants (mirrors of the repository constants; DO NOT TUNE)
# ----------------------------------------------------------------------------
REPLAY_N = 5
REPLAY_MIN_WITH = 3
REPLAY_MIN_NET = 2
ABL_N = 20
TAU_C = 0.05
BOOT_N = 10_000
BOOT_SEED = 0
GEN_N_IN = 10
GEN_N_OUT = 10
GEN_IN_MIN = 0.0
GEN_OUT_MIN = -0.05
REG_N = 40
TAU_R = 0.03
INT_N = 10
INT_HARM = -0.05
T_EVAL_N = 134
N_SEEDS = 3
H2_MIN_EFFECT = 0.05
K_CANDIDATES = 10


# ----------------------------------------------------------------------------
# Verbatim copy of inheritance/bootstrap.py::paired_cluster_bootstrap
# (commit 0ce0a36) used ONLY to validate the exact pass-probability table.
# ----------------------------------------------------------------------------
def paired_cluster_bootstrap_repo(deltas, *, seed, n_resamples=BOOT_N,
                                  confidence=0.95):
    n = len(deltas)
    point = sum(deltas) / n
    rng = random.Random(seed)
    means = []
    for _ in range(n_resamples):
        total = 0.0
        for _ in range(n):
            total += deltas[rng.randrange(n)]
        means.append(total / n)
    means.sort()
    alpha = 1.0 - confidence
    lower_index = max(0, int(alpha / 2 * n_resamples))
    upper_index = min(n_resamples - 1, int((1 - alpha / 2) * n_resamples))
    return point, means[lower_index], means[upper_index]


def stage_passes_repo(point, lb, tau_c=TAU_C):
    return point >= tau_c and lb > 0.0


# ----------------------------------------------------------------------------
# Generative model: paired cell probabilities
# ----------------------------------------------------------------------------
def cell_probs(p0, delta, f, model="symmetric"):
    """Return dict {(y0,y1): prob} for one random task instance."""
    r = min(1.0, max(delta, 0.0) / (1.0 - p0)) if p0 < 1 else 0.0
    h = min(1.0, max(-delta, 0.0) / p0) if p0 > 0 else 0.0
    # pre-noise joint
    pre = {
        (0, 0): (1 - p0) * (1 - r),
        (0, 1): (1 - p0) * r,
        (1, 0): p0 * h,
        (1, 1): p0 * (1 - h),
    }
    out = {k: 0.0 for k in pre}
    p1 = min(1.0, max(0.0, p0 + delta))
    for (y0, y1p), pr in pre.items():
        if model == "symmetric":
            out[(y0, y1p)] += pr * (1 - f)
            out[(y0, 1 - y1p)] += pr * f
        elif model == "redraw":
            out[(y0, y1p)] += pr * (1 - f)
            out[(y0, 1)] += pr * f * p1
            out[(y0, 0)] += pr * f * (1 - p1)
        else:
            raise ValueError(model)
    return out


def reg_fail_prob(p0, delta, f, model="symmetric"):
    """P(with-trait failure | Y0 = 1) on a T_reg task."""
    h = min(1.0, max(-delta, 0.0) / p0) if p0 > 0 else 0.0
    if model == "symmetric":
        return h * (1 - f) + (1 - h) * f
    p1 = min(1.0, max(0.0, p0 + delta))
    return h * (1 - f) + f * (1 - p1)


def realized_effect(cp):
    return cp[(0, 1)] - cp[(1, 0)]


# ----------------------------------------------------------------------------
# Exact stage operating characteristics
# ----------------------------------------------------------------------------
def compositions(n, k):
    """All k-tuples of non-negative ints summing to n."""
    if k == 1:
        yield (n,)
        return
    for i in range(n + 1):
        for rest in compositions(n - i, k - 1):
            yield (i,) + rest


def replay_pass_prob(cp, expression=1.0, source_solved=False):
    """Exact P(stage-1 pass). Cells ordered (00,01,10,11).
    source_solved: the source instance has Y0=1 fixed (trait mined from a
    G0 success at temp 0 => identical without-trait rerun)."""
    probs = [cp[(0, 0)], cp[(0, 1)], cp[(1, 0)], cp[(1, 1)]]
    n_var = REPLAY_N - 1 if source_solved else REPLAY_N
    # source instance distribution when solved: Y0=1 -> (1,0) or (1,1)
    if source_solved:
        p_src_11 = cp[(1, 1)] / (cp[(1, 0)] + cp[(1, 1)])
        src_options = [((1, 1), p_src_11), ((1, 0), 1 - p_src_11)]
    else:
        src_options = [(None, 1.0)]
    total = 0.0
    for src, psrc in src_options:
        for comp in compositions(n_var, 4):
            pr = multinomial.pmf(comp, n_var, probs) * psrc
            n00, n01, n10, n11 = comp
            if src == (1, 1):
                n11 += 1
            elif src == (1, 0):
                n10 += 1
            s_with = n01 + n11
            s_without = n10 + n11
            if s_with >= REPLAY_MIN_WITH and (s_with - s_without) >= REPLAY_MIN_NET:
                total += pr * expression ** s_with
    return total


def exact_p_resample_nonpositive(a, b, n=ABL_N):
    """P(resampled mean <= 0) when the delta multiset has a (+1), b (-1),
    n-a-b (0): (A,B,C) ~ Multinomial(n; a/n, b/n, c/n); P(A <= B)."""
    c = n - a - b
    pa, pb, pc = a / n, b / n, c / n
    total = 0.0
    for A in range(n + 1):
        for B in range(n - A + 1):
            if A <= B:
                C = n - A - B
                total += multinomial.pmf([A, B, C], n, [pa, pb, pc])
    return total


def ablation_pass_table(tau_c=TAU_C, n=ABL_N, n_boot=BOOT_N):
    """pass_prob[a][b] = P(stage-2 pass | a rescues, b breaks).
    LB>0 iff fewer than lower_index+1 resample means are <= 0; the count of
    such means is Binomial(n_boot, p_ab) over the RNG stream (ordering of
    the delta vector acts as the randomisation), so
    P(LB>0 | a,b) = P(Bin(n_boot, p_ab) <= lower_index)."""
    lower_index = int(0.025 * n_boot)
    tab = np.zeros((n + 1, n + 1))
    pab = np.zeros((n + 1, n + 1))
    for a in range(n + 1):
        for b in range(n + 1 - a):
            point = (a - b) / n
            if not (point >= tau_c):
                continue
            p = exact_p_resample_nonpositive(a, b, n)
            pab[a, b] = p
            tab[a, b] = binom.cdf(lower_index, n_boot, p)
    return tab, pab


def ablation_pass_prob(cp, tab, n=ABL_N):
    probs = [cp[(0, 1)], cp[(1, 0)], cp[(0, 0)] + cp[(1, 1)]]
    total = 0.0
    for a in range(n + 1):
        for b in range(n + 1 - a):
            if tab[a, b] == 0.0:
                continue
            c = n - a - b
            total += multinomial.pmf([a, b, c], n, probs) * tab[a, b]
    return total


def net_threshold_pass_prob(cp, n, min_delta):
    """P(net/n >= min_delta) for n paired instances (stage 3 both scopes)."""
    probs = [cp[(0, 1)], cp[(1, 0)], cp[(0, 0)] + cp[(1, 1)]]
    total = 0.0
    for a in range(n + 1):
        for b in range(n + 1 - a):
            if (a - b) / n >= min_delta:
                total += multinomial.pmf([a, b, n - a - b], n, probs)
    return total


def regression_pass_prob(h_reg, n=REG_N, tau_r=TAU_R):
    max_fail = math.floor(tau_r * n + 1e-9)
    return binom.cdf(max_fail, n, h_reg)


# ----------------------------------------------------------------------------
# Monte Carlo: end-to-end gate + interaction stage + arm-C composition
# ----------------------------------------------------------------------------
def sample_ablation_outcome(rng, cp, tab, n=ABL_N):
    """Sample (a,b) and the pass decision for one trait."""
    probs = [cp[(0, 1)], cp[(1, 0)], cp[(0, 0)] + cp[(1, 1)]]
    a, b, _ = rng.multinomial(n, probs)
    passed = rng.random() < tab[a, b]
    return (a - b) / n, passed


def with_trait_outcome(rng, y0, delta, f, p0, model):
    """Vectorised transform of a without-trait outcome vector."""
    r = min(1.0, max(delta, 0.0) / (1.0 - p0)) if p0 < 1 else 0.0
    h = min(1.0, max(-delta, 0.0) / p0) if p0 > 0 else 0.0
    u = rng.random(y0.shape)
    y1 = np.where(y0 == 0, u < r, ~(u < h)).astype(np.int8)
    un = rng.random(y0.shape)
    if model == "symmetric":
        y1 = np.where(un < f, 1 - y1, y1)
    else:
        p1 = min(1.0, max(0.0, p0 + delta))
        ur = rng.random(y0.shape)
        y1 = np.where(un < f, (ur < p1).astype(np.int8), y1)
    return y1.astype(np.int8)


def e2e_simulation(rng, *, p0, f, model, mixture, delta_help, delta_harm,
                   tab, n_reps, expression=1.0, source_solved=False):
    """Simulate K=10 candidates through all stages incl. A1 interaction."""
    types = np.array([0, 1, 2])  # 0 helpful, 1 neutral, 2 harmful
    deltas = {0: delta_help, 1: 0.0, 2: delta_harm}
    stage_p = {}
    for t, d in deltas.items():
        cp = cell_probs(p0, d, f, model)
        stage_p[t] = {
            "replay": replay_pass_prob(cp, expression, source_solved),
            "gen_in": net_threshold_pass_prob(cp, GEN_N_IN, GEN_IN_MIN),
            "gen_out": net_threshold_pass_prob(cp, GEN_N_OUT, GEN_OUT_MIN),
            "reg": regression_pass_prob(reg_fail_prob(p0, d, f, model)),
            "cp": cp,
        }
    out = {
        "n_admitted": [], "n_help": [], "n_neut": [], "n_harm": [],
        "n_eligible": [], "n_eliminated": [], "any_nonhelp": 0,
        "empty": 0, "n_pairs": [],
    }
    for _ in range(n_reps):
        ttypes = rng.choice(types, size=K_CANDIDATES, p=mixture)
        eligible = []
        contrib = {}
        for i, t in enumerate(ttypes):
            sp = stage_p[t]
            ok = rng.random() < sp["replay"]
            c, abl_ok = sample_ablation_outcome(rng, sp["cp"], tab)
            ok = ok and abl_ok
            ok = ok and rng.random() < sp["gen_in"]
            ok = ok and rng.random() < sp["gen_out"]
            ok = ok and rng.random() < sp["reg"]
            contrib[i] = c
            if ok:
                eligible.append(i)
        eliminated = set()
        n_pairs = 0
        if len(eligible) >= 2:
            for i, j in itertools.combinations(eligible, 2):
                n_pairs += 1
                y0 = (rng.random(INT_N) < p0).astype(np.int8)
                # both traits: effect i, effect j, noise twice (two prompts)
                yb = with_trait_outcome(rng, y0, deltas[ttypes[i]], 0.0, p0, model)
                yb = with_trait_outcome(rng, yb, deltas[ttypes[j]], 0.0, p0, model)
                f_pair = 1 - (1 - f) ** 2
                yb = with_trait_outcome(rng, yb, 0.0, f_pair, p0, model)
                dpair = (yb.astype(float) - y0).mean()
                if dpair < INT_HARM:
                    # eliminate lower contribution; tie -> earlier rank (lower index)
                    if contrib[i] > contrib[j]:
                        eliminated.add(j)
                    elif contrib[j] > contrib[i]:
                        eliminated.add(i)
                    else:
                        eliminated.add(max(i, j))
        admitted = [i for i in eligible if i not in eliminated]
        adm_types = [ttypes[i] for i in admitted]
        out["n_admitted"].append(len(admitted))
        out["n_eligible"].append(len(eligible))
        out["n_eliminated"].append(len(eliminated))
        out["n_pairs"].append(n_pairs)
        out["n_help"].append(sum(1 for t in adm_types if t == 0))
        out["n_neut"].append(sum(1 for t in adm_types if t == 1))
        out["n_harm"].append(sum(1 for t in adm_types if t == 2))
        if any(t != 0 for t in adm_types):
            out["any_nonhelp"] += 1
        if not admitted:
            out["empty"] += 1
    n_adm = np.array(out["n_admitted"])
    n_help = np.array(out["n_help"])
    nonempty = n_adm > 0
    prec = (n_help[nonempty] / n_adm[nonempty]).mean() if nonempty.any() else float("nan")
    return {
        "E_admitted": float(n_adm.mean()),
        "P_empty": out["empty"] / n_reps,
        "E_helpful": float(n_help.mean()),
        "E_neutral": float(np.mean(out["n_neut"])),
        "E_harmful": float(np.mean(out["n_harm"])),
        "P_any_nonhelpful_admitted": out["any_nonhelp"] / n_reps,
        "P_any_harmful_admitted": float(np.mean(np.array(out["n_harm"]) > 0)),
        "precision_given_nonempty": float(prec),
        "E_eligible_pre_interaction": float(np.mean(out["n_eligible"])),
        "E_eliminated_by_interaction": float(np.mean(out["n_eliminated"])),
        "P_any_eliminated": float(np.mean(np.array(out["n_eliminated"]) > 0)),
        "stage_pass_by_type": {
            {0: "helpful", 1: "neutral", 2: "harmful"}[t]: {
                k: float(v) for k, v in sp.items() if k != "cp"}
            for t, sp in stage_p.items()},
    }


# ----------------------------------------------------------------------------
# Monte Carlo: H1 / H2 decision power with cluster bootstrap over tasks
# ----------------------------------------------------------------------------
def boot_lb(rng, d, n_boot, chunk=64, alpha=0.05):
    """Percentile bootstrap lower bound of the mean, per row of d (R, n),
    with the same index convention as inheritance/bootstrap.py."""
    R, n = d.shape
    li = max(0, int(alpha / 2 * n_boot))
    lbs = np.empty(R)
    for s in range(0, R, chunk):
        dd = d[s:s + chunk]
        c = dd.shape[0]
        idx = rng.integers(0, n, size=(c, n_boot, n))
        samp = np.take_along_axis(dd[:, None, :], idx, axis=2)
        means = samp.mean(axis=2)
        means.sort(axis=1)
        lbs[s:s + c] = means[:, li]
    return lbs


def draw_uniform_with_sharing(rng, R, n, S, rho):
    """Uniforms (R, n, S) shared across seeds with prob rho per task."""
    u_sh = rng.random((R, n, 1))
    u_in = rng.random((R, n, S))
    mask = rng.random((R, n, 1)) < rho
    return np.where(mask, u_sh, u_in)


def arm_outcomes(rng, ya, delta, f, p0, rho, model, S=N_SEEDS):
    """ya: (R, n) without-inheritance outcomes; returns (R, n, S)."""
    R, n = ya.shape
    r = min(1.0, max(delta, 0.0) / (1.0 - p0)) if p0 < 1 else 0.0
    h = min(1.0, max(-delta, 0.0) / p0) if p0 > 0 else 0.0
    y0 = ya[:, :, None]
    u = draw_uniform_with_sharing(rng, R, n, S, rho)
    y1 = np.where(y0 == 0, u < r, ~(u < h)).astype(np.int8)
    un = draw_uniform_with_sharing(rng, R, n, S, rho)
    if model == "symmetric":
        y1 = np.where(un < f, 1 - y1, y1)
    else:
        p1 = min(1.0, max(0.0, p0 + delta))
        ur = draw_uniform_with_sharing(rng, R, n, S, rho)
        y1 = np.where(un < f, (ur < p1).astype(np.int8), y1)
    return y1.astype(np.int8)


def h2_eval_part(rng, *, R, p0, delta_B, delta_C, f_B, f_C, rho, model, n_boot):
    ya = (rng.random((R, T_EVAL_N)) < p0).astype(np.int8)
    yb = arm_outcomes(rng, ya, delta_B, f_B, p0, rho, model)
    yc = arm_outcomes(rng, ya, delta_C, f_C, p0, rho, model)
    diff = yc.astype(float) - yb  # (R, n, S)
    d_task = diff.mean(axis=2)     # cluster = task, averaged over seeds
    point = d_task.mean(axis=1)
    lb = boot_lb(rng, d_task, n_boot)
    seed_dir = (diff.mean(axis=1) > 0).sum(axis=1)  # (R,)
    c_point = point >= H2_MIN_EFFECT
    c_lb = lb > 0
    c_seed = seed_dir >= 2
    return {
        "P_point_ge_5pp": float(c_point.mean()),
        "P_lb_gt_0": float(c_lb.mean()),
        "P_seed_dir_2of3": float(c_seed.mean()),
        "P_eval_part": float((c_point & c_lb & c_seed).mean()),
        "P_lb_gt_0_and_seed": float((c_lb & c_seed).mean()),
        "realized_effect_mean": float(point.mean()),
        "sd_point": float(point.std()),
    }


def h2_reg_part(rng, *, R, r_B, r_C, rho, n_boot, corr=0.0):
    """Paired T_reg failure indicators per seed. corr>0 induces positive
    dependence (a C-failure implies a B-failure w.p. corr)."""
    uB = draw_uniform_with_sharing(rng, R, REG_N, N_SEEDS, rho)
    uC = draw_uniform_with_sharing(rng, R, REG_N, N_SEEDS, rho)
    fC = (uC < r_C)
    if corr > 0 and r_B > 0:
        # couple: where C fails, B fails w.p. max(corr, r_B)
        uX = draw_uniform_with_sharing(rng, R, REG_N, N_SEEDS, rho)
        fB = np.where(fC, uX < max(corr, r_B), uB < r_B)
    else:
        fB = (uB < r_B)
    diff = fB.astype(float) - fC.astype(float)
    d_task = diff.mean(axis=2)
    point = d_task.mean(axis=1)
    lb = boot_lb(rng, d_task, n_boot)
    ok = (point > 0) & (lb > 0)
    return {"P_reg_part": float(ok.mean()),
            "P_reg_point_gt_0": float((point > 0).mean())}


def h1_part(rng, *, R, p0, delta_C, f_C, rho, model, n_boot):
    ya = (rng.random((R, T_EVAL_N)) < p0).astype(np.int8)
    yc = arm_outcomes(rng, ya, delta_C, f_C, p0, rho, model)
    diff = yc.astype(float) - ya[:, :, None]
    d_task = diff.mean(axis=2)
    lb = boot_lb(rng, d_task, n_boot)
    point = d_task.mean(axis=1)
    return {"P_h1": float(((lb > 0) & (point > 0)).mean()),
            "realized_effect_mean": float(point.mean())}


# ----------------------------------------------------------------------------
# Markdown helpers
# ----------------------------------------------------------------------------
def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out)


def pct(x):
    return f"{100 * x:.1f}%"


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=20261012)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..",
                                                  "exp0001_power_sim_results"))
    ap.add_argument("--n-reps", type=int, default=1000, help="MC reps per H1/H2 cell")
    ap.add_argument("--n-boot", type=int, default=1000,
                    help="bootstrap resamples per MC rep (locked protocol is 10000; "
                         "see --validate for a 10k check)")
    ap.add_argument("--e2e-reps", type=int, default=3000)
    ap.add_argument("--p0", type=float, nargs="+", default=[0.3, 0.5],
                    help="UNVERIFIED base success grid (no prompting-only Qwen2.5-7B "
                         "ALFWorld number exists in the repo)")
    ap.add_argument("--f", type=float, nargs="+", default=[0.0, 0.05, 0.10, 0.20])
    ap.add_argument("--delta", type=float, nargs="+",
                    default=[-0.10, -0.05, 0.0, 0.05, 0.10, 0.20, 0.30, 0.40])
    ap.add_argument("--model", choices=["symmetric", "redraw"], default="symmetric")
    ap.add_argument("--skip-validate", action="store_true")
    ap.add_argument("--skip-h", action="store_true", help="skip H1/H2 MC")
    ap.add_argument("--skip-e2e", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    results = {"args": vars(args), "locked": {
        "replay": "S_with>=3/5 and net>=+2/5 and expression",
        "ablation": "n=20 point>=0.05 and percentile-bootstrap LB(10k, idx 250)>0",
        "gen_in": "n=10 net/10>=0", "gen_out": "n=10 net/10>=-0.05 (== net>=0)",
        "regression": "40 T_reg baseline=1, 1-S/40<=0.03 (== <=1 failure)",
        "interaction": "A1 both-vs-neither n=10, harmful iff delta<-0.05 (== net<=-1)",
        "h2": "C-B>=+5pp, LB>0, reg(B)-reg(C) LB>0, direction in >=2/3 seeds",
        "h1": "C-A LB>0"}}
    md = []

    # ---------------- Section 0: exact ablation pass table ----------------
    tab, pab = ablation_pass_table()
    results["ablation_table"] = {
        f"a={a},b={b}": {"p_resample_nonpos": float(pab[a, b]), "P_pass": float(tab[a, b])}
        for a in range(ABL_N + 1) for b in range(ABL_N + 1 - a) if (a - b) / ABL_N >= TAU_C}
    rows = []
    for a in range(1, 11):
        for b in range(0, 5):
            if a - b >= 1:
                rows.append((a, b, f"{(a-b)/ABL_N:+.2f}", f"{pab[a,b]:.4f}", pct(tab[a, b])))
    md.append("### T0. Stage-2 (ablation, n=20) exact pass probability by observed (rescues a, breaks b)\n")
    md.append("P(resample mean <= 0) is exact (trinomial); P(pass) = P(Bin(10000, p) <= 250) "
              "— i.e. the seeded 10k bootstrap's LB is > 0 only if fewer than 251 of the "
              "10k resample means are <= 0.\n")
    md.append(md_table(["a (+1)", "b (-1)", "point", "P(resample<=0)", "P(stage-2 pass)"], rows))
    # minimal net needed with b=0
    min_a = next(a for a in range(ABL_N + 1) if tab[a, 0] > 0.5)
    results["ablation_min_rescues_zero_breaks"] = min_a
    md.append(f"\nSmallest a with b=0 giving P(pass)>50%: **a={min_a}** "
              f"(observed contribution >= {min_a/ABL_N:+.2f}).\n")

    # tau_c grid equivalence at n=20 (secondary sensitivity, analysis only)
    tau_rows = []
    for tau in (0.03, 0.05, 0.08):
        tab_t, _ = ablation_pass_table(tau_c=tau)
        min_net = next(k for k in range(ABL_N + 1) if k / ABL_N >= tau)
        min_a_t = next(a for a in range(ABL_N + 1) if tab_t[a, 0] > 0.5)
        tau_rows.append((tau, min_net, min_a_t))
    md.append("\n### T0b. Secondary tau_c grid at n=20: integer net required by the point rule vs by LB>0\n")
    md.append(md_table(["tau_c", "net needed by point>=tau_c", "net needed by LB>0 (b=0)"], tau_rows))
    tr_rows = []
    for tau in (0.02, 0.03, 0.05):
        tr_rows.append((tau, math.floor(tau * REG_N + 1e-9)))
    md.append("\n### T0c. Secondary tau_r grid at n=40: failures tolerated\n")
    md.append(md_table(["tau_r", "max T_reg failures allowed (of 40)"], tr_rows))
    results["tau_grid"] = {"tau_c": tau_rows, "tau_r": tr_rows}

    # ---------------- Section 0v: validate against repo bootstrap ----------------
    if not args.skip_validate:
        val_rows = []
        cells = [(3, 0), (4, 0), (5, 1), (6, 1), (6, 2), (7, 2), (8, 3)]
        n_ord = 25
        for a, b in cells:
            base = [1.0] * a + [-1.0] * b + [0.0] * (ABL_N - a - b)
            hits = 0
            prng = random.Random(1234 + a * 100 + b)
            for _ in range(n_ord):
                vec = base[:]
                prng.shuffle(vec)
                point, lb, _ = paired_cluster_bootstrap_repo(vec, seed=BOOT_SEED)
                hits += stage_passes_repo(point, lb)
            val_rows.append((a, b, f"{pab[a,b]:.4f}", pct(tab[a, b]), f"{hits}/{n_ord}"))
        md.append("\n### T0v. Validation of the exact table against the repository bootstrap "
                  "(verbatim inheritance/bootstrap.py, seed 0, 10k resamples, 25 random task orderings per cell)\n")
        md.append(md_table(["a", "b", "p_resample<=0", "predicted P(pass)", "repo bootstrap pass count"], val_rows))
        results["validation"] = val_rows
        print(f"[validate] done {time.time()-t0:.1f}s", file=sys.stderr)

    # ---------------- Section 1: stage OCs ----------------
    stage_rows = []
    stage_json = []
    for p0 in args.p0:
        for d in args.delta:
            for f in args.f:
                cp = cell_probs(p0, d, f, args.model)
                re = realized_effect(cp)
                p_rep = replay_pass_prob(cp)
                p_abl = ablation_pass_prob(cp, tab)
                p_in = net_threshold_pass_prob(cp, GEN_N_IN, GEN_IN_MIN)
                p_out_spec = net_threshold_pass_prob(cell_probs(p0, 0.0, f, args.model), GEN_N_OUT, GEN_OUT_MIN)
                p_out_gen = net_threshold_pass_prob(cp, GEN_N_OUT, GEN_OUT_MIN)
                h_reg = reg_fail_prob(p0, d, f, args.model)
                p_reg = regression_pass_prob(h_reg)
                p_all_gen = p_rep * p_abl * p_in * p_out_gen * p_reg
                p_all_spec = p_rep * p_abl * p_in * p_out_spec * p_reg
                stage_rows.append((p0, f"{d:+.2f}", f, f"{re:+.3f}", pct(p_rep), pct(p_abl), pct(p_in),
                                   pct(p_out_gen), pct(p_out_spec), f"{h_reg:.3f}", pct(p_reg),
                                   pct(p_all_gen), pct(p_all_spec)))
                stage_json.append({"p0": p0, "delta": d, "f": f, "realized_effect": re,
                                   "replay": p_rep, "ablation": p_abl, "gen_in": p_in,
                                   "gen_out_generalist": p_out_gen, "gen_out_specialist": p_out_spec,
                                   "h_reg": h_reg, "regression": p_reg,
                                   "all_generalist": p_all_gen, "all_specialist": p_all_spec})
    results["stage_oc"] = stage_json
    md.append(f"\n### T1. Stage operating characteristics (exact; noise model = {args.model}; "
              "expression prob = 1; source instance treated like the variants)\n")
    md.append("`E[dlt]` = realized mean with-minus-without effect including the noise bias; "
              "`out(gen)` = trait effect also applies out of scope; `out(spec)` = trait is "
              "exactly neutral out of scope; `all` = product of the five stage pass probabilities "
              "(stages use disjoint instances) before the interaction stage.\n")
    md.append(md_table(["p0", "delta", "f", "E[dlt]", "replay", "ablation", "gen-in", "out(gen)",
                        "out(spec)", "h_reg", "regression", "all(gen)", "all(spec)"], stage_rows))

    # replay sensitivities: expression prob, source solved
    rep_rows = []
    for p0 in args.p0:
        for d in (0.10, 0.20, 0.40):
            for f in (0.0, 0.10):
                cp = cell_probs(p0, d, f, args.model)
                rep_rows.append((p0, f"{d:+.2f}", f, pct(replay_pass_prob(cp)),
                                 pct(replay_pass_prob(cp, expression=0.8)),
                                 pct(replay_pass_prob(cp, source_solved=True)),
                                 pct(replay_pass_prob(cp, expression=0.8, source_solved=True))))
    md.append("\n### T1b. Stage-1 replay sensitivities: expression probability e per successful episode, "
              "and 'source instance already solved by G0' (Y0=1 fixed on the source)\n")
    md.append(md_table(["p0", "delta", "f", "e=1, src~variants", "e=0.8", "src solved", "e=0.8 & src solved"], rep_rows))
    results["replay_sensitivity"] = rep_rows

    # regression as a function of h_reg only
    hr = [0.0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20]
    md.append("\n### T1c. Stage-4 regression pass probability vs per-task T_reg failure probability h_reg "
              "(exact binomial, 40 tasks, <=1 failure allowed)\n")
    md.append(md_table(["h_reg", "P(pass) tau_r=0.03", "tau_r=0.02 (0 fails)", "tau_r=0.05 (<=2 fails)"],
                       [(h, pct(regression_pass_prob(h)), pct(regression_pass_prob(h, tau_r=0.02)),
                         pct(regression_pass_prob(h, tau_r=0.05))) for h in hr]))
    results["regression_curve"] = {str(h): float(regression_pass_prob(h)) for h in hr}

    # redraw-model regression mapping
    md.append("\n### T1d. Mapping of noise rate f to h_reg for a neutral trait under the two noise models\n")
    md.append(md_table(["f", "symmetric: h_reg", "P(pass)", "redraw p0=0.3: h_reg", "P(pass)", "redraw p0=0.5: h_reg", "P(pass)"],
                       [(f, f"{reg_fail_prob(0.5,0,f,'symmetric'):.3f}", pct(regression_pass_prob(reg_fail_prob(0.5,0,f,'symmetric'))),
                         f"{reg_fail_prob(0.3,0,f,'redraw'):.3f}", pct(regression_pass_prob(reg_fail_prob(0.3,0,f,'redraw'))),
                         f"{reg_fail_prob(0.5,0,f,'redraw'):.3f}", pct(regression_pass_prob(reg_fail_prob(0.5,0,f,'redraw'))))
                        for f in args.f]))

    # specialist mapping: in-family effect vs ablation-pool effect
    spec_rows = []
    for p0 in args.p0:
        for dfam in (0.30, 0.50, 0.80):
            for share in (1/6, 1/3, 1.0):
                for f in (0.0, 0.05):
                    d_pool = dfam * share
                    cp_f = cell_probs(p0, min(dfam, 1 - p0), f, args.model)
                    cp_p = cell_probs(p0, d_pool, f, args.model)
                    spec_rows.append((p0, f"{dfam:+.2f}", f"{share:.2f}", f"{d_pool:+.3f}", f,
                                      pct(replay_pass_prob(cp_f)), pct(ablation_pass_prob(cp_p, tab)),
                                      pct(net_threshold_pass_prob(cp_f, GEN_N_IN, GEN_IN_MIN))))
    md.append("\n### T1e. Specialist traits: replay/in-scope see the in-family effect, the stratified "
              "20-task ablation pool sees share x in-family effect\n")
    md.append(md_table(["p0", "in-family delta", "family share of pool", "pool delta", "f", "replay", "ablation", "gen-in"], spec_rows))
    results["specialist"] = spec_rows
    print(f"[stage OC] done {time.time()-t0:.1f}s", file=sys.stderr)

    # ---------------- Section 2: end-to-end + arm-C composition ----------------
    if not args.skip_e2e:
        e2e_rows = []
        e2e_json = []
        mixtures = {"20/50/30": [0.2, 0.5, 0.3], "50/30/20": [0.5, 0.3, 0.2], "10/60/30": [0.1, 0.6, 0.3]}
        for p0 in args.p0:
            for mix_name, mix in mixtures.items():
                for dh in (0.10, 0.20, 0.40):
                    for f in (0.0, 0.05, 0.10, 0.20):
                        if mix_name != "20/50/30" and (dh != 0.20 or f not in (0.0, 0.05)):
                            continue
                        res = e2e_simulation(rng, p0=p0, f=f, model=args.model, mixture=mix,
                                             delta_help=dh, delta_harm=-0.10, tab=tab,
                                             n_reps=args.e2e_reps)
                        sp = res["stage_pass_by_type"]
                        e2e_rows.append((p0, mix_name, f"{dh:+.2f}", f,
                                         f"{res['E_eligible_pre_interaction']:.2f}",
                                         f"{res['E_eliminated_by_interaction']:.2f}",
                                         f"{res['E_admitted']:.2f}", pct(res["P_empty"]),
                                         f"{res['E_helpful']:.2f}", f"{res['E_neutral']:.3f}",
                                         f"{res['E_harmful']:.3f}", pct(res["P_any_nonhelpful_admitted"]),
                                         pct(res["precision_given_nonempty"]) if not math.isnan(res["precision_given_nonempty"]) else "n/a"))
                        e2e_json.append({"p0": p0, "mixture": mix_name, "delta_help": dh, "f": f, **res})
        results["e2e"] = e2e_json
        md.append(f"\n### T2. End-to-end: K=10 candidates per seed, mixture helpful/neutral/harmful, "
                  f"harmful delta = -0.10, {args.e2e_reps} MC reps, incl. A1 interaction stage\n")
        md.append(md_table(["p0", "mix H/N/Harm", "delta_help", "f", "E[eligible]", "E[elim by interaction]",
                            "E[admitted]", "P(arm C empty)", "E[helpful adm]", "E[neutral adm]",
                            "E[harmful adm]", "P(any non-helpful adm)", "precision | nonempty"], e2e_rows))
        print(f"[e2e] done {time.time()-t0:.1f}s", file=sys.stderr)

    # ---------------- Section 3/4: H2 and H1 power ----------------
    if not args.skip_h:
        R, nb = args.n_reps, args.n_boot
        eff_grid = [0.0, 0.02, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20]
        farm_grid = [0.10, 0.20, 0.30]
        rho_grid = [0.0, 1.0]
        p0_h = 0.4  # UNVERIFIED anchor for arm-level analyses
        h2_rows, h2_json = [], []
        for rho in rho_grid:
            for fa in farm_grid:
                for eff in eff_grid:
                    res = h2_eval_part(rng, R=R, p0=p0_h, delta_B=0.0, delta_C=eff, f_B=fa, f_C=fa,
                                       rho=rho, model=args.model, n_boot=nb)
                    h2_rows.append((rho, fa, f"{eff:+.2f}", f"{res['realized_effect_mean']:+.3f}",
                                    f"{res['sd_point']:.3f}", pct(res["P_point_ge_5pp"]), pct(res["P_lb_gt_0"]),
                                    pct(res["P_seed_dir_2of3"]), pct(res["P_eval_part"])))
                    h2_json.append({"rho": rho, "f_arm": fa, "effect": eff, **res})
        results["h2_eval"] = h2_json
        md.append(f"\n### T3. H2 T_eval component: P[point(C-B) >= +5pp AND LB>0 AND direction in >=2/3 seeds]; "
                  f"n=134 paired tasks x 3 seeds, cluster = task (seed-averaged), p0={p0_h} (UNVERIFIED), "
                  f"delta_B=0, f_B=f_C=f_arm; rho = share of tasks whose flips are identical across seeds "
                  f"(1 = seeds add no information); {R} reps x {nb} resamples\n")
        md.append(md_table(["rho", "f_arm", "true C-B", "E[obs C-B]", "sd(obs)", "P(point>=5pp)", "P(LB>0)",
                            "P(dir 2/3)", "P(T_eval part)"], h2_rows))
        # MDE at 80% for the T_eval part
        mde_rows = []
        for rho in rho_grid:
            for fa in farm_grid:
                pts = [(j["effect"], j["P_eval_part"]) for j in h2_json if j["rho"] == rho and j["f_arm"] == fa]
                mde = None
                for (e1, pw1), (e2, pw2) in zip(pts, pts[1:]):
                    if pw1 < 0.8 <= pw2:
                        mde = e1 + (0.8 - pw1) * (e2 - e1) / (pw2 - pw1)
                        break
                if mde is None and pts[-1][1] >= 0.8:
                    mde = pts[0][0]
                pts_lb = [(j["effect"], j["P_lb_gt_0_and_seed"]) for j in h2_json if j["rho"] == rho and j["f_arm"] == fa]
                mde_lb = None
                for (e1, pw1), (e2, pw2) in zip(pts_lb, pts_lb[1:]):
                    if pw1 < 0.8 <= pw2:
                        mde_lb = e1 + (0.8 - pw1) * (e2 - e1) / (pw2 - pw1)
                        break
                # realized-scale MDE: interpolate on E[observed C-B] (= nominal*(1-2f)
                # under symmetric noise with f_B=f_C), which is the effect actually
                # present on T_eval
                pts_r = [(j["realized_effect_mean"], j["P_eval_part"]) for j in h2_json
                         if j["rho"] == rho and j["f_arm"] == fa]
                mde_r = None
                for (e1, pw1), (e2, pw2) in zip(pts_r, pts_r[1:]):
                    if pw1 < 0.8 <= pw2:
                        mde_r = e1 + (0.8 - pw1) * (e2 - e1) / (pw2 - pw1)
                        break
                pts_rlb = [(j["realized_effect_mean"], j["P_lb_gt_0_and_seed"]) for j in h2_json
                           if j["rho"] == rho and j["f_arm"] == fa]
                mde_rlb = None
                for (e1, pw1), (e2, pw2) in zip(pts_rlb, pts_rlb[1:]):
                    if pw1 < 0.8 <= pw2:
                        mde_rlb = e1 + (0.8 - pw1) * (e2 - e1) / (pw2 - pw1)
                        break
                mde_rows.append((rho, fa, f"{mde:.3f}" if mde is not None else ">0.20",
                                 f"{mde_lb:.3f}" if mde_lb is not None else ">0.20",
                                 f"{mde_r:.3f}" if mde_r is not None else f">{pts_r[-1][0]:.3f}",
                                 f"{mde_rlb:.3f}" if mde_rlb is not None else f">{pts_rlb[-1][0]:.3f}"))
        md.append("\n### T3b. Minimum detectable C-B effect at 80% power (linear interpolation on the grid). "
                  "'nominal' = model input delta_C; 'realized' = E[observed C-B] on T_eval "
                  "(= nominal x (1-2 f_arm) under symmetric noise), i.e. the effect that is actually present\n")
        md.append(md_table(["rho", "f_arm", "MDE nominal: full criterion", "MDE nominal: LB>0 & seed only",
                            "MDE realized: full criterion", "MDE realized: LB>0 & seed only"], mde_rows))
        results["h2_mde"] = mde_rows
        print(f"[h2 eval] done {time.time()-t0:.1f}s", file=sys.stderr)

        reg_rows, reg_json = [], []
        rB_grid = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40]
        rC_grid = [0.0, 0.025, 0.05]
        for rho in rho_grid:
            for rB in rB_grid:
                for rC in rC_grid:
                    if rC >= rB:
                        continue
                    res = h2_reg_part(rng, R=R, r_B=rB, r_C=rC, rho=rho, n_boot=nb)
                    reg_rows.append((rho, rB, rC, f"{rB-rC:+.3f}", pct(res["P_reg_part"])))
                    reg_json.append({"rho": rho, "r_B": rB, "r_C": rC, **res})
        results["h2_reg"] = reg_json
        md.append("\n### T4. H2 regression component: P[reg(B)-reg(C) > 0 with cluster-bootstrap LB > 0] over the 40 T_reg tasks x 3 seeds "
                  "(per-task per-seed failure probabilities r_B, r_C, independent arms)\n")
        md.append(md_table(["rho", "r_B", "r_C", "true diff", "P(reg part)"], reg_rows))
        # positive-dependence sensitivity
        dep_rows = []
        for rB, rC in ((0.10, 0.025), (0.15, 0.05), (0.20, 0.05)):
            for corr in (0.0, 0.5, 1.0):
                res = h2_reg_part(rng, R=R, r_B=rB, r_C=rC, rho=1.0, n_boot=nb, corr=corr)
                dep_rows.append((rB, rC, corr, pct(res["P_reg_part"])))
        md.append("\n### T4b. Regression component with positive B/C failure dependence (C's skills are a subset of B's), rho=1\n")
        md.append(md_table(["r_B", "r_C", "P(B fails | C fails)", "P(reg part)"], dep_rows))
        results["h2_reg_dependence"] = dep_rows
        print(f"[h2 reg] done {time.time()-t0:.1f}s", file=sys.stderr)

        # joint H2 illustrative cells (independent components)
        joint_rows = []
        for rho in rho_grid:
            for fa in (0.10, 0.20):
                for eff in (0.05, 0.10, 0.15):
                    pe = next(j["P_eval_part"] for j in h2_json if j["rho"] == rho and j["f_arm"] == fa and j["effect"] == eff)
                    for rB, rC in ((0.10, 0.025), (0.20, 0.05), (0.30, 0.05)):
                        pr = next(j["P_reg_part"] for j in reg_json if j["rho"] == rho and j["r_B"] == rB and j["r_C"] == rC)
                        joint_rows.append((rho, fa, f"{eff:+.2f}", rB, rC, pct(pe), pct(pr), pct(pe * pr)))
        md.append("\n### T5. Joint H2 support probability = P(T_eval part) x P(regression part) (disjoint task sets => independent given parameters)\n")
        md.append(md_table(["rho", "f_arm", "true C-B", "r_B", "r_C", "P(eval)", "P(reg)", "P(H2 supported)"], joint_rows))
        results["h2_joint"] = joint_rows

        # H1
        h1_rows, h1_json = [], []
        for rho in rho_grid:
            for fa in farm_grid:
                for eff in (0.0, 0.025, 0.05, 0.075, 0.10, 0.15):
                    res = h1_part(rng, R=R, p0=p0_h, delta_C=eff, f_C=fa, rho=rho, model=args.model, n_boot=nb)
                    h1_rows.append((rho, fa, f"{eff:+.3f}", f"{res['realized_effect_mean']:+.3f}", pct(res["P_h1"])))
                    h1_json.append({"rho": rho, "f_arm": fa, "effect": eff, **res})
        results["h1"] = h1_json
        md.append(f"\n### T6. H1 power: P[(C-A) cluster-bootstrap LB > 0]; arm A identical across seeds (G0 at temp 0), p0={p0_h}\n")
        md.append(md_table(["rho", "f_C", "true C-A", "E[obs C-A]", "P(H1 supported)"], h1_rows))
        print(f"[h1] done {time.time()-t0:.1f}s", file=sys.stderr)

        # bootstrap resolution check: 1k vs 10k on two cells
        chk = []
        for eff in (0.08, 0.12):
            r1 = h2_eval_part(rng, R=300, p0=p0_h, delta_B=0.0, delta_C=eff, f_B=0.2, f_C=0.2, rho=1.0, model=args.model, n_boot=1000)
            r2 = h2_eval_part(rng, R=300, p0=p0_h, delta_B=0.0, delta_C=eff, f_B=0.2, f_C=0.2, rho=1.0, model=args.model, n_boot=10000)
            chk.append((f"{eff:+.2f}", pct(r1["P_lb_gt_0"]), pct(r2["P_lb_gt_0"])))
        md.append("\n### T6b. Bootstrap-resolution check (rho=1, f_arm=0.2, 300 reps): P(LB>0) with 1k vs 10k resamples\n")
        md.append(md_table(["true C-B", "1k resamples", "10k resamples"], chk))
        results["boot_check"] = chk

    results["runtime_s"] = time.time() - t0
    with open(os.path.join(args.out, "results.json"), "w") as fh:
        json.dump(results, fh, indent=1, default=float)
    with open(os.path.join(args.out, "tables.md"), "w") as fh:
        fh.write("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\n[done] {results['runtime_s']:.1f}s -> {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
