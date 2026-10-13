"""C2/C3/C4 independent exact computations (no code from exp0001_power_sim.py).

Per-instance model (the study's stated A2, re-implemented from its text):
  Y0 ~ Bern(p0)  (without trait, G0 at temp 0)
  effect: delta>0 -> failed instance rescued w.p. r = delta/(1-p0);
          delta<0 -> solved instance broken w.p. -delta/p0
  noise (symmetric): with-trait outcome inverted w.p. f.
Cells per instance: (Y1, Y0) in {11, 10, 01, 00}.
Stage-2 pass table: from c1_stage2_out.json (seed-0 empirical, true repo RNG
stream) -- also re-run with a deterministic threshold p_tail < 0.025.
"""
import json, math, itertools, sys
from functools import lru_cache
import numpy as np
from scipy.stats import binom

HERE = "<scratch>/out/repro/verify-power"
tab = json.load(open(f"{HERE}/c1_stage2_out.json"))["rows"]
PASS_EMP = {(r["a"], r["b"]): r["emp_pass"] for r in tab}


def cells(p0, delta, f):
    if delta >= 0:
        r = delta / (1 - p0)
        y1pre_given_y0_1 = 1.0
        y1pre_given_y0_0 = r
    else:
        y1pre_given_y0_1 = 1 + delta / p0
        y1pre_given_y0_0 = 0.0
    def flip(p):  # P(Y1=1) after symmetric flip
        return p * (1 - f) + (1 - p) * f
    p11 = p0 * flip(y1pre_given_y0_1)
    p01 = p0 * (1 - flip(y1pre_given_y0_1))
    p10 = (1 - p0) * flip(y1pre_given_y0_0)
    p00 = (1 - p0) * (1 - flip(y1pre_given_y0_0))
    return p11, p10, p01, p00


def multinom_pmf(counts, probs):
    n = sum(counts)
    lg = math.lgamma(n + 1) - sum(math.lgamma(c + 1) for c in counts)
    s = 0.0
    for c, p in zip(counts, probs):
        if c > 0:
            if p <= 0:
                return 0.0
            s += c * math.log(p)
    return math.exp(lg + s)


def replay_pass(p0, delta, f, e=1.0, source_rescued=False):
    c = cells(p0, delta, f)
    n_var = 4 if source_rescued else 5
    tot = 0.0
    for n11 in range(n_var + 1):
        for n10 in range(n_var + 1 - n11):
            for n01 in range(n_var + 1 - n11 - n10):
                n00 = n_var - n11 - n10 - n01
                pr = multinom_pmf((n11, n10, n01, n00), c)
                a11, a10, a01 = n11, n10, n01
                if source_rescued:
                    a10 += 1
                s_with = a11 + a10
                net = a10 - a01
                if s_with >= 3 and net >= 2:
                    tot += pr * e ** s_with
    return tot


def ablation_pass(p0, delta, f, table=PASS_EMP, n=20):
    _, p10, p01, _ = cells(p0, delta, f)
    tot = 0.0
    for a in range(n + 1):
        for b in range(n + 1 - a):
            ps = table.get((a, b), 0.0)
            if ps == 0:
                continue
            tot += multinom_pmf((a, b, n - a - b), (p10, p01, 1 - p10 - p01)) * ps
    return tot


def gen_pass(p0, delta, f, n=10):
    _, p10, p01, _ = cells(p0, delta, f)
    tot = 0.0
    for a in range(n + 1):
        for b in range(n + 1 - a):
            if a - b >= 0:
                tot += multinom_pmf((a, b, n - a - b), (p10, p01, 1 - p10 - p01))
    return tot


def reg_pass(h, n=40, max_fail=1):
    return float(binom.cdf(max_fail, n, h))


def h_reg_sym(p0, delta, f):
    h = (-delta / p0) if delta < 0 else 0.0
    return h * (1 - f) + (1 - h) * f


out = {}
print("=== C2: all-five-stage pass of a helpful GENERALIST trait (symmetric model, e=1) ===")
print(f"{'p0':>4}{'delta':>7}{'f':>6}{'replay':>9}{'abl':>8}{'gen_in':>8}{'gen_out':>8}{'reg':>8}{'ALL':>8}{'falseRej':>9}")
for p0 in (0.3, 0.5):
    for delta in (0.05, 0.10, 0.20, 0.30, 0.40):
        for f in (0.0, 0.05, 0.10):
            rp = replay_pass(p0, delta, f)
            ab = ablation_pass(p0, delta, f)
            gi = gen_pass(p0, delta, f)
            go = gen_pass(p0, delta, f)
            rg = reg_pass(h_reg_sym(p0, delta, f))
            allp = rp * ab * gi * go * rg
            out[f"{p0},{delta},{f}"] = dict(replay=rp, abl=ab, gen_in=gi, gen_out=go, reg=rg, all=allp)
            print(f"{p0:>4}{delta:>7.2f}{f:>6.2f}{rp:>9.4f}{ab:>8.4f}{gi:>8.4f}{go:>8.4f}{rg:>8.4f}{allp:>8.4f}{1-allp:>9.3f}")

print("\nClosed-form f=0 cross-check: ablation = P(Bin(20,delta)>=4) (no breaks at f=0)")
for d in (0.1, 0.2, 0.3, 0.4):
    print(f"  delta={d}: {1-binom.cdf(3,20,d):.4f}")

print("\n=== C2 adversarial sensitivities (f=0) ===")
print("(i) replay source instance is a G0 failure deterministically rescued by its own mined trait")
for p0 in (0.3, 0.5):
    for delta in (0.10, 0.20, 0.30, 0.40):
        rp = replay_pass(p0, delta, 0.0, source_rescued=True)
        ab = ablation_pass(p0, delta, 0.0)
        print(f"  p0={p0} delta={delta}: replay {rp:.4f} (vs {replay_pass(p0, delta, 0.0):.4f}), ALL {rp*ab:.4f}, falseRej {1-rp*ab:.3f}")

# per-stage decomposition of the false reject at f=0
print("\n(ii) stage-2 alone vs replay alone at f=0, p0=0.5")
for delta in (0.1, 0.2):
    print(f"  delta={delta}: replay={replay_pass(0.5, delta, 0):.4f} ablation={ablation_pass(0.5, delta, 0):.4f}")

print("\n=== C3: regression stage (40 T_reg, baseline=1 not rerun, pass iff <=1 failure) ===")
for h in (0.0, 0.01, 0.02, 0.025, 0.03, 0.05, 0.075, 0.10):
    print(f"  h={h:<6}: P(pass)={reg_pass(h):.4f}  [0 fails: {reg_pass(h,max_fail=0):.4f}; <=2 fails: {reg_pass(h,max_fail=2):.4f}]")
print("  symmetric-model neutral trait: h_reg = h(1-f)+(1-h)f with h=0 -> f; f=0.05 ->", round(reg_pass(h_reg_sym(0.5, 0.0, 0.05)), 4))
print("  redraw-model neutral trait (study's alt): h_reg = f(1-p0); f=0.05, p0=0.3/0.5 ->",
      round(reg_pass(0.05 * 0.7), 4), round(reg_pass(0.05 * 0.5), 4))
# what f would make a neutral trait pass 80% / 90% / 95%?
from scipy.optimize import brentq
for target in (0.5, 0.8, 0.9, 0.95):
    h = brentq(lambda x: reg_pass(x) - target, 1e-6, 0.5)
    print(f"  h_reg giving P(pass)={target}: {h:.4f}")
# the 'true regression equals threshold' reading: per-task failure prob 0.03
print("  P(pass | h=0.03) =", round(reg_pass(0.03), 4))
# alternative reading: trait breaks EXACTLY 3% of 40 = 1.2 tasks deterministically -> fixed set
print("  deterministic reading: a trait that breaks exactly k fixed T_reg tasks passes iff k<=1 (prob 1 or 0)")

print("\n=== C4: float/integer boundaries (exact code expressions) ===")
for s_with in range(0, 11):
    for s_without in range(0, 11):
        pass
deltas_out = sorted({(sw - swo) / 10 for sw in range(11) for swo in range(11)})
passing = [d for d in deltas_out if d >= -0.05]
print("  out-of-scope: smallest passing delta_out =", min(passing), "; largest failing =", max(d for d in deltas_out if d < -0.05))
print("  => delta_out >= -0.05  <=>  net >= 0 at n=10:", all(((d >= -0.05) == (round(d * 10) >= 0)) for d in deltas_out))
print("  in-scope rule delta_in >= 0.0 <=> net >= 0:", all(((d >= 0.0) == (round(d * 10) >= 0)) for d in deltas_out))
for s in (40, 39, 38, 37):
    drop = 1.0 - s / 40
    print(f"  s_with={s}: drop={drop!r}  <=0.02:{drop <= 0.02} <=0.03:{drop <= 0.03} <=0.05:{drop <= 0.05}")
print("  1.0-38/40 =", repr(1.0 - 38 / 40), "; == 0.050000000000000044:", (1.0 - 38 / 40) == 0.050000000000000044)
tol = {tau: max(40 - s for s in range(41) if 1.0 - s / 40 <= tau) for tau in (0.02, 0.03, 0.05)}
print("  max failures tolerated per tau_r cell (as coded):", tol)
# stage-2 point rule float safety
for net in (0, 1, 2):
    pt = sum([1.0] * net + [0.0] * (20 - net)) / 20
    print(f"  stage-2 point net={net}: {pt!r} >=0.03:{pt >= 0.03} >=0.05:{pt >= 0.05} >=0.08:{pt >= 0.08}")
# interaction harmful threshold at n=10
print("  interaction harmful iff delta_pair < -0.05: net=-1 ->", (-1 / 10) < -0.05, "; net=0 ->", 0.0 < -0.05)

json.dump(out, open(f"{HERE}/c2_c4_out.json", "w"), indent=1)
