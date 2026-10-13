"""Re-derive V2 gate pass probabilities with an internally consistent noise model.
noise loss rate l applies to every stage; in-scope/ablation/replay wins = l + e; out-of-scope wins = losses = l;
T_reg flip f either = l/p (review's assumption) or independent. Replay effect e_r may differ (family concentration).
"""
import math, sys
sys.dont_write_bytecode = True
sys.path.insert(0, "/home/user/heritagent")
from inheritance.bootstrap import paired_cluster_bootstrap, stage_passes
N2 = 20
region = {}
for b in range(N2 + 1):
    for c in range(N2 + 1 - b):
        if b - c <= 0:
            region[(b, c)] = False; continue
        r = paired_cluster_bootstrap([1.0]*b + [-1.0]*c + [0.0]*(N2-b-c), seed=0)
        region[(b, c)] = stage_passes(r.point_estimate, r.lower_bound, tau_c=0.05)

def mn(counts, probs):
    n = sum(counts); coef = math.factorial(n)
    for k in counts: coef //= math.factorial(k)
    pr = 1.0
    for k, q in zip(counts, probs):
        if k: pr *= q**k
    return coef*pr

def p_replay(p, win, loss):
    b1, b0 = p-loss, 1-p-win
    t = 0.0
    for w in range(6):
        for l in range(6-w):
            for s in range(6-w-l):
                z = 5-w-l-s
                if s+w >= 3 and w-l >= 2:
                    t += mn((w, l, s, z), (win, loss, b1, b0))
    return t

def p_s2(win, loss):
    return sum(mn((b, c, N2-b-c), (win, loss, 1-win-loss)) for (b, c), ok in region.items() if ok)

def p_nn(n, win, loss):
    return sum(mn((b, c, n-b-c), (win, loss, 1-win-loss)) for b in range(n+1) for c in range(n+1-b) if b >= c)

def p_reg(f, n=40, mx=1):
    return sum(math.comb(n, k)*f**k*(1-f)**(n-k) for k in range(mx+1))

print("e   l     p     e_r  f      | replay stage2 in    out   T_reg | P(promote) E|C|/10")
for p in (0.4, 0.667):
    for e in (0.2, 0.3):
        for l in (0.0, 0.025, 0.05):
            for e_r in (e, min(0.5, 2*e)):
                for fmode in ("l/p", 0.02):
                    f = l/p if fmode == "l/p" else fmode
                    if l == 0 and fmode != "l/p": continue
                    wr, lr = l+e_r, l
                    if wr > 1-p: continue
                    pr = p_replay(p, wr, lr); s2 = p_s2(l+e, l); pin = p_nn(10, l+e, l)
                    pout = p_nn(10, l, l); s4 = p_reg(f)
                    tot = pr*s2*pin*pout*s4
                    print(f"{e:<3} {l:<5} {p:<5} {e_r:<4} {f:<6.3f} | {pr:.3f}  {s2:.3f}  {pin:.3f} {pout:.3f} {s4:.3f} | {tot:.4f}  {10*tot:.2f}")
# review's own out-of-scope assumption in the 'no-loss' row
print("\nreview row e=0.2,d=0.2,p=0.4 uses out-of-scope win=loss=0.1 while in-scope loss=0:",
      round(p_nn(10, 0.1, 0.1), 3), "vs consistent (l=0):", round(p_nn(10, 0, 0), 3))
