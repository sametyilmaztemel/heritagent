"""Adversarial C1 check: compare the locked percentile-bootstrap LB>0 rule at n=20 with an
exact one-sided conditional sign (McNemar) test on discordant pairs."""
import json
from scipy.stats import binom
HERE = "<scratch>/out/repro/verify-power"
rows = {(r["a"], r["b"]): r["emp_pass"] for r in json.load(open(f"{HERE}/c1_stage2_out.json"))["rows"]}
def sign_p(a, b):  # one-sided P(X >= a | X ~ Bin(a+b, 1/2))
    return binom.sf(a - 1, a + b, 0.5)
print(" b | min a: bootstrap(always) | exact sign one-sided a=0.025 | a=0.05")
for b in range(0, 6):
    boot = min([a for (a, bb), p in rows.items() if bb == b and p == 1.0], default=None)
    s025 = min([a for a in range(1, 21 - b) if sign_p(a, b) <= 0.025], default=None)
    s05 = min([a for a in range(1, 21 - b) if sign_p(a, b) <= 0.05], default=None)
    print(f" {b} | {boot} | {s025} | {s05}")
print("(4,0): sign p =", sign_p(4, 0), "; (5,0):", sign_p(5, 0), "; (7,1):", sign_p(7, 1), "; (9,2):", sign_p(9, 2))
# stage-2 pass prob at f=0 under each rule: P(Bin(20,delta) >= k)
for d in (0.1, 0.2, 0.3):
    print(f"delta={d}: bootstrap(>=4) {binom.sf(3,20,d):.3f} | sign a=.05 (>=5) {binom.sf(4,20,d):.3f} | sign a=.025 (>=6) {binom.sf(5,20,d):.3f}")
