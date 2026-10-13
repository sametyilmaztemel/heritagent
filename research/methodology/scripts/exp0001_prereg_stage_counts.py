"""Expected per-stage marginal pass counts per seed (K=10), from the power-analysis T1 table
(symmetric noise, p0=0.5, generalist traits; values copied from EXP-0001-power-analysis.md T1).
All stages run for all candidates (gate.py: no scientific short-circuit), so marginal counts are observable."""
T1 = {  # (delta, f): (replay, ablation, gen_in, out_gen, regression, all_gen)
 ("help+0.20", 0.00): (.252, .589, 1.0, 1.0, 1.0, .148),
 ("help+0.20", 0.05): (.243, .408, .964, .964, .399, .037),
 ("help+0.20", 0.10): (.234, .285, .924, .924, .080, .005),
 ("neutral",   0.00): (0.0, 0.0, 1.0, 1.0, 1.0, 0.0),
 ("neutral",   0.05): (.005, .001, .818, .818, .399, 0.0),
 ("neutral",   0.10): (.017, .007, .725, .725, .080, 0.0),
 ("harm-0.10", 0.00): (0.0, 0.0, .349, .349, .001, 0.0),
 ("harm-0.10", 0.05): (.003, 0.0, .386, .386, 0.0, 0.0),
 ("harm-0.10", 0.10): (.013, .002, .421, .421, 0.0, 0.0),
}
MIX = {"baseline 20/50/30": (2, 5, 3), "optimistic 50/30/20": (5, 3, 2), "pessimistic 10/60/30": (1, 6, 3)}
stages = ["replay", "ablation", "gen_in", "gen_out", "regression", "all5"]
print("| mixture | f | " + " | ".join(stages) + " |"); print("|---|---|" + "---|"*len(stages))
for mn, (h, n, x) in MIX.items():
    for f in (0.0, 0.05, 0.10):
        row = [h*a + n*b + x*c for a, b, c in zip(T1[("help+0.20", f)], T1[("neutral", f)], T1[("harm-0.10", f)])]
        print(f"| {mn} | {f} | " + " | ".join(f"{v:.2f}" for v in row) + " |")
for q in (0.245, 0.47, 0.739, 0.927, 0.98):
    print(f"P(empty per seed)={q}: P(empty in >=2/3 seeds | independent)={q**3+3*q*q*(1-q):.3f}; all 3={q**3:.3f}")
