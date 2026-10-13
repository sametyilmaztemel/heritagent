"""R10: exhaustive check over every attainable outcome of the REAL
evaluators (binary success, n=10 out-of-scope, n=20 ablation, n=40 T_reg),
using the exact expressions in generalization.py / bootstrap.py /
regression.py / gate.py sensitivity:
 - out-of-scope 'delta_out >= -0.05' is identical to 'delta_out >= 0';
 - sensitivity tau_c=0.03 is identical to tau_c=0.05 (k/20 grid);
 - sensitivity tau_r=0.05 is identical to tau_r=0.03 (float, see R1);
so no looser cell can ever enlarge the primary eligible set and the
not_identifiable branch is unreachable from real evaluator output.
Run from the PR checkout with PYTHONPATH=."""
from inheritance.generalization import OUT_SCOPE_MIN_DELTA, IN_SCOPE_MIN_DELTA
from inheritance.gate import SENSITIVITY_TAU_C_GRID, SENSITIVITY_TAU_R_GRID

out_rule = [(sw - swo) / 10 >= OUT_SCOPE_MIN_DELTA for sw in range(11) for swo in range(11)]
zero_rule = [(sw - swo) / 10 >= 0.0 for sw in range(11) for swo in range(11)]
print("out-of-scope rule == (delta_out >= 0) for all 121 outcomes:", out_rule == zero_rule)

def abl(k_pos, k_neg):
    deltas = [1.0] * k_pos + [-1.0] * k_neg + [0.0] * (20 - k_pos - k_neg)
    return sum(deltas) / 20
pts = {abl(p, n) for p in range(21) for n in range(21 - p)}
same_c = all((x >= 0.03) == (x >= 0.05) for x in pts)
print("tau_c=0.03 cell == tau_c=0.05 cell for all attainable point estimates:", same_c)
drops = [1.0 - s / 40 for s in range(41)]
same_r = all((d <= 0.05) == (d <= 0.03) for d in drops)
print("tau_r=0.05 cell == tau_r=0.03 cell for all 41 T_reg outcomes:", same_r)
print("grid:", SENSITIVITY_TAU_C_GRID, SENSITIVITY_TAU_R_GRID)
