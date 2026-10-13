"""R13 - sensitivity_analysis (inheritance/ablation.py:152-178) vs the
pre-registered secondary analysis (design.md:71: "recompute gate decisions
on T_gate stage data only, varying tau_c in {0.03, 0.05, 0.08}").

(a) The returned 'meets_tau_c' is point-estimate-only; there is no field for
    the recomputed gate decision (point >= tau_c AND LB > 0). A consumer that
    reads meets_tau_c as "would pass under tau_c" gets the wrong answer.
(b) The analysis re-draws the bootstrap with a caller-chosen seed and
    n_resamples (neither is part of the pre-registered sensitivity grid), so
    at the primary tau_c=0.05 it can report an LB with the opposite sign
    from the primary child record on identical deltas.

Run from /home/user/heritagent:
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. <venv314>/bin/python <this file>
"""
from inheritance.ablation import sensitivity_analysis
from inheritance.bootstrap import paired_cluster_bootstrap, stage_passes

d = [1.0] + [0.0] * 19
for tau in (0.03, 0.05, 0.08):
    a = sensitivity_analysis(d, tau_c=tau, n_resamples=10_000, bootstrap_seed=0)
    gate = stage_passes(0.05, a["bootstrap"]["lower_bound"], tau_c=tau)
    print(f"(a) deltas=1x(+1),19x0 tau_c={tau}: meets_tau_c={a['meets_tau_c']} "
          f"LB={a['bootstrap']['lower_bound']} -> actual gate decision={gate}; "
          f"keys={sorted(a)}")

d2 = [1.0] * 10 + [-1.0] * 3 + [0.0] * 7
primary = paired_cluster_bootstrap(d2, seed=0)   # AblationEvaluator default seed
print(f"\n(b) primary (seed 0, 10k): LB={primary.lower_bound} "
      f"pass={stage_passes(primary.point_estimate, primary.lower_bound)}")
for seed, n in ((1, 10_000), (0, 500), (3, 2_000)):
    a = sensitivity_analysis(d2, tau_c=0.05, n_resamples=n, bootstrap_seed=seed)
    print(f"    sensitivity tau_c=0.05 seed={seed} n={n}: LB={a['bootstrap']['lower_bound']} "
          f"meets_tau_c={a['meets_tau_c']} -> implied gate={stage_passes(0.35, a['bootstrap']['lower_bound'])}")
