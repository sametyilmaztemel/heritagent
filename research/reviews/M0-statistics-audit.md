# M0 statistics and gate-code audit (CIG stages 1–2)

Independent reviewer (Claude subagent). Read-only audit; nothing was posted to GitHub and the checkout was not modified (`git status` clean after every run).

## 1. Scope

- Code: `main` @ `0ce0a36` ("Issue #11: replay and ablation evaluators"). Files: `inheritance/bootstrap.py`, `ablation.py`, `replay.py`, `overlay.py`, `runner.py`, plus `runtime/loop.py` and `runtime/skills.py` for expression semantics.
- Tests: `tests/inheritance/*`. Run on venv314 they give **56 passed**.
- Reference text: `experiments/manifests/EXP-0001-design.md` §4–5 (lines 47–48, 61–72), `EXP-0001-minimal-cig.yaml` (lines 78–82, 97–106), ADR-0002, issue #11 body, issue #12 body, and SPEC §10 (`SPEC.md:392`).
- Out of scope: stages 3–4, the aggregate verdict, and PR #28. One read-only grep of the PR #28 worktree at `4ca5517` is reported in §5 only.

## 2. Method

- Line-by-line comparison of the code with the locked prereg text.
- 13 repro scripts in `research/reviews/repro/m0-stats/`, each with a saved `.out` file. Run command from `/home/user/heritagent`: `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. <scratchpad>/venv314/bin/python <script>`. R10 was also run on venv311, venv312 and venv (3.13).
- Exact, non-Monte-Carlo checks. For n=20 binary outcomes each paired delta is −1, 0 or +1, which gives 231 possible (pos, neg) compositions. For each one, P(resample mean ≤ 0) was computed exactly from the multinomial and compared with the merged code's verdict. R06 compares the replay rule with an independent reference reading of the prereg over all 1024 outcome patterns × 2 expression regimes.

## 3. Findings

Ordered by severity. The "Repro" line names the script.

### F1 (major, prereg/statistics): τ_c = 0.05 never affects the stage-2 verdict at n = 20
- The code matches the text: `bootstrap.py:66-70` and `ablation.py:268-271` implement "point ≥ τ_c AND LB > 0" (design.md:48).
- With 20 binary paired outcomes, no composition with point < 0.20 has LB > 0. The smallest exact P(mean ≤ 0) among compositions with point in (0, 0.20) is 0.0388 (pos=3, neg=0); LB > 0 needs roughly < 0.025.
  - Every composition with point exactly 0.05 has P(mean ≤ 0) ≥ 0.358.
  - The smallest passing composition is +4/20 with zero harms (point 0.20, LB 0.05).
- The verdict count is identical (54 of 231) for every τ_c from 0.00 to 0.20. The pre-registered secondary grid τ_c ∈ {0.03, 0.05, 0.08} (design.md:71; yaml:105) changes **0/231** decisions, so that sensitivity analysis is vacuous for stage 2.
- Operating characteristic (R12: seed 0, iid simplification):

  | True per-task effect | P(pass stage 2) |
  |---|---|
  | +0.05 (= τ_c) | 0.016–0.056 |
  | +0.10 | 0.11–0.15 |
  | +0.20 | 0.27–0.59 |
  | null (a = b) | 0.021–0.022 |

- The tests do not catch this:
  - `test_bootstrap.py:54-60` asserts `stage_passes(0.05, 0.01)`, a combination that cannot occur at n=20.
  - The "real tau_c boundary at evaluator level" case (`test_ablation.py:140-155`) actually fails on LB, not on τ_c.
- Repro: `r01_tau_c_nonbinding.py`, `r12_stage2_operating_characteristic.py`.

### F2 (major, integrity): the bootstrap seed and task order decide the primary verdict, and neither is locked or pre-registered
- τ_c and n_resamples are locked (`ablation.py:199-205`), but `bootstrap_seed` is a free constructor argument (`ablation.py:188, 204`). It appears nowhere in the yaml, the design or ADR-0002.
- Example: composition pos=10, neg=3 (point 0.35, exact P(mean ≤ 0) = 0.0258) passes for **61/200** seeds.
  - Run end to end on identical scripted evidence, `bootstrap_seed=1` gives LB 0.05 and PASS; `bootstrap_seed=0` (the default) gives LB 0.00 and FAIL.
  - Both child records are schema-valid and share the id `CIG-0007/S11-ablation`.
- The RNG draws positions in the list, so reordering the same 20 deltas also flips the verdict at seed 0: 4 of 8 orderings pass (R03b).
- The Research/Critic agent accepted a configurable seed because it is "a reproducibility seed, not a threshold" (issue #11 comment, 2026-10-08T23:43Z). This evidence shows that near the boundary it works as a decision parameter.
- Mitigations already present: the child record stores the seed and the task order. PR #28 head `4ca5517` does not pass a seed (read-only grep), so it uses 0.
- Suggested fix: pin the seed (and a canonical task order) in the manifest. Better, since deltas only take the values −1, 0, +1 at n=20, compute the bootstrap distribution exactly (R01 shows this is cheap), which removes Monte Carlo noise entirely.
- Repro: `r02_bootstrap_seed_flips_primary_verdict.py`, `r03_percentile_index_and_order.py` (b).

### F3 (major, interpretation-dependent): the replay expression check only tests presence for constitutive traits, which is the M0 default
- `overlay.py:78-80` gives current candidates no regulation, so they are constitutive. `runtime/skills.py:15-19` makes constitutive skills always active, and `runtime/loop.py:165-166` emits them in `skills_expressed` at every step. The contracted `expressed_gene_ids` is built from those events (`runner.py:149-158`).
- So for every M0 candidate, `replay.py:227-233` cannot fail.
- Repro over the real `run_agent` path: the model gives byte-identical replies in both arms and never acts on the skill, yet every with-trait episode is marked as expressing the gene and `expression_ok=True, passed=True`. A regulated target (`consecutive_failures >= 2`) correctly fails the same check.
- Prereg (design.md:47): "trait must fire … (expression check, not just presence)". Issue #11: "trait must actually fire".
  - If "presence" means presence in the genome, the code is consistent.
  - If "fire" means the behaviour used the skill, this is unimplemented.
- Either way, `expression_check.passed=True` in M0 replay records carries no information.
- Repro: `r07_expression_check_is_presence.py`.

### F4 (minor): bootstrap inputs are not validated
- `bootstrap_seed=None` is accepted (`ablation.py:188`; `bootstrap.py:48` calls `random.Random(None)`, which seeds from OS entropy).
  - 12 primary runs on identical evidence: 7 PASS, 5 FAIL.
  - All 12 child records are schema-valid and record `"seed": null`. The cig schema's `measurements` is a free object.
- Deltas with NaN are accepted: point = nan with "CI" [1.0, 1.0]. Out-of-range deltas (5.0) and booleans are also accepted.
- This matters most for the public `sensitivity_analysis` API, which takes arbitrary deltas.
- Repro: `r05_bootstrap_seed_none_nondeterministic.py`, `r04_degenerate_cases.py`.

### F5 (minor): infrastructure-error statuses count as real outcomes
- `validate_outcome` (`runner.py:184-223`) never checks `status`, which is a free string (`runner.py:164`).
- With a trait that has no effect (both arms succeed everywhere) and 6 control-arm episodes returned as `status="adapter_error", success=False`, the result is deltas 6×(+1), point 0.30, LB 0.10, **PASS**.
- Lower risk with the real `run_agent`, whose RunResult status is only `finished` or `budget_exhausted` (`loop.py:104`), but the runner contract allows it.
- Repro: `r08_ablation_provenance_and_status.py` (b).

### F6 (minor): stage-2 allocation does not check or record task provenance
- `AblationAllocation.__post_init__` (`ablation.py:57-87`) accepts tasks with `bank_id="T_eval"`, `None`, or a replay-bank id.
- Its metadata (`ablation.py:89-98`) records no bank ids. By contrast `ReplayAllocation` enforces provenance (`replay.py:85-89`) and records it (`replay.py:99-107`).
- The yaml:67 contamination rule is therefore not enforced or auditable at stage 2.
- Repro: `r08_ablation_provenance_and_status.py` (a).

### F7 (minor): "one execution per (task, condition)" holds only within one `evaluate()` call
- `seen_pairs` is local (`replay.py:201`, `ablation.py:218`), and the ids are already validated distinct (`replay.py:77-79`, `ablation.py:70-71`). So the in-loop duplicate guard (`replay.py:207-210`, `ablation.py:231-234`) can never trigger.
- Calling `evaluate()` twice re-executes everything: 20 runner calls for replay and 80 for ablation. It emits two schema-valid children with the same `cig_id` and, under simulated backend drift, opposite `stage_pass` (replay False→True; ablation LB 0.00→0.05, FAIL→PASS). That is a best-of-k path.
- This may belong to #12 / PR #28 deduplication.
- Repro: `r11_repeat_evaluate_pseudoreplication.py`.

### F8 (minor): the percentile quantile definition is ad hoc and not pre-registered
- For B=10,000 at 95%, `bootstrap.py:57-59` picks `means[250]` (251st smallest) and `means[9750]` (250th largest). The tails are cut asymmetrically.
- A type-1 (inverted-CDF) 2.5th percentile would be `means[249]`. With seed 64 and pos10/neg3, exactly 250 resample means are ≤ 0: the code gives LB 0.05 (PASS), type 1 gives 0.0 (FAIL), and numpy's default type 7 gives 0.04875 (PASS).
- The word "percentile" appears nowhere in the prereg, ADR-0002 or issue #11, although the Research/Critic agent described "95% percentile paired bootstrap" as locked there.
- Repro: `r03_percentile_index_and_order.py` (a, b0).

### F9 (minor): `sensitivity_analysis` does not compute what §5 pre-registers
- `meets_tau_c` (`ablation.py:175`) checks the point estimate only. There is no recomputed gate decision and no point estimate in the output.
  - Example: deltas 1×(+1), 19×0 at τ_c=0.03 give `meets_tau_c=True`, but the actual gate verdict is FAIL.
- It also re-draws the bootstrap with a caller-chosen seed and n, which are not in the pre-registered grid. At τ_c=0.05 with seed 1 it reports LB 0.05 where the primary record says 0.0.
- Repro: `r13_sensitivity_api.py`.

### F10 (minor): replay mixes a non-episode entry into `episodes`
- `replay.py:251-252` appends `{"expression_violations": [...]}` to the episodes list and to `measurements.episodes`.
- 992 of 2048 evaluations had 11 entries instead of 10. Consumers that count episodes or read `e["task_id"]` will miscount or break.
- Repro: `r06_replay_rule_exhaustive.py`.

### F11 (minor, latent): the bootstrap API cannot express clusters with more than one observation
- `paired_cluster_bootstrap` treats every list element as a cluster. Design §5 (design.md:63) requires a cluster bootstrap over tasks for the T_eval contrasts, which pool 3 mining seeds.
- Passing 134 tasks × 3 seeds = 402 flat deltas gives CI [+0.012, +0.142], "excludes 0". Averaging per task first gives CI [−0.030, +0.189] on the same data. The width ratio is 1.69 (√3 ≈ 1.73).
- This is not a defect on main yet, because the T_eval contrast code does not exist.
- Repro: `r09_cluster_api_flattening_risk.py`.

### F12 (minor, docs/provenance)
- `bootstrap.py:21` labels 10k "ADR-0002 locked", but ADR-0002 does not mention 10k. The prereg states 10k only for the T_eval arm contrasts (design.md:63, yaml:99).
- `bootstrap.py:9-10` says "no hidden version-dependent defaults", but `randrange` is documented as not guaranteed stable across Python versions.
  - Empirically the output is identical on 3.11.17, 3.12.3, 3.13.16 and 3.14.6 (digest `084d1d4a60dd4b0f`, R10).
- `replay.py` and `ablation.py` use the name `EvaluationTask` in annotations without importing it. This is harmless under `from __future__ import annotations`.

### F13 (minor, test coverage)
- No evaluator-level test asserts that all-zero evidence FAILS. `test_ablation.py:82-98` and `test_ablation_frozen_contract.py:408-421` only assert Δ=0. The behaviour itself is correct (R04).
- Nothing tests verdict stability across seeds or orderings, and `test_bootstrap.py:28-38` actually asserts that different seeds give different CIs.
- Nothing compares the result with an independent quantile implementation.
- The τ_c tests use combinations that cannot occur (see F1).

## 4. Confirmed correct

- **Resampling unit:** the task instance, one paired d_i per task. Pairing is preserved, since d_i is resampled as a unit (`ablation.py:265-266`, `bootstrap.py:50-54`). Draws are with replacement, n per resample, statistic = mean.
- **Resample count:** 10,000, not caller-tunable (TypeError; `test_ablation.py:235-266`).
- **CI:** two-sided 95% percentile; LB ≈ 2.5th percentile.
  - The null false-pass rate is 0.021–0.022 (R12), consistent with a one-sided α ≈ 0.025.
  - LB > 0 is strict (`bootstrap.py:70`), and point ≥ τ_c is non-strict as the prereg says (design.md:48).
- **τ_c float boundary:** exact for binary deltas; for every composition, sum/20 equals k/20 exactly, and 1/20 == 0.05 (R04).
- **Degenerate cases:**
  - All-zero gives CI [0, 0] and FAIL, at both the bootstrap and evaluator levels.
  - All +1 gives [1, 1] and PASS; all −1 FAILS; empty input raises; confidence outside (0, 1) raises.
- **Determinism:** the same seed gives an identical result, also across Python 3.11–3.14.
- **Replay rule:** exact. 2048/2048 evaluations agree with an independent reading of design.md:47 (≥ 3/5 successes with the trait, net ≥ +2/5, expression in every successful with-trait episode; failed episodes exempt). Integer comparisons; `delta_success = Δ/5`.
- **Contamination:** target expression in the without-trait arm aborts with an integrity error and no child record. This is stricter than the prereg and reasonable.
- **No post-filtering in ablation:** tasks where the trait did not fire contribute Δ = 0, matching issue #11 and design §6.3.

## 5. Open questions

1. F1: before the run (`expected_results_filled: false`), should τ_c, n, or the rule itself be amended via an ADR? For example, "LB ≥ τ_c", or a larger ablation pool. Otherwise the "τ_c = 5pp" threshold and its sensitivity grid are inoperative.
2. F2/F8: pin `bootstrap_seed` and a canonical task order in the manifest, or switch to the exact bootstrap distribution? Also pre-register the quantile definition.
3. F3: what does "fire" mean for a constitutive trait? If it means behavioural use, which observable defines it (for example, a procedure-step reference in the trajectory)?
4. F5: what is the policy for infrastructure-error episodes: exclude and re-run, or abort the stage? Is it symmetric across arms?
5. The ablation pool is stratified ({10, 6, 4} in tests), but the bootstrap ignores strata. Is a stratified bootstrap intended? The prereg does not say.
6. SPEC §10 (`SPEC.md:392`) uses a strict "Contribution > τ_c" while the prereg uses ≥. Given F1 this does not change any verdict now, but the documents should agree.
7. F7/F11: does #12 / PR #28 own cross-call deduplication of children with the same id, and task-level clustering for the T_eval analysis across 3 mining seeds? A read-only grep shows PR #28 head `4ca5517` keeps the same `bootstrap.py` and `ablation.py` seed surface.
