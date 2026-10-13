# EXP-0001 — Operating characteristics and power analysis of the LOCKED design (simulation study)

**Status:** analysis-only research note by an independent third-agent researcher (Claude subagent), 2026-10-12. Nothing in this note changes any locked or pre-registered element of EXP-0001; every recommendation that would alter a locked rule is flagged `requires_adr = true` and is a *proposal for an ADR amendment*, not a decision (SPEC §33 pre-registration discipline). All numbers below are **operating characteristics under explicitly assumed effect models** (SPEC §34 wording); none is a prediction that "the experiment will fail", and none is empirical evidence about Qwen2.5-7B-Instruct or ALFWorld.

**Inputs (read-only):** `experiments/manifests/EXP-0001-design.md`, `EXP-0001-minimal-cig.yaml`, `research/decisions/ADR-0002-exp-0001-design.md` on `main` (commit `0ce0a36`) plus **Amendment A1** on PR #28 (`origin/issue-12-cig-gate`, head `7886852` at read time; the task brief named `4ca5517`, the branch has since moved), `research/hypotheses/hypotheses.md`, and the implementations `inheritance/bootstrap.py`, `replay.py`, `ablation.py` (main) and `generalization.py`, `regression.py`, `interaction.py`, `gate.py` (PR #28 branch).

**Reproduction:** `scripts/exp0001_power_sim.py` (this directory), fixed seed 20261012, numpy 2.5.4 / scipy 1.18.1, Python 3.14.6; full run ≈ 4–5 min on 4 cores; outputs in `exp0001_power_sim_results/{tables.md,results.json}`. The stage-2 decision table is validated against a verbatim copy of `inheritance/bootstrap.py::paired_cluster_bootstrap` (seed 0, 10 000 resamples), see T0v.

**Provenance note [corrected after independent verification 2026-10-13]:** the analysis was built against PR #28 head `7886852`. PR #28 has since been merged as main `6014bc7`. A diff of `7886852..6014bc7` changes provenance validation, evidence hashing and input validation in `gate.py`, `interaction.py` and `regression.py`. It changes no threshold, pass rule, bootstrap convention, eligible-set composition or elimination logic, and `generalization.py` is untouched. Every rule the simulation models is therefore identical in merged main.

---

> **Verification status (independent adversarial verification, 2026-10-13; full record in `research/methodology/VERIFICATION-2026-10-13.md`).** A recompute agent re-derived the headline numbers with its own code. A faithfulness agent checked the simulation against the code at merged main `6014bc7` and reran the script, which reproduced every stored table bit-for-bit.
> * **Confirmed:** false-reject rates of ≈ 99 % at +10 pp and ≈ 85 % at +20 pp (§0 item 2). Regression-stage binomials: 40 % at f = 0.05 and 66 % at a true 3 pp rate (item 3). Out-of-scope tolerance void at n = 10, and the τ_r = 0.05 float cell (item 8). Fidelity of the simulation to every locked stage and H1/H2 rule.
> * **Confirmed with correction:** the stage-2 pass rule, which needs the b ≥ 3 rows and has five ordering-dependent compositions, not three (item 1, §4.1, R2, R10). The H2 T_eval MDE, with the grid-exhausted cells filled in and the point rule shown to bind at ρ = 0 (item 5, §4.7, §4.9, §5.1). H2 regression power, where the prose swapped the ρ labels relative to T4 (item 6, §4.7, §4.9, §5.1, §5.2). The 2/3-seed rule under the null, which is ≈ 0.44–0.47 rather than 0.5 because of ties (§4.7). The wording "interaction stage never reached" (item 4, §4.6, §5.1). Provenance (above, §7).
> * **Refuted:** cross-candidate independence of the stage draws in the end-to-end composition (A4, T2). It understates P(arm C empty); the optimistic lower end is ≈ 33 %, not 25 % (item 4, §4.6, R5).
> * Verifiers' sensitivity notes are collected in §4.10. Numbers not listed here (e.g. T1e, T5, T6) were reproduced by rerunning the script but not re-derived independently.

---

## 0. Key results (operating characteristics under the assumed effect models; not predictions)

1. **Stage 2 is an integer rule.** With n = 20 paired instances, "percentile-bootstrap 95 % LB > 0" holds only for ≥ 4 net rescues with zero breaks (or ≥ 7 with one, ≥ 9 with two, ≥ 11 with three, ≥ 13 with four, ≥ 14 with five; never with ≥ 6 breaks). The τ_c = 0.05 point rule (net ≥ 1) is never binding, and the whole secondary τ_c grid {0.03, 0.05, 0.08} cannot change any stage-2 verdict. Validated against the repository bootstrap. Five compositions have RNG-ordering-dependent verdicts: (6,1), (8,2) and (10,3) inside the T0 range (a ≤ 10), plus the nearly deterministic (12,4) at ≈ 98 % and (13,5) at ≈ 0.1 %. [corrected after independent verification 2026-10-13]
2. **False-reject rates for genuinely helpful generalist traits** (all five stages, zero noise): ≈ 99 % at +10 pp, ≈ 85 % at +20 pp, ≈ 60 % at +30 pp, ≈ 35 % at +40 pp. Replay (needs ≥ 2 net rescues of 5) and stage 2 are the binding stages at f = 0; a single-family specialist with +50 pp in-family effect passes stage 2 only 8 % of the time because the stratified pool dilutes it to ≈ +8 pp.
3. **Any perturbation noise is converted into rejection by the regression stage** ("≤ 1 failure of 40"): a neutral trait with per-task flip rate f = 0.05 passes 40 % (symmetric) to 74 % (mean-neutral), with f = 0.10 8 % to 40 %. Stage 4 tests "perturbation-free", not "regression ≤ 3 pp": a trait whose true regression rate equals the 3 pp threshold passes 66 %.
4. **False-pass rates for neutral and harmful traits are ≈ 0** under every model run (≤ 0.1 %); inheritance precision ≈ 100 %, but expected admitted genes per seed are 0.0–1.3 of 10, and **P(arm C empty) ranges from 25 % (optimistic: +40 pp helpful traits, no noise) to 98 %** when candidates are drawn independently. The real candidates share their instance pools, and under that sharing the optimistic lower end is ≈ 33 %: the independent-draw model understates P(arm C empty) (see §4.6). The A1 interaction stage is rarely reached in most cells, but in the optimistic zero-noise cell P(≥ 2 eligible) ≈ 38 %. E[eliminated] = 0 holds there because f = 0 produces no harmful pairs. If reached, the stage flags 15–40 % of neutral pairs. [corrected after independent verification 2026-10-13]
5. **H2 T_eval component:** with redundant seeds (ρ = 1) the binding constraint is the CI, not the +5 pp point rule. With independent seeds (ρ = 0) and small effects the point rule also binds: at f_arm = 0.1 and nominal +5 pp, P(point ≥ 5 pp) = 33.6 % against P(LB > 0) = 43.5 %. Realized MDE at 80 % power with three informative seeds (ρ = 0) is ≈ 7–10 pp across f_arm ∈ {0.1, 0.2, 0.3} (≈ 9.6 pp at f_arm = 0.3). With redundant seeds it is ≈ 10–13 pp at f_arm = 0.1 and ≈ 12–17 pp at f_arm ≥ 0.2 (symmetric noise 15.5 / 16.9 pp, redraw 12.3 / 13.9 pp). Every cell is above the pre-registered +5 pp, including when arm C is less noisy than B. Nominal effects are further attenuated by (1 − 2 f_arm) under symmetric noise. [corrected after independent verification 2026-10-13]
6. **H2 regression component (40 T_reg × 3 seeds):** for a 5–7.5 pp true difference with both arms ≤ 10 pp regression, power is ≈ 12–35 % with redundant seeds (ρ = 1). With independent seeds (ρ = 0) it is ≈ 30–98 %, and reaches 80 % once r_C ≈ 0 and r_B ≥ 5–7.5 pp. (The 4 % figure in T4 belongs to a 2.5 pp-difference cell.) At ρ = 1, ≥ 80 % requires arm B to regress ≥ 20–30 pp of T_reg while C stays ≤ 5 pp, or B/C failures to be strongly coupled within task. The under-powering finding therefore holds when seeds are redundant (ρ ≈ 1) or when r_C is well above 0. It does not hold with independent seeds and r_C ≈ 0, and stage 4's selection makes r_C ≈ 0 the expected case (see §4.7). [corrected after independent verification 2026-10-13] Joint H2 support reaches 80 % only in the corner (ρ = 0, f_arm = 0.1, C − B ≥ +10 pp nominal, r_B ≥ 0.20).
7. **H1:** adequately powered for realized ≥ +10 pp if prompt perturbation is mean-neutral (70–92 % at ρ = 1), but **not identifiable as trait value**: under symmetric noise with p0 < 0.5 the zero-effect "support" rate is 10–43 % because arm A has no added prompt. A placebo-prompt control would make it identifiable (R4).
8. **Two quantisation facts** the manifest does not state: out-of-scope tolerance −0.05 at n = 10 is void (net ≥ 0 required); the implementation's `1.0 − successes/40` makes the secondary τ_r = 0.05 cell identical to τ_r = 0.03 (float boundary; primary rule unaffected).
9. **Not identifiable at this n:** +5–10 pp per-trait effects (stage 2), 1 pp vs 3 pp regression (stage 4), the H2 +5 pp threshold, and a 5–7.5 pp regression-rate difference between arms (the last one when seeds are redundant or r_C is well above 0; see item 6) [corrected after independent verification 2026-10-13]. The unmeasured quantities that most change the picture are the perturbation rate/shape (f, noise model) and the across-seed sharing ρ — both measurable before the run (R3, R4, R8).

All recommendations touching locked elements (R1–R4, R7, R9) are flagged `requires_adr = true` in §6 and are proposals for amendment, not decisions.

## 1. The locked rules exactly as implemented (what the simulation reproduces)

| Stage / decision | Locked rule (design §4–5, ADR-0002, A1) | Integer form at the locked n (what the code actually tests) |
|---|---|---|
| S1 replay | 5 instances (source + 4 same-family bank variants), `S_with ≥ 0.6·5` AND `ΔS ≥ +0.4·5`, expression in every successful with-trait episode | `S_with ≥ 3` AND `S_with − S_without ≥ 2` |
| S2 ablation | n = 20, point estimate ≥ τ_c = 0.05 AND 10k paired cluster percentile-bootstrap 95 % lower bound > 0 (`bootstrap.py`, `random.Random(0)`, LB = `sorted_means[250]`) | net rescues ≥ 1 by the point rule; **LB > 0 requires ≥ 4 net rescues with 0 breaks** (T0). With b breaks: a ≥ 7 / 9 / 11 / 13 / 14 for b = 1 / 2 / 3 / 4 / 5; never for b ≥ 6. Five compositions depend on the task ordering: (6,1), (8,2), (10,3), (12,4), (13,5). [corrected after independent verification 2026-10-13] |
| S3 in-scope | n = 10, Δ ≥ 0 | net ≥ 0 |
| S3 out-of-scope | n = 10, Δ ≥ −0.05 | **net ≥ 0** (−1/10 = −0.10 < −0.05): the −5 pp tolerance is void at n = 10 |
| S4A regression | 40 T_reg tasks, baseline = 1 by construction (not rerun), `1 − S_with/40 ≤ τ_r = 0.03` | **≤ 1 failure of 40** |
| S4B overhead | ≤ 512 tokens | deterministic; assumed to pass |
| S4C interaction (A1) | both-vs-neither on 10 tasks, harmful iff Δ_pair < −0.05; lower stage-2 contribution member of each harmful pair eliminated, all pairs resolved from complete evidence, no cascade | harmful iff **net ≤ −1** of 10 |
| H2 | (C − B) ≥ +5 pp on T_eval (134) with cluster-bootstrap 95 % CI excluding 0 AND reg(C) < reg(B) with CI on the difference excluding 0 AND direction in ≥ 2/3 mining seeds | see §2 A6 for the CI construction assumed |
| H1 | (C − A) CI excludes 0, positive | — |

Two arithmetic facts about the implementation (verified in Python 3.14 with the exact expressions used by the code):

* `regression.py` computes `drop = 1.0 - successes / 40`. For 2 failures this is `0.050000000000000044`, which is **not** `≤ 0.05`. The primary rule (τ_r = 0.03) is unaffected (1 failure → `0.025000000000000022 ≤ 0.03`), but the **secondary sensitivity cell τ_r = 0.05 in `gate.py::sensitivity_recomputation` behaves identically to τ_r = 0.03** (≤ 1 failure) instead of tolerating 2. The τ_r = 0.02 cell correctly tolerates 0 failures. (Implementation detail; not a locked-rule change.)
* All other comparisons (`1/20 ≥ 0.05`, `0/10 ≥ −0.05`, `−1/10 < −0.05`) are float-safe.

## 2. Assumptions (explicit; every one is a modelling choice, none is measured)

* **A1 — Binary paired outcomes, deterministic backbone.** ALFWorld is deterministic and eval temperature is 0, so each (task instance, condition) has one fixed outcome. A trait changes an instance's outcome only through a *discordant flip* (0→1 "rescue" or 1→0 "break"). All stage statistics are therefore functions of discordant-pair counts.
* **A2 — Generative model per instance.** `Y0 ~ Bernoulli(p0)` (without trait). Effect step: a trait with nominal effect δ > 0 rescues a failed instance with probability δ/(1 − p0); δ < 0 breaks a solved instance with probability −δ/p0 (so the mean effect before noise is exactly δ). **Noise step:** because any added prompt text perturbs a temperature-0 trajectory, even an exactly neutral trait flips outcomes at rate *f*. Primary model = **symmetric flip** (as in the task brief): the post-effect outcome is inverted with probability f. Its realized mean effect is `δ(1 − 2f) + f(1 − 2p0)`, i.e. symmetric noise is mean-neutral only at p0 = 0.5 and *looks mildly helpful* when p0 < 0.5 (reported as `E[dlt]` in every table). A **mean-neutral "redraw"** model (with probability f the outcome is redrawn from Bernoulli(p0 + δ)) is run for the regression stage, where the two models differ most (T1d).
* **A3 — Base rate p0 is UNVERIFIED by this analysis.** The repository contains no prompting-only Qwen2.5-7B-Instruct ALFWorld success rate (the SkillRL numbers in `research/related-work/skillrl.md` are post-RL). The sibling note `ALFWorld-baselines.md` (same output directory, written by another subagent in this research round; its "V" markings are that agent's, not re-verified here) reports two published training-free valid_unseen/temp-0 anchors for this model, 29.9 % (ExpRAG) and 45.5 % (SAGA), and a 14.8–31.2 % family on an unstated split. The grids used here — p0 ∈ {0.3, 0.5} for stage OCs, p0 = 0.4 for arm-level analyses — bracket those anchors; replace with the measured G0 rate on T_mine once available (R8). That note also observes that T_reg (40 G0-solved tasks out of 60 T_mine) presupposes a G0 rate ≥ 66.7 %, above every anchor; this analysis takes T_reg = 40 as given and does not re-examine that feasibility question.
* **A4 — Stage independence.** Stages use disjoint instance sets, so for a trait with fixed (p0, δ, f) the stage pass events are independent and the pre-interaction pass probability is the product. Expression probability per successful with-trait episode e = 1 (constitutive trait) by default; e = 0.8 and "source instance already solved by G0" are sensitivities (T1b). The overhead stage is assumed to pass. [corrected after independent verification 2026-10-13] Two limits apply. First, overhead is a deterministic property of each trait, so every `all(...)` pass probability and E[admitted] is an upper bound with respect to it. Second, the independence holds within a trait but **not across candidates**. The ablation pool (20), the interaction pool (10), T_reg (40) and the family-indexed replay/generalization banks are shared by all K = 10 candidates, so their without-trait outcomes Y0 are common. The end-to-end composition (T2) nonetheless draws each candidate independently. Per-trait marginals and E[admitted] are unaffected. The joint distribution is not: independence understates P(arm C empty) and the variance of the admitted count (§4.6).
* **A5 — T_reg.** Baseline success = 1 by construction and not rerun (`regression.py`). G0's lifetime on T_mine runs at temperature 0 (`runtime/loop.py:174`; the 0.7 is the miner LLM's temperature, `traits/miner/miner.py`), so the baseline and the with-trait rerun are in the same sampling regime. Per-task failure probability on T_reg is `h_reg = h(1 − f) + (1 − h)f` (symmetric) or `h(1 − f) + f(1 − p0 − δ)` (redraw), h = break probability.
* **A6 — H2/H1 CI construction (the locked text is ambiguous on seed pooling).** Assumed: cluster = task; the per-task statistic is the mean over the 3 mining seeds of (C − B); 10k percentile cluster bootstrap over the 134 tasks with the same index convention as `bootstrap.py`. The seed-direction criterion is evaluated per seed on the per-seed T_eval means. It is not applied to the H2 regression component, because the design states it only for C − B on T_eval; this too is an interpretive choice [corrected after independent verification 2026-10-13]. Alternative readings (per-seed CIs; pooling 402 (task, seed) rows as independent) give different power; the ambiguity itself is a finding (§6, R1).
* **A7 — Seed correlation ρ (bounds).** Arm A's G1 genome does not depend on the mining seed, so arm A's T_eval outcomes are identical across seeds. For arms B and C the flip realizations are either shared across seeds (ρ = 1: seeds carry no extra information, worst case) or independent (ρ = 0: three seeds ≈ three independent replicates, best case). Reality lies between; both bounds are reported.
* **A8 — Arm-level noise.** For H2/H1 the two arms are parameterised directly by true mean effects (δ_B = 0, δ_C = true C − B) and a common arm-level perturbation rate f_arm ∈ {0.1, 0.2, 0.3} that drives B/C discordance (the variance of the paired difference). Arm failures on T_reg are independent between arms (conservative); positive dependence (C's genome ⊂ B's) is a sensitivity (T4b).
* **A9 — Specialist vs generalist traits.** Replay and in-scope generalization sample from the trait's own family, so they see the in-family effect; the stratified 20-task ablation pool sees share × in-family effect (share ≈ 1/6 for a single-family trait among ~6 families). Out-of-scope is run both for an exactly neutral out-of-scope effect ("spec") and for the same effect everywhere ("gen").
* **A10 — Candidate mixtures.** K = 10 per seed, types helpful/neutral/harmful with mixture 20/50/30 % (baseline), 50/30/20 (optimistic), 10/60/30 (pessimistic); harmful nominal effect −10 pp; the same f for all candidates in a cell.
* **A11 — Monte Carlo resolution.** Stage-2 decisions are exact (trinomial + binomial, no MC). H1/H2 Monte Carlo uses 1 000 reps × 1 000 bootstrap resamples per cell; a 10k-resample check on two cells is in T6b. MC standard error on a reported probability is ≤ 1.6 pp.

## 3. Method

1. **Exact stage-2 table.** For each observed composition (a rescues, b breaks, 20 − a − b concordant), `P(resampled mean ≤ 0) = P(A ≤ B)` under Multinomial(20; a/20, b/20, c/20). The locked LB is `sorted_means[250]` of 10 000 seeded resamples, so `LB > 0 ⇔ at most 250 resample means are ≤ 0`; over the RNG stream (the task ordering of the frozen allocation acts as the randomisation) that count is Binomial(10 000, p_ab), giving `P(pass | a, b) = 1[(a − b)/20 ≥ 0.05] · P(Bin(10 000, p_ab) ≤ 250)`. Validated against the verbatim repository bootstrap (T0v).
2. **Exact stage OCs.** Replay (compositions of 5 into the four paired cells, with expression factor e^{S_with}), ablation (trinomial over (a, b) × the exact table), generalization (net ≥ 0 over 10), regression (Binomial(40, h_reg) ≤ 1).
3. **End-to-end Monte Carlo** (3 000 reps): K = 10 candidates per seed drawn from the mixture; independent stage draws with the exact probabilities; stage-2 point estimate sampled for pair resolution; A1 interaction stage simulated on 10 shared tasks for every pair in the eligible set (both-trait condition = both effects + two prompt perturbations, `f_pair = 1 − (1 − f)²`); elimination exactly as `gate.py::_resolve_harmful_pairs`.
4. **H2 / H1 Monte Carlo**: arm outcomes generated from A2/A7/A8 on 134 tasks × 3 seeds; per-task seed-averaged differences; vectorised percentile cluster bootstrap over tasks; decision rules applied literally; the regression component simulated on 40 tasks × 3 seeds with the same bootstrap. MDE at 80 % power by linear interpolation on the effect grid.

---

## 4. Results

All percentages are pass/support probabilities under the assumed effect models (A1–A11). "Nominal δ" is the model input; "realized" is the expected with-minus-without difference actually present after the noise step.

### 4.1 Stage 2 (ablation) is an integer rule: ≥ 4 net rescues with zero breaks

The locked LB > 0 requirement on the 10k percentile bootstrap over 20 paired deltas is exact and reproducible (T0, validated in T0v). It passes only these observed compositions (a rescues, b breaks):

| observed (a, b) | point estimate | P(resample mean ≤ 0) | P(stage-2 pass) |
|---|---|---|---|
| (3, 0) | +0.15 | 0.0388 | 0 % |
| **(4, 0)** | +0.20 | 0.0115 | **100 %** |
| (5, 1) | +0.20 | 0.0588 | 0 % |
| (6, 1) | +0.25 | 0.0284 | 2.1 % (bootstrap-RNG dependent) |
| (7, 1) | +0.30 | 0.0131 | 100 % |
| (8, 2) | +0.30 | 0.0289 | 0.9 % |
| (9, 2) | +0.35 | 0.0150 | 100 % |
| (10, 3) | +0.35 | 0.0258 | 31.9 % |

Consequences: (i) a trait must show an **observed contribution of at least +20 pp with no break** (or ≥ +30 pp with one break, ≥ +35 pp with two) to pass; (ii) the point-estimate threshold τ_c is never the binding constraint at n = 20 — the whole secondary grid τ_c ∈ {0.03, 0.05, 0.08} (net ≥ 1, ≥ 1, ≥ 2) is dominated by the LB > 0 requirement (net ≥ 4), so the pre-planned τ_c sensitivity analysis cannot change any stage-2 verdict (T0b); (iii) compositions whose verdict depends on the bootstrap RNG stream (i.e. on the frozen task ordering) are a reproducibility hazard only in the sense that a different allocation ordering of the same evidence could flip the verdict.

[corrected after independent verification 2026-10-13] The table above stops at b = 2. An independent enumeration of every composition with net ≥ 1 used two methods: the exact multinomial tail, and the seed-0 stream of `bootstrap.py` over 4 000 random orderings per composition. It extends the rule. With three breaks a ≥ 11 always passes, with four a ≥ 13, with five a ≥ 14, and with six or more breaks no composition passes. Five compositions depend on the ordering, with pass rates (6,1) ≈ 2 %, (8,2) ≈ 1 %, (10,3) ≈ 32 %, (12,4) ≈ 98 % and (13,5) ≈ 0.1 %. The last two fall outside the T0 range (a ≤ 10) and are nearly deterministic. A separate faithfulness check of 20 000 orderings per cell against a replica of the repository bootstrap agreed to ≤ 0.8 pp in every cell. That check adds a caveat: reading the ordering as a randomisation holds only if the frozen allocation order is exchangeable with respect to outcomes. A family-sorted order combined with a specialist trait would make the verdict order-dependent.

### 4.2 Stage operating characteristics (exact; symmetric noise; e = 1)

Selected rows of T1 (full grid in the appendix). `all` = product over the five stages before interaction (generalist trait).

| p0 | nominal δ | f | realized | replay | ablation | gen-in | gen-out | regression | all (generalist) |
|---|---|---|---|---|---|---|---|---|---|
| 0.5 | −0.10 | 0.00 | −0.100 | 0.0 % | 0.0 % | 34.9 % | 34.9 % | 0.1 % | 0.0 % |
| 0.5 | −0.05 | 0.05 | −0.045 | 0.4 % | 0.0 % | 56.5 % | 56.5 % | 1.8 % | 0.0 % |
| 0.5 | 0.00 | 0.00 | 0.000 | 0.0 % | 0.0 % | 100 % | 100 % | 100 % | 0.0 % |
| 0.5 | 0.00 | 0.05 | 0.000 | 0.5 % | 0.1 % | 81.8 % | 81.8 % | 39.9 % | 0.0 % |
| 0.5 | 0.00 | 0.10 | 0.000 | 1.7 % | 0.7 % | 72.5 % | 72.5 % | 8.0 % | 0.0 % |
| 0.5 | 0.00 | 0.20 | 0.000 | 5.3 % | 2.2 % | 64.7 % | 64.7 % | 0.1 % | 0.0 % |
| 0.5 | +0.05 | 0.00 | +0.050 | 2.0 % | 1.6 % | 100 % | 100 % | 100 % | 0.0 % |
| 0.5 | +0.10 | 0.00 | +0.100 | 7.5 % | 13.3 % | 100 % | 100 % | 100 % | 1.0 % |
| 0.5 | +0.10 | 0.05 | +0.090 | 8.9 % | 12.4 % | 91.7 % | 91.7 % | 39.9 % | 0.4 % |
| 0.5 | +0.10 | 0.10 | +0.080 | 10.2 % | 10.8 % | 85.3 % | 85.3 % | 8.0 % | 0.1 % |
| 0.5 | +0.20 | 0.00 | +0.200 | 25.2 % | 58.9 % | 100 % | 100 % | 100 % | 14.8 % |
| 0.5 | +0.20 | 0.05 | +0.180 | 24.3 % | 40.8 % | 96.4 % | 96.4 % | 39.9 % | 3.7 % |
| 0.5 | +0.20 | 0.10 | +0.160 | 23.4 % | 28.5 % | 92.4 % | 92.4 % | 8.0 % | 0.5 % |
| 0.5 | +0.30 | 0.00 | +0.300 | 46.5 % | 89.3 % | 100 % | 100 % | 100 % | 41.5 % |
| 0.5 | +0.30 | 0.05 | +0.270 | 42.3 % | 65.3 % | 98.5 % | 98.5 % | 39.9 % | 10.7 % |
| 0.5 | +0.40 | 0.00 | +0.400 | 66.1 % | 98.4 % | 100 % | 100 % | 100 % | 65.1 % |
| 0.5 | +0.40 | 0.05 | +0.360 | 59.4 % | 82.1 % | 99.5 % | 99.5 % | 39.9 % | 19.2 % |
| 0.5 | +0.40 | 0.10 | +0.320 | 53.0 % | 64.8 % | 98.2 % | 98.2 % | 8.0 % | 2.7 % |
| 0.3 | +0.10 | 0.00 | +0.100 | 6.0 % | 13.3 % | 100 % | 100 % | 100 % | 0.8 % |
| 0.3 | +0.20 | 0.00 | +0.200 | 21.3 % | 58.9 % | 100 % | 100 % | 100 % | 12.5 % |
| 0.3 | +0.40 | 0.00 | +0.400 | 62.0 % | 98.4 % | 100 % | 100 % | 100 % | 61.0 % |
| 0.3 | +0.40 | 0.05 | +0.380 | 58.5 % | 89.4 % | 99.8 % | 99.8 % | 39.9 % | 20.8 % |

Reading: **false-reject rates for genuinely helpful generalist traits** (1 − `all`): ≈ 99 % at +10 pp, ≈ 85–87 % at +20 pp, ≈ 59–63 % at +30 pp and ≈ 35–39 % at +40 pp with zero perturbation noise; any noise f ≥ 0.05 pushes the false-reject rate above 80 % even at +40 pp, and f ≥ 0.10 above 97 %. The per-stage culprits at f = 0 are replay (needs ≥ 2 net rescues among 5 instances ⇒ a ≥ 40 % discordance rate for 50 % pass) and stage 2 (needs ≥ 4 net rescues among 20 with 0 breaks); at f ≥ 0.05 the regression stage becomes co-dominant (T1c).

**False-pass rates for neutral and harmful traits** are ≈ 0 at every grid point (the largest value in the full grid is 0.1 %, p0 = 0.3, δ = +0.05, f = 0.05 — a trait that is marginally helpful, not neutral). Symmetric noise alone cannot manufacture ≥ 4 net rescues with 0 breaks except rarely (stage 2: 0.1–2.8 % for a neutral trait at f = 0.05–0.10; 10.7 % at p0 = 0.3, f = 0.20 because of the mean bias of symmetric noise), and the regression stage then rejects almost all of them.

### 4.3 Replay sensitivities (T1b)

| p0 | nominal δ | f | e = 1, source ∼ variants | e = 0.8 | source already solved | e = 0.8 & source solved |
|---|---|---|---|---|---|---|
| 0.5 | +0.10 | 0.0 | 7.5 % | 3.2 % | 5.2 % | 2.1 % |
| 0.5 | +0.20 | 0.0 | 25.2 % | 10.3 % | 18.1 % | 7.0 % |
| 0.5 | +0.40 | 0.0 | 66.1 % | 23.8 % | 52.5 % | 18.4 % |
| 0.5 | +0.40 | 0.1 | 53.0 % | 19.9 % | 39.4 % | 14.3 % |

A regulated trait that fires on only 80 % of successful with-trait episodes loses roughly 60 % of its replay pass probability through the expression check (which requires firing in **every** successful episode). If the source instance is one G0 already solves at temperature 0 (the trait was mined from that success), the source cannot contribute a rescue and the +2 net must come from the 4 variants.

### 4.4 Regression stage: a test of "perturbation-free", not of "regression ≤ 3 pp" (T1c, T1d)

| per-task failure prob. on T_reg, h_reg | P(pass), τ_r = 0.03 (≤ 1 fail) | τ_r = 0.02 (0 fails) | τ_r = 0.05 as intended (≤ 2 fails) |
|---|---|---|---|
| 0.01 | 93.9 % | 66.9 % | 99.3 % |
| 0.02 | 81.0 % | 44.6 % | 95.4 % |
| 0.03 | 66.2 % | 29.6 % | 88.2 % |
| 0.05 | 39.9 % | 12.9 % | 67.7 % |
| 0.075 | 18.8 % | 4.4 % | 41.4 % |
| 0.10 | 8.0 % | 1.5 % | 22.3 % |
| 0.20 | 0.1 % | 0.0 % | 0.8 % |

A trait whose *true* regression rate equals the threshold (3 pp) passes only 66 % of the time; one with a true 5 pp perturbation rate passes 40 %. For a neutral trait, h_reg = f under symmetric noise, f(1 − p0) under mean-neutral redraw (T1d: f = 0.05 → 40 % / 59 % / 74 % pass at symmetric / redraw p0 = 0.3 / redraw p0 = 0.5). Note the implementation artefact from §1: the τ_r = 0.05 cell, as coded, equals the τ_r = 0.03 column.

### 4.5 Specialist traits (T1e)

A single-family specialist with a large in-family effect faces the stratified 20-task pool where only ≈ 1/6 of instances are in scope:

| in-family δ | family share of pool | pool δ | replay | ablation (f = 0) | ablation (f = 0.05) |
|---|---|---|---|---|---|
| +0.30 | 1/6 | +0.050 | 41–47 % | 1.6 % | 3–6 % |
| +0.50 | 1/6 | +0.083 | 79–81 % | 8.0 % | 9–13 % |
| +0.50 | 1/3 | +0.167 | 79–81 % | 43.3 % | 31–41 % |
| +0.80 | 1/6 | +0.133 | 81–97 % | 27.3 % | 21–29 % |
| +0.80 | 1/3 | +0.267 | 81–97 % | 82.2 % | 58–69 % |

The in-scope/out-of-scope split (critic round 1, R1) protects specialists at stage 3 but **not at stage 2**, where the stratified pool dilutes a +50 pp in-family effect to +8 pp and the integer rule rejects it 92 % of the time.

### 4.6 End-to-end: expected arm-C composition from K = 10 (T2, 3 000 MC reps, incl. A1 interaction)

| p0 | mixture H/N/Harm | nominal δ_help | f | E[eligible] | E[admitted] | P(arm C empty) | E[neutral adm.] | E[harmful adm.] | P(any non-helpful admitted) |
|---|---|---|---|---|---|---|---|---|---|
| 0.5 | 20/50/30 | +0.10 | 0.00 | 0.02 | 0.02 | 98.0 % | 0.000 | 0.000 | 0.0 % |
| 0.5 | 20/50/30 | +0.20 | 0.00 | 0.30 | 0.30 | 73.9 % | 0.000 | 0.000 | 0.0 % |
| 0.5 | 20/50/30 | +0.20 | 0.05 | 0.07 | 0.07 | 92.7 % | 0.000 | 0.000 | 0.0 % |
| 0.5 | 20/50/30 | +0.40 | 0.00 | 1.30 | 1.30 | 24.5 % | 0.000 | 0.000 | 0.0 % |
| 0.5 | 20/50/30 | +0.40 | 0.05 | 0.41 | 0.41 | 65.1 % | 0.000 | 0.000 | 0.0 % |
| 0.5 | 20/50/30 | +0.40 | 0.10 | 0.05 | 0.05 | 94.8 % | 0.000 | 0.000 | 0.0 % |
| 0.5 | 50/30/20 | +0.20 | 0.00 | 0.74 | 0.74 | 47.0 % | 0.000 | 0.000 | 0.0 % |
| 0.5 | 10/60/30 | +0.20 | 0.00 | 0.15 | 0.15 | 85.3 % | 0.000 | 0.000 | 0.0 % |
| 0.3 | 20/50/30 | +0.40 | 0.00 | 1.24 | 1.24 | 27.1 % | 0.000 | 0.000 | 0.0 % |
| 0.3 | 50/30/20 | +0.20 | 0.00 | 0.64 | 0.64 | 51.6 % | 0.000 | 0.000 | 0.0 % |

Family-wise behaviour across the 10 candidates: inheritance precision (SPEC §18) is ≈ 100 % in every cell — under these models **no neutral or harmful candidate is admitted in 3 000 × 10 draws per cell** (one isolated admission in the whole grid) — but recall is so low that the expected arm-C genome holds 0.02–1.3 genes and is **empty with probability 25–98 %**.

[corrected after independent verification 2026-10-13] **The P(arm C empty) column is optimistic.** T2 draws each candidate's stage outcomes independently. In the locked design the ablation, interaction, T_reg and family-indexed replay/generalization pools are shared by all K = 10 candidates (A4). An independent instance-level Monte Carlo with shared Y0 on the ablation, replay and generalization pools (6 000 reps) leaves E[eligible] unchanged within MC error, but raises P(arm C empty) in every cell checked: 24.4 → 33.1 % (p0 = 0.5, +40 pp, f = 0), 27.1 → 30.2 % (p0 = 0.3, +40 pp, f = 0), 74.8 → 75.6 % (+20 pp), 66.7 → 68.9 % (+40 pp, f = 0.05). It also raises the variance of the admitted count by ≈ 37 % in the first cell. The lower end of the range is therefore ≈ 33 %, not 25 %. The means, precision and all marginal stage numbers stand, and the P(any non-helpful admitted) column is likewise mildly optimistic under the independent-draw reading.

The A1 interaction stage needs ≥ 2 eligible candidates and is rarely reached in most cells. In the optimistic cell (p0 = 0.5, +40 pp helpful, f = 0) P(≥ 2 eligible) is ≈ 38 %, against ≈ 5 % at f = 0.05 and ≈ 3 % at +20 pp, f = 0. E[eliminated] = 0.00 in every cell, because at f = 0 no pair is harmful, not because the stage is never reached [corrected after independent verification 2026-10-13]. When it *is* reached, the integer form of its rule (net ≤ −1 of 10, with two prompts' perturbation) flags a pair of two exactly neutral traits with probability 15–16 % (p0 = 0.3) to 27–40 % (p0 = 0.5) for f = 0.05–0.20 (exact calculation), i.e. it would then remove roughly one of every three to six admitted pairs for no causal reason.

If arm C is empty, G1(C) = G0 = G1(A): the H2 contrast degenerates to (A − B) and H1 to exactly 0. Design §6.5 already pre-declares "B ≈ C because few candidates are harmful" as a finding; the complementary degenerate case "C ≈ A because the gate admits nothing" is not pre-declared (see R5).

### 4.7 H2 decision power (T3–T5)

**T_eval component** — P[point(C − B) ≥ +5 pp AND LB > 0 AND direction in ≥ 2/3 seeds], 134 paired tasks × 3 seeds, p0 = 0.4 (unverified), δ_B = 0, f_B = f_C = f_arm, 1 000 reps × 1 000 resamples:

| ρ (seed sharing) | f_arm | nominal C−B | realized C−B | P(point ≥ 5 pp) | P(LB > 0) | P(dir 2/3) | **P(T_eval part)** |
|---|---|---|---|---|---|---|---|
| 0 (independent seeds) | 0.1 | 0.00 | 0.000 | 0.8 % | 2.9 % | 43.8 % | 0.8 % |
| 0 | 0.1 | +0.05 | +0.041 | 33.6 % | 43.5 % | 90.4 % | 33.6 % |
| 0 | 0.1 | +0.10 | +0.081 | 88.0 % | 90.1 % | 99.8 % | 87.7 % |
| 0 | 0.1 | +0.15 | +0.121 | 99.5 % | 99.5 % | 100 % | 99.5 % |
| 0 | 0.2 | +0.05 | +0.030 | 25.1 % | 19.9 % | 76.1 % | 19.6 % |
| 0 | 0.2 | +0.10 | +0.061 | 63.8 % | 54.0 % | 96.9 % | 53.9 % |
| 0 | 0.2 | +0.15 | +0.090 | 89.7 % | 84.1 % | 99.2 % | 84.1 % |
| 0 | 0.3 | +0.10 | +0.040 | 35.5 % | 24.7 % | 83.1 % | 24.7 % |
| 0 | 0.3 | +0.20 | +0.079 | 80.5 % | 65.0 % | 96.2 % | 64.9 % |
| 1 (seeds identical) | 0.1 | 0.00 | +0.001 | 9.0 % | 1.9 % | 47.1 % | 1.9 % |
| 1 | 0.1 | +0.05 | +0.039 | 39.7 % | 13.6 % | 83.2 % | 13.6 % |
| 1 | 0.1 | +0.10 | +0.079 | 78.2 % | 44.4 % | 95.8 % | 44.4 % |
| 1 | 0.1 | +0.15 | +0.119 | 94.3 % | 74.1 % | 99.6 % | 74.1 % |
| 1 | 0.1 | +0.20 | +0.158 | 99.5 % | 92.9 % | 100 % | 92.9 % |
| 1 | 0.2 | +0.10 | +0.060 | 58.1 % | 19.4 % | 86.9 % | 19.4 % |
| 1 | 0.2 | +0.15 | +0.088 | 77.0 % | 36.9 % | 95.4 % | 36.9 % |
| 1 | 0.2 | +0.20 | +0.120 | 91.8 % | 59.2 % | 98.6 % | 59.2 % |
| 1 | 0.3 | +0.20 | +0.080 | 69.3 % | 27.1 % | 90.7 % | 27.1 % |

Type-I behaviour is controlled by the CI (P(LB > 0) = 2–3 % at zero effect), not by the +5 pp point threshold (9–18 % at zero effect when ρ = 1). [corrected after independent verification 2026-10-13] At ρ = 1 the binding constraint is `LB > 0`. At ρ = 0 and small effects the point rule binds instead: at f_arm = 0.1 and nominal +5 pp, P(point ≥ 5 pp) = 33.6 % against P(LB > 0) = 43.5 %, and the two are about equal at +10 pp (88.0 % vs 90.1 %). The 2/3-seed rule is rarely the reason for failure once the CI excludes 0, and under the null it controls essentially nothing. With continuous, independent, symmetric seeds, P(C − B > 0 in ≥ 2 of 3 seeds) = 0.5 exactly. With the discrete per-seed means over 134 paired binary tasks, ties have probability 8.1 / 6.1 / 5.3 % (f_arm = 0.1 / 0.2 / 0.3), so the null probability is ≈ 0.44–0.46 at ρ = 0 and ≈ 0.46–0.47 at ρ = 1. At ρ = 1 the reason is different: identical seeds reduce the rule to the sign of one mean. These values agree with the T3 null cells (43.8–47.1 %). Read literally, "same direction in ≥ 2/3 seeds" holds with probability 1 for three untied seeds; the operative reading is "C − B > 0 in ≥ 2/3 seeds". Type-I control comes from `LB > 0`.

**Minimum detectable effect at 80 % power** (T3b; nominal → realized = nominal × (1 − 2 f_arm)):

| ρ | f_arm | MDE nominal (full criterion) | MDE nominal (LB > 0 & seed only) | **MDE realized (full criterion)** | MDE realized (LB > 0 & seed only) |
|---|---|---|---|---|---|
| 0 | 0.1 | 9.2 pp | 8.8 pp | **7.4 pp** | 7.0 pp |
| 0 | 0.2 | 14.3 pp | 14.3 pp | **8.5 pp** | 8.5 pp |
| 0 | 0.3 | > 20 pp | > 20 pp | **> 7.9 pp** (grid exhausted at 65 % power); **≈ 9.6 pp** by independent recompute | > 7.9 pp |
| 1 | 0.1 | 16.6 pp | 16.6 pp | **13.1 pp** | 13.1 pp |
| 1 | 0.2 | > 20 pp | > 20 pp | **> 12.0 pp** (59 % power at the grid end); **≈ 15.5 pp** by independent recompute | > 12.0 pp |
| 1 | 0.3 | > 20 pp | > 20 pp | **> 8.0 pp** (27 % power at the grid end); **≈ 16.9 pp** by independent recompute | > 8.0 pp |

Even in the most favourable cell (independent seeds, f_arm = 0.1) the realized MDE is ≈ 7.4 pp, above the pre-registered +5 pp support threshold; with seeds carrying no independent information (ρ = 1) it is ≈ 13 pp, and for f_arm ≥ 0.2 at ρ = 1 the grid never reaches 80 %.

[corrected after independent verification 2026-10-13] An independent recompute used the same model: p0 = 0.4, Y0 shared across seeds, independent symmetric arm noise f_B = f_C = f_arm. It used exact enumeration at ρ = 1 and Monte Carlo with exact bootstrap tails at ρ = 0, and it fills the three grid-exhausted cells (bold additions above). At ρ = 0 the full-criterion realized MDE is 7.3 / 8.6 / 9.6 pp for f_arm = 0.1 / 0.2 / 0.3, so the informative-seed range is ≈ 7–10 pp. If the rescue is shared across seeds it is 8.2 / 9.4 / 10.1 pp. At ρ = 1 it is 12.7 / 15.5 / 16.9 pp, consistent with T3b's 13.1 pp at f_arm = 0.1. Spot cells reproduce T3 (e.g. 87.8 % vs 87.7 %, 75.7 % vs 74.1 %). The conclusion "MDE > pre-registered +5 pp" survives every variant tried. That includes a less noisy arm C (f_C < f_B), which lowers the MDE by 1–2 pp, e.g. 5.6 pp at ρ = 0 and 10.6 pp at ρ = 1 for f_B = 0.1, f_C = 0. Pooling the 402 (task, seed) rows as independent clusters gives 6.8–9.4 pp, which is anti-conservative when ρ is near 1.

A closed-form normal-approximation cross-check (per-task difference variance from the same cell probabilities, n_eff = 134 or 402) gives MDE(LB > 0) ≈ 11.8 / 14.4 / 16.0 pp nominal at ρ = 1 for f_arm = 0.1 / 0.2 / 0.3 and ≈ 6.8 / 8.3 / 9.2 pp at ρ = 0, consistent with the Monte Carlo within its resolution (T6b shows the 1k-resample bootstrap overstates P(LB > 0) by ≈ 1–3 pp relative to 10k, so the MC powers are, if anything, slightly optimistic).

**Regression component** — P[reg(B) − reg(C) > 0 with LB > 0] on 40 T_reg tasks × 3 seeds (T4; arms independent):

| ρ | r_B | r_C | true difference | P(reg part) |
|---|---|---|---|---|
| 0 | 0.05 | 0.025 | +0.025 | 16.1 % |
| 0 | 0.10 | 0.025 | +0.075 | 67.9 % |
| 0 | 0.10 | 0.05 | +0.050 | 34.3 % |
| 0 | 0.15 | 0.05 | +0.100 | 74.4 % |
| 0 | 0.20 | 0.05 | +0.150 | 95.1 % |
| 1 | 0.05 | 0.0 | +0.050 | 13.5 % |
| 1 | 0.10 | 0.025 | +0.075 | 26.3 % |
| 1 | 0.10 | 0.05 | +0.050 | 12.0 % |
| 1 | 0.15 | 0.025 | +0.125 | 54.1 % |
| 1 | 0.20 | 0.05 | +0.150 | 53.5 % |
| 1 | 0.30 | 0.05 | +0.250 | 87.2 % |
| 1 | 0.40 | 0.05 | +0.350 | 97.6 % |

[corrected after independent verification 2026-10-13] With both arms near ceiling (reg ≤ 10 pp) and a 5–7.5 pp true difference, the 40-task criterion has ≈ 12–35 % power when seeds are redundant (ρ = 1). With independent seeds (ρ = 0) it has ≈ 30–98 %, as T4 itself shows. The independent recompute gives, at ρ = 0 / ρ = 1: (0.10, 0.05) 30.2 / 12.0 %; (0.075, 0.025) 41.9 / 14.7 %; (0.10, 0.025) 68.0 / 26.6 %; (0.05, 0) 83.9 / 13.8 %; (0.075, 0) 98.0 / 35.3 %. These match T4 within its MC error. The 4 % figure belongs to a 2.5 pp-difference cell (0.05 / 0.025, ρ = 1), and "≥ 80 % only when B regresses ≥ 20–30 pp" holds only at ρ = 1. **A validity point the power model treats as a free parameter:** each arm-C gene has already passed stage 4 on these same 40 T_reg tasks (≤ 1 failure), and an empty arm C is G0, which reproduces the baseline. With a deterministic backend, r_C is therefore ≈ 0 by selection (at most 1/40 for a single gene). The realistic case sits at the r_C = 0 end. There, at ρ = 1, power is P(Bin(40, r_B) ≥ 4) = 13.8 % at r_B = 5 pp and 57.7 % at 10 pp. It also means that reg(C) < reg(B) is partly guaranteed by construction. Positive dependence of failures (C's skills ⊂ B's; T4b, ρ = 1) raises P(reg part) from 28.8 % → 57.3 % (r_B = 0.10, r_C = 0.025) and from 54.8 % → 96.0 % (0.20 / 0.05) at full dependence — the regression criterion is therefore sensitive to an assumption (within-task failure coupling) that the design does not control.

**Joint H2 support** (T5, product of the two components):

| ρ | f_arm | nominal C−B | r_B / r_C | P(H2 supported) |
|---|---|---|---|---|
| 0 | 0.1 | +0.05 | 0.10 / 0.025 | 22.8 % |
| 0 | 0.1 | +0.10 | 0.20 / 0.05 | 83.4 % |
| 0 | 0.1 | +0.15 | 0.30 / 0.05 | 99.5 % |
| 0 | 0.2 | +0.10 | 0.20 / 0.05 | 51.3 % |
| 0 | 0.2 | +0.15 | 0.10 / 0.025 | 57.1 % |
| 1 | 0.1 | +0.05 | 0.10 / 0.025 | 3.6 % |
| 1 | 0.1 | +0.10 | 0.20 / 0.05 | 23.8 % |
| 1 | 0.1 | +0.15 | 0.30 / 0.05 | 64.6 % |
| 1 | 0.2 | +0.10 | 0.20 / 0.05 | 10.4 % |
| 1 | 0.2 | +0.15 | 0.30 / 0.05 | 32.2 % |

H2 reaches 80 % joint power only in the most favourable corner (independent seeds, low arm noise, nominal C − B ≥ +10 pp, and arm B regressing ≥ 20 pp of T_reg while C regresses ≤ 5 pp).

### 4.8 H1 power (T6) and the prompt-perturbation confound

| ρ | f_C | nominal C−A | realized C−A (symmetric) | P(H1 supported) |
|---|---|---|---|---|
| 0 | 0.1 | 0.000 | +0.020 | 20.9 % |
| 0 | 0.1 | +0.050 | +0.060 | 84.4 % |
| 0 | 0.2 | 0.000 | +0.040 | 33.0 % |
| 0 | 0.2 | +0.050 | +0.070 | 68.3 % |
| 0 | 0.3 | 0.000 | +0.061 | 42.7 % |
| 1 | 0.1 | 0.000 | +0.020 | 10.0 % |
| 1 | 0.1 | +0.050 | +0.059 | 43.6 % |
| 1 | 0.1 | +0.100 | +0.099 | 78.5 % |
| 1 | 0.2 | +0.100 | +0.100 | 62.7 % |
| 1 | 0.3 | +0.150 | +0.120 | 62.3 % |

Two readings. (a) Power: at ρ = 1 a realized +10 pp gives 48–79 % and +6 pp gives 37–44 %; only with independent seeds does +6 pp reach ≈ 84 %. (b) **Identifiability:** arm A carries no added prompt, so H1's (C − A) contrast includes whatever systematic effect prompt perturbation has. Under the symmetric-noise model with p0 < 0.5 that effect is +f(1 − 2p0) (+2 / +4 / +6 pp here) and the zero-effect rows show 10–43 % "support" — not a type-I error of the statistic but a confound of the design. Under the mean-neutral redraw model it vanishes:

| ρ | f_C | nominal C−A | realized C−A (redraw) | P(H1 supported), redraw | P(H1 supported), symmetric (from above) |
|---|---|---|---|---|---|
| 0 | 0.1 | 0.000 | 0.000 | 2.1 % | 20.9 % |
| 0 | 0.1 | +0.050 | +0.050 | 87.6 % | 84.4 % |
| 0 | 0.2 | 0.000 | 0.000 | 2.6 % | 33.0 % |
| 0 | 0.3 | 0.000 | −0.001 | 2.7 % | 42.7 % |
| 1 | 0.1 | 0.000 | 0.000 | 2.5 % | 10.0 % |
| 1 | 0.1 | +0.050 | +0.050 | 46.1 % | 43.6 % |
| 1 | 0.1 | +0.100 | +0.100 | 91.6 % | 78.5 % |
| 1 | 0.2 | +0.100 | +0.101 | 80.1 % | 62.7 % |
| 1 | 0.3 | +0.100 | +0.100 | 70.4 % | 47.5 % |

Under mean-neutral noise the H1 statistic is well calibrated (2.0–2.7 % at zero effect) and a realized +10 pp reaches 70–92 % power at ρ = 1; H1 is therefore adequately powered for ≥ +10 pp effects *if* the perturbation is mean-neutral — which is exactly the unmeasured quantity.

Which model is closer to reality is **unknown and not measurable within the locked design** (no placebo-prompt arm; see R4). The same perturbation cancels in H2 only to the extent that f_B ≈ f_C; arm B's prompt is ≈ 10 × longer than a typical arm-C prompt, so f_B > f_C is the more plausible case and the (C − B) contrast then also inherits a perturbation difference of unknown sign.

### 4.9 Sensitivity to the noise-model shape (mean-neutral "redraw" run, Appendix B)

The same script with `--model redraw` (perturbed outcomes redrawn from Bernoulli(p0 + δ); exactly mean-neutral; T_reg failure probability f(1 − p0 − δ) instead of f) changes the picture in one direction only — the regression stage and everything downstream of it become less punitive — while the integer rules of stages 1–3 and the H2 noise floor are unchanged:

| quantity (p0 = 0.5 unless noted) | symmetric | redraw |
|---|---|---|
| neutral trait, f = 0.05: P(regression pass) | 39.9 % | 73.6 % |
| neutral trait, f = 0.10: P(regression pass) | 8.0 % | 39.9 % |
| helpful +20 pp, f = 0.10: P(all stages) | 0.5 % | 8.9 % |
| helpful +40 pp, f = 0.05 / 0.10 / 0.20: P(all stages) | 19.2 / 2.7 / 0.0 % | 63.1 / 59.5 / 50.0 % |
| E[admitted] of K = 10, 20/50/30, δ_help = +40 pp, f = 0.10 | 0.05 (P empty 94.8 %) | 1.20 (P empty 27.4 %) |
| E[admitted], δ_help = +20 pp, f = 0.10 | 0.01 (98.5 % empty) | 0.18 (83.7 % empty) |
| neutral / harmful admitted (any cell) | 0.000 | 0.000 |
| H2 T_eval part, ρ = 1, f_arm = 0.1, realized +10 pp | 44.4 % (nominal +12.5 pp needed) | 83.3 % |
| H2 realized MDE, ρ = 0 / ρ = 1, f_arm = 0.1 | 7.4 / 13.1 pp | 7.0 / 9.7 pp |
| H2 realized MDE, ρ = 1, f_arm = 0.2 / 0.3 | > 12.0 / > 8.0 pp (grid exhausted); ≈ 15.5 / 16.9 pp by independent recompute [corrected after independent verification 2026-10-13] | 12.3 / 13.9 pp |
| H2 regression part, ρ = 1, r_B = 0.10, r_C = 0.025 | 26.3 % | 27.1 % |
| H1 at zero effect, f_C = 0.1–0.3 | 10–43 % | 2–3 % |

Conclusions robust to the noise shape: stage 2 needs ≥ +20 pp observed with no break; replay needs ≥ 2 net rescues of 5; false passes ≈ 0; the realized H2 MDE is ≥ 7 pp (independent seeds) and ≥ 10 pp (redundant seeds), above the pre-registered +5 pp; the 40-task regression criterion is under-powered when both arms regress ≤ 10 pp, provided seeds are redundant or r_C is well above 0 (at ρ = 0 with r_C ≈ 0 and r_B ≥ 5 pp it reaches ≥ 80 %; §4.7) [corrected after independent verification 2026-10-13]. Conclusions that depend on the noise shape (and thus on a measurable but unmeasured quantity, R3/R4): how many helpful candidates survive stage 4, whether arm C is empty, and whether H1 is confounded.

### 4.10 Sensitivity of conclusions (from the independent verification, 2026-10-13)

[corrected after independent verification 2026-10-13] The recompute agent added the following adversarial checks. They are model-based operating characteristics like everything above, not measurements.

* **The exact sign test is stricter than the bootstrap rule.** The locked percentile-bootstrap LB > 0 rule is *more lenient* than an exact one-sided sign/McNemar test on the discordant pairs. (4,0) passes the bootstrap, but its one-sided sign-test p = 0.0625. With 0 breaks an exact test needs 6 rescues at α = 0.025 or 5 at α = 0.05. At δ = +20 pp and f = 0, stage-2 pass would fall from 58.9 % to 37.0 % (α = 0.05) or 19.6 % (α = 0.025). So the strictness of stage 2 comes from n = 20, not from the bootstrap. Option R2(b), an exact test at a conventional α ≤ 0.05, would **tighten** stage 2, not loosen it.
* **Source-rescue replay assumption.** The replay sensitivities explored here (T1b) are one-sided pessimistic: they vary only toward "source already solved by G0". If instead the trait was mined from the source's *failure* and deterministically rescues it, the five-stage false-reject rate drops from ≈ 99 % to ≈ 95.8–96.6 % at +10 pp and from ≈ 85–87 % to ≈ 66.5–71.1 % at +20 pp (f = 0). Conversely, reading δ as a *relative* effect (rescue probability δ among failures) raises false-reject to 99.97 / 99.0 % (p0 = 0.5) and 99.86 / 96.6 % (p0 = 0.3) at +10 / +20 pp.
* **Symmetric T_reg noise is the pessimistic end.** Under symmetric flipping a neutral trait has h_reg = f. The redraw model gives 58.9 % (p0 = 0.3) and 73.6 % (p0 = 0.5) pass at f = 0.05 instead of 39.9 %. A latent-difficulty view, in which tasks G0 solves are more robust, would lower h_reg further. In the other direction, the T_reg baseline is never rerun, so any vLLM/backend nondeterminism is counted entirely as regression. The 66 % pass at a 3 pp rate is a superpopulation reading. On the fixed temperature-0 T_reg, a trait that breaks a fixed set of k tasks passes deterministically iff k ≤ 1.
* **Non-firing regulated traits leave prompts byte-identical.** When a regulated trait does not fire, the rendered skill block is the same as G0's ("No skills are currently expressed." in both; `runtime/skills.py`, `runtime/regulation.py`). At temperature 0 it can then perturb a T_reg task only through backend nondeterminism, so its h_reg is bounded by its firing rate. The e = 1 constitutive default used throughout is the case where f applies in full.
* **H2 regression contrast is partly circular** (see §4.7): r_C ≈ 0 by selection, so the realistic H2 regression case is the r_C = 0 end of T4.
* **Equal arm noise slightly overstates the H2 MDE.** If arm C is less noisy than B (f_C < f_B), the realized MDE drops by 1–2 pp but stays ≥ +5 pp in every cell tried (§4.7).

## 5. Interpretation

### 5.1 Which locked rules dominate the failure probability

| Rule | Integer form | Dominates for | Mechanism |
|---|---|---|---|
| Stage-2 `LB > 0` (n = 20) | ≥ 4 net rescues and 0 breaks | helpful traits with δ ≤ +30 pp; every specialist | the percentile LB of 20 ternary deltas is > 0 only when the resample mass at ≤ 0 is < 2.5 %, which 3 rescues cannot deliver (0.85²⁰ = 3.9 %) |
| Stage-1 replay (n = 5) | ≥ 2 net rescues of 5 and ≥ 3 successes, trait firing in every success | all helpful traits with δ < +40 pp; regulated traits (e < 1) | 5 instances resolve only ≥ 40 pp discordance; expression check compounds as e^{S_with} |
| Stage-4 regression (n = 40) | ≤ 1 failure | every trait once f ≥ 0.05 | Binomial(40, f) ≤ 1 is 40 % at f = 0.05 and 8 % at f = 0.10, independent of the trait's true value |
| H2 `LB > 0` (n = 134 × 3) | — | H2 overall | realized MDE 7–13 pp (ρ = 0 … 1, f_arm = 0.1) exceeds the pre-registered +5 pp; the +5 pp point rule is non-binding at ρ = 1 but binds at ρ = 0 for small effects [corrected after independent verification 2026-10-13] |
| H2 regression difference (n = 40 × 3) | — | H2 when both arms regress ≤ 10 pp | with redundant seeds (ρ = 1) 40 tasks cannot resolve a 5–7.5 pp difference (≈ 12–35 %); with independent seeds (ρ = 0) power is ≈ 30–98 %, and r_C ≈ 0 by selection [corrected after independent verification 2026-10-13] |
| Stage-3 out-of-scope `≥ −0.05` (n = 10) | net ≥ 0 | neutral traits with f > 0 (65–90 % pass) | tolerance void at n = 10; otherwise weak |
| τ_c grid | net ≥ 1 / 1 / 2 | nothing | dominated by LB > 0 |
| τ_r grid | 0 / 1 / 1(coded) or 2(intended) failures | secondary analysis only | float boundary at 2/40 |
| A1 interaction (n = 10) | net ≤ −1 | nothing in practice (needs ≥ 2 eligible; P(≥ 2 eligible) ≈ 38 % in the optimistic zero-noise cell, where no pair is harmful, and ≈ 3–5 % in the other cells checked) [corrected after independent verification 2026-10-13] | if reached, flags 15–40 % of neutral pairs |

### 5.2 What is identifiable at this n

* Stage 2 can identify helpful traits only in the regime δ_pool ≳ +30 pp with f ≈ 0 (≥ 89 %); +5 to +10 pp effects are indistinguishable from 0 (1.6–13 %). Specialists are identifiable only if their family covers ≥ 1/3 of the pool and the in-family effect is ≥ +50 pp.
* Stage 4 cannot distinguish a true 1 pp regression from 3 pp (94 % vs 66 %) and converts a 5 pp perturbation rate into a 60 % rejection; its verdict is dominated by f, which the design neither measures nor controls (A2, A5).
* The gate's **precision is identifiable and excellent** (≈ 0 false passes under every model here); its **recall is not** — the expected admitted count is < 1.5 in every cell, so EXP-0001 as locked will mostly measure "what an empty or one-gene arm C does", and a B ≈ C or C ≈ A outcome will not discriminate "gate unnecessary" (design §6.5) from "gate too strict".
* H2's T_eval component identifies realized effects ≥ 13 pp (ρ = 1, f_arm = 0.1) to ≥ 7 pp (ρ = 0); the pre-registered +5 pp threshold sits below the MDE in every cell. The seed-sharing fraction ρ — not measurable before the run — moves the MDE by almost a factor 2, and ρ = 1 is exact for arm A.
* With redundant seeds (ρ = 1), the regression half of H2 identifies a difference only when arm B regresses ≥ 15–30 pp of T_reg while arm C stays ≤ 5 pp. With independent seeds (ρ = 0) and r_C ≈ 0, which stage 4 makes the expected case, r_B ≥ 5 pp already gives ≥ 80 %. Part of reg(C) < reg(B) is then guaranteed by construction (§4.7). [corrected after independent verification 2026-10-13]
* H1 is not identifiable as "trait value" without separating prompt perturbation from trait effect.

## 6. Recommendations

Every item that would touch a locked or pre-registered element is marked **requires_adr = true**: it must go through an ADR amendment and the critic review (SPEC §32–33) *before* run start and must never be applied silently. Items marked false are analysis/implementation hygiene that change no locked rule.

| # | Recommendation | requires_adr | priority | owner |
|---|---|---|---|---|
| R1 | **Specify the H1/H2 estimator across mining seeds** in ADR-0002 / manifest: cluster = task with per-task seed-averaged (C − B) (assumed here), or per-seed CIs, or pooled (task, seed) rows. The three readings have different power and the locked text ("cluster bootstrap over tasks … across 3 independent mining seeds") admits all of them. Pre-register one and the bootstrap index convention (`bootstrap.py`). | **true** | P0 | builder + critic |
| R2 | **Re-examine the stage-2 rule at n = 20 before run start.** As locked it is "≥ 4 net rescues, 0 breaks" (observed ≥ +20 pp), makes τ_c and its whole sensitivity grid inert, and has 5 RNG/ordering-dependent compositions ((6,1), (8,2), (10,3), (12,4), (13,5)). Options to evaluate via ADR (not decided here): (a) raise n_ablation (this script can quantify; e.g. `--delta 0.10 --f 0` with a modified `ABL_N`), (b) replace percentile-LB > 0 by an exact one-sided test on discordant pairs (sign / McNemar-type) with a pre-registered α. At α ≤ 0.05 this would make stage 2 *stricter*, not more lenient: it needs 5–6 rescues with 0 breaks against the bootstrap's 4 (§4.10) [corrected after independent verification 2026-10-13], (c) keep the rule and pre-register the implied expected pass counts (≈ 0–1 of 10) explicitly. | **true** | P0 | critic (owner decides) |
| R3 | **Separate perturbation noise from regression in stage 4.** "≤ 1 failure of 40" is a test of f, not of regression ≤ 3 pp. Options via ADR: a pre-run *placebo probe* (G0 + neutral text of matched length on T_reg and on a T_gate subset) to measure f, with τ_r pre-registered relative to the measured floor; or a larger T_reg; or a paired placebo-vs-trait contrast instead of absolute drop. | **true** | P1 | critic + builder |
| R4 | **Add a placebo-prompt control (arm A′ or a pre-run probe)** so that H1 (C − A) is identifiable as trait value rather than trait + prompt perturbation; also lets f_B vs f_C be estimated for H2. Adding an arm changes the locked design and budget (≈ +134 × 3 episodes for A′ on T_eval; +40 × 3 on T_reg). | **true** | P1 | critic (owner decides) |
| R5 | **Fill the open pre-registration fields (design §8) with these operating characteristics**: expected candidates passing each stage (≈ 0–1.3 of 10 per seed under the models here), P(arm C empty) (25–98 % under independent candidate draws; the lower end is ≈ 33 % once the instance pools shared across candidates are modelled, §4.6 [corrected after independent verification 2026-10-13]), and pre-declare the analysis for the degenerate case C = A (H2 collapses to A vs B; H1 = 0 by construction) alongside the already-declared B ≈ C finding (§6.5). This fills, rather than changes, locked fields. | false (filling pre-registered fields, but must be done *before* run start and reviewed) | P1 | owner + critic |
| R6 | **Fix the float boundary in `regression.py` / `gate.py` sensitivity**: compare integer failure counts (`failures <= floor(tau_r * n)` computed exactly, e.g. with `fractions.Fraction`) so the secondary τ_r = 0.05 cell tolerates 2 failures as ADR-0002 intends; add a boundary test for 1/40, 2/40 and for 1/20 vs τ_c. Primary verdicts are unaffected. | false | P1 | builder |
| R7 | **State the integer forms of every threshold in the manifest** (replay 3/5 & +2/5 already is; add "stage 2 ⇒ ≥ 4 net rescues, 0 breaks at n = 20", "out-of-scope ≥ −0.05 ⇒ net ≥ 0 at n = 10", "regression ⇒ ≤ 1 failure", "interaction ⇒ net ≤ −1") so reviewers see what is actually tested; or change n where the stated tolerance is void. A clarification is still an amendment of a locked artifact. | **true** | P2 | builder + critic |
| R8 | **Re-run this script with measured inputs** once G0's T_mine run exists: `--p0 <measured G0 rate>` and, if R3/R4 give a measured f, `--f <measured>`; attach the output to the pre-registration before the expected-result fields are frozen. No locked rule changes. | false | P2 | any |
| R9 | **Reconcile the pre-registered +5 pp H2 threshold with the MDE** (realized 7–13 pp at f_arm = 0.1; ≥ 20 pp at f_arm = 0.3). T_eval size is fixed by ALFWorld's unseen split, so the only levers are more mining seeds (effective only if ρ < 1) or an honest pre-registered expectation that H2 is adequately powered only for ≥ +10–15 pp effects. Either is an ADR matter. | **true** | P1 | critic (owner decides) |
| R10 | Record in the CIG stage-2 child record the (a, b) composition and `p_resample≤0` next to the LB, so RNG-boundary cells (6,1), (8,2), (10,3), (12,4), (13,5) [corrected after independent verification 2026-10-13] are visible in post-run review; pure logging. | false | P2 | builder |

## 7. Limitations of this analysis

* The effect models are assumptions (A1–A11); the base rate p0 and the perturbation rate f are unmeasured, and the noise model's shape (symmetric vs mean-neutral) is unknown. Results should be read as *where the locked rules are informative*, not as predictions of the run's outcome.
* Family heterogeneity of p0, within-task seed correlation structure, and arm-B prompt-length effects are modelled only through grids/bounds (ρ, f_arm).
* Stage probabilities use exact combinatorics; H1/H2 use 1 000 reps × 1 000 resamples (MC s.e. ≤ 1.6 pp; 1k resamples slightly optimistic vs the locked 10k, T6b).
* The stage-2 pass table is validated against the repository's bootstrap; stage 3/4/interaction implementations were read on the PR #28 branch at head `7886852`. That branch has since merged as main `6014bc7`, and the merge changes no rule the simulation models (see the provenance note at the top). [corrected after independent verification 2026-10-13]

---

## Appendix A — full output tables, symmetric noise model (`exp0001_power_sim_results/tables.md`, seed 20261012)

#### T0. Stage-2 (ablation, n=20) exact pass probability by observed (rescues a, breaks b)

P(resample mean <= 0) is exact (trinomial); P(pass) = P(Bin(10000, p) <= 250) — i.e. the seeded 10k bootstrap's LB is > 0 only if fewer than 251 of the 10k resample means are <= 0.

| a (+1) | b (-1) | point | P(resample<=0) | P(stage-2 pass) |
|---|---|---|---|---|
| 1 | 0 | +0.05 | 0.3585 | 0.0% |
| 2 | 0 | +0.10 | 0.1216 | 0.0% |
| 2 | 1 | +0.05 | 0.3892 | 0.0% |
| 3 | 0 | +0.15 | 0.0388 | 0.0% |
| 3 | 1 | +0.10 | 0.2185 | 0.0% |
| 3 | 2 | +0.05 | 0.4114 | 0.0% |
| 4 | 0 | +0.20 | 0.0115 | 100.0% |
| 4 | 1 | +0.15 | 0.1162 | 0.0% |
| 4 | 2 | +0.10 | 0.2654 | 0.0% |
| 4 | 3 | +0.05 | 0.4243 | 0.0% |
| 5 | 0 | +0.25 | 0.0032 | 100.0% |
| 5 | 1 | +0.20 | 0.0588 | 0.0% |
| 5 | 2 | +0.15 | 0.1628 | 0.0% |
| 5 | 3 | +0.10 | 0.2944 | 0.0% |
| 5 | 4 | +0.05 | 0.4328 | 0.0% |
| 6 | 0 | +0.30 | 0.0008 | 100.0% |
| 6 | 1 | +0.25 | 0.0284 | 2.1% |
| 6 | 2 | +0.20 | 0.0955 | 0.0% |
| 6 | 3 | +0.15 | 0.1955 | 0.0% |
| 6 | 4 | +0.10 | 0.3147 | 0.0% |
| 7 | 0 | +0.35 | 0.0002 | 100.0% |
| 7 | 1 | +0.30 | 0.0131 | 100.0% |
| 7 | 2 | +0.25 | 0.0536 | 0.0% |
| 7 | 3 | +0.20 | 0.1246 | 0.0% |
| 7 | 4 | +0.15 | 0.2201 | 0.0% |
| 8 | 0 | +0.40 | 0.0000 | 100.0% |
| 8 | 1 | +0.35 | 0.0057 | 100.0% |
| 8 | 2 | +0.30 | 0.0289 | 0.9% |
| 8 | 3 | +0.25 | 0.0765 | 0.0% |
| 8 | 4 | +0.20 | 0.1484 | 0.0% |
| 9 | 0 | +0.45 | 0.0000 | 100.0% |
| 9 | 1 | +0.40 | 0.0024 | 100.0% |
| 9 | 2 | +0.35 | 0.0150 | 100.0% |
| 9 | 3 | +0.30 | 0.0452 | 0.0% |
| 9 | 4 | +0.25 | 0.0967 | 0.0% |
| 10 | 0 | +0.50 | 0.0000 | 100.0% |
| 10 | 1 | +0.45 | 0.0009 | 100.0% |
| 10 | 2 | +0.40 | 0.0074 | 100.0% |
| 10 | 3 | +0.35 | 0.0258 | 31.9% |
| 10 | 4 | +0.30 | 0.0609 | 0.0% |

Smallest a with b=0 giving P(pass)>50%: **a=4** (observed contribution >= +0.20).


#### T0b. Secondary tau_c grid at n=20: integer net required by the point rule vs by LB>0

| tau_c | net needed by point>=tau_c | net needed by LB>0 (b=0) |
|---|---|---|
| 0.03 | 1 | 4 |
| 0.05 | 1 | 4 |
| 0.08 | 2 | 4 |

#### T0c. Secondary tau_r grid at n=40: failures tolerated

| tau_r | max T_reg failures allowed (of 40) |
|---|---|
| 0.02 | 0 |
| 0.03 | 1 |
| 0.05 | 2 |

#### T0v. Validation of the exact table against the repository bootstrap (verbatim inheritance/bootstrap.py, seed 0, 10k resamples, 25 random task orderings per cell)

| a | b | p_resample<=0 | predicted P(pass) | repo bootstrap pass count |
|---|---|---|---|---|
| 3 | 0 | 0.0388 | 0.0% | 0/25 |
| 4 | 0 | 0.0115 | 100.0% | 25/25 |
| 5 | 1 | 0.0588 | 0.0% | 0/25 |
| 6 | 1 | 0.0284 | 2.1% | 1/25 |
| 6 | 2 | 0.0955 | 0.0% | 0/25 |
| 7 | 2 | 0.0536 | 0.0% | 0/25 |
| 8 | 3 | 0.0765 | 0.0% | 0/25 |

#### T1. Stage operating characteristics (exact; noise model = symmetric; expression prob = 1; source instance treated like the variants)

`E[dlt]` = realized mean with-minus-without effect including the noise bias; `out(gen)` = trait effect also applies out of scope; `out(spec)` = trait is exactly neutral out of scope; `all` = product of the five stage pass probabilities (stages use disjoint instances) before the interaction stage.

| p0 | delta | f | E[dlt] | replay | ablation | gen-in | out(gen) | out(spec) | h_reg | regression | all(gen) | all(spec) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.3 | -0.10 | 0.0 | -0.100 | 0.0% | 0.0% | 34.9% | 34.9% | 100.0% | 0.333 | 0.0% | 0.0% | 0.0% |
| 0.3 | -0.10 | 0.05 | -0.070 | 0.5% | 0.1% | 45.4% | 45.4% | 89.6% | 0.350 | 0.0% | 0.0% | 0.0% |
| 0.3 | -0.10 | 0.1 | -0.040 | 1.8% | 0.7% | 54.1% | 54.1% | 85.3% | 0.367 | 0.0% | 0.0% | 0.0% |
| 0.3 | -0.10 | 0.2 | +0.020 | 6.7% | 3.6% | 67.2% | 67.2% | 83.7% | 0.400 | 0.0% | 0.0% | 0.0% |
| 0.3 | -0.05 | 0.0 | -0.050 | 0.0% | 0.0% | 59.9% | 59.9% | 100.0% | 0.167 | 0.6% | 0.0% | 0.0% |
| 0.3 | -0.05 | 0.05 | -0.025 | 0.6% | 0.2% | 64.1% | 64.1% | 89.6% | 0.200 | 0.1% | 0.0% | 0.0% |
| 0.3 | -0.05 | 0.1 | -0.000 | 2.2% | 1.4% | 68.3% | 68.3% | 85.3% | 0.233 | 0.0% | 0.0% | 0.0% |
| 0.3 | -0.05 | 0.2 | +0.050 | 7.7% | 6.2% | 75.3% | 75.3% | 83.7% | 0.300 | 0.0% | 0.0% | 0.0% |
| 0.3 | +0.00 | 0.0 | +0.000 | 0.0% | 0.0% | 100.0% | 100.0% | 100.0% | 0.000 | 100.0% | 0.0% | 0.0% |
| 0.3 | +0.00 | 0.05 | +0.020 | 0.7% | 0.4% | 89.6% | 89.6% | 89.6% | 0.050 | 39.9% | 0.0% | 0.0% |
| 0.3 | +0.00 | 0.1 | +0.040 | 2.7% | 2.8% | 85.3% | 85.3% | 85.3% | 0.100 | 8.0% | 0.0% | 0.0% |
| 0.3 | +0.00 | 0.2 | +0.080 | 8.8% | 10.7% | 83.7% | 83.7% | 83.7% | 0.200 | 0.1% | 0.0% | 0.0% |
| 0.3 | +0.05 | 0.0 | +0.050 | 1.6% | 1.6% | 100.0% | 100.0% | 100.0% | 0.000 | 100.0% | 0.0% | 0.0% |
| 0.3 | +0.05 | 0.05 | +0.065 | 3.7% | 5.5% | 93.1% | 93.1% | 89.6% | 0.050 | 39.9% | 0.1% | 0.1% |
| 0.3 | +0.05 | 0.1 | +0.080 | 6.3% | 10.2% | 89.6% | 89.6% | 85.3% | 0.100 | 8.0% | 0.0% | 0.0% |
| 0.3 | +0.05 | 0.2 | +0.110 | 12.6% | 16.2% | 87.1% | 87.1% | 83.7% | 0.200 | 0.1% | 0.0% | 0.0% |
| 0.3 | +0.10 | 0.0 | +0.100 | 6.0% | 13.3% | 100.0% | 100.0% | 100.0% | 0.000 | 100.0% | 0.8% | 0.8% |
| 0.3 | +0.10 | 0.05 | +0.110 | 8.6% | 18.1% | 95.4% | 95.4% | 89.6% | 0.050 | 39.9% | 0.6% | 0.5% |
| 0.3 | +0.10 | 0.1 | +0.120 | 11.3% | 21.1% | 92.7% | 92.7% | 85.3% | 0.100 | 8.0% | 0.2% | 0.2% |
| 0.3 | +0.10 | 0.2 | +0.140 | 17.0% | 22.2% | 89.8% | 89.8% | 83.7% | 0.200 | 0.1% | 0.0% | 0.0% |
| 0.3 | +0.20 | 0.0 | +0.200 | 21.3% | 58.9% | 100.0% | 100.0% | 100.0% | 0.000 | 100.0% | 12.5% | 12.5% |
| 0.3 | +0.20 | 0.05 | +0.200 | 22.8% | 51.4% | 98.1% | 98.1% | 89.6% | 0.050 | 39.9% | 4.5% | 4.1% |
| 0.3 | +0.20 | 0.1 | +0.200 | 24.3% | 44.9% | 96.5% | 96.5% | 85.3% | 0.100 | 8.0% | 0.8% | 0.7% |
| 0.3 | +0.20 | 0.2 | +0.200 | 26.9% | 35.0% | 93.8% | 93.8% | 83.7% | 0.200 | 0.1% | 0.0% | 0.0% |
| 0.3 | +0.30 | 0.0 | +0.300 | 41.4% | 89.3% | 100.0% | 100.0% | 100.0% | 0.000 | 100.0% | 37.0% | 37.0% |
| 0.3 | +0.30 | 0.05 | +0.290 | 40.5% | 75.8% | 99.3% | 99.3% | 89.6% | 0.050 | 39.9% | 12.1% | 10.9% |
| 0.3 | +0.30 | 0.1 | +0.280 | 39.6% | 64.8% | 98.4% | 98.4% | 85.3% | 0.100 | 8.0% | 2.0% | 1.7% |
| 0.3 | +0.30 | 0.2 | +0.260 | 37.7% | 48.4% | 96.3% | 96.3% | 83.7% | 0.200 | 0.1% | 0.0% | 0.0% |
| 0.3 | +0.40 | 0.0 | +0.400 | 62.0% | 98.4% | 100.0% | 100.0% | 100.0% | 0.000 | 100.0% | 61.0% | 61.0% |
| 0.3 | +0.40 | 0.05 | +0.380 | 58.5% | 89.4% | 99.8% | 99.8% | 89.6% | 0.050 | 39.9% | 20.8% | 18.7% |
| 0.3 | +0.40 | 0.1 | +0.360 | 55.1% | 79.9% | 99.3% | 99.3% | 85.3% | 0.100 | 8.0% | 3.5% | 3.0% |
| 0.3 | +0.40 | 0.2 | +0.320 | 48.8% | 61.5% | 97.8% | 97.8% | 83.7% | 0.200 | 0.1% | 0.0% | 0.0% |
| 0.5 | -0.10 | 0.0 | -0.100 | 0.0% | 0.0% | 34.9% | 34.9% | 100.0% | 0.200 | 0.1% | 0.0% | 0.0% |
| 0.5 | -0.10 | 0.05 | -0.090 | 0.3% | 0.0% | 38.6% | 38.6% | 81.8% | 0.230 | 0.0% | 0.0% | 0.0% |
| 0.5 | -0.10 | 0.1 | -0.080 | 1.3% | 0.2% | 42.1% | 42.1% | 72.5% | 0.260 | 0.0% | 0.0% | 0.0% |
| 0.5 | -0.10 | 0.2 | -0.060 | 4.2% | 0.7% | 47.8% | 47.8% | 64.7% | 0.320 | 0.0% | 0.0% | 0.0% |
| 0.5 | -0.05 | 0.0 | -0.050 | 0.0% | 0.0% | 59.9% | 59.9% | 100.0% | 0.100 | 8.0% | 0.0% | 0.0% |
| 0.5 | -0.05 | 0.05 | -0.045 | 0.4% | 0.0% | 56.5% | 56.5% | 81.8% | 0.140 | 1.8% | 0.0% | 0.0% |
| 0.5 | -0.05 | 0.1 | -0.040 | 1.5% | 0.3% | 55.5% | 55.5% | 72.5% | 0.180 | 0.3% | 0.0% | 0.0% |
| 0.5 | -0.05 | 0.2 | -0.030 | 4.8% | 1.3% | 55.8% | 55.8% | 64.7% | 0.260 | 0.0% | 0.0% | 0.0% |
| 0.5 | +0.00 | 0.0 | +0.000 | 0.0% | 0.0% | 100.0% | 100.0% | 100.0% | 0.000 | 100.0% | 0.0% | 0.0% |
| 0.5 | +0.00 | 0.05 | +0.000 | 0.5% | 0.1% | 81.8% | 81.8% | 81.8% | 0.050 | 39.9% | 0.0% | 0.0% |
| 0.5 | +0.00 | 0.1 | +0.000 | 1.7% | 0.7% | 72.5% | 72.5% | 72.5% | 0.100 | 8.0% | 0.0% | 0.0% |
| 0.5 | +0.00 | 0.2 | +0.000 | 5.3% | 2.2% | 64.7% | 64.7% | 64.7% | 0.200 | 0.1% | 0.0% | 0.0% |
| 0.5 | +0.05 | 0.0 | +0.050 | 2.0% | 1.6% | 100.0% | 100.0% | 100.0% | 0.000 | 100.0% | 0.0% | 0.0% |
| 0.5 | +0.05 | 0.05 | +0.045 | 3.6% | 3.1% | 87.6% | 87.6% | 81.8% | 0.050 | 39.9% | 0.0% | 0.0% |
| 0.5 | +0.05 | 0.1 | +0.040 | 5.2% | 4.2% | 79.8% | 79.8% | 72.5% | 0.100 | 8.0% | 0.0% | 0.0% |
| 0.5 | +0.05 | 0.2 | +0.030 | 8.6% | 4.3% | 70.8% | 70.8% | 64.7% | 0.200 | 0.1% | 0.0% | 0.0% |
| 0.5 | +0.10 | 0.0 | +0.100 | 7.5% | 13.3% | 100.0% | 100.0% | 100.0% | 0.000 | 100.0% | 1.0% | 1.0% |
| 0.5 | +0.10 | 0.05 | +0.090 | 8.9% | 12.4% | 91.7% | 91.7% | 81.8% | 0.050 | 39.9% | 0.4% | 0.3% |
| 0.5 | +0.10 | 0.1 | +0.080 | 10.2% | 10.8% | 85.3% | 85.3% | 72.5% | 0.100 | 8.0% | 0.1% | 0.1% |
| 0.5 | +0.10 | 0.2 | +0.060 | 12.5% | 7.1% | 75.9% | 75.9% | 64.7% | 0.200 | 0.1% | 0.0% | 0.0% |
| 0.5 | +0.20 | 0.0 | +0.200 | 25.2% | 58.9% | 100.0% | 100.0% | 100.0% | 0.000 | 100.0% | 14.8% | 14.8% |
| 0.5 | +0.20 | 0.05 | +0.180 | 24.3% | 40.8% | 96.4% | 96.4% | 81.8% | 0.050 | 39.9% | 3.7% | 3.1% |
| 0.5 | +0.20 | 0.1 | +0.160 | 23.4% | 28.5% | 92.4% | 92.4% | 72.5% | 0.100 | 8.0% | 0.5% | 0.4% |
| 0.5 | +0.20 | 0.2 | +0.120 | 21.5% | 14.1% | 83.9% | 83.9% | 64.7% | 0.200 | 0.1% | 0.0% | 0.0% |
| 0.5 | +0.30 | 0.0 | +0.300 | 46.5% | 89.3% | 100.0% | 100.0% | 100.0% | 0.000 | 100.0% | 41.5% | 41.5% |
| 0.5 | +0.30 | 0.05 | +0.270 | 42.3% | 65.3% | 98.5% | 98.5% | 81.8% | 0.050 | 39.9% | 10.7% | 8.9% |
| 0.5 | +0.30 | 0.1 | +0.240 | 38.3% | 47.1% | 96.2% | 96.2% | 72.5% | 0.100 | 8.0% | 1.3% | 1.0% |
| 0.5 | +0.30 | 0.2 | +0.180 | 31.4% | 23.6% | 89.4% | 89.4% | 64.7% | 0.200 | 0.1% | 0.0% | 0.0% |
| 0.5 | +0.40 | 0.0 | +0.400 | 66.1% | 98.4% | 100.0% | 100.0% | 100.0% | 0.000 | 100.0% | 65.1% | 65.1% |
| 0.5 | +0.40 | 0.05 | +0.360 | 59.4% | 82.1% | 99.5% | 99.5% | 81.8% | 0.050 | 39.9% | 19.2% | 15.8% |
| 0.5 | +0.40 | 0.1 | +0.320 | 53.0% | 64.8% | 98.2% | 98.2% | 72.5% | 0.100 | 8.0% | 2.7% | 2.0% |
| 0.5 | +0.40 | 0.2 | +0.240 | 41.4% | 35.2% | 93.1% | 93.1% | 64.7% | 0.200 | 0.1% | 0.0% | 0.0% |

#### T1b. Stage-1 replay sensitivities: expression probability e per successful episode, and 'source instance already solved by G0' (Y0=1 fixed on the source)

| p0 | delta | f | e=1, src~variants | e=0.8 | src solved | e=0.8 & src solved |
|---|---|---|---|---|---|---|
| 0.3 | +0.10 | 0.0 | 6.0% | 2.8% | 5.2% | 2.3% |
| 0.3 | +0.10 | 0.1 | 11.3% | 5.3% | 9.4% | 4.1% |
| 0.3 | +0.20 | 0.0 | 21.3% | 9.7% | 18.1% | 7.8% |
| 0.3 | +0.20 | 0.1 | 24.3% | 11.0% | 19.7% | 8.4% |
| 0.3 | +0.40 | 0.0 | 62.0% | 26.1% | 52.5% | 20.8% |
| 0.3 | +0.40 | 0.1 | 55.1% | 23.6% | 44.3% | 17.9% |
| 0.5 | +0.10 | 0.0 | 7.5% | 3.2% | 5.2% | 2.1% |
| 0.5 | +0.10 | 0.1 | 10.2% | 4.4% | 6.9% | 2.8% |
| 0.5 | +0.20 | 0.0 | 25.2% | 10.3% | 18.1% | 7.0% |
| 0.5 | +0.20 | 0.1 | 23.4% | 9.7% | 16.1% | 6.3% |
| 0.5 | +0.40 | 0.0 | 66.1% | 23.8% | 52.5% | 18.4% |
| 0.5 | +0.40 | 0.1 | 53.0% | 19.9% | 39.4% | 14.3% |

#### T1c. Stage-4 regression pass probability vs per-task T_reg failure probability h_reg (exact binomial, 40 tasks, <=1 failure allowed)

| h_reg | P(pass) tau_r=0.03 | tau_r=0.02 (0 fails) | tau_r=0.05 (<=2 fails) |
|---|---|---|---|
| 0.0 | 100.0% | 100.0% | 100.0% |
| 0.01 | 93.9% | 66.9% | 99.3% |
| 0.02 | 81.0% | 44.6% | 95.4% |
| 0.03 | 66.2% | 29.6% | 88.2% |
| 0.05 | 39.9% | 12.9% | 67.7% |
| 0.075 | 18.8% | 4.4% | 41.4% |
| 0.1 | 8.0% | 1.5% | 22.3% |
| 0.15 | 1.2% | 0.2% | 4.9% |
| 0.2 | 0.1% | 0.0% | 0.8% |

#### T1d. Mapping of noise rate f to h_reg for a neutral trait under the two noise models

| f | symmetric: h_reg | P(pass) | redraw p0=0.3: h_reg | P(pass) | redraw p0=0.5: h_reg | P(pass) |
|---|---|---|---|---|---|---|
| 0.0 | 0.000 | 100.0% | 0.000 | 100.0% | 0.000 | 100.0% |
| 0.05 | 0.050 | 39.9% | 0.035 | 58.9% | 0.025 | 73.6% |
| 0.1 | 0.100 | 8.0% | 0.070 | 22.0% | 0.050 | 39.9% |
| 0.2 | 0.200 | 0.1% | 0.140 | 1.8% | 0.100 | 8.0% |

#### T1e. Specialist traits: replay/in-scope see the in-family effect, the stratified 20-task ablation pool sees share x in-family effect

| p0 | in-family delta | family share of pool | pool delta | f | replay | ablation | gen-in |
|---|---|---|---|---|---|---|---|
| 0.3 | +0.30 | 0.17 | +0.050 | 0.0 | 41.4% | 1.6% | 100.0% |
| 0.3 | +0.30 | 0.17 | +0.050 | 0.05 | 40.5% | 5.5% | 99.3% |
| 0.3 | +0.30 | 0.33 | +0.100 | 0.0 | 41.4% | 13.3% | 100.0% |
| 0.3 | +0.30 | 0.33 | +0.100 | 0.05 | 40.5% | 18.1% | 99.3% |
| 0.3 | +0.30 | 1.00 | +0.300 | 0.0 | 41.4% | 89.3% | 100.0% |
| 0.3 | +0.30 | 1.00 | +0.300 | 0.05 | 40.5% | 75.8% | 99.3% |
| 0.3 | +0.50 | 0.17 | +0.083 | 0.0 | 79.2% | 8.0% | 100.0% |
| 0.3 | +0.50 | 0.17 | +0.083 | 0.05 | 74.2% | 13.2% | 99.9% |
| 0.3 | +0.50 | 0.33 | +0.167 | 0.0 | 79.2% | 43.3% | 100.0% |
| 0.3 | +0.50 | 0.33 | +0.167 | 0.05 | 74.2% | 40.6% | 99.9% |
| 0.3 | +0.50 | 1.00 | +0.500 | 0.0 | 79.2% | 99.9% | 100.0% |
| 0.3 | +0.50 | 1.00 | +0.500 | 0.05 | 74.2% | 96.1% | 99.9% |
| 0.3 | +0.80 | 0.17 | +0.133 | 0.0 | 96.9% | 27.3% | 100.0% |
| 0.3 | +0.80 | 0.17 | +0.133 | 0.05 | 93.2% | 29.2% | 100.0% |
| 0.3 | +0.80 | 0.33 | +0.267 | 0.0 | 96.9% | 82.2% | 100.0% |
| 0.3 | +0.80 | 0.33 | +0.267 | 0.05 | 93.2% | 69.1% | 100.0% |
| 0.3 | +0.80 | 1.00 | +0.800 | 0.0 | 96.9% | 100.0% | 100.0% |
| 0.3 | +0.80 | 1.00 | +0.800 | 0.05 | 93.2% | 99.8% | 100.0% |
| 0.5 | +0.30 | 0.17 | +0.050 | 0.0 | 46.5% | 1.6% | 100.0% |
| 0.5 | +0.30 | 0.17 | +0.050 | 0.05 | 42.3% | 3.1% | 98.5% |
| 0.5 | +0.30 | 0.33 | +0.100 | 0.0 | 46.5% | 13.3% | 100.0% |
| 0.5 | +0.30 | 0.33 | +0.100 | 0.05 | 42.3% | 12.4% | 98.5% |
| 0.5 | +0.30 | 1.00 | +0.300 | 0.0 | 46.5% | 89.3% | 100.0% |
| 0.5 | +0.30 | 1.00 | +0.300 | 0.05 | 42.3% | 65.3% | 98.5% |
| 0.5 | +0.50 | 0.17 | +0.083 | 0.0 | 81.2% | 8.0% | 100.0% |
| 0.5 | +0.50 | 0.17 | +0.083 | 0.05 | 73.5% | 8.6% | 99.8% |
| 0.5 | +0.50 | 0.33 | +0.167 | 0.0 | 81.2% | 43.3% | 100.0% |
| 0.5 | +0.50 | 0.33 | +0.167 | 0.05 | 73.5% | 31.1% | 99.8% |
| 0.5 | +0.50 | 1.00 | +0.500 | 0.0 | 81.2% | 99.9% | 100.0% |
| 0.5 | +0.50 | 1.00 | +0.500 | 0.05 | 73.5% | 92.3% | 99.8% |
| 0.5 | +0.80 | 0.17 | +0.133 | 0.0 | 81.2% | 27.3% | 100.0% |
| 0.5 | +0.80 | 0.17 | +0.133 | 0.05 | 73.5% | 21.4% | 99.8% |
| 0.5 | +0.80 | 0.33 | +0.267 | 0.0 | 81.2% | 82.2% | 100.0% |
| 0.5 | +0.80 | 0.33 | +0.267 | 0.05 | 73.5% | 58.1% | 99.8% |
| 0.5 | +0.80 | 1.00 | +0.800 | 0.0 | 81.2% | 99.9% | 100.0% |
| 0.5 | +0.80 | 1.00 | +0.800 | 0.05 | 73.5% | 92.3% | 99.8% |

#### T2. End-to-end: K=10 candidates per seed, mixture helpful/neutral/harmful, harmful delta = -0.10, 3000 MC reps, incl. A1 interaction stage

| p0 | mix H/N/Harm | delta_help | f | E[eligible] | E[elim by interaction] | E[admitted] | P(arm C empty) | E[helpful adm] | E[neutral adm] | E[harmful adm] | P(any non-helpful adm) | precision | nonempty |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.3 | 20/50/30 | +0.10 | 0.0 | 0.01 | 0.00 | 0.01 | 98.7% | 0.01 | 0.000 | 0.000 | 0.0% | 100.0% |
| 0.3 | 20/50/30 | +0.10 | 0.05 | 0.01 | 0.00 | 0.01 | 98.8% | 0.01 | 0.000 | 0.000 | 0.0% | 100.0% |
| 0.3 | 20/50/30 | +0.10 | 0.1 | 0.00 | 0.00 | 0.00 | 99.7% | 0.00 | 0.000 | 0.000 | 0.0% | 100.0% |
| 0.3 | 20/50/30 | +0.10 | 0.2 | 0.00 | 0.00 | 0.00 | 100.0% | 0.00 | 0.000 | 0.000 | 0.0% | n/a |
| 0.3 | 20/50/30 | +0.20 | 0.0 | 0.25 | 0.00 | 0.25 | 77.3% | 0.25 | 0.000 | 0.000 | 0.0% | 100.0% |
| 0.3 | 20/50/30 | +0.20 | 0.05 | 0.10 | 0.00 | 0.10 | 90.9% | 0.10 | 0.000 | 0.000 | 0.0% | 100.0% |
| 0.3 | 20/50/30 | +0.20 | 0.1 | 0.01 | 0.00 | 0.01 | 98.6% | 0.01 | 0.000 | 0.000 | 0.0% | 100.0% |
| 0.3 | 20/50/30 | +0.20 | 0.2 | 0.00 | 0.00 | 0.00 | 100.0% | 0.00 | 0.000 | 0.000 | 0.0% | 100.0% |
| 0.3 | 20/50/30 | +0.40 | 0.0 | 1.24 | 0.00 | 1.24 | 27.1% | 1.24 | 0.000 | 0.000 | 0.0% | 100.0% |
| 0.3 | 20/50/30 | +0.40 | 0.05 | 0.42 | 0.00 | 0.42 | 64.8% | 0.42 | 0.000 | 0.000 | 0.0% | 100.0% |
| 0.3 | 20/50/30 | +0.40 | 0.1 | 0.07 | 0.00 | 0.07 | 93.5% | 0.07 | 0.000 | 0.000 | 0.0% | 100.0% |
| 0.3 | 20/50/30 | +0.40 | 0.2 | 0.00 | 0.00 | 0.00 | 99.9% | 0.00 | 0.000 | 0.000 | 0.0% | 66.7% |
| 0.3 | 50/30/20 | +0.20 | 0.0 | 0.64 | 0.00 | 0.64 | 51.6% | 0.64 | 0.000 | 0.000 | 0.0% | 100.0% |
| 0.3 | 50/30/20 | +0.20 | 0.05 | 0.22 | 0.00 | 0.22 | 79.7% | 0.22 | 0.000 | 0.000 | 0.0% | 100.0% |
| 0.3 | 10/60/30 | +0.20 | 0.0 | 0.12 | 0.00 | 0.12 | 88.4% | 0.12 | 0.000 | 0.000 | 0.0% | 100.0% |
| 0.3 | 10/60/30 | +0.20 | 0.05 | 0.05 | 0.00 | 0.05 | 95.4% | 0.05 | 0.000 | 0.000 | 0.0% | 100.0% |
| 0.5 | 20/50/30 | +0.10 | 0.0 | 0.02 | 0.00 | 0.02 | 98.0% | 0.02 | 0.000 | 0.000 | 0.0% | 100.0% |
| 0.5 | 20/50/30 | +0.10 | 0.05 | 0.01 | 0.00 | 0.01 | 99.3% | 0.01 | 0.000 | 0.000 | 0.0% | 100.0% |
| 0.5 | 20/50/30 | +0.10 | 0.1 | 0.00 | 0.00 | 0.00 | 99.9% | 0.00 | 0.000 | 0.000 | 0.0% | 100.0% |
| 0.5 | 20/50/30 | +0.10 | 0.2 | 0.00 | 0.00 | 0.00 | 100.0% | 0.00 | 0.000 | 0.000 | 0.0% | n/a |
| 0.5 | 20/50/30 | +0.20 | 0.0 | 0.30 | 0.00 | 0.30 | 73.9% | 0.30 | 0.000 | 0.000 | 0.0% | 100.0% |
| 0.5 | 20/50/30 | +0.20 | 0.05 | 0.07 | 0.00 | 0.07 | 92.7% | 0.07 | 0.000 | 0.000 | 0.0% | 100.0% |
| 0.5 | 20/50/30 | +0.20 | 0.1 | 0.01 | 0.00 | 0.01 | 99.1% | 0.01 | 0.000 | 0.000 | 0.0% | 100.0% |
| 0.5 | 20/50/30 | +0.20 | 0.2 | 0.00 | 0.00 | 0.00 | 100.0% | 0.00 | 0.000 | 0.000 | 0.0% | n/a |
| 0.5 | 20/50/30 | +0.40 | 0.0 | 1.30 | 0.00 | 1.30 | 24.5% | 1.30 | 0.000 | 0.000 | 0.0% | 100.0% |
| 0.5 | 20/50/30 | +0.40 | 0.05 | 0.41 | 0.00 | 0.41 | 65.1% | 0.41 | 0.000 | 0.000 | 0.0% | 100.0% |
| 0.5 | 20/50/30 | +0.40 | 0.1 | 0.05 | 0.00 | 0.05 | 94.8% | 0.05 | 0.000 | 0.000 | 0.0% | 99.4% |
| 0.5 | 20/50/30 | +0.40 | 0.2 | 0.00 | 0.00 | 0.00 | 100.0% | 0.00 | 0.000 | 0.000 | 0.0% | 100.0% |
| 0.5 | 50/30/20 | +0.20 | 0.0 | 0.74 | 0.00 | 0.74 | 47.0% | 0.74 | 0.000 | 0.000 | 0.0% | 100.0% |
| 0.5 | 50/30/20 | +0.20 | 0.05 | 0.18 | 0.00 | 0.18 | 83.5% | 0.18 | 0.000 | 0.000 | 0.0% | 100.0% |
| 0.5 | 10/60/30 | +0.20 | 0.0 | 0.15 | 0.00 | 0.15 | 85.3% | 0.15 | 0.000 | 0.000 | 0.0% | 100.0% |
| 0.5 | 10/60/30 | +0.20 | 0.05 | 0.04 | 0.00 | 0.04 | 96.5% | 0.04 | 0.000 | 0.000 | 0.0% | 100.0% |

#### T3. H2 T_eval component: P[point(C-B) >= +5pp AND LB>0 AND direction in >=2/3 seeds]; n=134 paired tasks x 3 seeds, cluster = task (seed-averaged), p0=0.4 (UNVERIFIED), delta_B=0, f_B=f_C=f_arm; rho = share of tasks whose flips are identical across seeds (1 = seeds add no information); 1000 reps x 1000 resamples

| rho | f_arm | true C-B | E[obs C-B] | sd(obs) | P(point>=5pp) | P(LB>0) | P(dir 2/3) | P(T_eval part) |
|---|---|---|---|---|---|---|---|---|
| 0.0 | 0.1 | +0.00 | -0.000 | 0.021 | 0.8% | 2.9% | 43.8% | 0.8% |
| 0.0 | 0.1 | +0.02 | +0.017 | 0.023 | 6.3% | 12.1% | 67.4% | 6.3% |
| 0.0 | 0.1 | +0.04 | +0.031 | 0.023 | 18.7% | 28.1% | 85.7% | 18.7% |
| 0.0 | 0.1 | +0.05 | +0.041 | 0.024 | 33.6% | 43.5% | 90.4% | 33.6% |
| 0.0 | 0.1 | +0.06 | +0.048 | 0.023 | 44.8% | 54.9% | 96.2% | 44.7% |
| 0.0 | 0.1 | +0.08 | +0.063 | 0.024 | 69.0% | 73.7% | 98.8% | 68.6% |
| 0.0 | 0.1 | +0.10 | +0.081 | 0.025 | 88.0% | 90.1% | 99.8% | 87.7% |
| 0.0 | 0.1 | +0.12 | +0.096 | 0.026 | 95.7% | 96.4% | 100.0% | 95.6% |
| 0.0 | 0.1 | +0.15 | +0.121 | 0.027 | 99.5% | 99.5% | 100.0% | 99.5% |
| 0.0 | 0.1 | +0.20 | +0.162 | 0.028 | 100.0% | 100.0% | 100.0% | 100.0% |
| 0.0 | 0.2 | +0.00 | -0.000 | 0.029 | 4.1% | 3.1% | 45.2% | 2.8% |
| 0.0 | 0.2 | +0.02 | +0.012 | 0.029 | 8.4% | 6.2% | 60.9% | 6.2% |
| 0.0 | 0.2 | +0.04 | +0.024 | 0.029 | 18.4% | 15.1% | 72.2% | 15.0% |
| 0.0 | 0.2 | +0.05 | +0.030 | 0.030 | 25.1% | 19.9% | 76.1% | 19.6% |
| 0.0 | 0.2 | +0.06 | +0.036 | 0.029 | 29.1% | 23.6% | 82.4% | 23.5% |
| 0.0 | 0.2 | +0.08 | +0.048 | 0.029 | 46.0% | 38.0% | 89.6% | 37.6% |
| 0.0 | 0.2 | +0.10 | +0.061 | 0.029 | 63.8% | 54.0% | 96.9% | 53.9% |
| 0.0 | 0.2 | +0.12 | +0.071 | 0.029 | 74.9% | 67.3% | 98.6% | 67.1% |
| 0.0 | 0.2 | +0.15 | +0.090 | 0.030 | 89.7% | 84.1% | 99.2% | 84.1% |
| 0.0 | 0.2 | +0.20 | +0.120 | 0.032 | 98.2% | 96.4% | 99.9% | 96.4% |
| 0.0 | 0.3 | +0.00 | +0.000 | 0.031 | 5.1% | 2.3% | 47.0% | 2.3% |
| 0.0 | 0.3 | +0.02 | +0.007 | 0.032 | 8.5% | 4.8% | 51.9% | 4.8% |
| 0.0 | 0.3 | +0.04 | +0.017 | 0.034 | 16.3% | 8.1% | 63.3% | 8.1% |
| 0.0 | 0.3 | +0.05 | +0.022 | 0.032 | 18.8% | 10.1% | 68.6% | 10.1% |
| 0.0 | 0.3 | +0.06 | +0.024 | 0.032 | 20.6% | 11.7% | 71.8% | 11.7% |
| 0.0 | 0.3 | +0.08 | +0.033 | 0.033 | 28.9% | 18.2% | 78.8% | 18.2% |
| 0.0 | 0.3 | +0.10 | +0.040 | 0.033 | 35.5% | 24.7% | 83.1% | 24.7% |
| 0.0 | 0.3 | +0.12 | +0.047 | 0.033 | 44.9% | 30.2% | 87.1% | 30.2% |
| 0.0 | 0.3 | +0.15 | +0.061 | 0.033 | 61.7% | 45.4% | 93.2% | 45.4% |
| 0.0 | 0.3 | +0.20 | +0.079 | 0.034 | 80.5% | 65.0% | 96.2% | 64.9% |
| 1.0 | 0.1 | +0.00 | +0.001 | 0.036 | 9.0% | 1.9% | 47.1% | 1.9% |
| 1.0 | 0.1 | +0.02 | +0.016 | 0.038 | 18.3% | 6.1% | 62.8% | 6.1% |
| 1.0 | 0.1 | +0.04 | +0.034 | 0.039 | 36.0% | 12.0% | 77.6% | 12.0% |
| 1.0 | 0.1 | +0.05 | +0.039 | 0.039 | 39.7% | 13.6% | 83.2% | 13.6% |
| 1.0 | 0.1 | +0.06 | +0.046 | 0.040 | 47.5% | 20.3% | 85.4% | 20.3% |
| 1.0 | 0.1 | +0.08 | +0.063 | 0.041 | 63.4% | 31.5% | 93.4% | 31.5% |
| 1.0 | 0.1 | +0.10 | +0.079 | 0.042 | 78.2% | 44.4% | 95.8% | 44.4% |
| 1.0 | 0.1 | +0.12 | +0.095 | 0.042 | 86.1% | 60.3% | 98.0% | 60.3% |
| 1.0 | 0.1 | +0.15 | +0.119 | 0.045 | 94.3% | 74.1% | 99.6% | 74.1% |
| 1.0 | 0.1 | +0.20 | +0.158 | 0.046 | 99.5% | 92.9% | 100.0% | 92.9% |
| 1.0 | 0.2 | +0.00 | -0.001 | 0.050 | 15.4% | 2.4% | 47.1% | 2.4% |
| 1.0 | 0.2 | +0.02 | +0.016 | 0.049 | 24.3% | 4.4% | 59.9% | 4.4% |
| 1.0 | 0.2 | +0.04 | +0.025 | 0.050 | 32.7% | 5.9% | 68.3% | 5.9% |
| 1.0 | 0.2 | +0.05 | +0.032 | 0.051 | 36.2% | 9.6% | 71.9% | 9.6% |
| 1.0 | 0.2 | +0.06 | +0.037 | 0.049 | 42.1% | 9.5% | 74.3% | 9.5% |
| 1.0 | 0.2 | +0.08 | +0.049 | 0.051 | 49.0% | 16.1% | 81.1% | 16.1% |
| 1.0 | 0.2 | +0.10 | +0.060 | 0.051 | 58.1% | 19.4% | 86.9% | 19.4% |
| 1.0 | 0.2 | +0.12 | +0.074 | 0.051 | 66.6% | 29.5% | 90.8% | 29.5% |
| 1.0 | 0.2 | +0.15 | +0.088 | 0.051 | 77.0% | 36.9% | 95.4% | 36.9% |
| 1.0 | 0.2 | +0.20 | +0.120 | 0.052 | 91.8% | 59.2% | 98.6% | 59.2% |
| 1.0 | 0.3 | +0.00 | -0.003 | 0.057 | 18.3% | 1.9% | 45.3% | 1.9% |
| 1.0 | 0.3 | +0.02 | +0.006 | 0.055 | 21.8% | 3.0% | 51.0% | 3.0% |
| 1.0 | 0.3 | +0.04 | +0.018 | 0.058 | 29.2% | 5.4% | 61.9% | 5.4% |
| 1.0 | 0.3 | +0.05 | +0.021 | 0.058 | 31.7% | 5.8% | 60.7% | 5.8% |
| 1.0 | 0.3 | +0.06 | +0.023 | 0.055 | 32.0% | 4.0% | 63.8% | 4.0% |
| 1.0 | 0.3 | +0.08 | +0.033 | 0.054 | 38.3% | 6.8% | 71.1% | 6.8% |
| 1.0 | 0.3 | +0.10 | +0.042 | 0.056 | 46.2% | 9.9% | 75.1% | 9.9% |
| 1.0 | 0.3 | +0.12 | +0.050 | 0.056 | 52.2% | 12.1% | 80.2% | 12.1% |
| 1.0 | 0.3 | +0.15 | +0.064 | 0.055 | 59.8% | 17.8% | 86.1% | 17.8% |
| 1.0 | 0.3 | +0.20 | +0.080 | 0.057 | 69.3% | 27.1% | 90.7% | 27.1% |

#### T3b. Minimum detectable C-B effect at 80% power (linear interpolation on the grid). 'nominal' = model input delta_C; 'realized' = E[observed C-B] on T_eval (= nominal x (1-2 f_arm) under symmetric noise), i.e. the effect that is actually present

| rho | f_arm | MDE nominal: full criterion | MDE nominal: LB>0 & seed only | MDE realized: full criterion | MDE realized: LB>0 & seed only |
|---|---|---|---|---|---|
| 0.0 | 0.1 | 0.092 | 0.088 | 0.074 | 0.070 |
| 0.0 | 0.2 | 0.143 | 0.143 | 0.085 | 0.085 |
| 0.0 | 0.3 | >0.20 | >0.20 | >0.079 | >0.079 |
| 1.0 | 0.1 | 0.166 | 0.166 | 0.131 | 0.131 |
| 1.0 | 0.2 | >0.20 | >0.20 | >0.120 | >0.120 |
| 1.0 | 0.3 | >0.20 | >0.20 | >0.080 | >0.080 |

#### T4. H2 regression component: P[reg(B)-reg(C) > 0 with cluster-bootstrap LB > 0] over the 40 T_reg tasks x 3 seeds (per-task per-seed failure probabilities r_B, r_C, independent arms)

| rho | r_B | r_C | true diff | P(reg part) |
|---|---|---|---|---|
| 0.0 | 0.05 | 0.0 | +0.050 | 85.7% |
| 0.0 | 0.05 | 0.025 | +0.025 | 16.1% |
| 0.0 | 0.1 | 0.0 | +0.100 | 99.5% |
| 0.0 | 0.1 | 0.025 | +0.075 | 67.9% |
| 0.0 | 0.1 | 0.05 | +0.050 | 34.3% |
| 0.0 | 0.15 | 0.0 | +0.150 | 100.0% |
| 0.0 | 0.15 | 0.025 | +0.125 | 96.3% |
| 0.0 | 0.15 | 0.05 | +0.100 | 74.4% |
| 0.0 | 0.2 | 0.0 | +0.200 | 100.0% |
| 0.0 | 0.2 | 0.025 | +0.175 | 99.8% |
| 0.0 | 0.2 | 0.05 | +0.150 | 95.1% |
| 0.0 | 0.3 | 0.0 | +0.300 | 100.0% |
| 0.0 | 0.3 | 0.025 | +0.275 | 100.0% |
| 0.0 | 0.3 | 0.05 | +0.250 | 100.0% |
| 0.0 | 0.4 | 0.0 | +0.400 | 100.0% |
| 0.0 | 0.4 | 0.025 | +0.375 | 100.0% |
| 0.0 | 0.4 | 0.05 | +0.350 | 100.0% |
| 1.0 | 0.05 | 0.0 | +0.050 | 13.5% |
| 1.0 | 0.05 | 0.025 | +0.025 | 4.1% |
| 1.0 | 0.1 | 0.0 | +0.100 | 60.1% |
| 1.0 | 0.1 | 0.025 | +0.075 | 26.3% |
| 1.0 | 0.1 | 0.05 | +0.050 | 12.0% |
| 1.0 | 0.15 | 0.0 | +0.150 | 87.7% |
| 1.0 | 0.15 | 0.025 | +0.125 | 54.1% |
| 1.0 | 0.15 | 0.05 | +0.100 | 29.1% |
| 1.0 | 0.2 | 0.0 | +0.200 | 96.7% |
| 1.0 | 0.2 | 0.025 | +0.175 | 73.0% |
| 1.0 | 0.2 | 0.05 | +0.150 | 53.5% |
| 1.0 | 0.3 | 0.0 | +0.300 | 100.0% |
| 1.0 | 0.3 | 0.025 | +0.275 | 95.5% |
| 1.0 | 0.3 | 0.05 | +0.250 | 87.2% |
| 1.0 | 0.4 | 0.0 | +0.400 | 100.0% |
| 1.0 | 0.4 | 0.025 | +0.375 | 99.7% |
| 1.0 | 0.4 | 0.05 | +0.350 | 97.6% |

#### T4b. Regression component with positive B/C failure dependence (C's skills are a subset of B's), rho=1

| r_B | r_C | P(B fails | C fails) | P(reg part) |
|---|---|---|---|
| 0.1 | 0.025 | 0.0 | 28.8% |
| 0.1 | 0.025 | 0.5 | 34.8% |
| 0.1 | 0.025 | 1.0 | 57.3% |
| 0.15 | 0.05 | 0.0 | 32.1% |
| 0.15 | 0.05 | 0.5 | 46.8% |
| 0.15 | 0.05 | 1.0 | 82.4% |
| 0.2 | 0.05 | 0.0 | 54.8% |
| 0.2 | 0.05 | 0.5 | 69.2% |
| 0.2 | 0.05 | 1.0 | 96.0% |

#### T5. Joint H2 support probability = P(T_eval part) x P(regression part) (disjoint task sets => independent given parameters)

| rho | f_arm | true C-B | r_B | r_C | P(eval) | P(reg) | P(H2 supported) |
|---|---|---|---|---|---|---|---|
| 0.0 | 0.1 | +0.05 | 0.1 | 0.025 | 33.6% | 67.9% | 22.8% |
| 0.0 | 0.1 | +0.05 | 0.2 | 0.05 | 33.6% | 95.1% | 32.0% |
| 0.0 | 0.1 | +0.05 | 0.3 | 0.05 | 33.6% | 100.0% | 33.6% |
| 0.0 | 0.1 | +0.10 | 0.1 | 0.025 | 87.7% | 67.9% | 59.5% |
| 0.0 | 0.1 | +0.10 | 0.2 | 0.05 | 87.7% | 95.1% | 83.4% |
| 0.0 | 0.1 | +0.10 | 0.3 | 0.05 | 87.7% | 100.0% | 87.7% |
| 0.0 | 0.1 | +0.15 | 0.1 | 0.025 | 99.5% | 67.9% | 67.6% |
| 0.0 | 0.1 | +0.15 | 0.2 | 0.05 | 99.5% | 95.1% | 94.6% |
| 0.0 | 0.1 | +0.15 | 0.3 | 0.05 | 99.5% | 100.0% | 99.5% |
| 0.0 | 0.2 | +0.05 | 0.1 | 0.025 | 19.6% | 67.9% | 13.3% |
| 0.0 | 0.2 | +0.05 | 0.2 | 0.05 | 19.6% | 95.1% | 18.6% |
| 0.0 | 0.2 | +0.05 | 0.3 | 0.05 | 19.6% | 100.0% | 19.6% |
| 0.0 | 0.2 | +0.10 | 0.1 | 0.025 | 53.9% | 67.9% | 36.6% |
| 0.0 | 0.2 | +0.10 | 0.2 | 0.05 | 53.9% | 95.1% | 51.3% |
| 0.0 | 0.2 | +0.10 | 0.3 | 0.05 | 53.9% | 100.0% | 53.9% |
| 0.0 | 0.2 | +0.15 | 0.1 | 0.025 | 84.1% | 67.9% | 57.1% |
| 0.0 | 0.2 | +0.15 | 0.2 | 0.05 | 84.1% | 95.1% | 80.0% |
| 0.0 | 0.2 | +0.15 | 0.3 | 0.05 | 84.1% | 100.0% | 84.1% |
| 1.0 | 0.1 | +0.05 | 0.1 | 0.025 | 13.6% | 26.3% | 3.6% |
| 1.0 | 0.1 | +0.05 | 0.2 | 0.05 | 13.6% | 53.5% | 7.3% |
| 1.0 | 0.1 | +0.05 | 0.3 | 0.05 | 13.6% | 87.2% | 11.9% |
| 1.0 | 0.1 | +0.10 | 0.1 | 0.025 | 44.4% | 26.3% | 11.7% |
| 1.0 | 0.1 | +0.10 | 0.2 | 0.05 | 44.4% | 53.5% | 23.8% |
| 1.0 | 0.1 | +0.10 | 0.3 | 0.05 | 44.4% | 87.2% | 38.7% |
| 1.0 | 0.1 | +0.15 | 0.1 | 0.025 | 74.1% | 26.3% | 19.5% |
| 1.0 | 0.1 | +0.15 | 0.2 | 0.05 | 74.1% | 53.5% | 39.6% |
| 1.0 | 0.1 | +0.15 | 0.3 | 0.05 | 74.1% | 87.2% | 64.6% |
| 1.0 | 0.2 | +0.05 | 0.1 | 0.025 | 9.6% | 26.3% | 2.5% |
| 1.0 | 0.2 | +0.05 | 0.2 | 0.05 | 9.6% | 53.5% | 5.1% |
| 1.0 | 0.2 | +0.05 | 0.3 | 0.05 | 9.6% | 87.2% | 8.4% |
| 1.0 | 0.2 | +0.10 | 0.1 | 0.025 | 19.4% | 26.3% | 5.1% |
| 1.0 | 0.2 | +0.10 | 0.2 | 0.05 | 19.4% | 53.5% | 10.4% |
| 1.0 | 0.2 | +0.10 | 0.3 | 0.05 | 19.4% | 87.2% | 16.9% |
| 1.0 | 0.2 | +0.15 | 0.1 | 0.025 | 36.9% | 26.3% | 9.7% |
| 1.0 | 0.2 | +0.15 | 0.2 | 0.05 | 36.9% | 53.5% | 19.7% |
| 1.0 | 0.2 | +0.15 | 0.3 | 0.05 | 36.9% | 87.2% | 32.2% |

#### T6. H1 power: P[(C-A) cluster-bootstrap LB > 0]; arm A identical across seeds (G0 at temp 0), p0=0.4

| rho | f_C | true C-A | E[obs C-A] | P(H1 supported) |
|---|---|---|---|---|
| 0.0 | 0.1 | +0.000 | +0.020 | 20.9% |
| 0.0 | 0.1 | +0.025 | +0.040 | 58.4% |
| 0.0 | 0.1 | +0.050 | +0.060 | 84.4% |
| 0.0 | 0.1 | +0.075 | +0.079 | 94.7% |
| 0.0 | 0.1 | +0.100 | +0.102 | 99.4% |
| 0.0 | 0.1 | +0.150 | +0.140 | 100.0% |
| 0.0 | 0.2 | +0.000 | +0.040 | 33.0% |
| 0.0 | 0.2 | +0.025 | +0.054 | 49.9% |
| 0.0 | 0.2 | +0.050 | +0.070 | 68.3% |
| 0.0 | 0.2 | +0.075 | +0.085 | 82.7% |
| 0.0 | 0.2 | +0.100 | +0.100 | 90.8% |
| 0.0 | 0.2 | +0.150 | +0.130 | 97.2% |
| 0.0 | 0.3 | +0.000 | +0.061 | 42.7% |
| 0.0 | 0.3 | +0.025 | +0.069 | 50.3% |
| 0.0 | 0.3 | +0.050 | +0.083 | 64.6% |
| 0.0 | 0.3 | +0.075 | +0.090 | 67.6% |
| 0.0 | 0.3 | +0.100 | +0.099 | 76.0% |
| 0.0 | 0.3 | +0.150 | +0.119 | 88.8% |
| 1.0 | 0.1 | +0.000 | +0.020 | 10.0% |
| 1.0 | 0.1 | +0.025 | +0.040 | 25.9% |
| 1.0 | 0.1 | +0.050 | +0.059 | 43.6% |
| 1.0 | 0.1 | +0.075 | +0.080 | 65.7% |
| 1.0 | 0.1 | +0.100 | +0.099 | 78.5% |
| 1.0 | 0.1 | +0.150 | +0.140 | 96.0% |
| 1.0 | 0.2 | +0.000 | +0.039 | 15.1% |
| 1.0 | 0.2 | +0.025 | +0.054 | 28.2% |
| 1.0 | 0.2 | +0.050 | +0.069 | 37.1% |
| 1.0 | 0.2 | +0.075 | +0.087 | 52.6% |
| 1.0 | 0.2 | +0.100 | +0.100 | 62.7% |
| 1.0 | 0.2 | +0.150 | +0.131 | 83.2% |
| 1.0 | 0.3 | +0.000 | +0.063 | 25.2% |
| 1.0 | 0.3 | +0.025 | +0.071 | 30.7% |
| 1.0 | 0.3 | +0.050 | +0.080 | 36.5% |
| 1.0 | 0.3 | +0.075 | +0.092 | 43.2% |
| 1.0 | 0.3 | +0.100 | +0.099 | 47.5% |
| 1.0 | 0.3 | +0.150 | +0.120 | 62.3% |

#### T6b. Bootstrap-resolution check (rho=1, f_arm=0.2, 300 reps): P(LB>0) with 1k vs 10k resamples

| true C-B | 1k resamples | 10k resamples |
|---|---|---|
| +0.08 | 16.0% | 12.7% |
| +0.12 | 27.3% | 26.3% |

---

## Appendix B — full output tables, mean-neutral redraw noise model (`exp0001_power_sim_results_redraw/tables.md`, same seed, `--model redraw --skip-validate`)

#### T0. Stage-2 (ablation, n=20) exact pass probability by observed (rescues a, breaks b)

P(resample mean <= 0) is exact (trinomial); P(pass) = P(Bin(10000, p) <= 250) — i.e. the seeded 10k bootstrap's LB is > 0 only if fewer than 251 of the 10k resample means are <= 0.

| a (+1) | b (-1) | point | P(resample<=0) | P(stage-2 pass) |
|---|---|---|---|---|
| 1 | 0 | +0.05 | 0.3585 | 0.0% |
| 2 | 0 | +0.10 | 0.1216 | 0.0% |
| 2 | 1 | +0.05 | 0.3892 | 0.0% |
| 3 | 0 | +0.15 | 0.0388 | 0.0% |
| 3 | 1 | +0.10 | 0.2185 | 0.0% |
| 3 | 2 | +0.05 | 0.4114 | 0.0% |
| 4 | 0 | +0.20 | 0.0115 | 100.0% |
| 4 | 1 | +0.15 | 0.1162 | 0.0% |
| 4 | 2 | +0.10 | 0.2654 | 0.0% |
| 4 | 3 | +0.05 | 0.4243 | 0.0% |
| 5 | 0 | +0.25 | 0.0032 | 100.0% |
| 5 | 1 | +0.20 | 0.0588 | 0.0% |
| 5 | 2 | +0.15 | 0.1628 | 0.0% |
| 5 | 3 | +0.10 | 0.2944 | 0.0% |
| 5 | 4 | +0.05 | 0.4328 | 0.0% |
| 6 | 0 | +0.30 | 0.0008 | 100.0% |
| 6 | 1 | +0.25 | 0.0284 | 2.1% |
| 6 | 2 | +0.20 | 0.0955 | 0.0% |
| 6 | 3 | +0.15 | 0.1955 | 0.0% |
| 6 | 4 | +0.10 | 0.3147 | 0.0% |
| 7 | 0 | +0.35 | 0.0002 | 100.0% |
| 7 | 1 | +0.30 | 0.0131 | 100.0% |
| 7 | 2 | +0.25 | 0.0536 | 0.0% |
| 7 | 3 | +0.20 | 0.1246 | 0.0% |
| 7 | 4 | +0.15 | 0.2201 | 0.0% |
| 8 | 0 | +0.40 | 0.0000 | 100.0% |
| 8 | 1 | +0.35 | 0.0057 | 100.0% |
| 8 | 2 | +0.30 | 0.0289 | 0.9% |
| 8 | 3 | +0.25 | 0.0765 | 0.0% |
| 8 | 4 | +0.20 | 0.1484 | 0.0% |
| 9 | 0 | +0.45 | 0.0000 | 100.0% |
| 9 | 1 | +0.40 | 0.0024 | 100.0% |
| 9 | 2 | +0.35 | 0.0150 | 100.0% |
| 9 | 3 | +0.30 | 0.0452 | 0.0% |
| 9 | 4 | +0.25 | 0.0967 | 0.0% |
| 10 | 0 | +0.50 | 0.0000 | 100.0% |
| 10 | 1 | +0.45 | 0.0009 | 100.0% |
| 10 | 2 | +0.40 | 0.0074 | 100.0% |
| 10 | 3 | +0.35 | 0.0258 | 31.9% |
| 10 | 4 | +0.30 | 0.0609 | 0.0% |

Smallest a with b=0 giving P(pass)>50%: **a=4** (observed contribution >= +0.20).


#### T0b. Secondary tau_c grid at n=20: integer net required by the point rule vs by LB>0

| tau_c | net needed by point>=tau_c | net needed by LB>0 (b=0) |
|---|---|---|
| 0.03 | 1 | 4 |
| 0.05 | 1 | 4 |
| 0.08 | 2 | 4 |

#### T0c. Secondary tau_r grid at n=40: failures tolerated

| tau_r | max T_reg failures allowed (of 40) |
|---|---|
| 0.02 | 0 |
| 0.03 | 1 |
| 0.05 | 2 |

#### T1. Stage operating characteristics (exact; noise model = redraw; expression prob = 1; source instance treated like the variants)

`E[dlt]` = realized mean with-minus-without effect including the noise bias; `out(gen)` = trait effect also applies out of scope; `out(spec)` = trait is exactly neutral out of scope; `all` = product of the five stage pass probabilities (stages use disjoint instances) before the interaction stage.

| p0 | delta | f | E[dlt] | replay | ablation | gen-in | out(gen) | out(spec) | h_reg | regression | all(gen) | all(spec) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.3 | -0.10 | 0.0 | -0.100 | 0.0% | 0.0% | 34.9% | 34.9% | 100.0% | 0.333 | 0.0% | 0.0% | 0.0% |
| 0.3 | -0.10 | 0.05 | -0.100 | 0.0% | 0.0% | 34.9% | 34.9% | 90.9% | 0.357 | 0.0% | 0.0% | 0.0% |
| 0.3 | -0.10 | 0.1 | -0.100 | 0.1% | 0.0% | 35.1% | 35.1% | 84.0% | 0.380 | 0.0% | 0.0% | 0.0% |
| 0.3 | -0.10 | 0.2 | -0.100 | 0.3% | 0.0% | 35.6% | 35.6% | 74.8% | 0.427 | 0.0% | 0.0% | 0.0% |
| 0.3 | -0.05 | 0.0 | -0.050 | 0.0% | 0.0% | 59.9% | 59.9% | 100.0% | 0.167 | 0.6% | 0.0% | 0.0% |
| 0.3 | -0.05 | 0.05 | -0.050 | 0.0% | 0.0% | 57.4% | 57.4% | 90.9% | 0.196 | 0.2% | 0.0% | 0.0% |
| 0.3 | -0.05 | 0.1 | -0.050 | 0.1% | 0.0% | 55.5% | 55.5% | 84.0% | 0.225 | 0.0% | 0.0% | 0.0% |
| 0.3 | -0.05 | 0.2 | -0.050 | 0.5% | 0.1% | 53.0% | 53.0% | 74.8% | 0.283 | 0.0% | 0.0% | 0.0% |
| 0.3 | +0.00 | 0.0 | +0.000 | 0.0% | 0.0% | 100.0% | 100.0% | 100.0% | 0.000 | 100.0% | 0.0% | 0.0% |
| 0.3 | +0.00 | 0.05 | +0.000 | 0.1% | 0.0% | 90.9% | 90.9% | 90.9% | 0.035 | 58.9% | 0.0% | 0.0% |
| 0.3 | +0.00 | 0.1 | +0.000 | 0.3% | 0.1% | 84.0% | 84.0% | 84.0% | 0.070 | 22.0% | 0.0% | 0.0% |
| 0.3 | +0.00 | 0.2 | +0.000 | 0.9% | 0.4% | 74.8% | 74.8% | 74.8% | 0.140 | 1.8% | 0.0% | 0.0% |
| 0.3 | +0.05 | 0.0 | +0.050 | 1.6% | 1.6% | 100.0% | 100.0% | 100.0% | 0.000 | 100.0% | 0.0% | 0.0% |
| 0.3 | +0.05 | 0.05 | +0.050 | 2.1% | 2.4% | 94.5% | 94.5% | 90.9% | 0.033 | 62.5% | 0.0% | 0.0% |
| 0.3 | +0.05 | 0.1 | +0.050 | 2.7% | 3.3% | 90.2% | 90.2% | 84.0% | 0.065 | 25.7% | 0.0% | 0.0% |
| 0.3 | +0.05 | 0.2 | +0.050 | 4.1% | 4.9% | 83.9% | 83.9% | 74.8% | 0.130 | 2.7% | 0.0% | 0.0% |
| 0.3 | +0.10 | 0.0 | +0.100 | 6.0% | 13.3% | 100.0% | 100.0% | 100.0% | 0.000 | 100.0% | 0.8% | 0.8% |
| 0.3 | +0.10 | 0.05 | +0.100 | 6.8% | 14.3% | 96.8% | 96.8% | 90.9% | 0.030 | 66.2% | 0.6% | 0.6% |
| 0.3 | +0.10 | 0.1 | +0.100 | 7.6% | 15.0% | 94.2% | 94.2% | 84.0% | 0.060 | 29.9% | 0.3% | 0.3% |
| 0.3 | +0.10 | 0.2 | +0.100 | 9.2% | 15.3% | 90.0% | 90.0% | 74.8% | 0.120 | 3.9% | 0.0% | 0.0% |
| 0.3 | +0.20 | 0.0 | +0.200 | 21.3% | 58.9% | 100.0% | 100.0% | 100.0% | 0.000 | 100.0% | 12.5% | 12.5% |
| 0.3 | +0.20 | 0.05 | +0.200 | 22.1% | 55.1% | 99.0% | 99.0% | 90.9% | 0.025 | 73.6% | 8.8% | 8.0% |
| 0.3 | +0.20 | 0.1 | +0.200 | 22.8% | 51.4% | 98.1% | 98.1% | 84.0% | 0.050 | 39.9% | 4.5% | 3.9% |
| 0.3 | +0.20 | 0.2 | +0.200 | 24.3% | 44.9% | 96.5% | 96.5% | 74.8% | 0.100 | 8.0% | 0.8% | 0.6% |
| 0.3 | +0.30 | 0.0 | +0.300 | 41.4% | 89.3% | 100.0% | 100.0% | 100.0% | 0.000 | 100.0% | 37.0% | 37.0% |
| 0.3 | +0.30 | 0.05 | +0.300 | 41.9% | 84.3% | 99.8% | 99.8% | 90.9% | 0.020 | 81.0% | 28.4% | 25.9% |
| 0.3 | +0.30 | 0.1 | +0.300 | 42.3% | 79.8% | 99.5% | 99.5% | 84.0% | 0.040 | 52.1% | 17.4% | 14.7% |
| 0.3 | +0.30 | 0.2 | +0.300 | 43.1% | 72.3% | 99.0% | 99.0% | 74.8% | 0.080 | 15.9% | 4.9% | 3.7% |
| 0.3 | +0.40 | 0.0 | +0.400 | 62.0% | 98.4% | 100.0% | 100.0% | 100.0% | 0.000 | 100.0% | 61.0% | 61.0% |
| 0.3 | +0.40 | 0.05 | +0.400 | 62.1% | 96.1% | 100.0% | 100.0% | 90.9% | 0.015 | 87.9% | 52.4% | 47.6% |
| 0.3 | +0.40 | 0.1 | +0.400 | 62.2% | 94.0% | 99.9% | 99.9% | 84.0% | 0.030 | 66.2% | 38.6% | 32.4% |
| 0.3 | +0.40 | 0.2 | +0.400 | 62.3% | 90.1% | 99.8% | 99.8% | 74.8% | 0.060 | 29.9% | 16.7% | 12.5% |
| 0.5 | -0.10 | 0.0 | -0.100 | 0.0% | 0.0% | 34.9% | 34.9% | 100.0% | 0.200 | 0.1% | 0.0% | 0.0% |
| 0.5 | -0.10 | 0.05 | -0.100 | 0.1% | 0.0% | 35.0% | 35.0% | 89.4% | 0.220 | 0.1% | 0.0% | 0.0% |
| 0.5 | -0.10 | 0.1 | -0.100 | 0.2% | 0.0% | 35.3% | 35.3% | 81.8% | 0.240 | 0.0% | 0.0% | 0.0% |
| 0.5 | -0.10 | 0.2 | -0.100 | 0.8% | 0.1% | 36.1% | 36.1% | 72.5% | 0.280 | 0.0% | 0.0% | 0.0% |
| 0.5 | -0.05 | 0.0 | -0.050 | 0.0% | 0.0% | 59.9% | 59.9% | 100.0% | 0.100 | 8.0% | 0.0% | 0.0% |
| 0.5 | -0.05 | 0.05 | -0.050 | 0.1% | 0.0% | 56.8% | 56.8% | 89.4% | 0.122 | 3.5% | 0.0% | 0.0% |
| 0.5 | -0.05 | 0.1 | -0.050 | 0.3% | 0.0% | 54.6% | 54.6% | 81.8% | 0.145 | 1.5% | 0.0% | 0.0% |
| 0.5 | -0.05 | 0.2 | -0.050 | 1.2% | 0.2% | 52.1% | 52.1% | 72.5% | 0.190 | 0.2% | 0.0% | 0.0% |
| 0.5 | +0.00 | 0.0 | +0.000 | 0.0% | 0.0% | 100.0% | 100.0% | 100.0% | 0.000 | 100.0% | 0.0% | 0.0% |
| 0.5 | +0.00 | 0.05 | +0.000 | 0.1% | 0.0% | 89.4% | 89.4% | 89.4% | 0.025 | 73.6% | 0.0% | 0.0% |
| 0.5 | +0.00 | 0.1 | +0.000 | 0.5% | 0.1% | 81.8% | 81.8% | 81.8% | 0.050 | 39.9% | 0.0% | 0.0% |
| 0.5 | +0.00 | 0.2 | +0.000 | 1.7% | 0.7% | 72.5% | 72.5% | 72.5% | 0.100 | 8.0% | 0.0% | 0.0% |
| 0.5 | +0.05 | 0.0 | +0.050 | 2.0% | 1.6% | 100.0% | 100.0% | 100.0% | 0.000 | 100.0% | 0.0% | 0.0% |
| 0.5 | +0.05 | 0.05 | +0.050 | 2.9% | 2.6% | 93.8% | 93.8% | 89.4% | 0.022 | 77.3% | 0.1% | 0.0% |
| 0.5 | +0.05 | 0.1 | +0.050 | 3.8% | 3.6% | 89.0% | 89.0% | 81.8% | 0.045 | 45.7% | 0.0% | 0.0% |
| 0.5 | +0.05 | 0.2 | +0.050 | 5.9% | 5.3% | 82.4% | 82.4% | 72.5% | 0.090 | 11.4% | 0.0% | 0.0% |
| 0.5 | +0.10 | 0.0 | +0.100 | 7.5% | 13.3% | 100.0% | 100.0% | 100.0% | 0.000 | 100.0% | 1.0% | 1.0% |
| 0.5 | +0.10 | 0.05 | +0.100 | 8.6% | 14.4% | 96.5% | 96.5% | 89.4% | 0.020 | 81.0% | 0.9% | 0.9% |
| 0.5 | +0.10 | 0.1 | +0.100 | 9.8% | 15.1% | 93.6% | 93.6% | 81.8% | 0.040 | 52.1% | 0.7% | 0.6% |
| 0.5 | +0.10 | 0.2 | +0.100 | 12.1% | 15.2% | 89.2% | 89.2% | 72.5% | 0.080 | 15.9% | 0.2% | 0.2% |
| 0.5 | +0.20 | 0.0 | +0.200 | 25.2% | 58.9% | 100.0% | 100.0% | 100.0% | 0.000 | 100.0% | 14.8% | 14.8% |
| 0.5 | +0.20 | 0.05 | +0.200 | 26.1% | 55.1% | 99.0% | 99.0% | 89.4% | 0.015 | 87.9% | 12.4% | 11.2% |
| 0.5 | +0.20 | 0.1 | +0.200 | 27.0% | 51.4% | 98.1% | 98.1% | 81.8% | 0.030 | 66.2% | 8.9% | 7.4% |
| 0.5 | +0.20 | 0.2 | +0.200 | 28.7% | 44.9% | 96.5% | 96.5% | 72.5% | 0.060 | 29.9% | 3.6% | 2.7% |
| 0.5 | +0.30 | 0.0 | +0.300 | 46.5% | 89.3% | 100.0% | 100.0% | 100.0% | 0.000 | 100.0% | 41.5% | 41.5% |
| 0.5 | +0.30 | 0.05 | +0.300 | 46.8% | 85.1% | 99.8% | 99.8% | 89.4% | 0.010 | 93.9% | 37.3% | 33.4% |
| 0.5 | +0.30 | 0.1 | +0.300 | 47.2% | 81.3% | 99.6% | 99.6% | 81.8% | 0.020 | 81.0% | 30.8% | 25.3% |
| 0.5 | +0.30 | 0.2 | +0.300 | 47.9% | 74.6% | 99.2% | 99.2% | 72.5% | 0.040 | 52.1% | 18.3% | 13.4% |
| 0.5 | +0.40 | 0.0 | +0.400 | 66.1% | 98.4% | 100.0% | 100.0% | 100.0% | 0.000 | 100.0% | 65.1% | 65.1% |
| 0.5 | +0.40 | 0.05 | +0.400 | 66.2% | 97.1% | 100.0% | 100.0% | 89.4% | 0.005 | 98.3% | 63.1% | 56.4% |
| 0.5 | +0.40 | 0.1 | +0.400 | 66.2% | 95.9% | 99.9% | 99.9% | 81.8% | 0.010 | 93.9% | 59.5% | 48.7% |
| 0.5 | +0.40 | 0.2 | +0.400 | 66.2% | 93.5% | 99.9% | 99.9% | 72.5% | 0.020 | 81.0% | 50.0% | 36.3% |

#### T1b. Stage-1 replay sensitivities: expression probability e per successful episode, and 'source instance already solved by G0' (Y0=1 fixed on the source)

| p0 | delta | f | e=1, src~variants | e=0.8 | src solved | e=0.8 & src solved |
|---|---|---|---|---|---|---|
| 0.3 | +0.10 | 0.0 | 6.0% | 2.8% | 5.2% | 2.3% |
| 0.3 | +0.10 | 0.1 | 7.6% | 3.5% | 6.5% | 2.9% |
| 0.3 | +0.20 | 0.0 | 21.3% | 9.7% | 18.1% | 7.8% |
| 0.3 | +0.20 | 0.1 | 22.8% | 10.4% | 19.0% | 8.1% |
| 0.3 | +0.40 | 0.0 | 62.0% | 26.1% | 52.5% | 20.8% |
| 0.3 | +0.40 | 0.1 | 62.2% | 26.1% | 51.9% | 20.6% |
| 0.5 | +0.10 | 0.0 | 7.5% | 3.2% | 5.2% | 2.1% |
| 0.5 | +0.10 | 0.1 | 9.8% | 4.2% | 6.8% | 2.7% |
| 0.5 | +0.20 | 0.0 | 25.2% | 10.3% | 18.1% | 7.0% |
| 0.5 | +0.20 | 0.1 | 27.0% | 11.1% | 19.3% | 7.4% |
| 0.5 | +0.40 | 0.0 | 66.1% | 23.8% | 52.5% | 18.4% |
| 0.5 | +0.40 | 0.1 | 66.2% | 23.8% | 52.4% | 18.3% |

#### T1c. Stage-4 regression pass probability vs per-task T_reg failure probability h_reg (exact binomial, 40 tasks, <=1 failure allowed)

| h_reg | P(pass) tau_r=0.03 | tau_r=0.02 (0 fails) | tau_r=0.05 (<=2 fails) |
|---|---|---|---|
| 0.0 | 100.0% | 100.0% | 100.0% |
| 0.01 | 93.9% | 66.9% | 99.3% |
| 0.02 | 81.0% | 44.6% | 95.4% |
| 0.03 | 66.2% | 29.6% | 88.2% |
| 0.05 | 39.9% | 12.9% | 67.7% |
| 0.075 | 18.8% | 4.4% | 41.4% |
| 0.1 | 8.0% | 1.5% | 22.3% |
| 0.15 | 1.2% | 0.2% | 4.9% |
| 0.2 | 0.1% | 0.0% | 0.8% |

#### T1d. Mapping of noise rate f to h_reg for a neutral trait under the two noise models

| f | symmetric: h_reg | P(pass) | redraw p0=0.3: h_reg | P(pass) | redraw p0=0.5: h_reg | P(pass) |
|---|---|---|---|---|---|---|
| 0.0 | 0.000 | 100.0% | 0.000 | 100.0% | 0.000 | 100.0% |
| 0.05 | 0.050 | 39.9% | 0.035 | 58.9% | 0.025 | 73.6% |
| 0.1 | 0.100 | 8.0% | 0.070 | 22.0% | 0.050 | 39.9% |
| 0.2 | 0.200 | 0.1% | 0.140 | 1.8% | 0.100 | 8.0% |

#### T1e. Specialist traits: replay/in-scope see the in-family effect, the stratified 20-task ablation pool sees share x in-family effect

| p0 | in-family delta | family share of pool | pool delta | f | replay | ablation | gen-in |
|---|---|---|---|---|---|---|---|
| 0.3 | +0.30 | 0.17 | +0.050 | 0.0 | 41.4% | 1.6% | 100.0% |
| 0.3 | +0.30 | 0.17 | +0.050 | 0.05 | 41.9% | 2.4% | 99.8% |
| 0.3 | +0.30 | 0.33 | +0.100 | 0.0 | 41.4% | 13.3% | 100.0% |
| 0.3 | +0.30 | 0.33 | +0.100 | 0.05 | 41.9% | 14.3% | 99.8% |
| 0.3 | +0.30 | 1.00 | +0.300 | 0.0 | 41.4% | 89.3% | 100.0% |
| 0.3 | +0.30 | 1.00 | +0.300 | 0.05 | 41.9% | 84.3% | 99.8% |
| 0.3 | +0.50 | 0.17 | +0.083 | 0.0 | 79.2% | 8.0% | 100.0% |
| 0.3 | +0.50 | 0.17 | +0.083 | 0.05 | 79.1% | 9.2% | 100.0% |
| 0.3 | +0.50 | 0.33 | +0.167 | 0.0 | 79.2% | 43.3% | 100.0% |
| 0.3 | +0.50 | 0.33 | +0.167 | 0.05 | 79.1% | 41.3% | 100.0% |
| 0.3 | +0.50 | 1.00 | +0.500 | 0.0 | 79.2% | 99.9% | 100.0% |
| 0.3 | +0.50 | 1.00 | +0.500 | 0.05 | 79.1% | 99.4% | 100.0% |
| 0.3 | +0.80 | 0.17 | +0.133 | 0.0 | 96.9% | 27.3% | 100.0% |
| 0.3 | +0.80 | 0.17 | +0.133 | 0.05 | 96.9% | 27.1% | 100.0% |
| 0.3 | +0.80 | 0.33 | +0.267 | 0.0 | 96.9% | 82.2% | 100.0% |
| 0.3 | +0.80 | 0.33 | +0.267 | 0.05 | 96.9% | 76.8% | 100.0% |
| 0.3 | +0.80 | 1.00 | +0.800 | 0.0 | 96.9% | 100.0% | 100.0% |
| 0.3 | +0.80 | 1.00 | +0.800 | 0.05 | 96.9% | 100.0% | 100.0% |
| 0.5 | +0.30 | 0.17 | +0.050 | 0.0 | 46.5% | 1.6% | 100.0% |
| 0.5 | +0.30 | 0.17 | +0.050 | 0.05 | 46.8% | 2.6% | 99.8% |
| 0.5 | +0.30 | 0.33 | +0.100 | 0.0 | 46.5% | 13.3% | 100.0% |
| 0.5 | +0.30 | 0.33 | +0.100 | 0.05 | 46.8% | 14.4% | 99.8% |
| 0.5 | +0.30 | 1.00 | +0.300 | 0.0 | 46.5% | 89.3% | 100.0% |
| 0.5 | +0.30 | 1.00 | +0.300 | 0.05 | 46.8% | 85.1% | 99.8% |
| 0.5 | +0.50 | 0.17 | +0.083 | 0.0 | 81.2% | 8.0% | 100.0% |
| 0.5 | +0.50 | 0.17 | +0.083 | 0.05 | 81.2% | 9.3% | 100.0% |
| 0.5 | +0.50 | 0.33 | +0.167 | 0.0 | 81.2% | 43.3% | 100.0% |
| 0.5 | +0.50 | 0.33 | +0.167 | 0.05 | 81.2% | 41.2% | 100.0% |
| 0.5 | +0.50 | 1.00 | +0.500 | 0.0 | 81.2% | 99.9% | 100.0% |
| 0.5 | +0.50 | 1.00 | +0.500 | 0.05 | 81.2% | 99.9% | 100.0% |
| 0.5 | +0.80 | 0.17 | +0.133 | 0.0 | 81.2% | 27.3% | 100.0% |
| 0.5 | +0.80 | 0.17 | +0.133 | 0.05 | 81.2% | 27.0% | 100.0% |
| 0.5 | +0.80 | 0.33 | +0.267 | 0.0 | 81.2% | 82.2% | 100.0% |
| 0.5 | +0.80 | 0.33 | +0.267 | 0.05 | 81.2% | 77.3% | 100.0% |
| 0.5 | +0.80 | 1.00 | +0.800 | 0.0 | 81.2% | 99.9% | 100.0% |
| 0.5 | +0.80 | 1.00 | +0.800 | 0.05 | 81.2% | 99.9% | 100.0% |

#### T2. End-to-end: K=10 candidates per seed, mixture helpful/neutral/harmful, harmful delta = -0.10, 3000 MC reps, incl. A1 interaction stage

| p0 | mix H/N/Harm | delta_help | f | E[eligible] | E[elim by interaction] | E[admitted] | P(arm C empty) | E[helpful adm] | E[neutral adm] | E[harmful adm] | P(any non-helpful adm) | precision | nonempty |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.3 | 20/50/30 | +0.10 | 0.0 | 0.01 | 0.00 | 0.01 | 98.7% | 0.01 | 0.000 | 0.000 | 0.0% | 100.0% |
| 0.3 | 20/50/30 | +0.10 | 0.05 | 0.01 | 0.00 | 0.01 | 98.7% | 0.01 | 0.000 | 0.000 | 0.0% | 100.0% |
| 0.3 | 20/50/30 | +0.10 | 0.1 | 0.01 | 0.00 | 0.01 | 99.4% | 0.01 | 0.000 | 0.000 | 0.0% | 100.0% |
| 0.3 | 20/50/30 | +0.10 | 0.2 | 0.00 | 0.00 | 0.00 | 99.9% | 0.00 | 0.000 | 0.000 | 0.0% | 100.0% |
| 0.3 | 20/50/30 | +0.20 | 0.0 | 0.25 | 0.00 | 0.25 | 77.8% | 0.25 | 0.000 | 0.000 | 0.0% | 100.0% |
| 0.3 | 20/50/30 | +0.20 | 0.05 | 0.19 | 0.00 | 0.19 | 83.0% | 0.19 | 0.000 | 0.000 | 0.0% | 100.0% |
| 0.3 | 20/50/30 | +0.20 | 0.1 | 0.10 | 0.00 | 0.10 | 90.7% | 0.10 | 0.000 | 0.000 | 0.0% | 100.0% |
| 0.3 | 20/50/30 | +0.20 | 0.2 | 0.01 | 0.00 | 0.01 | 98.8% | 0.01 | 0.000 | 0.000 | 0.0% | 100.0% |
| 0.3 | 20/50/30 | +0.40 | 0.0 | 1.24 | 0.00 | 1.24 | 25.9% | 1.24 | 0.000 | 0.000 | 0.0% | 100.0% |
| 0.3 | 20/50/30 | +0.40 | 0.05 | 1.06 | 0.00 | 1.06 | 32.5% | 1.06 | 0.000 | 0.000 | 0.0% | 100.0% |
| 0.3 | 20/50/30 | +0.40 | 0.1 | 0.78 | 0.00 | 0.78 | 44.5% | 0.78 | 0.000 | 0.000 | 0.0% | 100.0% |
| 0.3 | 20/50/30 | +0.40 | 0.2 | 0.33 | 0.00 | 0.33 | 71.9% | 0.33 | 0.000 | 0.000 | 0.0% | 100.0% |
| 0.3 | 50/30/20 | +0.20 | 0.0 | 0.64 | 0.00 | 0.64 | 51.7% | 0.64 | 0.000 | 0.000 | 0.0% | 100.0% |
| 0.3 | 50/30/20 | +0.20 | 0.05 | 0.43 | 0.00 | 0.43 | 64.3% | 0.43 | 0.000 | 0.000 | 0.0% | 100.0% |
| 0.3 | 10/60/30 | +0.20 | 0.0 | 0.12 | 0.00 | 0.12 | 88.3% | 0.12 | 0.000 | 0.000 | 0.0% | 100.0% |
| 0.3 | 10/60/30 | +0.20 | 0.05 | 0.09 | 0.00 | 0.09 | 91.1% | 0.09 | 0.000 | 0.000 | 0.0% | 100.0% |
| 0.5 | 20/50/30 | +0.10 | 0.0 | 0.02 | 0.00 | 0.02 | 98.4% | 0.02 | 0.000 | 0.000 | 0.0% | 100.0% |
| 0.5 | 20/50/30 | +0.10 | 0.05 | 0.02 | 0.00 | 0.02 | 97.6% | 0.02 | 0.000 | 0.000 | 0.0% | 100.0% |
| 0.5 | 20/50/30 | +0.10 | 0.1 | 0.01 | 0.00 | 0.01 | 98.5% | 0.01 | 0.000 | 0.000 | 0.0% | 97.7% |
| 0.5 | 20/50/30 | +0.10 | 0.2 | 0.00 | 0.00 | 0.00 | 99.6% | 0.00 | 0.000 | 0.000 | 0.0% | 100.0% |
| 0.5 | 20/50/30 | +0.20 | 0.0 | 0.30 | 0.00 | 0.30 | 73.6% | 0.30 | 0.000 | 0.000 | 0.0% | 100.0% |
| 0.5 | 20/50/30 | +0.20 | 0.05 | 0.25 | 0.00 | 0.25 | 78.0% | 0.25 | 0.000 | 0.000 | 0.0% | 100.0% |
| 0.5 | 20/50/30 | +0.20 | 0.1 | 0.18 | 0.00 | 0.18 | 83.7% | 0.18 | 0.000 | 0.000 | 0.0% | 100.0% |
| 0.5 | 20/50/30 | +0.20 | 0.2 | 0.07 | 0.00 | 0.07 | 93.3% | 0.07 | 0.000 | 0.000 | 0.0% | 100.0% |
| 0.5 | 20/50/30 | +0.40 | 0.0 | 1.30 | 0.00 | 1.30 | 24.5% | 1.30 | 0.000 | 0.000 | 0.0% | 100.0% |
| 0.5 | 20/50/30 | +0.40 | 0.05 | 1.27 | 0.00 | 1.26 | 25.7% | 1.26 | 0.000 | 0.000 | 0.0% | 100.0% |
| 0.5 | 20/50/30 | +0.40 | 0.1 | 1.21 | 0.01 | 1.20 | 27.4% | 1.20 | 0.000 | 0.000 | 0.0% | 100.0% |
| 0.5 | 20/50/30 | +0.40 | 0.2 | 1.01 | 0.02 | 0.99 | 34.5% | 0.99 | 0.000 | 0.000 | 0.0% | 100.0% |
| 0.5 | 50/30/20 | +0.20 | 0.0 | 0.74 | 0.00 | 0.74 | 47.1% | 0.74 | 0.000 | 0.000 | 0.0% | 100.0% |
| 0.5 | 50/30/20 | +0.20 | 0.05 | 0.64 | 0.00 | 0.63 | 51.7% | 0.63 | 0.000 | 0.000 | 0.0% | 100.0% |
| 0.5 | 10/60/30 | +0.20 | 0.0 | 0.14 | 0.00 | 0.14 | 86.5% | 0.14 | 0.000 | 0.000 | 0.0% | 100.0% |
| 0.5 | 10/60/30 | +0.20 | 0.05 | 0.13 | 0.00 | 0.13 | 88.1% | 0.13 | 0.000 | 0.000 | 0.0% | 100.0% |

#### T3. H2 T_eval component: P[point(C-B) >= +5pp AND LB>0 AND direction in >=2/3 seeds]; n=134 paired tasks x 3 seeds, cluster = task (seed-averaged), p0=0.4 (UNVERIFIED), delta_B=0, f_B=f_C=f_arm; rho = share of tasks whose flips are identical across seeds (1 = seeds add no information); 1000 reps x 1000 resamples

| rho | f_arm | true C-B | E[obs C-B] | sd(obs) | P(point>=5pp) | P(LB>0) | P(dir 2/3) | P(T_eval part) |
|---|---|---|---|---|---|---|---|---|
| 0.0 | 0.1 | +0.00 | +0.001 | 0.015 | 0.0% | 2.9% | 42.9% | 0.0% |
| 0.0 | 0.1 | +0.02 | +0.019 | 0.017 | 3.4% | 21.7% | 79.9% | 3.4% |
| 0.0 | 0.1 | +0.04 | +0.039 | 0.018 | 25.7% | 60.6% | 96.1% | 25.7% |
| 0.0 | 0.1 | +0.05 | +0.051 | 0.018 | 50.7% | 80.5% | 99.0% | 50.7% |
| 0.0 | 0.1 | +0.06 | +0.060 | 0.019 | 66.7% | 89.7% | 99.9% | 66.7% |
| 0.0 | 0.1 | +0.08 | +0.080 | 0.019 | 92.3% | 98.0% | 99.9% | 92.3% |
| 0.0 | 0.1 | +0.10 | +0.100 | 0.021 | 98.8% | 99.9% | 100.0% | 98.8% |
| 0.0 | 0.1 | +0.12 | +0.120 | 0.022 | 99.9% | 100.0% | 100.0% | 99.9% |
| 0.0 | 0.1 | +0.15 | +0.150 | 0.024 | 100.0% | 100.0% | 100.0% | 100.0% |
| 0.0 | 0.1 | +0.20 | +0.200 | 0.024 | 100.0% | 100.0% | 100.0% | 100.0% |
| 0.0 | 0.2 | +0.00 | +0.001 | 0.020 | 0.5% | 3.0% | 44.4% | 0.5% |
| 0.0 | 0.2 | +0.02 | +0.019 | 0.022 | 7.4% | 15.1% | 72.1% | 7.4% |
| 0.0 | 0.2 | +0.04 | +0.039 | 0.023 | 29.5% | 40.6% | 91.1% | 29.5% |
| 0.0 | 0.2 | +0.05 | +0.049 | 0.023 | 47.3% | 58.3% | 96.2% | 47.3% |
| 0.0 | 0.2 | +0.06 | +0.061 | 0.024 | 66.9% | 76.9% | 98.6% | 66.9% |
| 0.0 | 0.2 | +0.08 | +0.080 | 0.024 | 89.3% | 92.4% | 99.9% | 88.9% |
| 0.0 | 0.2 | +0.10 | +0.102 | 0.024 | 98.4% | 99.0% | 100.0% | 98.4% |
| 0.0 | 0.2 | +0.12 | +0.120 | 0.026 | 99.4% | 99.3% | 100.0% | 99.3% |
| 0.0 | 0.2 | +0.15 | +0.149 | 0.026 | 100.0% | 100.0% | 100.0% | 100.0% |
| 0.0 | 0.2 | +0.20 | +0.202 | 0.027 | 100.0% | 100.0% | 100.0% | 100.0% |
| 0.0 | 0.3 | +0.00 | +0.000 | 0.025 | 1.3% | 2.8% | 44.2% | 1.3% |
| 0.0 | 0.3 | +0.02 | +0.020 | 0.026 | 11.6% | 13.3% | 70.7% | 11.0% |
| 0.0 | 0.3 | +0.04 | +0.039 | 0.026 | 34.7% | 35.4% | 89.5% | 33.3% |
| 0.0 | 0.3 | +0.05 | +0.050 | 0.026 | 49.1% | 49.5% | 94.1% | 46.8% |
| 0.0 | 0.3 | +0.06 | +0.059 | 0.026 | 62.5% | 62.0% | 96.1% | 59.8% |
| 0.0 | 0.3 | +0.08 | +0.079 | 0.026 | 85.8% | 84.3% | 99.7% | 83.6% |
| 0.0 | 0.3 | +0.10 | +0.100 | 0.027 | 97.3% | 95.9% | 100.0% | 95.8% |
| 0.0 | 0.3 | +0.12 | +0.121 | 0.028 | 99.5% | 99.3% | 100.0% | 99.3% |
| 0.0 | 0.3 | +0.15 | +0.151 | 0.028 | 100.0% | 100.0% | 100.0% | 100.0% |
| 0.0 | 0.3 | +0.20 | +0.200 | 0.030 | 100.0% | 100.0% | 100.0% | 100.0% |
| 1.0 | 0.1 | +0.00 | +0.002 | 0.026 | 3.0% | 2.1% | 47.2% | 1.8% |
| 1.0 | 0.1 | +0.02 | +0.021 | 0.029 | 15.7% | 10.3% | 71.7% | 9.8% |
| 1.0 | 0.1 | +0.04 | +0.040 | 0.031 | 40.7% | 23.6% | 87.1% | 23.5% |
| 1.0 | 0.1 | +0.05 | +0.050 | 0.032 | 52.7% | 33.3% | 93.3% | 33.2% |
| 1.0 | 0.1 | +0.06 | +0.061 | 0.032 | 65.7% | 46.2% | 96.4% | 46.2% |
| 1.0 | 0.1 | +0.08 | +0.080 | 0.035 | 82.5% | 63.8% | 98.4% | 63.8% |
| 1.0 | 0.1 | +0.10 | +0.100 | 0.034 | 93.8% | 83.3% | 99.7% | 83.3% |
| 1.0 | 0.1 | +0.12 | +0.120 | 0.037 | 97.2% | 89.9% | 99.9% | 89.9% |
| 1.0 | 0.1 | +0.15 | +0.149 | 0.039 | 99.8% | 97.8% | 100.0% | 97.8% |
| 1.0 | 0.1 | +0.20 | +0.200 | 0.041 | 100.0% | 100.0% | 100.0% | 100.0% |
| 1.0 | 0.2 | +0.00 | +0.001 | 0.035 | 8.2% | 2.1% | 47.7% | 2.1% |
| 1.0 | 0.2 | +0.02 | +0.019 | 0.038 | 21.3% | 7.0% | 64.6% | 7.0% |
| 1.0 | 0.2 | +0.04 | +0.040 | 0.039 | 40.5% | 16.0% | 84.1% | 16.0% |
| 1.0 | 0.2 | +0.05 | +0.050 | 0.040 | 50.8% | 23.3% | 88.0% | 23.3% |
| 1.0 | 0.2 | +0.06 | +0.058 | 0.040 | 60.2% | 29.1% | 90.8% | 29.1% |
| 1.0 | 0.2 | +0.08 | +0.079 | 0.044 | 75.5% | 45.8% | 97.0% | 45.8% |
| 1.0 | 0.2 | +0.10 | +0.098 | 0.041 | 88.7% | 62.3% | 99.4% | 62.3% |
| 1.0 | 0.2 | +0.12 | +0.119 | 0.043 | 95.4% | 78.6% | 99.9% | 78.6% |
| 1.0 | 0.2 | +0.15 | +0.152 | 0.044 | 99.4% | 92.7% | 100.0% | 92.7% |
| 1.0 | 0.2 | +0.20 | +0.200 | 0.045 | 100.0% | 99.4% | 100.0% | 99.4% |
| 1.0 | 0.3 | +0.00 | -0.001 | 0.043 | 12.8% | 2.1% | 44.1% | 2.1% |
| 1.0 | 0.3 | +0.02 | +0.023 | 0.043 | 29.4% | 6.1% | 67.9% | 6.1% |
| 1.0 | 0.3 | +0.04 | +0.042 | 0.046 | 43.9% | 16.0% | 79.8% | 16.0% |
| 1.0 | 0.3 | +0.05 | +0.050 | 0.045 | 51.3% | 18.6% | 85.2% | 18.6% |
| 1.0 | 0.3 | +0.06 | +0.060 | 0.045 | 61.2% | 23.7% | 89.5% | 23.7% |
| 1.0 | 0.3 | +0.08 | +0.079 | 0.045 | 76.1% | 40.0% | 94.4% | 40.0% |
| 1.0 | 0.3 | +0.10 | +0.100 | 0.047 | 85.9% | 53.9% | 97.8% | 53.9% |
| 1.0 | 0.3 | +0.12 | +0.120 | 0.047 | 93.6% | 70.0% | 99.2% | 70.0% |
| 1.0 | 0.3 | +0.15 | +0.150 | 0.049 | 97.6% | 85.6% | 99.9% | 85.6% |
| 1.0 | 0.3 | +0.20 | +0.198 | 0.050 | 100.0% | 97.4% | 100.0% | 97.4% |

#### T3b. Minimum detectable C-B effect at 80% power (linear interpolation on the grid). 'nominal' = model input delta_C; 'realized' = E[observed C-B] on T_eval (= nominal x (1-2 f_arm) under symmetric noise), i.e. the effect that is actually present

| rho | f_arm | MDE nominal: full criterion | MDE nominal: LB>0 & seed only | MDE realized: full criterion | MDE realized: LB>0 & seed only |
|---|---|---|---|---|---|
| 0.0 | 0.1 | 0.070 | 0.050 | 0.070 | 0.051 |
| 0.0 | 0.2 | 0.072 | 0.064 | 0.072 | 0.065 |
| 0.0 | 0.3 | 0.077 | 0.076 | 0.076 | 0.075 |
| 1.0 | 0.1 | 0.097 | 0.097 | 0.097 | 0.097 |
| 1.0 | 0.2 | 0.123 | 0.123 | 0.123 | 0.123 |
| 1.0 | 0.3 | 0.139 | 0.139 | 0.139 | 0.139 |

#### T4. H2 regression component: P[reg(B)-reg(C) > 0 with cluster-bootstrap LB > 0] over the 40 T_reg tasks x 3 seeds (per-task per-seed failure probabilities r_B, r_C, independent arms)

| rho | r_B | r_C | true diff | P(reg part) |
|---|---|---|---|---|
| 0.0 | 0.05 | 0.0 | +0.050 | 84.0% |
| 0.0 | 0.05 | 0.025 | +0.025 | 16.9% |
| 0.0 | 0.1 | 0.0 | +0.100 | 99.9% |
| 0.0 | 0.1 | 0.025 | +0.075 | 69.6% |
| 0.0 | 0.1 | 0.05 | +0.050 | 32.6% |
| 0.0 | 0.15 | 0.0 | +0.150 | 100.0% |
| 0.0 | 0.15 | 0.025 | +0.125 | 95.3% |
| 0.0 | 0.15 | 0.05 | +0.100 | 76.8% |
| 0.0 | 0.2 | 0.0 | +0.200 | 100.0% |
| 0.0 | 0.2 | 0.025 | +0.175 | 99.9% |
| 0.0 | 0.2 | 0.05 | +0.150 | 95.9% |
| 0.0 | 0.3 | 0.0 | +0.300 | 100.0% |
| 0.0 | 0.3 | 0.025 | +0.275 | 100.0% |
| 0.0 | 0.3 | 0.05 | +0.250 | 100.0% |
| 0.0 | 0.4 | 0.0 | +0.400 | 100.0% |
| 0.0 | 0.4 | 0.025 | +0.375 | 100.0% |
| 0.0 | 0.4 | 0.05 | +0.350 | 100.0% |
| 1.0 | 0.05 | 0.0 | +0.050 | 14.0% |
| 1.0 | 0.05 | 0.025 | +0.025 | 3.9% |
| 1.0 | 0.1 | 0.0 | +0.100 | 56.4% |
| 1.0 | 0.1 | 0.025 | +0.075 | 27.1% |
| 1.0 | 0.1 | 0.05 | +0.050 | 12.4% |
| 1.0 | 0.15 | 0.0 | +0.150 | 86.8% |
| 1.0 | 0.15 | 0.025 | +0.125 | 51.4% |
| 1.0 | 0.15 | 0.05 | +0.100 | 30.0% |
| 1.0 | 0.2 | 0.0 | +0.200 | 97.1% |
| 1.0 | 0.2 | 0.025 | +0.175 | 73.3% |
| 1.0 | 0.2 | 0.05 | +0.150 | 55.2% |
| 1.0 | 0.3 | 0.0 | +0.300 | 100.0% |
| 1.0 | 0.3 | 0.025 | +0.275 | 95.1% |
| 1.0 | 0.3 | 0.05 | +0.250 | 85.3% |
| 1.0 | 0.4 | 0.0 | +0.400 | 100.0% |
| 1.0 | 0.4 | 0.025 | +0.375 | 99.8% |
| 1.0 | 0.4 | 0.05 | +0.350 | 97.3% |

#### T4b. Regression component with positive B/C failure dependence (C's skills are a subset of B's), rho=1

| r_B | r_C | P(B fails | C fails) | P(reg part) |
|---|---|---|---|
| 0.1 | 0.025 | 0.0 | 26.1% |
| 0.1 | 0.025 | 0.5 | 35.8% |
| 0.1 | 0.025 | 1.0 | 55.8% |
| 0.15 | 0.05 | 0.0 | 28.7% |
| 0.15 | 0.05 | 0.5 | 43.7% |
| 0.15 | 0.05 | 1.0 | 85.8% |
| 0.2 | 0.05 | 0.0 | 52.0% |
| 0.2 | 0.05 | 0.5 | 67.6% |
| 0.2 | 0.05 | 1.0 | 96.5% |

#### T5. Joint H2 support probability = P(T_eval part) x P(regression part) (disjoint task sets => independent given parameters)

| rho | f_arm | true C-B | r_B | r_C | P(eval) | P(reg) | P(H2 supported) |
|---|---|---|---|---|---|---|---|
| 0.0 | 0.1 | +0.05 | 0.1 | 0.025 | 50.7% | 69.6% | 35.3% |
| 0.0 | 0.1 | +0.05 | 0.2 | 0.05 | 50.7% | 95.9% | 48.6% |
| 0.0 | 0.1 | +0.05 | 0.3 | 0.05 | 50.7% | 100.0% | 50.7% |
| 0.0 | 0.1 | +0.10 | 0.1 | 0.025 | 98.8% | 69.6% | 68.8% |
| 0.0 | 0.1 | +0.10 | 0.2 | 0.05 | 98.8% | 95.9% | 94.7% |
| 0.0 | 0.1 | +0.10 | 0.3 | 0.05 | 98.8% | 100.0% | 98.8% |
| 0.0 | 0.1 | +0.15 | 0.1 | 0.025 | 100.0% | 69.6% | 69.6% |
| 0.0 | 0.1 | +0.15 | 0.2 | 0.05 | 100.0% | 95.9% | 95.9% |
| 0.0 | 0.1 | +0.15 | 0.3 | 0.05 | 100.0% | 100.0% | 100.0% |
| 0.0 | 0.2 | +0.05 | 0.1 | 0.025 | 47.3% | 69.6% | 32.9% |
| 0.0 | 0.2 | +0.05 | 0.2 | 0.05 | 47.3% | 95.9% | 45.4% |
| 0.0 | 0.2 | +0.05 | 0.3 | 0.05 | 47.3% | 100.0% | 47.3% |
| 0.0 | 0.2 | +0.10 | 0.1 | 0.025 | 98.4% | 69.6% | 68.5% |
| 0.0 | 0.2 | +0.10 | 0.2 | 0.05 | 98.4% | 95.9% | 94.4% |
| 0.0 | 0.2 | +0.10 | 0.3 | 0.05 | 98.4% | 100.0% | 98.4% |
| 0.0 | 0.2 | +0.15 | 0.1 | 0.025 | 100.0% | 69.6% | 69.6% |
| 0.0 | 0.2 | +0.15 | 0.2 | 0.05 | 100.0% | 95.9% | 95.9% |
| 0.0 | 0.2 | +0.15 | 0.3 | 0.05 | 100.0% | 100.0% | 100.0% |
| 1.0 | 0.1 | +0.05 | 0.1 | 0.025 | 33.2% | 27.1% | 9.0% |
| 1.0 | 0.1 | +0.05 | 0.2 | 0.05 | 33.2% | 55.2% | 18.3% |
| 1.0 | 0.1 | +0.05 | 0.3 | 0.05 | 33.2% | 85.3% | 28.3% |
| 1.0 | 0.1 | +0.10 | 0.1 | 0.025 | 83.3% | 27.1% | 22.6% |
| 1.0 | 0.1 | +0.10 | 0.2 | 0.05 | 83.3% | 55.2% | 46.0% |
| 1.0 | 0.1 | +0.10 | 0.3 | 0.05 | 83.3% | 85.3% | 71.1% |
| 1.0 | 0.1 | +0.15 | 0.1 | 0.025 | 97.8% | 27.1% | 26.5% |
| 1.0 | 0.1 | +0.15 | 0.2 | 0.05 | 97.8% | 55.2% | 54.0% |
| 1.0 | 0.1 | +0.15 | 0.3 | 0.05 | 97.8% | 85.3% | 83.4% |
| 1.0 | 0.2 | +0.05 | 0.1 | 0.025 | 23.3% | 27.1% | 6.3% |
| 1.0 | 0.2 | +0.05 | 0.2 | 0.05 | 23.3% | 55.2% | 12.9% |
| 1.0 | 0.2 | +0.05 | 0.3 | 0.05 | 23.3% | 85.3% | 19.9% |
| 1.0 | 0.2 | +0.10 | 0.1 | 0.025 | 62.3% | 27.1% | 16.9% |
| 1.0 | 0.2 | +0.10 | 0.2 | 0.05 | 62.3% | 55.2% | 34.4% |
| 1.0 | 0.2 | +0.10 | 0.3 | 0.05 | 62.3% | 85.3% | 53.1% |
| 1.0 | 0.2 | +0.15 | 0.1 | 0.025 | 92.7% | 27.1% | 25.1% |
| 1.0 | 0.2 | +0.15 | 0.2 | 0.05 | 92.7% | 55.2% | 51.2% |
| 1.0 | 0.2 | +0.15 | 0.3 | 0.05 | 92.7% | 85.3% | 79.1% |

#### T6. H1 power: P[(C-A) cluster-bootstrap LB > 0]; arm A identical across seeds (G0 at temp 0), p0=0.4

| rho | f_C | true C-A | E[obs C-A] | P(H1 supported) |
|---|---|---|---|---|
| 0.0 | 0.1 | +0.000 | -0.000 | 2.1% |
| 0.0 | 0.1 | +0.025 | +0.025 | 41.8% |
| 0.0 | 0.1 | +0.050 | +0.050 | 87.6% |
| 0.0 | 0.1 | +0.075 | +0.075 | 98.3% |
| 0.0 | 0.1 | +0.100 | +0.099 | 100.0% |
| 0.0 | 0.1 | +0.150 | +0.149 | 100.0% |
| 0.0 | 0.2 | +0.000 | -0.000 | 2.6% |
| 0.0 | 0.2 | +0.025 | +0.025 | 27.3% |
| 0.0 | 0.2 | +0.050 | +0.051 | 71.4% |
| 0.0 | 0.2 | +0.075 | +0.075 | 93.8% |
| 0.0 | 0.2 | +0.100 | +0.101 | 98.7% |
| 0.0 | 0.2 | +0.150 | +0.151 | 100.0% |
| 0.0 | 0.3 | +0.000 | -0.001 | 2.7% |
| 0.0 | 0.3 | +0.025 | +0.024 | 20.2% |
| 0.0 | 0.3 | +0.050 | +0.050 | 52.9% |
| 0.0 | 0.3 | +0.075 | +0.074 | 82.8% |
| 0.0 | 0.3 | +0.100 | +0.100 | 96.1% |
| 0.0 | 0.3 | +0.150 | +0.151 | 100.0% |
| 1.0 | 0.1 | +0.000 | +0.000 | 2.5% |
| 1.0 | 0.1 | +0.025 | +0.026 | 16.8% |
| 1.0 | 0.1 | +0.050 | +0.050 | 46.1% |
| 1.0 | 0.1 | +0.075 | +0.075 | 75.2% |
| 1.0 | 0.1 | +0.100 | +0.100 | 91.6% |
| 1.0 | 0.1 | +0.150 | +0.152 | 99.8% |
| 1.0 | 0.2 | +0.000 | -0.000 | 2.0% |
| 1.0 | 0.2 | +0.025 | +0.023 | 10.4% |
| 1.0 | 0.2 | +0.050 | +0.050 | 33.6% |
| 1.0 | 0.2 | +0.075 | +0.074 | 57.5% |
| 1.0 | 0.2 | +0.100 | +0.101 | 80.1% |
| 1.0 | 0.2 | +0.150 | +0.150 | 97.3% |
| 1.0 | 0.3 | +0.000 | +0.001 | 2.5% |
| 1.0 | 0.3 | +0.025 | +0.026 | 10.6% |
| 1.0 | 0.3 | +0.050 | +0.049 | 24.8% |
| 1.0 | 0.3 | +0.075 | +0.075 | 47.2% |
| 1.0 | 0.3 | +0.100 | +0.100 | 70.4% |
| 1.0 | 0.3 | +0.150 | +0.148 | 93.9% |

#### T6b. Bootstrap-resolution check (rho=1, f_arm=0.2, 300 reps): P(LB>0) with 1k vs 10k resamples

| true C-B | 1k resamples | 10k resamples |
|---|---|---|
| +0.08 | 49.0% | 47.0% |
| +0.12 | 76.7% | 79.3% |
