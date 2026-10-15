# ADR-0006 — EXP-0001 final-evaluation statistics: the open analysis choices are pre-registered

Date: 2026-10-14 · Status: **ACCEPTED by the owner** (Samet, 2026-10-14, explicit answers in chat to eight questions, each on the recommended option; D8 followed the own review of #68). **Codex cross-review requested (SPEC §32).** **Amended by ADR-0009** (owner, 2026-10-15): D2 extends D7 to every arm (each distinct genome is evaluated once and every arm and seed whose genome it is reads its cells), and D5 scopes the H2 regression clause (see D4). Artifacts: `evaluation/metrics/final_statistics.py`, `experiments/manifests/EXP-0001-minimal-cig.yaml` (`evaluation.statistics.final_rule`), `tests/evaluation/test_final_statistics.py` · Origin: issue #13 Phase A ("statistics: paired per-task arm contrasts on T_eval, 10k task-cluster bootstrap, primary H2/H1 rules exactly as preregistered, ≥ 2/3 seed direction requirement, regression contrast, bloat and adaptation-cost secondary outputs"); ADR-0005, which left the H1/H2 bootstrap to #13.

**Problem.**
- The locked text fixes the core of the final analysis:
  - "cluster bootstrap over tasks, 10k resamples", 95 % CIs (design §5, manifest `evaluation.statistics.primary`);
  - H2: "`C−B ≥ +5pp` on T_eval, bootstrap 95% CI excluding 0, AND regression rate(C) < regression rate(B) with the regression-difference CI excluding 0, AND the C−B direction holds in ≥ 2/3 mining seeds" (design §8);
  - H1: "`C−A > 0` with bootstrap 95% CI excluding 0" (design §8);
  - "all pairwise arm contrasts on T_eval" are reported (design §5).
- It leaves seven choices open. Each can move a verdict or a reported number after the data are in. An independent extraction of every locked statement on the final analysis, adversarially checked, found these:
  1. **How the 3 mining seeds are combined** for the pooled contrasts. Design §4: "Uncertainty comes from (a) distinct task instances and (b) independent mining seeds." Design §10 R1 and ADR-0002 attach "cluster bootstrap" to the task instances. Neither says how the seeds are pooled.
  2. **The bootstrap's method, seed and cluster order.** ADR-0005: "The final-evaluation statistics (H1/H2, #13) are not implemented yet and are not decided here. The same rule is the natural default, but #13 decides it."
  3. **A per-seed tie** in the 2/3 direction rule (C−B = 0 in a seed, for example when the gate promotes every candidate, so C = B).
  4. **The regression-rate formula** for the final evaluation.
  5. **Missing or aborted episodes**, and a run stopped by a hard cap.
  6. **The secondary outputs** (genome size in genes and tokens; descendant adaptation cost): definitions and whether they get intervals.
  7. **Arm A** is unchanged G0 in every seed and deterministic at temperature 0. Is it run once or three times? Design §4: "Duplicate deterministic executions are never treated as independent evidence."

**Decision (owner).**
- **D1 Seed pooling.** For each task, the paired arm difference is averaged over the mining seeds. The tasks are the bootstrap clusters. The 2/3 rule checks the seeds separately.
  - On T_eval every task has one cell per seed, so a cluster's value is the mean of three differences in {−1, 0, 1}.
  - On T_reg likewise, since every seed shares one T_reg set (D8).
- **D2 Bootstrap.** The ADR-0005 rule for every contrast:
  - two-sided 95 % percentile interval, 10,000 resamples, seed 0, the clusters in `task_id` order (Python string order);
  - each contrast (C−B, C−A and B−A on T_eval; reg(C)−reg(B) on T_reg) computed separately with that rule;
  - computed with exact rational arithmetic on the index stream of `inheritance.bootstrap.paired_cluster_bootstrap` (same `random.Random(seed)` draws, same order statistics). Float rounding then cannot decide the strict "CI excluding 0" comparison. Agreement with the float implementation is tested.
- **The rule outcomes, as implemented** (the builder's reading of the locked text, for Codex to confirm). "≥ +5pp" and "> 0" are compared exactly on the pooled point estimate. "CI excluding 0" is strict: a bound equal to 0 does not exclude 0.
  - H2 success clause: C−B ≥ 5/100 and lower bound > 0.
  - H2 regression clause: reg(C)−reg(B) < 0 and upper bound < 0.
  - H2 seed clause: C−B > 0 in at least 2 of the 3 seeds (D3).
  - H1: C−A > 0 and lower bound > 0. H1 has no seed clause.
- **D3 Seed direction.** A seed holds the C−B direction only if its own C−B on T_eval is strictly positive. A tie does not hold.
- **D4 Regression rate.** reg(arm) is the fraction of the arm's T_reg tasks it fails. Every T_reg task was solved by G0. The H2 contrast reg(C)−reg(B) is the same under any formula that subtracts a per-task baseline, so this choice fixes reported levels, not the H2 outcome.
  - [corrected after independent verification 2026-10-15] "Solved by G0" means solved in at least one temperature-0.7 lifetime attempt (ADR-0007 E3), not at the evaluation's temperature 0.
  - Scope (ADR-0009 D5): the H2 regression clause is measured on T_reg, the suite the gate's stage 4A already used to decide which traits arm C keeps (paired against G0 since ADR-0009 D1). It therefore checks, in sample, the regression the gate controls. It is not an out-of-sample estimate, and it favours C over B by construction.
- **D5 Incomplete data fail closed.**
  - A primary H1/H2 outcome exists only when every (arm, seed, task) cell on T_eval and T_reg holds exactly one recorded outcome.
  - An episode cut off by infrastructure is not a failure; it must be completed (part of the option the owner chose). A step budget that runs out is a failure, ALFWorld's `won` being false (ADR-0003 D6).
  - A run stopped by a hard cap reports `incomplete`, with its reason and descriptive rates only; it has no primary outcome (design §8: "stop the experiment, mark the run incomplete, and record the reason").
- **D6 Secondary outputs are descriptive only:** no interval, no test.
  - `genome_size_genes`: the arm genome's gene refs (policy slots plus skills).
  - `genome_size_tokens`: the token count of the model-visible skill block rendering all of the genome's skills, whatever their regulation, counted by the recorded `TokenCounter`. A genome without skills has 0 skill-block tokens.
  - `descendant_adaptation_cost`: on the transfer subset, per arm, the tasks solved, and over the solved tasks the tokens and steps to success. Unsolved tasks are counted as censored and never given a cost.
  - All are reported per run and pooled per arm.
- **D7 Arm A runs once.** Its single T_eval run, and one run over every seed's T_reg tasks, serve every seed's C−A and A rates. ADR-0009 D2 extends this to every arm: a genome evaluated once serves every arm and seed whose genome it is, so wherever genomes coincide the contrast between them is exactly 0.
- **D8 T_reg is one set shared by every seed** (decided after the own review of #68). The analysis requires one 40-task T_reg for all three seeds and fails closed on anything else. With T_reg sets that differ by seed, D1's per-task weighting and the reported (cell-weighted) regression rates could differ in sign, and the review built cases where the H2 regression clause held while every reported rate said C regressed more. One shared set removes that. The T_reg derivation (#13) must therefore produce one set; as #13's handoff already requires, fewer than 40 qualifying tasks fails closed.

**Reason.**
- **D1** matches "cluster bootstrap over tasks" literally. It uses all three seeds' data without counting one task's three runs as three independent clusters (pseudo-replication), and keeps the seeds' role in the separate 2/3 rule. Three seeds are too few to resample (10 distinct multisets).
- **D2** reuses the rule already pre-registered for stage 2, so the experiment has one bootstrap convention. Exact arithmetic follows the repo's precedent for strict boundaries (`inheritance/regression.py` compares the stage-4 drop with `Fraction` for the same reason).
- **D3** is the literal reading of "the C−B direction holds" and the conservative one: a seed in which the gate changed nothing does not count as support.
- **D4** is the stage-4 gate's own definition (`regression_drop = 1 - success_with`).
- **D5** follows the cap rule's "stop, mark incomplete" and removes any choice of imputation after the data are in.
- **D6** keeps the H3 and H5 previews from being read as primary results. Their pools and definitions are less settled than H1/H2.
- **D7** follows "Duplicate deterministic executions are never treated as independent evidence" and saves up to 2 × (134 + 40 + 20) = 388 episodes.
- The choices were made before any EXP-0001 data. They were not chosen for their effect on any verdict.

**Scope.**
- This covers the final analysis only. The gate's stages keep ADR-0002 and ADR-0005.
- Not decided here; these stay with #13's runner and the Critic:
  - how the T_reg derivation picks its one shared set (D8) (since decided by ADR-0007 E3: among the T_mine tasks G0 solved in at least one lifetime episode of any seed);
  - the 20-task transfer subset for adaptation cost, which has no pool yet (since decided by ADR-0007 E4: the ablation pool);
  - how the threshold-sensitivity grid (design §5, "report how arm-C composition and T_eval outcomes would change") is reported for compositions that were never run on T_eval.
- The output states rule outcomes only. It never calls a result validated (SPEC §32) and makes no causal or inheritance claim (SPEC §34).

**Research implication.**
- None for the hypotheses. This removes eight analysis degrees of freedom from the final evaluation.
- No novelty or causal claim.

**Experiment needed.** None. No EXP-0001 episode has run.

**Compatibility impact.**
- The locked manifest gains `evaluation.statistics.final_rule`. The H1, H2 and seed-consistency strings are unchanged, and `check_artifacts.py` still checks them exactly.
- The pre-registration's expectations and caps are unchanged. Its immutability clause applies once execution starts, and it has not started.
- D7 lowers the planned final-evaluation episode count (design §7: "3 arms × (134 + 40 + 20) × 3 mining seeds") by 388 (2 × (134 + 40 + 20)). The hard caps are unchanged.
- A new module and its tests; one import line in `scripts/ci/smoke_installed.py`; the manifest line; the takeover note. No existing behaviour changes.
