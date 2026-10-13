# EXP-0001 methodology and validity review (independent, complementary to design §6)

Author: independent Claude researcher subagent (third agent, read-only). Date: 2026-10-12.
Scope: read-only review. No repo edits, no GitHub writes. SPEC §34 anti-overclaim applies.

**Trees read.** `main` @ `0ce0a36` (/home/user/heritagent). PR #28 branch `origin/issue-12-cig-gate`. The task brief named head `4ca5517`. During this session the remote ref showed `7886852` and then `c6ce4c1` ("fix(inheritance): final 5-item correctness pass on gate evidence", 2026-10-13 00:32 +0300). ADR-0002 and the manifests did not change between `4ca5517` and `c6ce4c1` (checked with `git diff 4ca5517 origin/issue-12-cig-gate -- research/decisions/ experiments/`, empty). Code citations marked "PR" are at `c6ce4c1`.

**GitHub read.** Issues #6, #12 (body and 6 comments), #13, #20 (54 comments, summarized), #22, and the full issue list.

**Evidence labels.** **V** = I read it in a repo file, an issue, or a script output from this session. **M** = output of an explicit model with stated assumptions (not data). **U** = not verified. Any change to a locked or pre-registered element is marked **[requires ADR]**. Such a change is a proposal only and is never presented as decided.

**Sibling outputs this review builds on (not repeated here).**
- `out/research/methodology/ALFWorld-baselines.md`: ALFWorld split counts, Qwen2.5-7B anchors, H2 power on n=134, T_reg feasibility, stage-2 pass region.
- `research/reviews/repro/m0-stats/r01.out`: shows tau_c is non-binding.
- `out/research/reviews/test-portability.md`: the test suite collects fully only on Python 3.14; no CI.

**Scripts for this review.** `out/research/methodology/scripts/validity_checks.py`, output in `validity_checks.out`. Run with `venv314/bin/python -I -B`. The script imports the repo's locked `inheritance/bootstrap.py`. It runs in about 6 s.

> **Verification status (independent adversarial verification, 2026-10-13).** The full record is in `research/methodology/VERIFICATION-2026-10-13.md`. Verifiers checked code at merged `main` @ `6014bc7` (PR #28 merged). Line citations marked "PR" below are at `c6ce4c1` and may have shifted.
> - **Confirmed:** finding 13's budget item I5 (8,526-episode worst case; re-rated P2). Finding 9's mechanics: the τ_r = 0.05 cell behaves like τ_r = 0.03, tau_c is inert, and the stage-2 pass region is 4/0, 7/1, 9/2, 11/3. I12 (out-of-scope Δ ≥ −0.05 is the same as net ≥ 0 at n = 10).
> - **Confirmed with correction** (holds but overstated; corrected inline): finding 1 (stays P0); findings 2–6 (re-rated P0 → P1); finding 7 (stays P1). Also the "≥ 2/3 seeds" null probability in finding 10 (≈ 0.44–0.47 once ties are counted) and boundary robustness in section 3.8.
> - **Refuted:** no headline finding. Several sub-claims were narrowed or withdrawn: "satisfied by construction" (2.3), "cannot be represented" (2.5), "hidden" cap (2.4), "contradicts the design rule" (2.6), "setup abort" for one candidate (2.7), and "expected arm C ≈ G0" as a general statement (2.2).
> - Each corrected passage is marked "[corrected after independent verification 2026-10-13]". After verification, only finding 1 remains P0.

---

## 0. Executive summary (ranked)

| # | Pri | Finding (one line) | Class | ADR? |
|---|---|---|---|---|
| 1 | P0 | The Trait Miner accepts only success/failure pairs with identical `task` text and identical benchmark/split/environment provenance. The runtime hard-codes temperature 0, and the design runs **one** G0 episode per T_mine instance per seed. So the specified protocol gives no same-instance pair. Pairs could arise only by accident: identical task text across instances (provenance has no instance id), or cross-seed pairing. Cross-instance pairs would be few and confounded by scene layout. Cross-seed pairs break seed independence and are identical at temp 0. The lifetime sampling and pairing policy was explicitly deferred to #13 but never pre-registered. This is a critical-path design gap; the code is not broken. [corrected after independent verification 2026-10-13] | construct / critical path | yes |
| 2 | P1 | The gate is a 5-stage conjunction of low-power tests. Under an explicit task-independence model (not data), a trait with a +20pp true effect is promoted with probability of about 0.002–0.2 at a 2.5–5% noise loss rate, or about 0.14 with no losses. The result hinges on the unmeasured T_reg flip rate f and on stage 2. P(\|C\| = 0) is about 0.2–0.75, so an empty arm C is plausible, not certain. If it happens, H2 is formally "supported" (veto beats inherit-all), which is not evidence for causal selection. H1 and the SPEC §29 criterion would fail and expose this. [corrected after independent verification 2026-10-13] | statistical / construct | yes |
| 3 | P1 | T_reg is the stage-4 **selection** set for every arm-C trait **and** an H2 primary regression outcome set. This biases the H2 regression leg in C's favour, so that leg is not a held-out test. It is not satisfied by construction in general, because stage 4 bounds each trait alone, not the multi-trait G1. The T_eval leg is unaffected. T_reg is probably in-sample for the miner (depending on the pairing policy). It has no hash or post-G0 freeze point, no selection rule, and no fallback. [corrected after independent verification 2026-10-13] | construct / statistical | yes |
| 4 | P1 | The G0 seed planner gene caps every episode at **8 loop steps** (`max_plan_steps: 8`), i.e. at most 8 environment actions. This is an unreasoned placeholder from #4 test fixtures with no pre-registered value. It is unregistered, not hidden: it is logged in events and was required on #7. It is identical in all arms. The planner schema caps the value at 16, so a 50-step protocol needs a schema change. The cap probably makes T_reg (≥40/60 solved) infeasible (prediction). It is inconsistent with "~30 LLM calls/episode", erring toward over-budgeting. It departs from the 50-step limit stated for the SAGA and verl-agent/GiGPO anchors. [corrected after independent verification 2026-10-13] | construct / external / reproducibility | yes |
| 5 | P1 | Germline `genome/0.1` has no **honest** origin label for arm B's G1 genes (ungated or gate-rejected candidates). The schema would accept them as `seed`, or as `assimilation` with any `cig_record` string, because the loader never checks the record's verdict or existence. So arm B can be built only with false provenance, and the "assimilation = CIG-validated" invariant is unenforced for arm C too. On rendering, the documents are ambiguous: ADR-0001 R4 and eag-v0-schema.md describe a `project_skillrl` arm, whereas design:39 says all arms use the same payload. In the code, `project_skillrl` has no runtime caller, so B and C differ only in selection unless #13 wires it in. [corrected after independent verification 2026-10-13] | construct / critical path | yes |
| 6 | P1 | Stage-2 ablation runs unconditionally on a 20-task stratified pool, and its minimum pass is +20pp (4 wins and 0 losses). Under the reviewer's proportional allocation, a trait scoped only to Look, Heat or Cool **cannot** pass (at most 2–3 in-scope tasks). Pick/Clean/Pick2 specialists pass only by fixing all 4 in-family tasks with no losses. Traits scoped to 2 or more families are not blocked in principle. Stage 2 dilutes specialist effects rather than literally requiring improvement on unrelated tasks, which is a purpose-level tension with the design:49 rule (scoped to stage 3). The fix is a pre-registration choice, not a code bug. [corrected after independent verification 2026-10-13] | construct / purpose-level inconsistency | yes |
| 7 | P1 | A candidate whose scope covers the whole family universe has no out-of-scope family. The merged #12 code then raises a plain `ValueError`, and with no per-candidate catch this aborts the **entire gate batch** for that seed. This is deliberate under the #12 handoff ("reject an impossible allocation"). The gap is that no universal-scope rule is pre-registered. The miner ranks by `estimated_generality`, which may favour broad scopes (unmeasured inference). [corrected after independent verification 2026-10-13] | construct / critical path | yes |
| 8 | P1 | C−B confounds *which* skills with *how many* skills and prompt length. There is no size-matched random-subset control, so a C>B result cannot be attributed to causal validation. | construct | yes |
| 9 | P1 | The pre-registered threshold-sensitivity analysis is nearly degenerate. tau_c is non-binding (sibling r01). A float bug makes the tau_r=0.05 cell behave like tau_r=0.03. At most 2 distinct compositions exist per seed. | statistical / implementation | fix: no; grid: yes |
| 10 | P1 | Seeds. Arm A is the same deterministic G0 in all 3 seeds (388 duplicate episodes, which is pseudo-replication if pooled). Seed pooling for the H2 CI is unspecified. The "C − B > 0 in ≥2/3 seeds" rule has probability of about 0.5 under the null (≈ 0.44–0.47 at n = 134 once ties are counted). [corrected after independent verification 2026-10-13] | statistical | yes |
| 11 | P1 | No pre-registered measure of the gate's *predictive validity* (gate verdict vs held-out per-trait effect). This is the first half of SPEC §27's prototype question. | construct | yes |
| 12 | P1 | The teacher (miner) model is unspecified. SkillRL uses o3. Using a stronger or closed teacher changes what H1 means and conflicts with SPEC §21. | construct / reproducibility | yes |
| 13 | P1 (I5 budget item: P2) | Internal inconsistencies across SPEC / ADR-0002 / design / YAML / hypotheses.md (section 1: 20 items). Examples: the interaction stage runs per seed, so the worst case is 8,526 episodes (about 7.2k if the shared "neither" baseline is cached), not the stated 5.8–6.7k; three different H2 success definitions; stale status strings. [corrected after independent verification 2026-10-13] | consistency | some |
| 14 | P1 | SPEC §24 gaps: no G0/G1 genome_id (#22), T_reg unhashed, no vLLM/ALFWorld/data versions, no budgets, no lockfile, no CI, Python-3.14-only test collection. | reproducibility | mostly no |
| 15 | P2 | "Expression check" means "skill block was in the prompt", not "skill was used". Interaction is both-vs-neither with a 1-net-loss trigger. No joint-set check exists. The adaptation-cost metric is undefined at temp 0. Clustering by scene/family is ignored. valid_seen is unused. A contamination probe is missing. | construct / statistical / external | mixed |

Critical path to #13 is in section 6. In short: three new ADRs (lifetime protocol, arm-B representation, G0 budget / T_reg), plus #22, plus two untracked engineering items (ALFWorld adapter, split-manifest builder), plus a pilot on a disjoint train pool, all **before** #6 can be filled honestly. After independent verification, only finding 1 remains a P0 blocker, because it prevents candidate production. Findings 2–7 are P1 pre-registration fixes. They are not hard blockers of engineering work, but each must still be resolved by ADR before #6 is filled. [corrected after independent verification 2026-10-13]

---

## 1. Internal inconsistencies (exact quotes, locations)

Paths are relative to the repo root. "design" = `experiments/manifests/EXP-0001-design.md`; "YAML" = `EXP-0001-minimal-cig.yaml`; "ADR2" = `research/decisions/ADR-0002-exp-0001-design.md`.

| # | Item | Quote A (location) | Quote B (location) | Comment |
|---|---|---|---|---|
| I1 | T_gate size | ADR2:26 (round-1 table) "T_gate (~60) organized into sub-pools (ablation 20 / generalization reserve 20 / interaction 10 / replay reserve 10)" | design:19 "`T_gate` (~114 tasks …)"; YAML:57 `target_size: "~114 = ablation 20 + interaction 10 + banks ~84"` | Superseded by ADR2:43 (final R1), but the round-1 row has no "superseded" marker. ADR2:9 "Proposed change" still describes "sub-pools" and never mentions banks. V |
| I2 | Pool arithmetic | YAML:49 `task_families: ~6`; design:17 "ALFWorld's ~6 task families" | YAML:57 "banks ~84" (= 6×(4+10)) | 84 holds only for exactly 6 families at the exact minimums. ALFWorld has exactly six named families (sibling Table 1). The family vocabulary is not pinned; tests use `heat_and_place`, ALFWorld uses `pick_heat_then_place_in_recep`. V |
| I3 | "Disjoint" splits | design:17 "four disjoint splits"; ADR2:9 "3-arm, 4-split" | design:26 "`T_reg` (40 tasks, subset of T_mine that G0 solved)" | T_reg ⊂ T_mine, so the splits are not disjoint. V |
| I4 | Split hashing | design:27 "Splits are content-hashed and committed with the manifest" | YAML:66 `T_reg:  {source: subset-of-T_mine-solved-by-G0, size: 40}` (no `hash`, no stratification, no selection rule) | T_reg depends on outcomes, so it cannot be hashed at lock time. Its **selection rule** must be pre-registered instead. V |
| I5 | Budget, interaction | design:85 "The interaction stage adds ≤ ~900 in the worst case. Total ≈ 5.8–6.7k"; YAML:109–110 `episodes_est: 6000` … "interaction worst case ~900" | Interaction runs per mining seed: each seed has its own frozen set (45 pairs × 10 × 2 = 900 **per seed**) | Worst case = 180 + 3,900 + 1,746 + 2,700 = **8,526** episodes (`validity_checks.out` §5). V. This is the upper bound under ADR-0002/A1 accounting. If the shared "neither" baseline is cached, the total is still about 7.2k. Only a "once globally" reading of A1 stays near 6.7k, and that reading is inconsistent with the design's per-seed multipliers for every other stage. The "≈ 180–200k calls" figure inherits the error. Severity P2. [corrected after independent verification 2026-10-13] |
| I6 | LLM calls/episode | design:85 "× ~30 LLM calls ≈ 180–200k local-7B calls" | `runtime/seed_g0.py:17` `PLANNER_PAYLOAD = {"style": "react", "max_plan_steps": 8}`; `runtime/loop.py:151` `effective_max_steps = min(budgets.max_steps, planner.max_plan_steps)` | With an 8-step cap, an episode has ≤ 8 generate calls plus parse retries. Either the budget or the G0 cap is wrong. V. Under the cap the estimate errs toward over-budgeting compute, not under-budgeting. [corrected after independent verification 2026-10-13] |
| I7 | H2 operationalization | `research/hypotheses/hypotheses.md:8` "Arm C − arm B ≥ +5pp on T_eval AND regression rate(C) < regression rate(B)" (no CI) | design:66 / YAML:100 "… CI excl. 0, AND reg(C) < reg(B), CI excl. 0" | The two documents disagree on whether CIs are required. V |
| I8 | M0 success criterion | SPEC:1227 "G1'in held-out performance'ında reproducible pozitif improvement göstermesi ve unrestricted-skill baseline'a göre daha düşük regression" (positive held-out improvement + lower regression than B; no C−B margin) | design:66 H2 (C−B ≥ +5pp …); issue #13 body cites SPEC §29 as "Success criterion" | Two different milestone success definitions. "Reproducible" is undefined. V |
| I9 | H5 | hypotheses.md:11 "count of 'catastrophic' (>10pp) per-task drops C < B" | Absent from design §4/§5 and YAML:96 `metrics` | ">10pp per-task drop" has no meaning for binary per-task outcomes. V |
| I10 | Metric/split mapping | hypotheses.md:18 "EXP manifests must state which are computed and from which splits" | YAML:96 `metrics: [heldout_success, regression_rate, genome_size_genes, genome_size_tokens, descendant_adaptation_cost]` (no splits; no Inheritance Precision / Assimilation Gain / Genome Efficiency from SPEC §18) | V |
| I11 | Expression check | design:47 "trait must fire in every successful with-trait episode (expression check, not just presence)" | `runtime/skills.py:16-17` "constitutive skills always express"; `inheritance/replay.py:11` "target gene appears in skills_expressed" | For constitutive skills, which #12 says M0 candidates "normally" are, the check is exactly presence. V |
| I12 | Out-of-scope tolerance | design:51 "Out-of-scope … Pass: Δsuccess ≥ −0.05 — no harmful spillover" | n=10, so Δ ∈ {…, −0.1, 0, …} | Δ ≥ −0.05 is equivalent to wins ≥ losses, which is **identical** to the in-scope rule Δ ≥ 0. The stated tolerance does not exist. V (`validity_checks.out` §1) |
| I13 | tau_c meaning | ADR2:9 "τ_c = 5pp contribution with bootstrap LB > 0"; design:71 grid τ_c ∈ {0.03, 0.05, 0.08} | Repo bootstrap: minimum passing point estimate = **0.20**; 4 wins/0 losses, 7/1, 9/2, 11/3 (`validity_checks.out` §3; matches sibling r01) | tau_c never binds, so the tau_c sensitivity grid cannot change any decision. V |
| I14 | Sensitivity reach | design:71 "Report how arm-C composition **and T_eval outcomes** would change under each" | Issue #12 handoff §15 "ZERO T_eval access"; design §7 budget has no line for alternative-composition T_eval runs | The T_eval part needs extra arm-C variants on T_eval. Its budget and its reporting rule are unspecified. V |
| I15 | Mining pairs | design:43 "LLM differential distillation over success/failure pairs"; design:85 "Mining: 3 seeds × 60 tasks = 180 episodes" | `traits/miner/inputs.py:3-4` "ONE complete verified successful trajectory and ONE complete verified failed trajectory of the SAME task"; `runtime/loop.py:174` `GenerationSettings(temperature=0.0, …)` (hard-coded, no parameter) | One deterministic episode per task instance means zero same-instance pairs (finding 1). The pair check compares only the `task` string plus benchmark/split/environment provenance, so cross-instance pairs with identical task text are accepted. V. [corrected after independent verification 2026-10-13] |
| I16 | "Mining temperature" | YAML:20 `mining_temperature: 0.7`; design:79 "mining-phase temp 0.7 variance is covered by 3 independent mining seeds" | `traits/miner/miner.py:82` (`MiningSettings.temperature = 0.7`, applies to the teacher call only) | The temperature of the G0 *lifetime* is not stated anywhere. In code it is 0. So the 3 seeds share identical lifetimes and differ only in teacher sampling. V |
| I17 | Arm A across seeds | design:59 "each arm's G1 runs T_eval (across 3 independent mining seeds …)"; design:85 "3 arms × (134 + 40 + 20) × 3 mining seeds" | design:54 "Duplicate deterministic executions are never treated as independent evidence" | Arm A = G0 in every seed, so the runs are 3 identical deterministic copies (388 duplicate episodes). V |
| I18 | Status strings | ADR2:3 "Status: REVISED — critic round 1 applied, re-review requested"; ADR2:19 "all five points … resolved below; no unresolved questions remain"; design:3 "REVISED after critic round 1 … re-review requested"; YAML:2-3 "REVISED after critic round 1 and round 2 … See … (sections 2-5, 9-10)"; YAML:10 `status: design-revised-re-review-requested` | YAML:12 `locked: true`; issue #20 (2026-10-02) "PR #2 merged: ADR-0002 / EXP-0001 design"; ADR2 has round-2, final-round and A1 sections; YAML omits design §11 | Lock state is ambiguous in the documents themselves. V |
| I19 | K semantics | design:43 "Cap: ≤ 10 candidates per run" | design:39 "the top **K = 10** are selected" | Behaviour when fewer than 10 candidates are mined is unspecified (B and C would then have < 10). V |
| I20 | Amendment naming | ADR-0001 "Amendment A1 — somatic-only `origin="acquired"`" | ADR2 (PR) "Amendment A1 — M0 pairwise interaction contrast" | Two different "Amendment A1"s. Cross-references are ambiguous; use qualified ids (ADR-0002/A1). V |

SPEC-level divergences. These are acceptable for M0 but must be stated in claims:
- SPEC:338 defines contribution as leave-one-out within the genome, F(G∪t) − F(G∖t). EXP-0001 measures add-one-to-G0, and the final arm-C set is never jointly evaluated (finding 15).
- SPEC:396 requires "Generalization(t) > τ_g" (strict, positive). The design's in-scope rule is Δ ≥ 0, which is non-inferiority and is passed by a no-op trait. Per SPEC §34 ("general"), call the stage "unseen-instance non-inferiority".
- SPEC:689 lists "→ attribution" in the HeritAgent pipeline, but EXP-0001 has no attribution stage. H2 therefore tests a 4-stage gate, not the SPEC §17 pipeline.
- SPEC:801 says the first paper should cover at least 3 capability classes. EXP-0001 covers one (section 4).

---

## 2. Construct validity

### 2.1 [P0] The lifetime protocol cannot produce the miner's input (finding 1)

- **Evidence (V).** The miner contract is "MiningPair binds ONE complete verified successful trajectory and ONE complete verified failed trajectory of the SAME task" (`traits/miner/inputs.py:3-4`). The miner prompt says "two complete runs of the SAME task, one successful and one failed" (`traits/miner/prompts/differential-skill-v0.txt`). `run_agent` takes no temperature or seed argument and builds `GenerationSettings(temperature=0.0, …)` internally (`runtime/loop.py:140,174`). The design budgets one G0 episode per T_mine task per seed (design:85).
- **Evidence added by verification (V, `main` @ `6014bc7`).** `MiningPair.__post_init__` rejects a pair only when the `task` strings differ, and `_require_matching_context` compares only benchmark/split/environment provenance. Provenance carries no game-instance id. So two different ALFWorld instances with identical task text are accepted as a pair; the verifier reproduced this. The `MiningBatch` docstring says "Pairing policy is experiment-runner responsibility (#13), not the miner's". The Critic's #9 handoff asked only for the same `task_family` and the same benchmark/environment context. It is ambiguous whether `mining_temperature: 0.7` (YAML:20, design:79) is meant to apply to the lifetime; in code it applies only to the teacher.
- **Impact.** With deterministic G0 at temp 0 (design:54), each task instance yields exactly one outcome, so the specified protocol gives the miner no same-instance success/failure pair. Pairs could arise only by accident, in two ways. (a) Different instances share identical task text. Whether this can happen depends on what the unbuilt ALFWorld adapter passes as `task`, and such pairs would be few and confounded by scene layout. (b) Pairing across the 3 mining seeds. That breaks their independence, and at temp 0 it yields identical trajectories anyway. The pairing and lifetime policy was explicitly deferred to #13 but never pre-registered. This is a critical-path design gap, not a code defect, and it stays P0. [corrected after independent verification 2026-10-13] Any fix (k stochastic rollouts per task at temperature T, or cross-task pairing) changes four things: (a) the mining budget; (b) what "independent mining seeds" means (it would then cover lifetime variance as well as teacher variance); (c) the definition of "G0 solved" for T_reg, since a success sampled at T>0 does not imply a temp-0 success, which creates regression-to-the-mean in stage 4 and in H2's regression outcome; (d) the H3 "lifetime" interpretation.
- **Recommendation [requires ADR].** Pre-register the lifetime protocol. It covers rollouts per task k, lifetime temperature, seed derivation and the pairing rule. The pairing unit (same instance vs same family) must be stated explicitly, and instance identity must be added to the pair check. [corrected after independent verification 2026-10-13] Define "G0-solved" for T_reg as **solved by G0 at eval settings (temp 0)** in a separate deterministic pass, not as solved in some lifetime rollout. Update the §7 budget accordingly.

### 2.2 [P1] Gate power, and H2 can be supported by an empty arm C (finding 2)

**Model (M; assumptions in `validity_checks.py` §4).**
- Each task falls independently into one of four paired cells.
- G0 success rate p. Discordance d = P(win) + P(loss). Net effect e = P(win) − P(loss).
- Out-of-scope tasks have e = 0 and the same d.
- T_reg tasks are G0 successes, so the per-task flip-to-fail probability is f = loss / p.
- Stages are treated as independent because they use disjoint task sets.

Selected rows at p = 0.40. The p = 0.667 table, the minimum p at which T_reg is feasible, is qualitatively the same; see `validity_checks.out`.

| e (true) | d | replay | stage 2 | in ≥0 | out ≥0 | T_reg (≤1/40) | P(promote) | E[\|C\|] of 10 |
|---|---|---|---|---|---|---|---|---|
| 0 (null) | 0.20 | 0.047 | 0.022 | 0.647 | 0.647 | 0.000 | ≈ 0 | 0 |
| +0.10 | 0.10 (no losses) | 0.069 | 0.133 | 1.000 | 0.725 | 1.000 | 0.0066 | 0.07 |
| +0.20 | 0.20 (no losses) | 0.237 | 0.589 | 1.000 | 0.647 | 1.000 | 0.090 | 0.90 |
| +0.20 | 0.30 | 0.291 | 0.377 | 0.946 | 0.616 | 0.032 | 0.0021 | 0.02 |
| +0.30 | 0.30 (no losses) | 0.447 | 0.893 | 1.000 | 0.616 | 1.000 | 0.246 | 2.46 |
| +0.30 | 0.40 | 0.480 | 0.603 | 0.978 | 0.599 | 0.032 | 0.0055 | 0.05 |

T_reg pass probability by per-task flip rate f: f = 0.02 → 0.81; 0.05 → 0.40; 0.10 → 0.08; 0.15 → 0.01.

**Model consistency note (M, from independent verification).** The table's "no losses" rows still give out-of-scope tasks the row's d, which is a 10% loss rate at e = d = 0.2. So the 0.09 upper figure mixes zero noise in scope with nonzero noise out of scope. A model with one consistent noise loss rate l on every pool (the verifier's `consistent_model.py`, reusing the repo bootstrap) gives the following for e = +0.2. With l = 0: P(promote) ≈ 0.14 (0.38 if the effect is concentrated in the replay family). With l = 0.025–0.05: about 0.002–0.2. The low end requires T_reg flips spread evenly over G0 successes (f = l/p, about 6–12% per task); with f = 0.02 it is 0.06–0.20. For e = +0.3, P ranges from 0.007 to 0.72. The binding stages are T_reg and stage 2, and the deciding quantity f is unmeasured. For 10 i.i.d. candidates, P(\|C\| = 0) is 0.74 at P = 0.03, 0.43 at P = 0.08 and 0.22 at P = 0.14. [corrected after independent verification 2026-10-13]

**Readings.**
- (a) False promotion is negligible. Per-candidate null P(promote) ≤ 2e-6, and FWER over 10 candidates ≤ 2e-5. Multiplicity across 10 × 5 tests is **not** the threat, because the gate is an intersection-union conjunction.
- (b) False **rejection** compounds. Stage 4 (≤ 1 failure of 40) rejects nearly every trait whose per-task flip rate on G0-solved tasks reaches about 0.1 (pass 0.08 at f = 0.10), though it passes 0.81 at f = 0.02. Losses are what a 7B model at temp 0 produces under any prompt perturbation, but their rate f is unmeasured (see the model consistency note above). [corrected after independent verification 2026-10-13] Out-of-scope (wins ≥ losses on 10 tasks) rejects about 35–40% of genuinely harmless traits.
- (c) Discordance is plausibly high. Published memory-content changes on the same 134 games moved Qwen2.5-7B by ±11pp (SAGA, via the sibling, V there). That regime is the model's pessimistic corner, where the per-task T_reg flip rate is f ≳ 6%, and there |C| = 0 is the likely outcome in every seed. Across the modelled range, E[|C|] per seed for +20pp candidates is about 0–2, and P(|C| = 0) is about 0.2–0.75. An empty arm C is therefore a plausible outcome, not a certainty. [corrected after independent verification 2026-10-13]
- (d) H2 as written (design:66 / YAML:100) has no non-degeneracy clause; the verifier confirmed this from the documents. If |C| = 0, then C = G0 = A and reg(C) = 0, so H2 is formally "supported" whenever B is significantly harmful (C − B ≥ 5pp with a CI excluding 0, and reg(B) with a CI excluding 0). That outcome supports "veto beats inherit-everything", not "causal validation selects good traits". Design §6.5 pre-declares the B ≈ C case as a finding but not the C ≈ A case. Mitigation: H1 (C − A > 0, CI excluding 0) and the SPEC §29 / #13 M0 success criterion would fail when C = A, and the per-seed CIG records would show |C| = 0. The result would therefore not pass silently as project success, but the H2 label itself would be an overclaim. [corrected after independent verification 2026-10-13]

**Caveats (U).** Effects are assumed homogeneous. T_reg tasks are in-sample for the miner (section 2.3), which may lower f below the out-of-sample value. The source task in replay is in-sample, which raises replay pass rates.

**Recommendations.**
- **[requires ADR]** Add an H2 non-degeneracy condition: H2 counts as supported only if arm C is non-empty in ≥ 2/3 seeds. Pre-declare "C = A, C > B" as a separate "gate-as-veto" finding. Severity after verification: P1, a pre-run fix that must land before #6. [corrected after independent verification 2026-10-13]
- **[no ADR; pre-run]** Run a **calibration pilot** on a train pool **disjoint from every EXP-0001 split** (about 3,300 train games available; sibling §1.3). Measure G0 success, the null discordance d0 and the flip rate f0 using 3–5 length-matched placebo skill texts. Feed these into #6's "expected number of candidates passing each stage". If the pilot shows E[|C|] ≈ 0, revisit tau_r, the replay rule and the stage-2 n via ADR **before** the run, never after.

### 2.3 [P1] T_reg is a selection set and an outcome set at once (finding 3)

- **Evidence (V).**
  - Issue #12 handoff §5: "T_reg is the frozen 40-task subset of T_mine that G0 already solved … Do **not** rerun the without-trait baseline … `baseline_success = 1.0`". PR `inheritance/regression.py:231-232` `drop = 1.0 - successes / len(allocation.tasks)`; `passed = drop <= TAU_R`.
  - design:59 G1 evaluation "runs T_eval … + T_reg. Metrics: … regression rate on T_reg (primary)".
  - design:66 H2 "regression rate(C) < regression rate(B) with CI excluding 0".
- **Impact.**
  - (i) **Circularity (selection on the outcome set).** Every arm-C trait was individually filtered to ≤ 1/40 failures on these 40 tasks. Arm B carries the same traits unfiltered, including those that failed on these exact tasks. This biases the reg(C) < reg(B) comparison in C's favour, so the regression leg of H2 is not a held-out test of the gate. It is **not** satisfied by construction in general. Stage 4 bounds each trait alone, not the multi-trait composite that G1 runs. The bound is tight only when a single trait is promoted and the G1 prompt reproduces the gate overlay, which has not been verified. An additive-flip illustration by the verifier (K = 10, stage 4 as the only filter; an assumption, U) gives composite T_reg reg(C) of about 0.07–0.09. In that illustration the in-sample inflation of the B − C gap, beyond what fewer traits explain, is about 6–12pp. H2's T_eval leg is held-out and unaffected. That T_reg consists of G0-solved tasks is itself spec-mandated (SPEC §18 Regression Rate). The defect is its reuse as the selection set. [corrected after independent verification 2026-10-13]
  - (ii) **Probably in-sample.** T_reg ⊂ T_mine, and the miner's success-side trajectories come from G0-solved T_mine tasks, so traits may contain instance-specific procedures. This would understate regression on new tasks. Whether a given T_reg task actually enters mining depends on the pairing policy, which is unspecified (#13; section 2.1). [corrected after independent verification 2026-10-13]
  - (iii) **Baseline = 1 by construction.** Under deterministic decoding, stage 4 measures *perturbation sensitivity* of fragile G0 successes, not semantic harm (section 2.2).
  - (iv) **Feasibility.** The sibling showed that ≥ 40/60 G0 successes needs p ≥ 0.667. Anchors are 15–46%, and the 8-step cap (section 2.4) lowers this further.
- **Recommendation [requires ADR].** Keep T_reg as the gate-internal stage-4 set if desired. Move H2's *primary* regression outcome to a held-out definition that costs no extra episodes: **regression rate(X) = share of arm-A-solved T_eval tasks that arm X fails**. This is paired, held-out and never touched by the gate. Pre-register a deterministic T_reg selection rule (stratified, canonical task-id order) and a fallback for when G0 solves fewer than 40. Because T_reg depends on G0 outcomes and cannot be hashed at lock time, also add a post-G0 freeze point with a committed hash, analogous to the candidate-set freeze. Severity after verification: P1. This is a pre-run ADR fix that contaminates only the regression leg of H2, and it must be resolved before #6 locks the pre-registration. [corrected after independent verification 2026-10-13]

### 2.4 [P1] The 8-step episode cap is an unregistered pre-registration element (finding 4)

- **Evidence (V).** `runtime/seed_g0.py:17` `"max_plan_steps": 8`. `runtime/loop.py:150-151` "heritable planner policy limit vs external loop budget … `effective_max_steps = min(budgets.max_steps, planner.max_plan_steps)`". The manifest has no `max_steps`, `max_total_tokens` or `max_tokens_per_request` field. design:37 promises "Equal **evaluation budget** (max steps per task)" without a value.
- **Evidence corrected and added by verification (V, `main` @ `6014bc7`).**
  - A step is one planner decision (one loop iteration), not one tool action. A Final Answer, a rejected or unknown tool call, or an invalid-argument call each uses up a step. Parse retries do not. So the bound is **at most** 8 environment actions.
  - With `Budgets(max_steps=50)` the run still ends at step 8, with reason `planner_max_plan_steps`.
  - The planner payload schema caps `max_plan_steps` at 16, so a 50-step protocol needs a schema change plus a new planner gene version.
  - The value 8 originates in #4 test fixtures (`de7ef59`, copied into `seed_g0` in `353645b`). No ADR, issue, SPEC or design text justifies it.
  - It is **unregistered, not hidden**. It is a versioned planner gene, it is logged in `run_started` and `budget_exhausted` events, and it was an explicit Codex requirement on #7.
  - It is identical in arms A, B and C, so it is not an arm confound.
  - The 50-step limit is stated for the SAGA 45.5% anchor and for verl-agent/GiGPO. The ExpRAG 29.9% row does not state one (sibling §2–3).
  [corrected after independent verification 2026-10-13]
- **Impact.** All harm statements below are predictions; no ALFWorld adapter or run exists yet. [corrected after independent verification 2026-10-13]
  - Floor effects. Pick2's minimum plan with known object locations is about 8 actions (go/take/go/put ×2), leaving no slack for search. This is my own derivation from ALFWorld action semantics (U).
  - T_reg is likely infeasible (prediction).
  - The regime is step-starved, which inflates the value of efficiency skills.
  - Results are not directly comparable to the anchors that state a 50-step limit (external validity).
  - The §7 call budget is inconsistent with the cap (I6). It errs toward over-budgeting compute.
  - The genome's planner gene, not `Budgets`, sets the eval budget, so "equal budget" is really a property of the shared planner gene.
- **Recommendation [requires ADR].** Pre-register the G0 planner parameters, which means a new planner gene version if changed (ADR-0001 D6 version-increment rule). A 50-step protocol also needs a schema maximum above 16. This is a cheap pre-run fix; severity after verification is P1. [corrected after independent verification 2026-10-13] Pre-register `Budgets.max_steps`, `max_total_retries`, `max_tokens_per_request` and `max_total_tokens` (recommend `None`, or prompt-excluded). Report the `budget_exhausted` rate per arm as a diagnostic. If max_total_tokens is ever set, arm B's longer prompts exhaust it first.

### 2.5 [P1] Arm-B representation and rendering parity (finding 5)

- **Evidence (V).** `genome/schema/eag-0.1.genome.schema.json:66` `"origin": {"enum": ["seed", "mutation", "recombination", "assimilation"]}`. Lines 80–87: assimilation requires `source_trajectory` and `cig_record`. ADR-0001 A1: "the germline `genome/0.1` schema still rejects `acquired`". ADR-0001 R4: "projection adapter emits strict SkillRL skills for the baseline arm". `genome/validation/projection.py:55-56` `project_skillrl` "additionally strips procedure[]". design:39 "Runtime prompt assembly consumes only the projected payload (`name`, `principle`, `when_to_apply`, `procedure`)".
- **Evidence added by verification (V, `main` @ `6014bc7`, verifier repro).**
  - Representation: a one-skill G1 genome is **accepted** with `origin="seed"` (no CIG) and with `origin="assimilation"` plus `cig_record="CIG-REJECTED-0001"`. It is rejected with `acquired`, with `unrestricted_inheritance` (not in the enum), and with assimilation lacking a `cig_record`. The schema types `cig_record` only as a string, and the loader never checks the record's verdict or existence.
  - Rendering: `project_skillrl` has no runtime caller; grep finds only its definition, the export and one test. `compile_runtime_config` → `project_for_runtime` → `project_skill`, the overlays (`inheritance/overlay.py`, `interaction.py`) and `skill_block` all keep `procedure[]`.
  - Docs: ADR-0001 R4 and `architecture/eag-v0-schema.md:21/:245` say the stripping adapter serves "the baseline arm". Design:39 and the manifest say all arms assemble from {name, principle, when_to_apply, procedure}.
- **Impact** (two documentation-level design gaps; no code defect). [corrected after independent verification 2026-10-13]
  - (i) Arm B's G1 genome cannot be built **honestly**. It can be built and validated today only with provenance that does not describe what happened: `seed`, or `assimilation` with a reject-verdict or arbitrary CIG record. This undermines lineage auditability and the "assimilation = CIG-validated" invariant, which the schema does not enforce for arm C either. The missing genome_id is a #22 issue common to all arms, not specific to B. The genome size metric (H5) does not depend on the origin label.
  - (ii) The documents are ambiguous on rendering. If #13 wires `project_skillrl` into arm B while C uses `project_skill`, then B − C mixes selection with skill format. Format alone can move Qwen2.5-7B from 83.6% to 1.5% (ExpRAG Table 17, via the sibling). With the current code paths, B and C differ only in selection.
- **Recommendation [requires ADR, before #13 G1 assembly].** Add an explicit germline origin (e.g., `unrestricted_inheritance`) or a typed arm-B genome variant whose provenance records "no gate". Add validation that `assimilation` cites a promote-verdict CIG record. [corrected after independent verification 2026-10-13] State that **all arms use the identical projection and renderer** (`project_skill` → `skill_block`). Keep `project_skillrl` for a separately named SkillRL-format arm only. Add a pre-run rendering smoke test of the 10-skill arm-B prompt (length, format, context fit).

### 2.6 [P1] Stage 2 structurally rejects narrow-scope specialists (finding 6)

- **Evidence.** Proportional stratification of 20 ablation tasks over the ALFWorld train family counts (790/308/650/459/533/813; sibling Table 1, V there) gives Pick 4, Look 2, Clean 4, Heat 3, Cool 3, Pick2 4 (largest remainder; my computation, reproduced by the verifier). The design says only "stratified". `AblationAllocation` (`inheritance/ablation.py:49-87`) accepts any `frozen_strata` that sums to 20, so these exact quotas are a pre-registration choice, not code. [corrected after independent verification 2026-10-13] The stage-2 pass region needs at least 4 wins with 0 losses, i.e. a point estimate of at least 0.20 (V, repo bootstrap). A single-family specialist can win at most its in-family tasks: Look 0.10, Heat or Cool 0.15. These can never pass. Pick, Clean and Pick2 need G0 to fail all 4 in-family tasks and the trait to fix all 4 with zero losses elsewhere.
- **Conflict (purpose-level).** design:49 says "a specialist trait must not be required to improve unrelated task types", which motivated the stage-3 scope split in critic round 1 R1. That rule sits in the stage-3 bullet. Stage 2 does not literally require improvement on unrelated tasks, because a specialist's Δ there is 0. But it dilutes the specialist's effect over them, so stage 2 defeats the rule's purpose. Design threat §6.3 covers regulated traits under unconditional ablation, but not scope. Traits declared for 2 or more families have 6–8 in-scope ablation tasks and are not blocked in principle. [corrected after independent verification 2026-10-13]
- **Impact.** Arm C can only contain broadly helpful traits or high-effect family traits. Many natural ALFWorld skills are family-specific (microwave/heat, fridge/cool, sinkbasin/clean, desklamp/look). The H2 comparison is then partly "general traits only" versus "all traits", a confound in what the gate selects for. This is a power and dilution limitation that pre-registration can fix; it is not a code bug. [corrected after independent verification 2026-10-13]
- **Recommendation [requires ADR].** Choose one, and pre-register it:
  - (a) scope-conditional stage 2: compute the contribution on in-scope ablation tasks drawn from a family-indexed ablation bank, with the out-of-scope harm check unchanged;
  - (b) a per-family minimum in the ablation pool that is large enough for the pass region;
  - (c) explicitly declare that M0 gates only general traits, and say so in claims.

### 2.7 [P1] Universal scope and the ranking incentive (finding 7)

- **Evidence (V).** Miner prompt: "applicability.task_families: a NON-EMPTY subset of the allowed families"; "1.0 = plausibly reusable across several declared families". Ranking primary key = `estimated_generality` (design:39). PR `inheritance/generalization.py:84-86` requires exactly 10 out-of-scope tasks from families outside the declared applicability, otherwise `ValueError`. The issue #12 handoff §3: "#12 should reject an impossible allocation rather than improvise".
- **Evidence added by verification (V, code trace at `main` @ `6014bc7`; no repro).**
  - The check that actually fires for a universal-scope candidate is `validate_against_applicability` (`generalization.py:110-123`), called from `evaluate()` (:240).
  - `gate.py:266-274` runs the evaluators with no try/except, so the `ValueError` aborts the whole gate batch for that seed.
  - The proposal schema has `minItems: 1` and no `maxItems`. `miner.py:247` checks only membership in `batch.family_universe`, and ranking sorts by `(-estimated_generality, discovery_order)` (`miner.py:300`).
  - The prompt calls `estimated_generality` "a bounded ranking heuristic ONLY".
  [corrected after independent verification 2026-10-13]
- **Impact.** A candidate whose scope covers the whole family universe aborts the **entire gate batch** for its seed, not just its own stage 3. It counts as a setup error, not a scientific reject. The `ValueError` is deliberate under the #12 handoff, which assigned feasibility to #13 and the split manifest, so this is not a defect in the merged #12 code. The real gap is that ADR-0002 and the design pre-register no universal-scope rule, and #13 does not yet supply one. Without a rule, the run either crashes or a rule gets improvised mid-run, which design §2 forbids. The ranking incentive is a plausible but unmeasured inference. `estimated_generality` is a model self-report with no code coupling to scope breadth, so broad scopes *may* be favoured, which would partly reverse the scope-gaming threat in design §6.6. [corrected after independent verification 2026-10-13]
- **Recommendation [requires ADR].** Pre-register the universal-scope rule. Options: out-of-scope stage vacuous with in-scope drawn across all families; force at least one excluded family at extraction (cap or validate applicability); or an explicit reject verdict. [corrected after independent verification 2026-10-13] Also pre-register the deterministic in-/out-of-scope family-count plan (e.g., round-robin over declared and over excluded families). Today #13 "supplies" it with no rule, which is a researcher degree of freedom.

### 2.8 [P1] What C − B identifies (finding 8)

- **Evidence (V).** design:37 "arm B will have larger prompts; prompt overhead is part of the treatment". H2's name (SPEC §15) is "Causal validation … beats automatic inheritance".
- **Impact.** C − B mixes three things: (i) selection by causal evidence, (ii) fewer skills and a shorter prompt (dilution), and (iii) the τ_k length filter. A positive C − B is equally consistent with "any small subset beats ten".
- **Recommendation [requires ADR].** Add arm **D**: a size-matched random subset of the frozen candidates (|D| = |C| per seed, with a pre-registered RNG seed). Optionally add arm **E**: the top-|C| candidates by miner rank (a cheap-heuristic control). Cost: about 194 episodes per arm per seed. The H2 attribution claim should require C > D, not only C > B. If |C| = 0 this is moot (section 2.2).

### 2.9 [P1] Gate predictive validity is unmeasured (finding 11)

- SPEC §27 asks whether CIG can reliably tell whether a trait is worth inheriting, and whether that selection helps descendants. EXP-0001 measures only arm-level outcomes. Nothing measures whether promote/reject verdicts agree with held-out per-trait effects, and "Inheritance Precision" (SPEC §18) is not computed (I10).
- **Recommendation [requires ADR].** Add a pre-registered secondary analysis: per-candidate held-out contribution G0 ∪ t versus G0. The G0 runs are already arm A. Run it on T_eval *after* verdicts are frozen (no selection use) or on valid_seen. Report the rank correlation between gate stage-2 Δ and held-out Δ, plus the precision and recall of the promote verdict. Cost: 10 × 134 × 3 ≈ 4.0k episodes on T_eval, or 4.2k on valid_seen. This is the only direct test of the CIG classifier itself.

### 2.10 [P1] The teacher is a hidden variable (finding 12)

- **Evidence (V).** `TraitMiner(model: ModelAdapter, …)`. The YAML `model:` block names only Qwen2.5-7B-Instruct. `research/related-work/skillrl.md:7` "teacher OpenAI o3"; line 34 "Strong closed-source teacher dependence".
- **Impact.** A stronger teacher turns H1's "lifetime-acquired capability" into teacher-knowledge injection. A closed teacher conflicts with SPEC §21 ("araştırmanın kendisi belirli bir proprietary API'nin davranışına bağımlı olmamalıdır" — the research itself must not depend on a specific proprietary API's behaviour). A self-teacher (Qwen-7B) may produce low-quality skills.
- **Recommendation [requires ADR].** Pre-register the teacher identity, revision and hash. If it differs from the backbone, report it as a named factor and avoid "lifetime learning" wording in claims.

### 2.11 [P2] Other construct issues

- **Expression check (I11).** It is a prompt-exposure check. Rename it "exposure check" in CIG records and in the paper. Optionally add a behavioural-use diagnostic, such as the share of steps whose actions match a procedure step, or a cited step id. The diagnostic is descriptive only and needs no ADR. Using it as a pass rule **[requires ADR]**.
- **Interaction.**
  - ADR-0002/A1 "both vs neither" with "harmful iff `delta_pair < -0.05` (with n=10 this is a >=10pp observed drop)" fires on **one net task loss**. It cannot separate a true negative interaction from either trait's individual effect on the interaction pool, or from perturbation noise. It cannot detect *redundancy*, i.e. positive but sub-additive pairs, which is the bloat-relevant case for H5.
  - Higher-order interactions are never tested, so the final arm-C genome (all promoted traits together) is never evaluated on T_gate.
  - Recommendation: a descriptive (no-ADR) **joint-set check**, running the final arm-C genome versus G0 on the ablation and interaction pools once per seed (about 30 episodes), reported alongside. Using it as a gate rule **[requires ADR]**.
- **Replay in-sample.** One of the five replay instances is the source task the trait was distilled from. Report the source-task outcome separately from the 4 variants (descriptive, no ADR).
- **Adaptation cost.**
  - design:59 says "tokens/actions to first success" on "a 20-task T_gate transfer subset". At temp 0 there is one deterministic attempt per task, so "first success" is undefined.
  - The subset is unspecified. If it overlaps the ablation pool, arm C was selected on it.
  - **[requires ADR]** Define the metric, for example tokens and actions per solved task on held-out tasks, and pick a subset disjoint from all gate pools.
- **Placebo calibration.** A length-matched neutral skill run through all 5 stages gives an empirical null for each stage statistic. As a gate-run control candidate (excluded from arms B and C) this **[requires ADR]**. As a pre-run pilot on a disjoint pool it needs no ADR (section 2.2).

---

## 3. Statistical-conclusion validity

1. **Multiplicity (10 candidates × 5 stages).** Not the binding risk. The conjunction controls false promotion: model FWER ≤ 2e-5 (M). The binding risk is compounded Type II error (section 2.2). Do not add multiplicity corrections; they would make the gate even more conservative.
2. **Winner's curse.**
   - Promoted traits are selected on stage-2 LB > 0, so their stage-2 Δ is at least 0.20 by construction and is biased upward.
   - The interaction tie-break uses these discrete, inflated Δs. Values cluster at 0.20 and 0.25, so many ties fall through to miner rank.
   - Never report CIG-record Δs as effect sizes. T_eval (arm contrasts) and the section 2.9 per-trait held-out analysis give the unbiased estimates.
   - Selection-then-evaluation on T_eval is otherwise clean: T_eval is not used by the gate. **No ADR needed** for the reporting rule.
3. **Threshold sensitivity is close to vacuous (finding 9).**
   - tau_c ∈ {0.03, 0.05, 0.08} never binds (I13).
   - PR `inheritance/regression.py:231` computes `drop = 1.0 - successes/40`, compared with `<= tau` in PR `inheritance/gate.py:436,456`. For 38/40, drop = 0.050000000000000044 > 0.05. So the τ_r = 0.05 cell allows ≤ 1 failure instead of the intended ≤ 2, identical to the primary (V, `validity_checks.out` §2).
   - Net effect: the 9-cell grid can produce only 2 distinct compositions (τ_r = 0.02 and everything else).
   - Recommendations: (a) **no ADR**, because this restores the pre-registered rule: compare integer failure counts (`failures <= floor(tau*40 + 1e-9)`) or use `fractions.Fraction`, and add a 38/40 @ 0.05 boundary test; (b) **[requires ADR]** replace the tau_c grid with one that binds, e.g. a CI-level grid {80, 90, 95%} or a minimum-wins grid.
4. **Seed structure (finding 10).**
   - (a) Arm A ×3 seeds is deterministic duplication (I17). Run it once and reuse it.
   - (b) Whether H2's CI pools seeds is unspecified: mean per task over seeds then bootstrap tasks, or a separate CI per seed. Tasks are crossed with seeds. Between-seed variance (which traits get mined) is the main uncertainty about the *procedure*, and 3 seeds cannot estimate it.
   - (c) "C − B > 0 in ≥ 2/3 seeds" has probability 0.5 under a symmetric null, ignoring ties. With the discrete per-seed means over 134 paired binary tasks it is about 0.44–0.47. At ρ = 1 the seeds are identical and the rule reduces to the sign of one mean, which is also about 0.46–0.47. It is weak alone, though it is conjoined with the CI; the LB > 0 condition is what controls type I error. [corrected after independent verification 2026-10-13]
   - **[requires ADR]** Pre-register: primary = task-cluster bootstrap on per-task seed-means, with inference stated as conditional on the 3 realised seeds; sensitivity = two-way (seed × task) bootstrap or a mixed-effects logistic model; arm A run once.
5. **Regression CI on n = 40.** On top of the circularity (section 2.3), if arm C is empty (a plausible outcome, section 2.2), reg(C) = 0 [corrected after independent verification 2026-10-13] and "CI excluding 0" needs reg(B) ≥ about 4–5 of 40 (the sibling's MDE table, V there). The section 2.3 T_eval-based definition (n = number of A-solved T_eval tasks, about 40–60 at a 30–45% G0 rate; U) is no larger, but it is held-out.
6. **Clustering.** "Cluster bootstrap over tasks" treats tasks as independent. valid_unseen games share scenes and families (scene count U). **[requires ADR for primary; none for secondary]** Add a family-blocked or scene-blocked bootstrap as a pre-registered secondary.
7. **Determinism is assumed, not measured.**
   - The design spends deterministic duplicates anyway: stage-2 "without" = G0 on the same 20 tasks for all 10 candidates (180 duplicates per seed), and the interaction "neither" arm repeats identical G0 configs on the same 10 tasks for 45 pairs. Dedup would bring the interaction worst case from 900 to 460 per seed.
   - **[no ADR]** Use these duplicates as a free **determinism audit**: report the concordance of identical (config, task) episodes. Add a pre-run vLLM bitwise-repeat smoke test, and record batch size / batch-invariance settings. vLLM batch-invariant mode availability is U.
   - If concordance < 100%, the design §4 assumption that "exactly one execution per (task instance, condition)" is valid fails and must be disclosed.
8. **Bootstrap seed.** The stage-2 bootstrap seed is caller-supplied (`inheritance/ablation.py:268-269`, `self._bootstrap_seed`). Pre-register its derivation. Verdicts at the minimal passing compositions (4/0, 7/1, 9/2, 11/3) are robust (sibling exact P values 0.0115–0.015 versus 0.025). However, a few compositions are bootstrap-stream/ordering-dependent. Their pass rates over task orderings are (6,1) ≈ 2%, (8,2) ≈ 1%, (10,3) ≈ 32%, (12,4) ≈ 98% and (13,5) ≈ 0.1%, so the seed derivation matters there. This is P2 with no ADR. [corrected after independent verification 2026-10-13]

---

## 4. External validity

Brief; see the sibling for numbers.

- **Single domain and single backbone.** SPEC:801 prefers at least 3 capability classes for the first paper, while SPEC §29 allows one domain for M0. Every EXP-0001 claim must be scoped to "ALFWorld (text), Qwen2.5-7B-Instruct, frozen weights, temp 0, this G0 prompt". Do not use "general" or "model-independent" (SPEC §34).
- **Backbone currency.** Qwen2.5-7B is defensible as a SkillRL-lineage comparability anchor (sibling §6). Newer 8–9B models (e.g., Qwen3.5-9B) are not clearly better on this harness (sibling, V there). A one-model result invites the "does it hold on a 2026 model" question. H4 (#17) is the planned answer; it is not part of M0.
- **Saturation.** None for training-free 7B: anchors are 15–46% on different protocols. Per-family floors (Heat, Cool, Pick2 under some protocols), made worse by the 8-step cap (section 2.4), mean family-specific traits may show no effect because of floor effects, not because they are worthless.
- **Distribution shift between gate and evaluation.** T_gate is drawn from train (seen rooms). T_eval is valid_unseen (unseen rooms). The gate is validated in-distribution and evaluated under room shift. valid_seen (140) is unused. **[requires ADR]** Add valid_seen as a pre-registered secondary held-out set. This separates instance generalization from room generalization, and also gives a T_eval-free venue for section 2.9.
- **Contamination (U).** ALFWorld (2020) game files and ReAct ALFWorld prompts (2022) are public. Qwen2.5 pretraining exposure is plausible but unverified; I found no primary statement. It affects all arms' base level and may modulate skill uptake. **[no ADR]** Add a cheap probe: completion of ALFWorld expert-trajectory prefixes and game descriptions. Report it; do not claim it.

---

## 5. Reproducibility gaps against SPEC §24

SPEC:1027: "Sonuçlar seed, model revision, genome revision, dataset split ve code commit olmadan raporlanmamalıdır." (Results must not be reported without seed, model revision, genome revision, dataset split and code commit.)

| SPEC §24 / standard item | Manifest state (V) | Code state (V) | Gap / recommendation |
|---|---|---|---|
| Seeds | `mining_seeds: [11, 23, 41]` | Bootstrap seed caller-supplied; lifetime seed not defined (section 2.1) | Pre-register all seed derivations (lifetime, teacher, bootstrap, arm-D RNG) |
| Model revision | `revision: pin-at-run-time`, "record weights hash at run start" | ModelAdapter metadata carries `revision` | Also pin the vLLM version, dtype, max_model_len, tensor-parallel size, GPU type, batch/scheduling settings (determinism) |
| Genome revision | `initial: G0-seed-react` (label) | `runtime/seed_g0.py:24` `G0_GENOME_ID = "G-0f1e2d3c"` (hand-written, not content-derived); #22 open | #22 ADR + `compute_genome_id()` before #13; G1 ids for A/B/C; arm-B representation (section 2.5) |
| Dataset split | `hash: commit-at-lock` for T_mine/T_gate/T_eval; T_reg none | No split files, no split builder, no issue tracking one | Split-manifest builder: loader-filtered per-family counts (the TextWorld loader filters games; sibling §1.3), ALFWorld data version (e.g., json_2.1.1, U), family vocabulary, pool/bank membership, hashes, T_reg rule |
| Code commit | `code-commit` in `record.required_at_run_time` | — | Plus a lockfile (pyproject has lower bounds only: `jsonschema>=4.18`, `pytest>=8`; no vllm/alfworld/textworld pins) |
| Budgets | absent | `Budgets(max_steps, max_total_retries, max_tokens_per_request=512, max_total_tokens=None)`; planner cap 8 | Pre-register (section 2.4) |
| Prompts | `miner-prompt-hash` | Miner template + output-schema hashes recorded (V) | Also hash the G0 ReAct system/user template and the skill renderer (`skill_block`) |
| Token counter (τ_k) | `expression_overhead_cap_tokens: 512` | PR: counter identity + rendered sha256 recorded (#12 §6) | Pin tokenizer id/revision in the manifest |
| Environment / CI | — | No `.github/`; #18 open; suite collects fully only on 3.14 (sibling: missing `from pathlib import Path` in `tests/somatic/test_store_replay_parity.py:63`) | Land the one-line fix plus a 3.10–3.14 CI matrix before #13; all PR test counts so far are builder-reported (#20) |
| Determinism evidence | Policy asserted (design:54) | None | Pre-run repeat test + in-run duplicate-episode concordance (section 3.7) |

---

## 6. Critical path: what must exist before #13 can run

The current issue graph is "#13 depends on #12, #6" (issue #13 body). #22 says "Blocks #13". Several required items have **no issue at all**.

| Order | Item | Status (V unless noted) | Blocks |
|---|---|---|---|
| A1 | **[P0] ADR: lifetime sampling protocol** (k rollouts, temperature, seeds, pairing unit and instance identity in the pair check; temp-0 "G0-solved" for T_reg) | Not started; no issue. Policy explicitly deferred to #13 but not pre-registered | Mining (no candidates without it), T_reg, budget |
| A2 | **[P1] ADR: G0 budgets** (planner max_plan_steps, with a schema maximum above 16 if 50 steps; Budgets values) and **T_reg** selection rule, post-G0 freeze/hash, plus fallback / held-out regression definition | Not started; no issue | T_reg feasibility, H2 regression outcome, #6 |
| A3 | **[P1] ADR: arm-B genome representation + identical projection across arms** (plus validation that `assimilation` cites a promote-verdict CIG record) | Not started; no issue | G1 assembly provenance (arm B; the assimilation invariant for arm C). genome_id is blocked by #22 for all arms (B2) |
| A4 | **[P1]** Gate pre-registration fixes from sections 2.2, 2.6–2.8 and 3 (H2 non-degeneracy, specialist stage 2, universal-scope rule, arm D, seed pooling, sensitivity grid) | Proposals only | #6 integrity |
| B1 | #12 / PR #28 merge | Merged; `main` is now `6014bc7` (independent verification, 2026-10-13). This supersedes the earlier state in which the Critic merge gate was CLOSED at `7886852`. No CI runs as of my original read. [corrected after independent verification 2026-10-13] | Gate execution |
| B2 | #22 genome_id ADR + implementation | Open, no PR | #13 (per #22) |
| B3 | Float fix for τ_r comparisons (section 3.3) | Not raised | Honest sensitivity |
| C1 | **ALFWorld `ToolEnvironment` adapter** + benchmark-success outcome annotation + family labels | Absent: `runtime/loop.py:5` "No benchmark logic, no ALFWorld code"; no issue | Everything live |
| C2 | **Split-manifest builder** + committed hashes (pools, banks, interaction pool, stage-3 allocation rule) | Absent; no issue | Gate inputs, §24 |
| C3 | G1 assembly: somatic validated → germline `origin=assimilation` with CIG ref; arm-B per A3 | "downstream (#22/#13-era work)" (#12 handoff) | Arms B/C |
| C4 | T_eval evaluation + statistics module (H1/H2 tests per section 3.4, arm A once, per-family reporting) | Absent | Analysis |
| D1 | Live vLLM validation: weights hash, determinism repeat test, arm-B context fit, tokenizer counter | ModelAdapter tested against fakes only (#20: "no GPU/live model") | Pilot, run |
| D2 | **Calibration pilot** on a disjoint train pool: G0 success, d0, f0 with placebo skills, E[\|C\|] | Not planned | Honest #6 numbers; A2/A4 calibration |
| E1 | #18 CI + Python-portability fix | Open | Attested test claims |
| E2 | #6 pre-registration (expected results, per-stage pass counts, compute cap) | Open; cannot be filled credibly before D2 | #13 |
| F | Compute: the stated 5.8–6.7k is an undercount; the real range is about 7.2k (shared "neither" cached) to 8.5k (worst case, I5; P2) episodes [corrected after independent verification 2026-10-13], plus arm D (~0.6k) and the optional section 2.9 analysis (~4k) | Not allocated (#6 field empty) | #13 |

Suggested sequencing:
- A1–A3 and B2 first. They are cheap paperwork and unblock C3/C2.
- C1 and C2 in parallel with B1.
- Then D1 and D2.
- Then A4 calibration via ADR, informed by D2 and **before** E2.
- Then E2.
- #13 last.

No locked element should change after D2 except through an ADR dated before #6 is filled.

Priority after independent verification: A1 is the only P0 item, because without it the miner receives no mining pairs and hence no candidates. A2–A4 are P1. They must be decided by ADR before E2 (#6) is filled. Unlike A1, they do not stop the experiment from producing candidates. The rows A1–A4, B1 and F above carry the corrected priorities and status. [corrected after independent verification 2026-10-13]

---

## 7. Coordination note

All issue and PR traffic I read is posted through the owner account, signed "Builder" or "Research/Critic". From the repo alone I cannot tell which product (Zcode or Codex) plays which role (U). Nothing in this review is posted anywhere. If the orchestrator relays any of this, put finding 1 (P0; it blocks candidate production) to both agents first. Then put findings 2–6 (P1 after verification; each can degrade or confound #13 and must be resolved by ADR before #6 is filled). Use the corrected wording of each finding, not the original. [corrected after independent verification 2026-10-13]

## 8. Unverified / not done

- The gate pass probabilities are model outputs (M). They depend on assumed p, d and e, independence across tasks, and homogeneous effects. Only the pilot (D2) can replace them with data.
- The minimum Pick2 plan length under an 8-step cap is my reasoning from ALFWorld action semantics (U).
- The ALFWorld scene count in valid_unseen (U), Qwen2.5 contamination (U), and vLLM batch-invariance options (U) were not verified.
- PR head `c6ce4c1` was not code-reviewed beyond the regression/sensitivity lines cited.
