# EXP-0001 — Minimal Causal Inheritance Gate experiment (design, pre-registration)

**Status: PREREGISTERED-LOCKED (2026-10-14, before EXP-0001 execution).** Originally designed and locked 2026-10-02; revised same day per Research/Critic review on PR #2 / issue #5 (see §9); expected results and compute cap pre-registered 2026-10-14, before any EXP-0001 execution (see §8). Machine-readable manifest: `EXP-0001-minimal-cig.yaml`. Decision record: `research/decisions/ADR-0002-exp-0001-design.md`.

This is Milestone 0 (SPEC §29): one model, one domain, one agent, two generations. It exists to test the load-bearing hypothesis H2 before anything else is built (SPEC §27).

## 1. Hypotheses under test

- **Primary — H2 (Causal Gate):** CIG-gated inheritance produces better held-out performance and lower regression than unrestricted skill inheritance.
- **Secondary — H1 (Assimilation, directional):** assimilated traits improve descendant zero-shot performance vs no inheritance.
- **Observational — H5 (partial):** genome size/bloat comparison between gated and unrestricted arms.

## 2. Environment and model

- **Environment: ALFWorld** (text-based embodied tool use; same family as SkillRL's setup for future comparability). Deterministic simulator; LLM is the only stochastic component. Eval temperature 0.
- **Model:** Qwen2.5-7B-Instruct, local (vLLM), via ModelAdapter. Rationale: matches SkillRL's base model; local → reproducible, cheap CIG replay loops; supports later H4 backbone swap. *ADR-0011 (owner, 2026-10-16): the model is Qwen3.8-27B (a current family), served text-only with thinking off and without batch-invariant kernels; the SkillRL base-model match no longer holds, and the G0 success near 30 % assumed in ADR-0007 E1 and ADR-0009 was written for Qwen2.5-7B (the new pilot reports, the owner decides).*
- **Split discipline — four disjoint splits, task-family-stratified.** Contamination between gate data and final evaluation would invalidate H2, so all train-side splits are drawn with **task-family-stratified quotas** across ALFWorld's ~6 task families (proportional to family availability with a minimum of ~8 tasks per family in `T_mine` and ~4 per family in `T_gate`; target sizes are approximate). Stratification prevents a family-composition confound between the arm-identical G0 phase and the gate phase:
  - `T_mine` (~60 tasks, ALFWorld train split, stratified): G0's lifetime; trajectories → Trait Miner.
  - `T_gate` (~114 tasks, train split, stratified, disjoint instances from T_mine), organized as:
    - *ablation pool* (20, stratified): stage-2 contribution estimation;
    - *interaction pool* (10): pairwise co-expression checks;
    - *`replay_bank[family]`* — **family-indexed** bank, disjoint, **≥ 4 valid unseen instances per family** (stage-1 replay variants are drawn from the source task's family bank);
    - *`generalization_bank[family]`* — **family-indexed** bank, disjoint, **≥ 10 valid unseen instances per family** (stage-3 in-/out-of-scope sampling per trait family). **Feasibility fallback:** if ALFWorld availability prevents ≥ 10 (or ≥ 4) for some family, a **family-specific pre-registered maximum** is recorded in the split manifest before run start — never improvised mid-run.
  Banks exist to guarantee per-family feasibility for single-family specialist traits; **bank construction adds no LLM calls** — only per-trait evaluations on bank instances drive compute.
  - *`benefit_bank[family]`* (ADR-0012 D1, owner, 2026-10-19) — 192 ALFWorld train games per family (1,152) that are in no pool of the split manifest and not in the calibration pilot, the smallest `sha256("heritagent-split|0|benefit_bank|<family>|<game_id>")`, in ascending order; recorded in its own manifest `EXP-0001-benefit-bank.json` with its sha256 and the split and pilot hashes it was drawn against; a family with fewer fails closed; no LLM calls. Stage 2 and the out-of-scope check draw from it. Since ADR-0012 the gate reads neither `replay_bank` nor `generalization_bank` (both stay in the split manifest), and the ablation pool is the transfer subset only (ADR-0007 E4).
  - `T_eval` (134 tasks, official ALFWorld unseen eval split): **final evaluation only** — never visible to miner, gate, or threshold tuning. Untouched by stratification decisions.
  - `T_reg` (40 tasks, subset of T_mine that G0 solved): regression suite (arm-internal, run at G1 evaluation time).
  - Splits are content-hashed and committed with the manifest; any change after execution starts = new experiment id (ADR-0012 Q2, owner, 2026-10-19; it read "any change"). The benefit bank's separate manifest, added before any EXP-0001 episode, leaves `EXP-0001-splits.json` and its hash unchanged.

## 3. Arms (identical G0, identical seeds, identical eval budget)

| Arm | Inheritance policy | G1 genome |
|---|---|---|
| A — no inheritance | none | G0 genome unchanged (skills empty) |
| B — unrestricted | all mined candidate traits injected as skill genes | G0 ∪ all candidates |
| C — CIG-gated | only candidates passing all four gate stages | G0 ∪ validated candidates |

Arm B operationalizes the SkillRL-style persistence policy (library growth + always-available skills) without weight training — the spec's §17 "most critical comparison". Note: arm B will have larger prompts; prompt overhead is part of the treatment being tested (bloat is an outcome, per H5), not a confound to normalize away. Equal **evaluation budget** (max steps per task) is enforced across arms.

**Candidate-set freeze (applies to both arms):** after mining, candidates are ranked (primary: miner-reported estimated generality; tie-break: earliest discovery order), the top **K = 10** are selected, and this **frozen set — with its ranking — is committed as an artifact before any arm branches**. Arms B and C start from the exact same frozen candidates; B injects all of them, C gates the same set. At extraction each candidate also declares machine-readable scope — **`applicability.task_families[]`** (frozen with the candidate set; used **only** for gate sampling and audit: since ADR-0012 D4–D5 it selects the stage-2 benefit-bank games and the stage-3 out-of-scope games). **Runtime input hygiene — all arms:** `applicability.task_families[]` and every other evaluation-only metadata field is **stripped from all runtime model inputs in every arm** (A, B, and C — not only the strict SkillRL projection), so no arm — and arm C in particular — can ever receive evaluation-only oracle metadata at runtime. Runtime prompt assembly consumes only the projected payload (`name`, `principle`, `when_to_apply`, `procedure`); the assembly function takes the stripped projection as its input and tests assert the absence of evaluation-only fields (checklist for issues #4/#7). A sensitivity analysis over `K ∈ {5, 10, 20}` is planned as a follow-up (EXP-0005, secondary — see §5).

## 4. Pipeline

**G0 lifetime:** G0 (seed genome: ReAct-style planner + retry policy, no skills) runs T_mine; trajectories recorded (full functional phenotype per SPEC §7). Trait Miner v0: LLM differential distillation over success/failure pairs (SkillRL-adapted, cited) → candidate skill genes (`{name, principle, when_to_apply, procedure}` + somatic envelope). Cap: ≤ 10 candidates per run (keeps CIG budget bounded; cap itself recorded).

**CIG stages (arm C only), thresholds v2 (ADR-0012, owner, 2026-10-19; the v0 rules of 2026-10-02 are kept as historical records in §9–§11 and ADR-0002):**

Per mining seed the gate runs in six rounds (ADR-0012 D2): R1 stage 1 and stage-2 look 1; R2–R4 stage-2 looks 2–4 for every candidate still running; R5 the out-of-scope check and 4A; R6 4C. A round starts after the previous round's episodes are recorded.

1. **Replay (expression check)** — the candidate plays its source task (the T_mine game of its mining pair's success) once with the trait. **Pass:** the target gene is expressed in that episode, or the episode fails. No outcome threshold, no without-trait execution (ADR-0012 D3).
2. **Benefit (record key `ablation`)** — up to 192 `benefit_bank` games over the frozen applicability families (m families in name order; allocation game j is the ⌊j/m⌋-th game, in bank order, of the (j mod m)-th family). Looks after n_k = 48, 96, 144, 192 games; look k plays games [n_(k−1), n_k) in round R_k against that round's cells. Over the first n_k games: n⁺ = with-trait solved and cell failed, n⁻ the reverse, net = n⁺ − n⁻, point = net / n_k, p_k = P(Bin(n⁺ + n⁻, ½) ≥ n⁺) in exact arithmetic (1 if n⁺ + n⁻ = 0). **Pass** at look k iff p_k ≤ α_k and point ≥ τ_c = 0.05, (α₁, α₂, α₃, α₄) = (0.0005, 0.0005, 0.001, 0.003). **Futility:** at looks 1–3 stop and fail iff net ≤ ⌊n_k/40⌋; no efficacy at look 4 fails. Games are never filtered by expression; expression is recorded. The ADR-0005 bootstrap interval over the games played is reported, descriptive only. The candidate's contribution is its point at the stopping look (ADR-0012 D4).
3. **Generalization — split by scope** (a specialist trait must not be required to improve unrelated task types):
   - **In-scope:** the child records stage 2's paired outcomes by reference. **Pass:** Δsuccess ≥ 0 over them. It runs no episode and adds no test (ADR-0012 D5).
   - **Out-of-scope** (the applicability omits a family): 24 `benefit_bank` games over the omitted families, round-robin in name order, each family's first games in bank order, in R5. **Pass:** Δsuccess ≥ −0.05 in exact arithmetic (net loss ≤ 1 of 24). An all-family applicability passes vacuously (ADR-0007 E6).
   - **Sampling reuse semantics:** no replacement *within* a candidate's games; every candidate that needs n games of a family gets the same first n, and this **reuse across candidates is logged** in each CIG record, as is each round's cell reuse. Scope declarations are immutable once the gate starts (see threat §6.6).

**Determinism / stochastic-source policy (M0; ADR-0012 D2, D8):** eval temperature is 0 and ALFWorld is deterministic. In each round G0 (no trait) plays once, at the gate's locked settings, every game a with-trait execution of that round is compared with: the game's **cell** for (seed, round). Every without-trait outcome of the round (stages 2, 3 and 4A) and every neither outcome (4C) reads it. A cell is read only in its round, never pooled with another, never shared across seeds or with the final evaluation. A round's cells and with-trait executions form one queue in ascending sha256 of their ledger keys. So every gate stage runs **exactly one execution per (seed, round, task instance, condition)**. Duplicate executions (a game's cells in two rounds, temperature-0 repeats, the lifetime's temperature-0.7 attempts) are never evidence for a gate decision; the cross-round and cross-seed cell disagreement is reported descriptively. The server has no batch-invariant kernels (ADR-0011 D2). Uncertainty comes from (a) distinct task instances and (b) independent mining seeds. If repeated stochastic runs are scientifically desired in a later experiment, the stochastic source (e.g., a pre-registered temperature grid or seed set) must be introduced and pre-registered explicitly — backend nondeterminism is never relied upon.
4. **Interaction + regression** — 4A: T_reg (40 tasks) with genome ∪ trait in R5, paired against the seed's cells; **pass:** (success_without − success_with)/40 ≤ 0.03 (τ_r; ADR-0009 D1). 4B: expression overhead τ_k ≤ 512 tokens/trait. 4C: if ≥ 2 eligible skills, both vs neither on the 10 interaction tasks in R6, neither read from the seed's R6 cells; a pair is harmful iff Δ_pair < −0.05, and the member with the lower stage-2 contribution is removed, an exact tie keeping the earlier frozen rank (ADR-0002 A1, ADR-0012 D6).

**Which stages run (ADR-0012 D7).** Stages 1, 2 and 4B run for every candidate, and its in-scope child is computed from its stage-2 record. The out-of-scope check, 4A and 4C run only for candidates whose stage-2 p_k met α_k at some look; the others' children carry `stage_pass: false`, `measurements.not_run: true` and the reason. All five record keys are written for every candidate.

Gate outputs are CIG records (schema `cig/0.1`); rejected candidates stay in somatic memory (SPEC §9 fail path).

**G1 evaluation:** each arm's G1 runs T_eval (across 3 independent mining seeds — uncertainty sources are task instances and mining seeds, not backend nondeterminism) + T_reg. Metrics: T_eval success (primary), regression rate on T_reg (primary), genome size (genes, expression tokens), descendant adaptation cost on a 20-task T_gate transfer subset (tokens/actions to first success — H3 preview).

## 5. Statistics (locked)

- Report per-task success with bootstrap 95% CIs (cluster bootstrap over tasks, 10k resamples) for all pairwise arm contrasts on T_eval.
- Stage-level gate statistics are computed over **distinct task instances only** — one execution per (seed, round, instance, condition) (ADR-0012 D2); no repeated-run pseudo-replication. The stage-2 decision is §4's group-sequential exact paired sign test, recomputed by the gate from the record; its ADR-0005 bootstrap interval is descriptive only (ADR-0012 D4).
- Seed-level consistency requirement: direction of C−B must hold in ≥ 2/3 mining seeds.
- H2 supported iff: (C − B) ≥ +5pp on T_eval with CI excluding 0 AND regression rate(C) < regression rate(B) with CI excluding 0 on the difference.
- H1 supported iff (C − A) CI excludes 0 (positive).
- No p-hacking escape hatch: secondary metrics are reported regardless of outcome; all gate thresholds and this plan are committed before the run.

**Secondary sensitivity analyses (pre-planned now, reported as clearly secondary; primary conclusions above remain the pre-locked ones and are never retuned):**
- *Threshold sensitivity:* recompute gate decisions on T_gate stage data only, varying τ_c ∈ {0.03, 0.05, 0.08} and τ_r ∈ {0.02, 0.03, 0.05}. Report how arm-C composition and T_eval outcomes would change under each — as analysis, not as a new primary result. `T_eval` is never used for threshold selection. *ADR-0012 D7 (2026-10-19): τ_c is recomputed on the recorded stage-2 paths, and a grid value under which the test would have continued past the observed stopping look is recorded as not identifiable; τ_r on 4A where it ran; no grid value changes a not-run child.*
- *Candidate-cap sensitivity (follow-up, EXP-0005):* re-run the B/C arm construction with K ∈ {5, 10, 20} from the same frozen ranked candidate list.

## 6. Threats to validity (to monitor in post-run review)

1. Trait Miner quality variance → same miner, same prompts across arms; miner log committed.
2. Gate overfitting to T_gate → mitigated by disjoint T_eval and family stratification; check trait generality across families via the out-of-scope stage.
3. Regulated traits misvalued under unconditional ablation (SkillShapley lesson) → ablation evaluates skills under their `express_when` conditions.
4. Determinism: ALFWorld is deterministic and eval LLM temperature is 0 ⇒ exactly one execution per (seed, round, task, condition) everywhere in gating (ADR-0012 D2); mining-phase temp 0.7 variance is covered by 3 independent mining seeds. Backend nondeterminism (batching, floating point) is never counted as evidence; any future stochastic repeats require an explicitly pre-registered stochastic source.
5. Baseline strength: if arm B ≈ arm C because few candidates are harmful, that is a *finding* (gate unnecessary in this regime), not a failed experiment — record as such.
6. **Scope-declaration gaming:** a trait could declare narrow applicability to face an easy in-scope check while arm B suffers its cost everywhere. Mitigations: (a) scope declared at extraction, frozen before gate stages, recorded in the CIG record; (b) out-of-scope harmlessness is still required; (c) trait utility in arm C is conditional on scope, but T_eval success is unconditional — misdeclared scope shows up as lost T_eval performance for arm C relative to arm B. *ADR-0012 (owner, 2026-10-19): the declared scope now also selects the stage-2 and out-of-scope samples, which enlarges this surface; (a)–(c) are unchanged.*

## 7. Budget estimate (order-of-magnitude, for planning)

Mining: 3 seeds × 60 tasks = 180 episodes. CIG per seed: 10 candidates × [replay 5×2 + ablation 20×2 + generalization 20×2 + regression 40] = 1,300 episodes → 3,900 across seeds (single deterministic run per (task, condition); ~130 episodes per candidate vs ~350 under the earlier repeated-run design). Final evaluation: 3 arms × (134 + 40 + 20) × 3 mining seeds ≈ 1.75k episodes. The interaction stage adds ≤ ~900 in the worst case (all pairs of 10 validated skills). Total ≈ 5.8–6.7k episodes × ~30 LLM calls ≈ 180–200k local-7B calls — feasible on one GPU node over a few days. **Bank construction (`replay_bank`, `generalization_bank`; ~84 instances) adds no LLM calls** — only per-trait evaluations on bank instances are billed, and per-trait counts are unchanged by the family-indexed layout.

*ADR-0012 D9, D11 (owner, 2026-10-19; M, not decision rules):* the estimate above was written for Qwen2.5-7B and the locked gate. For the recalibrated gate, the planning mixture expects 6,181 / 6,879 / 7,373 episodes at δ = 0 / 0.05 / 0.18 (p99 up to 8,598; largest seen 9,632; up to 13,934 in the heaviest simulated worlds), about 163k–194k calls at the pilot's 26.3 per episode. With every candidate running every look and stage, every pair running 4C and every seed's C non-empty, the run needs at most 16,724 episodes (V arithmetic); the caps (§8) are set from the simulated tails, not from this bound. The benefit bank (1,152 games) adds no LLM calls.

## 8. Pre-registration (LOCKED 2026-10-14, before EXP-0001 execution)

**Pre-registration completed before EXP-0001 execution; no EXP-0001 outcome data had been collected or inspected when these expectations were locked.** (This statement refers to actual EXP-0001 outcome data only; the synthetic test/dev fixtures used while building the merged CIG implementation are not EXP-0001 outcomes.)

### Expectations are NOT decision rules

The values below are EXPECTATIONS — directional forecasts recorded for post-run review against SPEC §33. They are **not** success thresholds and must never be used to retune anything mid-run. The **PRIMARY SUPPORT CRITERIA** in §5 (locked) remain the only decision rules, unchanged by this pre-registration:

- H2 supported iff: `C−B ≥ +5pp` on T_eval, bootstrap 95% CI excluding 0, AND regression rate(C) < regression rate(B) with the regression-difference CI excluding 0, AND the C−B direction holds in ≥ 2/3 mining seeds;
- H1 supported iff: `C−A > 0` with bootstrap 95% CI excluding 0.

In particular, the expected C−B central value of +7pp does **not** move the +5pp threshold. Null/negative outcomes remain valid scientific results.

- [x] **Expected result, H2 (direction + rough magnitude):** CIG-gated arm C > unrestricted arm B on held-out `T_eval`, with lower regression. Expected `C−B` held-out success: approximately **+5 to +10 percentage points**, central expectation approximately **+7pp**. Regression rate for C expected approximately **3–8pp lower** than B. These are expectations, NOT success thresholds.
- [x] **Expected result, H1:** arm C shows positive held-out improvement over the no-inheritance arm A. Expected `C−A`: approximately **+5 to +15 percentage points**, central expectation approximately **+10pp**. Expectation, NOT a success threshold — the H1 support criterion stays `C−A > 0` with bootstrap 95% CI excluding 0.
- [x] **Expected number of candidates passing each gate stage** — a **rough expectation per mining seed, not a quota** (frozen candidate set: K = 10 per seed):
  - Stage 1 — Replay: ~**6 / 10**;
  - Stage 2 — Ablation: ~**4 / 10**;
  - Stage 3 — Generalization (combined): ~**4 / 10** (audit detail: in-scope pass ~5–6/10; out-of-scope harmlessness ~7–9/10; combined stage-3 pass ~4/10);
  - Stage 4 — Interaction + regression + overhead: ~**3 / 10** (most candidates are expected to pass the token-overhead check alone; attrition is primarily expected from the 40-task regression stage and occasional harmful-pair interaction);
  - Final promoted into arm C: ≈ **3 / 10**.

  These numbers may never be used to justify threshold changes during the experiment.

  *Dated note (ADR-0012 D11, owner, 2026-10-19):* these expectations stay as recorded, written for Qwen2.5-7B and the locked gate. ADR-0012's projections for the recalibrated gate (M, not decision rules; per seed 3 traits at +15 pp in all families, 2 at +25 pp in two, 5 with no effect; δ = 0 / 0.05 / 0.18): helpful traits promoted 4.71 / 4.02 / 2.37 of 5; no-effect traits promoted 0 / 0.0034 / 0.0115 of 15; arm C empty in a seed ≈ 1e-6 / 3e-4 / 0.049.
- [x] **Compute allocation and wall-clock cap (hard resource envelope):**
  - at most **1 local GPU/accelerator node**;
  - **14,500 total experiment episodes — hard cap** (ADR-0012 D9, owner, 2026-10-19, before any EXP-0001 episode; locked 2026-10-14 at 7,000; expectation ≈ 6,200–7,400, §7 note);
  - **507,500 model calls — hard cap** (35 per episode; ADR-0012 D9; locked at 225,000; expectation ≈ 163k–194k, §7 note);
  - **96 wall-clock hours — hard cap**;
  - mining seeds fixed: **[11, 23, 41]**.

  **Hard rule:** if any hard cap would be exceeded, stop the experiment, mark the run incomplete, and record the reason. Before each gate round the run also stops so unless the caps leave room for the round's largest possible count, given the candidates still in it, plus the final evaluation's largest (1,358 episodes), the calls converted at the run's mean per episode so far; the per-episode worst-case check stays (ADR-0012 D9, ADR-0007 E5). The budget is never silently enlarged; any rerun with a larger cap requires a new preregistration amendment / experiment revision BEFORE new data collection. Hardware throughput differences never justify changing scientific thresholds.

### Pre-registration immutability

Once EXP-0001 execution has started, the expected results, thresholds, support criteria, gate-survival expectations, and compute caps in this §8 are **immutable under the same EXP-0001 identity**. If a change is required, the current run is preserved as incomplete/aborted, an amendment / new experiment revision is pre-registered BEFORE any additional outcome collection, and the change is recorded in the audit trail.

## 9. Review resolution (critic round 1, 2026-10-02)

All five requested changes from PR #2 / issue #5 applied; decisions recorded in ADR-0002:

*Note (ADR-0012, owner, 2026-10-19):* items 1 and 3 are historical records. The in-scope check now records stage 2 by reference and the out-of-scope check plays 24 benefit-bank games (D5); replay is an expression check on the source task alone (D3). Current rules: §4.

1. Generalization stage split into **in-scope** (Δ ≥ 0 on unseen instances matching declared applicability) and **out-of-scope** (Δ ≥ −0.05 on non-matching families) — specialist traits are no longer penalized for not improving unrelated types (§4 stage 3).
2. Raw 60/40 allocation replaced with **task-family-stratified quotas** (§2); `T_eval` remains the untouched official unseen split.
3. Replay now uses **valid same-family resampling** from a disjoint reserve; synthetic object/room substitution dropped for M0 unless later proven semantics-preserving (§4 stage 1).
4. **Candidate-set freeze** before arm branching: ranked list, top K=10, same frozen set for arms B and C, committed artifact; K-sensitivity {5,10,20} planned as EXP-0005 (§3, §5).
5. **Post-primary threshold-sensitivity plan** added (τ_c, τ_r grids on T_gate data only, clearly secondary, no T_eval retuning) (§5).

## 10. Review resolution (critic round 2, 2026-10-02)

All three requested changes from PR #2 / issue #5 applied; decisions recorded in ADR-0002:

*Note (ADR-0012, owner, 2026-10-19):* historical records where superseded: item 1's one-execution rule is now one per (seed, round, instance, condition) (D2), its instance thresholds are replaced by D3 and D4, and its cost by §7's note; item 2's reuse rule now covers the benefit bank, not the replay reserve (D4, D5); item 3's applicability also selects the stage-2 games (D4).

1. **Pseudo-replication removed:** every gate stage runs exactly one deterministic execution per (task instance, condition) at eval temp 0; duplicate deterministic runs are never treated as independent evidence. Uncertainty sources are distinct task instances (cluster bootstrap) and independent mining seeds. A determinism/stochastic-source policy was added: any future repeated stochastic runs require an explicitly pre-registered stochastic source. Stage thresholds re-expressed over instances (replay: ≥ 3/5 success-with, Δ ≥ +2/5; ablation CI over the 20 instances). Per-candidate gate cost drops from ~350 to ~130 episodes (§4, §6.4, §7).
2. **Per-trait sampling reuse semantics defined:** no replacement *within* a trait's evaluation set; cross-trait reuse allowed and logged per trait in the CIG records (intended paired design for comparability); applies to both the generalization reserve and the replay reserve (§2, §4 stage 3).
3. **Machine-readable, evaluation-only applicability:** `applicability.task_families[]` declared at extraction, frozen with the candidate set, used only for stage-3 sampling/audit — never a runtime regulation predicate or agent-visible oracle. Field definition lives in the EAG schema (PR #1, commit `9ef2f5a`); the SkillRL projection adapter strips it (§3).

## 11. Review resolution (final round, 2026-10-02)

Both remaining blocks from PR #2 / issue #5 applied; decisions recorded in ADR-0002:

*Note (ADR-0012, owner, 2026-10-19):* historical records where superseded: the gate no longer reads item 1's banks (stage 1 plays the source task alone, D3; stage 2 and the out-of-scope check draw from the benefit bank, D4–D5); item 2's "the total budget (§7) are unchanged" no longer holds (§7 note, §8 caps).

1. **Family-indexed evaluation banks replace the small global reserves:** `replay_bank[family]` with ≥ 4 valid unseen instances per family and `generalization_bank[family]` with ≥ 10 per family — disjoint from T_mine, T_reg, ablation pool, interaction pool, each other, and from the untouched `T_eval`. **Feasibility fallback:** where ALFWorld availability prevents the guarantee, a family-specific **pre-registered maximum** is recorded in the split manifest before run start (never improvised mid-run). Stage 1 draws variants from `replay_bank[family-of-source-task]`; stage-3 in-scope draws from `generalization_bank[f]` over the trait's declared families. Reuse semantics unchanged: cross-trait reuse allowed and logged, within-trait no replacement (§2, §4).
2. **Bank construction adds no LLM calls** — only per-trait evaluations on bank instances drive compute; per-trait counts and the total budget (§7) are unchanged by the family-indexed layout.
3. **Applicability stripped from all runtime model inputs in every arm:** `applicability.task_families[]` and all evaluation-only metadata are removed from every arm's runtime inputs (A, B, C) — not only the strict SkillRL projection — so arm C cannot accidentally receive oracle metadata. Runtime prompt assembly consumes only the projected payload (`name`, `principle`, `when_to_apply`, `procedure`); assembly takes the stripped projection as input and tests assert absence of evaluation-only fields (checklist for #4/#7) (§3).
