# EXP-0001 — Minimal Causal Inheritance Gate experiment (design, pre-registration)

**Status: REVISED after critic round 1 (2026-10-02) — re-review requested.** Originally designed and locked 2026-10-02; revised same day per Research/Critic review on PR #2 / issue #5 (see §9). Machine-readable manifest: `EXP-0001-minimal-cig.yaml`. Decision record: `research/decisions/ADR-0002-exp-0001-design.md`. Expected-result fields must still be filled in before the run starts (SPEC §33).

This is Milestone 0 (SPEC §29): one model, one domain, one agent, two generations. It exists to test the load-bearing hypothesis H2 before anything else is built (SPEC §27).

## 1. Hypotheses under test

- **Primary — H2 (Causal Gate):** CIG-gated inheritance produces better held-out performance and lower regression than unrestricted skill inheritance.
- **Secondary — H1 (Assimilation, directional):** assimilated traits improve descendant zero-shot performance vs no inheritance.
- **Observational — H5 (partial):** genome size/bloat comparison between gated and unrestricted arms.

## 2. Environment and model

- **Environment: ALFWorld** (text-based embodied tool use; same family as SkillRL's setup for future comparability). Deterministic simulator; LLM is the only stochastic component. Eval temperature 0.
- **Model:** Qwen2.5-7B-Instruct, local (vLLM), via ModelAdapter. Rationale: matches SkillRL's base model; local → reproducible, cheap CIG replay loops; supports later H4 backbone swap.
- **Split discipline — four disjoint splits, task-family-stratified.** Contamination between gate data and final evaluation would invalidate H2, so all train-side splits are drawn with **task-family-stratified quotas** across ALFWorld's ~6 task families (proportional to family availability with a minimum of ~8 tasks per family in `T_mine` and ~4 per family in `T_gate`; target sizes are approximate). Stratification prevents a family-composition confound between the arm-identical G0 phase and the gate phase:
  - `T_mine` (~60 tasks, ALFWorld train split, stratified): G0's lifetime; trajectories → Trait Miner.
  - `T_gate` (~60 tasks, train split, stratified, disjoint instances from T_mine), organized as sub-pools:
    - *ablation pool* (20, stratified): stage-2 contribution estimation;
    - *generalization reserve* (20, stratified): per-trait in-scope / out-of-scope sampling (stage 3);
    - *interaction pool* (10): pairwise co-expression checks;
    - *replay reserve* (10, ≥1 per family): same-family unseen instances for stage-1 replay variants.
  - `T_eval` (134 tasks, official ALFWorld unseen eval split): **final evaluation only** — never visible to miner, gate, or threshold tuning. Untouched by stratification decisions.
  - `T_reg` (40 tasks, subset of T_mine that G0 solved): regression suite (arm-internal, run at G1 evaluation time).
  - Splits are content-hashed and committed with the manifest; any change = new experiment id.

## 3. Arms (identical G0, identical seeds, identical eval budget)

| Arm | Inheritance policy | G1 genome |
|---|---|---|
| A — no inheritance | none | G0 genome unchanged (skills empty) |
| B — unrestricted | all mined candidate traits injected as skill genes | G0 ∪ all candidates |
| C — CIG-gated | only candidates passing all four gate stages | G0 ∪ validated candidates |

Arm B operationalizes the SkillRL-style persistence policy (library growth + always-available skills) without weight training — the spec's §17 "most critical comparison". Note: arm B will have larger prompts; prompt overhead is part of the treatment being tested (bloat is an outcome, per H5), not a confound to normalize away. Equal **evaluation budget** (max steps per task) is enforced across arms.

**Candidate-set freeze (applies to both arms):** after mining, candidates are ranked (primary: miner-reported estimated generality; tie-break: earliest discovery order), the top **K = 10** are selected, and this **frozen set — with its ranking — is committed as an artifact before any arm branches**. Arms B and C start from the exact same frozen candidates; B injects all of them, C gates the same set. A sensitivity analysis over `K ∈ {5, 10, 20}` is planned as a follow-up (EXP-0005, secondary — see §5).

## 4. Pipeline

**G0 lifetime:** G0 (seed genome: ReAct-style planner + retry policy, no skills) runs T_mine; trajectories recorded (full functional phenotype per SPEC §7). Trait Miner v0: LLM differential distillation over success/failure pairs (SkillRL-adapted, cited) → candidate skill genes (`{name, principle, when_to_apply, procedure}` + somatic envelope). Cap: ≤ 10 candidates per run (keeps CIG budget bounded; cap itself recorded).

**CIG stages (arm C only), initial thresholds (SPEC §10 allows adaptive thresholds later; v0 fixed):**

1. **Replay validation** — source task + 4 **valid same-family resampled variants**: unseen instances of the same task family drawn from the replay reserve (disjoint from all other pools). No synthetic object/room substitution in M0 — substitutions are only admissible later if proven to preserve valid ALFWorld state semantics. 5 runs each, with vs without trait, eval temp 0. **Pass:** mean success-with ≥ 0.6 AND Δsuccess ≥ +0.4 on the replay cluster; trait must fire at least once per successful episode (expression check, not just presence).
2. **Controlled ablation / contribution** — 20 tasks from the stratified ablation pool, 3 runs each with vs without. Contribution = Δsuccess; **pass:** bootstrap 95% CI lower bound > 0 AND point estimate ≥ +0.05 (τ_c).
3. **Generalization — split by scope** (a specialist trait must not be required to improve unrelated task types):
   - **In-scope:** 10 unseen instances from families matching the trait's declared applicability (`when_to_apply`, declared by the miner at extraction time and frozen before any gate stage runs). **Pass:** Δsuccess ≥ 0 — the trait retains non-negative value on unseen instances where it claims to apply.
   - **Out-of-scope:** 10 instances from families *outside* the declared applicability. **Pass:** Δsuccess ≥ −0.05 — no harmful spillover where the trait should not apply.
   - Per-trait sampling is without replacement from the generalization reserve; allocations are logged in the CIG record. Scope declarations are immutable once the gate starts (see threat §6.6).
4. **Interaction + regression** — re-run T_reg (40 tasks) with genome ∪ trait; **pass:** regression ≤ 3pp (τ_r). Interaction check: if ≥ 2 validated skills, pairwise co-expression on 10 T_gate tasks; harmful pairs (Δ < −5pp together) → keep the higher-contribution trait only. Expression overhead τ_k ≤ 512 tokens/trait.

Gate outputs are CIG records (schema `cig/0.1`); rejected candidates stay in somatic memory (SPEC §9 fail path).

**G1 evaluation:** each arm's G1 runs T_eval (3 seeds for mining/run-level variance) + T_reg. Metrics: T_eval success (primary), regression rate on T_reg (primary), genome size (genes, expression tokens), descendant adaptation cost on a 20-task T_gate transfer subset (tokens/actions to first success — H3 preview).

## 5. Statistics (locked)

- Report per-task success with bootstrap 95% CIs (cluster bootstrap over tasks, 10k resamples) for all pairwise arm contrasts on T_eval.
- Seed-level consistency requirement: direction of C−B must hold in ≥ 2/3 seeds.
- H2 supported iff: (C − B) ≥ +5pp on T_eval with CI excluding 0 AND regression rate(C) < regression rate(B) with CI excluding 0 on the difference.
- H1 supported iff (C − A) CI excludes 0 (positive).
- No p-hacking escape hatch: secondary metrics are reported regardless of outcome; all gate thresholds and this plan are committed before the run.

**Secondary sensitivity analyses (pre-planned now, reported as clearly secondary; primary conclusions above remain the pre-locked ones and are never retuned):**
- *Threshold sensitivity:* recompute gate decisions on T_gate stage data only, varying τ_c ∈ {0.03, 0.05, 0.08} and τ_r ∈ {0.02, 0.03, 0.05}. Report how arm-C composition and T_eval outcomes would change under each — as analysis, not as a new primary result. `T_eval` is never used for threshold selection.
- *Candidate-cap sensitivity (follow-up, EXP-0005):* re-run the B/C arm construction with K ∈ {5, 10, 20} from the same frozen ranked candidate list.

## 6. Threats to validity (to monitor in post-run review)

1. Trait Miner quality variance → same miner, same prompts across arms; miner log committed.
2. Gate overfitting to T_gate → mitigated by disjoint T_eval and family stratification; check trait generality across families via the out-of-scope stage.
3. Regulated traits misvalued under unconditional ablation (SkillShapley lesson) → ablation evaluates skills under their `express_when` conditions.
4. ALFWorld stochasticity: simulator deterministic, LLM temp 0 at eval; mining-phase temp 0.7 variance covered by 3 seeds.
5. Baseline strength: if arm B ≈ arm C because few candidates are harmful, that is a *finding* (gate unnecessary in this regime), not a failed experiment — record as such.
6. **Scope-declaration gaming:** a trait could declare narrow applicability to face an easy in-scope check while arm B suffers its cost everywhere. Mitigations: (a) scope declared at extraction, frozen before gate stages, recorded in the CIG record; (b) out-of-scope harmlessness is still required; (c) trait utility in arm C is conditional on scope, but T_eval success is unconditional — misdeclared scope shows up as lost T_eval performance for arm C relative to arm B.

## 7. Budget estimate (order-of-magnitude, for planning)

Mining: 60 tasks × ~30 steps. CIG per candidate ≈ 5×5×2 (replay) + 20×3×2 (ablation) + 20×2 (gen) + 40 (reg) ≈ 350 episodes; × up to 10 candidates ≈ 3.5k episodes. Final: 3 arms × (134 + 40 + 20) × 3 seeds ≈ 1.7k episodes. Total ≈ 5–6k episodes × ~30 LLM calls ≈ 150–200k local-7B calls — feasible on one GPU node over a few days.

## 8. Pre-registration fields to fill BEFORE run start

- [ ] Expected result, H2 (direction + rough magnitude): ______
- [ ] Expected result, H1: ______
- [ ] Expected number of candidates passing each gate stage: ______
- [ ] Compute allocation and wall-clock cap: ______

## 9. Review resolution (critic round 1, 2026-10-02)

All five requested changes from PR #2 / issue #5 applied; decisions recorded in ADR-0002:

1. Generalization stage split into **in-scope** (Δ ≥ 0 on unseen instances matching declared applicability) and **out-of-scope** (Δ ≥ −0.05 on non-matching families) — specialist traits are no longer penalized for not improving unrelated types (§4 stage 3).
2. Raw 60/40 allocation replaced with **task-family-stratified quotas** (§2); `T_eval` remains the untouched official unseen split.
3. Replay now uses **valid same-family resampling** from a disjoint reserve; synthetic object/room substitution dropped for M0 unless later proven semantics-preserving (§4 stage 1).
4. **Candidate-set freeze** before arm branching: ranked list, top K=10, same frozen set for arms B and C, committed artifact; K-sensitivity {5,10,20} planned as EXP-0005 (§3, §5).
5. **Post-primary threshold-sensitivity plan** added (τ_c, τ_r grids on T_gate data only, clearly secondary, no T_eval retuning) (§5).
