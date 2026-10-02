# EXP-0001 — Minimal Causal Inheritance Gate experiment (design, pre-registration)

**Status: DESIGN — locked before implementation; expected-result fields must be filled in before the run starts (SPEC §33).** Machine-readable manifest: `EXP-0001-minimal-cig.yaml`. Decision record: `research/decisions/ADR-0002-exp-0001-design.md`.

This is Milestone 0 (SPEC §29): one model, one domain, one agent, two generations. It exists to test the load-bearing hypothesis H2 before anything else is built (SPEC §27).

## 1. Hypotheses under test

- **Primary — H2 (Causal Gate):** CIG-gated inheritance produces better held-out performance and lower regression than unrestricted skill inheritance.
- **Secondary — H1 (Assimilation, directional):** assimilated traits improve descendant zero-shot performance vs no inheritance.
- **Observational — H5 (partial):** genome size/bloat comparison between gated and unrestricted arms.

## 2. Environment and model

- **Environment: ALFWorld** (text-based embodied tool use; same family as SkillRL's setup for future comparability). Deterministic simulator; LLM is the only stochastic component. Eval temperature 0.
- **Model:** Qwen2.5-7B-Instruct, local (vLLM), via ModelAdapter. Rationale: matches SkillRL's base model; local → reproducible, cheap CIG replay loops; supports later H4 backbone swap.
- **Split discipline — the four-disjoint-split rule.** Contamination between gate data and final evaluation would invalidate H2, so:
  - `T_mine` (60 tasks, from ALFWorld train split): G0's lifetime; trajectories → Trait Miner.
  - `T_gate` (40 tasks, train split, disjoint types-instances from T_mine): all CIG stages (replay variants, ablation, gate-generalization, regression checks).
  - `T_eval` (134 tasks, official ALFWorld unseen eval split): **final evaluation only** — never visible to miner, gate, or threshold tuning.
  - `T_reg` (40 tasks, subset of T_mine that G0 solved): regression suite (arm-internal, run at G1 evaluation time).
  - Splits are content-hashed and committed with the manifest; any change = new experiment id.

## 3. Arms (identical G0, identical seeds, identical eval budget)

| Arm | Inheritance policy | G1 genome |
|---|---|---|
| A — no inheritance | none | G0 genome unchanged (skills empty) |
| B — unrestricted | all mined candidate traits injected as skill genes | G0 ∪ all candidates |
| C — CIG-gated | only candidates passing all four gate stages | G0 ∪ validated candidates |

Arm B operationalizes the SkillRL-style persistence policy (library growth + always-available skills) without weight training — the spec's §17 "most critical comparison". Note: arm B will have larger prompts; prompt overhead is part of the treatment being tested (bloat is an outcome, per H5), not a confound to normalize away. Equal **evaluation budget** (max steps per task) is enforced across arms.

## 4. Pipeline

**G0 lifetime:** G0 (seed genome: ReAct-style planner + retry policy, no skills) runs T_mine; trajectories recorded (full functional phenotype per SPEC §7). Trait Miner v0: LLM differential distillation over success/failure pairs (SkillRL-adapted, cited) → candidate skill genes (`{name, principle, when_to_apply, procedure}` + somatic envelope). Cap: ≤ 10 candidates per run (keeps CIG budget bounded; cap itself recorded).

**CIG stages (arm C only), initial thresholds (SPEC §10 allows adaptive thresholds later; v0 fixed):**

1. **Replay validation** — source task + 4 perturbed variants (object/room substitutions) from T_gate distribution; 5 runs each, with vs without trait, eval temp 0. **Pass:** mean success-with ≥ 0.6 AND Δsuccess ≥ +0.4 on the replay cluster; trait must fire at least once per successful episode (expression check, not just presence).
2. **Controlled ablation / contribution** — 20 T_gate tasks, 3 runs each with vs without. Contribution = Δsuccess; **pass:** bootstrap 95% CI lower bound > 0 AND point estimate ≥ +0.05 (τ_c).
3. **Held-out generalization** — 20 further T_gate tasks of task types *different* from the source task's type where available. **Pass:** Δsuccess ≥ 0 (no degradation) — we do not require gains here; we require the trait not to be source-instance memorization that hurts elsewhere.
4. **Interaction + regression** — re-run T_reg (40 tasks) with genome ∪ trait; **pass:** regression ≤ 3pp (τ_r). Interaction check: if ≥ 2 validated skills, pairwise co-expression on 10 T_gate tasks; harmful pairs (Δ < −5pp together) → keep the higher-contribution trait only. Expression overhead τ_k ≤ 512 tokens/trait.

Gate outputs are CIG records (schema `cig/0.1`); rejected candidates stay in somatic memory (SPEC §9 fail path).

**G1 evaluation:** each arm's G1 runs T_eval (3 seeds for mining/run-level variance) + T_reg. Metrics: T_eval success (primary), regression rate on T_reg (primary), genome size (genes, expression tokens), descendant adaptation cost on a 20-task T_gate transfer subset (tokens/actions to first success — H3 preview).

## 5. Statistics (locked)

- Report per-task success with bootstrap 95% CIs (cluster bootstrap over tasks, 10k resamples) for all pairwise arm contrasts on T_eval.
- Seed-level consistency requirement: direction of C−B must hold in ≥ 2/3 seeds.
- H2 supported iff: (C − B) ≥ +5pp on T_eval with CI excluding 0 AND regression rate(C) < regression rate(B) with CI excluding 0 on the difference.
- H1 supported iff (C − A) CI excludes 0 (positive).
- No p-hacking escape hatch: secondary metrics are reported regardless of outcome; all gate thresholds and this plan are committed before the run.

## 6. Threats to validity (to monitor in post-run review)

1. Trait Miner quality variance → same miner, same prompts across arms; miner log committed.
2. Gate overfitting to T_gate → mitigated by disjoint T_eval; check trait generality across task types in stage 3.
3. Regulated traits misvalued under unconditional ablation (SkillShapley lesson) → ablation evaluates skills under their `express_when` conditions.
4. ALFWorld stochasticity: simulator deterministic, LLM temp 0 at eval; mining-phase temp 0.7 variance covered by 3 seeds.
5. Baseline strength: if arm B ≈ arm C because few candidates are harmful, that is a *finding* (gate unnecessary in this regime), not a failed experiment — record as such.

## 7. Budget estimate (order-of-magnitude, for planning)

Mining: 60 tasks × ~30 steps. CIG per candidate ≈ 5×5×2 (replay) + 20×3×2 (ablation) + 20×2 (gen) + 40 (reg) ≈ 350 episodes; × up to 10 candidates ≈ 3.5k episodes. Final: 3 arms × (134 + 40 + 20) × 3 seeds ≈ 1.7k episodes. Total ≈ 5–6k episodes × ~30 LLM calls ≈ 150–200k local-7B calls — feasible on one GPU node over a few days.

## 8. Pre-registration fields to fill BEFORE run start

- [ ] Expected result, H2 (direction + rough magnitude): ______
- [ ] Expected result, H1: ______
- [ ] Expected number of candidates passing each gate stage: ______
- [ ] Compute allocation and wall-clock cap: ______
