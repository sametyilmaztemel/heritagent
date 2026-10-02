# ADR-0002 — Adopt EXP-0001 design (Milestone 0: minimal CIG experiment)

Date: 2026-10-02 · Status: REVISED — critic round 1 applied, re-review requested (SPEC §32) · Artifacts: `experiments/manifests/EXP-0001-design.md`, `EXP-0001-minimal-cig.yaml`

**Problem.** H2 (gated inheritance beats unrestricted inheritance) is the project's load-bearing hypothesis (SPEC §17); it must be tested at minimum cost before any population/evolution machinery is built (SPEC §27–29).

**Current design.** None — no experiment has been run.

**Proposed change.** Lock EXP-0001 as a 3-arm, 4-split, single-domain experiment: ALFWorld + Qwen2.5-7B-Instruct (local); arms A (no inheritance) / B (unrestricted = SkillRL-style skill persistence, frozen weights) / C (CIG-gated); **task-family-stratified** splits (T_mine/T_gate with sub-pools/T_eval/T_reg) with the contamination rule "T_eval never visible to miner, gate, or threshold tuning"; **frozen ranked candidate set** (K=10) shared by arms B and C before branching; gate stages with **in-scope/out-of-scope generalization** and same-family-valid-resampling replay; fixed v0 gate thresholds (τ_c = 5pp contribution with bootstrap LB > 0; in-scope Δ ≥ 0, out-of-scope Δ ≥ −5pp; τ_r = 3pp regression; τ_k = 512 tokens); cluster-bootstrap statistics with pre-locked support conditions plus pre-planned **secondary** threshold- and cap-sensitivity analyses (no T_eval retuning).

**Reason.** (a) ALFWorld/Qwen2.5-7B matches SkillRL's setup, making arm B a faithful frozen-weights reconstruction of the strongest adjacent baseline and enabling future direct comparison; (b) the four-split rule is the only way H2's held-out claim survives reviewer scrutiny — gate data and eval data must be disjoint; (c) local model makes the ~5–6k-episode CIG replay/ablation loop affordable and reproducible; (d) pre-locked thresholds and success conditions pre-empt confirmation bias (SPEC §33).

**Research implication.** Tests H2 (primary), H1 (secondary), H5-partial (bloat/regression), previews H3 (adaptation cost). Negative or null results are pre-declared as findings: if few candidates are harmful and B ≈ C, that is evidence about the regime, not a failed run (design §6.5).

**Experiment needed.** This ADR *is* the experiment design. Pre-registration fields (expected results, compute cap) must be filled before the run; model weights hash, split hashes, and miner prompt hash recorded at run start.

**Compatibility impact.** Requires: EAG v0 schema (ADR-0001) implemented for genome/manifest + somatic envelope; ModelAdapter (vLLM); trajectory recorder; Trait Miner v0 (SkillRL-adapted distillation — cited, not claimed); CIG stages 1–4 with record emission. No population manager, no mutation, no recombination, no lineage graph beyond parent pointers.

**Cross-review asks:** all five points from the critic round are resolved below; no unresolved questions remain from the builder side.

## Review resolution — critic round 1 (2026-10-02, PR #2 / issue #5)

| # | Critic requirement | Resolution |
|---|---|---|
| R1 | Split generalization into in-scope (help where applicable) and out-of-scope (harmless where not applicable) — do not penalize specialist traits | Stage 3 split: in-scope Δ ≥ 0 on unseen instances matching the trait's declared applicability; out-of-scope Δ ≥ −0.05 on non-matching families. Scope declared by the miner at extraction, frozen before gate stage 1, recorded in the CIG record (design §4) |
| R2 | Replace raw 60/40 allocation with task-family-stratified quotas; T_eval untouched | Stratified quotas with per-family minimums; T_gate (~60) organized into sub-pools (ablation 20 / generalization reserve 20 / interaction 10 / replay reserve 10); T_eval = official unseen split, untouched (design §2) |
| R3 | Prefer valid same-family resampling over synthetic object/room substitution unless proven semantics-preserving | Replay variants = 4 unseen same-family instances from the disjoint replay reserve; synthetic substitution dropped for M0 with an explicit re-admission condition (design §4 stage 1) |
| R4 | Freeze the candidate set before arm branching; same candidates for B and C; record ranking and cap logic; plan K={5,10,20} sensitivity | Frozen ranked list (miner-estimated generality, discovery-order tie-break), top K=10, committed artifact before branching; K-sensitivity planned as EXP-0005 (design §3, §5) |
| R5 | Add a post-primary threshold-sensitivity plan (secondary, no retuning on T_eval) | Pre-planned secondary analysis: τ_c ∈ {0.03, 0.05, 0.08}, τ_r ∈ {0.02, 0.03, 0.05} grids computed on T_gate stage data only; primary conclusions immutable (design §5, manifest `sensitivity_secondary`) |

## Review resolution — critic round 2 (2026-10-02, PR #2 / issue #5)

| # | Critic requirement | Resolution |
|---|---|---|
| R1 | Remove pseudo-replication from deterministic temp=0 repeated runs; uncertainty from instances + mining seeds; pre-register any stochastic source explicitly | Determinism/stochastic-source policy added (design §4): exactly **one deterministic execution per (task instance, condition)** in every gate stage; duplicate deterministic runs are never independent evidence; uncertainty = distinct task instances (cluster bootstrap) + independent mining seeds; future repeated stochastic runs require an explicitly pre-registered stochastic source (never backend nondeterminism). Stage thresholds re-expressed over instances: replay ≥ 3/5 success-with and Δ ≥ +2/5; ablation CI over the 20 instances. Per-candidate gate cost drops ~350 → ~130 episodes |
| R2 | Define per-trait sampling reuse semantics for gate pools (within-trait no-replacement; cross-trait reuse allowed+logged, or enlarge pools) | Locked: no replacement **within** a trait's evaluation set; **cross-trait reuse allowed and logged** (per-trait allocations in CIG records) as the intended paired design for comparability; applies to both the generalization reserve and the replay reserve (design §2, §4 stage 3; manifest `sampling_reuse`) |
| R3 | Frozen machine-readable applicability before gating, not a runtime oracle | `applicability.task_families[]` on skill payloads: produced at extraction, frozen with the candidate set, **evaluation-only** (stage-3 sampling/audit); never a runtime regulation predicate or agent-visible oracle; agent-facing scope remains free-form `when_to_apply`. Field normatively defined in the EAG schema (PR #1, commit `9ef2f5a`) and stripped by the SkillRL projection adapter (design §3; manifest `genome.applicability`) |

## Review resolution — final round (2026-10-02, PR #2 / issue #5)

| # | Critic requirement | Resolution |
|---|---|---|
| R1 | Global replay/generalization reserves too small for per-trait same-family/in-scope counts → family-indexed banks with guaranteed per-family counts (or pre-registered family-specific maxima) | Global reserves replaced by `replay_bank[family]` (≥ 4 valid unseen instances per family) and `generalization_bank[family]` (≥ 10 per family), disjoint from T_mine/T_reg/ablation/interaction/T_eval; stage 1 draws from `replay_bank[family-of-source-task]`, stage-3 in-scope from `generalization_bank[f]` over the trait's declared families. Feasibility fallback: family-specific **pre-registered maximum** recorded in the split manifest before run start when ALFWorld availability prevents the minimum. Reuse semantics unchanged: cross-trait reuse allowed + logged, within-trait no replacement. Bank construction adds no LLM calls; per-trait counts and total budget unchanged (design §2, §4, §7; manifest `evaluation_banks`) |
| R2 | State explicitly that `applicability.task_families[]` is stripped from all runtime model inputs in every arm | Runtime input hygiene added (design §3; manifest `genome.runtime_input_policy`): evaluation-only metadata stripped from **all arms'** runtime inputs (A, B, C — not only the strict SkillRL projection); runtime prompt assembly consumes only the projected payload (`name`, `principle`, `when_to_apply`, `procedure`); tests assert absence of evaluation-only fields (checklist for #4/#7) |
