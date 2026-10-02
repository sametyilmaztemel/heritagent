# ADR-0002 — Adopt EXP-0001 design (Milestone 0: minimal CIG experiment)

Date: 2026-10-02 · Status: PROPOSED (awaiting cross-review per SPEC §32) · Artifacts: `experiments/manifests/EXP-0001-design.md`, `EXP-0001-minimal-cig.yaml`

**Problem.** H2 (gated inheritance beats unrestricted inheritance) is the project's load-bearing hypothesis (SPEC §17); it must be tested at minimum cost before any population/evolution machinery is built (SPEC §27–29).

**Current design.** None — no experiment has been run.

**Proposed change.** Lock EXP-0001 as a 3-arm, 4-split, single-domain experiment: ALFWorld + Qwen2.5-7B-Instruct (local); arms A (no inheritance) / B (unrestricted = SkillRL-style skill persistence, frozen weights) / C (CIG-gated); four disjoint splits (T_mine/T_gate/T_eval/T_reg) with the contamination rule "T_eval never visible to miner, gate, or threshold tuning"; fixed v0 gate thresholds (τ_c = 5pp contribution with bootstrap LB > 0; τ_g = non-negative cross-type generalization; τ_r = 3pp regression; τ_k = 512 tokens); cluster-bootstrap statistics with pre-locked support conditions.

**Reason.** (a) ALFWorld/Qwen2.5-7B matches SkillRL's setup, making arm B a faithful frozen-weights reconstruction of the strongest adjacent baseline and enabling future direct comparison; (b) the four-split rule is the only way H2's held-out claim survives reviewer scrutiny — gate data and eval data must be disjoint; (c) local model makes the ~5–6k-episode CIG replay/ablation loop affordable and reproducible; (d) pre-locked thresholds and success conditions pre-empt confirmation bias (SPEC §33).

**Research implication.** Tests H2 (primary), H1 (secondary), H5-partial (bloat/regression), previews H3 (adaptation cost). Negative or null results are pre-declared as findings: if few candidates are harmful and B ≈ C, that is evidence about the regime, not a failed run (design §6.5).

**Experiment needed.** This ADR *is* the experiment design. Pre-registration fields (expected results, compute cap) must be filled before the run; model weights hash, split hashes, and miner prompt hash recorded at run start.

**Compatibility impact.** Requires: EAG v0 schema (ADR-0001) implemented for genome/manifest + somatic envelope; ModelAdapter (vLLM); trajectory recorder; Trait Miner v0 (SkillRL-adapted distillation — cited, not claimed); CIG stages 1–4 with record emission. No population manager, no mutation, no recombination, no lineage graph beyond parent pointers.

**Cross-review asks:** (1) adequacy of 60/40/20/20 task allocations given ALFWorld's ~6 task types; (2) whether replay "perturbed variants" (object/room substitution) are implementable cleanly in ALFWorld or should fall back to same-type resampling; (3) whether the ≤10-candidate cap biases against arm B's best case.
