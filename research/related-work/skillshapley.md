# SkillShapley — detailed notes

**Paper:** *SkillShapley: Boundary-Adaptive Shapley Valuation for Skill Step Attribution in LLM Agents* — Chang Liu, Yuqi Zhang, Yiman Zhong, Boyi Liu, Hengjun Wang (Beihang University); Shuyue Wei (corresponding, Shandong University). arXiv:2608.13173 (submitted 13 Aug 2026), preprint only. Reviewed 2026-10-02.

## Mechanism

- Valued unit: an **instruction block (step) inside a single fixed skill** (its `skill.md` body manually segmented into decision rules, API examples, validation, pitfalls; frontmatter always stays so the skill loads). Player = block; coalition = step-subset skill variant run on M=3 fixed benchmark instances (OpenHands harness, temperature 0); utility v(S) = average reward (often discrete {0, 1/3, 2/3, 1}).
- Two empirical findings drive the design: sharp performance cliffs in rewards; step interactions largely additive rather than synergistic.
- **BAES (Boundary-Adaptive Edge Shapley)** sampler: (a) warm-up evaluates anchors (empty, full, singletons, (n−1)-subsets) plus intermediate configs with many one-flip cached neighbors until Kendall-τ stratum rankings stabilize; (b) adaptive acquisition greedily evaluates the config maximizing a score combining high-variance/under-sampled strata and proximity to low-reward cached states, reusing each evaluation across all one-flip edges. Cost = unique configs evaluated; budgets B = 3n² (300/243/363 configs for their three skills) vs 512–2048 full enumeration.
- No trajectory permutation/replay — counterfactuals are step-removal variants. BAES is deliberately biased toward ranking recovery.

## Results

Three SkillsBench task–skill pairs (offer-letter-generator/docx n=10; manufacturing-FJSP-optimization n=9; dialogue-parser n=11). Attribution baselines: individual scores, LOO, random removal, LeastCore, exact Shapley (exact-Shapley top-step removal curves degrade fastest = best identifies critical steps). Approximation baselines: MC, QMC, paired MC, size-k truncation — BAES lower MAE vs exact at matched budgets (pilot: 99 configs → 206 reusable one-flip edges vs 130 for MC). Example attributions: FJSP P9 φ=0.1155; offer-letter P2 φ=−0.0194.

## Overlap vs HeritAgent chain

Attribution is used **only for skill editing/explanation** — estimate values → delete/reinforce low-value steps → validate via removal curves; authoring guidance. **No retention/inheritance decision mechanism, no threshold gate, no cross-generation lineage, no genome-level composition effects.** Single static skill on a fixed benchmark; a human consumes the φ signal. Related but distinct: SkillSV (line-level skill valuation for token pruning); Shapley-Coop (NeurIPS 2025, multi-agent credit assignment) — also not inheritance mechanisms.

## Limitations (stated)

Exact Shapley feasible only for few steps; requires fixed player set + stable benchmark signal (ill-defined for dynamic-length workflows or subjective criteria); assembly-line tightly-coupled steps inflate Shapley values (structural necessity ≠ usefulness); BAES biased with fill-in fallbacks.

## Consequences for HeritAgent

- Their additivity finding (step interactions mostly additive) supports CIG v0's default of cheap LOO/marginal ablation, escalating to Shapley-style sampling only on interaction evidence.
- If CIG stage-4 interaction testing becomes the compute bottleneck, BAES-style budgeted sampling is the candidate accelerator (cite; do not reimplement blindly — it is ranking-oriented, our gate needs calibrated effect sizes).
- Their "structural necessity ≠ usefulness" caveat maps to our regulatory-gene valuation problem: a skill that only fires under regulation may look useless under unconditional ablation → ablation must respect `express_when` conditions (design note for EXP-0001).
