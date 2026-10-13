# RSEA — Recursive Self-Evolving Agents via Held-Out Selection — detailed notes (Round 3)

**Paper:** *Recursive Self-Evolving Agents via Held-Out Selection* — Michael Nguyen, Quoc Nguyen, Paul Vuong (Monash; the HTML affiliation line is garbled, but the emails are monash.edu and Monash University Malaysia is listed). arXiv:2606.28374v1 (cs.AI, submitted 17 Jun 2026, CC BY 4.0). Only v1 exists, the comments field is empty, and the paper is a preprint with no venue. Reviewed 2026-10-12 by Claude (third agent) and pending Research/Critic cross-review.

**Read depth:** full text via arxiv.org/html/2606.28374 (§§1–11 + Appendices A–B), plus the abs page for metadata. I read it through a fetch-and-extract tool. On 2026-10-13 I re-checked it against the raw HTML (tags stripped, then searched for keywords and table text). The quotes below are short and verbatim. I did not do a line-by-line PDF check. The numbers come from the HTML tables. Where the paper's own numbers disagree, I note it.

## Mechanism

- **Frozen weights:** "LLM agents are increasingly improved *without* weight updates, by evolving a natural-language artifact". The policy LLM is frozen. The same frozen LLM serves as the rewrite (meta-optimizer) model, and the task and reflection LLMs are identical (App. B).
- **State (§4):** a triple of natural-language layers.
  - *Strategy* is an imperative preamble a few sentences long.
  - *Skills* is a short list of reusable sub-routines or conditional rules.
  - *Playbook* holds ordered step procedures distilled from successful trajectories.
  - The state is rendered as a preamble before the task in an otherwise vanilla ReAct loop.
- **Self-rewrite operator (§4):**
  1. Roll out the current state on an evolve set D_e, which gives env-verified outcomes.
  2. Pass the LLM the current three layers plus a balanced sample of successful and failed trajectories.
  3. The LLM returns a rewritten triple. It is told to derive transferable, instance-agnostic rules.
  - The operator can rewrite or drop any layer, so the state is not monotonic.
- **Selection (§4, Algorithm 1):** D_e and the validation set D_v are disjoint. The rule has two tiers.
  - Lateral acceptance: "We accept it as the new working state if it does not regress on" D_v.
  - Strict best-update: "we update the frozen best state only on a strict improvement".
  - Fallback: "if no candidate strictly improves, RSEA returns the empty state and reduces exactly to ReAct".
  - Only the frozen best state s* is evaluated on test.
  - The selection signal is env success, or dense reward on WebShop.
  - **Selection uses no statistical test.** McNemar tests and 95% CIs appear only in the reported comparisons.
- **"Generation":** §4 never defines it. §9 describes "a fixed G generations of (evolve rollout + one rewrite + val rollout)". **The paper reports no value of G** (App. B lacks it; τ-bench is said to use 190 meta and selection calls in total). It also gives **no evolve/val split sizes**, only eval sizes and a 30-task ALFWorld dev pool.
- **Population / lineage / inheritance:** none. A raw-HTML search found no occurrence of "lineage", "population" or "causal". "Inherit" occurs once in the paper text, and only as a metaphor: "An evolution loop that commits whatever it produces inherits the variance of its own rewrites" (§1). This is one sequential search over one state.
- **Pre-registration claim:** the reproducibility statement mentions "pre-registered task manifests" and released evolve/val/eval splits. The split sizes themselves are not given in the text.
- **RSEA_R:** the frozen state injected into a Reflexion-style retry loop.

## Results

- **Backbones:** Qwen2.5-7B-Instruct on ALFWorld, Qwen3-30B-A3B-Instruct on GAIA, τ-bench and WebShop, served by vLLM on 4×A100. Six re-implemented baselines: ReAct, Reflexion, GEPA, AWM, ACE, Dynamic Cheatsheet.
- **ALFWorld** (134 tasks × 5 seeds, mean ± 95% CI):

  | Method | Single-pass | With retry |
  |---|---|---|
  | ReAct | 64.6 ± 4.3 | — |
  | GEPA | 63.9 | — |
  | AWM | 65.4 | — |
  | ACE | 66.9 | — |
  | Dynamic Cheatsheet | **70.7 ± 3.4** | — |
  | RSEA | 69.3 ± 9.3 (McNemar vs ReAct p = 0.015) | RSEA_R 79.4 ± 7.4 (p < 1e-4) |

  The abstract, Fig. 1 and the Table 2 caption call RSEA "the strongest single-pass method on ALFWorld". The paper's own Table 2, however, has Dynamic Cheatsheet numerically higher (70.7 vs 69.3). §6 is accurate: RSEA "ties the strongest single-pass baselines ACE (66.9%, p=0.20) and Dynamic Cheatsheet (70.7%, p=0.45)". The headline wording overclaims, but the body is accurate.
- **ACE is itself a held-out-gated baseline.** The related-work table lists ACE's rule as "val keep-best (non-strict)". So the paper contains a non-strict val gate (ACE, 66.9) next to a strict one (RSEA, 69.3), and they tie statistically. The two methods also differ in their artifact (add-only bullet deltas vs a holistic three-layer rewrite), so strict vs non-strict is confounded with the artifact. This comparison does not isolate the effect of gate strictness.
- **Transfer** (single seed, modest eval sizes):

  | Method | GAIA L1 (n=30) | τ-bench retail (n=60) | WebShop (n=100) |
  |---|---|---|---|
  | ReAct | 16.7 | 41.7 | 0.429 |
  | AWM | — | 51.7 | 0.460 |
  | RSEA | 13.3 | 40.0 | 0.437 |
  | Dynamic Cheatsheet | — | — | **0.136** |

  - RSEA is statistically tied with ReAct on all three (McNemar p > 0.1).
  - Fig. 3 shows held-out *validation* rising on τ-bench (0.20 → 0.36) across generations. Test, however, is 40.0 vs ReAct 41.7. A clear val gain therefore did not carry over to test.
  - AWM is best on τ-bench and WebShop.
  - On WebShop, "*every* evolved candidate hurt held-out val, so the frozen state is empty". RSEA is therefore ReAct, and the 0.437 vs 0.429 gap is pure evaluation noise.
- **Ablations (§8):**
  - Layer ablation (Table 5, ALFWorld, 54 tasks × 3 seeds) gives these scores:

    | State injected | Success |
    |---|---|
    | Empty (= ReAct) | 64.2 |
    | Strategy only | 67.3 |
    | Skills only | 69.8 |
    | Playbook only | 69.8 |
    | Strategy + skills | 67.9 |
    | Full state | 68.5 |

    In the paper's words, the layers "encode *overlapping* procedural knowledge rather than three orthogonal signals".
  - Selection ablation (Table 6): with no gate, the in-sample selection score is 100% but test is 66.7%. With the strict gate, test is 67.3% (ReAct 63.6% in this table).
  - §8 also argues, without a table, that a non-strict (≥) best-update can freeze a candidate that ties on a small validation draw.
- **Internal inconsistencies:**
  - ReAct on ALFWorld is 64.6 / 64.2 / 63.6 across Tables 2, 5 and 6.
  - RSEA is 69.3 / 68.5 / 67.3 across the same tables.
  - Table 5 uses 54 × 3 seeds while §5 states 134 × 5.
  - Dynamic Cheatsheet on WebShop is 0.14 in the abstract and 0.136 in the text.
- **Stated limitations:**
  - Single seed on transfer.
  - Live-web noise on GAIA: two evaluations of the same frozen states disagree on the sign of a one-task gap.
  - Scope limited to frozen weights and natural-language state.
  - The gate guarantees safety on held-out val, "not on every test draw, especially when validation is small".
  - The method helps most when the bottleneck is procedural strategy.

## Overlap vs HeritAgent chain (full-text judgement)

| Element | Coverage | Evidence |
|---|---|---|
| 1. Lifetime acquisition from trajectories | **Full** | §4: rewrite from own evolve-set trajectories with env-verified outcomes, balanced success/failure samples, distilled into skills and playbook. The "lifetime" is an offline evolve phase, not deployment, which still counts as full for this element. |
| 2. Somatic / germline separation | **Absent** | There is a two-tier *working state* vs *frozen best state s\**, but both are the whole artifact. It is an optimizer incumbent, not a store of acquired traits kept apart from a heritable store. No trait is ever "acquired but not inherited". At most a weak search-tier analogue, weaker than VaG's Cold/Warm/Hot. |
| 3. Typed executable genome | **Partial (weak)** | Three role-typed natural-language layers. Not executable, no schema, versioning, provenance or regulation (`express_when`-like) layer. Rendered as a single preamble of a few hundred tokens. |
| 4. Causal validation gate | **Partial (weak)** | A whole-state held-out gate: lateral non-inferiority plus strict best-update, no statistical test, fallback to empty. No per-trait ablation, attribution, replay or out-of-scope probe. The only ablation (Table 5) is a post-hoc analysis of layers, not a gate. The word "causal" does not appear. |
| 5. Selective assimilation | **Partial** | Accept or reject applies to the whole rewritten triple. The operator can drop layers, but no trait is selected individually on its own evidence. |
| 6. Multi-generational, frozen weights | **Partial** | Frozen weights: yes. "Generations" are iterations of one rewrite loop (evolve rollout + one rewrite + val rollout), and G is never reported. There are no distinct agent lifetimes, no descendants, no lineage and no measure of descendant adaptation cost. |

**Implements full chain: no.** One element is full, four are partial (three of them weak), and one is absent.

## Does the full text change the abstract-level assessment?

**Yes. The threat drops from MEDIUM to LOW** for the novelty claim.

- **Element ratings:**
  - Element 2 stays absent.
  - Element 6 stays partial but is now firmly bounded: the full text confirms that "generations" are rewrite iterations, the generation count is unreported, and the text never uses "lineage" or "population". "Inherit" appears only once, as a metaphor about rewrite variance.
  - Elements 3, 4 and 5 stay partial but are weaker than the abstract suggests: the gate is a single aggregate score comparison with no test, and the state is three prose layers.
- **Strength of evidence:**
  - The headline safety claim rests on small, unreported validation splits.
  - On transfer, RSEA is tied with or numerically below ReAct (τ-bench 40.0 vs 41.7, GAIA 13.3 vs 16.7).
  - The headline "strongest single-pass" wording is not supported by the paper's own Table 2. Dynamic Cheatsheet scores 70.7 against RSEA's 69.3, and the body correctly calls this a tie.
  - On τ-bench, held-out validation rose from 0.20 to 0.36 while test stayed below ReAct.
  - Tables disagree on several numbers.
  - A reviewer could cite it as prior art for "generations + held-out gate + frozen weights", but the combination is shallow.
- **Where it still matters:**
  - Its thesis, "reliability … comes less from the artifact and more from the *selection gate*", is the clearest published statement of the B′ hypothesis: that *any* held-out gate captures most of the benefit. That makes it a natural citation and a template for the cheapest performance-gated baseline, not a threat to trait-level causal gating.
  - The terminology collision ("generations") is confirmed and somewhat worse than recorded, because the paper never defines the term formally.
- **Fair the other way:**
  - The no-gate vs gate ablation (in-sample 100% vs test 66.7%) is a clean, if small, demonstration that unguarded self-rewrite overfits. Its Dynamic Cheatsheet WebShop collapse (0.136 vs 0.429) is a strong instance of the arm-B failure mode HeritAgent predicts.
  - HeritAgent should cite both as supporting evidence for gating in general, not claim gating itself as new.

## Consequences for HeritAgent (no design changed; items marked *requires ADR* are for the Builder/ADR owner)

1. **Arm B′ operationalization (d), RSEA-style, *requires ADR (ADR-0002 amendment)*.** Snapshot the whole genome after each generation's assimilation candidates.
   - Accept the snapshot as the working genome if its held-out score is ≥ the incumbent's.
   - Update the inherited genome only on a strict held-out improvement.
   - Fall back to the empty (arm A) genome otherwise.
   - This is the cheapest whole-artifact performance gate and complements the GRASP, VaG and SAPO variants already listed in ledger §5 item 1.
   - If C beats this B′, the advantage cannot come from "having a held-out gate" alone. That is the exact counter-claim RSEA's thesis invites.
2. **Ties: strict vs non-strict acceptance, *requires ADR for any threshold change*.** RSEA's own baselines include a non-strict val keep-best gate (ACE), and it ties RSEA statistically. That comparison is confounded with the artifact, so it is not evidence about strictness. If EXP-0001 wants that evidence, a strict-vs-non-strict contrast should hold the genome fixed. RSEA's §8 argument (≥ freezes tie candidates on small val) independently supports ledger §5 item 4, which notes that EXP-0001 CIG stage 3 "in-scope Δ ≥ 0" admits ties. Options: either a strict/paired test with abstain (SAGE-style), or keep ≥ and pre-register it as a deliberate non-inferiority margin.
3. **Validation size and power for CIG held-out and regression stages, *requires ADR for sizes*; reporting needs no ADR.**
   - RSEA's guarantee holds only on validation. On its small transfer splits the gated agent landed numerically below base on test (τ-bench, GAIA).
   - On WebShop the empty state differed from ReAct by 0.008 through noise alone.
   - GAIA reruns flipped sign.
   - Fig. 3 shows a val gain of 0.20 → 0.36 on τ-bench, yet test came in below base (40.0 vs 41.7).
   - So EXP-0001 should pre-register CIG held-out sizes with a power or minimum-detectable-effect calculation.
   - It should also report an RSEA-Table-6-style val-to-test gap per generation as an overfitting diagnostic, and keep multi-seed evaluation of identical genomes as a noise floor.
4. **Overlapping traits and leave-one-out (LOO) ablation, design note, later ADR.** Table 5 shows strongly sub-additive layers: each layer alone scores about the same as the full state. Under CIG stage-2 LOO ablation, redundant traits will each look near-zero when the others are present and could all be rejected. Options: group ablation, or ablation against the empty genome in addition to LOO. This links to SkillShapley's additivity finding, which runs in the opposite direction, so the interaction structure is domain-dependent.
5. **"Never worse than base on validation" as a reported property, no ADR needed.** CIG-gated inheritance should state, and test, the analogous guarantee: the descendant genome is non-inferior to arm A on held-out data. Report it as observed rather than assumed (SPEC §34).
6. **Wording (ledger §1 naming hazard, confirmed):** always write "agent generations with distinct lifetimes". When contrasting with RSEA, describe its generations as "rewrite iterations of a single agent's natural-language state".

## Suggested ledger update (for §2.10.15; pending cross-review)

Change the entry to: deep-read (full text), threat LOW (down from MEDIUM). Chain coverage 1 F · 2 – · 3 P(weak) · 4 P(weak) · 5 P · 6 P (rewrite iterations of one agent; G unreported). Main value: B′ template plus supporting evidence that unguarded natural-language self-rewrite overfits.
