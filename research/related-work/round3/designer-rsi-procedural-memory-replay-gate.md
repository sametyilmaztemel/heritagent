# Designer-RSI — Evolving Procedural Memory from User Traffic for Agentic Graphic Design — detailed notes (Round 3)

**Paper:** *Designer-RSI: Evolving Procedural Memory from User Traffic for Agentic Graphic Design*. Authors: Hongyang Du, Lan Yan, Christian Flores, Asim Kadav. The HTML lists the affiliations as Adobe and Brown University, with the corresponding author at brown.edu. arXiv:2609.22086v1 (cs.AI; cross-listed cs.CV), submitted 18 Sep 2026, CC BY 4.0. The comments field reads "9 pages, 7 figures". Only v1 exists, and no venue is given. Reviewed 2026-10-13 by Claude (third agent); pending Research/Critic cross-review.

**Naming note:** the "Designer-RSI" prefix appears only in the arXiv metadata title. The HTML/LaTeX title and body never use "Designer-RSI" or "RSI". The system is called "Evolve" in tables, and the loop is called the "evolving loop" or "harness". Cite the paper by the arXiv title and id.

**Read depth:** full text from arxiv.org/html/2609.22086: §§1–6 plus Appendices A–K, with the abs page for metadata. I downloaded the raw HTML and stripped the tags. Math alttext was kept so the numbers could be read. The quotes below are short and verbatim from that text. Figures 4–5 were read from their captions and the inline Fig. 5(b) table, not from the rendered images. I did not do a line-by-line check against the PDF.

## Mechanism

- **Frozen weights:** a frozen frontier model drives equivalents of Photoshop, Illustrator and InDesign through 230+ tools. The paper states: "The foundation models, tools, renderer, evaluator, and evolution roles remain fixed; only the skill library changes." Disabling skill retrieval gives back the original agent, and that is the control (§2).
- **Roles (§3):**
  - Prompter: poses design briefs drawn from user data plus LLM variants.
  - Solver: the agent itself.
  - Grader: a multimodal judge, "Graphic-Eval".
  - Reflector: turns failures into SKILL.md edits.
  - Only the SKILL.md files change.
- **Skill representation (App. C.1).** Each skill is one Markdown SKILL.md with YAML front matter. This is a typed, versioned schema:
  - `description`: 200–600 characters, and the sole retrieval signal.
  - `app_mode` ∈ {raster, vector, page_layout, any}.
  - `tool_sequence`: the 5–10-tool critical path.
  - `prerequisites`: other skills, pulled in one hop.
  - `status` ∈ {candidate, stable, deprecated}, plus a version counter.
  - The body has five required sections: when to use, clarify, steps (each naming a tool and its parameters), tips, and an error-handling table.
  - "Per-skill statistics live outside the file, in a separate store, so the skill itself stays in clean, shippable form."
  - The paper never says how `status` is used in the loop.
- **Retrieval and injection (App. C.2–C.3):**
  - A frozen LLM distills the request into ≤6 operation keywords. These are scored against skill descriptions by token overlap, which ranks 2–4 skills per turn.
  - Progressive disclosure: the model sees a catalog and calls `load_skill(name)`.
  - Tool-list narrowing is supported but "off by default in our experiments".
- **Cold start (App. G):** 76 skills are generated from product help documentation in four phases. Two phases use no LLM, and there are human review checkpoints after Phases 1 and 3. So "no human labels" applies to evolution only, not to the seed bank.
- **Widening (§3.1):**
  - A frozen LLM extracts and canonicalizes the subtasks in each trajectory.
  - A subtask is uncovered if no retrieved skill addresses it. Subtasks tied to a skill blamed for a poor outcome also count.
  - Uncovered subtasks go into a "persistent coverage pool keyed by canonical label".
  - At k_min = 3 occurrences, a frozen LLM distills a candidate skill.
  - The candidate is gated **against the no-skill agent**. If rejected, it is discarded, but its occurrences stay in the pool.
- **Deepening (§3.2, App. E):**
  - Scoring: s_j = 0.7·completeness + 0.3·aesthetic. A trajectory with s_j < τ = 0.6 is a failure, and the failure counts "against every skill it retrieved, with no blame weighting".
  - A round fires every 50 new graded records. Every skill with failure count ≥ m = 2 is selected, capped at 50 per round.
  - The Reflector gets the failure rationales plus this skill's own successful calls (tool sequences, waypoints, thinking tokens) and emits a targeted section edit.
  - After repeated gate failures it escalates to a major rewrite. An optional exploration cycle can then decide update, delete or keep, and "keep" routes the records to the coverage pool.
  - In the paper's words: "selection is a cheap, permissive heuristic, while the gate—not the heuristic—decides what ships."
- **Replay gate (§3.3, App. F).** This is the core of the comparison with the CIG.
  - *Explicit confound control.* (i) The gate never uses absolute judge scores, because of VLM drift. (ii) "asset retrieval and other upstream state can differ between arms". So each prompt gets "several contexts", each context is frozen, and both arms are replayed fresh in the same batch: "within each context the only difference under test is the skill condition."
  - *Arms.* A rewrite (V2) runs against the incumbent (V1). A mint runs against the agent with no skill retrieved.
  - *Replay set.* For a rewrite: 2 well-scoring + 2 poorly-scoring prompts from trajectories that retrieved the skill. For a mint: prompts from the triggering cluster, not conditioned on score.
  - *Judging.* Pairwise judging with presentation order swapped gives a per-record win rate r. A tie band of half-width δ sits around 0.5, and **δ is not reported**. A prompt counts as won only if the candidate wins a majority of its contexts. **The number of contexts per prompt is not reported.**
  - *Acceptance (Eq. 5):* accept ⇔ (no record with r < 0.5−δ) ∧ (some record with r > 0.5+δ). "The rule never trades a regression on one case for a gain on another." Ties everywhere are rejected. The gate is motivated by safe policy improvement (Thomas 2015; Laroche 2019), but it uses **no confidence bound or statistical test**.
  - Rejected proposals are logged with before/after snapshots, so the history records "what was tried and refused".
- **Generations / inheritance / causality:** none. A text search of the full extracted text found zero occurrences of "inherit", "lineage", "causal" and "counterfactual". "Generation" always means image generation. There are five "rounds" of one deployed bank.

## Results

- **Evolution:**
  - 1,406 non-overlapping briefs (user traffic plus LLM-augmented variants), 1,869 graded trajectories, 5 rounds. The bank grows 76 → 139.
  - Gate rejections: 100/231 rewrites (131 committed) and 67/136 mints (69 committed). 6 skills were discarded or merged.
  - R1 is mostly repair (39/59 rewrites and 4/10 mints pass). Minting peaks in R2–R3 (22/46 and 26/40).
  - Minted skills are not paraphrases: nearest-neighbour distance is 0.215 vs 0.158 median, Mann–Whitney p < 1e-8, Cliff's δ = 0.58.
  - **The backbone used to generate the evolution trajectories is not stated.**
- **Internal held-out benchmark:**
  - 200 human-authored briefs, disjoint from the evolution set and fixed across rounds. Claude-Sonnet-4. The App. J gallery shows "the same 193 prompts", a discrepancy the paper does not explain.
  - The completeness metric uses the same Graphic-Eval judge family as the evolution reward. App. D says this is deliberate: "a skill is only rewritten against the criterion it will later be measured on."
  - R5 vs no skill: ≥0.5 completeness goes 86% → 93%, ≥0.9 goes 43% → 56% (+13 pp), and =1.0 goes 24% → 32%.
  - **Non-monotone:** R4 falls below no skill at ≥0.3 (90% vs 94%). The authors attribute this to never-revised minted skills that "can misfire on requests outside that cluster", since a mint is verified only against the tools-only baseline of its own gap cluster.
- **External benchmarks:** 300 prompts each. A single run is implied; no seeds or CIs are given.
  - T2I average quality: +3.83 (Opus-4.6), +7.70 (Sonnet-4) and +9.67 (Qwen3.6-27B).
  - Per-benchmark regressions exist: Opus GenEval2 −3.32; Qwen DPG-Bench −3.20 quality and −15.3 success.
  - Sonnet GenEval2 success 72.7 → 99.3.
  - Design win rates vs Base, judged by GPT-5.4: 67.6% (Opus), 61.8% (Sonnet), 62.8% (Qwen). Qwen GraphicBench is 49.0%.
  - The bank is reused across three backbones. This is cross-backbone transfer of a frozen bank, with the evolving backbone unstated.
- **Ablation (§5, Table 3):** 200 unseen prompts, one batch.

  | Arm | Completeness | Win rate vs Base |
  |---|---|---|
  | Base | 69.08 | — |
  | Cold start | 68.62 | 46.4% |
  | + rewrite (cold-start skills at latest version) | 69.02 | 48.6% |
  | + new skills (all 139 skills at v1) | 69.79 | 49.4% |
  | Evolve | 74.04 | 58.5% (p = 0.025) |

  - Interaction: 74.04 − 68.62 − (0.40 + 1.17) = +3.85, which the paper calls "superadditive".
  - **The test behind p = 0.025 is not named**, and no CI is given.
  - The single-axis arms are **post-hoc sub-banks of the one co-evolved run**, not separately evolved runs. Each mint and rewrite was gated in the presence of the other axis, so the ablation does not show what widening-only evolution would produce.
  - **There is no ungated or performance-gated comparison arm.** The paper never evaluates the loop without the replay gate, so the gate's own contribution is unmeasured.
- **Stated limitations:**
  - "The replay gate is also local to the evaluated cases: rejecting observed regressions on the replay set does not guarantee monotonic improvement over the full user-traffic distribution."
  - Natural-language skills cannot override strong model defaults (App. I).
  - Long skills lose fidelity.
  - Prompts and trajectories are not released (internal benchmark, content policy).

## Chain coverage vs HeritAgent (SPEC §3)

| Element | Verdict | Evidence |
|---|---|---|
| 1. Lifetime acquisition | **Full** | New skills are minted from recurring uncovered subtasks in deployed user traffic (§3.1). Existing skills are revised from their own successful and failed trajectories (§3.2). The seed bank comes from documentation, not experience, so it is not part of this. |
| 2. Somatic/germline separation | **Absent** (weak analogues only) | There is one deployed bank, and proposals are transient: rejected ones are discarded. Weak analogues: a `status ∈ {candidate, stable, deprecated}` field whose lifecycle is never described; a persistent coverage pool, which holds *evidence* (subtask occurrences), not traits; personalized user skills "excluded from the public skill pool" (App. I, K), which is a private-vs-shared split but not a promotion pathway; per-skill statistics kept outside the shippable file. None of these is a somatic staging store with assimilation as a promotion event. |
| 3. Typed executable genome | **Partial** (upgraded from the abstract-level "absent") | Typed YAML front matter (`app_mode` enum, `tool_sequence`, `prerequisites` dependencies, `status` enum, version counter); a mandatory five-section body including an error-handling situation→action table; tool names validated against the live catalog at seeding (App. G). Against the EAG it is still a flat library of natural-language procedures. It has no policy or regulation layer, no per-trait validation record attached to the artifact (statistics live in a separate store), no genome-level versioning or lineage, and the tool sequences are guidance, not executed code. |
| 4. Causal validation gate | **Partial** (upgraded from the abstract-level "not matched") | This is a per-change controlled paired intervention, with explicit confound control: frozen per-context upstream state, same-batch replay, order-swapped pairwise judging, mint vs no-skill as a with/without test of one trait, and a no-loss + ≥1-win rule. It covers roughly CIG stages 1–2 (replay + controlled with/without) and a local slice of stage 5 (regression protection on 2 good prompts). Missing: attribution under interactions (mints are tested against the *no-skill* agent, not the current bank, and the paper itself reports cross-cluster misfires in R4); held-out generalization in-scope or out-of-scope (replay prompts come from the skill's own trajectories or cluster); statistical inference (no test, no CI, δ and the context count unreported, n = 4 prompts for rewrites); and any heritability role, since the gate decides deployment within one continually updated system. The paper never uses the word "causal". |
| 5. Selective assimilation | **Partial** | Gated admission into the deployed bank, with 167 of 367 proposals rejected. The target is a live deployed library, not a heritable genome, and admission is per change (rewrite or mint), not per individually validated lifetime trait leaving a staging store. |
| 6. Multi-generational, frozen weights | **Partial** | Frozen weights: full. Five rounds of one bank are continual adaptation, with no agent generations, distinct lifetimes, lineage or descendants. The round-5 bank is reused across three frozen backbones, which is a portability result, not inheritance. |

**Implements the full chain: no.** Coverage vector: 1 F · 2 – · 3 P · 4 P · 5 P · 6 P.

## Change vs the abstract-level triage

- **Elements 3 and 4 go up from absent to partial.** The full text shows a typed, versioned skill schema. It also shows a gate that is more than a "regression check against past successes": it is an explicitly confound-controlled, per-change with/without (mint) or candidate-vs-incumbent (rewrite) replay. Element 6 is scored partial for consistency with §2.10 entries such as RSEA. The triage listed it as matched, but the full text confirms only frozen weights and rounds, not generations.
- **Threat stays MEDIUM; it does not go up.** The added overlap sits squarely inside the already-crowded within-lifetime pre-persistence-gate cluster (§1 row 4; §3 items 9–11: GRASP, VaG, SAGE, SkillBoost, HarnessEvolve). Designer-RSI adds a stricter variant (no-loss rather than a net-gain budget) and a new domain (open-ended graphic design with no oracle and a VLM judge), but no heritability, no staging store, no genome and no gated-vs-ungated evidence.
- **Two details lower the threat to H2 specifically.** (a) The paper has no ungated or performance-gated comparison, so it adds no evidence that a gate beats unrestricted persistence. (b) Its own R4 observation says the opposite of what a strong gate claim needs: changes admitted by a local, in-cluster gate still produced a distribution-level regression. That supports, as one anecdotal observation, the motivation for CIG's held-out and interaction-attribution stages. It is one run, n = 200, no CI, so it can be cited only as "reported", never as evidence for HeritAgent.

## What remains different

1. No somatic/germline split. There is one live deployed bank, transient proposals, and no promotion event from a staging store.
2. The skill schema is typed but is a flat procedural-memory library. There are no policies or regulation, no per-trait validation record in the artifact, and no genome or lineage versioning.
3. The gate is a local paired replay with no held-out or out-of-scope generalization, no interaction-aware attribution (mints tested against no-skill, not against the bank), no statistical test, small replay sets and unreported δ and context counts. It gates deployment, not heritability.
4. There are no agent generations, inheritance, descendants or adaptation-cost measurement. "Rounds" are continual updates of one system.
5. There is no ungated or performance-gated control, so the gate's effect is unmeasured. The single-axis ablation arms are post-hoc sub-banks.

## Overlap HeritAgent must acknowledge (candidate §3 additions — pending cross-review)

- **Strict no-regression (no lost case, ≥1 won case) per-change acceptance with matched, frozen-context, same-batch paired replay to control upstream-state and judge-drift confounds**, motivated by safe policy improvement. This is in an oracle-free domain with a VLM pairwise judge, which extends §3 items 10–11.
- **Coupled acquisition ("widening") and revision ("deepening") of a skill bank under one gate; recurrence-thresholded minting from a persistent uncovered-subtask pool; failure-count-triggered revision with success-contrast "do-not-regress" context; escalation from targeted to major rewrite; rejected-proposal logging.** This extends items 5 and 14.
- **Typed SKILL.md front matter with app scope, tool critical path, prerequisites, status and a version counter, with per-skill statistics held outside the shippable artifact.** This is another typed-skill-schema analogue for item 22. The "statistics outside the artifact" choice parallels the §5.6 "gate outside the genome" invariant.

**Citation/naming hazard found:** Designer-RSI cites "SkillAudit: Ground-truth-free skill evolution via paired trajectory auditing" (Gao et al., 2026) as arXiv:2606.14239. The current arXiv title of 2606.14239 is *SelfEvoSkill: Paired Execution Audits for Skill Revision without External Outcome Supervision*, with the same first authors (Haowen Gao, Haoran Chen, Can Wang, Shasha Guo …). I checked this on the abs page on 2026-10-13. The ledger lists SelfEvoSkill 2606.14239 as "opened and judged low/no relevance". That judgement should be re-checked, because it pairs with/without trajectories of a candidate skill to localize refinements, which is a §3 item-10 mechanism. Always cite it by arXiv id plus its current title.

## Implications for EXP-0001 / CIG — all **require ADR**; nothing was changed

1. **Arm B′ operationalization (§5.1), requires an ADR-0002 amendment.** Designer-RSI's Eq. 5 is a fourth candidate rule, "(d) strict no-loss / ≥1-win paired replay vs incumbent (or vs no-trait for new traits) on a 2-good + 2-bad replay set." It is cheap and published, and it removes all trait-level attribution, held-out and out-of-scope stages, which makes it a natural performance-gated baseline. If chosen, the tie band δ and the number of contexts per prompt must be pre-registered, because the paper does not report them.
2. **Statistics for CIG stages, requires an ADR if thresholds change.** The paper's per-context frozen upstream state and same-batch replay of both arms is a variance and confound control worth adopting in CIG stage 1/2 replay. It cancels judge drift and environment-state differences, and it is cheap to add. Its no-loss rule over a handful of prompts is underpowered and has no error-rate guarantee, so it should not replace the bootstrap CI LB > 0 and ≥ +0.05 already in stage 2. If adopted, report n01/n10 counts as well (§5.4).
3. **Interaction stage justification, no ADR (citation only).** The reported R4 regression from mints gated only against the no-skill agent is a concrete, citable example of why candidate traits should be tested against the *current genome* (CIG attribution under interactions). Cite it as reported, single run.
4. **Reward/metric judge sharing, requires an ADR if EXP-0001 uses an LLM judge.** The paper deliberately uses the same judge family for the evolution reward and the reported metric. EXP-0001 should keep the CIG judge and the held-out evaluation judge distinct, or report both.
