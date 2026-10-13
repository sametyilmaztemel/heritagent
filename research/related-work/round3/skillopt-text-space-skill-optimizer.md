# SkillOpt — Executive Strategy for Self-Evolving Agent Skills — detailed notes (Round 3)

**Paper:** *SkillOpt: Executive Strategy for Self-Evolving Agent Skills*. Authors: Yifan Yang, Ziyang Gong, Weiquan Huang, Qihao Yang, Ziwei Zhou, Zisu Huang, Yan Li, Xuemei Gao, Qi Dai, Bei Liu, Kai Qiu, Yuqing Yang, Dongdong Chen, Xue Yang, Chong Luo. Affiliations: Microsoft, Shanghai Jiao Tong University, Tongji University and Fudan University. That is 15 names on both the abs page and the HTML. The abs page's "view PDF" string, however, says "Yifan Yang and 13 other authors". Cite the 15 listed names.
- **arXiv:** 2605.23904, cs.AI with cross-list cs.CL. v1 is dated 22 May 2026 and v2 25 May 2026.
- **Format:** 27 pages, 4 figures, 6 tables. No venue is listed.
- **Code:** the paper links aka.ms/SkillOpt. I did not open it.
- **Review:** 2026-10-13 by Claude (third agent), pending Research/Critic cross-review.

**Read depth:** full text from arxiv.org/html/2605.23904v2.
- I downloaded the raw HTML, stripped the tags and read the text.
- Covered: §§1–5, Tables 1–6, Algorithm 1, and Appendices A–C, including the prompt contracts (skimmed), the C.3 safeguards and the C.4 design principles.
- I checked the quotes below against the stripped text by searching for them.
- Figure 3 (val/test curves) and Figure 4 could only be read through their captions and prose. I did not check the PDF line by line.

## Mechanism

- **Frozen target, separate optimizer.**
  - A frozen target model M runs tasks with a single skill document s, a Markdown `best_skill.md` of about 300–2,000 tokens.
  - The skill is prepended to the system prompt (direct chat) or rendered as a per-task `SKILL.md` (Codex CLI and Claude Code CLI harnesses).
  - A separate optimizer model O edits s. O is GPT-5.5 by default; Table 5 also uses a "target-matched" optimizer.
  - Model weights never change.
- **Splits (§3.1, App. C):** D_tr supplies experience, D_sel gates updates, and D_test is reported only.
  - The default split is 2:1:7 with `split_seed=42`.
  - ALFWorld uses 39 train tasks, 140 selection environments and 134 test environments.
  - Selection-split sizes for the other benchmarks are not given in the text.
- **Forward pass:** a rollout batch from D_tr, size B = 40 by default, under the current skill. It records messages, tool calls, verifier feedback and execution traces.
- **Backward pass:**
  1. Failures and successes are split into reflection minibatches (B_m = 8).
  2. O proposes add/delete/replace edits from each minibatch: corrective edits from failures, "preserve" edits from successes.
  3. Edits are merged hierarchically, giving priority to failures.
  4. O then ranks the merged pool by "expected utility". The ranking prompt (C.2.6) scores systematic impact, complementarity, generality and actionability. These are the LLM's own judgements, not measured effects.
- **Bounded update ("textual learning rate"):**
  - The ranked pool is clipped to the top L_t edits. L_t = 4 by default, with cosine decay to a floor of 2.
  - Patch ops are `append`, `insert_after`, `replace` and `delete`.
  - Each merged edit records a support count and a source type.
- **Validation gate (§3.5, Alg. 1, lines 16–28):** the candidate skill (the incumbent plus up to L_t edits *as one bundle*) is scored on D_sel with the same frozen model and harness. Quotes:
  - "If it improves over the current selection score, it becomes the new current skill; if it also exceeds the best score so far, it becomes best_skill."
  - "The validation gate is intentionally strict: a candidate skill is accepted only when its selection-split score is strictly greater than the current selection score, so ties are rejected and the deployed skill never silently drifts."
  - The test is a mean-score comparison on a single evaluation. A hash cache means an identical candidate is never re-scored.
  - There is no statistical test, no per-edit evaluation and no separate regression budget. Regressions are netted in the mean.
- **Rejected-edit buffer:** an epoch-local buffer of rejected edits, the score drop each caused, and the failure patterns observed. It is fed back to later reflection calls in the same epoch as "negative feedback".
- **Epoch-wise slow/meta update (§3.6, from epoch 2):**
  - The same sampled training items (20 by default) are run under the previous-epoch and current-epoch skills, and grouped into "improvements, regressions, persistent failures, and stable successes".
  - O writes longitudinal guidance into a protected region (`SLOW_UPDATE_START` … `SLOW_UPDATE_END`) that step-level edits cannot overwrite. The result still passes the D_sel gate.
  - A separate *meta skill* holds optimizer-side memory of which edit patterns helped. It is "optimizer-side only … not shipped with the target model".
- **Provenance:** each step writes an `edit_apply_report.json` with per-edit accept/skip status, "so the source of every change to best_skill.md is recoverable".
- **Vocabulary:** the paper frames all of this as a deep-learning analogy (batch, minibatch, learning rate, schedule, validation, momentum). It uses none of the following words: "causal", "inherit", "lineage", "population", "variance", "confidence" or "significance". "Generation" appears only as "edit generation". It claims to be "to our knowledge, the first systematic controllable text-space optimizer for agent skills".

## Results (as reported; single run per cell, no seeds, CIs or tests)

- **Table 1 (52 cells):** 7 target models × 6 benchmarks in direct chat, plus GPT-5.5 × 5 benchmarks in each of Codex and Claude Code. SkillOpt is "best or tied-best" in all 52.
  - "Tied" is undefined. There is at least one exact tie: GPT-5.4-mini LiveMath, where Trace2Skill and SkillOpt both score 32.8.
  - GPT-5.5 direct chat averages 58.8 with no skill and 82.3 with SkillOpt (+23.5).
  - The best per-cell baseline averages 76.9 (an "oracle" over 6 methods), so SkillOpt is +5.4 above it.
  - Codex: +24.8 over no skill and +14.0 over EvoSkill. Claude Code: +19.1 and +3.2.
  - In Codex SearchQA, EvoSkill falls 20.4 points below no-skill (61.4 vs 81.8).
- **Baseline coverage is uneven:**
  - TextGrad and GEPA run only in direct chat; EvoSkill runs only in the harnesses.
  - GEPA (Pareto/validation-based) and EvoSkill (held-out frontier) are themselves *validation-gated* optimizers.
  - The paper notes that Trace2Skill evaluates "without SkillOpt's iterative validation gate".
- **Ablations (Tables 2–3; GPT-5.5 as both target and optimizer; SearchQA / SpreadsheetBench / LiveMath):**

  | Setting | SearchQA | SpreadsheetBench | LiveMath |
  |---|---|---|---|
  | Default | 87.1 | 77.5 | 61.3 |
  | Without edit budget ("without lr") | 84.6 | 75.7 | 57.3 |
  | Dynamic lr | 85.8 | 71.8 | 54.0 |
  | Without rejected buffer | 85.5 | 72.9 | 58.9 |
  | Without meta skill | 85.1 | 75.7 | 58.1 |
  | Without meta skill and slow update | 86.3 | **55.0** | 59.7 |

  - The hyperparameter sweeps are mostly flat on SearchQA (±1.5 points) and noisy on LiveMath. For example, the reflection minibatch size moves LiveMath between 54.8 and 64.5, and the edit budget L_t ∈ {1, 2, 4, 8, 16} moves it between 56.5 and 66.9. The pattern is non-monotone, which suggests large run-to-run noise.
  - **There is no "without validation gate" ablation.** The Introduction says the ablations show "held-out gating prevents harmful proposals from accumulating". The "Gate strictness and edit observability" paragraph, however, is descriptive only and has no numbers, and there is no strict-vs-non-strict comparison.
  - The gate's value is argued indirectly from three things: only 1–4 edits were accepted, Trace2Skill (ungated) does worse, and in Fig. 3 "validation checkpoints track held-out test performance".
- **Transfer (Table 4), all positive:**

  | Transfer | Benchmark / target | No skill → transferred skill | Gain |
  |---|---|---|---|
  | Cross-model (from GPT-5.4) | SpreadsheetBench, GPT-5.4-mini | 36.1 → 45.5 | +9.4 |
  | Cross-model (from GPT-5.4) | SpreadsheetBench, GPT-5.4-nano | — | +3.0 |
  | Cross-model (from GPT-5.4) | LiveMath, GPT-5.4-mini | — | +4.5 |
  | Cross-model (from GPT-5.4) | LiveMath, GPT-5.4-nano | 23.2 → 28.8 | +5.6 (above the in-domain 27.2) |
  | Cross-harness, Codex → Claude Code | SpreadsheetBench | 22.1 → 81.8 | +59.7 |
  | Cross-harness, Claude Code → Codex | SpreadsheetBench | 27.5 → 71.1 | +43.6 |
  | Cross-harness, Codex → Claude Code | LiveMath | — | +1.6 |
  | Cross-harness, Claude Code → Codex | LiveMath | — | +12.8 |
  | Cross-benchmark, OlympiadBench → Omni-MATH | GPT-5.4 / -mini / -nano | — | +3.7 / +1.8 / +1.3 |

  This is one-hop reuse of a frozen artifact, with no further learning in the new setting.
- **Optimizer strength (Table 5):** a target-matched optimizer recovers 56–74% of the strong-optimizer gain.
- **Edit economy (Table 6):**
  - 1–4 accepted edits per benchmark (median 2.5). LiveMath (+29.3) and OfficeQA (+39.0) each came from *one* accepted edit.
  - Final skills are 379–1,995 tokens.
  - Training cost is 20.8M–213.8M tokens per benchmark, or 0.6M–46.4M tokens per test point gained.
- **Stated limitations (App. B):**
  - The method needs reliable automatic scoring and a held-out selection split.
  - It costs extra rollouts and optimizer calls.
  - It builds one skill for one domain, not a library.
  - Skills can encode training-distribution heuristics, so "careful held-out evaluation remains necessary before transferring".

### Reporting issues found on full read (fair to flag; none invalidates the method)

1. **Headline cells coincide with the best sweep cells, not the stated default.**
   - The default configuration is cosine schedule, L_t = 4 (§4, App. C, and the light-blue Table 3 rows), and it scores 87.1 / 77.5 / 61.3 on SearchQA / SpreadsheetBench / LiveMath.
   - The Table 1 GPT-5.5 direct-chat cells are 87.3 / 80.7 / 66.9. These are exactly the "constant" schedule row (87.3 / 80.7, Table 2e) and the L_t = 8 row (LiveMath 66.9, Table 2d).
   - The paper does not say how the Table 1 configurations were chosen. If they were picked per benchmark from the sweep using test scores, that would be configuration selection on the test split. I cannot verify this from the text, so treat it as an open question for the authors, not a finding of misconduct.
2. **Split inconsistency.** The Table 2 caption fixes panel (a) to a 4:1:5 split. Appendix C ("Ablation protocol") says the train-size ablation uses 2:1:7.
3. **Two-tier state appears vacuous as written.**
   - Algorithm 1 initialises s_cur = s_best and score_cur = score_best.
   - It updates s_cur only when score_cand > score_cur, and then also updates s_best when score_cand > score_best.
   - As written, cur and best therefore always coincide. Unlike RSEA, there is no lateral "accept if not worse" tier.
4. **No variance reporting anywhere.** Every cell is one optimization run and one test evaluation. The ablation sweeps move LiveMath by up to ~12 points non-monotonically, which suggests run-to-run noise of that order on that benchmark.
5. **Gate claim not ablated** (see the Ablations bullet above).

## Overlap vs HeritAgent chain (full-text judgement)

| Element | Coverage | Evidence |
|---|---|---|
| 1. Lifetime acquisition from trajectories | **Full** | §3.2–3.3: edits are derived from the frozen target's own scored rollouts (messages, tool calls, verifier feedback, execution traces). Failure minibatches yield corrective rules and success minibatches yield preserving rules. The "lifetime" is an offline training loop with an external frontier optimizer, not deployment. That still counts as full for this element, as it did for RSEA and EvoSkill. |
| 2. Somatic / germline separation | **Absent** (weak analogues only) | There is no store of acquired-but-not-yet-heritable traits. A candidate exists only for one D_sel evaluation and is then committed or discarded. The cur/best two-tier state collapses (issue 3). Analogues worth citing: (a) the optimizer-side meta skill and rejected-edit buffer are training state that is never shipped, an exported vs private boundary; (b) the protected slow-update region separates fast and slow content within the one artifact. Neither is a staging store of lifetime traits with a promotion event. |
| 3. Typed executable genome | **Partial (weak)** | One untyped Markdown document. Its only structure is the protected `SLOW_UPDATE` region versus free text. Patch ops are typed (append / insert_after / replace / delete) and there are per-edit accept/skip logs (`edit_apply_report.json`) and content hashes. There is no schema of policies, skills and regulation, no per-trait validation record inside the artifact, and nothing executable. |
| 4. Causal validation gate | **Partial** | The gate is an interventional candidate-vs-incumbent comparison on a disjoint selection split with the same frozen model and harness: strict >, ties rejected, applied to every update including slow updates (C.4: "every candidate skill is evaluated on a selection split before acceptance"). It is applied to a **bundle of up to L_t edits**, so it cannot attribute gain to an individual edit. There is no per-edit ablation, no attribution, no replay across environments, no out-of-scope probe, no separate regression bound, no statistical test, and one cached evaluation. The paper never uses the word "causal". This is a performance gate, not a trait-level causal gate. |
| 5. Selective assimilation | **Partial** | It is highly selective in effect: 1–4 accepted edits survive out of many proposed, and "the deployed skill is correspondingly compact rather than the union of every reflection". Selection, however, is (i) LLM-judged ranking by "expected utility" and clipping to L_t, then (ii) an aggregate gate on the bundle. No edit is admitted on its own measured evidence, and there is no assimilation into a heritable store distinct from the working artifact. |
| 6. Multi-generational, frozen weights | **Partial** | Frozen target weights: yes, explicitly. There are no generations, lineage, descendants or population; "generation" occurs only as "edit generation". The closest analogue is the one-hop reuse of a frozen skill in another model, harness or benchmark (Table 4). The recipient does no further learning and no adaptation cost is measured, which corresponds to H4's transfer question, not to inheritance. |

**Implements full chain: no.** One element is full, four are partial (one of them weak) and one is absent. Elements 2 and 6 (the somatic/germline split and inheritance across agent generations) are missing entirely.

## Does the full text change the abstract-level assessment?

The ledger (§2.11) rated SkillOpt at abstract level as a **low-threat** adjacent work: a performance gate, not a causal trait gate, with no inheritance, and a candidate baseline for a held-out-only arm. **Threat to the novelty claim stays LOW.** The full text confirms all three points and adds detail that weakens SkillOpt as a gate competitor:
- the gate acts on an edit *bundle*;
- there is no statistical test and a single cached evaluation;
- there is no gate ablation;
- there is no variance reporting;
- there is a possible configuration-selection issue in the headline table.

The full text **raises its relevance as an empirical baseline** (impact on H2's effect size: medium, not novelty).
- **What it shows:** a single, whole-artifact, tie-rejecting held-out gate wrapped around bounded LLM edits gives very large gains on frozen frontier and small models: +9 to +27 points averaged per model, from only 1–4 accepted edits. It beats two other validation-gated optimizers (GEPA, EvoSkill).
- **Why it matters for H2:** this is the strongest published instance of the "any held-out gate captures most of the benefit" counter-hypothesis. If HeritAgent's arm B′ is a weak gate, a reviewer can point to SkillOpt as the stronger performance-gated alternative.
- **Engineering ideas it supports:** bounded edits (no-budget ablation 84.6 / 75.7 / 57.3 vs 87.1 / 77.5 / 61.3), negative-feedback memory, and protected consolidation regions.

**Fair the other way:**
- **Gate design:** SkillOpt's gate is stricter than several in the ledger cluster. It rejects ties, unlike RSEA's lateral tier, ACE's non-strict keep-best and the CIG v0 stage-3 "Δ ≥ 0". It also gates the slow/consolidation updates, not only fast edits.
- **Provenance:** its per-edit accept/skip logging gives recoverable provenance for every change.
- **Separation of concerns:** keeping optimizer memory out of the deployed artifact is a clean form of the "gate/optimizer state outside the heritable artifact" principle.
- **Transfer evidence:** its transfer results are uniformly positive, though single-run and few in number. That is more favourable for H4-style portability than SkillRL's negative no-SFT result.

## Consequences for HeritAgent (no design changed; *requires ADR* items are for the Builder/ADR owner)

1. **Arm B′ operationalization (e), SkillOpt-style. *Requires ADR (ADR-0002 amendment).*** Per generation, propose candidate traits from somatic evidence and cap them at L_t per step (for example 4). Accept the bundle into the inherited genome iff the mean held-out selection score strictly exceeds the incumbent's. Keep a rejected-trait buffer as optimizer-only feedback, and use no per-trait ablation, attribution or out-of-scope stage.
   - It is cheap.
   - It has public code: aka.ms/SkillOpt is linked, but I did not inspect it or check its licence.
   - It is empirically the strongest published whole-artifact performance gate.
   - It complements the B′ variants (a)–(d) in ledger §5 item 1 (GRASP, VaG, SAPO, RSEA).
   - Whichever B′ is chosen, match its held-out evaluation budget (episodes) to arm C's. SkillOpt's training cost is 20–214M tokens per benchmark, so an unmatched comparison is easy to get wrong.
2. **Bundle "hitchhiker" analysis. *Pre-registered secondary analysis; requires ADR only if it becomes an arm.*** SkillOpt accepts edits in bundles of ≤ L_t. CIG's per-trait claim can be tested directly: run CIG stage-2 leave-one-out (LOO) / controlled ablation on the traits inside each B′-accepted bundle, and report the fraction of individually null or harmful traits that a bundle gate admits. This measures exactly what a trait-level gate adds over SkillOpt-style gating. Note the RSEA lesson: for redundant traits, LOO ablation needs a group- or empty-genome reference.
3. **Gate statistics. *Requires ADR for any threshold change; reporting format needs none.***
   - SkillOpt compares one cached evaluation per candidate with a strict >, and takes the maximum over many candidates (4 epochs × several steps) on a fixed D_sel. That is an optimizer's-curse setting with no correction.
   - For CIG: (a) re-evaluate or seed-replicate rather than cache when a decision is close to the threshold; (b) keep the stage-2 bootstrap CI lower bound > 0; (c) resolve the stage-3 "Δ ≥ 0" tie question explicitly, either SAGE-style paired test with abstain, or a declared non-inferiority margin. SkillOpt's tie rejection is one published precedent, RSEA and ACE the opposite.
   - Report the number of candidates evaluated per accepted trait, so that selection-split overfitting can be judged. SkillOpt does not report this.
4. **Configuration-selection discipline. *Pre-registration; no ADR for reporting.*** Because the SkillOpt headline cells coincide with best-of-sweep cells (issue 1), EXP-0001 should:
   - pre-register CIG thresholds and B′ hyperparameters;
   - select any hyperparameter on the selection split only;
   - always report the pre-registered default configuration;
   - report multi-seed variance for identical genomes as a noise floor (consistent with the RSEA note, item 3).
5. **Regression accounting (differentiator to keep).** SkillOpt nets regressions in the mean and uses improvement/regression groupings only as optimizer *input* (the slow update). CIG's separate regression bound τ_r and n01/n10 reporting are a real methodological difference, and should be stated as such with SkillOpt cited. No ADR needed.
6. **Optimizer state outside the artifact (supports ledger §5 item 6).**
   - SkillOpt's meta skill and rejected buffer are never shipped. This is a published precedent for keeping gate and optimizer state out of the heritable artifact.
   - Its protected `SLOW_UPDATE` region, which fast edits cannot overwrite, is a precedent for write-protected genome regions such as regulation fields.
   - Cite both. Any EAG write-protection rule belongs in a later EAG/CIG ADR.
7. **Rejected-trait memory as lineage data. *Later ADR-0001 amendment when the lineage graph is built.*** SkillOpt's ablation suggests rejected-edit memory helps the proposer (−1.6 / −4.6 / −2.4 without it; single runs). HeritAgent could record CIG rejections, with the score drop and the stage that failed, as lineage-graph evidence available to descendants' trait miners. This is a design option, not a claim.
8. **H4 framing.** Cite SkillOpt's cross-harness transfer (SpreadsheetBench +59.7 / +43.6; LiveMath +1.6 / +12.8) as evidence that transfer is positive but asymmetric and benchmark-dependent. This adds to ledger §3 item 24.

## Suggested ledger updates (pending cross-review)

- **§2.11 SkillOpt entry:** mark it deep-read (full text); threat stays LOW. Chain coverage: 1 F · 2 – · 3 P(weak) · 4 P · 5 P · 6 P (frozen weights + one-hop transfer only). Main value: the strongest whole-artifact B′ template, plus the reporting caveats above. Remove it from the §2.12 "not read" list.
- **§3 additions (attach to items 11, 14, 23 and 26):**
  - bounded per-step edit budgets as a "textual learning rate" for skill documents;
  - tie-rejecting strict held-out acceptance of edit bundles;
  - rejected-edit buffers as negative feedback to the proposer;
  - epoch-wise consolidation into a write-protected region that is still gated;
  - optimizer-side memory kept out of the deployed artifact;
  - per-edit accept/skip provenance logs;
  - "skill as the trainable external state of a frozen agent" with deep-learning-optimizer framing.
- **Naming hazard:** avoid "textual learning rate", "slow/meta update" and "meta skill" for CIG or EAG components.
