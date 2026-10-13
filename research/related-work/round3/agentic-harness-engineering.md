# Agentic Harness Engineering (AHE): detailed notes (Round 3)

**Paper:** *Agentic Harness Engineering: Observability-Driven Automatic Evolution of Coding-Agent Harnesses*, by Jiahang Lin\*, Shichun Liu\*, Chengjun Pan\*, Lizhi Lin, Shihan Dou, Zhiheng Xi, Xuanjing Huang, Hang Yan, Zhenhua Han†, Tao Gui† and Yu-Gang Jiang.
- **Affiliations:** Fudan University, Peking University, and Shanghai Qiji Zhifeng Co., Ltd. (\* equal contribution; † corresponding author.)
- **arXiv:** 2604.25850 (cs.CL). v1 was submitted on 28 Apr 2026. The version read here is v4, last revised 18 May 2026. It is a preprint, and I found no venue on the abs page.
- **Code:** the paper states github.com/china-qijizhifeng/agentic-harness-engineering. I did not open or check the repository.
- **Review status:** reviewed 2026-10-13 by Claude (third agent). It is pending Research/Critic cross-review.

**Read depth:** full text from arxiv.org/html/2604.25850 (v4). That covers §§1–5, Limitations and Appendices A–E, including the full Evolve Agent prompt (B.2) and the case study (C). I downloaded the raw HTML, stripped the tags locally and read the whole text. The quotes below are verbatim from that text. Figures 1, 3, 4, 12 and 13 are images. I used only the numbers the text states and could not read the per-iteration curve values. I did not check the PDF.

## Mechanism

- **Frozen base model:** "with the base model held fixed and only the explicit harness edited" (§3). All three role agents (Code Agent, Agent Debugger and Evolve Agent) use GPT-5.4. The Code Agent runs at high reasoning and the Evolve Agent at xhigh (App. A, Table 4).
- **Component substrate (NexAU, §3.1):** the harness exposes "seven orthogonal component types as explicit files at fixed mount points in a single workspace: system prompt, tool description, tool implementation, middleware, skill, sub-agent configuration, and long-term memory."
  - "Each logical edit becomes one commit on the workspace's git history, which yields file-level diffs and rollback granularity for free."
  - New components must be registered in `code_agent.yaml` (B.2).
  - The seed H0 has bash only, with "no middleware, no skills, no sub-agents".
- **Experience distillation (§3.2):** an Agent Debugger turns k rollouts per task into a per-task root-cause report plus a benchmark-level overview. Raw traces stay available for drill-down.
- **Decision observability (§3.3):** each edit gets a change-manifest entry. The entry carries failure evidence, root cause, targeted fix, `predicted_fixes`, `risk_tasks`, `constraint_level` and `why_this_component` (JSON schema in B.2).
  - "In the next round, the loop intersects the predicted-fix and predicted-regression sets with the observed task-level deltas to produce a per-edit verdict."
  - The Evolve Agent then sorts each edit into KEEP, IMPROVE, or ROLLBACK + PIVOT.
  - App. C.3 says edits are "subject to the same automatic rollback if their predicted fixes do not appear."
  - **The paper gives no exact verdict rule.** It states no threshold, no statistic, and no way to handle several edits shipped in the same round.
- **Outer loop (Algorithm 1):** Rollout(H_{t−1}) → Clean → Attribute(C_{t−1}, T_{t−1}, T_t) → Rollback → AgentDebugger → Evolve → Commit (git tag) → "if Pass@1(T_t) > Pass@1(H_best) then H_best ← H_t".
  - The returned H_best is selected **on the same 89 Terminal-Bench 2 tasks that drive evolution**. The paper has no evolve/validation/test split.
  - It ran T = 10 iterations with k = 2 rollouts per task, in "a single AHE campaign", taking about 32 hours (§4.2, Table 4).
- **Governance (§3.3, B.2):**
  - The Evolve Agent may write only inside `workspace/`.
  - The verifier, tracer, LLM configuration and runs directory are read-only.
  - The seed system-prompt rules cannot be deleted.
  - The prompt forbids "task-specific logic or hardcoded solutions" and "reverse-engineer[ing] test cases from trajectories".
  - The change manifest is written to the experiment root, "NOT inside workspace/". In other words, the evidence ledger sits outside the evolvable artifact.
- **Memory split (B.2):** `LongTermMEMORY.md` is "Persistent cross-session knowledge -- MODIFIABLE" by the Evolve Agent. `ShortTermMEMORY.md` is "managed by code agent at runtime, DO NOT MODIFY". The paper reports no mechanism that promotes short-term content into long-term memory.
- **Multi-variant mode (B.2):** this appears in the prompt only. The workspace layout includes `variant_N/`, `variant_selection.json` and a "MANDATORY Strategy Constraint" for parallel agents. The main text never describes using variants in the reported run, so I do not count it.

## Results (as stated in text)

- **Terminal-Bench 2 (89 tasks; 4 easy / 55 medium / 30 hard; k = 2; Table 1):**

  | Harness | pass@1 |
  |---|---|
  | NexAU0 (seed) | 69.7% |
  | **AHE** | **77.0%** |
  | Codex | 71.9% |
  | Terminus-2 | 62.9% |
  | OpenCode | 47.2% |
  | ACE | 68.9% |
  | TF-GRPO | 72.3% |

  - On the Hard tier AHE scores 53.3%, below Codex at 56.7%.
  - **The paper reports no confidence intervals or significance tests anywhere.** It acknowledges "a promising but high-variance setting" (Limitations).
- **SWE-bench-verified (500 tasks; harness transferred frozen, Table 2):**

  | Harness | Success | Tokens per trial (k) |
  |---|---|---|
  | AHE | 75.6% | 461 |
  | NexAU0 | 75.2% | 526 |
  | ACE | 74.6% | 679 |
  | TF-GRPO | 74.2% | 582 |

  - The success gain over the seed is +0.4 pp. Without CIs it is not distinguishable from noise.
  - AHE regresses on scikit-learn, pydata and astropy, by −6.3, −4.6 and −4.5 pp respectively.
  - The token saving (−12% vs the seed) is the more robust result.
- **Cross-model transfer (Terminal-Bench 2, harness frozen, Fig. 3 text):**
  - deepseek-v4-flash: +10.1 pp
  - qwen-3.6-plus: +6.3 pp
  - gemini-3.1-flash-lite: +5.1 pp
  - GPT-5.4 medium and xhigh: +2.3 pp each
  - The authors say the transfer numbers "conflate harness portability with operating-point coupling" (Limitations).
- **Component ablation (Table 3; one AHE component swapped into the seed, the others held at seed defaults):**

  | Variant | pass@1 |
  |---|---|
  | + memory only | 75.3% |
  | + tool only | 73.0% |
  | + middleware only | 71.9% |
  | + system_prompt only | 67.4% (a regression) |
  | AHE full | 77.0% |

  - The single-component gains sum to +11.1 pp against +7.3 pp for full AHE. The authors call this "Components interact non-additively, capping the aggregate gain."
  - On Hard, memory-only (63.3%) beats full AHE (53.3%). The authors "leave interaction-aware evolution to future work."
  - The ablation is a **post-hoc analysis**, not part of the loop.
- **Self-attribution reliability (§4.4.2, App. D):**
  - Fix predictions: precision 33.7% and recall 51.4%, against random baselines of 6.5% and 10.6%.
  - Regression predictions: precision 11.8% and recall 11.1%, against random baselines of 5.6% and 5.4%.
  - Cumulatively, "43 unique regression predictions and only 5 landed … 40 regressions the agent did not foresee actually occurred."
  - The section headline is "**Regression blindness.**"
  - App. D gives a cumulative regression precision of 11.6%; §4.4.2 gives a cross-iteration mean of 11.8%. These are two different aggregations, not an error.
- **Case study (App. C):** four "winning rounds" (iterations 2, 5, 6 and 8) and eight manifest entries. The trajectory pairs share "the same random seed". This is a qualitative before/after comparison, not a gate.

## Overlap vs HeritAgent chain (full-text judgement)

| Element | Coverage | Evidence |
|---|---|---|
| 1. Lifetime acquisition | **Partial** | Harness edits, skills and 12 long-term-memory "boundary-case lessons" (§4.4.1) are distilled from the Code Agent's own rollout trajectories (§3.2). However, a separate meta-agent acquires them offline between benchmark rounds. There is no agent "lifetime", and the Code Agent does not acquire them while acting. |
| 2. Somatic/germline separation | **Absent** | There is one live workspace. Edits are committed to H_t and expressed in the next round's rollouts *before* any verdict (Algorithm 1). ShortTermMEMORY (runtime, session-scoped) vs LongTermMEMORY (evolver-edited) is a weak analogue: nothing is staged and then promoted. Variant workspaces are prompt-only. |
| 3. Typed executable genome | **Partial (strong)** | Seven typed component classes at fixed mount points, a YAML registry, git commits and tags per edit and iteration, and a `constraint_level` per manifest entry. Middleware (execution hooks, guards, finish-hooks) is a functional analogue of "regulation". Missing: functional traits as the typed unit (types are *file classes*, and one fix spans several files and levels), per-trait validation records attached to the heritable artifact, and a policies/skills/regulation schema. |
| 4. Causal validation gate | **Partial (weak)** | The per-edit verdict is an *observational* before/after comparison between consecutive rounds on the same 89 tasks, with k = 2. All edits in a round ship together, so attribution is confounded, and the rule is unspecified. It is retrospective: edits are expressed first and rolled back later. There is no controlled ablation at gate time, no held-out set, and no explicit regression stage. The authors show that the predictive regression check fails ("regression blindness"). The component ablation is controlled but post-hoc and per component class. |
| 5. Selective assimilation | **Partial** | Per-edit KEEP / IMPROVE / ROLLBACK at file granularity plus best-so-far selection of H_best is selective *retention*. There is no separate heritable store into which validated traits are assimilated as a distinct lifecycle event. |
| 6. Multi-generational, frozen weights | **Partial** | Ten git-tagged iterations of one harness lineage on frozen GPT-5.4. The frozen harness is reused unchanged on five other base models. The iterations are versions of one artifact, not parent → descendant agents with distinct lifetimes. A full-text search finds no "lineage" or "germline" anywhere. "Inherit" occurs only in App. B prompts, in the software sense (Python class and sub-agent config inheritance). "Generations" occurs once, in the B.2 prompt, meaning consecutive loop iterations ("mixes **two** generations of work"). |

**Full chain implemented: no.** The paper has no somatic store, no prerequisite gate, and no agent generations.

## Does the full text change the abstract-level assessment?

My abstract-level reading (ledger §2.10.12) was **MEDIUM**, with coverage 1 – · 2 – · 3 P · 4 P · 5 – · 6 P. The full-text reading is **1 P · 2 – · 3 P(strong) · 4 P(weak) · 5 P · 6 P**, and I put the threat at **LOW for the chain** (down from MEDIUM).

**What moves up:**
- Elements 1 and 5 move from absent to partial. Trajectory-derived lessons and skills, and per-edit keep/rollback, are both present.
- Element 3 is a little stronger than the abstract suggested. The typed component classes and registry, the manifest held outside the workspace, and the protected seed rules all exist.
- Middleware is a real execution-time "regulation" layer. The paper's case study (the iteration 5/6/8 publish-state guard) shows prose rules failing until middleware or tools enforce them. The paper states this for the guard case only ("prompt advice … was already present, but no execution-time mechanism enforced it", App. C.1.2). That is direct support for typing regulation separately in the EAG.

**What moves down:**
- The "falsifiable contract" is weaker as a gate than the abstract implies:
  - The verdict is retrospective and observational.
  - Co-shipped edits confound attribution.
  - The rule has no stated threshold.
  - k = 2 and a single campaign give no CIs.
  - H_best is selected in-sample on the evolution benchmark.
- The authors' own analysis shows that self-declared predictions fail on regressions.
- The headline transfer to SWE-bench is +0.4 pp in success. The token reduction is the real effect.

**Net:** AHE is less a competitor to the CIG than **independent evidence for the CIG's design premises**:
- Retrospective self-attribution is regression-blind, which argues for measured regression protection rather than predicted.
- Components interact non-additively, which argues for CIG stage 3 (attribution under interactions).
- In-distribution aggregate gains can transfer weakly, which argues for the held-out stage.

The typed-harness overlap with element 3 is unchanged and remains the reason to cite it. It is already covered by §3 item 22.

## Consequences for HeritAgent (no design changed; items marked *requires ADR* are for the Builder/ADR owner)

1. **Cite as motivating evidence (no ADR).**
   - "Regression blindness" (11.8% / 11.1% vs a random baseline of about 5.5%) is published evidence that an evolver's predicted at-risk set cannot replace a measured regression stage. It supports CIG stage 5 as measured, not predicted.
   - Non-additivity (+11.1 pp summed singles vs +7.3 pp full) supports stage 3.
   - Neither result is ours to claim (SPEC §34). Both are measured on a single campaign with no CIs.
2. **EAG typing rationale (no ADR now; later EAG-schema ADR).**
   - Cite the "system prompt alone regresses" result (67.4% vs 69.7%), together with the case study showing prose rules failing until a tool or middleware guard enforces them. Together they argue that regulation and executable traits should be separate types, not prose policies.
   - Caveat: the "structural transfers, prose does not" generalisation is the authors' reading of a single ablation.
   - Note also that AHE's types are *file classes*. A single functional fix spans several levels, for example a prompt rule plus a shell guard (iteration 5). This argues for the EAG's typed unit being the *trait*, with typed sub-parts, rather than the file.
3. **Arm B′ (performance-gated baseline): *requires ADR* (amendment to ADR-0002).** AHE's loop gives a cheap, published "retrospective keep/rollback on next-round task deltas" operationalisation. The rule is: keep an edit iff its predicted fixes appear, and keep the best-so-far by aggregate pass@1. Two uses:
   - It could serve as a weaker variant of B′, the "self-attributed performance gate", sitting between B (unrestricted) and the GRASP/SAGE-style held-out net-gain B′ already listed in ledger §5.1.
   - Alternatively, cite it to explain why B′ must use a *held-out* split. AHE selects H_best on the same tasks it evolves on, so its 77.0% is an in-sample best-of-10 figure, which is the optimizer's-curse problem SAGE names.

   Either way, choose and pre-register one before the run.
4. **CIG statistics: *requires ADR* (threshold changes only; reporting formats do not).**
   - Under AHE's protocol (89 tasks, k = 2) one task flip moves pass@1 by about 1.1 pp. Several per-round differences in AHE are of that order.
   - EXP-0001 should report per-stage n01/n10 flip counts, with paired per-task tests and CIs, so that "round-to-round delta" verdicts are not used as evidence.
   - It should also forbid co-shipping several candidate traits into one evaluation unless the design is factorial or uses leave-one-out ablation. AHE's co-shipped edits make per-edit verdicts unidentifiable.
5. **Gate and ledger outside the heritable artifact (no ADR; already ledger §5.6).** AHE keeps the verifier read-only and writes the manifest outside `workspace/`. This is prior art for "gate logic outside the genome". Cite it, do not claim it.
6. **H4 framing (later ADR, per ledger §5.10).** AHE's cross-model gains are confounded with the operating point (step budget and timeout were fitted to GPT-5.4 high), as the authors say. For EXP H4, the operating point (step budget, timeout, reasoning tier) must be a declared, controlled precondition of an inherited genome.

## Suggested ledger update (for §2.10.12; pending cross-review)

Change the entry to: **deep-read (full text, v4), threat LOW (down from MEDIUM).** Chain coverage 1 P · 2 – · 3 P(strong) · 4 P(weak) · 5 P · 6 P.

- The gate is retrospective, observational, co-shipped, unthresholded, k = 2, single campaign, and selects in-sample.
- The authors document regression blindness and non-additive component interaction.
- Main value: (a) the typed-component rationale for the EAG, including middleware as regulation; (b) supporting evidence for CIG stages 3 and 5; (c) a template for a weak "self-attributed performance gate" variant of B′ (requires ADR).
- Keep §3 item 22 as is. Add to item 15: "retrospective self-attributed edit verdicts are regression-blind (AHE)".
