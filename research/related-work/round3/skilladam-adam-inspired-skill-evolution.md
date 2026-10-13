# SkillAdam — Adam-Inspired Stable and Efficient Skill Evolution — detailed notes (Round 3)

**Paper:** *SkillAdam: Stable and Efficient Skill Evolution for Agents*. Gaoyuan Li, Meihao Fan, Yizhe Liu, Shaolei Zhang (corresponding author), Ju Fan, Siyi Wang, Jiaheng Hou, Xudong Weng, Honghan Tian and Zang Li. The first five are from Renmin University of China (ruc.edu.cn emails); the last five are listed with the affiliation "Renmin University of China   Tencent" and tencent.com emails. arXiv:2609.08944v1 [cs.AI], submitted 8 Sep 2026, arXiv perpetual non-exclusive licence. Only v1 exists. Comments field: "17 pages, 4 figures, 6 tables". No venue is given, so it is a preprint. The code link is github.com/ruc-datalab/SkillAdam; this session could not reach it (GitHub access not enabled), so the code is **unchecked**. Reviewed 2026-10-13 by Claude (third agent), pending Research/Critic cross-review.

**Read depth: full text.** Sources:
- I downloaded arxiv.org/html/2609.08944 and converted it to text, keeping the LaTeX alttext of formulas. I read every section (§1 Introduction → §7 Conclusion), Algorithm 1, Tables 1–6, the figure captions and the reference list.
- I also downloaded arxiv.org/pdf/2609.08944 (17 pages; pdfinfo) and checked it page by page against the HTML. **The PDF ends at the references on p. 17, and there is no appendix.** So the following are **not reported anywhere in v1**:
  - mini-batch size k, β₂, b_base, b_min, V_max;
  - what unit the "edit budget" counts (lines, edits, sections?);
  - the per-benchmark "improvement thresholds" and "regression boundaries" of the acceptance gate;
  - the "benchmark-specific stopping rule";
  - the test-set sizes of the five short-horizon benchmarks;
  - prompt templates.
- A text search of the paper finds no "causal", "inherit", "lineage", "genome", "germline", "population", "overfit" or "limitation". "Generation" appears only in the sense of patch/skill generation. There is no limitations section.

**Arithmetic I checked.** All of the following match the paper:
- The §5.4 claims of "average improvement of 14.45% and 31.20%" over HumanSkill and LLMSkill are **relative** gains of the five-benchmark mean. My computation gives 14.45% and 31.20%; the absolute gains are +10.1 pp and +19.1 pp.
- DP-Avg values reproduce from case counts: 28.33 = (27/60 + 7/60)/2; 21.67 = (25/60 + 1/60)/2; 19.17 = (22/60 + 1/60)/2.
- Shopping per-level counts for the full method: L1 13/25, L2 9/25, L3 5/10.

## Mechanism

The paper treats the optimization of **one Markdown skill document** for **one benchmark** as discrete, gradient-free optimization. It borrows two Adam principles "functionally, not numerically" (§4.2, Table 1).

- **Initialization.** An initial skill S₀ is written by an LLM from "fixed baseline execution trajectories" and their feedback. SkillOpt is given the same S₀.
- **Iteration t (Algorithm 1).**
  1. Sample a mini-batch B_t from the optimization pool and roll out the current skill S_{t−1}. This gives F_t^roll: per-case metric vectors E plus diagnostics C (error descriptions, judge rationales).
  2. An LLM patch generator produces g_t = G(S_{t−1}, trajectories, F_t^roll, M_{t−1}, σ_t). The candidate is S̃_t = Apply(S_{t−1}, g_t).
  3. **The candidate is evaluated on the same mini-batch B_t** (F_t^val), so the two evaluations are "directly comparable case-level feedback".
  4. A benchmark-specific acceptance gate decides a_t ∈ {0, 1}. If a_t = 1 the candidate replaces the current skill; otherwise the current skill is kept.
- **"First moment": the Evolving Issue Tracker (EIT) M_t.**
  - It is a set of structured issue entries I_j = (error pattern p_j, status z_j, set of attempted solutions with observed outcomes A_j).
  - An LLM function U_EIT updates it after every evaluated candidate, whether accepted or rejected. It links failures to existing issues, creates new entries, records each attempt and its outcome, and "reopens an issue when the same failure recurs".
  - The paper's role for it: decide *what* to revise, and keep later revisions from undoing earlier effective corrections ("Direction Stability").
- **"Second moment": the volatility-driven edit budget.**
  - Per-case paired improvement δ_{t,i} = s(E^val) − s(E^roll) on B_t.
  - V̂_t is the sample variance of the δ values.
  - V_t = β₂·V_{t−1} + (1 − β₂)·V̂_t.
  - The next budget is σ_{t+1} = max(b_min, round(b_base·[1 − clip(V_t / V_max, 0, 1)])).
  - High dispersion of per-case effects (some cases helped, some hurt) shrinks the allowed size of the next edit ("Update Adaptivity"). The variance is used **only for step size, never for acceptance**.
- **Acceptance gate (§5.3 "Acceptance Protocol").**
  - Rule, quoted: "A proposed revision is accepted when at least one designated metric reaches its improvement threshold and every protected metric remains within its regression boundary. An auxiliary metric can therefore support acceptance when the primary metric is unchanged."
  - The paper is explicit that "SkillAdam does not use an additional validation set for this decision." The gate is therefore an **in-sample** paired comparison, on the same cases from which the patch was just generated.
  - The thresholds and boundaries are not reported.
- **Output.** Only S_{T_max} is returned. The EIT and V are optimizer state and are discarded.
- **Setup (§5.3).**
  - No harness, direct chat, frozen target model. Skills are "injected as natural-language instructions".
  - GPT-5.5 is the target for six benchmarks (medium reasoning effort, temperature 1.0). DeepPlanning uses Claude Sonnet 4.5 at temperature 0.0.
  - For the six non-DeepPlanning benchmarks, SkillAdam **merges SkillOpt's train and selection partitions into one optimization pool**, while SkillOpt keeps its split.
  - DeepPlanning uses an odd/even case split.
  - One seed (42). "Each test case is evaluated once."

## Results (as reported; single run, point estimates, no CIs or tests)

- **Short-horizon benchmarks (Table 2; GPT-5.5).** Baselines are **copied from SkillOpt's paper**, not re-run.
  - SkillAdam vs SkillOpt:

    | Benchmark | SkillAdam | SkillOpt |
    |---|---|---|
    | SearchQA | 87.5 | 87.3 |
    | Spreadsheet | 81.1 | 80.7 |
    | OfficeQA | 72.1 | 72.1 |
    | DocVQA | 92.3 | 91.2 |
    | LiveMath | 67.7 | 66.9 |

  - The margins are **0 to 1.1 pp**: "four strict wins and one tie". Test sizes are unreported, there is one run, and sampling is at temperature 1.0. These margins cannot be distinguished from noise, and the data use differs (SkillAdam also optimizes on the selection partition).
- **Long-horizon benchmarks (Table 3).**
  - ALFWorld (GPT-5.5): NoSkill 83.6, reproduced SkillOpt 87.3, SkillAdam 89.6.
  - DeepPlanning (Sonnet 4.5, reproduced): DP-Shopping 31.7 / 41.7 / 45.0 (NoSkill / SkillOpt / SkillAdam); DP-Travel 0.0 / 1.7 / 11.7; DP-Avg 15.8 / 21.7 / 28.3.
  - The headline +6.7 pp DP-Avg comes mostly from DP-Travel. That is **7/60 vs 1/60 cases**. The paper reports no paired data. My bound: even if all six discordant cases favour SkillAdam, the exact two-sided p is about 0.03, and it is weaker with any reverse discordance.
- **Ablation (Table 4; cumulative; DeepPlanning only).**
  - Full M+B: 28.3. Without the edit budget (−B): 21.7. Without budget and memory (−B −M): 19.2.
  - The budget effect is again DP-Travel 7/60 vs 1/60, and Shopping 27/60 vs 25/60 (2 cases).
  - The memory effect is Shopping 25/60 vs 22/60 (3 cases). Shopping L1 actually **drops** when memory is added (56 → 48).
  - The authors themselves note that the cumulative design "does not independently isolate the effect of memory when the edit budget is enabled or the interaction".
  - *Observation (cannot verify):* the −B row (41.7 / 1.7 / 21.7) is identical to the reproduced SkillOpt row in Table 3. This may be a coincidence at n = 60.
- **Cross-model transfer (Table 5).**
  - GPT-5.5-optimized skills are deployed verbatim on GPT-5.4-mini over six benchmarks.
  - Average target score: SkillAdam 67.8, SkillOpt 63.1, NoSkill 48.9. Retention: 81.3% vs 76.2%.
  - SkillAdam wins 5/6 on score. The largest gap is ALFWorld (82.8 vs 66.4, where SkillOpt falls *below* NoSkill at 73.1).
  - NoSkill numbers are taken from SkillOpt's paper.
- **Skill quality (Figure 3).**
  - A GPT-5.5 judge (same family as the executor), 3 calls per skill, six 1–5 dimensions, blind to method name.
  - SkillAdam beats SkillOpt on 10/10 slices; mean 3.88 vs 3.30.
- **Cost (Table 6; DeepPlanning).** Optimization-phase tokens 74.0M vs 226.6M (−67.3%); API requests 2,830 vs 9,071 (−68.8%).
- **Dynamics (Figure 4; Shopping L1; one run).**
  - Accepted SkillAdam skills stay above the 40% initial accuracy. SkillOpt's accepted updates fluctuate, and its high-scoring later candidates are not retained.
  - SkillAdam rejected a revision at iteration 8 that had a high post-hoc test score. The authors concede that, with one run and memory and gate acting together, this "characterizes the overall optimization behavior rather than isolating the causal effect of memory alone".

## Overlap vs HeritAgent chain (full-text judgement)

| Element | Coverage | Evidence |
|---|---|---|
| 1. Lifetime acquisition | **Full** (with caveats) | The skill is constructed from the agent's own execution trajectories plus evaluator feedback (initialization), then revised iteratively from rollouts (§4.1 "Rollout", "Skill Update"). This is the same class of trajectory-driven skill learning the ledger already rates F for GRASP and VaG. Caveats: (a) it is offline optimization over a training pool with a privileged benchmark evaluator, not acquisition during deployment; (b) the unit is revisions of **one** document per benchmark, not the acquisition of discrete, separately identifiable traits. |
| 2. Somatic / germline separation | **Absent** | There is a single current skill S_t that is both the working artifact and the output. There is no staging store of candidate traits. Rejected candidates are only logged as attempts inside the EIT and are never promoted later. The only boundary is between **optimizer state** (EIT M_t, volatility V_t), which is discarded, and the **returned skill**. This is an optimizer-vs-output split, not a lifetime-store vs heritable-store split. At most it is a weak analogue of OurArk's "private state not inherited". |
| 3. Typed executable genome | **Absent** | The artifact is "a Markdown-formatted natural-language instruction" (§3.1), injected as text. There is no schema, no trait records, no policy/skill/regulation typing, no executable code, and no versioning or lineage of skill versions (only accept/keep). The structured, typed records in the paper are the EIT issue entries (pattern, status, attempts with outcomes). Those belong to the optimizer, not to the delivered artifact. |
| 4. Causal validation gate | **Partial (weak)** | The gate is a prerequisite for replacement (Eq. 23–24) and is a **paired candidate-vs-incumbent comparison on identical cases** with a multi-metric rule: an improvement threshold on a designated metric plus regression boundaries on protected metrics. That gives it the shape of an interventional before/after contrast with regression protection. Against CIG it is weak on every evidential axis. (a) It is **in-sample**: the gate cases are the same mini-batch the patch was generated from, and "SkillAdam does not use an additional validation set". (b) There is one sample per case, at temperature 1.0 for GPT-5.5. (c) There is no ablation or attribution, because the whole document is the unit. (d) There is no held-out stage, in-scope or out-of-scope. (e) Regression protection is per-metric aggregate on the batch, not a protected case set. (f) Thresholds are unreported and there is no statistical test. The per-case paired δ variance is computed but used only for step size. The gate decides within-lifetime replacement of the live document, never heritability. |
| 5. Selective assimilation | **Partial (weak)** | Accept/reject of each revision into the single live skill. Nothing is assimilated into a store distinct from the working state, and there is no per-trait selection, because the whole document is replaced or kept. |
| 6. Multi-generational, frozen weights | **Partial** (frozen half full; generational half absent) | "The target model remains frozen during optimization" (§5.3). There are no generations, descendants, lineages or populations. The only inheritance-like event is a one-shot, verbatim, non-re-validated transfer of the final skill from GPT-5.5 to GPT-5.4-mini (§5.6). That is the cross-backbone transfer already conceded in ledger §3 item 24. |

**Implements full chain: no.** One element is full (with caveats), three are partial (two of them weak), and two are absent. SkillAdam is primarily an **optimizer-design** paper for single-document skill refinement. Its contribution (cross-iteration optimizer memory plus variance-adaptive edit size) is orthogonal to HeritAgent's claimed contribution (what is inherited, and on what evidence).

## Does the full text change the abstract-level assessment?

There was no §2.10 entry. The prior assessments were two mentions:
- In §2.10.10 and §2.12, SkillAdam is listed among the "commit-gate competitors named by deep reads (titles confirmed on arXiv, not read)".
- In §3 item 14 it is cited as "concurrent work" for candidate-vs-incumbent paired checks, structural-growth control and the "skill-evolution overfitting" observation. This came from SkillEvoReg's bibliography.

Both mentions together imply roughly MEDIUM-LOW for element 4.

**Threat to the chain claim: LOW, down from the implied "commit-gate competitor".**

- **Why lower.**
  - The gate is the weakest seen in this cluster. It is an in-sample, same-batch, single-sample, unreported-threshold accept rule, with **no validation set by design**. It is weaker than SkillOpt's own selection split, GRASP's balanced held-out probe, SAGE's per-item paired test, and VaG's held-out add-one replay.
  - It is not trait-level: the whole document is the unit.
  - It never decides persistence into anything other than the live document, and there are no generations.
  - SkillAdam does not use the word "overfit" at all, and it does not regularize library growth. The §3 item-14 attribution of the "skill-evolution overfitting observation" to SkillAdam is therefore inaccurate. The closest SkillAdam content is the observation that "successive revisions may focus on different problems and undo one another", plus edit-*size* control (see the suggested ledger fix below).
- **Fair the other way: points a reviewer could legitimately raise.**
  - (i) **Per-case paired effect dispersion as a reliability signal.** SkillAdam explicitly treats "improves some cases but degrades others" (high Var δ) as evidence against broad change. That is conceptually close to the CIG argument that a mean gain hiding case-level regressions should not be trusted. SkillAdam uses the signal for step size, not admission, but it is prior art for the *statistic*.
  - (ii) **A structured cross-iteration record of issues, attempted fixes and their outcomes, including rejected attempts, that conditions later updates.** This is functionally a negative-results log, close to what CIG validation records plus a somatic store of rejected candidates would hold.
  - (iii) **Evidence (weak: one run, n = 60 per domain) that an in-sample paired gate plus memory beats SkillOpt** while using about a third of the tokens. A reviewer may cite this to argue that cheap in-sample gating suffices. Our counter is that SkillAdam has no held-out or regression-set evaluation inside the gate, and its own Figure 4 shows a strong candidate *rejected* by the in-sample gate.
  - (iv) **Transfer retention of 81% to a smaller backbone without re-validation**, the kind of number H4 will be compared against.
  - HeritAgent must therefore not present the following as new: "paired candidate-vs-incumbent acceptance with protected-metric regression boundaries"; "adaptive edit-size control"; "memory of past attempts as a stabilizer of skill evolution".

## Consequences for HeritAgent

No design was changed. Items marked *requires ADR* are for the Builder/ADR owner.

1. **Arm B′: SkillAdam as the "weakest published performance gate", not as the B′ of choice. *Requires ADR (ADR-0002 amendment) only if adopted.***
   - A SkillAdam-style in-sample gate would be: accept iff the candidate beats the incumbent on the *same* episodes it was mined from, on a designated metric, with protected metrics inside a regression boundary. It is cheaper than GRASP or SAGE options (a)–(e) in ledger §5 item 1 and the SAGE/SkillCommit notes.
   - It is a poor *sole* B′, because a reviewer will dismiss it as a straw man next to GRASP or SAGE.
   - Recommendation: keep a held-out performance gate (GRASP- or SAGE-style) as the primary B′. Mention SkillAdam's in-sample gate only as a possible lower rung if a "gate ladder" (in-sample → held-out → full CIG) is pre-registered to localise which CIG property carries any effect.
2. **Disjointness of trait-mining and gate episodes. *Requires ADR if not already specified.***
   - SkillAdam generates the patch from batch B_t and gates it on the same B_t. This is the textbook in-sample optimism SAGE calls the "Optimizer's Curse". SkillAdam's own Figure 4 (a rejected candidate with high test accuracy, accepted ones with lower) shows the in-sample gate is noisy in both directions.
   - EXP-0001 already separates gate-holdout from final test. It should also state explicitly that the episodes from which a trait was mined are **not** reused as CIG stage-2/3 evidence. Stage-1 replay on the source environment is the intended exception and should be labelled as such.
3. **Report per-case paired effect dispersion in CIG stages 2–4. *Reporting only, no ADR; using it as a threshold requires ADR.***
   - Alongside the mean Δ and bootstrap CI, report the per-episode paired δ distribution: variance, and the help/hurt (n10/n01) counts recommended in ledger §5 item 4.
   - A high-dispersion trait, one that helps some contexts and hurts others, is a candidate for a regulation condition (`express_when`) rather than unconditional assimilation. This is a concrete mechanism-level use of SkillAdam's statistic that SkillAdam does not make.
4. **Optimizer/somatic memory as an uncontrolled inheritance channel. *Requires ADR (ADR-0001/0002 clarification).***
   - SkillAdam's gains are partly attributed to an attempt-and-outcome memory that is *not* part of the delivered skill. In HeritAgent the analogous object is the somatic store plus CIG records of rejected candidates.
   - Decide and pre-register whether descendants can read ancestors' rejected-trait records or issue histories. If they can, that is information inheritance outside the EAG, and it must be either held constant across arms A/B/B′/C or counted as part of what is inherited.
   - Likewise, if any arm's within-lifetime learner uses an issue-tracker-style memory, all arms must use the same within-lifetime learner, so that arm differences reflect the inheritance rule, not the optimizer.
5. **Cost reporting. *Reporting only, no ADR.*** SkillAdam reports optimization-phase input/output tokens and API requests separately, with per-request cost held roughly constant. EXP-0001's descendant adaptation cost and the CIG's own gate cost should be reported in the same units (tokens and requests), so that "CIG is expensive" can be quantified against published optimizers.
6. **Statistics and power.**
   - SkillAdam's headline long-horizon effects are 6 cases out of 60 (DP-Travel) and 2–3 cases out of 60 (Shopping), from one seed and one evaluation per case. Its short-horizon "state of the art" claim rests on 0–1.1 pp margins against numbers copied from another paper, with different data use.
   - This is the same under-powering risk flagged for EXP-0001 in the SAGE and SkillCommit notes. Pre-registered seeds, exact paired tests and minimum detectable effects *require ADR*.
   - If HeritAgent ever compares against SkillAdam or SkillOpt numbers, it must re-run them, not copy them.
7. **Wording and naming.**
   - Do not name CIG/EAG components "issue tracker", "optimization memory", "edit budget", "volatility" or "first/second moment" without citing SkillAdam.
   - Do not claim Adam- or gradient-analogy framings, adaptive edit-size control, or paired same-case candidate-vs-incumbent acceptance with protected-metric regression boundaries.
   - When contrasting, say: "an in-sample, single-document refinement optimizer with an optimizer-private memory; no held-out, trait-level or heritability gate".
8. **Round-4 queue.** Works cited by SkillAdam. Titles were confirmed on arXiv abstract pages on 2026-10-13; none were read:
   - **SkillAxe** 2606.10546 (*Sharpening LLM-Authored Agent Skills Through Evaluation-Guided Self-Refinement*);
   - **SkillOS** 2605.06614 (*Learning Skill Curation for Self-Evolving Agents*; it trains a curator of an external skill repository, which is potentially relevant to selective assimilation);
   - **CoEvoSkills** 2604.01687 (*Self-Evolving Agent Skills via Co-Evolutionary Verification*; a surrogate verifier, relevant to the gate).
   - SkillOpt 2605.23904 (title re-confirmed) remains in the §2.12 queue. It is SkillAdam's main baseline and, unlike SkillAdam, uses a separate selection split.
   - SkillAdam cites none of VaG, GRASP, SAGE, SEA or SkillCommit.

## Suggested ledger update (new §2.10 entry or §2.12 resolution; pending cross-review)

- **SkillAdam 2609.08944 (Sep 2026 preprint; Renmin University of China + Tencent): deep-read, threat LOW.** The full text lowers the implied "commit-gate competitor" rating.
- **Chain coverage:** 1 F (caveats: offline pool, one document) · 2 – · 3 – · 4 P (weak: in-sample same-batch paired accept rule with protected-metric regression boundaries; no validation set by design; thresholds unreported) · 5 P (weak: accept/keep of the live document) · 6 P (frozen weights; one-shot verbatim GPT-5.5 → GPT-5.4-mini transfer, 81.3% retention).
- **Evidence caveats:** one seed and one evaluation per case; short-horizon margins 0–1.1 pp against baselines copied from SkillOpt with different data use; long-horizon effects of 6/60 and 2–3/60 cases; cumulative ablation only; hyperparameters, gate thresholds and test sizes unreported; no appendix in v1; code unchecked.
- **Fix §3 item 14:** SkillAdam does not discuss overfitting. Re-attribute it to "cross-iteration optimizer memory of issues and attempted fixes; variance-adaptive edit-size control of skill-document updates; paired same-case candidate-vs-incumbent multi-metric acceptance with protected-metric regression boundaries (in-sample)". Keep SkillEvoReg as the source for the overfitting observation.
