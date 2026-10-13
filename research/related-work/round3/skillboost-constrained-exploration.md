# SkillBoost — Constrained Exploration-Exploitation against Skill Overfitting — detailed notes (Round 3)

**Paper:** *Rethinking Self-Evolution: A Constrained Exploration-Exploitation Process for Mitigating Skill Overfitting* — Hongqiang Lin†, Chao Liu† (equal contribution), Xiaofan Bai, Xuan Jin, Yuhong Li, Nenggan Zheng\*, Xipeng Cao\* (corresponding). Affiliations: Zhejiang University (Lin, Zheng) and Alibaba Group (others). arXiv:2607.26643 (cs.AI). v1 was submitted on 29 Jul 2026 and v2 on 19 Aug 2026, both 663 KB. The licence is CC BY 4.0. The comments field is empty, and the paper is a preprint with no venue. Code is claimed at github.com/HQ-Lin/SkillBoost. That page returned HTTP 403 through this session's proxy, so **I could not verify the code**. Reviewed 2026-10-13 by Claude (third agent), pending Research/Critic cross-review.

**Read depth:** full text of v2 via arxiv.org/html/2607.26643. I downloaded the raw HTML, stripped the tags and read the text end to end: §§1–5 and Appendices A–F, plus Appendix G in part (I read the ALFWorld v1→v2 and BFCL skill pairs and skimmed the rest). Quotes below are short and verbatim from that text. Numbers come from the HTML tables. I did not cross-check the PDF.

## Mechanism

- **Setting:** frozen LLM parameters θ (§3.1: "θ denotes the frozen LLM parameters"). The trainable state is a single SKILL.md per benchmark and model.
  - The skill is "a structured state vector, whose coordinates correspond to key components (e.g., Metadata, Workflow, Constraints, References)".
  - Evolution is cast as an MDP: state = skill version, action = a set of targeted add/delete/replace edits on components, transition = deterministic application of the edits, reward = performance change.
- **Each iteration (Algorithm 1)** runs four stages:
  1. **Forward rollout** on the dataset 𝒟 (the *training split*) collects step-level trajectories τ_i of observations, tool calls and outputs. The paper says this provides "causal evidence for fine-grained failure attribution". That is the only occurrence of "causal" in the paper. The evidence is observational trace reading, not intervention.
  2. **Structured exploitation (attribution).** An LLM analyzer agent runs three checks over the failed trajectories:
     - workflow compliance: skipped steps, ambiguous instructions, or ungrounded conclusions;
     - reasoning-chain reconstruction, which finds the first deviating step j\*;
     - root-cause clustering, which separates editable "strategy defects" from "capability gaps".

     The output is a diagnosis g_t = {(cause_k, target_k)}, where each target_k is the skill component to edit.
  3. **Prior-guided exploration (Best-of-N).** The analyzer produces N repair briefs that share g_t and differ only in repair strategy π(n), meaning edit scope (narrow or wider) and cluster priority (App. D). Each brief yields one candidate skill. The default is N = 4.
  4. **Verified acceptance.** Gain r(s′) = Eval(s′,𝒟) − Eval(s_t,𝒟). The two case-level metrics are Fix (fail→pass) and Regress (pass→fail), so r ∝ Fix − Regress.
     - Rule: "Accept(s′) ⇔ r(s′) > 0 ∧ Regress(s′) < ϵ". The highest-gain accepted candidate becomes s_{t+1}. If no candidate is accepted, the skill stays unchanged.
     - Cost is controlled by a two-phase cascade (App. D). Phase A screens all N candidates on the previous round's *failure set* ("typically 5–30 instances"). Phase B evaluates the top K = 2 on the full set.
     - App. F also lists "Guard samples 10–20: Random correct cases added against regressions".
- **On which data the gate is evaluated:** the main text defines Eval over the same 𝒟 used for the forward rollout. App. B's worked example says "Back-test v2 on full training set". Its numbers are LiveMath train split = 35 questions, v1 8/35 → v2 13/35, with 6 fixes and 1 regression, so the candidate was accepted. App. D calls the Phase-B set the "complete validation set". **Read in context, the acceptance gate is an in-sample (training-split) Fix/Regress test, not a held-out gate.** Table III's "Train" column is the same split. Held-out *test* splits are used only for reporting. App. D's "validation" wording conflicts with this, and the paper never describes a separate validation split.
- **The regression threshold ϵ never gets a numerical value.** It is absent from the hyperparameter table (Table XI) and from the text. The worked example uses "regression count ≥ repaired cases → reject", which is just the r > 0 condition. The number of iterations T is also not reported; Fig. 1 shows "six skill versions". Split sizes are not reported except LiveMath train = 35.
- **Statistics.** No per-question statistical test is used inside the gate: a single evaluation decides acceptance. Decoding is at temperature 0.1 with seed 42, and results come from a single run. App. E's significance analysis is at the configuration level only: n = 20 model–benchmark pairs, paired t-test, Wilcoxon and sign test against SkillOpt and against the best baseline per setting. The authors themselves note that it "measures the consistency of the gains across evaluation settings rather than per-question variance".
- **Versioning:** the evolved SKILL.md frontmatter carries `type: task_skill`, `current_version`, `parent_version` and `mutation_brief` (App. G). This is a version chain of one document. No `parent_version` points to another agent.
- **Theory (§3.3, App. C):** under idealised uniform sampling in an ϵ-ball, Best-of-N directional progress is ≈ ϵ√(2 ln N / d), so gains saturate in N. This motivates N = 4. It is not a guarantee about the gate.

## Results

- **Backbones and benchmarks:**
  - Claude-opus-4-6, Qwen-3.7-max, Qwen-3.6-plus, DeepSeek-v4-pro and Kimi-k2.6 on SpreadsheetBench, BFCL-v4, LiveMathematicianBench and ALFWorld: 20 pairs.
  - DocVQA on Qwen-3.6-plus, Qwen-3.7-plus and Kimi-k2.6: 3 pairs, with no Trace2Skill or SkillOpt baselines.
  - That gives the "23 model–benchmark configurations".
- **Baselines:** no skill, human skill, one-shot LLM skill, Trace2Skill, SkillOpt.
- **Main table (Table I, held-out test):** SkillBoost is best in all 20 pairs and gains over no-skill in every one. Examples:
  - Claude LiveMath: 28.6 → 76.0 (SkillOpt 49.6).
  - Claude SpreadsheetBench: 50.0 → 82.5.
  - Qwen-3.6-plus ALFWorld: 60.6 → 87.5.
  - Smallest gains: Qwen-3.6-plus BFCL +2.3 and Qwen-3.7-max BFCL +3.3.

  I re-checked the per-model mean gains of +10.6 (Kimi) and +28.4 (Claude) from the table.
- **Overfitting (Table III, Δ = Test − Train):** SkillOpt ranges from −1.9 to −26.7 and Trace2Skill from −3.0 to −17.7. SkillBoost ranges from −3.8 to +2.5, which the text calls "near zero". The worst SkillBoost values are −3.8 (Kimi Spreadsheet) and −1.3 (DeepSeek Spreadsheet).
- **Gate ablation (Table V):** "w/o Gate" means the top candidate on the failure set is accepted without a full-set back-test. Accuracy falls on all six cells, for example Claude BFCL 48.5 → 37.7 and DeepSeek Sheet 80.0 → 67.5. The ablation removes the whole back-test. **It does not isolate ϵ (the regression bound) from the r > 0 condition, and no ϵ sweep is reported.**
- **Attribution ablation (Table IV):** removing any one of the workflow, reasoning or root-cause checks lowers test accuracy in all nine cells. Example: DeepSeek Math 46.4 → 27.2 without reasoning reconstruction.
- **Rejection example (§4.4):** "a rule expansion from 87 to 150 lines fixes 20 cases but breaks 23 (net −3), so the gate rejects it."
- **Transfer (Table VI):**
  - Cross-model: skills optimised with Claude transferred to DeepSeek and Kimi beat the targets' human seed skills by +0.7 to +14.3.
  - Cross-benchmark: the LiveMath skill moved to OlympiadBench gives DeepSeek 44.0 → 50.0 and Kimi 46.0 → 60.0.
  - The comparison is against *human skills only*, not against target-native SkillBoost runs. "Transfer" here means a frozen document reused by another model, not inheritance by a descendant agent.
- **Other findings:**
  - Skill length grows monotonically under SkillOpt, while SkillBoost stays compact and prunes on BFCL (Fig. 3).
  - N saturates after 4 (Table VII: Sheet 61.1 / 66.8 / 77.9 / 79.6 / 80.4 for N = 1, 2, 4, 6, 8).
  - A structured SKILL.md beats a scrambled one by 0.6–7.2 points (Table X).
  - Optimisation costs 95K–154K tokens per round. Evaluation costs 3–8M tokens per run, 85–97% of the total.
- **Stated limitation (§6):** the method depends on the base model's ability to attribute failures, generate edits and evaluate candidates. The paper has no separate limitations section.

## Overlap vs HeritAgent chain (full-text judgement)

| Element | Coverage | Evidence |
|---|---|---|
| 1. Lifetime acquisition from trajectories | **Full** | §3.2 runs a forward rollout on the training split and refines the skill from failed trajectories. As with RSEA, the "lifetime" is an offline training phase starting from a seed skill, which counts as full for this element. |
| 2. Somatic / germline separation | **Absent** | One live SKILL.md per run. Candidates are transient, not a staging store of acquired traits. Nothing is acquired-but-not-inherited, and there is no promotion event between stores. |
| 3. Typed executable genome | **Partial (weak)** | Section-typed components ("Metadata, Workflow, Constraints, References"), component-targeted edits, and version frontmatter (`current_version`, `parent_version`, `mutation_brief`). It is a single natural-language document, not an executable genome of policies + skills + regulation. It has no per-trait validation records and no expression/regulation layer. *(Upgrade from the abstract-level "–".)* |
| 4. Causal validation gate | **Partial (weak)** | Before commit, the whole candidate version is back-tested on the training split with a Fix/Regress decomposition, r > 0 ∧ Regress < ϵ. Attribution is an LLM's reading of trajectories (observational), used to *localise edits*, not to *validate* them. The gate has no controlled ablation, no held-out or out-of-scope probe, no per-item statistics, and ϵ is unreported. It is weaker than GRASP's or VaG's held-out gates. The paper's single use of "causal" is for observational trajectories. |
| 5. Selective assimilation | **Partial (weak)** | It is selective: most candidates are rejected and the best accepted one is chosen. The unit, however, is a whole candidate version (one edit set covering one or more failure clusters). Individually validated traits are not admitted into a store. Edits are localised to attributed components, but they are not validated per component. |
| 6. Multi-generational, frozen weights | **Partial** | Frozen weights: yes (§3.1). The "generations" are skill versions v1 → v_T of one optimisation run. There are no agent generations with distinct lifetimes, and no descendants inherit or re-adapt. Cross-model reuse of the final skill is a transfer test, not an inheritance mechanism. |

**Implements full chain:** no. Element 2 is absent and 3–6 are partial, most of them weakly.

## Changed vs abstract-level assessment

The abstract-level vector was 1 F · 2 – · 3 – · 4 P · 5 P · 6 P. After the full read it is **1 F · 2 – · 3 P(weak) · 4 P(weak) · 5 P(weak) · 6 P**.

**Threat goes down, from MEDIUM to LOW** for the HeritAgent novelty claim. Reasons:

1. The "verified acceptance" gate is *in-sample*. It is evaluated on the same training split the failures came from, plus 10–20 guard cases, and is not a held-out net-gain gate. It is therefore a strict subset of the gates already recorded in §3 items 9–11 (GRASP, VaG, SAGE, DarwinX).
2. The regression bound ϵ is never quantified or ablated. The gate ablation removes the entire back-test, so it supports only the already-recorded finding that "the acceptance gate drives gains" (§3 item 15).
3. Its "attribution" is LLM root-cause diagnosis that steers *where to edit*. It is not interventional contribution evidence that decides *whether to keep*. It therefore does not pre-empt CIG's ablation or attribution stages.
4. There is no inheritance semantics at all: one document, one run, version chain only.

Element 3 rises slightly (component-typed SKILL.md with parent_version frontmatter), but this is weaker than the Agentic Harness Engineering and GRASP analogues already listed. The work stays in the "verified acceptance with regression budget" cluster as the ledger said. Its specific contributions are worth citing: the overfitting-gap measurement (Test − Train) and the empirical evidence that *structured failure attribution* improves skill repair quality.

## Consequences for HeritAgent (each item **requires ADR** if it changes EXP-0001; nothing was changed here)

1. **Arm B′ operationalisation (ADR-0002 amendment).** SkillBoost gives a fourth candidate, the weakest in the ladder: (d) "in-sample Fix/Regress gate". Admit iff Fix − Regress > 0 on the training/replay set and Regress < ϵ, with no held-out stage. It sits *below* the GRASP/VaG-style held-out gates already proposed in ledger §5.1. If B′ is a single arm, the held-out variant (a) or (b) is still the stronger, reviewer-natural baseline. Reporting (d) as a secondary cheap-gate analysis would show how much of C's advantage over B comes from any back-test at all versus from held-out and trait-level stages. **Requires ADR.**
2. **Overfitting-gap reporting.** Report Δ = held-out − training (or replay-set) score per arm and per generation in EXP-0001, so results are comparable with SkillBoost's Table III. This changes only the reporting format, not a threshold. An ADR note is recommended but not required.
3. **CIG statistics.** SkillBoost's gate has no per-item test, one seed and an unreported ϵ. CIG must not repeat this. Pre-register the regression bound τ_r numerically, report the n01/n10 (Regress/Fix) contingency per CIG stage (already proposed in ledger §5.4), and run per-item paired tests rather than configuration-level t-tests. This is consistent with existing §5.4 items. Any threshold change **requires ADR**.
4. **Separate localisation from validation.** SkillBoost uses attribution only to localise edits. CIG's attribution stage should be documented explicitly as *validation evidence* (contribution estimates under intervention), and the LLM-diagnosis step that proposes traits should be kept outside the gate. This is compatible with ledger §5.6 (gate outside the genome). It is a design-documentation note for the CIG ADR. **Requires ADR** if it changes the stage definition.
5. **Length control.** Their Fig. 3 (monotone growth of skills under SkillOpt) and the 87 → 150-line rejected rule reinforce ledger §5.5: report genome or prompt token counts per arm. Secondary analysis only.

## Naming hazards

- Avoid "verified acceptance", "structured exploitation", "prior-guided exploration", "Repair Brief" and "skill overfitting" as HeritAgent component names. "Skill-evolution overfitting" is already attributed to SkillEvoReg in §3 item 14, and SkillBoost uses "skill overfitting" independently.
- "Attribution" here means LLM failure localisation. When HeritAgent says "attribution", qualify it as contribution estimation under intervention.

## Anti-overclaim notes (SPEC §34)

- SkillBoost's own "state-of-the-art" claim rests on single-run accuracies with configuration-level significance tests. Do not cite its numbers as per-question-significant.
- "Near-zero gap" holds for most cells but not all (−3.8 on Kimi Spreadsheet).
