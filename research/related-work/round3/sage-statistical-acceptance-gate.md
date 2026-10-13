# SAGE — A Statistical Acceptance Gate for Self-Evolving Agents — detailed notes (Round 3)

**Paper:** *SAGE: A Statistical Acceptance Gate for Self-Evolving Agents*. The authors are Yihao Wang, Linhan Xia, Rui Liu, Zhaofeng Zhang, Hongyu Wu, Yang Yang, Jinglu He, Yu Guo and Kai Lei. The affiliations listed in the HTML are Peking University, University of Oklahoma, Tencent, Imperial College London, University of Michigan, University of Edinburgh, Xunce Technology and GienTech Technology. The corresponding author is at pkusz.edu.cn. arXiv:2609.36043v1 [cs.AI], submitted 28 Sep 2026, arXiv non-exclusive licence. The comments field reads "10 pages, 2 figures, 2 tables". Only v1 exists and no venue is given, so it is a preprint. Reviewed 2026-10-13 by Claude (third agent), pending Research/Critic cross-review.

**Read depth: full text.** I downloaded arxiv.org/html/2609.36043 with curl and converted it to text, keeping the LaTeX `alttext` of every formula so the equations and numbers stay intact. I read §§1–6 and the references [1]–[26]. The abs page supplied the metadata. Quotes below are short and verbatim. The numbers are from Tables 1–2. I recomputed the averages they quote and they match: C1 − SkillOpt = 6.20, SAGE − C1 = 2.53, SAGE − SkillOpt = 8.73. I did not check the PDF.

## Mechanism

- **Setting.** The agent runs on a frozen model and improves itself by editing "a persistent skill document that encodes their workflow, tool-use rules, and decision logic". SAGE reuses SkillOpt (arXiv:2605.23904) exactly: "every edit is proposed by the same SkillOpt optimizer and the only difference is the gate that accepts them". The paper does not describe how edits are derived from experience. That part is entirely SkillOpt's: a textual learning rate, a rejected-edit buffer and slow epoch-wise updates (§2).
- **Baseline gate g0 (Eq. 3).** Accept edit e_n if J_D(S^(t)+e_n) > J_D(S^(t)), where J_D is the aggregate binary-verifier score on a held-out validation set D.
- **Component 1: per-item paired ledger (§3.2).** The incumbent and the candidate are run on the same validation items, and each item is classified as both-correct, both-wrong, a win (w) or a regression (ℓ). Two statistics follow:
  - a discounted net gain Δ_λ = w − λℓ, with λ ≥ 1, citing the "first-do-no-harm principle of asymmetric counterfactual utilities" (Ben-Michael et al. JASA 2024; Träuble et al. NeurIPS);
  - a regression fraction ρ = ℓ / n_S, where n_S is the number of validation items the incumbent already solves.
- **Component 2: hypothesis-test denoising (§3.3).** Let q = Pr(W | W ∪ L). The null is H0: q ≤ λ/(1+λ). The p-value is the exact conditional binomial tail p_λ = Pr[Bin(m, q_λ) ≥ w], where m = w + ℓ. For λ = 1 this is the one-sided exact McNemar test.
  - The paper argues explicitly *against* shrinking aggregate scores. Scores of different edits on a shared validation set are correlated through item difficulty, so the paired view is the right fix. It cites Smith & Winkler 2006 on the Optimizer's Curse and Hawinkel et al. 2025 on the winner's curse under dependence.
  - When m = 0 the gate abstains.
  - Worked example from the paper: with λ = 1, 5 wins to 3 losses is not significant (p = 0.363 by my calculation), while 8–0 is (p = 0.0039).
- **Accept rule (Eq. 11).** A = I[Δ_λ > 0, p_λ < α, ρ ≤ τ]. Otherwise the gate abstains and keeps S^(t).
  - At (λ, α, τ) = (1, 1, 1) the rule reduces exactly to g0.
  - Under the conservative settings, every edit SAGE accepts would also be accepted by g0. This holds for a *fixed incumbent*. Along a trajectory the two incumbents diverge after the first decision that differs, so the abstract's phrase "commits only a subset of the baseline's edits" is a per-step statement.
- **Not specified:**
  - how several edits in one mini-batch that each pass the gate are combined;
  - the values of λ, α and τ that were used ("tuned on held-out data only");
  - the sizes of the validation and test sets;
  - the number of seeds or runs;
  - any code release. No repository link was found in the text.
- **Vocabulary.** A text search found no occurrence of "causal", "inherit", "lineage" or "population". "Generation" appears only in the sense of edit proposal. "Counterfactual" appears only in the cited title about asymmetric counterfactual utilities.

## Results

- **Setup (§4.1).**
  - Five text benchmarks: LiveMath, SpreadsheetBench, SearchQA, OfficeQA Pro and ALFWorld. DocVQA is omitted because not every backbone is multimodal.
  - Four open backbones, as named in the paper: DeepSeek-V4-flash, GLM-5.2, MiniMax-M3 and Qwen3.6.
  - Equal budget for both gates: the same number of proposed edits and the same token cost.
- **Regression rate (Table 1).**
  - The paper defines it as "the fraction of the items the current skill already solves that an accepted edit breaks". It does not say whether this is measured on the gate's validation items or on test.
  - Under the naive gate the rate ranges from 2.4% to 45.6%. LiveMath (27.5–39.4%) and OfficeQA (38.7–45.6%) are the worst.
  - SAGE reaches exactly 0.0% in all 12 LiveMath, SSB and OfficeQA cells. It is lower in 19 of 20 cells and equal in 1 (GLM-5.2 on SearchQA, 2.4% vs 2.4%).
  - **Caveat.** If the metric is computed on the validation ledger, an exact 0.0% follows almost by construction whenever τ is small. The paper does not report τ, so this cannot be checked.
- **Final score (Table 2; presumably on test, but not stated).**
  - SAGE is highest in all 20 cells. Mean gains are SkillOpt → C1 (paired criterion without the test) +6.20 and C1 → SAGE +2.53.
  - The largest gains are on LiveMath and OfficeQA: DeepSeek-V4 rises from 34.15 to 48.78 on LiveMath, and MiniMax-M3 rises from 43.19 to 60.12 on OfficeQA.
  - The smallest SAGE − C1 gaps are 0.25 (DeepSeek SSB 50.97 → 51.22) and 0.82 (Qwen ALFWorld 72.09 → 72.91).
  - Every number is a point estimate. There are no CIs, no seeds and no significance tests across runs, so "highest in 20/20" cannot be separated from noise for the small gaps. This is ironic in a paper about the Optimizer's Curse.
- **Ablation.**
  - The only ablation is C1 vs full SAGE.
  - §3.1 claims the boundary-recovery property is "verified in the ablation study", but no table shows the (1, 1, 1) setting.
  - There is no sensitivity analysis over λ, α or τ.
- **Discussion (§5.1).** The benefit "grows with the regression risk": SAGE is close to a pass-through on low-noise, verifiable tasks such as ALFWorld and SearchQA.
- **Stated limitations (§5.2):**
  - binary verifier only;
  - the sign test ignores the size of an improvement;
  - "the statistical guarantee is also per-comparison in scope … the validation set is reused across optimization steps, so trajectory-level error rates are not controlled";
  - hyperparameters still need manual tuning;
  - text-only evaluation.
- **Future work (§5.3).** Signed-rank tests for graded rewards; adaptive α via empirical-Bernstein bounds or confidence sequences (Howard et al. 2021); designing the optimizer and the gate together; and multi-agent joint acceptance, where an edit's effect "is no longer decidable from local evidence alone".

## Overlap vs HeritAgent chain (full-text judgement)

| Element | Coverage | Evidence |
|---|---|---|
| 1. Lifetime acquisition | **Full (inherited from SkillOpt; caveat)** | The system as run self-edits a persistent skill document of a frozen agent from experience: §1, "agents increasingly improve themselves by editing a persistent skill document". The acquisition mechanism is SkillOpt's, not SAGE's contribution, and this paper does not describe it. I still mark it full, because a reviewer would credit the system with it. |
| 2. Somatic / germline separation | **Absent** | There are only an incumbent and transient candidates (S^(t) vs S^(t)+e_n). SkillOpt's rejected-edit buffer records failed proposals; it is not a store of acquired-but-not-yet-heritable traits. No trait exists outside the one live document. |
| 3. Typed executable genome | **Absent** | One untyped natural-language skill document. No schema, trait types, regulation layer, per-trait provenance or validation records. |
| 4. Causal validation gate | **Partial (strongest statistics in the cluster so far)** | For each edit, the incumbent is compared with incumbent + edit on identical held-out items. This is an add-one interventional contrast, tested with an exact one-sided conditional binomial (McNemar) test, an asymmetric regression penalty, a regression cap relative to solved items, and abstention. It is a *single* stage at *edit* level and *in distribution*. There is no replay across environments, no ablation of a trait inside a multi-trait genome, no attribution under interactions and no out-of-scope probe. Error control is per comparison only, which the paper itself admits. The paper never calls its evidence "causal". |
| 5. Selective assimilation | **Partial** | Accept or abstain, edit by edit, into the persistent document. The selection is selective, but it goes into the live working artifact, not into a heritable genome distinct from lifetime state. There is no assimilation event between two stores. |
| 6. Multi-generational, frozen weights | **Partial (weak: frozen weights only)** | The paper uses a "frozen target model" / "frozen agent". It has no generations, descendants, lineages or inheritance. §5.3 mentions multi-agent extensions, but not descendant agents. |

**Implements full chain: no.** One element is full (and it comes from SkillOpt), three are partial (one of them weak) and two are absent. This matches the abstract-level row in ledger §2.10.14 (1 F · 2 – · 3 – · 4 P · 5 P · 6 P).

## Does the full text change the abstract-level assessment?

**Element ratings: unchanged. Threat to the novelty claim: down from MEDIUM to LOW. Relevance to EXP-0001 design: up.**

- **Why down.**
  - The full text confirms the scope is a single agent with a single document, an edit-level gate, and no inheritance or genome. The paper never treats acceptance as a prerequisite for heritability, only for the next incumbent.
  - The empirical evidence is thinner than the abstract suggests: one point estimate per cell; no seeds, CIs, set sizes or hyperparameter values; no code; a regression metric whose data source is unstated and which may be zero by construction; and a "verified in the ablation study" boundary claim with no matching table.
  - A reviewer could cite SAGE against "statistical held-out acceptance of learned skills" (already conceded in ledger §3 items 11–12). It cannot be cited against the conjunction: a somatic store, a typed genome, multi-stage trait-level evidence, and inheritance across generations.
- **Fair the other way.**
  - SAGE is the most statistically careful single-stage gate seen in the cluster. In some respects it is stronger than what EXP-0001 currently specifies:
    - an exact test on discordant pairs instead of an aggregate bootstrap;
    - an explicit argument for that choice based on correlation between items;
    - asymmetric costs;
    - a regression denominator relative to solved items;
    - abstention when there is no evidence.
  - Its ablation, with the optimizer fixed and only the gate varied (C1 +6.2, test +2.5), is direct, if noisy, evidence that **the acceptance rule alone moves final performance by several points**. That is exactly the confound arm B′ is meant to control.
  - If HeritAgent's arm C is compared only with naive or unrestricted inheritance, a SAGE-like gate is a ready explanation for any C > B gap.

## Consequences for HeritAgent

No design was changed. Items marked *requires ADR* are for the Builder/ADR owner.

1. **Arm B′ operationalization (e), SAGE-style. *Requires ADR (ADR-0002 amendment).***
   - Admit a candidate trait to the genome iff, on a shared held-out set and at the parent-genome incumbent, it satisfies Δ_λ > 0, p_λ < α and ρ ≤ τ.
   - Pre-register λ = 1, α = 0.05 and τ.
   - Use *no* replay across environments, controlled ablation in genome context, attribution or out-of-scope stage.
   - This is the strongest published single-stage performance gate. Choosing a naive aggregate gate for B′ would make B′ a strawman.
   - It complements ledger §5 item 1 (a)–(c) and RSEA option (d).
   - If budget allows only one B′, this one is the most defensible.
2. **Test statistic for CIG stages 2–4 on binary outcomes. *Requires ADR for any threshold or test change; reporting needs no ADR.***
   - EXP-0001 stage 2 uses a bootstrap CI lower bound > 0 and ≥ +0.05 on what appears to be an aggregate difference. Stage 3 in scope uses Δ ≥ 0, which admits ties.
   - SAGE's argument (paired, discordant-only, exact) applies directly whenever with/without runs share items and seeds.
   - Report (w, ℓ, m) per trait per stage in any case. This is the same data as SkillGen's n01/n10 and needs no ADR.
   - For stage 3, m = 0 → abstain is a principled replacement for "Δ ≥ 0", unless a non-inferiority margin is intended. That margin should then be pre-registered explicitly (cf. the RSEA note, item 2).
3. **Regression-stage denominator. *Requires ADR if changed.*** SAGE defines regression as ℓ / n_S, i.e. relative to items the incumbent already solves, not to all items. Check which denominator EXP-0001's τ_r uses. Without a stated denominator, regression rates are not comparable to SAGE's or GRASP's.
4. **Multiplicity and reuse of validation across candidates, stages and generations. *Requires ADR.***
   - SAGE explicitly leaves trajectory-level error uncontrolled.
   - CIG applies up to five stages to many candidate traits in every generation. Per-comparison α will inflate false assimilations over a lineage.
   - Options:
     - Holm or Benjamini–Hochberg (BH) correction across candidates in one generation;
     - fresh held-out items per generation;
     - SEA-style anytime-valid control (ledger §2.10.13), which matches SAGE's own §5.3 pointer to confidence sequences.
   - Pre-register one option.
5. **Power and held-out size. *Requires ADR for sizes.***
   - With λ = 1 and α = 0.05, at least m = 5 discordant pairs are needed even if all are wins (0.5^5 = 0.031).
   - With 20–40 episodes per candidate (ledger §5 item 1 budget) and modest effects, many candidates will yield fewer than 5 discordant pairs, so a SAGE-style gate, and CIG, will mostly abstain.
   - EXP-0001 should pre-register a minimum-detectable-effect / discordance calculation for each stage and report abstention rates per arm.
   - Abstention-driven conservatism is itself a confound for "smaller genome" (H-genome-size) claims. Separate "rejected on evidence" from "abstained for lack of evidence".
6. **Stochastic execution.** SAGE folds the agent's randomness into the probability space but uses one run per item. If EXP-0001 replays with several seeds per item, the pairs are not independent. Use item-level clustering (e.g. per-item majority) or a stratified exact test. This is a design note; the analysis plan needs an ADR.
7. **Wording and naming.** Do not present paired, regression-aware or abstaining acceptance as HeritAgent's contribution (ledger §3 items 11–12 already cover this). "SAGE" is a common ML acronym, so cite it as "SAGE (Wang et al., arXiv:2609.36043)" and never use the name for a HeritAgent component. When contrasting with it, say "edit-level acceptance into a single live skill document within one agent's optimization run".

## Suggested ledger update (§2.10.14; pending cross-review)

- Mark the entry deep-read (full text) and lower the threat from MEDIUM to LOW for the chain claim.
- Chain coverage is unchanged: 1 F (via SkillOpt) · 2 – · 3 – · 4 P (single-stage, edit-level, exact paired test, per-comparison error only) · 5 P · 6 P (frozen weights only).
- Evidence caveats: point estimates only; no seeds, CIs, set sizes, hyperparameters or code; regression-metric source unstated.
- Main value:
  - the B′ template (e);
  - the statistical design for CIG binary-outcome stages;
  - support for H2's premise that the gate matters.
