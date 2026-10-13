# SAPO (Co-Evolving Skill Generation and Policy Optimization): detailed notes

**Paper:** *Co-Evolving Skill Generation and Policy Optimization*, by Zhiwei Zhang, Yudi Lin, Nikki Lijing Kuang, Linlin Wu, Xiaomin Li, Songtao Liu, Fenglong Ma. arXiv:2606.08755 (v1 7 Jun 2026, cs.CL, CC BY 4.0). The arXiv abs page lists no venue or comments. The HTML abstract gives a code link (github.com/zzwjames/skill_augmented_agent), which I did not open or check. Framework name: **SAPO** (Skill-Augmented Policy Optimization). Reviewed 2026-10-12 by an independent researcher (Claude), round-2 sweep hit "causal-skill-validation".

**Read depth:** full text, from the arXiv HTML read in two passes (chars 0–100k and 100k–end). I read it through a summarising fetch tool, so the equations below are as that tool transcribed them, not copied by hand from the PDF. I did not read the PDF or the code. The number of seeds and the variance of the headline numbers were not confirmed.

## Mechanism

- **Setting:** online RL for language agents with a skill bank (the SkillRL lineage). Backbone Qwen2.5-7B-Instruct, plus Qwen3-4B-Instruct-2507 in Appendix D.2. GRPO on verl, AdamW, lr 1e-6. **The policy weights are updated.**
- **Two-tier store:** a **temporary bank** ("candidates awaiting promotion") and a **long-term bank** (capacity 45 skills). Retrieval draws from both. The long-term bank is **initialized from SkillRL's bank**. The paper does not give its size or say who generated it.
- **Candidate induction (Eq. 2):** the *policy itself* writes one JSON skill per induction unit of K queries (default K=4; K∈{1,4,8} tested). The input is the queries, the retrieved skills, a skill-writing template, and the base rollouts. This removes the dependence on a proprietary teacher (o3/GPT-5.4 in SkillRL-style pipelines; Appendix E estimates about $30 per ALFWorld run for GPT-5.4 generation).
- **Prospective paired validation (Eqs. 1, 3, 4, 5):** the per-query rollout budget G is split into two matched halves on the same query and the same retrieved skills. One half uses the base skills only; the other uses the base skills plus the candidate ŝ.
  - Intra-prompt utility: u(x, ŝ) = mean reward(with ŝ) − mean reward(base).
  - Cross-prompt utility: U_ŝ = mean of u over the K queries.
  - In their words, the reward gap "estimates the candidate's marginal contribution beyond the retrieved skills". It is "prospective skill validation" done "without additional rollout overhead", because it re-splits the existing budget.
- **Promotion rule (Eq. 6):** applied every T_prom = 5. Appendix C says "epochs" and §3 says "training steps"; the paper is inconsistent. A temporary skill is promoted to the long-term bank only if all three of these hold:
  1. U > 0;
  2. it ranks in the top ρ of the temporary bank (ρ=0.2 for ALFWorld/WebShop, 0.1 for Search-QA);
  3. its maximum similarity to the long-term skills is below γ=0.8.

  All other temporary skills are **discarded**.
- **Generator training (Eq. 7):** an asymmetric, W-REINFORCE-style loss. Skills with positive utility are reinforced (weight λ∈[0,1]) and skills with negative utility are suppressed more strongly. A KL step (Eq. 9) distills the generator conditioned on rollouts into a reduced-input scorer.
- **Scoring, pruning and reranking (Eq. 8):** the score is the length-normalized log-likelihood of the skill under the policy. When the long-term bank is over capacity, the lowest-scoring skills are pruned. Retrieval is an embedding pool (6 for ALFWorld, 10 for WebShop) reranked greedily to top-k=3.
- **Skill format:** JSON `{skill_id, title (4–8 words), principle (3–5 sentences), when_to_apply (2–3 sentences)}`. These are prompt-injected natural-language skills. They are not executable, and there is no type system or versioning.

## Results

| Benchmark | SAPO | Baselines |
|---|---|---|
| ALFWorld, All success | **92.2** | SkillRL 89.9, Skill0 87.9, D2Skill 87.8, GRPO 79.3 |
| WebShop, score / success | **90.5 / 78.1** | SkillRL 85.2 / 72.7, D2Skill 83.4 / 73.4 |
| Search-QA, mean of 7 | **47.8** | SkillRL 45.5, Skill0 44.4, EvolveR 43.1, Search-R1 38.5 |
| Qwen3-4B on ALFWorld | 82.0 | 72.7 (D2Skill with o3) |

- Ablations (ALFWorld / WebShop score / WebShop success):
  - w/o validation: 90.6 / 83.0 / 75.0
  - w/o generator training: 90.2 / 83.2 / 73.4
  - w/o scoring: 91.4 / 86.5 / 76.6
  - full: 92.2 / 90.5 / 78.1

  The validation stage is worth +1.6 on ALFWorld and +7.5 on the WebShop score. Without seed variance I cannot say whether the ALFWorld gap is significant.
- Appendix D.1: Claude-generated skills have mixed, sometimes negative, utility. The authors say this shows that frontier-LLM skills are not reliably good.
- Appendix D.3: SkillRL often declines after its peak. The authors blame unfiltered skill evolution that lets low-quality or outdated skills in.

## Overlap vs HeritAgent chain

| Element | Coverage | Evidence |
|---|---|---|
| 1. Lifetime acquisition from trajectories | **Full** | Skills are induced by the policy from its own base rollouts during online training. |
| 2. Somatic/germline separation | **Partial (structural only)** | There is an explicit temporary bank, a gated promotion, a long-term bank, and discard of unpromoted candidates. That is a probation-store / persistent-store split with gated promotion. But the long-term bank is the agent's *own* persistent memory inside one training run, not a heritable store passed to descendants. There is no lifetime vs. inheritance distinction, and no notion of an individual agent's lifetime. |
| 3. Typed executable functional genome | **Absent** | The skills are a fixed JSON prompt record. `when_to_apply` is a primitive activation condition, as in SkillRL. There are no types, versions, policies or regulation layer, and the skills are not executable. |
| 4. Causal validation gate before persistence | **Partial (strong partial; the closest hit so far)** | This is an interventional with/without comparison on matched rollouts, done *before* admission, and U>0 is a necessary condition for promotion. That matches the *ablation* core of CIG stage 2 and is evaluated in context (marginal given the co-retrieved skills). It lacks: <br>• **Held-out generalization:** utility is measured only on the K induction queries, which are training prompts. <br>• **Regression protection:** no check that already-promoted skills or other tasks get worse. <br>• **Interaction/attribution analysis:** beyond the single marginal gap, there is none. <br>• **Variance control:** each half gets only G/2 rollouts, with no confidence threshold, so promotion is a top-ρ ranking of a point estimate. <br>• **A frozen measuring instrument:** the policy is changing between steps. <br>• It guards a within-run store, not heritability. |
| 5. Selective assimilation into heritable genome | **Partial** | Promotion is selective (positive, top-ρ, novelty), and later pruning is by likelihood score. But the destination is not heritable. Learned skill knowledge is *also* absorbed into the weights through GRPO and generator training, which is the opposite of keeping capability in an explicit artifact. |
| 6. Multi-generational, frozen weights | **Absent** | Weights are trained. There is one run per backbone and no generations or lineages. Seeding the bank from SkillRL is a one-off initialization, not a demonstrated inheritance mechanism. No frozen-policy or cross-agent evaluation of the bank was reported. |

## Limitations

**Stated:** there is no dedicated limitations section. Early in training, the 4B generator produces lower-utility skills than GPT-5.4 and only catches up later (§5.4).

**Observed:**
- The utility estimate is a difference of two G/2-rollout means, with no reported confidence threshold or variance analysis.
- Admission is confounded by policy drift. The policy that generates and scores the skills is also the one being trained on skill-augmented rollouts.
- Validation is only in-distribution (the induction queries).
- Pruning by policy likelihood is a self-consistency signal, not a re-test of utility. A skill can stop being useful without losing likelihood, and the reverse can also happen.
- Search-QA out-of-domain tests evaluate the whole system, not the transfer of individual skills.
- The T_prom unit is inconsistent between sections.

## Consequences for HeritAgent

- **The novelty position narrows.** "Checking with vs. without whether a candidate skill helps, before it is stored" is now prior art (SAPO; Skill-α does it per edit; ASSAY does it after the fact by masking). A two-tier temporary/long-term bank with utility-gated promotion is also prior art (SAPO, structurally; LOGOS, governance-flavoured). CIG can no longer be described as "the first pre-persistence causal check". What it can be positioned on:
  - the gate decides **heritability across generations** under **frozen weights**, not admission to a within-run store;
  - it is **multi-stage**: replay across environments, held-out generalization and regression protection, which SAPO lacks;
  - it writes into a **typed EAG**, not a JSON prompt bank.
- **Ledger §1 row 2 (somatic/germline): "zero direct hits" is now too strong.** Suggested wording: "no LLM-agent work separates lifetime-acquired from heritable state; probation→promotion skill stores exist within one run (SAPO 2606.08755; LOGOS 2607.10878)".
- **Ledger §1 row 4 (causal gate):** add SAPO as the closest *pre-persistence interventional* gate. It is within one lifetime, single-stage, in-distribution, and uses a moving policy.
- **Experimental implication (recommendation, not a SPEC change):**
  - A SAPO-style single-stage gate (paired with/without on the source queries, U>0, top-ρ) is a natural **"CIG stage-2-only" ablation arm**, run with frozen weights. If full CIG does not beat it on held-out performance, regression rate and descendant adaptation cost, CIG's extra stages are unjustified. This baseline is cheap to build because the paired-rollout estimator uses no extra budget.
  - SAPO's Appendix D.3 (SkillRL declines after its peak under unfiltered evolution) supports framing arm B (unrestricted inheritance) as the risk baseline.
  - The +1.6 ALFWorld gain from validation is a warning that gates may help only a little on saturated benchmarks. Power the EXP-0001 comparisons accordingly.
- **Writing:** gloss SAPO as "within-lifetime admission control by marginal utility" and CIG as "gated heritability". Do not call SAPO's test "causal" on its behalf, since the paper does not use that word. Its own terms are "prospective skill validation" and "marginal utility".

## Required architecture changes

None to the architecture. Ledger wording changes as listed above. Optionally add a SAPO-style stage-2-only gate as an EXP-0001 ablation arm.
