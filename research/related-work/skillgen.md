# SkillGen — detailed notes

**Paper:** *SkillGen: Verified Inference-Time Agent Skill Synthesis* — Yuchen Ma (LMU Munich / MCML; eq. contrib.), Yue Huang (Notre Dame; eq. contrib.), Han Bao, Haomin Zhuang, Xiangliang Zhang (Notre Dame), Swadheen Shukla, Michel Galley (Microsoft Research), Stefan Feuerriegel (LMU Munich / MCML). arXiv:2605.10999 (cs.LG; cross-listed cs.AI, cs.MA), v1 9 May 2026, only version. No venue/comments field on the abstract page. Paper states code at github.com/yccm/SkillGen (repo not opened: proxy returned 403, so unverified). Read in full via arXiv HTML v1 + PDF text (main text, Appendix A–D). Reviewed 2026-10-12.

## Mechanism

- **Input:** a base agent, a skill-training pool split 70/30 into an *induction subset* and a *construction-time verification subset*, and a task-level evaluator (benchmark score / env check / LLM judge). **Output: one** auditable skill per (base model, benchmark) pair. Single skill, not a library.
- **Skill = inference-time intervention** s = (structured prompt, task metadata, optional executable scripts, optional reference docs), Anthropic-Agent-Skills style. The prompt body uses a fixed three-part schema: task context / reusable success patterns / failure-avoidance lessons. After round 1, refinement may only edit the natural-language body; the tool/script interface is frozen.
- **Stage 1, baseline elicitation:** roll out the base agent (no skill) on the induction subset to get success and failure trajectories; cache no-skill outcomes Y⁰ on the verification subset.
- **Stage 2, contrastive behavioral induction** (induction agent): a task summary; per-failure root-cause summaries, embedded and k-means clustered (≤8 clusters); the same for successes; and *local contrast*: for each failure, the nearest same-task-type success by embedding, with an LLM noting "behavior present in success / omitted in failure" (≤20 pairs).
- **Stage 3, generation → verification → refinement loop**, K = 8 rounds:
  - The verification agent loads candidate s and rolls out on every verification instance, comparing against cached Y⁰ on identical inputs (same instance id, seed 42).
  - **Potential-outcomes framing (explicitly Rubin 2005):** Δ(s) = E[Yˢ(x) − Y⁰(x)]. Repairs n₀₁ (0→1), regressions n₁₀ (1→0), net gain G_m(s) = n₀₁ − n₁₀. For a fixed, non-adaptively chosen skill, the estimator is unbiased (their caveat).
  - Feedback: instances are split into repaired / regressed / unresolved, and the agent writes keep/remove/add/emphasize edit instructions for the next round.
  - **Selection:** best-of-K by construction-time G_m.
  - **Verification gate:** the selected skill is `active` iff G_m ≥ max{g_abs, ⌈g_rel·m⌉, 1}, run with g_abs = 2 and g_rel = 0.05. Otherwise it is `deprecated` and replaced by the empty intervention. There are also up to 30 "baseline-success guard checks" to expose regressions on already-solved instances.
- **Deployment:** the active skill goes into a dedicated system-prompt slot. Reference docs load on demand (`skill_load_reference`), and scripts expose only declared `skill_`-prefixed functions. Base-model weights are never updated. The auxiliary roles (induction, generation, verification) are fixed to GPT-5.4-Mini for every base agent.

## Results (held-out paired rollouts, disjoint from the skill-training pool)

- Table 1 (8 base agents × 10 benchmark-splits = 80 cells):
  - Average gain is +3.27 to +10.08 pp per model; GPT-5.4-Nano gains +10.08 (54.27 → 64.35).
  - Per cell: 50 improve, 25 unchanged, 5 regress. ALFWorld improves in 14 of 16 entries; ScienceWorld improves for all 8 agents.
  - Benchmarks: ALFWorld IOD/OOD, LiveCodeBench, MCPBench, Mind2Web, PubMedQA, ScienceWorld, SocialMaze.
- **Baselines** (Trace2Skill, SkillX, EvoSkill, CoEvoSkills), adapted to a single-skill, no-tool interface on ALFWorld IOD/OOD and ScienceWorld with 3 models: SkillGen has the largest average gain. The authors caveat that this adaptation does not measure the baselines' native multi-skill capability.
- **Ablations** (ALFWorld OOD, ChemLLMBench yield): ICL instead of a skill, no refinement, **no verification gate**, no failure lessons, plain-text only. Full SkillGen wins every dataset-model pair.
- **Cross-model transfer:** a skill from a source model runs unchanged on an evaluator model. Of 120 off-diagonal pairs, 70% are non-negative and 42% exceed +5 pp. The best skill author is not the strongest model (Qwen-2.5-7B on ALFWorld).
- τ-Bench retail: +5.3 pp average on models whose skill passed the gate. ChemLLMBench yield: +16.1 pp average across 6 models.
- **Gate as selector (RQ6):** at round 8 the latest candidate's expected Δ is −3.1 pp, while the best verified candidate reaches +8.1 pp.
- Cost: about 5.6M tokens per skill on average (about $8.2 at GPT-5.4-Mini prices).

## Overlap vs HeritAgent chain

| Element | Coverage |
|---|---|
| 1. Trajectory-mined capabilities | **Full** (as a mechanism). Success/failure contrastive induction, with nearest-success local contrast. Overlaps SkillRL's differential distillation, and SkillGen is finer-grained (paired neighbor contrast). Acquisition is an offline batch over base-agent rollouts, not an agent's own lifetime, but the mechanism is fully covered. |
| 2. Somatic/germline separation | **Absent.** The candidate → `active`/`deprecated` status is a *deployment staging* label for one artifact within one construction run. There is no persistent somatic store, no germline registry, and no notion of what passes to descendants. |
| 3. Typed executable genome | **Absent** (one primitive). Each skill has a fixed typed schema (prompt / metadata / scripts / docs; context / success / failure body), with a frozen tool interface after round 1. But it is a single skill, not a whole-agent genome of policies + skills + regulation. There is no versioned lineage and no composition. |
| 4. Causal Inheritance Gate | **Partial: the strongest overlap of any paper so far on gate semantics.** It covers CIG stage 2 (controlled ablation: paired with/without on identical inputs, explicit potential-outcomes framing). It partly covers stage 4 (regression: n₁₀ is counted and fed back, plus guard checks). The gate is a hard accept/reject threshold. **Missing:** (a) regressions are *netted* against repairs (G = n₀₁ − n₁₀), not bounded separately; a skill can pass with many regressions, and their own App. C.9 reports an accepted skill with 51 repairs / 57 regressions on held-out. SPEC §10 has a separate RegressionRisk < τ_r. (b) No interaction testing or attribution among co-resident traits (single skill, so it doesn't arise). (c) The gate is decided on a 30% split of the *same* task pool; held-out/OOD evaluation is reporting only, not a gate stage. (d) No separate replay-reproducibility stage (one rollout per instance, seed 42). (e) What it gates is **deployment**, not heritability. |
| 5. Selective assimilation into heritable genome | **Partial.** There is selective, evidence-gated admission (active vs deprecated, with rejection to the empty intervention), but the target is a prompt slot for the current deployment, not a heritable genome. Nothing accumulates. |
| 6. Multi-generational, frozen weights | **Partial.** Frozen weights only (no parameter updates; skills transfer across 6 models). No generations or descendants. The 8 refinement rounds are iterations of one construction, and cross-model transfer is lateral reuse of a fixed skill, not inheritance. |

## Limitations (no explicit section; drawn from App. C.9 / D and the protocol)

- **Accepted skills can still regress on held-out data** (authors call this overgeneralization). For Llama-3.1-8B on ALFWorld OOD, the accepted skill drops 67.45% → 65.10% (51 repairs / 57 regressions). For Mistral-Nemo on ChemLLMBench yield, it drops 43.33% → 20.00% (3 repairs / 10 regressions).
- **Selection bias.** Best-of-K selection on the same verification split used by the gate means the reported G_m of the selected skill is optimistically biased. The unbiasedness statement holds only for non-adaptive skills (their own caveat). Verification sets are small in places (30-instance τ-Bench / ChemLLMBench construction pools).
- Single seed (42) and one paired rollout per instance. No significance tests on the Table 1 cells (95% bootstrap CIs only in the refinement-round figure).
- The baseline comparison uses forced single-skill, no-tool adaptations.
- The gate's evidence covers only the evaluated task distribution (Broader Impacts).

## Consequences for HeritAgent

- **Narrows the CIG novelty claim further.** Do not claim as ours: "skills as interventions with paired with/without evaluation on identical inputs", repair/regression (fixed vs broken) accounting, or a hard net-gain acceptance gate before a skill is used. SkillGen states these explicitly and calls itself "the first agentic framework that models inference-time skill synthesis as an intervention problem". Together with SkillForge, ASSAY, Skill-α and LOGOS, "causally validating a skill before use" is now prior art. CIG must be positioned only as (i) a gate on **heritability / germline assimilation** across generations, (ii) over a **multi-trait genome** with interaction/attribution stages, (iii) with **regression bounded separately** (τ_r) rather than netted, and (iv) with held-out generalization **inside** the gate.
- **Adopt and cite:** their repair/regression contingency table (n₀₁, n₁₀) as the reporting format for CIG stage 2 and stage 4. Their App. C.9 case (an accepted skill that is net-positive on verification but net-negative on held-out, with more regressions than repairs) is direct published evidence for keeping τ_r separate from τ_c. It also supports making generalization a gating stage rather than a reporting step. Cite it in SPEC §9/§10 motivation.
- **Methodological hazard to avoid:** best-of-K selection followed by a threshold on the same split, the winner's-curse pattern they caveat. CIG should decide promotion on evidence not used for candidate selection, or correct for it.
- **Their "gate off" ablation is a useful template for EXP-0001:** ungated vs gated skill admission measured as held-out Δ and regression counts. That is the single-generation version of our arm B vs gated-arm comparison. Our version adds generations, genome size, and descendant adaptation cost (H1–H6).
- Cross-model transfer (70% non-negative, best author ≠ strongest model) is relevant to H4 (cross-backbone genome expression). It is moderately positive evidence for frozen-weight skill portability, and it partly offsets SkillRL's "skills transfer poorly without cold-start SFT" negative result.
- **Threat level: medium** (agree with sweep). It does not implement the chain: no genome, no somatic/germline split, no generations. It is the closest published instance of CIG stage-2 semantics, and Related Work must name it explicitly next to ASSAY and SkillForge.

**Suggested ledger §3 addition:** "9. Paired with/without skill evaluation as an intervention, repair/regression counting, and net-gain acceptance gates before deploying a skill (SkillGen 2605.10999; also ASSAY, Skill-α, SkillForge)."
