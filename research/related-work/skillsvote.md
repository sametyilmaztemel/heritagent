# SkillsVote: detailed notes

**Paper:** *SkillsVote: Lifecycle Governance of Agent Skills from Collection, Recommendation to Evolution*, by Hongyi Liu, Haoyan Yang, Tao Jiang, Bo Tang, Feiyu Xiong, Yuyu Luo, Zhiyu Li (MemTensor (Shanghai) Technology, Harbin Institute of Technology, Soochow University, HKUST(GZ); corresponding authors Yuyu Luo and Zhiyu Li). arXiv:2605.18401. v1 was submitted 18 May 2026; v2 was revised 14 Jun 2026. Preprint, no venue stated. The paper lists a website (skills.vote) and code at GitHub `MemTensor/skills-vote`. The arXiv abs page links no code, and I could not reach the repo from this environment (HTTP 403 via proxy), so it is **unverified**. Reviewed 2026-10-12 from the full arXiv HTML v2 (main text, appendices 6–12, attribution and evolution schemas, case studies).

## Mechanism

The paper governs a skill library through four lifecycle stages: collection, recommendation, attribution and evolution. Weights are frozen throughout. The solver is the Codex CLI on GPT-5.2 medium, GPT-5.4 mini medium and GPT-5.5 xhigh. No training happens anywhere.

- **Collection / profiling (preprocessing):** they crawled >1.68M GitHub SKILL.md packages. These went through Anthropic's skill format validator, exact-content dedup and a Claude Code agentic profiler (runtime, quality and verifiability profiles), leaving >290K categorized skills. A curated subset of about 10K skills comes from repos with >1k stars. For "verifiable" skills they synthesize Harbor tasks with an executable verifier. These tasks are used to *profile* skills (success rate, cost). **They are not used as a gate on library updates in the evolution loop.**
- **Recommendation (pre-task exposure control):** a separate agent searches the skill folders with filesystem tools and selects a small, complementary set. Only those skill dirs are copied into the solver's context, along with a short usage guide. If recommendation fails, it falls back to exposing all skills.
- **Attribution (post-task):** the *same solver session is resumed* (Codex resume) and asked to split its own trajectory into "subtasks". A subtask has one objective, one primary evaluation signal and at most one linked skill. For each subtask the session records:
  - the judge type: environment feedback, human, or none;
  - an attribution label from 11 categories. The success labels are `success_no_skill_seen`, `success_viewed_skill_but_not_used` and `success_skill_used_with_extra_exploration`. The failure labels are `fail_skill_issue`, `fail_agent_limit`, `fail_client_env`, `fail_external_env` and `fail_unknown_env`. The rest are uncertain labels;
  - line-level `skill_refs` (which spans of a skill were actually used);
  - a "reusable exploration" delta.

  Evidence: online mode gives only the verifier's test counts. Offline "oracle mode" can also show the gold solution and tests, with a prompt that forbids copying answers into skills.
- **Evidence-gated evolution:** a subtask is *admissible* only if it is labelled successful and has non-empty reusable exploration. Failed and uncertain subtasks are kept only as diagnostic records. Admissible units are *aggregated*: units supporting the same procedure are merged into one proposal. They are then *routed* to one of edit-existing-skill (`error_fix` / `knowledge_addition` / `prerequisite_addition`), `create_skill` or `skip`. Before an edit, the old skill version is backed up under the batch timestamp.
- **Two usage modes:**
  - *Online:* the library starts empty and evolves along a test-time task stream. Libraries are isolated per job; on SWE-Bench Pro each repo is its own job.
  - *Offline:* a library is evolved on 48 Terminal-Bench Pro tasks (batches of 4), then **frozen** and transferred to Terminal-Bench 2.0 for recommendation only.

## Results

- **Terminal-Bench 2.0 (avg@5, 89 tasks; 85 for GPT-5.5 after 4 cyber-blocked tasks):**
  - Online, versus no skills: +2.6 / +1.1 / +0.9 pp (GPT-5.2 / 5.4-mini / 5.5). Baseline for GPT-5.5 is 79.8, SkillsVote 80.7.
  - The online baselines regress on GPT-5.5: ReasoningBank −2.1, skill-creator −1.0.
  - Offline frozen TB-Pro library: +7.8 / +5.8 / +1.4 pp.
  - Curated 10K library: +3.7 / +4.0 / −1.7 pp.
  - Hard split, online: SkillsVote *drops* 6.7 pp on GPT-5.2 Hard and 2.1 pp on GPT-5.5 Hard.
- **SWE-Bench Pro public (avg@1, 731 tasks, single run):**
  - Online: +2.7 / +2.6 / +1.2 pp.
  - ReasoningBank and skill-creator regress on GPT-5.5 (−2.5 / −2.6).
  - Curated offline: −3.4 / +1.5 / −2.3.
- **Routing (SkillRouter hard pool, 79,141 skills):** with GPT-5.5, SkillsVote reaches FC@10 62.7 versus 46.7 for SkillRouter. The advantage depends on model capability: GPT-5.4 mini collapses at ≥50k skills.
- **Negative transfer:** exposing the whole online library directly gives net-negative task-level deltas on TB2 Hard, and recommendation removes the net negative. The exact numbers were lost in the HTML math rendering, so I am not quoting them.
- **Statistics:** no confidence intervals, seeds beyond avg@5, or significance tests are reported. Most online gains are 1–3 pp, and SWE-Bench Pro is a single rollout per task.
- **Ablation of the gate:** there is **no component ablation** of attribution. The closest is the skill-creator baseline, which keeps recommendation but replaces attribution-guided evolution with an ungated "turn the trajectory into a skill" prompt.
- **Limitations section:** none.

## Overlap vs HeritAgent chain

| Element | Coverage |
|---|---|
| 1. Trajectory-mined capabilities | **Full**: skills are created and edited from attributed trajectory subtasks, both online (test-time stream) and offline (historical trajectories). |
| 2. Somatic/germline separation | **Absent**: one live library is both the in-lifetime working store and the transferred artifact. The attributed-subtask records are a per-batch staging buffer before persistence, not a second store with a different heritability status. Admitted edits take effect in the same library the agent keeps using. |
| 3. Typed executable genome | **Partial**: skills are directory artifacts (SKILL.md plus scripts, references and assets) with applicability boundaries and prerequisites (a regulation-like field via `prerequisite_addition`), timestamped backups (primitive versioning), and typed JSON schemas for profiling, recommendation, attribution and evolution records. But the unit is a skill library, not a typed whole-agent functional genome (no policies or regulation types, no gene/locus structure, no version graph). |
| 4. Causal validation as prerequisite | **Partial**: there *is* an evidence gate before persistence, the strongest "gating-before-persistence" prior after LOGOS. But the evidence is **observational, LLM self-attribution**: the solver resumes its own session and labels responsibility from the trace plus verifier counts (or the gold solution in offline oracle mode). There is no intervention (no with/without-skill ablation per candidate), no held-out generalization check, no regression test before admission, and no statistical threshold. The paper's own wording is "attribution" and "responsibility assignment", never causal. The only interventional evidence (per-task deltas versus no skills, Fig. 5) is a post-hoc library-level analysis, not a gate. |
| 5. Selective assimilation | **Partial**: the admit / aggregate / route-to-edit-create-skip step is explicitly selective and conservative. But the target is a persistent skill library, not a heritable genome, and the criterion is "successful + reusable", not a measured effect size. |
| 6. Multi-generational, frozen weights | **Partial**: weights are frozen (all API models). Offline transfer is a single parent-library to fresh-agent hop (TB-Pro, then frozen, then TB2). There is no lineage, no repeated generations, and no descendant adaptation-cost measurement. Online libraries are isolated per job and never carried forward. |

**Full chain implemented:** no. Elements 2 and 6 (as generations) are missing, element 4 is observational rather than interventional, and element 3 is a skill library rather than a whole-agent genome.

## Threat assessment

Medium; the sweep's rating stands. It is the clearest **concurrent** instance of "evidence-gated admission into a persistent capability store, with outcome attribution, frozen weights". Its online result is also prior evidence that an attribution-gated library beats an ungated trajectory-to-skill baseline (skill-creator) at the strongest backbone. That result is small, but it is consistent across 6 pairs. Do not dismiss it as "just retrieval". Do not inflate it either: it has no germline, no generations and no interventional per-trait test, and its gains are in the 1–3 pp range with no CIs.

## Consequences for HeritAgent

1. **Must-not-claim additions (ledger §3):**
   - Evidence-gated admission of trajectory-derived skills into a persistent library.
   - Outcome/responsibility attribution of trajectory segments to skill vs agent vs environment vs evaluation signal.
   - Pre-task skill exposure control as a fix for negative transfer.
   - Trajectory-built frozen skill libraries transferring to an unseen benchmark.
   - The general finding that "gated skill persistence beats ungated persistence". SkillsVote shows a weak observational version of this.
2. **Sharpen the CIG differentiator in writing:** SkillsVote has an *observational, self-reported, trace-level* admission filter. CIG has an *interventional, trait-level* prerequisite: replay, then controlled ablation, then attribution, then held-out generalization, then regression protection, with thresholds and a bootstrap lower bound (ADR-0002), gating *germline* assimilation across generations. Cite it in Related Work next to LOGOS as concurrent gating-before-persistence work. Gloss LOGOS as governance/authorization and SkillsVote as LLM-judged outcome attribution.
3. **Experiment design implication (suggestion, not decided):** H2 frames arm B as "automatic inheritance of every *successful* skill". SkillsVote's admission rule is almost exactly "successful + reusable + self-attributed". A reviewer will ask whether CIG's interventions add value over a cheap observational gate. Consider either an extra arm B′ (SkillsVote-style self-attribution gate on the same frozen candidate set, as in ADR-0002 R4), or at least a secondary analysis comparing CIG verdicts with LLM self-attribution labels on the same candidates. This would show where they disagree, and whether disagreements predict T_eval / T_reg outcomes. Without it, the H2 contrast "C beats B" is open to the objection that most of the gain is achievable observationally.
4. **Borrowable ideas (cite if adopted):**
   - The subtask unit (one objective / one judge signal / at most one skill) as Trait Miner segmentation.
   - Line-level `skill_refs` as a cheap pre-filter for which genes to ablate.
   - The judge-signal typing (environment / human / none). This is a useful pre-filter so CIG does not spend replay budget on candidates with no objective signal.
   - The oracle-mode leakage guard. It is relevant to the T_eval contamination rule: SkillsVote lets gold solutions inform attribution offline, which HeritAgent's four-split rule forbids for gate/eval data.
5. **Negative evidence to engage:** gains shrink as the backbone gets stronger (GPT-5.5: +0.9 / +1.2 pp; curated library negative). Skills also hurt on Hard splits in some settings. Both bear on H4/H5 and on how big the H2 effect will be (ADR-0002's 5 pp threshold may be ambitious on strong backbones).
