# Literature Sweep — Round 2 (2026-10-12, issue #19)

Author: Claude (third agent) — adjudication of sweeps run by Claude subagents. **Pending Research/Critic cross-review.** No repository file was modified; this report and the updated ledger live under the scratchpad `out/research/` tree until cross-reviewed.

Scope: works close to HeritAgent's six-element chain (SPEC §3; ledger §1): lifetime acquisition (L1), somatic/germline separation (S2), typed executable genome (T3), causal validation gate as prerequisite (C4), selective assimilation (A5), multi-generational inheritance with frozen weights (G6). Round 1 (2026-10-02) is in `adjacent-sweep.md` and ledger §2.1–2.9.

*Round-3 completion addendum (2026-10-13): see §6. Every medium/high-threat work has now been read in full; changed verdicts are tabulated in §6.2. Sections 0–5 are kept as the Round-2 record.*

## 0. Headline

- **49 new unique works** found and confirmed (47 on arXiv abstract pages, 2 on conference virtual pages; 1 further listing is unverifiable); **0 already in the ledger**.
- Threat levels: **1 high**, **17 medium**, **30 low**, **1 none** (methodology audit).
- **10 deep reads** (full text) of the high/medium set; **8 medium** works assessed at abstract level only (queued).
- **Verdict (a): no work implements the full six-element chain.** Closest: *Code Is the Body / OurArk* (2607.28691) — G6 full; L1, S2, T3, A5 partial; C4 absent.
- Three chain elements move from "open" to "partially open" (S2, T3, A5); C4 is crowded at the within-lifetime level. Claim position retained with narrowed wording (ledger §1); 20 new must-not-claim items (ledger §3.9–3.28); 21 round-1 citations audited, 8 corrected (ledger §4); 10 experiment-design implications flagged as *requires ADR* (ledger §5).

## 1. Method

Four query families were run by independent subagents (bio-heredity, causal-skill-validation, generational-evolution, promotion-gates), each capped at ~15 reported papers, followed by deduplication (by arXiv id / title), deep reads of the medium/high set as budget allowed, and a citation-accuracy check of every reference in the round-1 ledger. Hard rules: no paper is reported unless its abstract page (or conference page) was opened; no arXiv ids, authors or numbers invented; unverifiable items are labelled as such.

### 1.1 Query families and queries (96 total)

**bio-heredity (25):**
1. Baldwin effect genetic assimilation LLM agents skills arXiv 2026
2. agent genome heritable skills offspring agents inherit validated capabilities frozen weights arXiv
3. somatic germline separation LLM agent memory heritable vs episodic
4. Lamarckian inheritance LLM agents evolution acquired skills passed to offspring 2026
5. "genetic assimilation" language model agents prompt skills evolution generations
6. "agent DNA" OR "agent genome" typed genome LLM agents executable genome inheritance arXiv 2026
7. epigenetic regulation gene expression LLM agent skills activation heritable genome ICLR 2027 submission
8. multi-generational LLM agents inherit skill library offspring validation before inheritance frozen model
9. genotype-phenotype mapping LLM agent genome evolution prompts tools memory germline arXiv 2026
10. openreview ICLR 2027 submission agent evolution inheritance heritable skills genome Baldwin
11. Darwinian versus Lamarckian evolution self-improving LLM coding agents archive inherit learned experience
12. germline LLM agent germline skills heritable soma agent architecture
13. arxiv 2609 evolving agents offspring inherit skills parents causal ablation validated before inheritance
14. self-replicating LLM agents lineage genome skills mutation crossover population skill genome
15. Baldwinian learning evolution LLM agent population in-context learning internalization genotype 2026
16. heritable LLM agent memory next generation agents genes persist skill transmission
17. Artemis agent genome prompts tool wiring crossover mutation LLM agents
18. genetic assimilation OR canalization OR epigenetic inheritance agent skills internalization September 2026
19. Semantic Scholar agents inherit capabilities across generations skill inheritance lineage regression test gate
20. heredity LLM agents breeding offspring Mendelian follow-up Genomebook
21. arxiv 2026 agent genome germline self-evolving LLM frozen next-generation skills inherited descendants held-out
22. Morphogenetic Harness developmental biology LLM agents
23. evolution of skills agents across generations cultural vs genetic inheritance arXiv 2026
24. Baldwin effect large language model agents evolution learning guides evolution 2026
25. self-evolving agent lineage skill library inheritance child agent ablation causal held-out generational frozen LLM September 2026

**causal-skill-validation (23):**
1. arXiv 2026 agent skill verification counterfactual ablation harmful skills pruning
2. arXiv 2026 per-skill Shapley value skill library quality LLM agent
3. agent memory validation before persistence causal utility memory admission gate LLM agents arXiv 2026
4. Agent Skills benchmark ecosystem safety SKILL.md empirical study arXiv 2026
5. ASSAY 'Not All Skills Help' measuring and repairing agent skill libraries
6. skill library inheritance across agent generations validated skills before consolidation held-out gate arXiv 2026
7. ICLR 2027 submission OpenReview agent skills causal ablation skill curation
8. SkillForge evidence-based skill verification refinement arXiv 2608.24747 follow-up
9. skills hurt LLM agents regression tax decomposing why skills help and hurt
10. self-evolving agent experience memory contamination harmful experience filtering counterfactual utility validation
11. Agent Skills Can Be Harmful 2608.11888 follow-ups (search)
12. Skill-alpha progressive skill generation rollback reward follow-up
13. openreview.net ICLR 2027 skill library counterfactual masking (domain-restricted)
14. skill repair pruning faulty skills arXiv September 2026
15. heritable agent skills lineage generations Lamarckian parent child verification gate
16. agent genome skills policies versioned genome successor agents germline arXiv 2026
17. leave-one-out memory entry contribution counterfactual utility agent memory arXiv 2026
18. EvoSkill OR Trace2Skill OR Skill-Pro validation gate held-out promotion frozen model
19. skill ablation randomized masking skill contribution estimator library curation September 2026
20. Causal Memory Intervention CMI agent memory controlled interventions
21. SkillCenter paired baseline placebo oracle arms
22. Semantic Scholar citations of ASSAY follow-up October 2026
23. skills transfer across model upgrades obsolete skills retire skill utility decay

**generational-evolution (27):**
1. Darwin Godel Machine follow-up self-improving coding agent archive arXiv 2026
2. open-ended evolution of LLM agents lineage archive generations arXiv 2026
3. cross-model transfer of evolved agent scaffolds skill library different backbone LLM arXiv 2026
4. agent offspring inherit memory skills from parent agent generations arXiv 2026
5. population-based search agent scaffold evolution AlphaEvolve agent harness arXiv 2026
6. OpenReview ICLR 2027 submission self-evolving agent skill inheritance generations
7. Lamarckian inheritance LLM agents multi-generation validated skills heritable gate
8. self-referential agent improvement archive stepping stones Godel agent 2026
9. evolved scaffold transfers to other foundation models frozen weights DGM Huxley Godel Machine
10. agent evolution lineage regression ablation of inherited modifications attribution
11. multi-generational agent benchmark skill accumulation across generations cultural evolution
12. Huxley-Godel Machine clade metaproductivity
13. HyperAgents self-referential self-improving agents Meta
14. germline somatic LLM agent heritable skills frozen weights counterfactual ablation before inheritance
15. openreview ICLR 2027 self-evolving/self-improving agent archive lineage population
16. arXiv September 2026 self-improving coding agent open-ended evolution archive
17. agent genome evolution LLM executable genome typed genes crossover mutation
18. LLM agents reproduction breeding offspring inherit learned skills generations
19. Baldwin effect LLM agent skills genetic assimilation arXiv 2026
20. DarwinX / evolutionary selection LLM agents frozen model population
21. agent harness evolution transfer across backbones portability benchmark 2026
22. NeurIPS 2026 accepted self-evolving agents skill library transfer across generations
23. inheritance of memory across agent generations negative transfer harmful inherited memory
24. multi-generation self-improving agent inherits archive frozen FM held-out regression descendants adaptation cost
25. AgentSpawn skill inheritance follow-up spawned agents memory transfer validation
26. GRASP regression-aware skill proposal
27. Hierarchical Self-Improvement frozen LLM archive harness configurations

**promotion-gates (21):**
1. arXiv 2026 promotion gate learned skills held-out validation before persisting self-evolving agent
2. LOGOS living logic AI agent teams evolve with humans agent packs release candidates promotion
3. regression testing of agent updates continual learning tool-using agents without regression skill library arXiv 2026
4. safe self-modification LLM agent verification gate before accepting self-edit arXiv 2026
5. EvolveR experience-driven self-evolving agents distillation principles
6. pre-registered evaluation self-improving agents protocol held-out contamination 2026
7. SEAGym evaluating self-evolving agents update validation held-out transfer arXiv
8. ICLR 2027 submission self-evolving agent skills ablation causal contribution gate openreview
9. heritable skills across agent generations frozen LLM validated inheritance lineage skill library parent child arXiv 2026
10. counterfactual ablation of learned skills before consolidation into long-term memory agent germline OR Lamarckian OR genome LLM agent 2026
11. MUSE self-evolving agent experience skill validation arXiv 2026 follow-up verified skill admission
12. attribution as governance agent updates which edit caused improvement credit assignment self-improving agent audit trail
13. leave-one-out ablation skill admission causal effect learned skill persisted only if held-out improvement multi-generation agent evolution frozen weights
14. agent genome typed versioned inheritable skills policies offspring agents validated before inheritance LLM 2026 preprint
15. openreview ICLR 2027 self-evolving agent regression gate skills library submission
16. Godel agent self-improving lineage inheritance Baldwin effect LLM agents learned behaviors assimilated into heritable code 2026 arXiv
17. continual learning tool-using agents catastrophic forgetting skills non-regression guarantee sequential tasks arXiv September 2026
18. CASCADE cumulative agentic skill creation follow-up 2026 skill validation consolidation sharing agents
19. next generation agent inherits validated skills from predecessor agent skill library handoff successor agent evaluation frozen LLM 2026
20. RELAI Verifiable Continual Learning agents regression control optimization
21. agent self-evolution evolutionary transmission Lamarckian learned skills offspring LLM agents population generations arXiv 2026

### 1.2 Sources checked

- **arXiv abstract pages** (WebFetch) for every reported paper — mandatory before reporting; ids, titles, authors and dates confirmed there.
- **arXiv full text**: PDF for 2609.18560 (Gilestro; also PDF metadata/outline for 2607.04240), 2608.05810 (VaG v2), 2608.07545 (DarwinX v1), 2609.30861 (SkillEvoReg), 2606.08755 (SAPO), 2605.18401 (SkillsVote), 2605.10999 (SkillGen), 2603.02766 (EvoSkill), 2605.29668 (GRASP); arXiv HTML for 2607.28691 (Code Is the Body).
- **Conference pages**: icml.cc/virtual/2026/67403 (Morphogenetic Harness, ICML 2026 SCALE workshop); neurips.cc/virtual/2026/168230 (Gate Presence, NeurIPS 2026 Agents-in-the-Wild workshop).
- **Aggregators via WebSearch**: alphaxiv, emergentmind, papers.cool, huggingface papers, hyper.ai mirrors, awesomepapers — used for discovery only; nothing reported from a snippet alone.
- **ACL Anthology / doi.org** for the CapaBench venue and Genomebook DOI (citation check).
- **Repo files (read-only)**: `research/novelty-ledger.md`, `research/related-work/adjacent-sweep.md`, `research/related-work/skillrl.md`, `SPEC.md` §3/§9/§15–17/§30/§34, `experiments/manifests/EXP-0001-design.md`, `research/decisions/ADR-0002-exp-0001-design.md`.
- **Not reachable / not queryable**: OpenReview ICLR 2027 submissions (domain-restricted search returned no relevant indexed hits — no ICLR 2027 paper is reported); NeurIPS 2026 accepted list; Semantic Scholar and Google Scholar (no direct fetch); bioRxiv (HTTP 429 twice; Genomebook unverified); github.com/yccm/SkillGen (proxy 403).
- Zcode/Codex agent-communication tracking was not part of this literature subtask.

## 2. Deduplicated paper table (49 works)

Chain tags are the *sweep-level* tags (abstract-based) unless a deep read refined them (ledger §2.10 carries the refined F/P/– vectors). Depth: **full** = full text read, note in `related-work/`; **abs** = abstract page only; **page** = conference page only.

| # | Work | id | Date | Venue | Threat | Chain tags | Found by | Depth | Why it matters (one line) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Code Is the Body: Agent-Owned Software Bodies for Recursive Evolution and Descent (OurArk) | 2607.28691 | 2026-07-30 | arXiv cs.SE | **high** | S2 T3 A5 G6 | generational | full | Inheritable body vs non-inherited private state; validation-before-birth; multi-generation descent with frozen weights; no causal gate, untyped |
| 2 | When Self-Evolution Backfires: Pre-Commit Gating against Skill Contamination (VaG) | 2608.05810 | 2026-08-06 (v2 09-17) | arXiv cs.AI | medium (med-high for C4/H2/H5) | L1 A5 (+C4 partial on deep read) | bio, gates | full | Add-one held-out A/B + marginal-gain gate before skills persist; contamination-lineage argument; cross-backbone pool transfer |
| 3 | GRASP: Gated Regression-Aware Skill Proposer | 2605.29668 | 2026-05-28 (v3 08-21) | EMNLP 2026 Main (per arXiv) | medium | L1 C4 A5 | generational, gates | full | Held-out net-gain + hard regression budget; gate not writing drives gains; asymmetric interface-dependent transfer |
| 4 | DarwinX: Evolving Agent Harnesses Through Natural Selection | 2608.07545 | 2026-07-31 | arXiv cs.NE | medium | A5 G6 | bio, generational | full | Preserve-and-extend bounded-regression admission on heritable harness lineages; frozen weights; archive; proxy overfits held-out by 31.7 |
| 5 | The evolution of sex for artificial intelligence (Gilestro) | 2609.18560 | 2026-09-16 | arXiv cs.LG/cs.NE/q-bio.PE | medium | L1 G6 | bio | full | Lamarckian multigenerational acquired-skill inheritance on frozen base via LoRA; declinable merge gate; population-genetics vocabulary |
| 6 | Co-Evolving Skill Generation and Policy Optimization (SAPO) | 2606.08755 | 2026-06-07 | arXiv cs.CL | medium | L1 C4 A5 | causal | full | Paired with/without rollouts before storage; temporary→long-term bank; weights trained |
| 7 | SkillGen: Verified Inference-Time Agent Skill Synthesis | 2605.10999 | 2026-05-09 | arXiv cs.LG | medium | L1 C4 | causal | full | Skills as interventions (potential outcomes), n01/n10 repair/regression accounting, hard net-gain gate; netting admits regressing skills |
| 8 | SkillsVote: Lifecycle Governance of Agent Skills | 2605.18401 | 2026-05-18 (v2 06-14) | arXiv | medium | L1 C4 A5 | causal, gates | full | Evidence-gated admission with LLM self-attribution of trajectory segments; observational, not interventional |
| 9 | EvoSkill: Automated Skill Discovery for Multi-Agent Systems | 2603.02766 | 2026-03-03 | arXiv cs.AI | medium | L1 A5 G6 | causal | full | Frozen model; held-out-gated top-k frontier; git-backed lineage; frontier-relative admission can accept regressions vs parent |
| 10 | SkillEvoReg: Regularizing Agent Skill Evolution Against Overfitting | 2609.30861 | 2026-09-25 | arXiv | medium | L1 C4 | bio, gates | full | "Causal Counterexample Validation" per update (detect-and-repair, never rejects); skill dropout; growth regularization |
| 11 | Hypothesis-Driven Skill Optimization (HDSO) | 2606.22330 | 2026-06-21 | arXiv cs.AI/cs.SE | medium | L1 C4 A5 | generational | abs | Paired control/treatment before admission to approved repository; frozen curator/executor |
| 12 | Agentic Harness Engineering | 2604.25850 | 2026-04-28 (v4 05-18) | arXiv cs.CL | medium | T3 C4 G6 | generational | abs | Component-typed revertible harness with per-edit predictions; structural components transfer across models, prose does not |
| 13 | Self-Evolving Agents with Anytime-Valid Certificates (SEA) | 2607.00871 | 2026-07-01 | arXiv cs.AI | medium (read first) | T3 C4 A5 G6 | gates | abs | Versioned harness + adapter around frozen base; anytime-valid certified gate; single agent |
| 14 | SAGE: A Statistical Acceptance Gate for Self-Evolving Agents | 2609.36043 | 2026-09-28 | arXiv cs.AI | medium | L1 C4 A5 | gates | abs | Per-item paired test with abstain; optimizer's curse; regression rate →0 in 19/20 |
| 15 | Recursive Self-Evolving Agents via Held-Out Selection (RSEA) | 2606.28374 | 2026-06-17 | arXiv cs.AI | medium | L1 T3 A5 G6 | gates | abs | "Generations" + held-out non-inferiority + layered NL state; terminology collision |
| 16 | SkillBoost (Rethinking Self-Evolution: Constrained Exploration-Exploitation) | 2607.26643 | 2026-07-29 (v2 08-19) | arXiv cs.AI | medium | L1 C4 A5 | gates | abs | Component-level failure localization + regression-limited verified acceptance |
| 17 | HarnessEvolve | 2609.00829 | 2026-09-01 | arXiv cs.LG | medium | L1 C4 A5 | gates | abs | Quality gate + performance non-regression gate + held-out snapshot selection; reference trajectories from ground truth |
| 18 | Self-Healing Harness for Runtime Oversight of Agent Self-Modification | 2609.24130 | 2026-09-21 | arXiv | medium | L1 C4 A5 | gates | abs | External replay/regression admission gate; 55% of rejected proposals fixed target but broke a working case |
| 19 | HarnessBank: Semantic Gene-Bank Search with Gated Verification | 2607.13683 | 2026-07-15 (v2 07-30) | arXiv cs.CL | low | A5 | bio, generational | abs | "Gene bank"/"offspring" naming; gate is an efficiency filter |
| 20 | Machine Zygote: Causal Biparental Heredity in a Germline–Soma Artificial Agent | 2609.17300 | 2026-09-15 | arXiv cs.NE/cs.RO | low | S2 | bio | abs | Germline–soma + interventional heredity test, non-LLM, one generation; terminology hazard |
| 21 | FlowEvo | 2607.21596 | 2026-04-18 (v2 08-20) | arXiv | low | L1 A5 | bio | abs | Executable skills bank with utility-tracked suppression of negative-transfer skills |
| 22 | SkillMAS | 2605.09341 | 2026-05-10 (v2 05-16) | arXiv | low | L1 A5 | bio | abs | Credit from traces; evidence-gated MAS restructuring |
| 23 | MetaSkill-Evolve | 2607.05297 | 2026-07-06 | arXiv cs.AI | low | L1 G6 | bio | abs | Two-timescale task-skill/meta-skill on frozen backbone (MGM group) |
| 24 | The Geno-Synthetic Algorithm | 2605.13365 | 2026-05-13 | arXiv cs.NE | low | T3 | bio | abs | Typed heterogeneous genotypes + assembly operator (EC) |
| 25 | Biological Motifs for Agentic Control | 2607.04240 | 2026-07 (exact date unconfirmed) | arXiv | low | T3 | bio | abs (+PDF outline) | GRN/epigenetic vocabulary for agent control |
| 26 | Morphogenetic Harness | — (icml.cc/virtual/2026/67403) | 2026-07-10 | ICML 2026 SCALE ws poster | low | T3 | bio, generational | page | Colony evolution gives nothing beyond pass@k — control for population effects |
| 27 | Genes in Intelligent Agents (Learngene) | 2306.10225 | 2023-06-17 (v2 10-27) | arXiv (AAAI listing seen, not opened) | low | G6 | bio | abs | Original "heritable genes in agents" + Lamarckian claim (RL weights) |
| 28 | ARTEMIS: Evolving Excellence | 2512.09108 | 2025-12-09 (v2 2026-09-03) | arXiv | low | T3 | bio | abs | Agent configuration as evolvable genome with semantic operators |
| 29 | When Child Inherits: Subagent Spawn in Multi-Agent Networks | 2605.08460 | 2026-05-08 | arXiv | low | — | bio, generational | abs | Memory inheritance as attack surface — motivation for a selective gate |
| 30 | Safety in Self-Evolving LLM Agent Systems | 2606.23075 | 2026-06-22 | arXiv cs.CR (survey) | low | L1 G6 | causal, generational | abs (+partial body) | Lamarckian/lineage-persistent threat framing; authors unverifiable |
| 31 | What Is a Skill Worth? Structure-Aware Shapley (SkillSV) | 2608.04562 | 2026-08-05 | arXiv cs.AI | low | C4 | causal | abs | Dependency-aware coalitions + length-neutral padding for intra-skill attribution |
| 32 | WikiSkill | 2608.27454 | 2026-08-27 | arXiv | low | L1 A5 | causal | abs | Experience→knowledge→skills layering; gate claim secondhand |
| 33 | SkillOpt | 2605.23904 | 2026-05-22 | arXiv | low | L1 A5 | causal, gates | abs | Strict-improvement held-out gate per edit, frozen model |
| 34 | MeClear | 2609.09115 | 2026-09-08 | arXiv | low | C4 | causal | abs | LOO + Shapley memory attribution, query-time suppression |
| 35 | Causal Intervention-Based Memory Selection (CMI) | 2605.17641 | 2026-05-17 | arXiv cs.AI | low | C4 | causal | abs | "Causal" interventional memory selection at retrieval time |
| 36 | The Regression Tax | 2607.22520 | 2026-07-24 | arXiv cs.AI | low | — | causal | abs | Gains vs regressions; "osmosis" confound for masking ablations |
| 37 | Strategy-Level RSI / SkillApt | 2609.26863 | 2026-09-22 (v3 10-09) | arXiv | low | C4 | causal | abs | Paired evidence for LOAD/ABSTAIN activation; beats successes-only history |
| 38 | Practice Makes Unsafe: Skill Misevolution | 2608.12851 | 2026-08-13 (v2 10-07) | arXiv cs.AI | low | L1 A5 | causal | abs | Write-time + retrieval-time safety check on skills |
| 39 | Safe Skill Retirement for Physical Agents | 2609.29543 | 2026-09 (page date inconsistent — unverified) | arXiv cs.AI | low | C4 | causal | abs | Two-gate counterfactual retirement certificate; task-only gates pass unsafe changes |
| 40 | Counterfactual Trace Auditing of LLM Agent Skills | 2605.11946 | 2026-05-12 (v2 05-28) | arXiv | low | C4 | causal | abs | Trace-level effects hidden by aggregate pass rate |
| 41 | Huxley-Gödel Machine | 2510.21614 | 2025-10-24 (v3 10-29) | ICLR 2026 (per ICLR site) | low | G6 | generational | abs | Clade-metaproductivity selection; DGM follow-up missed in round 1 |
| 42 | Hyperagents | 2603.19461 | 2026-03-19 | arXiv cs.AI | low | G6 | generational | abs | Self-referential DGM successor; meta-level accumulation across runs |
| 43 | MemCollab | 2603.23234 | 2026-03-24 (v2 05-28) | arXiv cs.AI | low | L1 | generational | abs | Naive cross-backbone memory transfer can hurt — H4 risk |
| 44 | EvoAgentBench | 2607.05202 | 2026-07-06 | arXiv cs.AI | low | L1 | generational | abs | Ability-transfer benchmark; automatic self-evolution brittle |
| 45 | Rethinking Self-Evolving Agent Skills: Feedback Dynamics | 2608.02636 | 2026-07-31 | arXiv cs.SE | low | L1 A5 | generational | abs | 55/388 candidates helped; validation-filtered sparse search |
| 46 | Ratchet: How Reliable Must an LLM Judge Be to Retire a Skill? | 2605.22148 | 2026-05-21 (v3 08-07) | arXiv | low | L1 C4 A5 | gates | abs | Judge-error bound for contribution-based retirement; LLM-written skills +0.0 |
| 47 | Falsifiable Release Gates for Self-Improving Systems | 2607.13070 | 2026-07-11 (v2 07-21) | arXiv | low | C4 A5 | gates | abs | Pre-declared acceptance suites + standing invariants |
| 48 | Gate Presence Is Not a Safety Invariant | — (neurips.cc/virtual/2026/168230; OpenReview Rbq78da3LI) | 2026 | NeurIPS 2026 Agents-in-the-Wild ws poster | low | C4 A5 | gates | page | Editable gates get semantically weakened — keep CIG logic outside the genome |
| 49 | Most Self-Improving LLM-Agent Papers Fail Two or More Evaluation Traps (methods audit) | **unverifiable** (alphaxiv listing, non-standard id) | 2026-09-25 | preprint | none | — | gates | page | Five-trap reviewer checklist; cite only after canonical id is found |

Counts by sweep-level chain tag (49 works; a work may carry several): L1 28 · S2 2 · T3 9 · C4 24 · A5 23 · G6 14 · none 4. Only two works carry S2 and only one of them is an LLM-agent work (OurArk, by functional analogy; the paper does not use the terms).

## 3. Adjudication

### 3.1 (a) Does any work implement the full chain?

**No.** Refined coverage after deep reads (F = full, P = partial, – = absent; order L1 S2 T3 C4 A5 G6):

| Work | L1 | S2 | T3 | C4 | A5 | G6 | Full chain? |
|---|---|---|---|---|---|---|---|
| OurArk 2607.28691 | P | P | P | – | P | F | no — C4 absent (test suite + human merge) |
| VaG 2608.05810 | F | – | P | P | P | P | no — within one agent; no soma; untyped |
| GRASP 2605.29668 | F | – | P | P | P | P | no — one lifetime library |
| DarwinX 2608.07545 | P | – | P | P | P | F | no — variant-level gate; no soma; no trait schema |
| Gilestro 2609.18560 | P | – | – | P | P | P | no — Lamarckian adapters; gate only on merges |
| SAPO 2606.08755 | F | P | – | P | P | – | no — weights trained; one run |
| SkillGen 2605.10999 | F | – | – | P | P | P | no — one skill, deployment gate |
| SkillsVote 2605.18401 | F | – | P | P | P | P | no — observational gate; one library |
| EvoSkill 2603.02766 | F | – | P | P | P | F | no — scalar frontier gate; one search run |
| SkillEvoReg 2609.30861 | P | – | – | P | – | P | no — detect-and-repair; no inheritance |
| SEA 2607.00871 (abs) | – | – | P | P | P | P | no (abstract) — single agent; re-check in full |
| RSEA 2606.28374 (abs) | F | – | P | P | P | P | no (abstract) — rewrite iterations, NL state |

Element C4 at the *trait level with a held-out and regression stage as a prerequisite for heritability across agent generations* is absent in every work; S2 as a *gated staging store of lifetime traits* is absent in every work; T3 as a *typed schema of policies+skills+regulation with per-trait validation records* is absent in every work. The claim position therefore does not flip; it is narrowed to the conjunction (ledger §1).

### 3.2 (b) Entries added to ledger §2

Full SPEC §30 format for the 18 high/medium works (ledger §2.10.1–2.10.18); short entries for the 30 low-threat works grouped by theme (ledger §2.11); a Round-3 queue (ledger §2.12).

### 3.3 (c) Additions to §3 "must NOT claim"

Twenty items (ledger §3.9–3.28), consolidated from the ten deep-read `must_not_claim_additions` lists plus abstract-level findings. The most consequential: pre-persistence gating per se (item 9); paired with/without admission tests, skills-as-interventions, repair/regression accounting (10); regression budgets and preservation probes (11); inheritable-artifact vs private-state boundary and validation-before-birth (18); biology-framed Lamarckian multigenerational inheritance on a frozen base (20); regression-protected admission as a prerequisite for ancestry (21).

### 3.4 (d) Wording changes in §1

- Table rows for S2, T3, A5 changed from "Open" to "Partially open" with named analogues; row C4 changed to "Crowded within one lifetime; open only at the heritability level"; rows L1 and G6 extended.
- Round-1 phrase "zero direct hits" for S2 withdrawn.
- New "Round-2 narrowing of the claim" paragraph and "Round-2 verdict" paragraph; confidence now "moderate — two structured sweeps (2026-10-02, 2026-10-12)" with explicit coverage gaps.
- "What remains ours to defend" rewritten as the conjunction (i)–(iv), adding performance-gated inheritance as a required comparator.
- Anti-overclaim addendum on the word "causal" (SkillEvoReg, CMI vs SAPO, GRASP usage).
- Five new naming hazards (gene bank; germline–soma; generations; learngene; CMI/CCV names) and a CapaBench title correction.

### 3.5 (e) Corrections from the citation check

21 references checked; 13 verified, 7 mismatches (all minor-to-moderate), 2 unverifiable (see table below and ledger §4). No round-1 arXiv id was wrong. Corrections applied inline in ledger §2.1–2.9 and summarized in ledger §4.

| Reference | Result | Correction |
|---|---|---|
| AgentSquare 2410.06153 | mismatch | equal-contribution marks removed; ICLR 2025 venue unverified from arXiv (confirm on OpenReview) |
| DGM 2505.22954 | verified | — |
| MGM 2608.07645 | verified (abstract has no numbers) | body-level figures flagged as not re-verified |
| Genomebook (bioRxiv DOI) | unverifiable (429) | all details flagged as unverified in R2 |
| SkillRL 2602.08234 | verified body; venue partial | WebShop 85.2 is score, success is 72.7%; NeurIPS 2026 only via arXiv Comments |
| ShapleyFlow 2010.14592 | verified | — |
| CapaBench 2502.00510 | mismatch | ACL title adopted; arXiv title differs; "Who's the MVP"/v3 rename unverified; 24 vs 16 coalition inconsistency |
| SkillShapley 2608.13173 | verified | authors completed |
| SkillForge 2608.24747 | mismatch (no title) | title added (RL agents) |
| Agent Skills Can Be Harmful 2608.11888 | verified | numbers added |
| ASSAY 2606.15390 | mismatch | subtitle "…Agent Knowledge"; randomized masking unconfirmed from abstract |
| Skill-α 2608.01678 | mismatch (wording) | "causal per-edit gate" → "counterfactual per-edit scoring" |
| AgentSpawn 2602.07072 | mismatch (scope) | full title, sole author, code-generation scope |
| LOGOS 2607.10878 | verified | "held-out execution evidence" marked as paraphrase |
| MUSE-Autoskill 2605.27366 | verified | — |
| CASCADE 2512.23880 | verified | — |
| Voyager 2305.16291 | verified | GPT-4/self-verification not re-checked |
| OMEGA/RECLAIM 2605.25062 | mismatch | title/author added; Baldwin section unconfirmed from abstract |
| Survey 2507.21046 | verified | full title |
| D2Skill 2603.28716 (skillrl.md) | unverifiable | characterization flagged |
| ACL 2026 Findings DOI | verified | — |

### 3.6 (f) EXP-0001 / baseline implications — flagged, not changed

All are *requires ADR* (ledger §5). The single most important: a **performance-gated arm B′** (GRASP/VaG/SAPO/EvoSkill-style held-out or paired gate without trait-level stages) between arm B and arm C — otherwise C > B is attributable to "any held-out gate". Secondary: acceptance-rate-matched random control (Gilestro); observational-attribution comparison (SkillsVote); SAGE-style paired test with abstain for stage 3's tie-admitting "Δ ≥ 0"; n01/n10 reporting (SkillGen); length-neutral secondary analysis (Regression Tax osmosis; methods audit); gate-outside-genome invariant (Gate Presence); provenance for contamination tracking (VaG); convention-conflict probes (Gilestro); pass@k controls and whole-curriculum baseline for H6; H4 reframed around known asymmetric/interface-dependent transfer.

## 4. Deep-read summaries (10)

Full notes: `related-work/code-is-the-body-ourark.md`, `vag-pre-commit-gating.md`, `grasp.md`, `darwinx.md`, `evolution-of-sex-for-ai.md`, `sapo-co-evolving-skill-generation.md`, `skillgen.md`, `skillsvote.md`, `evoskill.md`, `skillevoreg.md`.

1. **OurArk / Code Is the Body (2607.28691, high).** Agent = (body B: versioned Git repo of manifest/runtime/adapters/skills/policies/tests/evolution code; private state S: memories/credentials/logs, not inherited; reasoner R: replaceable, weights not inherited) under a human custodian. Genesis descent copies a clean ancestor body, rewrites identity, records immutable parent-at-birth provenance, runs the inherited test suite and creates the birth commit only on pass. Evolution from six sources; each change in an isolated worktree, validated, PR'd, merged only by a human. Post-divergence inheritance is pull-based and selective. Evidence: one linear lineage (4 agents / 3 descents, 2 test-verified), regression suites, mocked model calls; no benchmark, ablation, gated-vs-ungated comparison or population. **Different:** no causal gate; body/private-state boundary is not a somatic staging store (nothing in S is promoted; body edits become heritable on human merge); untyped repository with an explicit rejection of a central schema; trajectory acquisition only a candidate source; heritability never measured. **Implication:** soften S2 wording; add §3 items; optional borrowings (birth-time inherited-contract check; birth-commit provenance).
2. **VaG (2608.05810, medium-high for C4/H2/H5).** Frozen-weight terminal agent distills NL skills per round; Cold<Warm<Hot tiers; Gate 1 = schema + add-one held-out A/B replay (ties pass) + LLM review; Gate 2 = greedy marginal-gain joint selection. Ungated pool rises then falls (62→50%, 179 skills); VaG monotone to 72% with 37 skills; source-only rollback recovers little (contamination irreversible); ablations −10/−8 pp; frozen pool lifts four backbones +8 to +16 pp. One backbone, one seed, ~30 pp Wilson bands, headline on the distillation split. **Different:** no soma; untyped; no replay-reproducibility, attribution, CI threshold, protected-set regression or re-validation; gate controls runtime admission in one agent, not heritability. **Implication:** VaG-style baseline arm; reword "prerequisite vs post-hoc"; provenance of distillation context; no tie admission.
3. **GRASP (2605.29668, medium; EMNLP 2026).** ≤10-skill library; K=4 proposed edits; balanced 36-item held-out probe; accept iff net (fixes−regressions) > 0 and R(c) ≤ R0. MedAgentBench 40.6→88.8; gate ablations show the gate and budget, not writing, drive gains; gate given to baselines helps in-domain not OOD; frozen libraries transfer asymmetrically and interface-dependently (FHIR-AgentBench degraded below baseline). **Different:** one lifetime; library-level single-edit gate on an in-distribution probe; no trait-level ablation, attribution, multi-environment replay or OOD stage; untyped. **Implication:** performance-gated arm B′; reuse balanced-probe construction; H4 framing.
4. **DarwinX (2608.07545, medium).** Population of layer-typed harness variants (skill layer + code layer) of a proprietary agent on a frozen model; preserve-and-extend contract (g > 0, R ≤ δ vs parent); verifier promote/revert; preservation probe and avg@k confirmation before a child steers search; merges kept only on union coverage; parents by cumulative lineage gain; failure/teacher/self evidence through one edit interface; shared failure-theme memory. TB2.1 75.5→83.2; held-out TerminalWorld 25→28/41 (p = 0.45) with 31.7-point proxy overfit; WebArena-Infinity 43.5→93.0; every merge reverted. Authors disclaim per-skill causal attribution. **Different:** no soma (mutate-then-select); variant/edit-level gate; held-out not a gate condition; no trait schema/regulation; offline proposer acquisition. **Implication:** regression-protection-as-prerequisite is prior art; CIG defends trait-level ablation/attribution, held-out as gate condition, soma→germline lifecycle; DarwinX-style ablation arm.
5. **Gilestro (2609.18560, medium for the biology framing).** Population-genetics mapping at three tiers; LLM tier: 3 lineages × 6 generations × 3 seeds, frozen Qwen2.5-1.5B, one new supervised LoRA skill per generation, child continues parent adapter (Lamarckian); merge arms never/obligate/own-ancestor/declinable/forced-stop. Obligate merging collapses; declinable ≈ never ≈ forced stop ≈ whole-syllabus single model; declinable gate's effect explained by timing; own-ancestor beats contemporary merge; functional disagreement predicts merge damage (ρ ≈ 0.45). **Different:** SFT skills, no agentic environment; no soma (adapter is both working and heritable state); untyped weight deltas; gate only on merges; trained adapters. **Implication:** whole-curriculum baseline; acceptance-rate-matched control; convention-conflict probes; cite as the Lamarckian counterpart.
6. **SAPO (2606.08755, medium).** GRPO-trained policy writes a JSON skill per K=4 queries; matched half-budget rollouts with/without the candidate on the same queries give marginal utility; temporary bank → long-term bank (cap 45) every 5 steps iff U > 0, top-ρ, similarity < 0.8; utility also trains the generator; likelihood-based pruning/reranking; bank seeded from SkillRL. ALFWorld 92.2; removing validation costs 1.6 (ALFWorld) / 7.5 (WebShop score). Never uses "causal". **Different:** one training run, weights trained; probation within a run; untyped JSON; single-stage in-distribution gate without CI, held-out, regression or interaction stages. **Implication:** SAPO-style stage-2-only ablation arm; soften "zero direct hits" for S2.
7. **SkillGen (2605.10999, medium).** One auditable skill per (model, benchmark) from contrastive induction; K=8 rounds of paired with/without verification under explicit potential outcomes; n01 repairs vs n10 regressions; best-of-K then hard gate G ≥ max{2, ⌈0.05m⌉, 1} → active/deprecated; frozen weights; +3.27 to +10.08 pp held-out across 8 models; 70% of 120 transfer pairs non-negative. **Different:** no genome, soma or generations; deployment gate; regressions netted (App. C.9: 51 repairs vs 57 regressions accepted); no interaction/attribution; held-out for reporting only; best-of-K winner's curse. **Implication:** bound regression separately; n01/n10 reporting; decide promotion on data not used for selection.
8. **SkillsVote (2605.18401, medium).** Four-stage lifecycle on a frozen Codex/GPT-5.x solver: profiling of >1.68M GitHub skills; pre-task recommendation; post-task LLM self-attribution of trajectory subtasks (11 categories; verifier counts online, gold solution offline "oracle mode"); evidence-gated evolution admitting only successful reusable subtasks. Online +0.9 to +2.7 pp; ungated baselines regress; frozen library transfers +1.4 to +7.8 pp. No CIs or ablations. **Different:** one live library; observational self-attribution gate without ablation, held-out, regression or thresholds; SKILL.md library not a typed genome; single-hop transfer; oracle mode conflicts with our contamination rule. **Implication:** contrast observational vs interventional gating; optional self-attribution comparison arm.
9. **EvoSkill (2603.02766, medium).** Frozen Opus 4.5 in Claude Code; Proposer sees failure traces plus ground truth and proposes one skill change; child scored once on held-out validation; admitted to top-k frontier (K=3) if not full or beats the weakest; round-robin parents; git-backed lineage with generation counters; OfficeQA 60.6→67.9 (17-item validation set); one skill transfers zero-shot to BrowseComp. **Different:** no soma; scalar frontier-relative gate (can admit a child worse than its parent), unconditional below capacity, no CIs, no ablation/attribution/regression/OOD/interaction; untyped; "generations" = mutation depth in one run; supervised proposer. **Implication:** performance-gated arm; cite frontier-relative admission as motivation for parent comparison and τ_r; no superiority implication (SPEC §34).
10. **SkillEvoReg (2609.30861, medium).** Regularization layer on SkillOpt/SkillEvolBench/ContinualSkillBench updaters: skill dropout during update generation; complexity-aware local regularization; Causal Counterexample Validation (one training source, one LLM-generated attack on the candidate delta, both versions evaluated; at most one scope-constrained repair, never a reject; causal identification disclaimed). One proprietary LLM, one trajectory per condition; smaller skills with similar/better held-out scores and some regressions; CCV evidence 12 signals / 9 repairs; CCV never fired in the ablation. **Different:** single library, untyped atoms, no inheritance; CCV single-source/single-case, detect-and-repair, not a persistence prerequisite. **Implication:** C4 row stays partially open; do not claim counterexample validation; optional clean-pair rule; Round-3 reads SkillCommit/GSE/SkillAdam.

## 5. Residual uncertainty

1. **Coverage** *(Round-2 state; resolved in Round 3 — see §6.1 and §6.5)*. Eight medium-threat works (SEA, RSEA, HDSO, AHE, SAGE, SkillBoost, HarnessEvolve, Self-Healing Harness) were assessed from abstracts only; SEA could rise to high if its full text has a lineage/generation section. SkillCommit (2608.15165), HarnessX (2606.14249), SkillOpt (2605.23904), GSE, SkillAdam and the OurArk/EvoSkill-cited priors (MOSS, Autogenesis, Self-Harness, Agent libOS, EvoGit, GEPA, Feedback Descent) were not read.
2. **Venues.** OpenReview ICLR 2027 submissions, the NeurIPS 2026 accepted list, Semantic Scholar and Google Scholar were not reachable; absence from search is not proof of absence. No ICLR 2027 paper is reported.
3. **Unverifiable items.** Methods-audit preprint (non-standard id); Genomebook bioRxiv details (429); authors of 2606.23075; exact date of 2607.04240; date of 2609.29543; CapaBench v1 title and rename-version mapping; AgentSquare ICLR 2025 venue from arXiv; SkillRL NeurIPS 2026 acceptance beyond the arXiv comment; ASSAY's randomized-masking mechanism; OMEGA/RECLAIM's Baldwin section; the "and 3 other authors" artifact on 2607.28691; SkillGen code repo; secondhand claims (WikiSkill gate rule, ARTEMIS "agent-as-genome", Biological Motifs "epigenetic state", SkillMAS disclaimer).
4. **Evidence quality of the threats themselves.** Several closest works are single-seed, single-backbone or small-n (VaG; EvoSkill 17-item validation; SkillsVote single rollout; SkillEvoReg one trajectory per condition; OurArk mocked model calls). Threat levels reflect mechanism overlap, not strength of evidence.
5. **Adjudication is single-agent.** Threat levels and must-not-claim wording were set by Claude's subagents and consolidated by Claude; the Research/Critic agent has not yet cross-reviewed. Chain-tag assignment at the sweep level was abstract-based and is coarser than the deep-read vectors.
6. **Builder/Critic agent communications (Zcode/Codex) were tracked separately** (see `research/coordination/agent-coordination-report.md`), not in this literature subtask. The `main` ledger was last edited 2026-10-04 (issue #9 updates, commit 7a066c0); this candidate ledger was built on that version.


---

## 6. Round-3 completion addendum (2026-10-13)

Author: Claude (third agent) — adjudication of full reads and abstract-page triage run by Claude subagents. **Pending Research/Critic cross-review.** No repository file was modified; this addendum and the updated ledger live under the scratchpad `out/research/` tree. Hard rules unchanged: no paper reported unless opened and confirmed; no ids, authors or numbers invented; unverifiable items labelled as such.

### 6.1 Scope

No new sweep was run. Round 3 completed the Round-2 reading queue (§5 item 1 above):

- **14 full reads** (notes in `related-work/round3/`): the eight abstract-level medium entries from ledger §2.10.11–2.10.18 (HDSO, Agentic Harness Engineering, SEA, SAGE, RSEA, SkillBoost, HarnessEvolve, Self-Healing Harness); the five queued commit-gate competitors (SkillCommit, GSE, SkillAdam, HarnessX, SkillOpt); and Designer-RSI (previously seen only in a search snippet).
- **14 abstract-page triages** of queued priors: MOSS, Autogenesis, Self-Harness, Agent libOS, EvoGit (cited by OurArk); GEPA, Feedback Descent (cited by EvoSkill); Evo-Bench, SHE, HarnessDev, the Learngene line, EvoOntology, EvoTreeNAD, RELAI-VCL (search-snippet-only in Round 2).
- Sources: arXiv HTML or PDF full text for the 14 reads (SEA: §§1–11 + App. B–E; HarnessX: v3 HTML main + App. 9, 10.1, 10.3, 11, 13; SkillOpt: HTML v2 + App. A–C; SkillAdam: 17-page PDF, no appendix; SkillCommit v1 with App. A/B missing); arXiv abstract pages for the triages and for the new leads SGM 2510.10232 and SelfEvoSkill 2606.14239; AAAI-26 pages returned HTTP 502.

### 6.2 Newly read / triaged works and changed verdicts

Coverage vectors use the six elements in order (L1 S2 T3 C4 A5 G6; F = full, P = partial, – = absent). "R2 verdict" is the Round-2 threat level and its basis.

| # | Work | id | Depth (R3) | R2 verdict | R3 verdict | R3 coverage | Change | Reason (one line) | Note (`related-work/round3/`) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | SEA — Self-Evolving Agents with Anytime-Valid Certificates | 2607.00871 | full | medium (abs; "possible HIGH", read first) | **medium, lower end** | P – P P P P | HIGH flag withdrawn; L1 – → P | no lineage/generation/inheritance section; one agent, one run; certified gate (Alg 4) disabled in the full stack and accepted 0 edits run alone; "steering adapter" is a directive softmax | sea-anytime-valid-certificates.md |
| 2 | RSEA — Recursive Self-Evolving Agents via Held-Out Selection | 2606.28374 | full | medium (abs) | **low** | F – P P P P | ↓ | "generations" are rewrite iterations of one agent; whole-triple untested keep-better; single-seed transfer ties ReAct; sub-additive layers | rsea-held-out-selection.md |
| 3 | HDSO — Hypothesis-Driven Skill Optimization | 2606.22330 | full | medium (abs) | **medium** (scope shifted) | F – P P P P | T3 – → P; R2 error corrected | has a staged in-scope/out-of-scope paired gate with exact McNemar, uptake log and rollback records on ALFWorld / frozen Qwen; no ungated arm; delivery interface may explain much of the gain | hdso-hypothesis-driven-skill-optimization.md |
| 4 | Agentic Harness Engineering (AHE) | 2604.25850 | full | medium (abs) | **low** | P – P P P P | ↓; L1, A5 – → P | retrospective, co-shipped keep/rollback on the same 89 tasks, in-sample best-of-10; the authors' own regression prediction fails (11.8/11.1%) | agentic-harness-engineering.md |
| 5 | SAGE — Statistical Acceptance Gate | 2609.36043 | full | medium (abs) | **low** (statistics relevance ↑) | F – – P P P | ↓ | one untyped document, edit-level single-stage gate, per-comparison error only; point estimates, no seeds/CIs/sizes/code | sage-statistical-acceptance-gate.md |
| 6 | SkillBoost — Constrained Exploration-Exploitation | 2607.26643 | full | medium (abs) | **low** | F – P P P P | ↓; T3 – → P | gate is an in-sample back-test on the training split; ε unquantified, never ablated; "attribution" is observational diagnosis | skillboost-constrained-exploration.md |
| 7 | HarnessEvolve | 2609.00829 | full | medium (abs) | **low** | F – P P P P | ↓; T3 – → P | performance gate tests the training batch the edit came from, δ = 0 admits ties, val reused for argmax + early stopping; miner (−29.1) not gate (−6.8) drives gains | harnessevolve-reference-trajectories.md |
| 8 | Self-Healing Harness | 2609.24130 | full | medium (abs) | **low** | F P P P P P | ↓; S2, T3 – → P | single replay vs historical scores, ≤ 2 protected cases, ≈ 2/3 of promotions skipped the protected test; its own "B′" arm not run; "55%" = 26% of replay verdicts | self-healing-harness-admission-control.md |
| 9 | SkillCommit — Behaviorally Validated Scope Expansion | 2608.15165 | full | queued ("most likely commit-gate competitor") | **low** (medium for 3 sub-claims) | F P P P P P | new | source-retention replay with no no-patch control; commit stage worth 2 of 45 cases; App. A/B missing | skillcommit-validated-scope-expansion.md |
| 10 | GSE — Globalized Skill Evolution | 2608.06153 | full | queued; cited in §3.14 | **medium** | F P P P P P | new | typed dependency/co-usage/conflict skill relations + interaction-targeted regression replay are prior art; gate is replay + zero-margin, cluster-level, no stats | gse-globalized-skill-evolution.md |
| 11 | SkillAdam | 2609.08944 | full | queued; cited in §3.14 for "overfitting" | **low** | F – – P P P | new; §3.14 corrected | in-sample same-batch single-sample gate with unreported thresholds; never discusses overfitting; single seed, 6/60-case effects | skilladam-adam-inspired-skill-evolution.md |
| 12 | HarnessX — Typed Harness Foundry | 2606.14249 | full | queued; §3.22 "per abstract" | **medium** (element 3 only) | P – P P P P | new | runtime-contract typed, hashable configuration + per-edit manifests with attribution signatures; seesaw gate lets −14/−24 pp through; held-out not evaluated; co-evolution unfreezes weights | harnessx-typed-harness-foundry.md |
| 13 | SkillOpt | 2605.23904 | full | low (abs, §2.11) | **low, confirmed** (baseline relevance ↑) | F – P P P P | confirmed | bundle-level strict held-out gate with one cached evaluation, no test, no gate ablation; strongest whole-artifact performance baseline; headline cells match best sweep rows | skillopt-text-space-skill-optimizer.md |
| 14 | Designer-RSI — Procedural Memory from User Traffic | 2609.22086 | full (after abs triage: medium) | snippet only | **medium** | F – P P P P | new; T3, C4 raised from triage | confound-controlled paired replay gate (frozen contexts, same batch, order-swapped judging); typed SKILL.md; no ungated arm; round-4 regression | designer-rsi-procedural-memory-replay-gate.md |
| 15 | MOSS | 2605.22794 | abs | queued (OurArk prior) | low | L1 T3 A5 (loose) | new | source rewriting verified by failure-batch replay, consent-gated promotion | — |
| 16 | Autogenesis Protocol | 2604.15034 | abs | queued (OurArk prior) | low | L1 A5 (weak) | new | versioned resources, propose/assess/commit with rollback | — |
| 17 | Self-Harness | 2606.09498 | abs | queued (OurArk prior) | low | L1 A5 (weak) | new | regression-tested harness self-modification; held-in and held-out gains | — |
| 18 | Agent libOS | 2606.03895 | abs | queued (OurArk prior) | low | L1 | new | permission/approval admission control, not evidence gating | — |
| 19 | EvoGit | 2506.02049 | abs | queued (OurArk prior) | low | T3 (loose) | new | Git phylogeny of code; human promote/prune | — |
| 20 | GEPA | 2507.19457 | abs | queued (EvoSkill prior) | low | L1 A5 (loose) | new | reflective Pareto prompt evolution; background citation | — |
| 21 | Feedback Descent | 2511.07919 | abs | queued (EvoSkill prior) | none | — | new | pairwise-feedback text optimization; no chain elements | — |
| 22 | Evo-Bench | 2608.09096 | abs | snippet only | low | T3 G6 (loose) | new | harness-improvement benchmark; evaluation neighbour | — |
| 23 | SHE | 2608.09885 | abs | snippet only | low | L1 T3 A5 (loose) | new | typed safety-harness evolution with safety + utility selection | — |
| 24 | HarnessDev | 2609.01437 | abs | snippet only | low | L1 T3 G6 (loose) | new | harness creation/evolution benchmark; unstable gains | — |
| 25 | Learngene line (2306.10225; AAAI-26 reprint + Adaptive-Learngene) | 2306.10225 | abs (AAAI pages 502) | low (R2 short entry); AAAI listing unopened | low | L1 A5 (loose) | AAAI identity ambiguous | weight-fragment inheritance with trained descendants | — |
| 26 | EvoOntology | 2609.15779 | abs | snippet only | low | A5 (loose) | new | backbone-conditioned paired acceptance of ontology edits | — |
| 27 | EvoTreeNAD | 2609.29016 | abs | snippet only | none | — | new | fitness-guided NAS genealogy; no agent-skill elements | — |
| 28 | Do Agent Optimizers Compound? / RELAI-VCL | 2607.14004 | abs | "judged low/no relevance" (R2 §2.12) | low (basis stated) | L1 A5 (loose) | moved out of the unentered list | regression-controlled lifelong harness optimization; numbers from a snippet only | — |

New leads (title/abstract only, queued in ledger §2.12): SGM 2510.10232 (from SEA); SkillCAT 2606.13317 (from SkillCommit); SkillOS 2605.06614 and SkillEvolver 2605.10500 (from HDSO); SkillAxe 2606.10546 and CoEvoSkills 2604.01687 (from SkillAdam); SelfEvoSkill 2606.14239 re-check (from Designer-RSI, which cites it as "SkillAudit").

### 6.3 Verdict (a) re-check — does any work implement the full chain?

**No.** The §3.1 table is extended with the Round-3 full-read vectors (order L1 S2 T3 C4 A5 G6):

| Work | L1 | S2 | T3 | C4 | A5 | G6 | Full chain? |
|---|---|---|---|---|---|---|---|
| SEA 2607.00871 (full) | P | – | P | P | P | P | no — one agent, one run; certified gate inactive; no lineage section (the Round-2 HIGH condition is not met) |
| RSEA 2606.28374 (full) | F | – | P | P | P | P | no — rewrite iterations of one agent; untested whole-state keep-better |
| HDSO 2606.22330 (full) | F | – | P | P | P | P | no — one deployment; add-one gate against the current repository; no re-validation |
| AHE 2604.25850 (full) | P | – | P | P | P | P | no — retrospective in-sample verdict; file-class types |
| SAGE 2609.36043 (full) | F | – | – | P | P | P | no — one untyped document; edit-level in-distribution gate |
| SkillBoost 2607.26643 (full) | F | – | P | P | P | P | no — in-sample back-test; version chain of one run |
| HarnessEvolve 2609.00829 (full) | F | – | P | P | P | P | no — in-sample bundle gate; snapshot pool is version selection |
| Self-Healing Harness 2609.24130 (full) | F | P | P | P | P | P | no — one online stream; historical-score replay; no held-out |
| SkillCommit 2608.15165 (full) | F | P | P | P | P | P | no — abstraction tiers, not heritability; source-retention replay only |
| GSE 2608.06153 (full) | F | P | P | P | P | P | no — one bank per run; cluster-level replay gate; no inheritance |
| SkillAdam 2609.08944 (full) | F | – | – | P | P | P | no — same-batch gate; optimizer state discarded |
| HarnessX 2606.14249 (full) | P | – | P | P | P | P | no — in-sample seesaw gate; routed variants, not lineages |
| SkillOpt 2605.23904 (full) | F | – | P | P | P | P | no — bundle-level strict held-out gate; one document |
| Designer-RSI 2609.22086 (full) | F | – | P | P | P | P | no — local replay gate; one deployed bank |

Every work with S2 = P has only a provisional tier, an abstraction level or a transient buffer inside one live store; every C4 = P is either in-sample or whole-artifact / single-edit; no work has agent generations with distinct lifetimes. Element C4 at the trait level with held-out and regression stages as a prerequisite for heritability across generations, S2 as a gated staging store of lifetime traits, and T3 as a schema of policy + skill + regulation trait types carrying per-trait validation records remain absent in every work read. The claim position does not flip; its Round-2 conjunction wording stands (ledger §1).

### 6.4 What changed in the ledger

- **§1:** Round-3 paragraph on what the full reads concede; table rows updated with Round-3 analogues per element; Round-3 completion verdict (SEA HIGH flag withdrawn; six medium → low; GSE, HarnessX, Designer-RSI new medium; SkillCommit, SkillAdam, SkillOpt low); confidence restated as two sweeps plus a completion pass in which every medium/high work has been read in full; anti-overclaim addendum (SGM / SEA / SAGE precede CIG as statistically principled gates; no "certified self-improvement" attribution to SEA; "causal" discipline); nine Round-3 naming hazards (incl. the "B′" collision with Self-Healing Harness).
- **§2.10:** entries 2.10.11–2.10.18 rewritten at full-read level with corrected mechanisms and evidence; new full-format entries 2.10.19 GSE, 2.10.20 HarnessX, 2.10.21 Designer-RSI; numbering kept stable with a note.
- **§2.11:** SkillOpt short entry replaced by its full read; new short full-read entries for SkillCommit and SkillAdam; a 14-item abstract-level triage block.
- **§2.12:** queue rewritten — Round-3 clearances, seven new leads, SelfEvoSkill re-check, HarnessBank / WikiSkill still unread, unverifiable items.
- **§3:** items 12, 14, 15, 22 corrected and extended; new items 29–35; the "what remains ours" paragraph gains a Round-3 addendum on wording (interventional validation evidence; agent generations with distinct lifetimes; the combination of trait types with per-trait records; a held-out performance-gated comparator at least as strong as SAGE / GRASP).
- **§4:** a Round-3 corrections table (11 rows) for statements in Round-2 entries that full reads overturned.
- **§5:** item 1 rewritten as a gate ladder with a recommended held-out primary and a frozen-candidate offline "gate ladder" secondary analysis; items 4, 5, 6, 9, 10 extended; new items 11–21 (joint-genome check; placebo traits; no in-sample evidence; hygiene filter; delivery confound; composite candidates; expression precondition; schema additions; optimizer memory as an inheritance channel; judge separation; reporting). Every item states whether it requires an ADR; no design was changed.

### 6.5 Residual uncertainty after Round 3

1. **Coverage.** Every medium/high-threat work in the ledger has now been read in full. Still unread: SGM 2510.10232 (origin of SEA's gate; abstract suggests non-LLM scope), SkillCAT, SkillOS, SkillEvolver, SkillAxe, CoEvoSkills (titles only); SelfEvoSkill (re-check); the full texts of HarnessBank and WikiSkill; the 14 triaged priors (abstract level — Autogenesis and Self-Harness most likely to show a stronger gate in the body). No new sweep was run, so works posted after 2026-10-12 are not covered.
2. **Venues.** OpenReview ICLR 2027, the NeurIPS 2026 accepted list, Semantic Scholar, Google Scholar and the AAAI-26 Learngene pages (502) remain unreachable; absence from search is not proof of absence.
3. **Unverifiable items from the full reads.** SkillOpt's headline-vs-sweep configuration selection and its inconsistent split ratios; SAGE's hyperparameters, set sizes and regression-rate data source; SkillCommit's missing appendices; HarnessEvolve's split sizes and budgets; GSE's replay metric; Designer-RSI's δ and contexts per prompt; SkillAdam's thresholds and code; SEA's Alg 4 under the full stack (disabled); RELAI-VCL's numbers (snippet only); the Round-2 unverifiable items (methods-audit id, Genomebook, 2606.23075 authors, 2607.04240 and 2609.29543 dates, CapaBench titles, AgentSquare venue, SkillRL acceptance, ASSAY mechanism, OMEGA Baldwin section).
4. **Evidence quality of the threats.** The Round-3 reads lowered more threats than they raised, largely because the full texts show thinner evidence than the abstracts implied (single runs: SkillCommit, GSE, HarnessEvolve, SkillAdam, SkillOpt, HarnessX, Designer-RSI, AHE, Self-Healing Harness; point estimates only: SAGE; a certified gate that accepted nothing: SEA). Threat levels still reflect mechanism overlap, not strength of evidence, and none of these numbers should be used as effect-size priors.
5. **Adjudication is single-agent.** Full reads, triage and consolidation were done by Claude's subagents and Claude; the Research/Critic agent has not cross-reviewed. Several "my inference" readings are flagged in the entries (SEA Table 4 control runs; HDSO Tables 1 and 3 delivery effect; Self-Healing Harness forward-path count by subtraction).
6. **Builder/Critic communications (Zcode/Codex)** are covered in `research/coordination/`, not here. The candidate ledger was built on the 2026-10-04 `main` ledger; `main` has no later ledger edits (checked at 6014bc7).
