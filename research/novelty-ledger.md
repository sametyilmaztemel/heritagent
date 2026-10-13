# Novelty Ledger

Living document per SPEC §30. Every relevant paper is recorded here with: paper / date / core mechanism / overlap with HeritAgent / what remains different / required architecture changes. Update whenever a new paper appears; adjust the novelty claim if needed.

- **Last updated:** 2026-10-04 (Round 1 + issue #9 miner updates: SkillRL NeurIPS 2026 status, SkillForge 2608.24747, Agent Skills Can Be Harmful 2608.11888; see `related-work/` for detailed notes) (Round 2-3 addendum proposed 2026-10-13, pending Research/Critic cross-review)
- **Next review:** before any paper submission, and on any new arXiv hit in the sweep categories

---

## 1. Current novelty position (2026-10-02)

**Intended claim:** HeritAgent's contribution is the *integration and the gating semantics*: an explicit somatic/germline separation for LLM agents, a typed executable functional agent genome (EAG) as the first-class heritable artifact, and a Causal Inheritance Gate (replay → controlled ablation → attribution → held-out generalization → regression protection) as a **prerequisite** for germline assimilation — evaluated across generations with frozen foundation-model weights.

**Verified status of the chain (post-review):**

| Chain element (SPEC §3) | Status in literature |
|---|---|
| Lifetime-acquired capabilities from trajectories | **Well covered** — do not claim (Voyager 2023; SkillRL 2026; MUSE-Autoskill; CASCADE) |
| Somatic vs germline store separation | **Open** — zero direct hits for LLM agents |
| Typed, versioned executable functional genome (policies + skills + regulation) | **Open** — closest are untyped code genomes (DGM/MGM) and behavioural-descriptor genomes (Genomebook) |
| Causal validation as prerequisite for inheritance | **Partially open** — causal skill *measurement* exists within one lifetime (ASSAY, Skill-α); promotion gates exist but governance-flavored (LOGOS); nobody gates *germline assimilation* on trait-level causal evidence |
| Selective assimilation into heritable genome | **Open** as part of the full chain |
| Multi-generational inheritance, frozen weights | **Well covered as a mechanism** — do not claim alone (DGM, MGM, Genomebook); our claim is what is inherited and how it is validated, not that generations exist |

**Round-1 verdict (sweep agent + seven paper reviews):** no single work implements the full chain. Confidence: moderate — based on one structured sweep (2023–2026, ~12 query families). This is NOT a "first work" claim; monitoring continues.

**Anti-overclaim reminder (SPEC §34):** the words *first, novel, better, causal, general, model-independent, evolutionary improvement, inheritance* require experimental evidence. Use "contribution estimate" / "association" unless controlled interventions were run.

### Naming hazard

CapaBench (Yang et al., Findings of ACL 2026, arXiv:2502.00510) **renamed its framework to "ShapleyFlow"** in its v3/ACL version — a collision with the original ShapleyFlow (Wang et al., AISTATS 2021, arXiv:2010.14592). In all HeritAgent writing: cite "ShapleyFlow (Wang et al. 2021)" for full-graph Shapley axiomatics, "CapaBench (Yang et al. 2026)" for agent-module attribution, and never reuse either name for a HeritAgent component.

---

## 2. Ledger entries

Format per SPEC §30. Detailed notes live in `research/related-work/<name>.md`.

### 2.1 AgentSquare — ICLR 2025

- **Paper:** *AgentSquare: Automatic LLM Agent Search in Modular Design Space* — Yu Shang*, Yu Li*, Keyu Zhao, Likai Ma, Jiahe Liu, Fengli Xu, Yong Li (Tsinghua). arXiv:2410.06153 (Oct 2024, v3 Feb 2025). Code: github.com/tsinghua-fib-lab/AgentSquare.
- **Core mechanism:** MoLAS modular design space — agent = (Planning, Reasoning, Tool Use, Memory) with uniform IO interfaces; LLM-driven module evolution + recombination search with an in-context performance surrogate.
- **Overlap:** typed modular agent spec (genome-like but flat 4-tuple); module pools retained by whole-agent fitness; frozen weights; iterative/generational search. Verified: +17.2% avg over best human design across 6 benchmarks (matches SPEC §3).
- **What remains different:** no trajectory-mined lifetime capabilities; no somatic/germline separation; no per-trait causal gate (surrogate scores ≠ controlled ablation); no versioned genome with skills/regulation; search is per-task and one-time, not cross-lifetime inheritance.
- **Required architecture changes:** none. Cite as modular-search prior art; position EAG's versioning/skills/regulation against their flat tuple.

### 2.2 Darwin Gödel Machine (DGM) — 2025 preprint

- **Paper:** *Darwin Gödel Machine: Open-Ended Evolution of Self-Improving Agents* — Jenny Zhang, Shengran Hu, Cong Lu, Robert Lange, Jeff Clune (UBC/Vector/Sakana). arXiv:2505.22954 (May 2025).
- **Core mechanism:** self-modifying coding agent (whole Python codebase = genotype); archive-based branching tree; parent selection ∝ performance + progeny viability; compile + editing-functionality regression check + benchmark subset as the only gate.
- **Overlap:** generations, lineages, archive, frozen weights (element 6 fully); empirical validation of whole agents; stepstones/backtracking.
- **What remains different:** no somatic store (traits go straight into inherited code); no per-trait credit assignment; gate is coarse (compile + 10-task check + 50-task benchmark) and was shown gameable (reward hacking documented); genome is untyped arbitrary code; transfer measured post-hoc, not as a gate.
- **Required architecture changes:** none. Cite for archive/lineage design and for the reward-hacking cautionary tale → motivates CIG's multi-stage gate (SPEC §9).

### 2.3 Mendel Gödel Machine (MGM) — Aug 2026 preprint

- **Paper:** *Mendel Gödel Machine: Recursive Self-Improving Coding Agents via Comparative Evolution* — Changzhi Liu, Yilun Liu, Sikuan Yan, Volker Tresp, Yunpu Ma (UESTC + LMU Munich/MCML). arXiv:2608.07645. Not a Clune-lab work.
- **Core mechanism:** reaction-norm mutation (edit conditioned on multiple tasks' trajectories to separate genotype-level defects from task-specific accidents) + cross-lineage hybridization (failing agent edited using another lineage's trajectory on the same task). Thompson sampling over Beta posteriors on an archive tree.
- **Overlap:** cross-lineage evidence use, comparative controlled *diagnosis*, executable versioned genotype (untyped code), frozen backbone (verified by cross-model transfer: Polyglot scaffold on DeepSeek 96.89%).
- **What remains different:** their "controlled inheritance" = controlled **diagnosis of defects**, not controlled **gating of heritability** — children enter the archive on a syntactic validity check; usefulness emerges only statistically, post hoc, population-level. No somatic/germline split; no per-trait ablation/attribution/held-out gate; no assimilation step.
- **Required architecture changes:** none to the claim; borrow the reaction-norm insight for CIG replay design (validate a candidate across **multiple** environments, not just its source task — already implied by SPEC §9 replay + generalization stages).

### 2.4 Genomebook — Mar 2026 preprint

- **Paper:** *Genomebook: Mendelian inheritance of behavioural traits in large language model agents across eight generations* — Manuel Corpas (Univ. of Westminster), bioRxiv 10.64898/2026.03.22.713494 (posted 2026-03-24). No arXiv version.
- **Core mechanism:** 26 behavioural/personality traits → 60 diploid loci (22 autosomes + sex chromosomes), additive/dominant/recessive modes, effect sizes 0.15–0.50; phenotype = SOUL.md + DNA.md prompt injection into frozen Claude; pairwise breeding with compatibility-based selection; 8 generations, 626 agents on a simulated social network.
- **Overlap:** explicit genotype/phenotype vocabulary for LLM agents, Mendelian machinery, multi-generation inheritance, frozen weights (element 6 + genetics framing).
- **What remains different — and this is the key contrast for our paper:** Genomebook's genotype parameterizes **behavioural descriptors** ("leadership drive", "obsessive focus"), not **executable functional capabilities**. No lifetime learning at all (traits hand-authored at founding; heritability is by design, not earned); no task trajectories; no validation gate; authors themselves concede results "may reflect prompt conditioning rather than genetic causation"; no external benchmark.
- **Required architecture changes:** none. This is the cleanest foil for "functional vs behavioural genome" — cite early in Related Work.

### 2.5 SkillRL — Feb 2026 (NeurIPS 2026)

- **Paper:** *SkillRL: Evolving Agents via Recursive Skill-Augmented Reinforcement Learning* — Peng Xia et al. (aiming-lab / Huaxiu Yao group). arXiv:2602.08234. **Status update 2026-10-04: accepted at NeurIPS 2026.** Trait Miner v0 (#9) explicitly credits SkillRL's experience-based differential skill distillation (success/failure trajectories → reusable skills) as the adapted acquisition mechanism; HeritAgent's distinction remains the somatic→germline lifecycle with causal gating.
- **Core mechanism:** o3-teacher differential distillation from success/failure trajectories → SkillBank (general + task-specific skills with `when_to_apply`), GRPO **weight training** (weights NOT frozen), recursive skill-library growth at validation checkpoints (≤3 skills/update, triggered by weak categories <0.4 success).
- **Overlap:** trajectory → reusable skill extraction with activation conditions (element 1 + primitive regulation); ALFWorld 89.9 / WebShop 85.2 headline numbers.
- **What remains different:** no somatic/germline separation; skill persistence is heuristic (rate-limited growth + Top-K retrieval), not causally validated — no per-skill ablation, attribution, or regression tests before a skill persists; "recursive evolution" is within **one training run**, not across agent generations; weights are fine-tuned (contradicts our frozen-weights setting).
- **Required architecture changes:** adopt their success/failure differential distillation as Trait Miner v0 (cite — adapted, not claimed); their SkillBank→Top-K is exactly the "unrestricted/persistent-skill baseline" arm B of EXP-0001.

### 2.6 ShapleyFlow — AISTATS 2021

- **Paper:** *Shapley Flow: A Graph-based Approach to Interpreting Model Predictions* — Jiaxuan Wang, Jenna Wiens, Scott Lundberg (U. Michigan). arXiv:2010.14592. (AISTATS, **not** NeurIPS — correct SPEC §3 if it says otherwise.)
- **Core mechanism:** Shapley attribution on **edges of a full causal DAG** (features, latents, model output); unique Shapley-axiom solution generalizing SHAP/ASV/Owen values; Monte-Carlo over ~10k orderings.
- **Overlap:** justifies full-graph Shapley as our attribution primitive when genome interactions exist (SPEC §9 stage 4).
- **What remains different:** single-prediction explainability, no agents, no persistence, no decision made on the attribution.
- **Required architecture changes:** none; citation grounding for CIG attribution semantics.

### 2.7 CapaBench — Findings of ACL 2026

- **Paper:** *Attribution-Based Analysis and Optimization of Modular Agentic Workflows* — Yingxuan Yang et al. (SJTU/UChicago/Toronto/Meituan). arXiv:2502.00510 (v1 Feb 2025 titled *"Who's the MVP?..."*; ACL version renamed framework to "ShapleyFlow" — see naming hazard above).
- **Core mechanism:** fixed 4-module pipeline (Planning/Reasoning/Action/Reflection); exact 2⁴ coalition enumeration → per-module Shapley; used to pick which LLM goes in which slot.
- **Overlap:** agent-level module attribution via Shapley; finding that Reflection contributes ~zero; mixing best-per-slot models beats best single model.
- **What remains different:** attribution for evaluation/configuration of interchangeable model calls in a static workflow — never a gate on inheritance of learned capabilities; no lineage.
- **Required architecture changes:** none; cite as the agent-attribution prior that HeritAgent moves "from explanation/configuration to lifecycle governance".

### 2.8 SkillShapley — Aug 2026 preprint

- **Paper:** *SkillShapley: Boundary-Adaptive Shapley Valuation for Skill Step Attribution in LLM Agents* — Chang Liu et al. (Beihang/Shandong). arXiv:2608.13173.
- **Core mechanism:** Shapley over instruction **blocks within one skill**; BAES sampler (anchor warm-up + variance-guided adaptive acquisition, one-flip caching); budgets ~3n² vs 2ⁿ enumeration; used to prune/reinforce low/high-value steps.
- **Overlap:** step-level marginal contribution inside skills; removal-curve validation.
- **What remains different:** single static skill, fixed benchmark, human-in-the-loop editing decision; no retention/inheritance threshold, no generations, no genome composition effects.
- **Required architecture changes:** candidate reuse of BAES-style budgeted sampling if CIG stage-4 interaction testing gets expensive — defer to implementation time.

### 2.9 Adjacent works from the sweep (short entries)

- **SkillForge** (arXiv:2608.24747, 2026) — evidence-based skill verification/refinement. Reduces any claim that "verifying extracted skills" is itself novel. HeritAgent's CIG must therefore be positioned as an **inheritance prerequisite tied to the somatic→germline lifecycle** (gating what becomes HERITABLE), not as skill verification per se. Added 2026-10-04 per issue #9 handoff.
- **Agent Skills Can Be Harmful** (arXiv:2608.11888, 2026) — shows functional failures and efficiency regressions induced by seemingly relevant skills, using differential comparisons for attribution. Strengthens the motivation for CIG regression/cost protection (H5, τ_r) and selective reuse; also supports the EXP-0001 arm B framing (unrestricted skill reuse as the risk baseline). Added 2026-10-04 per issue #9 handoff.

Detailed notes in `related-work/adjacent-sweep.md`.

- **ASSAY** — *Not All Skills Help: Measuring and Repairing Agent Skill Libraries* (arXiv:2606.15390, Jun 2026). Per-skill causal contribution via randomized masking; offline library restructuring; harmful-skill suppression per task. **Closest to CIG's measurement core, but single-lifetime curation — no inheritance, no genome.** Consider adopting randomized masking as the cheap ablation estimator inside CIG stage 2 (cite).
- **Skill-α** — *Progressive Agent Skill Generation via RL* (arXiv:2608.01678, Aug 2026). Per-edit rollback reward (counterfactual before/after execution). Causal per-edit gate, but no generations/genome/bio framing.
- **AgentSpawn** (arXiv:2602.07072, Feb 2026). Parent→offspring memory transfer at spawn time; selective slicing, no validation gate, no typed genome.
- **LOGOS** (arXiv:2607.10878, Jul 2026). Versioned "agent packs"; learned prompts/skills stay "untrusted release candidates" until held-out execution evidence + authorization → "promotion". Assimilation-gate-like, but governance/compliance-oriented; no trait-level causal attribution; no generational lifecycle.
- **MUSE-Autoskill** (arXiv:2605.27366) and **CASCADE** (arXiv:2512.23880): skill lifecycle / consolidation / cross-agent sharing; cumulative, not multi-generational; no causal gate.
- **Voyager** (arXiv:2305.16291, 2023): canonical lifetime skill library, executable code skills, self-verification, frozen GPT-4. Always cite; skills transfer across environments, not generations; verification = execution success, not causal ablation.
- **OMEGA / RECLAIM** (arXiv:2605.25062, May 2026): position paper with an explicit Baldwin-effect/genetic-assimilation section — **concept only, no implementation**. Cite to show the biology framing is in the air; our claim is the implemented, measured loop.
- *A Survey of Self-Evolving Agents* (arXiv:2507.21046, TMLR): taxonomy (what/when/how evolves) — useful Related Work scaffold; nothing on somatic/germline or assimilation gates.
- SSRN *When Does Diversity Rescue Evolutionary Search of LLM-Based Agents?*: six-gene "agent genome" as search space (genes = design components, not lifetime-assimilated traits).

---

## 3. What HeritAgent must NOT claim as novel (running list)

1. Modular agent design spaces and module recombination (AgentSquare).
2. Generations, archives, lineages, open-ended agent evolution (DGM).
3. Cross-lineage evidence use / hybridization / "controlled inheritance" as comparative diagnosis (MGM).
4. Genotype/phenotype and Mendelian vocabulary for LLM agents (Genomebook).
5. Trajectory → reusable skill extraction and skill libraries (Voyager, SkillRL, CASCADE, MUSE-Autoskill).
6. Causal/contribution attribution methods themselves (ShapleyFlow, CapaBench, SkillShapley, ASSAY, Skill-α).
7. Frozen-backbone evolution (all of the above).
8. Baldwin-effect/genetic-assimilation **rhetoric** (OMEGA/RECLAIM).

**What remains ours to defend empirically:** the somatic/germline architecture, the EAG as a typed heritable artifact of *functional* capabilities, CIG as a prerequisite for assimilation (not post-hoc curation), and the demonstration that gated inheritance beats unrestricted inheritance on held-out performance, regression rate, genome size, and descendant adaptation cost (EXP series; H1–H6).


---

## 4. Round 2-3 addendum (2026-10-13) — proposed by Claude (third agent), pending Research/Critic cross-review

**Status.** Everything below is a *proposal* from the independent third agent; §1-§3 above are unchanged and no SPEC, ADR or experiment design is altered (4.6 items are flagged "requires ADR"). Chain vectors follow the six SPEC §3 elements in order (1 lifetime acquisition, 2 somatic/germline, 3 typed executable genome, 4 causal gate, 5 selective assimilation, 6 multi-generational frozen weights); F = full, P = partial, – = absent. Basis: Round 2 (2026-10-12; 96 queries, 49 new works confirmed (47 on arXiv abstract pages, 2 on conference pages), 10 full reads, citation check of 21 Round-1 references) and a Round-3 completion pass (2026-10-13; no new sweep; 14 full reads, 14 abstract-page triages); every medium/high-threat work is now read in full.

### 4.1 Verdict

- **No work found implements all six chain elements.** Closest: *Code Is the Body / OurArk* (arXiv:2607.28691; 1 P, 2 P, 3 P, 4 –, 5 P, 6 F): multi-generation descent of a versioned executable body with non-inherited private state and frozen weights, but its gates are an inherited regression-test suite plus a human merge; no benchmark, ablation or gated-vs-ungated comparison.
- **Other closest works.** Element 4 in one lifetime: VaG (2608.05810), GRASP (2605.29668), HDSO (2606.22330), SkillOpt, SAGE. Regression-protected admission for heritable variants: DarwinX (2608.07545). Biology-framed multigenerational inheritance on a frozen base: Gilestro (2609.18560; heritable unit is trained LoRA weights). Certified admission: SEA (2607.00871), Statistical Gödel Machine (2510.10232, abstract only).
- **Shape of the position.** Elements 2, 3, 5 move from "open" to "partially open"; element 4 is crowded within one lifetime and open only at the heritability level; elements 1 and 6 are well covered as mechanisms. Round 3 lowered more threats than it raised (six medium to low; SEA "possible HIGH" withdrawn; GSE, HarnessX, Designer-RSI new medium).
- **Still absent in every work read:** (a) trait-level causal validation with held-out and regression stages as a prerequisite for heritability across agent generations; (b) a gated somatic staging store with assimilation as a discrete promotion event; (c) a typed policy + skill + regulation schema with per-trait validation records; (d) agent generations with distinct lifetimes (every "generation" read is a version/rewrite iteration of one artifact).
- **Confidence: moderate.** Two structured sweeps plus a completion pass. Gaps in 4.7 (unread leads, unreachable venues, nothing after 2026-10-12). Threat levels reflect mechanism overlap, not evidence strength (most closest works are single-seed or small-n). Adjudication is single-agent. **Not** a "first work" claim.

### 4.2 Proposed changes to §1 claim position (proposal — exact wording)

**(a) Add after "Intended claim":**

> *Narrowing (proposed 2026-10-13).* The claim is the **conjunction** only. Each part alone has published or concurrent prior art in some form: pre-persistence gating of learned skills with interventional evidence, an inheritable-artifact vs private-state boundary, probation/promotion tiers, component-typed versioned harnesses, regression-protected admission of heritable variants, and biology-framed multigenerational inheritance of acquired skills on a frozen base. What no work found does is to gate the promotion of **individually validated lifetime-acquired traits** from a **somatic staging store** into a **typed genome of policies + skills + regulation** on **trait-level interventional evidence**, and to measure the result across **agent generations with distinct lifetimes** (held-out performance, regression, genome size, descendant adaptation cost) against both unrestricted and performance-gated inheritance.

**(b) Chain-table "Status" cells (proposed):** *2* **partially open** (withdraw Round-1 "zero direct hits"; analogues: OurArk body vs private state, VaG Cold/Warm/Hot, SAPO temporary/long-term, LOGOS release candidates; in every read work the working and persisted stores coincide once a change is accepted). *3* **partially open, narrower** (typing features are individually prior art: HarnessX, GSE, SkillCommit, HDSO, DarwinX; absent: policy + skill + regulation trait types with per-trait validation records). *4* **crowded within one lifetime, open at the heritability level** (in-sample gates vs held-out gates on whole artifacts or single edits; evidence that a gate *causes* gains is thin, mostly single-run). *5* **partially open** (all read works select into one live artifact). *1* and *6*: well covered; extend the named works (4.3).

**(c) "What remains ours to defend" (proposed):** (i) a somatic staging store of lifetime-acquired traits distinct from the heritable genome, with assimilation as a discrete promotion event; (ii) the EAG as a typed, versioned heritable artifact of *functional* capabilities — the **combination** of policy / skill / regulation trait types with per-trait validation records — with gate logic held outside the artifact; (iii) the CIG as **interventional validation evidence** (paired, concurrent, temperature-0 with/without contrasts against the *current* genome on evidence not used to propose or select the trait; replay, controlled ablation, attribution under interactions, in- and out-of-scope held-out generalization, separate regression bound, abstain verdict) required before germline assimilation; (iv) measurement across **agent generations with distinct lifetimes** and frozen weights that gated inheritance beats unrestricted **and performance-gated** inheritance (a held-out gate at least as strong as SAGE or GRASP). Any one of (i)-(iii) alone is prior art in some form; per SPEC §34 none is claimed until measured.

**(d) Anti-overclaim addendum (proposed):** "causal" is already used for paired behavioural comparisons (SkillEvoReg CCV; Causal Memory Intervention 2605.17641), so "causal gate" is not new vocabulary; describe CIG attribution as *validation evidence under intervention*, distinct from the LLM failure-localisation that proposes traits. Do not call CIG the first statistically principled admission gate (SGM, SEA, SAGE precede it); do not say SEA demonstrated certified self-improvement (its certified gate accepted 0 edits); do not use single-run numbers from GSE, HarnessEvolve, SkillCommit, SkillAdam or SkillOpt as effect-size priors.

### 4.3 Most relevant new works

Threat: H high, M medium, L low. Notes are relative to `research/`; † = low-threat work included for a specific reason. "none" = abstract-level only.

| Work (id; date) | Chain 1-6 | Threat | What remains different | Note |
|---|---|---|---|---|
| Code Is the Body / OurArk (2607.28691; 2026-07-30) | P P P – P F | **H** | No causal gate (tests + human merge); private state never promoted; untyped repo; heritability unmeasured | related-work/code-is-the-body-ourark.md |
| VaG (2608.05810; 2026-08-06) | F – P P P P | M (med-high: 4, H2, H5) | No soma; ties admitted, no CI or regression set; skills not agents; one backbone/seed | related-work/vag-pre-commit-gating.md |
| GRASP (2605.29668; 2026-05-28; EMNLP 2026 per arXiv) | F – P P P P | M | One-lifetime library; in-distribution net-fix probe; no per-trait ablation or OOD stage | related-work/grasp.md |
| DarwinX (2608.07545; 2026-07-31) | P – P P P F | M | Whole-variant gate, no soma/trait ablation; held-out for reporting only (proxy overfits by 31.7) | related-work/darwinx.md |
| Gilestro (2609.18560; 2026-09-16) | P – – P P P | M (biology framing) | Heritable unit is trained LoRA weights; only gate is a declinable merge (acted as early stop) | related-work/evolution-of-sex-for-ai.md |
| SAPO (2606.08755; 2026-06-07) | F P – P P – | M | One GRPO run, weights trained; within-run probation; single-stage in-distribution gate | related-work/sapo-co-evolving-skill-generation.md |
| SkillGen (2605.10999; 2026-05-09) | F – – P P P | M | One skill per model-benchmark; regressions netted against repairs (51 vs 57 admitted) | related-work/skillgen.md |
| SkillsVote (2605.18401; 2026-05-18) | F – P P P P | M | Observational gate (LLM self-attribution); one live library | related-work/skillsvote.md |
| EvoSkill (2603.02766; 2026-03-03) | F – P P P F | M | Compares to weakest frontier member, not parent; 17-item validation; "generations" = search depth | related-work/evoskill.md |
| SkillEvoReg (2609.30861; 2026-09-25) | P – – P – P | M | Detect-and-repair, never rejects; no inheritance or assimilation | related-work/skillevoreg.md |
| HDSO (2606.22330; 2026-06-21) | F – P P P P | M (scope: EXP-0001 stage-3 design) | In/out-of-scope staged paired gate already on ALFWorld + frozen Qwen; but one deployment, add-one only, no ungated arm | related-work/round3/hdso-hypothesis-driven-skill-optimization.md |
| SEA (2607.00871; 2026-07-01) | P – P P P P | M (lower end) | Certified gate is design prior art, not demonstrated (0 edits); one agent/run, no lineage | related-work/round3/sea-anytime-valid-certificates.md |
| GSE (2608.06153; 2026-08-06) | F P P P P P | M (typed relations; interaction regression) | One bank per run, no inheritance; replay + zero-margin gate; admitted delta differs from what was checked | related-work/round3/gse-globalized-skill-evolution.md |
| HarnessX (2606.14249; 2026-06-12) | P – P P P P | M (element 3 only) | Typed runtime config, not trait types; in-sample seesaw gate lets −14/−24 pp through; no held-out | related-work/round3/harnessx-typed-harness-foundry.md |
| Designer-RSI (2609.22086; Sep 2026) | F – P P P P | M | Confound-controlled paired replay is prior art; local scope, no held-out/out-of-scope stage; one deployed bank | related-work/round3/designer-rsi-procedural-memory-replay-gate.md |
| † SAGE (2609.36043; 2026-09-28) | F – – P P P | L (statistics relevance up) | One untyped document, edit-level gate; supplies the discordant-pair test CIG should borrow | related-work/round3/sage-statistical-acceptance-gate.md |
| † SkillOpt (2605.23904; 2026-05-22) | F – P P P P | L (baseline relevance up) | Bundle-level held-out gate on one document; +19 to +25 pp from 1-4 gated edits | related-work/round3/skillopt-text-space-skill-optimizer.md |
| † SkillCommit (2608.15165; 2026-08-15) | F P P P P P | L (M for 3 sub-claims) | Abstraction tiers not heritability; source-retention replay, no no-patch control | related-work/round3/skillcommit-validated-scope-expansion.md |
| † RSEA (2606.28374; 2026-06-17) | F – P P P P | L (term collision) | "Generations" = rewrite iterations of one agent; untested whole-state keep-better | related-work/round3/rsea-held-out-selection.md |
| † Machine Zygote (2609.17300; 2026-09-15) | – P – – – – (sweep tag) | L (term hazard) | Non-LLM, one generation, no lifetime learning; germline-swap design citable | none |

Other Round-3 full reads (low; `related-work/round3/`): AHE 2604.25850, SkillBoost 2607.26643, HarnessEvolve 2609.00829, Self-Healing Harness 2609.24130 (naming hazard "B′"), SkillAdam 2609.08944. About 40 further works are low/none (candidate ledger §2.11).

### 4.4 Proposed additions to §3 must-not-claim

Not novel as standalone features (claim only the 4.2 conjunction); numbering continues the staged candidate ledger (items 9-35):

- **Gating (9-15):** pre-persistence gating of mined skills (VaG, SAPO, SkillGen, HDSO, GRASP, SkillsVote, Self-Healing Harness, SkillBoost, SAGE, HarnessEvolve, LOGOS); paired with/without held-out replay, skills as interventions, n01/n10 accounting (VaG, SAPO, SkillGen, HDSO, SkillApt, Regression Tax, ASSAY); hard regression budgets, non-inferiority, protected-set re-tests (GRASP, DarwinX, RSEA, SkillBoost, SAGE); exact paired test with abstain and anytime-valid certified admission (SAGE; SGM 2510.10232 [abstract only]; SEA) — CIG is not the first statistically principled gate; marginal-gain subset selection (VaG, DarwinX); candidate-vs-incumbent checks and regularization (SkillEvoReg, SkillBoost, SkillAdam, GSE); that unconditional accumulation is non-monotone and the gate, not skill writing, drives gains (VaG, GRASP, SAGE, Ratchet). Cite the Self-Healing Harness "55%" only as "detected conflicts" (211 of 383 replay-decided retirements).
- **Lineage, soma, biology (16-21, 28):** contamination through derived-skill lineages (VaG; Safety in Self-Evolving LLM Agent Systems; When Child Inherits); tiered quarantine/promotion (VaG, SAPO, LOGOS); inheritable-artifact vs private-state boundary, validation before birth, pull-based adoption (OurArk — claim only the gated staging store, promotion lifecycle and typed trait schema); "germline-soma", "causal heredity", germline-swap interventions (Machine Zygote); biology-framed multigenerational inheritance on a frozen base, Lamarckian framing, population-genetics vocabulary (Gilestro); preserve-and-extend variant admission (DarwinX), clade selection (HGM), cross-run meta accumulation (Hyperagents), frontier/git lineage (EvoSkill); GRN/epigenetic vocabulary (Biological Motifs).
- **Typing and schema (22, 29, 30, 32, 33):** typed harness components, contract checks, hashable configuration, change manifests (HarnessX, AHE); typed skill packages with hypothesis/evidence records, in/out-of-scope staged screening, uptake logs (HDSO); typed SKILL.md (Designer-RSI, SkillBoost); typed inter-skill relations (GSE — phrase any analogue as "measured interaction records"); validated-scope sets, commit/split/reject (SkillCommit); confound-controlled paired replay (Designer-RSI); layered NL state (RSEA, WikiSkill), typed genotypes (Geno-Synthetic), configuration-as-genome (ARTEMIS).
- **Transfer, attribution, release gates (23-27, 31, 34, 35):** held-out-gated skill retention (EvoSkill, SkillOpt, LOGOS); cross-backbone transfer and its known asymmetry (VaG, GRASP, MemCollab, EvoAgentBench, AHE, MGM, HGM) — H4 measures a known conditional phenomenon; attribution of segments / memories / skill units (SkillsVote, MeClear, CMI, SkillSV, Ratchet); release gates, gate outside the optimizer, write-time safety checks (Falsifiable Release Gates, Gate Presence, SafeEvolve); unqualified "causal" for paired comparisons (SPEC §34); that individually admitted updates jointly degrade (Self-Healing Harness, HarnessX, AHE) — CIG's joint-genome stage is *motivated by* it; whole-artifact held-out gates, keep/rollback, in-sample batch pipelines, fork-on-conflict pools (SkillOpt, RSEA, AHE, HarnessEvolve, HarnessX); abstract-level only: MOSS, Autogenesis, Self-Harness, Agent libOS, EvoGit, GEPA.

**Naming hazards to add to §1 (proposed):** "gene bank"/"offspring harnesses" (HarnessBank 2607.13683); "germline-soma", "causal heredity" (Machine Zygote 2609.17300); bare "generations" (RSEA, EvoSkill) — say "agent generations with distinct lifetimes"; "learngene" (2306.10225); "Causal Memory Intervention", "Causal Counterexample Validation"; **"arm B′"** (Self-Healing Harness uses it for immediate admission — relabel the performance-gated arm, e.g. "arm P"); "SAGE"; and vocabulary of SkillCommit (commit, source replay, scope expansion), HarnessX (seesaw, AEGIS, foundry), SEA (certificate, hold, steering adapter), HDSO (hypothesis ledger, uptake), AHE (change manifest, attribution signature).

### 4.5 Citation corrections to existing entries

21 Round-1 references were checked on arXiv abstract pages (and ACL Anthology / doi.org); no arXiv id was wrong. (The staged files say both "8 corrected" and "7 mismatches"; the table lists what was flagged.)

| Existing entry | Proposed correction |
|---|---|
| §2.1 AgentSquare 2410.06153 | Remove equal-contribution asterisks; "ICLR 2025" not shown on arXiv — confirm on OpenReview (SPEC §3 states it too). |
| §2.3 MGM 2608.07645 | Abstract has no numbers; mark body-level figures as not re-verified. |
| §2.4 Genomebook (bioRxiv 10.64898/2026.03.22.713494) | Unverifiable (HTTP 429); mark details as carried from Round 1; re-check before citing. |
| §2.5 SkillRL 2602.08234 | WebShop 85.2 is a score; success is 72.7%. NeurIPS 2026 only in arXiv Comments — write "per arXiv comment". |
| §2.7 CapaBench 2502.00510 | Cite ACL title + arXiv id; current arXiv title "Understanding and Optimizing Agentic Workflows via Shapley value"; "Who's the MVP?" v1 title and "renamed in v3" unverified (v3 dated 4 Nov 2025); 24-vs-16 coalition inconsistency. |
| §2.9 SkillForge 2608.24747 | Add missing title; note RL-agent scope. |
| §2.9 ASSAY 2606.15390 | Subtitle "…Agent Knowledge"; "randomized masking" unconfirmed from abstract (the stage-2 adoption suggestion rests on it). |
| §2.9 Skill-α 2608.01678 | "Causal per-edit gate" → "counterfactual per-edit scoring". |
| §2.9 AgentSpawn 2602.07072; LOGOS 2607.10878 | AgentSpawn: full title, sole author, cs.SE scope. LOGOS: "held-out execution evidence" is our paraphrase of "test evidence". |
| §2.9 OMEGA/RECLAIM 2605.25062; Survey 2507.21046 | Add titles and (OMEGA) sole author; Baldwin-effect section unconfirmed from abstract. |
| `related-work/skillrl.md` D2Skill 2603.28716 | Unverifiable; "privileged-teacher reliance" flagged. |
| Verified | DGM, ShapleyFlow, CASCADE, MUSE-Autoskill ("Under Review"), SkillShapley (add full author list), CapaBench ACL DOI; Voyager GPT-4/self-verification not re-checked. |

### 4.6 Experiment-design implications (proposals only; nothing changed)

EXP-0001 (ADR-0002) compares A (none) / B (unrestricted persistence) / C (CIG-gated); the reviewer-natural comparison is no longer B vs C alone.

1. **Performance-gated arm between B and C — requires ADR (amend ADR-0002).** Else any C > B is attributable to "any held-out gate" (SkillOpt: +19 to +25 pp from 1-4 gated edits). Ladder: held-out principled (SAGE-style Δ_λ > 0 ∧ p_λ < α ∧ ρ ≤ τ; GRASP-style net fix − regression > 0 ∧ R(c) ≤ R0); whole-artifact held-out (SkillOpt, RSEA, VaG add-one + marginal gain); staged in/out-of-scope (HDSO-style, about 124 vs 130 episodes per candidate); CIG-stage subsets; in-sample gates (never the sole arm); certified (secondary only). Proposed: one pre-registered held-out primary; freeze the K = 10 candidate stream and apply every rung offline as a secondary "gate ladder"; identical Trait Miner across arms; per-trait vs per-bundle stated, with a bundle "hitchhiker" analysis. Relabel "arm B′" (4.4).
2. **Acceptance-rate-matched random-accept control** (Gilestro: the gate acted as an early stop) — requires ADR if an arm; else secondary analysis.
3. **Observational-attribution comparison** (SkillsVote vs CIG verdicts on frozen candidates) — pre-registration note, no new arm.
4. **Gate statistics** — requires ADR for threshold/test changes. Stage-3 "Δ ≥ 0" admits ties (as VaG, RSEA, HarnessEvolve, GSE) while SkillOpt and HDSO reject them: abstain at m = 0 / exact discordant-pair test, or pre-register ≥ as a deliberate non-inferiority margin. Report (w, l, m) per trait per stage. At least 5 discordant pairs are needed for α = 0.05, so at 20-40 episodes most candidates will abstain: pre-register a minimum detectable effect, add an abstain verdict, report "rejected" separately from "abstained", pre-register multiplicity (K = 10 × 4 stages); stage-4 comparator must be a concurrent re-run of the current genome.
5. **Length / prompt-size control** (Regression Tax "osmosis", SkillSV padding) — secondary analysis; ADR only as an arm.
6. **Gate outside the genome** (Gate Presence; SEA; AHE) — requires ADR when populations are introduced; EXP-0001 v0 already complies.
7. **Distillation-context provenance in the Lineage Graph** (VaG) — requires ADR-0001 amendment. 8. **Convention-conflict probes** (Gilestro) — requires a later EAG-schema ADR.
9. **Population experiments (H6):** control for pass@k and agent count (Morphogenetic Harness); "single agent taught the whole curriculum" baseline — requires ADR (not EXP-0001).
10. **H4 framing:** transfer is asymmetric and interface-dependent (GRASP, MemCollab, EvoAgentBench, AHE, HDSO); typed interface/operating-point preconditions; model-conditional held-out stage — requires a later ADR (SPEC §19).
11. **Joint-genome check before committing G1** (Self-Healing Harness RQ4; HarnessX −14/−24 pp): G0 ∪ admitted traits on T_reg and T_gate, a failure rule, cumulative regression budget vs the founding ancestor — requires ADR.
12. **Placebo traits** (length-matched, irrelevant) in arm C's candidate set to measure the false-pass rate — requires ADR.
13. **No in-sample evidence:** mining episodes not reused as stage-2/3 evidence; stage-1 replay paired with/without — requires ADR if not already specified.
14. **Leakage / bloat hygiene filter identical across arms** (HarnessEvolve) — requires ADR if part of CIG; pre-register either way.
15. **Delivery-interface confound** (HDSO Table 3, my inference): identical delivery across B, performance-gated arm and C; optional progressive-disclosure B — requires ADR to add an arm.
16. **Composite candidates:** no co-shipped traits in one evaluation unit unless factorial/leave-one-out; consider group ablation (RSEA Table 5) — requires ADR.
17. **Expression precondition and expression-conditioned regression** (HarnessX, HDSO, GSE) — requires ADR as a hard gate condition; secondary analysis otherwise.
18. **CIG record / EAG schema additions** (falsification_plan, rollback_condition, uptake, reject-reason taxonomy, abstain verdict, scope + exclusion, measured interaction records) — requires ADR-0001 amendment.
19. **Optimizer / somatic memory as an uncontrolled inheritance channel** (SkillAdam, SkillOpt, HDSO): pre-register whether descendants can read ancestors' rejection records; constant across arms — requires ADR clarification.
20. **Judge separation** (CIG judge vs held-out evaluation judge) — requires ADR if an LLM judge is used.
21. **Reporting** (no ADR): overfitting gap per arm and generation; final-generation numbers, never peak; multi-seed noise floor; paired-delta distribution; gate and adaptation cost in tokens; re-run SkillOpt/SkillAdam comparators rather than copying numbers.

### 4.7 Residual queue and unreachable sources

- **Round-4 queue (title/abstract only):** SGM 2510.10232 (origin of SEA's gate; cite in must-not-claim 12); SkillCAT 2606.13317; SkillOS 2605.06614; SkillEvolver 2605.10500; SkillAxe 2606.10546; CoEvoSkills 2604.01687. **Re-check:** SelfEvoSkill 2606.14239 (Designer-RSI cites it as "SkillAudit"; cite by id plus current title).
- **Could rise on a full read:** HarnessBank 2607.13683 (medium if its gate is evidence-based); WikiSkill 2608.27454 (gate rule secondhand); Autogenesis 2604.15034 and Self-Harness 2606.09498 (abstract-level; most likely to show a stronger gate). Ten further priors were triaged on abstracts only (candidate ledger §2.11).
- **Next sweep:** re-run the four Round-2 query families with date ≥ 2026-10-12 before any submission.
- **Unreachable / unverifiable:** OpenReview ICLR 2027; NeurIPS 2026 accepted list; Semantic Scholar; Google Scholar; AAAI-26 Learngene pages (502); bioRxiv (429; Genomebook); github.com/yccm/SkillGen (403); SkillAdam and SkillOpt code; the methods-audit preprint (non-standard id — do not cite); authors of 2606.23075; dates of 2607.04240 and 2609.29543; SkillCommit appendices A/B; SAGE hyperparameters; HarnessEvolve split sizes; GSE replay metric; Designer-RSI tie band; SEA's Alg 4 in the full stack.
- **Process caveats:** adjudication is single-agent; Zcode/Codex communications were not tracked, so repository ledger edits after 2026-10-04 must be merged before adoption.

### 4.8 Pointers

- **Full candidate ledger** (complete §1-§5, 21 Round-2/3 entries in SPEC §30 format, must-not-claim items 9-35, Round-3 corrections, full design implications): `research/related-work/novelty-ledger-round3-candidate.md`.
- **Sweep report** (query log, 49-work table, adjudication, deep-read summaries, Round-3 addendum): `research/related-work/round2-sweep-2026-10-12.md`.
- **Per-paper notes:** `research/related-work/<name>.md` and `research/related-work/round3/<name>.md`, as cited in 4.3.
