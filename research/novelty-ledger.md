# Novelty Ledger

Living document per SPEC §30. Every relevant paper is recorded here with: paper / date / core mechanism / overlap with HeritAgent / what remains different / required architecture changes. Update whenever a new paper appears; adjust the novelty claim if needed.

- **Last updated:** 2026-10-04 (Round 1 + issue #9 miner updates: SkillRL NeurIPS 2026 status, SkillForge 2608.24747, Agent Skills Can Be Harmful 2608.11888; see `related-work/` for detailed notes)
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
