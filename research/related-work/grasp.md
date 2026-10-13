# GRASP — detailed notes

**Paper:** *GRASP: Gated Regression-Aware Skill Proposer for Self-Improving LLM Agents*. Authors: Johannes Moll, Jean-Philippe Corbeil, Jiazhen Pan, Martin Hadamitzky, Daniel Rueckert, Lisa Adams, Keno Bressem. arXiv:2605.29668. Versions: v1 28 May 2026, v2 14 Aug 2026, v3 21 Aug 2026 (current). The arXiv comment says **"Accepted at EMNLP 2026 (Main Conference)"** and points to code/data at github.com/jomoll/GRASP. I did not open the repository. Subjects: cs.AI, cs.CL. Reviewed 2026-10-12.

**Read depth:** I read the full text of the arXiv HTML v3 (main body plus appendices) in two passes through a fetch/summarise tool, so I worked from structured extractions and not from my own line-by-line reading. Two consequences:
- Some exact values come from the extractions and should be checked against the PDF before anyone quotes them. These are the formal statement of the regression budget (Eq. 1) and the λ weighting.
- One extraction said Appendix E.3 gives "no numeric value" for the regression budget. The §2.3 definition is a *relative* constraint (no regressions beyond the current library's count), so it has no free numeric threshold.

## Mechanism

- **Setting:** prompt-level self-improvement with **frozen weights**. No parameters are updated (§3.4). The same model acts as executor and as skill writer within a run. Training runs 5 epochs over a dev set, batch size 48, and keeps the checkpoint with the best validation accuracy.
- **Skill representation (§2.1, App. B.4):**
  - Each skill is a Markdown file with YAML frontmatter: `name`, `description`, `tags`, `version`, and `provenance` (epoch, update cycle, action ∈ {ADD, MODIFY, REMOVE}, probe score).
  - The body holds a trigger condition, a behavioural rule, an optional verification step for schema-constrained actions, and a contrastive wrong-vs-correct example plus pitfalls.
  - The library starts empty and is **capacity-bounded** (≤10 skills in Table 7). An ADD at capacity is allowed only together with a REMOVE.
  - The authors call the library "auditable", "versioned and reversible". Versioning is metadata only: I found no rollback procedure or lineage tracking beyond picking the best checkpoint.
- **Failure-driven proposals (§2.2):**
  1. An LLM gives each failing trace an open-vocabulary failure-mode label.
  2. Traces are grouped by label.
  3. The writer produces up to K=4 single-skill edits (ADD / MODIFY / REMOVE), cycling from the largest group to the smallest. Each call sees that group's failing traces, sampled passing traces, a summary of the active skills, and the other failure labels. The paper calls this "comparative proposal generation".
- **Acceptance gate (§2.3, Eq. 1, Alg. 1):**
  - *Probe:* N=36 dev samples. Half were previously failing and half previously passing, stratified by task type, and drawn from earlier dev episodes. The probe is out-of-sample relative to the batch that produced the proposals.
  - *Baseline:* the current library is re-run on the probe to get fresh fix and regression counts F0 and R0.
  - *Score:* a candidate c is scored (F(c) − F0) − (R(c) − R0). It is accepted only if the score is > 0 **and** the hard regression budget R(c) ≤ R0 holds.
  - Invalid-action regressions get weight λ=2 in the score but are counted unweighted in the budget.
  - Only the highest-scoring passing candidate is applied; otherwise the library stays unchanged.
  - If the winner causes any regression, a contrastive revision may narrow its trigger. The revision replaces the winner only if it scores strictly higher and still meets the budget.

## Results

**Base models:** gpt-oss-120b, DeepSeek V4 Flash, Gemini 3.1 Flash Lite, GPT-4.1, GPT-5.4 (low effort).

**MedAgentBench test (Table 1):**

| Model | Before | After GRASP |
|---|---|---|
| gpt-oss-120b | 40.6 | 88.8 |
| DeepSeek V4 Flash | 47.7 | 70.0 |
| Gemini 3.1 Flash Lite | 54.2 | 71.4 |
| GPT-4.1 | 45.5 | 84.9 |
| GPT-5.4 (low effort) | 45.1 | 85.4 |

For gpt-oss-120b, the best baseline (Evo-MedAgent) reaches 67.8.

**Other benchmarks:**
- MedAgentBench-v2 (gpt-oss-120b): 61.1 → 76.9. The authors make no superiority claim on the v2 OOD split, where GRASP is 1.1 points below SkillX and the CI includes zero.
- FHIR-AgentBench: 51.5 → 58.2 (Δ +6.8, 95% CI [3.7, 10.0]).
- Non-clinical (Table 6, gpt-oss-120b): ALFWorld 23.3 → 51.7, WebShop 20.7 → 41.3, DBBench 65.6 → 70.6, OS Interaction 48.6 → 49.5 (within noise).

**Ablations (Table 4, MedAgentBench, gpt-oss-120b, test):**

| Variant | Test |
|---|---|
| Full GRASP | 88.8 |
| w/o failure grouping | 84.4 |
| w/o regression budget | 81.8 |
| Fixes-only selection | 80.2 |
| Append-only (ADD only) | 80.2 |
| No gate, K=4 | 63.5 |
| No gate, K=1 | 40.1 |
| Matched compute, proposer-selected | 70.8 |
| Matched compute, random-selected | 67.2 |

The paper concludes that the gains come from the gate and the regression budget, not from skill writing alone.

**Gating the baselines (Table 5, §4.6):** adding the same gate to all five baselines raises their in-domain accuracy but not their OOD accuracy. Their OOD change is about zero on average, while GRASP gets +23.7 OOD.

**Transfer of frozen libraries (Tables 2–3):**
- *Cross-model:* a library written by GPT-5.4 and run by a weaker executor raises OOD (gpt-oss 56.3 → 77.8; Gemini 41.7 → 71.1). On the gpt-oss test split it does worse than gpt-oss's own library (88.8 → 76.0). Transfer is asymmetric: libraries from weaker writers run by stronger executors do worse than self-training.
- *Cross-benchmark:*
  - MedAgentBench → v2 gives 78.1 test, against 76.9 for a library trained on v2 itself.
  - Every library moved into FHIR-AgentBench pushes it **below** its 51.5 baseline (47.2 / 45.7 / 48.3).
  - A DBBench control library is neutral or harmful on clinical targets.
  - The paper explains this as transfer holding only across a shared tool-calling convention.

**Stated limitations:**
- The benchmarks measure procedural reliability, not clinical correctness.
- The probe is the main cost (~411 agent calls per batch).
- Compute is a possible confound against unprobed baselines. The authors address it with matched-compute ablations, but the gated-baseline experiment covers only MedAgentBench with gpt-oss-120b.
- The paper offers no way to detect writer–executor mismatch.
- The method does not help in open-ended action spaces with diffuse consequences (OS Interaction). It needs recurring failure modes plus a verifiable signal the probe can use.
- Three models are accessed through provider APIs that may change without notice.

## Overlap vs HeritAgent chain

| Element | Coverage |
|---|---|
| 1. Lifetime acquisition from trajectories | **Full.** Skills are written from labelled failing traces contrasted with passing traces, all within one run. |
| 2. Somatic/germline separation | **Absent.** There is one library. An accepted edit goes straight into the agent's working library; there is no separate heritable store and no candidate/somatic tier. |
| 3. Typed executable genome | **Partial (weak).** Each skill is a structured artifact: YAML frontmatter, trigger ≈ regulation, version and provenance fields, a capacity bound. But skills are untyped natural-language Markdown rules, not executable code. There is no typed whole-agent spec covering policies, skills and regulation, and versioning is metadata without lineage or rollback machinery. |
| 4. Causal validation gate | **Partial (strong partial).** Each candidate is a single-skill edit, evaluated as an intervention: library + edit against the re-run current library on a held-out, balanced probe, with a hard regression budget. This matches CIG's held-out-generalization and regression-protection stages and τ_r in spirit, and it is an interventional before/after comparison per edit. **Missing:** controlled ablation of a skill *inside* the evolved library after later edits; attribution under skill interactions; replay across multiple environments; an OOD check inside the gate (Table 5 shows the in-domain probe gate does not secure OOD transfer for other methods); and any use of this evidence as a condition for *heritability*. Causal claims are at the level of mechanism components; per-skill causal attribution is explicitly not done. |
| 5. Selective assimilation | **Partial.** The gate is selective: only the best passing candidate is applied, and MODIFY/REMOVE are first-class. But this is selective *persistence into the lifetime library*, not assimilation into a separate heritable genome. |
| 6. Multi-generational, frozen weights | **Partial.** Weights are frozen (full on that half). There are no generations, lineages or descendants. Moving a frozen library to another backbone or benchmark is a one-shot transfer test, not inheritance by an agent that keeps learning and passes the library on. |

## Why this matters for the novelty position

- GRASP is the cleanest **peer-reviewed** (EMNLP 2026 Main, per the arXiv comment) instance of "a learned skill persists only if it shows net held-out gain within a hard regression budget". Round 1 listed skill gates only as LOGOS (governance-flavoured), Skill-α (rollback reward) and ASSAY (offline masking). GRASP is stronger prior art than any of these for the specific mechanism *held-out probe + regression budget → accept/reject a skill edit*.
- So HeritAgent must **not** claim as novel:
  - gating skill-library edits on held-out evidence;
  - regression budgets or regression-aware acceptance for agent skills;
  - showing that the gate, not skill writing, drives the gains;
  - showing that frozen skill libraries transfer across backbones.
- What remains different, and must be argued and shown empirically rather than asserted:
  1. CIG gates **heritability** (somatic → germline), not lifetime library edits.
  2. CIG asks for **trait-level** controlled ablation and attribution under interaction, where GRASP uses a library-level net count.
  3. CIG adds multi-environment replay and an explicit generalization stage. GRASP's own Table 5 shows that in-domain probe gating does not secure OOD gains for other methods, which motivates a separate OOD stage.
  4. HeritAgent has a typed EAG and multi-generational descendants with adaptation cost as an outcome.
- **H4 relevance (cross-backbone transfer of the inherited genome):**
  - GRASP gives direct prior evidence that transfer is **asymmetric** (strong writer → weak executor helps; the reverse hurts) and **interface-dependent**: every library moved into FHIR-AgentBench degraded it.
  - H4 must be framed as testing these known failure modes, not as discovering transfer. The EAG expression layer should make interface or tool-calling assumptions explicit, for example as typed preconditions.
  - SkillRL's "no cold-start SFT → skills transfer poorly" result points the same way.

## Consequences for HeritAgent

- **Ledger:**
  - Add GRASP to ledger §2, near the LOGOS / Skill-α / ASSAY entries.
  - Downgrade the §1 wording on causal validation. "Promotion gates exist but governance-flavored (LOGOS)" is now incomplete, because a non-governance, regression-aware held-out skill gate exists in peer-reviewed form.
  - The "partially open" status still holds, but only for *trait-level causal evidence as a prerequisite for germline assimilation across generations*.
- **§3 must-not-claim list:** add "held-out, regression-budgeted acceptance of skill-library edits; frozen skill-library transfer across backbones (GRASP)".
- **EXP design:**
  - GRASP's gate is a natural **strong baseline arm** between EXP-0001 arm B (unrestricted inheritance) and full CIG. The question becomes: does trait-level CIG beat a GRASP-style net-gain + regression-budget gate on held-out performance, regression rate, genome size and descendant adaptation cost?
  - Without that arm, a reviewer can reasonably ask whether CIG's gains are just "a held-out gate". GRASP's own ablations show that a gate alone accounts for most of the effect within one lifetime.
- **CIG design:**
  - Borrow the probe construction (balanced previously-failing / previously-passing, stratified by task type, re-run baseline counts) for the held-out and regression stages, and cite it.
  - Record probe cost (~411 calls per batch in GRASP) as a budget reference for SPEC §9 stage costs.
- **Follow-up leads:** GRASP compares against Evo-MedAgent, SkillX and SAGE (skill/memory baselines). I did not open these papers, so they are unverified here. They are candidates for the next sweep pass.
- **Anti-overclaim (SPEC §34):** GRASP calls its library "versioned and reversible" and says "the library evolves rather than grows". Do not let this vocabulary blur the line: GRASP has no generations, no germline and no genome, and HeritAgent's distinction rests on those plus the inheritance-prerequisite semantics.
