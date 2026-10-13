# SkillCommit — Behaviorally Validated Scope Expansion — detailed notes (Round 3)

**Paper:** *SkillCommit: Evolving Agent Skills through Behaviorally Validated Scope Expansion*. Yu He (Nanjing University) and Weikai Yang (HKUST (Guangzhou); corresponding author). arXiv:2608.15165v1 [cs.AI], submitted 15 Aug 2026, arXiv perpetual non-exclusive licence. Only v1 exists. No venue, comments field or code link was found, so it is a preprint. The PDF is 9 pages in AAAI-style two-column format. Reviewed 2026-10-13 by Claude (third agent), pending Research/Critic cross-review.

**Read depth: full text (main body).** Sources:
- I downloaded arxiv.org/html/2608.15165 and converted it to text, keeping the LaTeX alttext of formulas. I read every section (Introduction → Conclusion) and the reference list.
- I also downloaded arxiv.org/pdf/2608.15165 (9 pages) and checked it against the HTML.
- The body points to **Appendix A** (prompt templates) and **Appendix B** (γ, L, m per dataset, generation settings, method budgets). **Neither appendix is in v1**: the HTML and the PDF both end at the references. These hyperparameters are therefore unverifiable.

Arithmetic checked against Table 1 and §Main Results: all of the following match what the paper states.
- No-Skill mean 50.80 → SkillCommit mean 78.15.
- Margins over the best baseline: +14.48, +28.08 and +28.01.
- 31 of the 72 baseline cells fall below No Skill and 6 equal it.
- Figure 4 four-task averages: 48.97, 75.25 and 69.75.
- The Figure 3 help/hurt counts reproduce the Table 1 Airline and NBA accuracies exactly. For example, NBA No Skill is 13/45; adding 21 helps and subtracting 2 hurts gives 32/45 = 71.11.

## Mechanism

The paper frames the method as "skill-library construction as a sequence of behaviorally validated scope expansions". Three stages run online over an *induction* split. Executor and teacher weights are frozen throughout.

1. **Feedback-Guided Patch Induction.**
   - When the executor fails on a source instance x_i, a teacher sees x_i, "the verified ground-truth answer" and the failed trajectory τ_i⁻. It then writes a reference trajectory τ_i⁺ and a candidate patch c_i containing "behavioral constraints, decision rules, or checkable intermediate steps".
   - The teacher may be the same model as the executor, with a separate prompt and privileged feedback. Patches that contain the final answer are discarded.
   - **Source-replay validation:** the executor is re-run on x_i with only c_i. If it now solves x_i, c_i is admitted as a "source-validated instance-specific patch". If not, the author is given the replay and the verifier feedback and tries again, for up to K = 3 attempts.
   - Reported yield: "88.0% of failed instances yield a validated patch within 3 attempts".
2. **Behavioral Compatibility Grouping.** Embeddings are used only to retrieve candidates (top R = 4 active skills).
   - **Expand an existing skill p.** Replay p on the new source x_t, and replay the new patch q_t on p's covered instances W(p). The compatibility score is C(p, q_t) = [ρ(p, x_t) + Σ_{x_j ∈ W(p)} ρ(q_t, x_j)] / (1 + |W(p)|), and the pair passes if C ≥ γ. An LLM "pattern author" then checks whether the two share "a shared procedural mechanism".
   - **Form a new skill.** Among residual (unabsorbed) patches, find an *anchor* patch whose procedure transfers to the source instances of every other group member. The group must have at least m members (m = 2 or 3) and must include q_t. Reverse transfer is an optional stricter rule. The group with the broadest coverage wins.
3. **Source-Preserving Consolidation.**
   - The pattern author induces a candidate skill with the fields **`when` / `repair` / `avoid_when`** (applicability, procedure, exclusion). It sees only the anonymised patch texts, with no questions, answers or provenance.
   - **Full-scope source replay** runs the candidate on every source instance in its claimed scope S. Refinement is allowed for up to L attempts. "All candidate versions together with their replay outcomes are retained for auditing".
   - Three outcomes follow:
     - **Commit** if the candidate succeeds on every source. An expansion becomes a new active version, and previous versions are kept "for traceability".
     - **Split** if the best version succeeds on at least m sources. Its scope is narrowed to those sources and it is re-validated once.
     - **Reject** otherwise. The hierarchy is unchanged, and the patches "remain available as instance-specific skills".
   - "SkillCommit uses source replay as the sole criterion for consolidation."

**Not specified in v1:**
- γ, L, per-dataset m and generation settings (all deferred to the missing App. B);
- how skills are retrieved and injected at *test* time;
- library sizes or token counts. "Compact" is claimed throughout and never measured;
- seeds or repeated runs, CIs or significance tests;
- any **no-patch control replay**;
- a limitations section;
- code.

**Vocabulary:** a text search finds no "causal", "inherit", "lineage", "germline", "genome" or "population". "Generation" appears only for patch generation. "Commit" is used in the version-control sense, for activating a validated abstraction.

## Results (as reported; single run, point estimates)

- **Splits.** Each benchmark is split into induction and held-out test cases:
  - RuleArena Airline: 100 / 100;
  - RuleArena NBA: 170 / 45;
  - OpenExempt EV and EC: 210 / 105 each;
  - KOR-Bench Logic and Cipher: 175 / 75 each.
- **Metrics.** Strict accuracy, except macro-F1 for OpenExempt.
- **Configurations.** Executors are DeepSeek-V4-Flash and Qwen3.6-35B-A3B in matched author = user settings. Two transfer settings use a Qwen3.5-9B user, with Qwen3.6-35B-A3B or GPT-5.4 as author.
- **Baselines.** No Skill, ExpeL, ReasoningBank, ACE and Trace2Skill, "adapted to the same benchmark interface and induction data".
- **Table 1.** SkillCommit is best in all 18 task × configuration cells. Six-task average gains over No Skill are +19.33 (DeepSeek) and +30.02 (Qwen-35B).
  - The baselines are frequently harmful: 31/72 of their cells are below No Skill.
  - Some baseline numbers are strikingly low. For example, ACE scores 6.67 on NBA, and ExpeL scores 7.76 on EV with Qwen-35B. These results may reflect the adaptation to a shared interface rather than the methods themselves.
- **Figure 3 help/hurt (DeepSeek, Airline and NBA).**
  - SkillCommit: 24 helps / 0 hurts (Airline) and 21 / 2 (NBA).
  - ExpeL: 24 / 8 and 14 / 3.
  - ACE: 3 / 24 and 3 / 13.
  - Trace2Skill: 2 / 24 and 7 / 7.
  - For SkillCommit vs No Skill these counts are significant by exact sign test (my computation: NBA 21 vs 2, p ≈ 3e-5; Airline 24 vs 0, p ≈ 6e-8). This is the paper's strongest evidence.
- **Transfer (Figure 4, 4 tasks).** The Qwen-9B user rises from 48.97 to 75.25 with Qwen-35B-authored skills and to 69.75 with GPT-5.4-authored skills. The library is transferred frozen, without re-validation on the recipient.
- **Ablation (Table 2, RuleArena NBA only; n = 45, so 1 case = 2.22 pp).**
  - Full: 71.11. Without patch induction: 53.33. Without grouping: 66.67. Without consolidation: 66.67.
  - Patch-induction-only: 64.44. Grouping-only: 60.00. Consolidation-only: 57.78.
  - The −4.44 pp drops for grouping and consolidation are **2 test cases each**, one run, no CI. They cannot be separated from noise (a 2–0 discordance gives an exact p = 0.25).
  - So the paper's own ablation shows that most of the gain comes from **source-replay admission of single patches**. The hierarchical abstraction and the commit gate, which are the paper's headline, have a small measured marginal effect that is not statistically supported.

## Overlap vs HeritAgent chain (full-text judgement)

| Element | Coverage | Evidence |
|---|---|---|
| 1. Lifetime acquisition | **Full** (with a caveat) | Online, failure-triggered patch induction from the agent's own trajectories plus verifier feedback (§Feedback-Guided Patch Induction). Caveat: acquisition uses a teacher with the **ground-truth answer** (privileged supervision), and it runs on a separate induction split rather than during deployment. |
| 2. Somatic / germline separation | **Partial (weak)** | There are two tiers: source-validated *residual instance patches* and committed *higher-level skills*. A patch is promoted (absorbed) only through grouping plus full-scope replay. But the tiers differ in *abstraction level*, not in heritability. Residual patches are admitted and "remain available as instance-specific skills", so the lower tier is not a quarantined staging store. There is no notion of a private versus an inheritable state. The ablation "deploying source-validated instance patches directly" shows that the lower tier is itself expressed. |
| 3. Typed executable genome | **Partial** | Each committed skill has a fixed schema `when` / `repair` / `avoid_when`. The `avoid_when` exclusion condition is a regulation-like field. Each skill also has an explicit validated scope W(p), version history with previous versions preserved, and an audit log of every candidate version with its replay outcomes. This is the closest analogue I have seen in this cluster to EAG's per-trait validation records. But there is a single trait type (procedural guidance text), with no policy/skill/regulation typing, no executable code and no whole-agent specification. The test-time retrieval and expression mechanism is not described. |
| 4. Causal validation gate | **Partial** | It is a three-stage, replay-based gate that is a *prerequisite* for admission and commit: (i) source replay (the patch must make the frozen executor solve its source); (ii) directed cross-instance replay with a threshold γ plus an LLM mechanism check; (iii) full-scope source replay with commit, split or reject. Stage (iii) is in-scope regression protection, and stage (ii) is a crude within-induction transfer test. Against CIG it falls short on every evidential axis: (a) it is not interventional in the paired sense. There is no with/without contrast beyond "previously failed, now succeeds" on **one** stochastic replay. There is no no-patch control replay, and selecting instances on failure invites regression-to-the-mean false admissions. (b) There is no ablation of a skill inside the library context and no attribution under interactions. (c) There is no held-out data inside the gate, since all replays use induction instances, and no out-of-scope probe: `avoid_when` is written by the author, never tested. (d) There is no regression check outside a skill's own source set when it is committed. (e) Outcomes are binary with no statistics and no repeated sampling. It operates within one lifetime, for library persistence, never for heritability. |
| 5. Selective assimilation | **Partial** | Commit, split and reject is a selective, evidence-gated consolidation into the *active* library, and rejected abstractions leave the hierarchy unchanged. But the target is the live working library, not a heritable genome distinct from lifetime state. Rejected material stays usable as residual patches, so this is a reorganisation event rather than an assimilation event between two stores. |
| 6. Multi-generational, frozen weights | **Partial** (frozen half full; generational half absent) | "All model parameters remain frozen throughout skill construction and evaluation." There are no generations, descendants or lineages. The only inheritance-like event is a one-shot frozen transfer of a finished library to a different, smaller executor, with no re-validation on the recipient and no recipient lifetime. This is the cross-backbone transfer already conceded in ledger §3 item 24. |

**Implements full chain: no.** One element is full. Five are partial, two of them weak (2, and the generational half of 6). Compared with SAGE (two absent), SkillCommit touches more elements. That is mainly because of its two-tier hierarchy and its versioned, scope-annotated, audited skill records.

## Does the full text change the abstract-level assessment?

There was no §2.10 entry. The only prior assessment is the line in the §2.10.10 SkillEvoReg entry (repeated in §2.12): "validates abstractions before commitment — most likely commit-gate competitor" (title confirmed, not read). That line implies roughly MEDIUM for element 4.

**Threat to the chain claim: settles at LOW. It stays MEDIUM for three narrow sub-claims.**

- **Why lower than "most likely commit-gate competitor".**
  - The "commit" gate is *source-retention replay*. It checks only that an abstraction still solves the induction instances it was built from, on one binary replay each. It is a sufficiency check, not interventional, trait-level or held-out evidence, and it never decides heritability.
  - The paper's own ablation attributes most of the gain to single-patch source-replay admission, an idea that VaG, SkillGen and SkillCAT already cover. The commit gate contributes +4.44 pp on 45 items (2 cases, one run).
  - There is no inheritance, no genome distinct from the live library, and no generations.
  - Key details (γ, L, m, generation settings and both appendices) are missing from v1. The claimed compactness is never measured.
- **Fair the other way: points a reviewer could legitimately raise.**
  - (i) **Staged promotion with scope as a first-class, replay-validated property.** Instance patch → group → committed abstraction, with commit, split and reject, and full version and audit history. This is structurally close to "somatic store → gate → germline" plus "per-trait validation records". It extends ledger §3 items 14, 17 and 22.
  - (ii) **An ordered multi-stage replay pipeline** (local validity → cross-instance transfer → in-scope regression protection) as a prerequisite for commit. This resembles the CIG stage ordering (replay → … → held-out → regression), though every stage is weaker.
  - (iii) **A typed exclusion field (`avoid_when`)** as a regulation-like part of each skill.
  - (iv) **Help/hurt evidence** that ungated experience methods overwrite correct behaviour (ACE 3/24) while replay-gated admission does not (24/0). This is direct, statistically strong support for H2's premise *within one lifetime*, which ledger §3 item 15 already concedes.
  - HeritAgent must therefore not describe "promotion of validated instance-level traits into a higher-level, versioned, scope-annotated store with commit/split/reject" as new. It can claim only the conjunction: a *heritability* gate on trait-level interventional evidence across generations.

## Consequences for HeritAgent

No design was changed. Items marked *requires ADR* are for the Builder/ADR owner.

1. **Arm B′ option (f): "SkillCommit-style replay-gated" inheritance. *Requires ADR (ADR-0002 amendment).***
   - A candidate trait is admitted to the genome iff it (i) makes the frozen executor solve its own source episode(s) on replay and (ii) still solves every source episode in its declared scope after any merge or abstraction.
   - There is no held-out stage, no ablation in genome context, no attribution and no out-of-scope probe.
   - This is a cheap, published, non-statistical "replay-gated" baseline that sits between unrestricted inheritance (B) and the full CIG (C). It complements the SAGE-style option (e) and ledger §5 item 1 (a)–(d).
   - If C beats a SAGE-style B′ but not a SkillCommit-style B′, or the reverse, that would localise which CIG stage carries the effect.
2. **No-trait control replay in CIG stage 1. *Requires ADR if not already specified.***
   - SkillCommit admits a patch when one replay *with* the patch succeeds on an instance the executor previously failed. There is no replay *without* the patch, so stochastic flips and regression to the mean (the instance was selected *because* it failed) are counted as validation.
   - CIG stage 1 (replay) should pre-register paired with/without replays on the same seeds, or k ≥ 2 samples per arm, and should not condition the baseline arm on the observed failure.
   - Report per-trait (w, ℓ, m) as in the SAGE note, item 2.
3. **Scope as a declared, tested trait attribute. *Requires ADR (schema change).***
   - SkillCommit's W(p) (validated scope) plus `avoid_when` is a useful design pattern. If the EAG trait schema does not already carry a declared applicability scope and an exclusion condition, consider adding them.
   - CIG's out-of-scope held-out probe should then *test* the declared exclusion. SkillCommit never tests it, which is a concrete point of difference worth measuring.
4. **Split as a third gate outcome. *Requires ADR.*** SkillCommit has three outcomes: commit, split (narrow the scope to the supported subset and re-validate) and reject. CIG appears to be binary (assimilate or not). A "narrow-scope assimilation" outcome would change genome-size accounting. If adopted, it must be counted separately in H-genome-size and regression metrics, and the re-validation must be on fresh items, or the narrowing step overfits the gate.
5. **Statistics and power.**
   - SkillCommit's ablation differences are 2/45 cases from one run. This is the same under-powering risk flagged for EXP-0001 (SAGE note, item 5).
   - EXP-0001 should not report stage-level ablations of the CIG on test sets of this size without exact paired tests and multiple seeds. Pre-registering sizes and tests *requires ADR*.
6. **Privileged teacher as a confound for descendant adaptation cost.** SkillCommit's acquisition uses the ground-truth answer. If any HeritAgent acquisition path uses verifier or ground-truth access, descendant adaptation-cost comparisons across arms must hold that access constant. This is a design note; it needs an ADR only if the acquisition interface changes.
7. **Wording and naming.**
   - Avoid "commit", "source replay", "scope expansion" and "split" as names for CIG stages or EAG operations without citing SkillCommit.
   - Do not call "patch-level replay validation" or "hierarchical consolidation under replay" HeritAgent contributions.
   - When contrasting, say: "within-lifetime consolidation of a single live skill library, gated by source-retention replay on induction instances; no held-out, interventional or heritability gate".
8. **Round-4 queue.** SkillCAT (arXiv:2606.13317, *Contrastive, Assessment-Augmented and Topology-Aware Skill Self-Evolution for LLM Agents*; title confirmed on arXiv, not read). SkillCommit says it "replays candidate patches on their source tasks before hierarchical merging", which makes it a direct prior for SkillCommit's stages 1 and 3. Also note that SkillCommit cites none of VaG, GRASP, SAGE or SEA.

## Suggested ledger update (new §2.10 entry or §2.12 resolution; pending cross-review)

- **SkillCommit 2608.15165 (Aug 2026 preprint): deep-read, threat LOW for the chain claim.** It is MEDIUM only for three narrow sub-claims: the staged promotion vocabulary, scope-annotated, versioned and audited skill records, and an ordered replay pipeline.
- **Chain coverage:** 1 F · 2 P (weak; abstraction tiers, not heritability tiers) · 3 P (`when`/`repair`/`avoid_when`, scope, versions, audit log) · 4 P (three-stage binary replay on induction instances; no control replay, held-out data, ablation or attribution) · 5 P (commit/split/reject into the live library) · 6 P (frozen weights only; one-shot frozen cross-model transfer).
- **Evidence caveats:** single run; NBA test n = 45; γ, L and m unreported; appendices missing in v1; compactness unmeasured; no code.
- **Add to §3:** "replay-validated scope expansion and source-retention commit/split/reject of hierarchical skill abstractions; per-skill validated-scope sets with version and audit history; typed `avoid_when` exclusion conditions" (SkillCommit; SkillCAT per SkillCommit's description).
