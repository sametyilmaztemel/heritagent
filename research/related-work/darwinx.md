# DarwinX — detailed notes

**Paper:** *DarwinX: Evolving Agent Harnesses Through Natural Selection*. Authors: Yifan Zhang, Yutong Dai (first authors), Juntao Tan, Luyu Yang (core), Rishi Mullur, Thai Hoang, Zhiyuan Hu, James Zhu, Phil Mui, Silvio Savarese, Ran Xu, Zeyuan Chen (Salesforce AI Research / Salesforce Agentforce). arXiv:2608.07545v1 [cs.NE; also cs.AI, cs.LG, cs.SE], submitted 31 Jul 2026. No code link on the abs page. The evolved agent "Monet" is **Salesforce-proprietary**; DarwinX is the procedure that evolves Monet's harness. Read depth: **full text** (v1 PDF, 33 pp. incl. appendices A–E). Reviewed 2026-10-12.

## Mechanism

- **Setting:** the base model is frozen in every matched comparison (GPT-5.5, GPT-5.6 Sol, Opus 4.8). What evolves is the *harness*, which has two editable layers: **skill** (prompts, memory, distilled knowledge) and **code** (tools, control flow, agent loop). Edits are additive deltas, Δ = Δcode ⊕ Δskill ⊕ Δprompt ⊕ Δtool.
- **Archive:** a tree. Each node holds a harness snapshot plus its edit delta, per-task scores, trial evidence and distilled lessons. Every scored variant is kept, losers included.
- **Preserve-and-extend contract (§2.1):** per-task Δ_t = p̂_t(child) − p̂_t(parent) (avg@k). Net gain g = Σ Δ_t; bounded regression R = Σ (−Δ_t)₊. The child is admitted iff g > 0 and R ≤ δ. A "reasoned verifier agent" f(g, R, trial evidence, shared memory) then returns promote or revert. A promoted child is re-tested at higher fidelity with a **preservation probe** (re-samples the lineage's known solved set) and full avg@k confirmation *before it may steer search*.
- **Two speeds by design:** the enabler is deliberately permissive ("favors recall during exploration"), so a variant can enter the tree on a noisy signal. Only confirmed nodes join the steering set S used for parent selection.
- **Inheritance eligibility (§2.3):** children are classified by solved-set relation. Improvers (S(c) ⊋ S(p)) and neutral children (S(c) = S(p)) are *eligible for inheritance*. Stepstones (S(c) ⊊ S(p)) and trade-off nodes are not; they contribute only distilled lessons.
- **Recombination:** complementary specialists are merged as H = H0 ⊕ Δ (shared ancestor). The merged child is kept iff S(child) ⊇ ∪ S(v_i), so a merge can only add coverage.
- **Parent selection:** exploit the node with the highest cumulative lineage gain G(c) = G(p) + g(c) with prob. 1−β, otherwise broaden across the population.
- **Learning signals (§2.4), one edit interface:**
  - failure-derived: summaries of failed trajectories;
  - teacher-derived: a reference solver's successful trajectory, used on "walls" (tasks no agent rollout solves);
  - self-derived: contrast of the agent's own pass and fail rollouts, used on variance-band tasks.
  All of them become harness edits; none touch weights.
- **Shared memory K_g (§2.6):** a population-level aggregate of worked/regressed/failure themes. Both proposer and verifier read it, and it steers toward global capabilities rather than per-task patches.

## Results (matched-model deltas are the load-bearing ones)

- **Terminal-Bench 2.1** (89 tasks; evolution and report on the same suite, i.e. in-domain). GPT-5.5: base 75.5 → **83.2** avg@5 (+7.7). GPT-5.6 Sol/medium: 84.7 (verified-leaderboard frontier). Neutral Terminus-2 on GPT-5.5/xhigh: 78.0. Per-task: 36 improve / 43 unchanged / 9 regress (30 vs 6 at a 10-pt threshold). Reward-hacking audit: 1 confirmed shortcut trial out of 445.
- **TerminalWorld** (94 train / 41 disjoint held-out, pass@1). Opus 4.8: 25 → **28/41** (68.3%); GPT-5.5: 20 → 23. Authors concede the matched Opus comparison is McNemar p = 0.45 and the margin over Claude Code p = 1.0, so it is "suggestive". The training subset saturates to 1.000 while held-out is 68.3%, i.e. the **in-loop proxy overfits by 31.7 pts**. Four specialists solve 24–27 held-out tasks and the merge solves 28. A separately skill-bundled pre-TW reference also reaches 28/41 (App. C).
- **WebArena-Infinity** (evolve on 300 synthetic intents with an LLM judge; report on 1,260 real tasks with deterministic verifiers). GPT-5.5: **43.5 → 93.0** audit-clean pass@1. Invalid trajectories fall 293 → 17. Note: **every recombination attempt was reverted on WAI**; gains came from a short primary lineage (26 kept / 36 reverted). The base harness scores 20.0 on Elation-Rx and 25.0 on Gmail, so part of the +49.5 is fixing a weak starting harness.
- **SWE-bench Verified transfer** (TB2.1 harness run unchanged, Opus 4.8): 84.2% vs 80.8% for a fix-skill reference. One direction only; the authors call the band "narrow".
- **"Ablation" (RQ5)** is a skill-bundle diff (7 added skills, all verification/artifact-contract) plus per-cluster gains. The authors state explicitly: *"exploratory attribution, not a per-skill causal ablation: the skills were co-selected, not independently randomized."*

## Overlap vs HeritAgent chain

| Element | Coverage |
|---|---|
| 1. Lifetime acquisition from trajectories | **Partial.** Edits are mined from trajectories: failure summaries, teacher distillation, and self pass/fail contrast (≈ SkillRL-style differential distillation). But acquisition is an offline proposer mutating a harness, not an agent acquiring traits during its own lifetime. |
| 2. Somatic/germline separation | **Absent.** A candidate edit is instantiated straight into a child genotype (classic mutate-then-select). There is no per-individual store of acquired-but-not-yet-heritable traits. Nearest structures a reviewer may map: (a) the provisional "in-tree but not steering" tier; (b) non-eligible children that pass on only distilled lessons into the population-level K_g. Both are evolutionary-algorithm probation/culture mechanisms, not soma. |
| 3. Typed executable genome | **Partial.** The executable harness is versioned in an archive tree. Edits are layer-typed (code / skill / prompt / tool) and additive, which makes deltas mergeable. Skills carry "when to use" + procedure (App. Tables 13–14), roughly primitive regulation. Table 7 says "typed nodes", but no genome schema, trait type system, or first-class regulatory genes are specified, and the harness is a proprietary codebase. Stronger than DGM's raw code, weaker than an EAG. |
| 4. Causal validation gate as heritability prerequisite | **Partial. Strongest overlap in the ledger to date on the *regression* side.** Passing preserve-and-extend + verifier + preservation probe + avg@k confirmation is literally required before a variant may become an ancestor or be merged. Because each mutation is one additive edit, child-vs-parent on matched model/tasks is in effect a **per-edit paired intervention** (cf. Skill-α's rollback reward). Missing vs CIG: (i) no trait-level ablation inside a composed genome (co-selected skills never isolated; authors disclaim causal attribution); (ii) no attribution stage; (iii) no held-out-generalization condition *in the gate* (held-out data is used only for final reporting, and the gate is shown to overfit its proxy); (iv) screening on rotating avg@3 subsets is noisy and the enabler is deliberately permissive; (v) interaction testing is limited to merge union-coverage. In CIG-stage terms: replay ≈ preservation probe (yes); controlled ablation ≈ per-edit child/parent contrast (partial); attribution (no); held-out generalization (no); regression protection (**yes, full**). |
| 5. Selective assimilation into heritable genome | **Partial.** Selective retention into the heritable lineage is the paper's core contribution: only S(c) ⊇ S(p) variants are inheritance-eligible, and merges are kept iff they cover the union. But selection is over whole genotype variants, not assimilation of individually validated traits from a somatic store into a genome. |
| 6. Multi-generational, frozen weights | **Full.** Weights frozen throughout. Lineages reach depth ~11 on WAI (Fig. 7b). Explicit parent/child/merge inheritance. The harness is run across base models (GPT-5.5, GPT-5.6 Sol, Opus 4.8), though survival under a base-model swap is listed as *unmeasured* (App. E). |

**Full chain:** no. Elements 2 and the trait-level core of element 4 (ablation/attribution/held-out as the gate) are absent.

## Limitations (stated)

- Whole-system evaluation only. Archive, parent selector, merge operator and effort are not independently randomized, and operator contributions are "plausible rather than causal".
- Recombination's value relative to single-lineage mutation "still requires controlled ablation". It failed on every WAI attempt.
- TerminalWorld is small (41 tasks; p = 0.45). The in-loop proxy overfits.
- SWE-V transfer is one-directional and inside a narrow band.
- The validity audit is static + LLM, not a formal sandbox.
- avg@k cost restricts the method to offline periodic evolution.
- Implicit: proprietary agent, no code, leaderboard rows not controlled. Most headline numbers are single matched pairs without seed CIs, except the TB2.1 ±1.2.

## Consequences for HeritAgent

- **Do-not-claim additions (§3):**
  - Variant-level regression protection or bounded-regression "preserve-and-extend" admission as a precondition for a variant becoming an ancestor.
  - Preservation probes over previously solved tasks.
  - Two-speed (provisional vs confirmed) noise-aware admission with avg@k re-measurement.
  - Recombination of complementary specialists with union-coverage acceptance.
  - A unified failure/teacher/self-contrast edit interface under frozen weights.
  - Natural-selection/"inheritance" framing for frozen-model harness lineages.
  - "Preservation as a policy/safety interface" (App. E, concept only).
- **CIG positioning must narrow.** After DarwinX, "a gate that blocks regressions before inheritance" is prior art, and SPEC §17's "lower regression" outcome alone cannot carry H2. The defensible part of CIG is now (a) **trait-level** controlled ablation and attribution *within a composed genome*, (b) **held-out generalization as a gate condition rather than a reporting split** (DarwinX's own 31.7-pt proxy-overfit result is the motivating evidence; cite it), and (c) the somatic→germline lifecycle that decides *which acquired trait* is assimilated, as opposed to which whole variant survives.
- **Baseline gap in EXP-0001.** The current arms are A / B (unrestricted) / C (CIG). There is no arm for "regression-gated inheritance without per-trait causal evidence". Without one, a reviewer can attribute any C > B effect to regression protection that DarwinX already provides. Recommendation: add a DarwinX-style arm, or a CIG ablation that keeps only stage 4 (regression) plus net-gain, to show the extra value of stages 1–3 (also matches SPEC §16 baseline D and the "HeritAgent – generalization gate" ablation). Decision for the Builder/ADR owner, not made here.
- **Reusable ideas (cite if adopted):**
  - cumulative lineage gain for parent ranking when variants are scored on different subsets;
  - union-coverage acceptance as a cheap epistasis check for merged genomes;
  - verifier-validity audit (evaluation-plane / privileged-knowledge classes) as a guard against genome traits that encode shortcuts.
- **Leads cited by DarwinX, not in the ledger** (abs pages checked only; full text not reviewed):
  - **HarnessX**, arXiv:2606.14249, Chen et al. (abstract: "typed harness primitives" via a substitution algebra, trace-driven evolution engine AEGIS, trajectories also used as model-training signal). DarwinX describes it as a per-edit gate + "seesaw test" with variant isolation and GRPO co-training. **Relevant to element 3 ("typed") and element 4. Needs its own deep-read.**
  - **SkillOpt**, arXiv:2605.23904, Yang et al. (abstract: a skill edit is accepted only if it strictly improves a held-out validation score; frozen agent). A held-out per-edit gate within one skill document, which bears on CIG's held-out stage. **Needs its own deep-read.**
  - Not opened at all, so their content is unverified: Agentic Harness Engineering (arXiv:2604.25850), EvoSkill (arXiv:2603.02766), Trace2Skill (arXiv:2603.25158), Red Queen Gödel Machine (arXiv:2606.26294), Hyperagents (arXiv:2603.19461).
