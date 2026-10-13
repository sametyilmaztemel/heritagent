# HarnessEvolve — Learning from Reference Trajectories — detailed notes (Round 3)

**Paper:** *HarnessEvolve: Learning from Reference Trajectories for Reliable Agent Self-Evolution*. Authors: Wen Jiang\*, Mingmin Chu\*, Yimeng Tian\*, Qianxin Zhang\*, Haofei Yang, Rui Yang, Yang Liu, Tao Lv, Fangming Li (\* equal contribution). The HTML lists the affiliation as ICT AI Competence Center, Huawei Technologies, Shanghai; the corresponding authors are Rui Yang and Fangming Li. arXiv:2609.00829v1 [cs.LG; cross-list cs.AI], submitted 1 Sep 2026. Only v1 exists, with no venue and no comments field, so it is a preprint. Licence: arXiv perpetual non-exclusive. Reviewed 2026-10-13 by Claude (third agent), pending Research/Critic cross-review.

**Read depth: full text.** I downloaded arxiv.org/html/2609.00829 with curl and converted it to text, keeping the LaTeX `alttext` of every formula. I read §§1–5, all references and Appendices A–C (Algorithms 1–2). The abs page supplied the metadata. I could not read the figures (Fig. 1 overview, Fig. 2 evolution curves, Fig. 3 before/after harness diffs), because they are images. I did not check the PDF. Quotes are short and verbatim. Numbers are from Tables 1–4, and I recomputed the margins the text quotes: 86.9 − 65.3 = 21.6 and 92.8 − 90.1 = 2.7, both correct.

## Mechanism

- **Setting (§3.1).** A task corpus of (q, a\*) pairs is split into train, val and test. A base execution agent "powered by a frozen LLM" has its *harness* ("prompts, skills, tools, and execution logic") optimized on train to maximize val accuracy. Test is used once at the end.
- **Four role-separated agents (§3.1, Fig. 1):**
  - A_exec, the agent being optimized;
  - A_eval, which scores accuracy and verifies references;
  - A_opt, which diagnoses errors and proposes edits;
  - A_gate, the quality and performance gates.
  - The authors motivate the split as "ensuring that optimization decisions are auditable and cannot be gamed by the execution agent". The gate agent is "an isolated judge, structurally decoupled from the optimization agent, preventing self-serving evaluation" (§3.5).
- **Reference trajectories (§3.2–3.3).** Before optimization starts, A_exec is run on every training task *with its ground-truth answer*, up to T_att = 5 attempts. A_eval accepts a trajectory only if it is "genuine": it invokes the appropriate tools and does not "trivially restat[e] the provided answer". Accepted trajectories are cached as τ⁺. Tasks without one fall back to single-trajectory analysis.
- **Diagnosis and clustering (§3.4, Alg. 1).**
  - A_opt compares each failed trajectory τ⁻ with τ⁺ and emits (severity, error cause, fix hint). It locates the "first action divergence point" t\*, which is treated as the root cause; later divergences are treated as cascades.
  - Signals are clustered by cause, not by distance, and singleton clusters are kept ("long-tail protection").
  - A_opt then writes one candidate update set Δ_harness for the batch, prioritizing large and severe clusters.
  - Credit assignment here is an **LLM-judged alignment against a privileged, answer-conditioned trajectory**. It is not an intervention.
- **Quality gate (§3.5).**
  - *Data leakage:* an LLM-as-judge scores whether the edits "directly embed failed queries and their ground-truth answers into the harness files" (score 0–1, rejected above η_leak = 0.8).
  - *Prompt bloat:* the gate counts newly injected in-context examples (rejected above η_blo = 5).
  - Both checks apply per edited file. A rejected update returns to A_opt with a natural-language reason, up to T_rev = 3 revisions, and is then abandoned.
- **Performance gate (§3.5, Eq. 1).**
  - A_candidate = Build(A_current, Δ_verified) is accepted iff two conditions hold:
    - Acc(cand, B_j) − Acc(cur, B_j) ≥ δ on the **current training mini-batch** B_j, which is the same batch whose failures produced the edit;
    - the maximum degradation over the R previous batches is ≤ ε.
  - Settings: δ = 0.0, ε = 0.025, R = 2, b = 40.
  - With δ = 0 a tie is accepted, so an edit that changes nothing on B_j passes.
  - With b = 40, ε = 0.025 equals one item per replay batch.
  - Accepted candidates "immediately become the active agent" and are stored in a snapshot pool 𝕍. Ten consecutive rejections (P_batch) end the epoch.
- **Epoch-end selection (Eq. 2).** All snapshots in 𝕍 are scored on T_val and the argmax becomes A_current for the next epoch. Limits are E = 20 epochs and P_ep = 5 epochs without improvement. The same val set is reused for every argmax and for early stopping.
- **What is edited (§4.1, App. A).**
  - On the in-house LAMAgent framework: "the full project code, skills, prompts, and tools".
  - On OpenClaw: skills (the whole skill directory, including reference files and scripts), AGENTS.md, SOUL.md and tools.
  - Fig. 3 (caption only) shows edits to skill.md, scripts created inside skills, the system prompt, tool descriptions and execution-logic code.
- **Vocabulary.** A text search finds no occurrence of "causal", "counterfactual", "inherit", "lineage", "population" or "seed". "Generation" appears only as reference-trajectory or candidate generation. There is no limitations section and no code link.

## Results

- **Backbones.**
  - Qwen3.6-27B, "post-trained with domain-specific fine-tuning" (the base is frozen during evolution, but it is not an off-the-shelf model);
  - DeepSeek-V4-Flash.
- **Baselines.** GEPA, ACE and SkillOpt, under "identical environmental constraints and an initial harness". Budgets, numbers of rollouts and token costs per method are not reported.
- **In-house benchmarks (Table 1; LAMAgent; Qwen / DeepSeek).**
  - CloudCoreNetwork-QA: Base 43.4 / 47.5 → HarnessEvolve 86.9 / 85.9. The best baselines are GEPA 65.3 (Qwen) and SkillOpt 65.3 (DeepSeek).
  - Wireless-QA: 79.0 / 85.9 → 89.7 / 92.8, against SkillOpt 89.0 (Qwen) and ACE 90.1 (DeepSeek).
  - The Qwen Wireless-QA margin over SkillOpt is only +0.7.
- **Open benchmarks (Table 2; OpenClaw; DeepSeek only).** Results are Base → HarnessEvolve, with the best baseline in brackets.

  | Benchmark | Base → HarnessEvolve | Best baseline | Margin |
  |---|---|---|---|
  | SearchQA | 86.5 → 92.9 | ACE 90.0 | +2.9 |
  | OfficeQA | 62.8 → 70.9 | ACE 68.9 | +2.0 |
  | SpreadsheetBench | 44.3 → 76.4 | SkillOpt 74.6 | +1.8 |
- **Cross-framework transfer (Table 3).**
  - Skills optimized on OpenClaw were deployed without re-optimization on Hermes, OpenCode, LAMAgent and DeepSeek Harness.
  - The result is "improve or maintain" in all 12 cells. Hermes SearchQA is unchanged (95.0 → 95.0) and Hermes SpreadsheetBench gains only +1.1.
  - The gains are large on SpreadsheetBench: OpenCode 57.5 → 87.9 and LAMAgent 45.0 → 80.0.
  - Only *skills* were transferred, not the whole harness. The backbone used for Table 3 is not stated.
- **Ablation (Table 4; CloudCoreNetwork-QA, Qwen only).**
  - Full system: 86.9.
  - M1, without reference trajectories: 57.8. This is below both GEPA (65.3) and SkillOpt (61.9) in the same cell.
  - M2, without error clustering: 68.6.
  - M3, without the quality gate: 80.1.
  - **There is no "without performance gate" or "without epoch-end selection" ablation.** The two mechanisms closest to CIG (regression protection and held-out selection) therefore have no isolated evidence.
- **Statistics.**
  - Every number is a single point estimate. There are no seeds, CIs or tests. The word "significant" is used without a test.
  - Split sizes are not reported, and two of the five benchmarks are in-house and unavailable.
  - Fig. 2 is described as showing that "a substantial fraction of candidate updates are rejected" and that peak accuracy in 𝕍 rises monotonically. I could not read the plotted values.

## Overlap vs HeritAgent chain (full-text judgement)

| Element | Coverage | Evidence |
|---|---|---|
| 1. Lifetime acquisition | **Full** | Harness edits, including new skill scripts, are derived from the agent's own failed trajectories on the training stream (§3.4, Alg. 2 lines 7–10). One caveat: the "lifetime" is an offline optimization run with a separate optimizer agent and privileged answer-conditioned references, not acquisition during deployment. Other ledger entries with the same set-up (SkillOpt, SAGE) are rated full, so this one is too. |
| 2. Somatic / germline separation | **Absent** | Accepted edits "immediately become the active agent" (§3.5), so they are expressed before any held-out evidence. The snapshot pool 𝕍 with epoch-end val argmax is a two-tier *version* selection (batch-accepted → val-selected). It is not a store of individually staged traits apart from a heritable one, and nothing is non-heritable by design. Like the AHE short-/long-term memory, it is at most a weak analogue. |
| 3. Typed executable genome | **Partial (weak)** | The optimized artifact is heterogeneous and *executable*: skill directories with scripts, prompts, tool descriptions and execution-logic code (§4.1, App. A Fig. 3), versioned as whole-harness snapshots. However, the paper gives no schema, no trait types, no registry, no per-trait provenance or validation records and no regulation layer. The component kinds are implicit file categories. This is weaker than AHE (typed component classes, a registry and git tags), which the ledger rates partial (strong). The abstract-level rating was "–"; the full text shows enough executable multi-component structure to justify a weak partial. |
| 4. Causal validation gate | **Partial (weak)** | Three filters apply: (a) an LLM-judge leakage check and a count cap on prompt bloat; (b) a candidate-vs-current paired comparison on the *training* batch the edit was mined from, with δ = 0 so ties are admitted, plus non-regression on R = 2 recent training batches within one item; (c) a val-set argmax over snapshots. Weaknesses: (b) is an add-one before/after contrast, but on in-sample data and for a *bundle* Δ_harness that mixes edits from several clusters, so no individual trait is tested. Neither held-out set is untouched by selection. There is no replay across environments, no controlled ablation within a multi-trait artifact, no attribution and no out-of-scope probe. No statistics are used. The only trajectory-level "attribution" is the LLM-judged first-divergence diagnosis *before* the edit, which is diagnosis, not validation. The paper does not call any of this causal. |
| 5. Selective assimilation | **Partial** | Edits are accepted or rejected into the live harness, and an epoch-level best snapshot is selected. The process is selective, but it acts at bundle and version level and goes into the working artifact. There is no separate heritable store and no per-trait assimilation event. |
| 6. Multi-generational, frozen weights | **Partial (weak: frozen weights only)** | "Powered by a frozen LLM" (§3.1), although Qwen3.6-27B was domain-fine-tuned beforehand. There are no generations, descendants or lineage, only epochs and batches of one harness. Cross-framework reuse of the evolved skills (Table 3) is transfer, not inheritance. |

**Implements full chain: no.** One element is full, four are partial (three of them weak) and one is absent. The abstract-level vector was 1 F · 2 – · 3 – · 4 P · 5 P · 6 P. The full text moves element 3 from – to P (weak). The others are unchanged, but element 4 is weaker than the abstract implied.

## Does the full text change the abstract-level assessment?

**Threat to the chain claim: down from MEDIUM to LOW. Relevance to EXP-0001 design hygiene (leakage, in-sample evidence, miner confound): medium.**

- **Why down.**
  - The abstract's "performance gate … without degrading recent batches" plus "held-out validation" read like a regression-protected held-out gate. The full text shows something weaker:
    - the acceptance test runs on the *training* mini-batch that generated the edit;
    - δ = 0 admits ties;
    - the regression check covers only the last two training batches, with a one-item tolerance;
    - "held-out" is a validation set reused for repeated argmax selection and early stopping, which is exactly the optimizer's-curse setting SAGE warns about.
  - Acceptance is per bundle, not per trait.
  - The performance gate is never ablated, so the paper gives no evidence that its gate (as opposed to its reference-guided miner) drives the gains. Its own ablation says the miner dominates: removing reference trajectories costs 29.1 points, while removing the quality gate costs 6.8.
  - There are no inheritance, genome, staging store or generations.
  - Evidence quality is low: single runs, two in-house datasets, a domain-fine-tuned backbone, no budgets, no code.
- **Fair the other way.**
  - It is a strong practical system with large reported gains (e.g. +43.5 points over Base on CloudCoreNetwork-QA with Qwen). It edits the *whole* executable harness, including code, which is closer to the EAG's "executable" aspect than skill-document optimizers (SkillOpt, SAGE) are.
  - The quality gate is a genuine, ablated contribution: leakage of training answers and prompt bloat, plus revise-with-reason before rejection. Neither the CIG nor EXP-0001 currently names it.
  - The role-separated, isolated gate agent is concurrent support for keeping gate logic outside the optimized artifact (ledger §5 item 6).
  - Table 3 is positive evidence that evolved skills transfer across agent frameworks without re-optimization, though it is single-run and only for skills. That is relevant to H4 and to "inheritance across interfaces".
  - A reviewer will still group it with GRASP, SAGE, SkillBoost and DarwinX under "gated self-evolution with regression checks" (ledger §3 items 9 and 11 already concede this). It does not pre-empt trait-level, multi-stage evidence as a prerequisite for germline assimilation across agent generations.

## Consequences for HeritAgent

No design was changed. Items marked *requires ADR* are for the Builder/ADR owner.

1. **Arm B′: do not use HarnessEvolve's gate as the primary operationalization. *Requires ADR (ADR-0002 amendment) if adopted at all.***
   - The gate (in-sample batch Δ ≥ 0, last-R non-regression, val argmax) is a weak, practitioner-style performance gate. Using it as B′ would invite a strawman objection.
   - The SAGE-style option (e) in the SAGE note, or the GRASP or VaG options in ledger §5 item 1, remain more defensible.
   - It could serve as a secondary "weak B′" that brackets the range, but only if the budget allows.
2. **Leakage and bloat hygiene filter applied identically to all arms. *Requires ADR if it becomes part of the CIG; if it is applied as a pre-gate hygiene step to arms A, B, B′ and C alike, it is an analysis-plan item that should still be pre-registered.***
   - Without it, arm B (unrestricted) can gain on any train/test overlap by memorizing answers. That would inflate or deflate the B–C contrast for reasons unrelated to causal validation.
   - Candidate rule: reject a candidate trait that embeds a training query or answer verbatim (string match plus an LLM judge), and cap injected examples per trait. Cite HarnessEvolve for the leakage/bloat gate; ledger §3 item 11 already lists it as not ours.
   - Connects to τ_k (genome-size control).
3. **No in-sample evidence for any CIG stage. *Requires ADR if EXP-0001 stage 1/2 currently replays the episodes a trait was mined from.***
   - HarnessEvolve accepts on the batch that produced the edit, which is selection on training data.
   - CIG replay and ablation evidence should come from episodes disjoint from the mining episodes, or the plan should state explicitly which stages are allowed to be in-sample. Check against EXP-0001's stage definitions.
4. **Admitting ties (δ = 0) is a known weakness; it repeats ledger §5 item 4 for EXP-0001 stage 3 "Δ ≥ 0". *Requires ADR for any threshold change.*** HarnessEvolve is a second published instance of a tie-admitting gate, alongside VaG. Use it as a citation when arguing for abstain-on-tie or a pre-registered non-inferiority margin.
5. **Miner confound. *Design note; pre-registration item, no ADR unless arms differ.***
   - Table 4 shows that the proposal/diagnosis quality moves accuracy by about 29 points, far more than the gate ablation (about 7).
   - In EXP-0001, every arm must share the *identical* Trait Miner, prompts and candidate stream, with only the assimilation rule varying. Better still, freeze the candidate set per generation and apply the gates offline to the same candidates.
   - Otherwise any C vs B difference may come from the miner.
6. **Bundle vs per-trait acceptance. *Requires ADR (B′ definition).*** HarnessEvolve accepts a whole Δ_harness. B′ must state whether it gates per trait (a fair comparison with CIG) or per bundle (a practitioner baseline). The choice changes genome size and regression attribution.
7. **Validation reuse across selections and generations.** The repeated val argmax, plus early stopping on the same set, repeats the multiplicity concern in the SAGE note (item 4). CIG held-out sets reused across generations need fresh items or a correction. *Requires ADR* (already flagged).
8. **Privileged reference trajectories. *Design note; requires ADR only if the Trait Miner would use ground-truth answers.***
   - Answer-conditioned "reference" trajectories are a cheap form of credit assignment, and the first-divergence step is a useful diagnostic for the Trait Miner.
   - However, they require ground-truth answers on the mining episodes. If adopted, those episodes must be disjoint from all CIG held-out and test sets, and the paper's "genuine trajectory" verifier should be adopted with them.
   - Never describe this as causal attribution (SPEC §34).
9. **H4 framing.** Table 3 adds HarnessEvolve to the cross-framework transfer cluster (ledger §3 item 24). Transfer was skills-only, improve-or-maintain, and varied strongly by framework (+0.0 to +35.0). Frame H4 as measuring a known, conditional effect.
10. **Wording.** Do not claim "decoupled execution / optimization / gating agents", "leakage/bloat gating" or "snapshot pool with held-out selection" as HeritAgent mechanisms. "HarnessEvolve" is a generic-sounding name; cite it as "HarnessEvolve (Jiang et al., arXiv:2609.00829)". When contrasting, say "bundle-level acceptance into a single live harness on in-sample training batches, with validation-set snapshot selection, within one optimization run".

## Suggested ledger update (§2.10.17; pending cross-review)

- Mark the entry deep-read (full text). Lower the threat from MEDIUM to LOW for the chain claim.
- Revised chain vector: 1 F · 2 – · 3 P (weak; executable multi-component harness, untyped) · 4 P (weak; in-sample bundle gate, δ = 0, R = 2 recent-batch non-regression, reused val argmax, no statistics, performance gate not ablated) · 5 P · 6 P (frozen weights only; Qwen domain-fine-tuned).
- Evidence caveats: single runs; no CIs, seeds, split sizes, budgets or code; two in-house datasets; ablation on one cell only.
- Main value:
  - the leakage/bloat hygiene filter for all arms;
  - the warning against in-sample gate evidence;
  - the miner-confound control;
  - cross-framework skill-transfer evidence for H4.
