# HDSO — Hypothesis-Driven Skill Optimization — detailed notes (Round 3)

**Paper:** *Hypothesis-Driven Skill Optimization for LLM Agents*. Authors: Fangxin Shang, Yehui Yang (corresponding author), AI Lab, Qifu Technology, Beijing. arXiv:2606.22330v1 [cs.AI], 21 Jun 2026. v1 is the only version. The arXiv metadata lists exactly these two authors. No code link and no venue appear in the paper. Reviewed 2026-10-13 by Claude (third agent). **Pending Research/Critic cross-review.**

**Read depth: full text.**
- I read the arXiv HTML v1 in full, myself, after converting it to plain text: §1–6, Algorithms 1–2, Tables 1–4, App. A (curator system prompt and JSON schemas) and App. B (two approved skills, verbatim).
- I recovered the equations from the HTML math `alttext`. I viewed Fig. 2 as an image. For Fig. 1 I read only the caption.
- I did not open the PDF. The HTML conversion looked complete: every section listed in the table of contents is present.

## Mechanism

**Setting (§3.1).**
- A frozen executor E_θ and a frozen curator C_φ. The objective is max over S′ of E_{x∼D_eval} E_τ[U(τ)] "s.t. θ, φ remain fixed". Only the external repository S changes.
- The curator sees task inputs, actions, observations, outcomes, approved skills and validation summaries. It does not see the executor's private reasoning.
- Operating mode (§1): the executor serves online, and the curator improves the repository offline from the logs "during low-load periods".

**Four invariants (§3.1).**
1. The no-skill path *is* the executor-only baseline.
2. Candidates are structured artifacts with scope and falsification conditions.
3. Validation is "prospective and paired": control and treatment run on the same task indices.
4. Rejected candidates stay in a hypothesis ledger.

**Executor (§3.2, Alg. 1).**
- A ReAct executor with *progressive disclosure*. At each step, at most 3 compact skill cards are retrieved, and the executor may request one full skill body.
- Each request is logged as an "uptake event".
- Skill payloads can include restricted `pre_prompt` / `post_action` code hooks. These "must preserve baseline behavior when uncertain".

**Curator loop (§3.3, Alg. 2).**
- *State.* Three objects: the approved repository S, a hypothesis bank H and an evidence ledger L.
- *Rounds.* N_cyc = 3 cycles. Each cycle collects discovery traces (3 fresh traces per task type, then non-overlapping refills). The curator then proposes up to 8 hypotheses, each paired with a candidate skill.
- *Proposal schema (App. A).* Each proposal has these fields: `claim`, `evidence`, `source_trace_ids`, `attribution`, `applicability_condition`, `expected_improvements`, `guardrails`, `risks`, `falsification_plan`, `promotion_condition`, `rollback_condition`, `confidence`, and a `candidate_skill`.
- *Candidate skill.* Fields: `name`, `description`, `body`, `invocation_rule`, `task_types`, `keywords`, and `payloads[]`. Each payload has `kind` ∈ {text, tool, code}, `activation`, `entrypoint` and `safety`.
- *Proposal rules (system prompt).* Every hypothesis needs "a same-task-type contrast between at least one successful trace and one failed trace". The prompt also says "Any revised skill is a new unvalidated candidate and must be tested again."
- *Quarantine.* Fig. 2 labels the candidate "Candidate Skill (not deployed)".

**Paired gate (§3.3, Eq. 3–5; §4.1).**
- For a candidate s on a validation set V, control runs with S and treatment runs with S ∪ {s}.
- The counts are b = treatment-only wins and c = control-only wins. The gate computes net(s;V) = b − c and Δ_SR.
- The validation stages (target tasks + guardrail tasks):

| Stage | Target tasks | Guardrail tasks |
|---|---|---|
| screening-small | 4 | 2 |
| screening-medium | 8 | 4 |
| confirmation (on "independent task indices") | 32 | 12 |

- **Pass rule:**
  - Screening needs ≥ 1 net win and ΔSR ≥ 0.05.
  - Guardrail regressions (paired losses outside the intended scope) must be 0 in screening-small and ≤ 1 afterwards.
  - Confirmation uses the same success-rate threshold and allows ≤ 1 guardrail regression.
  - The gate also records step delta, invalid-action delta, "per-scope regressions" and skill uptake.
- **Exact one-sided McNemar/binomial test.** It is computed as p = Pr[Binom(b+c, 0.5) ≥ b] at α = 0.05, but it is "logged for analysis and audit, but it is not a hard promotion threshold".
- **Curator review.**
  - The curator then reviews the behaviour pairs. It returns `accept|reject` plus a separate `direction_assessment` ∈ {validated_direction, promising_needs_revision, wrong_direction, insufficient_evidence}.
  - It also assigns a `failure_attribution` ∈ {hypothesis, skill_content, invocation_scope, executor_variance, validation_power, none}.
  - Accepted candidates are consolidated into S. Rejected ones go to H/L as "negative evidence for next cycle".
- **Validation data.** All gate pools are "held-out pools sampled from the same split", meaning ALFWorld train. Final evaluation is on the ALFWorld evaluation set: 140 episodes per seed, 3 seeds.

## Results (verbatim numbers)

**Table 1 (Avg. SR, Δ vs executor-only):**

| Setting | Executor-only → HDSO | Δ |
|---|---|---|
| Qwen3-8B → 8B | 40.5 → 47.4 | +6.9 |
| Qwen3-8B curator → Qwen3.5-9B executor | 44.3 → 52.1 | +7.8 |
| Qwen3.6-27B → 27B | 61.7 → 65.7 | +4.0 |

- Avg. Steps falls slightly in each row.
- The external rows are *cited, not rerun*, as deltas only: MemP +1.8, SkillOS-base +5.2, ReasoningBank +7.8.

**Other results:**
- **Table 2 (27B→27B, per task type).** The gains come from heat (+18.7), cool (+8.0), look_at (+7.6) and pick_and_place (+2.9). Clean and two-object are 0.0. Paired wins are small integers (HDSO-only wins 12 against baseline-only wins 5 in total).
- **Table 3 (frozen transfer, curator off):**

| Repository | Executor | Delivery | Δ |
|---|---|---|---|
| 8B repository | 8B | full skill context | +1.6 |
| 27B repository | 8B | progressive disclosure | +5.7 |
| 27B repository | 27B | full skill context | +1.2 |

- **Table 4 (20% flipped success/failure labels during discovery, screening and confirmation).** 47.6 (+7.1), against 47.4 with clean feedback. Only one narrow skill was consolidated ("Sinkbasin Navigation Optimization"), and it was requested in 28 of 420 evaluation episodes. The authors explicitly decline to read label noise as beneficial.
- **§5 failure analysis.**
  - Active 27B→8B curation failed: "small positive effects reversed or vanished in medium validation".
  - 27B→9B approved one narrow look_at skill and gave no full-evaluation gain.
  - 8B→27B rules were already covered by the executor, or too local.
  - Qwen3.5-9B never produced a stable repository as curator.

**Not reported** (my check of the full text):
- no HDSO-without-gate ablation;
- no ablation of the individual gate stages or of curator review;
- no confidence intervals or per-seed variance;
- no count of proposed, accepted or rejected candidates;
- no repository size for the main runs (only the noisy run is described: one skill);
- no regression analysis on the final evaluation set beyond the Table 2 win/loss counts;
- no benchmark other than ALFWorld.

## Overlap vs HeritAgent chain (full-text judgement)

| Element | Coverage | Evidence |
|---|---|---|
| 1. Lifetime acquisition from trajectories | **Full** | The curator turns the deployed executor's own traces into candidate skills over 3 cycles (Alg. 2). Each hypothesis is grounded in `source_trace_ids` and in a required success/failure contrast. |
| 2. Somatic/germline separation | **Absent** (weak analogue) | Candidates are "not deployed" (Fig. 2) and are expressed only in treatment arms. Rejected or revised hypotheses sit in a ledger as negative evidence, and the online executor is separated from offline curation. None of this is a somatic store whose traits the agent *uses during its life* and later promotes (SPEC §5). The approved repository is the single live and persistent store, and no heritable store distinct from it exists. Same verdict as VaG's Cold/Warm/Hot tiers. Interesting parallel: SPEC §9 sends failed traits to "somatic memory"; HDSO's ledger feeds rejected hypotheses back to the proposer. |
| 3. Typed executable genome | **Partial** (the abstract-level entry said absent) | There is a typed JSON skill package: invocation rule, task_types scope, and typed payloads (text/tool/code with `activation`, `entrypoint`, `safety`). Code payloads are executable `pre_prompt`/`post_action` hooks. A per-skill hypothesis record (claim, falsification plan, promotion/rollback conditions) and an evidence ledger of paired outcomes are attached. This is the closest found analogue of "typed skill + per-trait validation record". **Missing:** no policy or regulation layer as separate types (the invocation rule is the only regulation-like element); no versioning, parent pointers or lineage; no notion of the repository as a heritable artifact. |
| 4. Causal validation gate | **Partial** (stronger in design than the abstract suggested; weak in demonstrated evidence) | **Design.** Prospective paired add-one control/treatment on matched task indices, with staged screening → confirmation on independent tasks. **Target (in-scope) vs guardrail (out-of-scope) tasks** with per-scope regression limits, plus an exact McNemar statistic (logged only). An LLM review of behaviour pairs adds an explicit failure-attribution taxonomy and an "insufficient_evidence" verdict. **Missing:** no replay across environments; no controlled ablation inside the evolved composite (each test is a single add-one test against the current S, and approved skills are never re-tested when S changes); no attribution under interactions; validation pools are drawn from the train-split distribution; thresholds are permissive (screening-small passes on 1 net win out of 4 target tasks); the p-value is not binding; and the gate is not a heritability prerequisite. **Evidence:** the paper never isolates the gate's contribution (no ungated arm). See "Assessment" below. |
| 5. Selective assimilation | **Partial** | Only supported candidates are consolidated into S, and in the noisy run the lifecycle "tends to admit narrow, testable skills". But S is the working runtime repository of one deployment, not a heritable genome, and assimilation is not a promotion from a somatic store. |
| 6. Multi-generational, frozen weights | **Partial** (frozen half full; generational half absent) | Curator and executor weights are fixed throughout. There are no agent generations, lineages or descendants. The three curator cycles are iterations within one run. The only inheritance-like event is a one-shot frozen transfer of an approved repository to another executor (Table 3), with no re-validation on the recipient. |

**Coverage summary: 1 F · 2 – · 3 P · 4 P · 5 P · 6 P. implements_full_chain = false.**

## Assessment: does the full text change the abstract-level entry (ledger §2.10.11)?

- **Corrections to the ledger entry.**
  - The abstract-level "no regression-protection or held-out stage described" is wrong. HDSO has out-of-scope guardrail tasks with paired-loss budgets (0, then ≤ 1), "per-scope regression checks", and a confirmation stage on independent task indices. These are held-out from discovery but come from the same ALFWorld train split.
  - Element 3 moves from – to **P**.
  - The 8B→9B row is a *heterogeneous active* run with the largest gain (+7.8). "Cross-model curation succeeds only when …" refers to the failed 27B→8B, 27B→9B, 8B→27B and 9B-curator runs (§5).
- **Design-level threat to CIG: up.** On the same benchmark (ALFWorld) with frozen Qwen models, HDSO already does the following:
  - paired add-one with/without validation before persistence;
  - the **in-scope vs out-of-scope split** that EXP-0001 stage 3 uses (HDSO's target vs guardrail tasks);
  - a multi-stage screening → confirmation schedule;
  - an exact discordant-pair statistic;
  - an uptake/expression log (cf. the EXP-0001 stage-1 "trait must fire" check);
  - per-candidate records with falsification and rollback conditions;
  - a rejected-hypothesis store.

  HeritAgent cannot present any of these as new. All of them should be cited.
- **Evidential threat: no higher, possibly lower, than SAGE/GRASP/VaG.** HDSO never shows that its gate causes the gains.
  - There is no ungated arm. The external baselines are cited deltas from other papers, which the authors concede are not byte-identical.
  - The delivery interface is a large confound. *My inference:* if Table 1 evaluates the approved 8B repository with progressive disclosure, then the same repository delivered as full context (Table 3) gives +1.6 against +6.9. Under that reading most of the measured gain depends on delivery, not admission. The paper does not state this comparison as such.
  - Effects rest on 3 seeds × 140 episodes with no CIs. Per-type paired wins are 0–4.
- **What is not threatened.** The conjunction: a somatic staging store, a typed policy+skill+regulation genome with lineage, trait-level ablation/attribution/replay stages, gating of *germline* assimilation, and measurement across agent generations. HDSO is single-deployment curation.
- **Net threat: MEDIUM (unchanged level).** Within MEDIUM the threat shifts: higher for wording about CIG stage design (in-scope/out-of-scope, paired add-one, staged confirmation, hypothesis records), lower as empirical evidence that a gate beats ungated persistence. For the latter, GRASP and VaG remain the stronger priors.

## What HeritAgent must not claim as novel (additions to ledger §3)

- **Persistent skill updates framed as *falsifiable hypotheses*.** Each candidate carries claim, evidence trace ids, applicability condition, expected effect, guardrails, falsification plan, and promotion and rollback conditions. Rejected hypotheses are kept as auditable negative evidence and fed back to the proposer (HDSO §3.3, App. A). Extends §3 items 9–10.
- **Target-scope vs guardrail (out-of-scope) paired validation with per-scope regression limits, run in staged screening → confirmation on independent tasks** (HDSO §4.1). Directly precedes EXP-0001 stage 3's in-scope/out-of-scope split. Add to §3 item 11.
- **An exact one-sided McNemar/binomial test on discordant pairs as the paired skill-admission statistic** (HDSO Eq. 5; logged only). Add to §3 item 12.
- **LLM review of paired behaviour differences that separates accept/reject from a diagnosis of direction and failure attribution** (hypothesis vs skill content vs invocation scope vs executor variance vs validation power).
- **Decoupled frozen curator / frozen executor with heterogeneous pairs, and progressive disclosure of skill cards as the delivery interface** (HDSO; SkillOS 2605.06614 for the curator/executor split with a *trained* curator).
- **The observation that a stronger curator's skills can pass early screening on a weaker executor and then vanish at medium validation** (§5). This is a known capability-mismatch failure mode relevant to H4.

## What remains different

- **One deployment.** No agent generations with distinct lifetimes, no descendants and no descendant adaptation cost.
- **No somatic/germline split.** Candidates are never used in life, and the approved repository is both the runtime and the persistent store.
- **No genome.** Skills are typed packages, but there is no genome of policies+skills+regulation, no versioning or lineage, and no regulation layer beyond `invocation_rule`.
- **A weaker gate than CIG.** It is single add-one against the current repository. It has no multi-environment replay, no controlled ablation within the composite, no attribution under interaction and no re-validation of incumbents. Its held-out pools are in-distribution, its screening pass is permissive (1 net win out of 4), and its statistical test is non-binding. Curator LLM judgement is part of the gate itself; HeritAgent's design keeps the gate logic fixed and outside the heritable artifact.
- **No gate-attributable evidence.** There is no ungated or performance-gated comparison inside the paper.

## Implications for EXP-0001 / CIG (each flagged; nothing was changed)

1. **Arm B′: HDSO-style staged paired gate as a budget-matched, domain-matched operationalisation. Requires ADR (ADR-0002 amendment).**
   - Candidate option (e) for ledger §5 item 1: admit a frozen candidate iff it passes three add-one stages with HDSO's rules, and drop the curator-review step so the gate stays non-LLM and identical across seeds.
   - Stages: screening-small (4 in-scope + 2 out-of-scope tasks: ≥ 1 net win, ΔSR ≥ 0.05, 0 out-of-scope losses), screening-medium (8 + 4: ≤ 1 loss), confirmation (32 + 12 independent tasks: ΔSR ≥ 0.05, ≤ 1 loss).
   - It has no replay-across-variants, no ablation-pool CI and no interaction stage. That makes it a clean "performance-gated, no trait-level stages" baseline.
   - It uses the same benchmark (ALFWorld), the same frozen-weights regime and paired execution.
   - **Budget.** A candidate that passes everything costs (6 + 12 + 44) × 2 = 124 episodes, against ADR-0002's ≈130 episodes per CIG candidate. B′ vs C would then compare *what* the gate checks at nearly equal episode cost. This is a strong argument for preferring it.
   - **Pool sizes.** The 44-task confirmation set exceeds EXP-0001's T_gate sub-pools (20/20/10/10). It would have to come from T_gate with logged reuse, or be scaled down pro rata. That choice must be pre-registered.
2. **Stage 3 thresholds. Requires ADR if changed; citation only otherwise.**
   - HDSO's in-scope rule (≥ 1 net win and ΔSR ≥ 0.05) is *stricter* than EXP-0001 stage-3 in-scope "Δsuccess ≥ 0", which admits ties.
   - The two out-of-scope rules are not directly comparable. HDSO caps *gross* paired losses (≤ 1 of 12 tasks, whatever the wins). EXP-0001 bounds the *net* Δ ≥ −0.05 over 10 tasks, which in practice means zero net harm: a loss offset by a win passes. HDSO's gross-loss count is closer to "regression protection". Both could be reported.
   - Reinforces ledger §5 item 4 (tie admission). Cite HDSO as prior art for the in-scope/out-of-scope split in the EXP-0001 rationale. The wording/citation change needs no ADR; a threshold change does.
3. **Report discordant-pair counts (b, c) and an exact McNemar p per stage. No ADR for reporting; making it binding requires ADR.**
   - Under ADR-0002's one-deterministic-execution-per-(instance, condition) policy, the exact discordant-pair test is the natural paired statistic. It complements the cluster bootstrap.
   - Combine with SkillGen's n01/n10 reporting (ledger §5 item 4).
4. **Delivery-interface confound. Requires ADR if an arm is added; reporting otherwise.**
   - EXP-0001 injects all genes in context in both B and C (design §3: "B injects all of them"). HDSO's Table 3 suggests that full-context delivery of a validated repository yields much smaller gains than retrieval with progressive disclosure.
   - A reviewer could argue that any C > B is a prompt-load effect, since C has fewer genes, rather than gate selectivity.
   - The existing token-matched/length-neutral control (ledger §5 item 5) partly answers this. A secondary arm "B with retrieval-based delivery" (top-k cards) would answer it directly. Delivery must stay identical between B, B′ and C in any case.
5. **CIG record schema. Requires ADR-0001 amendment.** Borrow, with citation:
   - a per-candidate `falsification_plan` and `rollback_condition`;
   - a reject-reason taxonomy that separates *harm* from *underpowered* (`validation_power`, `executor_variance`, `invocation_scope`). This agrees with the SEA note's suggestion of an abstain/"insufficient evidence" verdict distinct from reject, and matters for H5 analyses of why genomes stay small;
   - an uptake/expression count per stage.
6. **Negative-evidence feedback to the miner. Not applicable to EXP-0001; requires ADR later.** EXP-0001 freezes K = 10 candidates before branching, so HDSO-style feedback of rejected hypotheses into proposal does not apply. For multi-generation experiments it is a design choice that affects both arms and must be symmetric or ablated.
7. **H4 (cross-backbone). Later ADR (SPEC §19).**
   - A repository validated on the 27B executor transferred to an 8B executor (+5.7 with progressive disclosure) with no re-validation, but *active* 27B-curator→8B-executor validation failed.
   - So validation on the donor model is not a reliable predictor of benefit for the recipient, in either direction. This supports making the CIG held-out stage model-conditional for cross-backbone inheritance (ledger §5 item 10).

## Follow-up leads (titles verified on arXiv abstract pages only; not read)

- **SkillOS: Learning Skill Curation for Self-Evolving Agents.** Ouyang et al., arXiv:2605.06614 (7 May 2026). RL-trained curator updates an external SkillRepo for a frozen executor; later related tasks evaluate the updates. HDSO calls it "the closest comparison". It is relevant to element 1 and to curator/executor separation. Threat: probably low–medium, because the abstract has no gate before persistence. Queue for Round 3.
- **SkillEvolver: Skill Learning as a Meta-Skill.** arXiv:2605.10500 (11 May 2026). A meta-skill authors, deploys and refines skills, with a "fresh-agent overfit audit" and detection of "silent-bypass" (a skill that is valid but never invoked). The bypass detection is relevant to EXP-0001's stage-1 expression check. Threat: low–medium; queue.
- Also cited by HDSO, not checked: SkillOpt 2605.23904 (already in §2.12), EmbodiSkill 2605.10332, SE-GA 2605.16883.

## Anti-overclaim (SPEC §34)

- HDSO is itself careful. It writes "Our claim is not that hypothesis-driven skills dominate all memory methods on raw score", treats the label-noise result as robustness only, and limits its scope to ALFWorld.
- HeritAgent must not describe HDSO as demonstrating that hypothesis gating causes the gains. The paper does not isolate the gate.
- HeritAgent equally must not describe CIG's paired with/without testing, in-scope/out-of-scope split, staged confirmation, or per-candidate falsification records as new. HDSO (Jun 2026) precedes them on the same benchmark.
