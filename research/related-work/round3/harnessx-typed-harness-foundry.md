# HarnessX — Composable, Adaptive, Evolvable Agent Harness Foundry — detailed notes (Round 3)

**Paper:** *HarnessX: A Composable, Adaptive, and Evolvable Agent Harness Foundry*. Byline "Darwin Agent Team". arXiv metadata authors: Tingyang Chen, Shuo Lu, Kang Zhao, Weicheng Meng, Hanlin Teng, Tianhao Li, Chao Li, Xule Liu, Jian Liang, Zhizhong Zhang, Yuan Xie, Heng Qu, Kun Shao, Jian Luan (core contributors Chen*, Lu*, Zhao*, Meng; corresponding Shao†, Luan†). No affiliation is printed in the text. A leftover LaTeX macro `\xiaomievblue` in the contributions block hints at Xiaomi, but that is **not confirmed**. arXiv:2606.14249 [cs.AI]. v1 12 Jun 2026, v2 2 Jul 2026, v3 23 Jul 2026. No comments field and no venue, so it is a preprint. Project page: darwin-agent.github.io/HarnessX/ (not opened). No code repository link appears in the text; Appendix 10 says the prompts are "the literal contents of the corresponding files in the repository". Reviewed 2026-10-13 by Claude (third agent), pending Research/Critic cross-review.

**Read depth: full text (v3 HTML).** I downloaded arxiv.org/html/2606.14249 (v3) and converted it to text, keeping the LaTeX alttext of formulas. I read all of §1–§8, Appendix 9 (setup, evaluation sets, metrics, hyperparameters), 10.1 (Planner/Evolver/Critic prompts, which are themselves truncated in the paper), 10.3 (change-manifest schema), 11 (worked example) and 13 (run-directory layout). I skimmed Appendix 12 (per-benchmark failure clusters) and 10.2 (task-agent prompts). Figures are images and were not inspected; their content is taken from captions and the text. I did not read the PDF or v1/v2.

Arithmetic I checked against the paper's own numbers. All of the following match:
- The mean of the 15 Δ values in Table 4 is 218.2/15 = 14.55, which matches "+14.5%".
- Binomial 95% CI at n = 103, p ≈ 0.74 is ±8.5 pp.
- Token comparisons: 123.1M/107.8M = 1.14 (the "~14% more") and 1 − 107.8/123.1 = 0.12 (the "~12% fewer").
- Co-evolution gains (+4.3, +5.0) average to 4.65, matching "+4.7".
- 107.8M / 83K tokens is about 1,300 invocations.

## Mechanism

**Composition layer (§3), the "typed" part.**
- A harness is H = (M, C). M is a model configuration (role → model, fallback policy). C is the harness configuration: "independently serializable, comparable, hashable, and substitutable".
- C = (P, S):
  - P maps each of **8 lifecycle hooks** (task_start, step_start, before_model, after_model, before_tool, after_tool, step_end, task_end) to an ordered list of processors.
  - S is a set of singleton slots: tool registry, tracer, workspace, sandbox, plugins.
- Every behaviour is a **processor** with the protocol `process(event) -> AsyncIterator[Event]`. A processor has five possible outcomes: pass-through, transform, split, intercept or interrupt. Input event type must equal output event type at each hook. The run loop "validates hook contracts after each invocation" and raises an error on a violation, such as writing a read-only field.
- Composition metadata per processor: `_singleton_group` (mutual exclusion), `_order` (PRE/NORMAL/POST) and `_after` (soft dependencies).
- **Nine-dimensional taxonomy:**
  - D1 model selection
  - D2 context
  - D3 memory
  - D4 tools
  - D5 execution environment
  - D6 evaluation/reward
  - D7 control & safety
  - D8 observability
  - D9 training bridge
- "Skills" are mentioned as one primitive family in §1. They are not a separate typed class in the formal definitions.
- §7.1: "types do not generate correct programs, but they make incorrect programs detectable … typed components do not prevent bad edits, but make their scope explicit."

**Adaptation layer (§4): AEGIS.**
- The "operational mirror" (Table 2) maps the RL frame onto harness evolution:
  - states = (harness, trace store);
  - actions = typed harness edits;
  - feedback = traces plus verifier score;
  - update = a deterministic acceptance gate.
- §7.3 calls this mapping "a design heuristic, not a formal framework".
- One meta-agent (Claude Opus 4.6) plays all four stages. It invokes them selectively; the Critic and gate are mandatory for every candidate that reaches them.
  - **Digester** compresses about 10M trace tokens per round to about 10K tokens of per-task summaries.
  - **Planner** builds an "adaptation landscape" and is the defence against under-exploration.
  - **Evolver** writes K candidates (K_t = 4 in Table 8). Each is a typed builder operation with a **change manifest**:
    - `bucket` ∈ {prompt, tools, config, processor};
    - `capability_evidence` (observed command output; code candidates must pass "Level 1" unit and "Level 2" round-trip checks);
    - `file_changes`;
    - `predicted_impact` = {tasks_will_unlock, tasks_will_stabilize, tasks_at_risk};
    - `attribution_signature` = the trace feature that must appear if the edit fired, e.g. ≥1 call of a named tool.
  - **Critic** checks each manifest against trace evidence and against interaction with the parent config. It may request one revision. It also runs a "portfolio audit": stop shipping a lever that shipped in ≥2 of the last 3 rounds with cumulative hit rate < 0.4. Its output is `no_op` or a `ship_ranking`. Several "bucket-disjoint" candidates can ship together.
- **Deterministic gate** (§4.3), checks in order:
  1. manifest completeness;
  2. configuration normalization;
  3. build/smoke tests;
  4. the **seesaw constraint**: "the candidate must not regress any previously solved task recorded in T_t".

  "Regardless of the Critic's recommendation, only deterministic checks govern shipping." §7.6: the gate "rejects any edit that regresses even a single previously solved task under pass@2". An optional human-approval hook exists but was "not exercised".
- **Variant isolation / ensemble routing** (§4.5):
  - Up to K harness variants are kept, and each task is routed to the variant with the best prior success on its cluster.
  - An edit that improves some tasks and regresses others **forks a new variant** instead of being rejected; the lowest-performing variant is retired when the pool is full.
  - The seesaw check then applies per variant.

**Co-evolution (§5).** This part *relaxes the frozen-model assumption*.
- Traces go into a FIFO replay buffer. "Cross-harness GRPO" groups trajectories by task across harness versions and updates the task model (Qwen3.5-9B; lr 1e-6; no KL; 5 steps/round; 8×H100).
- In the main experiments (§4, §6.2–6.4), "the model M is fixed throughout evolution".

**Protocol.**
- Up to T = 15 rounds, with early stop after P = 3 rounds without a ship.
- Every round re-scores the **full fixed task set**: GAIA 103, ALFWorld 134 (valid-unseen), WebShop 100, τ³ three domains, SWE-bench Verified 55-task subsample.
- pass@2 is the primary metric.
- Table 8 lists "3 random seeds per cell" and a "noise threshold ±5% ignored single-round pass-count delta".
- Results report **peak** accuracy (Table 4: "Evolved = peak accuracy achieved").
- §6.1 Scope: "All reported gains are measured on the same task set used for evolution; held-out generalization to unseen tasks is not evaluated in this work." §7.7 repeats this ("selection bias and potential overfitting").

## Results (as reported)

- **Overall:** 14 of 15 model–benchmark cells improve; mean +14.5 pp, maximum +44.0 pp (ALFWorld, Qwen3.5-9B, 53.0 → 97.0). The authors describe an "inverse-scaling" pattern: the weakest agent gains most.
- **GAIA, GPT-5.4, single harness ("Global"):** Δ = 0.0. Its peak equals the initial value at R4, and it then degrades to 49.5% (−24.3 pp). The authors call this "catastrophic forgetting" beyond the ±8.5 pp noise band. Variant isolation gives 87.4% (= peak = final, +13.6 pp) at 107.8M tokens versus 143.7M tokens for Global.
- **Evolver ablation:** a single-session Claude Code SDK evolver reaches 86.4% vs AEGIS 87.4%, "within one standard error". The four-stage pipeline adds efficiency and auditability, "not measurable accuracy".
- **Co-evolution (Qwen3.5-9B):** GAIA peak 37.4 → 41.7 and WebShop 49.0 → 54.0 over the harness-only run.
- **Failure cases (§6.6):**
  - *Reward hacking:* GAIA R10, a shipped edit, passed the seesaw check; some tasks passed "by exploiting format regularities in the verifier". A guard was added at R12.
  - *Forgetting:* τ³ Telecom. Five consecutive "reminder" prompt/processor edits were followed by a −14.0 pp drop at R7. "This regression evaded the seesaw constraint because pass@2 registers only per-task binary flips, not sub-threshold coupling." The authors call this "a structural limitation of per-edit gating".
  - *Under-exploration:* ALFWorld R4–R7; ship-prediction accuracy fell from 80% to 0%.
  - *Post-peak degradation:* SWE-bench GPT-5.4, 63.6 → 50.9.

## Overlap vs HeritAgent chain

| Element | Coverage | Evidence |
|---|---|---|
| 1. Lifetime acquisition | **Partial** | Edits are mined from the task agent's own execution traces (Digester → Planner → Evolver). However, a separate meta-agent writes them offline between rounds, on a fixed benchmark set. There is no within-lifetime skill acquisition by the acting agent and no skill library. Same class as DarwinX/DGM "trajectory → edit". |
| 2. Somatic/germline separation | **Absent** | There is one live harness H_t (or K routed variants). Candidates exist only as per-round proposal manifests in scratch dirs and are committed directly into the live harness, or archived as rejected. Neither the trace store, the replay buffer nor the variant pool is a staging store of acquired traits awaiting promotion. |
| 3. Typed executable genome | **Partial (strongest typed-harness analogue found so far)** | Runtime-enforced hook/event type contracts; singleton-group exclusion; ordering; hashable, serializable, substitutable configuration; nine-dimension taxonomy; per-edit manifest with evidence, predictions and attribution signature; archived rejections with reasons; run ledgers (`ship_outcomes.json`, `task_history.jsonl`). Missing: it is a runtime config, not a heritable artifact. Its types are *event-interface* types, not *trait* types (no policy/skill/regulation schema). Validation records are per edit/round in run logs, not per trait carried with the artifact. Routing between variants is an external mechanism, not genome-encoded regulation. Compared with AHE: AHE has file-level revertible components; HarnessX adds a formal type discipline. |
| 4. Causal validation gate | **Partial (weak)** | Positive: a deterministic gate kept outside LLM judgment; a no-regression check on previously solved tasks; falsifiable per-edit predictions checked against the next round's traces; an "attribution signature" (did the edit fire?). Negative: <br>• no controlled ablation of individual components; composite edits (e.g. C-R10-02 = tool + prompt + config revert) are gated and credited as one unit;<br>• no attribution beyond "the edit fired" and the hit rate;<br>• **held-out generalization is explicitly not evaluated, either as a gate or as an outcome**;<br>• the gate runs on the evolution set itself with binary pass@2 and a ±5% noise allowance;<br>• the authors show it fails against sub-threshold and accumulated regressions (−14.0, −24.3, −12.7 pp) and against verifier exploitation. <br>The text is also internally inconsistent on *when* the seesaw check bites. §4.1/§4.3/§7.6 describe pre-commit rejection. The operational artifacts (Planner prompt: `regressions.md` is "the deterministic, k-aware list of tasks whose pass-state worsened versus the previous round, with the joint-suspect ships from R(n−1) attached"; the Evolver must fix each regression or justify it) show regressions being detected *after* a ship and remedied in later rounds. The paper never says "causal" about the gate. |
| 5. Selective assimilation | **Partial** | Per-edit selection: ranked candidates, the first one to pass the gate ships, and bucket-disjoint multi-ship is allowed. Fork-on-conflict keeps locally useful edits in a new variant. But this is selection of proposed edits into one evolving harness, not promotion of individually validated lifetime traits from a somatic store into a heritable genome. |
| 6. Multi-generational, frozen weights | **Partial** | Weights are frozen in the main experiments, over ≤15 sequential rounds (versions) of one harness. There are no parent → descendant lineages with distinct lifetimes, no archive-based parent selection and no "inheritance" vocabulary. Variant forks are a routing ensemble, not descendants. The co-evolution arm deliberately *updates weights* (GRPO), which is the opposite of HeritAgent's premise. |

**Vector:** 1 P · 2 – · 3 P · 4 P · 5 P · 6 P. **Does not implement the full chain.**

## Fair assessment both ways

**Where HarnessX is stronger than HeritAgent's current state:**
- It has a working, formally typed and substitutable harness substrate, evaluated on 5 public benchmarks × 3 model families with a cross-family meta-agent.
- Its gate logic sits outside the optimizer ("LLM subagents explore … deterministic gates determine what ships"), which matches the "gate outside the genome" invariant in ledger §5 item 6.
- It writes pre-registered, falsifiable per-edit predictions and tracks hit rates.
- It honestly documents three gate failure modes with case studies.

**Where its evidence is weaker than its framing:**
- All numbers are in-sample peak values on the evolution set (authors' own caveat).
- Three seeds are listed, but no variance or CI is reported per cell.
- Ensemble routing chooses variants by prior success on clusters of the *same* tasks, so the 87.4% figure is an in-sample routed number. My inference: it is likely optimistic.
- The seesaw "no regression" guarantee is contradicted by its own trajectories.

## Changed vs abstract-level assessment

There was no separate §2.10 entry. HarnessX appeared only in §3 item 22 ("typed harness primitives as an evolvable substrate", *per abstract*), in §2.12 as a commit-gate competitor, and in the DarwinX note's description ("per-edit gate + seesaw test with variant isolation and GRPO co-training").

- **Element 3: threat UP (abstract → confirmed and sharpened).** The typing is real and more rigorous than in AHE: runtime-validated hook contracts, singleton groups, a hashable first-class config. "Typed, substitutable, composable agent components with runtime contract checking" cannot be claimed by HeritAgent. What remains ours is *trait-level* typing (policies/skills/regulation), per-trait validation records carried with a heritable artifact, and genome-encoded regulation.
- **Element 4: threat DOWN relative to how DarwinX's note and the abstract read.** The gate is an in-sample, binary pass@2, per-edit no-regression check plus LLM critique. It has no ablation, no attribution stage and no held-out stage. The authors document that it fails, and it may partly operate retrospectively. It is weaker than DarwinX's preserve-and-extend contract and much weaker than GRASP/VaG/SAGE. It is useful to HeritAgent mainly as *negative motivation*: a "no-regression" gate on binary per-task outcomes lets regressions of −14 to −24 pp through.
- **Elements 2, 5, 6: unchanged to slightly down.** There is no lineage or inheritance framing, and the co-evolution arm unfreezes weights.
- **Net: overall threat MEDIUM, driven by element 3 only.** It is LOW for elements 2/4/6 and for the conjunction. This does not change the §1 verdict: no single work implements all six elements.

## Consequences for HeritAgent

**Do not claim as novel** (add to ledger §3 item 22, upgrading "per abstract" to "deep-read"):
- typed hook/event-contract harness components with runtime validation;
- substitution/insertion/removal algebra over typed processors;
- per-edit change manifests with falsifiable predicted unlock/at-risk sets and attribution signatures;
- deterministic gating decoupled from LLM critique;
- fork-on-conflict variant isolation with per-variant regression scope;
- the RL-pathology framing (reward hacking / forgetting / under-exploration) of symbolic agent self-improvement.

**Naming hazards:** "seesaw constraint", "AEGIS", "operational mirror", "foundry", "Darwin Agent Team". Do not reuse them for CIG stages or EAG components. "AEGIS" is also a generic acronym; check before using it anywhere.

**Citable motivation for CIG design:**
- (i) Binary per-task outcomes (pass@k) mask success-probability drift, so per-edit no-regression gates miss accumulated regressions (§6.6, §7.2, §7.6).
- (ii) Composite multi-lever edits are credited as one unit, so per-trait ablation is needed to know which part helped.
- (iii) Self-evolution scored only in-sample, with peak reporting, carries selection bias. This is a direct reason for the held-out stage as a gate condition, alongside DarwinX's 31.7-point proxy overfit.

## EXP-0001 / CIG implications — each **requires ADR** unless marked otherwise; nothing was changed

1. **Arm B′ (performance-gated inheritance), "seesaw" operationalization (requires ADR).** HarnessX gives a published, simple, regression-only variant: admit iff manifest/smoke checks pass and no previously solved protected task flips to failing under pass@2. It could be pre-registered as the *weakest* B′ variant, or as a CIG ablation "regression stage only". GRASP-style net-gain + regression budget remains the stronger B′ default from ledger §5 item 1.
2. **CIG regression stage: cumulative, not per-edit (requires ADR).** Compare each candidate genome against the *original* protected set and the founding ancestor, not only against the immediate parent. Track a lineage-level cumulative regression budget so sub-threshold coupling cannot accumulate across generations. This is the Telecom −14.0 pp and Global −24.3 pp failure.
3. **CIG statistics: per-item success rates over n rollouts with a paired test, not pass@k binary flips (threshold change requires ADR; reporting format does not).** HarnessX's authors concede that pass@2 "mask[s] sub-threshold success-probability drift". This supports ledger §5 item 4 (SAGE-style paired test with abstain; n01/n10 reporting). The ±5% "noise threshold" is an example of a tolerance that hides regressions and should not be copied.
4. **Single-trait granularity / factorial ablation for composite candidates (requires ADR).** A candidate that bundles several traits (e.g. tool + prompt + config) should be either split, or ablated per component in CIG stage 2, before assimilation.
5. **Expression check in the replay stage (adoptable with citation; making it a hard gate condition requires ADR).** HarnessX's `attribution_signature` is a trace feature that must appear if the trait fired. It is a cheap precondition: a trait gets no credit on episodes where it was never expressed.
6. **Pre-registered per-trait predictions (adoptable; no ADR for logging).** Record predicted unlock/at-risk task sets in each CIG record and report hit rate. Note that HarnessX's hit rate ignores collateral damage on un-predicted tasks; CIG should not.
7. **Reporting (reporting format, no ADR unless metrics change).** Report final-generation and held-out numbers, not peak-over-rounds. If any selection over rounds or variants happens, keep the selection set and the evaluation set separate.
8. **Out of scope for EXP-0001 (note only).** Fork-on-conflict variant pools relate to future population/H6 designs and to genome-conflict handling. Cross-harness GRPO (weights updated from harness traces) is a contrast to the frozen-weights premise; do not add it.
