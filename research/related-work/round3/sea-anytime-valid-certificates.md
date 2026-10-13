# SEA — Self-Evolving Agents with Anytime-Valid Certificates — detailed notes

**Paper:** *Self-Evolving Agents with Anytime-Valid Certificates* — Biswa Sengupta (sole author). arXiv:2607.00871, cs.AI, v1 1 Jul 2026 (the only version listed; 63 KB). The abstract page has no comments field, so no venue is claimed: treat it as a preprint. The paper says there is an "executable reference implementation from which all pseudo-code in this paper is distilled". I found no code URL in the HTML text. Reviewed 2026-10-12 by Claude (third agent). **Pending Research/Critic cross-review.**

**Read depth: full text.**
- I downloaded the arXiv HTML (LaTeXML) and converted it to plain text, rendering math from its alttext. I read §§1–11 and Appendices B–E in full.
- Appendix A (pseudocode for Alg 1–11) I searched rather than read line by line. I looked at the Alg 4 (SGM-CS), Alg 5 (SDC-QD), Alg 10 and Alg 11 lines about accept, commit and certificates.
- All numbers below come from the rendered tables and text (Tables 3–4, §8, §9, App. D).
- No lineage, generation, population or inheritance section exists. Searching for *lineage, generation, inherit, offspring, descendant, parent, child, germ, soma, population, genome, heredit* finds only incidental uses: "statistical descendant of Schmidhuber's Gödel machine", "inheriting the abstention semantics", "corpus generation", "generation-level repair", and "children"/"parents" inside the beam-search pseudocode.

## Mechanism

**Four layers (§3, Fig. 1).** The deployed policy is π_t = L0 ∘ L1^(t) ∘ L2^(t).
- **L0** is a frozen LLM, "Never updated".
- **L1** is a "steering adapter". In practice it is a softmax over a *finite set of hand-written natural-language steering directives* appended to the system prompt, with logits θ as the only trainable parameters. It is updated by a closed-form REINFORCE score-function gradient. It is "not weight-fine-tuned in any reported run". A LoRA/prefix/soft-prompt version is described for open weights but not used.
- **L2** is a "mutable, versioned harness": system prompt, tool definitions, step and exploration budgets, memory, the grown abstraction (macro) library, and a pipeline of adopted self-repair primitives. "Every structural edit changes the harness's identity, so the policy version is well defined at all times". A policy's version is (θ, L2).
- **L3** is the loop controller. It "Holds certificates, gates, budgets, and archives; never part of the agent's forward pass."

**Certificate ledger (§3, App. C).** Every controller round emits an immutable certificate with these fields:
- `algorithm`, `round`;
- `decision ∈ {accept, hold, reject, nsf}`, where nsf means "no solution found" (abstain);
- `delta_spent`, `cumulative_delta`;
- a metrics map and a note.

The composite controller merges these into one ledger row per round. A per-decision JSONL audit trail is used to attribute flips and regressions to named mechanisms.

**Five L3 controllers (§4).** Each wraps a published guarantee. The paper is explicit that "their composition is an empirical construct we do not prove correct under the endogenous loop".

| Controller | What it does |
|---|---|
| Alg 1 PPB-CL | Time-uniform PAC-Bayes forgetting gate, a performative trust region on θ, and an anchor-task re-evaluation. |
| Alg 2 PNMP-A | Real-data-anchored Nash mirror-prox, with an e-process drift gate on a learned preference model. In the results it acts as "the strategy-directive learner". |
| Alg 3 PA-COCOA | Counterfactual credit assignment over directives, driven by parameter-free coin betting. |
| **Alg 4 SGM-CS** | The certified self-edit gate. It builds directly on the Statistical Gödel Machine (Wu et al. 2025, arXiv:2510.10232; see the lead below). It gates harness edits. |
| Alg 5 SDC-QD | Stitch/DreamCoder MDL macro mining from successful trajectories, with a MAP-Elites archive and a downstream-utility gate. |

**How Alg 4 SGM-CS decides (§4.4, Alg. 4).**
- The frozen LLM proposes edits to its own harness by cycling three knobs: the force-edit threshold, the step budget, and the strategy-guidance text.
- Baseline and candidate harnesses run concurrently on the same task batch under **common random numbers** (aligned per-task seeds, isolated working copies).
- The decision is a lower confidence bound (LCB) on the paired process-reward gain. It uses a paired-difference confidence sequence at the round's CTHS level δ_k, inflated by ε·W1 (an exact 1-D Wasserstein "performative" shift).
- A wild-bootstrap trend test widens the bound when the baseline is non-stationary.
- A one-task pre-gate pilot can reject early as nsf.
- Familywise error over an unbounded stream of edits is controlled by a horizon-free, normalised "confirm-triggered harmonic spending" schedule, δ_k = δ0 / (Z·k·log²(k+1)), with δ0 = 0.05.
- "By construction the protocol favors safety over progress—it may rationally abstain (nsf) indefinitely."
- Even when the harness gate abstains, a shadow-best patch is kept, and rejected harnesses that produced strong patches are requeued (at most 2 retries).

**Verifier tier (§5, Alg 6–10).** These mechanisms supply signal and variation to the controllers; they are not controllers themselves.
- A graded milestone verifier, with an error-progression bonus.
- A closed-loop `run_tests` tool.
- A process reward: the best verifier state reached during the episode.
- Best-of-N, which was *removed* from the live stack.
- Verified one-line micro-step beam search (Alg 7).
- **Self-authored reproduction oracles (Alg 8).** These are written from the issue text alone and admitted only if they fail on the unpatched base. The held-out gold tests are "reserved for terminal measurement" and "never steer" the search.
- Verified self-repair of harness operators, adopted by measured fix-rate (Alg 10).
- A slow loop that distils verified traces into a LoRA is "designed but not trained in any reported run".

**Library growth and pruning (App. D, §7).**
- The macro library is capped at 3 entries per (repository, status) context and 12 in total.
- A trailing-window retirement pass (24 rollouts, bad-rate 0.5) prunes it.
- A macro is admitted only if downstream lift is ≥ 0.05, "not merely an MDL gain".

## Experimental evidence (§8–9)

**Setup.**
- SWE-bench Verified, a seeded random draw of **52 instances**: 24 Django, 27 Matplotlib, 1 Flask.
- Official execution-based grader. **One run per cell**; "Each instance is solved once".
- Four bases crossed off/on (gemma4:31b, qwen3.6:27b, gpt-5.4-mini, gpt-5.5), plus GLM 5.2, which was run only as no-op control and full suite.

**Important configuration detail:** the reported full stack ("Algorithms–A") **disables Alg 4**, the certified harness-edit gate in the title, "for wall-clock cost": a scheduled Alg 4 round costs about 90 minutes, against about 7 minutes for the rest of the stack. Best-of-N is also removed.

**Table 3 (resolved out of 52, off → on):**

| Base | Off | On | Δ |
|---|---|---|---|
| Gemma | 18 | 22 | +4 |
| Qwen | 24 | 25 | +1 |
| Gpt-mini | 25 | 29 | +4 |
| Gpt | 28 | 34 | +6 |
| GLM 5.2 | 24 | 28 | +4 |

For GLM 5.2 the "off" cell is the no-op control. For Gpt, the no-op control (directives on, algorithms off) scores 29, so the deconfounded gains are **+5 (Gpt, 29 → 34) and +4 (GLM)**.

**Table 4 (Gpt single-algorithm runs, resolved out of 52):**

| Config | Resolved | Note |
|---|---|---|
| Baseline | 28 | |
| Control | 29 | |
| Alg 1 | 30 | |
| Alg 2 | 34 | |
| Alg 3 | 31 | |
| Alg 4 | 36 | "0 accepts — ran control config" |
| Alg 5 | 28 | "≈ 0 accepts" |
| Alg 6 | 26 | |
| Alg 7 | 32 | |
| Alg 8 | 32 | |
| Alg 9 | 29 | |
| Alg 10 | 30 | |
| Full | 34 | |

**§9.2 (event logs).**
- The live value is "carried by Alg 2/Alg 3 (directive shaping) and Alg 7/Alg 8 (verified search and self-oracle)". The "gated, library, and forgetting controllers Alg 4, Alg 5, Alg 1 contribute less".
- run_tests calls rise by about 50%.
- Regressions are few: 4 p2p-flagged instances for the full stack, 3 for the control.
- 17/52 instances are solved by every config and 10 by none.

**My inference, not stated by the author as a variance estimate:** the Alg 4 cell ran the control configuration but scored 36, against 29 for the control run. So two runs of effectively the same configuration differed by 7 out of 52. That is larger than the +5 deconfounded gain. The author calls the 36 "a high draw" and reports "magnitudes rather than significance" (§10).

**Stated limitations (§10):**
- The compositions of guarantees are open conjectures.
- The performative sensitivity ε is a hyperparameter that cannot be estimated online.
- The gates are conservative ("Alg 4 guarantees safety, not progress").
- The controllers "cannot manufacture capability" the base lacks.
- Single runs only; no multi-seed results.
- One harness only.
- The distillation step is not trained.

The forgetting threshold τ_forget is "raised to 1.1 above the anytime floor for live multi-step SWE runs" (default 0.1) (App. D). I did not verify the scale of the bound, so I cannot say how binding the Alg 1 gate is in practice.

## Overlap vs HeritAgent chain (full-text judgement)

| Element | Coverage | Evidence |
|---|---|---|
| 1. Lifetime acquisition from trajectories | **Partial** (abstract level said absent) | Within one run over the 52-instance stream, the agent adapts from its own rollouts. Alg 2/3 make 52 policy updates each to the directive softmax. Alg 5 SWE variant: "successful trajectories, encoded as operation sequences, become the corpus, and the mined macros become callable harness entries", and failure signatures become anti-macros. Alg 10 adopts repair primitives by measured fix-rate. Partial, not full: the acquired content is mostly a *re-weighting of a fixed hand-written directive set*, and the one genuine trajectory-to-skill path (Alg 5 macros) committed "≈ 0" macros in its ablation. |
| 2. Somatic/germline separation | **Absent** | There is one shared evolving policy (Alg 11: "over one shared evolving policy"). Accepted edits go straight into the deployed harness. The nearest analogues are weak: incumbent-vs-candidate concurrent deployment, a requeue pool for rejected-but-promising harnesses, and a shadow-best *patch*. None is a staging store of acquired traits awaiting promotion into a separate heritable store. No heritable store exists. |
| 3. Typed executable genome | **Partial** (moderately strong analogue) | There is a layered, versioned policy object (θ, L2). The harness has typed components: prompt, tools, step and explore→edit budgets, memory, macro library, repair pipeline. These loosely map onto policies, skills and regulation; the explore→edit threshold and step budgets are regulation-like knobs. Macros are "a composition within a fixed typed λ-calculus". Version identity changes on every structural edit, and an immutable certificate ledger exists. **Missing:** a declared schema of traits, per-trait validation records attached to the artifact, lineage or parent pointers, and any role as a heritable unit. Certificates are per controller-round, not per trait. |
| 4. Causal validation gate | **Partial** in design, **weak in the reported evidence** | Alg 4 is an interventional, paired (common-random-number) baseline-vs-candidate comparison per edit, with an anytime-valid LCB, familywise error budget, performativity correction and abstention. Alg 1 re-runs frozen anchor tasks as a regression gate. Alg 5 needs a positive downstream lift, and Alg 10 a measured fix-rate. **Missing:** controlled ablation of a component *inside* the evolved composite; attribution under interaction; a held-out generalization stage inside the gate (the gate's tasks are the round's own batch, and the held-out grader is terminal-only by design); multi-environment replay; and any use as a heritability prerequisite. **Empirically, the certified gate is barely exercised:** Alg 4 is off in the full stack and accepted 0 edits in its only run. The paper's attribution evidence is experiment-level (no-op control and single-algorithm cells), not gate-level. |
| 5. Selective assimilation | **Partial** | Edits, macros and repair primitives enter the persistent harness only through gates. The library is capped and self-prunes by retirement, which is relevant to H5 genome-size control. But this is selective persistence into the *working* policy within one run, not assimilation into a separate heritable genome. |
| 6. Multi-generational, frozen weights | **Partial** (frozen half full; generational half absent) | L0 is frozen and θ is a directive softmax, so no weights change in any run. There are no generations, lineages, descendants or populations. The evolved harness is never passed to a new agent with its own lifetime, and it is not transferred across bases either: each base runs its own loop. |

Coverage summary: **1 P · 2 – · 3 P · 4 P · 5 P · 6 P**. **implements_full_chain = false.**

## Does the full text change the abstract-level assessment?

- **No upgrade to HIGH.** The ledger's condition for an upgrade (a lineage or generation section) is not met. SEA is a single self-modifying agent within one run.
- **Element 1 goes up (– → P).** The full text shows trajectory-mined macros and anti-macros, plus online adaptation of the directive policy.
- **Element 4 stays P but carries less weight as *demonstrated* prior art.** The headline certified gate (Alg 4) is disabled in the reported stack and accepted zero edits when run alone. The measured gains come from directive shaping and verified test-time search, and the author says so (§9.2). As *design* prior art the gate is stronger than the abstract suggests: CRN pairing, performativity correction, a horizon-free familywise spending schedule and abstention, with explicit provenance from SGM.
- **Element 3 stays P, and is somewhat stronger than recorded.** The harness covers prompt, tools, budget and threshold knobs, a library and a repair pipeline. Version identity is defined by (θ, L2), and the gating controller sits outside the forward pass. This is the closest found analogue to "EAG as a governed artifact + CIG outside the genome". But it has no trait schema, no per-trait validation records and no heritability.
- **Net threat: MEDIUM, held at the lower end. The "highest priority / possible HIGH" flag is withdrawn.**
  - SEA threatens any HeritAgent wording that presents the following as new:
    - certified or statistically gated admission of self-modifications with an auditable ledger;
    - familywise error control over an unbounded edit stream;
    - a frozen base + evolvable harness + external gate decomposition.
  - SEA does not threaten the conjunction claim: somatic → germline promotion of individually validated lifetime traits on trait-level causal evidence, across agent generations.
- **Ledger correction for §2.10.13.**
  - The "steering adapter" is a softmax over hand-written directives, not a weight adapter. The paper says so; record it.
  - Add that Alg 4 is off in the reported full stack and accepted 0 edits in its single run.
  - The abstract's "four base models" is right for the off/on cross, but a fifth (GLM 5.2) appears with control and full suite only.

## Follow-up lead (verified on its arXiv abstract page only)

**SGM: A Statistical Gödel Machine for Risk-Controlled Recursive Self-Modification.** Xuening Wu, Shenqin Yin, Yanlan Kang, Xinhang Zhang, Qianya Xu, Zeping Chen, Wenqiang Zhang. arXiv:2510.10232 (11 Oct 2025).
- From the abstract: it admits a modification only when superiority is certified at a confidence level (e-values, Hoeffding bounds), allocates a global error budget across rounds, and introduces Confirm-Triggered Harmonic Spending (CTHS).
- It is evaluated on CIFAR-100, ImageNet-100, RL and black-box optimization. It does not appear to evaluate LLM agents, though I read the abstract only.
- This is the *origin* of the gate SEA uses. It belongs in ledger §3 item 12 next to SEA and SAGE ("e-value-gated self-edits with familywise error budget: SGM 2025; SEA; SAGE").
- Threat to the chain: low. It is a statistical gate for one self-modifying learner, with no skills, genome or generations. I did not read its full text.

## What HeritAgent must not claim as novel (additions to §3)

- A frozen base + mutable versioned harness + an external controller that holds gates and certificates and is "never part of the agent's forward pass" (SEA §3).
- An immutable per-decision certificate or audit ledger with accept/hold/reject/abstain decisions and error-spend accounting (SEA App. C).
- Anytime-valid, familywise-controlled admission of self-edits, CRN-paired baseline/candidate comparison and abstention, which are SGM 2025 and SEA. CIG may *use* these; it must cite them.
- Keeping the held-out grader out of the improvement loop, as a design principle (SEA §5.2).
- Capped, self-pruning skill or macro libraries with a downstream-utility admission bar.

## What remains different

- SEA has one agent and one run, with no agent generations with distinct lifetimes, no inheritance and no descendant adaptation cost.
- SEA has no somatic/germline split and no promotion event.
- SEA's harness is a versioned configuration, not a typed trait genome with per-trait validation records.
- SEA's gate is a statistical superiority test on an aggregate paired reward for a harness-knob edit. It has no trait-level controlled ablation inside the evolved composite, no attribution under interaction, no held-out generalization or out-of-scope stage, and no multi-environment replay.
- SEA's gate is not a heritability prerequisite.
- SEA's reported evidence covers 52 instances on one benchmark, with single runs and the certified gate inactive.

## Implications for EXP-0001 / CIG (each requires ADR; nothing was changed)

1. **Arm B′ (performance-gated inheritance, ledger §5 item 1): add a candidate operationalisation, but do not make it the primary B′. Requires ADR (ADR-0002 amendment).**
   - Option (d) would be an SGM/SEA-style certified gate: admit a candidate iff an anytime-valid paired LCB on the with-vs-without gain is > 0 at a CTHS-spent level, with no trait-level stages.
   - SEA's own Alg 4 accepted **0** edits on 52 instances. At EXP-0001 scale (20-task ablation pool, one deterministic execution per condition), a familywise anytime-valid gate at δ0 = 0.05 is likely to admit nothing. B′ would then collapse to arm A, an uninformative baseline.
   - Prefer GRASP/SAGE-style fixed-n paired gates (§5 options a–c) for B′. Report a certified-gate variant only as a secondary analysis on the frozen stage data.
2. **Multiplicity across candidates and stages. Requires ADR (threshold or statistics change).**
   - EXP-0001 tests K = 10 candidates × 4 stages at fixed per-test thresholds (stage-2 CI LB > 0 and ≥ +0.05; τ_r = 3pp), with no familywise or false-discovery control across candidates.
   - Because EXP-0001 is one-shot with a fixed n, anytime-validity is *not* required. The fixed-horizon analogue of SEA/SGM's budget is Holm/Bonferroni or a BH correction across the K stage-2 tests.
   - Pre-register whether CIG controls per-candidate error (current design) or familywise or false-discovery error. As a reporting-only step, add the count of candidates expected to pass under each regime to the §8 checklist.
   - Anytime-valid sequential testing (confidence sequences, e-processes) becomes relevant only if a later CIG version runs adaptive or early-stopped stage evaluation or open-ended multi-generation admission. In that case cite Howard et al. 2021, Ramdas et al. 2023, SGM and SEA, as the ledger already suggests for stage-5 regression thresholds.
3. **An abstain decision distinct from reject. Requires ADR (ADR-0001 / CIG record schema).**
   - SEA's certificate vocabulary {accept, hold, reject, nsf} and its `delta_spent` / `cumulative_delta` fields are a ready template for CIG records.
   - "Insufficient evidence" (for example a CI straddling the threshold, or too few valid bank instances under the feasibility fallback) should be recordable separately from "evidence of harm". This matters for H5 analyses of why genomes stay small.
4. **Run-to-run noise. No ADR needed; reporting only.** SEA's two effectively identical control-configuration runs differed by 7 out of 52 (my inference from Table 4). This supports EXP-0001's 3 mining seeds and the "C−B direction in ≥ 2/3 seeds" rule. It also argues against reading single-seed arm differences of about 5–10pp as effects. Worth one sentence in the pre-registration rationale.
5. **Gate outside the genome (ledger §5 item 6).** SEA's L3 "never part of the agent's forward pass", with gates and budgets held by the controller, is a citable precedent for the invariant that CIG thresholds and logic are non-heritable. Record it as an explicit invariant when populations are introduced. **Requires ADR** at that point.
6. **Not applicable to EXP-0001:** SEA's performativity correction (ε·W1). EXP-0001 uses fixed, disjoint ALFWorld splits, so the evaluated distribution does not depend on the deployed genome. Revisit only for curriculum or self-generated-task experiments.

## Anti-overclaim (SPEC §34)

- SEA itself is careful: "we do not derive new bounds or rates here", the guarantees under the endogenous loop are "open conjectures", and patches are "promising, never resolved".
- HeritAgent must not describe SEA as having "certified self-improvement", or as showing that its gate produced the gains. The paper shows the opposite for Alg 4.
- HeritAgent must equally not imply that its own CIG is the first statistically principled admission gate for self-modification. SGM (2025) and SEA (2026) precede it.
