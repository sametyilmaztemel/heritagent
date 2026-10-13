# Self-Healing Harness — Runtime Oversight of Agent Self-Modification — detailed notes (Round 3)

**Paper:** *Self-Healing Harness for Runtime Oversight of Agent Self-Modification*. Authors: Sina Tayebati, Divake Kumar, Nastaran Darabi, Ranganath Krishnan, Amit Ranjan Trivedi. Affiliations listed in the HTML: University of Illinois Chicago; AI Labs, Capital One. arXiv:2609.24130v1 [cs.AI], submitted Mon 21 Sep 2026, CC BY 4.0. Only v1 exists; no venue or comments field, so it is a preprint. Tracing/evaluation infrastructure is "PandaProbe (Tayebati 2026)", linked to github.com/chirpz-ai/pandaprobe; no release of the Harness code itself is stated in the text. Reviewed 2026-10-13 by Claude (third agent), pending Research/Critic cross-review.

**Read depth: full text.** I downloaded arxiv.org/html/2609.24130 with curl and converted it to text, keeping the LaTeX `alttext` of formulas. I read the main text (Introduction, Background and Related Work, The Self-Healing Harness, Experimental Setup, Results RQ1–RQ5, Limitations, Conclusion) and the Technical Appendix (Experiments not run; Metric and interval definitions; Mechanism details; Algorithms 1–3; Notation; Hyperparameters; Comparison to prior mechanisms, Table 7; Additional result tables 8–11). I skimmed the reference list. Metadata are from the abs page. I did not check the PDF. Model names below are as written in the paper; I did not verify them. Quotes are short and verbatim. I recomputed the rule-lifecycle totals (Table 10: 2,306 = 1,244 + 1,062, consistent per run) and the retirement split (211 + 172 = 383). The Table 2 promotion sub-counts sum to 419, while the table header says "Promote (420)". This is a minor inconsistency.

## Mechanism

- **Framing.** Persistent self-modification as *admission control*: "The agent may propose changes to its operating instructions, while an external runtime gate controls persistence." The weights π are fixed. Only the rule set R in the policy-inducing context changes: c(R) = c0 ⊕ ι(R), where ι(R) is an index of rule files, and rule bodies are pulled by tool call only ("pull model").
- **Detect.** A per-metric trajectory gate over per-turn trace scores. It keeps a running peak and a counter of traces since the last gain. A stall fires when z_j ≥ w and the peak is below θ. A regression fires when the score drops below the peak by more than δ_drop. Settings: w = 10, θ = 0.5, γ_gain = 0.02, δ_drop = 0.15. Metrics come in tiers:
  - tier 1: task completion and an embedding-based coherence score;
  - tier 2: LLM-judged tool-call and argument correctness;
  - tier 3: explanatory, off in the study;
  - tier 0: an optional outcome verifier.
  Only a tier-2/tier-0-corroborated "breach" becomes a replayable case.
- **Notice / Heal.** A notice is written to an external workspace. The agent pulls it, inspects the trace and writes a candidate natural-language rule r (text, rationale, scope, tags). Candidates are de-duplicated by normalized text. Rules are partitioned into free-form scope files.
- **Validate (Algorithm 2).** A candidate gets "provisional execution authority" and becomes persistent across episodes only on promotion.
  - *Replay path:* replay ≤ 3 failure cases matched to the rule's signature and ≤ 2 protected cases, "not rank[ed] … for semantic proximity", under c(R ∪ {r}). Re-score on M1 ∪ M2.
  - Promote iff ∃i: Δ_ĵ(s_i) ≥ δ_prom, AND ∀i,j: Δ_j(s_i) > −δ_reg, with δ_prom = δ_reg = 0.05. Non-regression takes precedence. Inconclusive replay falls back to a forward trial after ≤ 3 attempts.
  - **The replay delta is Δ_j(s_i) = m′_j(s_i) − m_j(s_i), "with m′ the replay score and m the score recorded when the case was captured"** (Eq. 12). The comparison is against a *historical* score, not a concurrent re-run of the incumbent context c(R). So Δ mixes the rule's effect with run-to-run noise (temperature control is unavailable on the endpoints) and with any rules admitted since capture. The authors state: "The test is a bounded empirical procedure, not a certificate of causation".
  - *Forward path:* promote if the flagged rate over the next n = 3 sessions is 0 or falls by ≥ δ_prom from p0, with p0 = 1 when there is no history. The authors call it "materially weaker: three sessions are unmatched evidence, there is no protected-case component … and the default p0 = 1 makes promotion comparatively easy."
- **Regression guard.** This is a corpus-level re-test. It replays the whole captured evaluation corpus under the current active set and fails it if any case drops by ≥ δ_reg on any metric. It covers only captured breach cases.
- **Lifecycle.** Rules move candidate → active → retired. "Each promotion records the case, metric, and improvement that justified it."
- **Threat model.** The agent is non-adversarial and cannot edit the evaluator, the gate, the replay corpus or the promotion record.
- **Vocabulary.** A text search found no occurrence of "genome", "lineage" or "descendant". "Causal" appears only in a cited reference title. "Causation" appears only in the disclaimer above. "Generation" means rule or update generation. "Inherits" refers to the guard inheriting evaluator reliability.

## Results

- **Setup.** A paired A/B design. Baseline is the unmodified agent loop; Harness is the same loop plus Detect/Notice/Heal/Validate. Benchmarks and splits: AppWorld test_normal (168 tasks), Terminal-Bench sample@2.0 (10), τ²-bench airline (50) and retail (114). Four models, as named: gpt-5.6-terra, gpt-5.6-luna, claude-haiku-4.5 and claude-sonnet-5. k = 4 trials, 100-turn budget, **seed 1 only**, 16 matched pairs. Each split is "processed once as a continuous adaptation stream without a separate learning and evaluation partition". Statistics: paired task-level percentile bootstrap (10k) on the task-completion score s̄.
- **RQ1 (Table 1).**
  - The Harness has higher s̄ in 16/16 pairs. Only 2/16 CIs exclude zero:
    - AppWorld gpt-terra 0.701 → 0.723, CI [+0.006, +0.040], p = .01;
    - τ² retail sonnet-5 0.669 → 0.701, CI [+0.003, +0.061], p = .03.
  - pass^k is higher in 12/16 and tied in 4. pass@1 is higher in 14/16, with one tie and one loss.
  - Effects are small (τ² +0.012 to +0.032). Most are "directional".
  - Fair note: a 16/16 sign pattern would be unlikely under a symmetric null if pairs were independent (two-sided sign test ≈ 3×10⁻⁵, my calculation). But pairs share tasks and models, and there is a single seed, so this is suggestive only.
- **RQ2: the 55% datum (Table 2).**
  - 2,306 rules were proposed: 1,244 promoted and 1,062 retired (Table 10).
  - Of **383 replay-decided retirements, 211 (55%) "improved their triggering failure but regressed protected behavior"**. The other 172 showed no target improvement.
  - Replay-decided promotions: 420 (sub-counts sum to 419).
  - "Replay-decided verdicts are a subset of the totals above, the remainder having been resolved by forward trial." By subtraction, **≈ 824 of the 1,244 promoted rules (≈ 66%) were promoted by the forward path, which has no protected-case test**. This includes all 70 Terminal-Bench promotions, where replay is unavailable and there were zero retirements.
  - Denominators matter. 211 is 55% of replay rejections, but **26% of replay-decided verdicts (211/803)** and 9% of all proposals.
  - The authors say these are "detected conflicts under sparse protection, not an estimate of total collateral-regression incidence".
  - They do not control for noise. There is no null or placebo rule, no replay of the incumbent context, and "repeated-replay agreement of verdicts is unmeasured". The appendix concedes that "some protected-case drops might be evaluator noise".
- **RQ3.**
  - Promotion rates: 0.34–0.44 on AppWorld and 0.41–0.63 on τ².
  - Replay share is 84–92% on AppWorld and τ² retail, 58–72% on τ² airline (a turn/message-hop budget mismatch truncated replays) and 0% on Terminal-Bench.
  - The authors say this variation is observational and confounded.
- **RQ4: accumulated-set risk (Table 4, five configurations).**
  - In AppWorld gpt-luna, the paired Δs̄ declines from +0.017 (Q1) to −0.016 (Q4), slope p = 0.02 (permutation). Over the same period the active set grows from 13 to 70 rules and input tokens per turn rise from 1.50× to 2.52× Baseline.
  - "Every rule passed the gate individually, so the harm is a property of the corpus rather than of any member."
  - The corpus guard could not see it, for three reasons:
    - coverage: it replays only captured cases;
    - granularity: it checks per-case drops of ≥ 0.05, while the decline was a diffuse aggregate shift;
    - mechanism: it may not reproduce retrieval-dilution state.
  - The other four configurations are mixed (three positive slopes, none significant).
- **RQ5.**
  - Context usage is 1.25× Baseline. Cost is 1.72× Baseline including repair. Wall-clock time is 1.7–7× Baseline.
  - In development, judge bias against the Harness's own meta-tools was found and reduced (14% → 0.17% of rationales). Both arms use the corrected prompts.
- **Stated limitations.**
  - "The primary missing comparison is between evidence-gated and immediate admission of the same agent-authored rules (arm B′), with the downstream corpus guard fixed. The Baseline–Harness comparison therefore does not isolate the effect of pre-admission validation."
  - Not run either: no-barrier, inlined-rule, tier-3, replay-only and forward-only arms.
  - No held-out transfer: "the protocol measures online adaptation rather than held-out transfer".
  - A single seed despite path dependence.
  - Unmeasured sensitivity to δ_prom, δ_reg and w.
  - ≤ 2 unranked protected cases per round.
  - Text-only rules and a non-adversarial agent.
- **Novelty claim (Table 7).** The paper claims the combination of five properties on an agent-authored update: P (persistent), S (self-generated), V (validated), R (regression-guarded) and T (runtime-triggered). It compares 11 systems (Reflexion, Voyager, Generative Agents, ExpeL, AutoManual, AWM, A-MEM, RAG memory, STOP, DGM, Constitutional AI). It does not cite the 2026 pre-persistence-gate cluster in ledger §2.10 (VaG, GRASP, SAGE, SkillBoost, HarnessEvolve, …).

## Overlap vs HeritAgent chain (full-text judgement)

| Element | Coverage | Evidence |
|---|---|---|
| 1. Lifetime acquisition | **Full** | The agent authors behavioural rules from its own detected failures during a live deployment stream ("Heal: Agent-Authored Behavioral Updates"; "earlier tasks … affect later behavior through proposed, validated, promoted, and retired rules"). |
| 2. Somatic / germline separation | **Partial (weak) — raised from "–" at abstract level** | There is a two-tier authority distinction inside one rule set: candidates have "provisional execution authority during evaluation" and acquire "persistent cross-episode authority only after" promotion. Retired rules leave the active set. This is a probation tier of the kind ledger §3 item 17 already concedes (VaG Cold/Warm/Hot, SAPO, LOGOS). It is *not* a boundary between a heritable artifact and lifetime state. R = active ∪ candidate is one live store, "persistent" means across episodes of one deployment, and there is no second store that a descendant receives. |
| 3. Typed executable genome | **Partial (weak) — raised from "–" at abstract level** | Rules carry metadata (text, rationale, scope, tags) and live in scope files. Importantly, each promotion "records the case, metric, and improvement that justified it". That is a per-rule validation and provenance record, one feature of the EAG. Otherwise the artifact is untyped. There is one kind of item (a natural-language rule), no policy/skill/regulation types, nothing executable beyond prompt text, no schema or versioning of the set as a unit, and the scopes are free-form. |
| 4. Causal validation gate | **Partial (weaker than SAGE and weaker than EXP-0001's CIG)** | Per-candidate replay with target improvement plus multi-metric non-regression on ≤ 2 unranked protected cases, plus a corpus-level re-test. Against the CIG stages: (i) "replay" compares to *historical* captured scores, not a paired with/without re-run, from one stochastic execution, with no noise control; (ii) there is no controlled ablation of the rule in the context of the active set; (iii) there is no attribution under interactions (RQ4 shows the interaction problem but no stage addresses it); (iv) there is no held-out generalization stage, in or out of scope, and no learning/evaluation partition; (v) the regression protection is real but sparse. About two-thirds of promotions came through the forward path, which has no protected-case test. The authors explicitly decline the causal reading: "not a certificate of causation". |
| 5. Selective assimilation | **Partial** | Rules are selectively promoted into the persistent active set, with retirement. The target is the live working context of the same agent, not a heritable genome separate from lifetime state, and there is no assimilation event between stores. |
| 6. Multi-generational, frozen weights | **Partial (frozen weights only)** | "π stays constant while the rule set R evolves"; the paper stresses compatibility with closed-weight models. There are no generations, descendants, offspring or inheritance. The frozen corpus is never transferred, even to held-out tasks (listed under "Experiments not run"). |

**Implements full chain: no.** One element is full, five are partial, and none is absent. Elements 2, 3 and 6 are weak partials. Compared with the abstract-level row in ledger §2.10.18 (1 F · 2 – · 3 – · 4 P · 5 P · 6 P), the full text *raises* elements 2 and 3 from absent to weak partial. The reasons are the provisional/persistent authority tiers and the per-promotion evidence records.

## Does the full text change the abstract-level assessment?

**Element ratings: slightly up (2 and 3: – → weak partial). Threat to the chain claim: down from MEDIUM to LOW. Relevance to EXP-0001 design: up, with a correction to how the ledger uses the 55% datum.**

- **Why the threat goes down.**
  - Scope is one agent within one online deployment stream. Persistence means cross-episode authority for the same agent. There is no descendant, no genome as a heritable unit and no held-out evaluation.
  - The gate is single-shot replay against historical scores. It has no ablation, no attribution, no held-out generalization and sparse protection, and two-thirds of admissions skip the protected test entirely.
  - The headline gains are small. Only 2/16 CIs exclude zero, and there is a single seed.
  - Most importantly, **the paper itself did not run the comparison that would show gating beats immediate admission (its own "arm B′")**. So it gives no evidence on the H2-type question "does a gate beat ungated persistence". It shows only that Harness beats no adaptation, directionally.
  - A reviewer can cite it against "gating persistence of self-authored updates with target-improvement + non-regression" (already conceded in ledger §3 items 9, 11 and 17). It cannot be cited against the conjunction: a somatic store → typed genome, multi-stage trait-level evidence, and inheritance across agent generations.
- **Fair the other way.**
  - It is a clean, deployable runtime design.
  - It is unusually candid about its limits. The appendix pre-specifies exactly the B / B′ / B″ decomposition needed to isolate gate value.
  - It gives the first *detected* evidence in this cluster that **individually admitted rules can degrade performance as a set** (RQ4, AppWorld gpt-luna, p = 0.02), that a per-case corpus guard does not catch it, and that the cause may be context dilution or rule interference. This directly supports CIG's interaction stage and genome-size/bloat outcomes (H5). It also warns that CIG's per-trait admission could pass traits that jointly harm.
  - Its per-promotion evidence record and its separation of gate, evaluator and replay corpus from the agent's tool surface resemble two EAG/CIG design choices: per-trait validation records, and gate logic outside the heritable artifact. Ledger §3 items 18, 22 and 26 already concede analogues of both; add this paper as another citation.
- **Correction to the ledger's use of the 55% datum (§2.10.18, §3 item 15).** The ledger states the datum without its denominator and treats it as support for CIG's regression stage. The full text supports a narrower statement: among replay-decided *rejections*, 211/383 (55%) improved the triggering failure while dropping ≥ 0.05 on some metric of ≤ 2 protected cases. That is 26% of replay-decided verdicts, measured against historical scores from single stochastic replays, with no noise or placebo control. The authors themselves treat it as "detected conflicts", not an incidence estimate, and concede possible evaluator noise. It is still informative, but cite it as such.

## Consequences for HeritAgent

No design was changed. Items marked *requires ADR* are for the Builder/ADR owner.

1. **Naming hazard: "arm B′".** This paper defines "B′" as *immediate admission of agent-authored rules with the same downstream corpus guard* and "B″" as fully uncontrolled persistence. HeritAgent's ledger §5 item 1 uses "arm B′" for *performance-gated inheritance*, which is nearly the opposite. In any writing that cites this paper, rename HeritAgent's arm (e.g., "arm P — performance-gated") or always say "HeritAgent arm B′ (performance-gated)". *Requires ADR only if the arm label in ADR-0002 is changed; otherwise a wording note.*
2. **Arm B′ (HeritAgent sense) operationalization (f), "Harness-style = CIG stages 1 + 4 only". *Requires ADR (ADR-0002 amendment).***
   - Admit a candidate iff replay on its source cluster improves success AND the T_reg regression ≤ τ_r. Drop stage 2 (controlled ablation / contribution) and stage 3 (in-/out-of-scope generalization).
   - This is the cheapest B′ to build, because it reuses EXP-0001's own stage implementations. It isolates exactly the contribution of the ablation and generalization stages, which is the part of CIG that the published cluster lacks.
   - It complements options (a)–(c) in ledger §5 item 1, RSEA (d) and SAGE (e). If two B′ variants are affordable, pair (e) SAGE-style (strongest statistics) with (f) (closest to published runtime practice).
3. **Paired concurrent contrast, not historical baselines. *No ADR needed — confirms the current design; record as a rationale.*** This paper's Eq. 12 compares a candidate replay with the score recorded at capture, which confounds the rule's effect with run-to-run variance and with corpus drift since capture. EXP-0001 stages 1–2 already run with vs without at temperature 0, one deterministic execution per (instance, condition). Keep that, and cite this paper as the motivating contrast. Also check that EXP-0001's stage 4 "regression ≤ 3pp" is computed against a *re-run* of the current genome on T_reg, not against G0's stored T_reg outcomes. The design text ("re-run T_reg with genome ∪ trait") does not say what the comparator is. *Requires ADR only if the comparator is changed.*
4. **Joint-genome check before committing G1. *Requires ADR.*** RQ4 shows that per-candidate admission does not control set-level degradation. EXP-0001 stage 4 tests each trait as genome ∪ trait, with pairwise co-expression only when there are ≥ 2 validated skills. Add one pre-registered *joint* T_reg (and T_gate) run of G0 ∪ {all admitted traits} before G1 is committed, with a pre-registered rule for what happens if it fails (e.g., greedy removal by lowest contribution). Without it, CIG could reproduce this paper's corpus-drift failure across generations, where it would compound.
5. **Gate noise calibration: a placebo trait. *Requires ADR.*** Neither this paper nor most of the cluster measures how often a gate rejects or admits a no-op change. Inject one or two length-matched, semantically irrelevant "placebo traits" into arm C's frozen candidate set and report their pass rate per CIG stage. At temperature 0 this measures prompt-perturbation sensitivity rather than sampling noise. That is still the relevant false-pass/false-reject rate for H2 and H5. The cost is about 130 episodes per placebo per seed.
6. **Protected-set size and selection. *No ADR — supports current design.*** This paper's ≤ 2 unranked protected cases are its main blind spot by its own account. EXP-0001's T_reg is 40 tasks drawn from what G0 solved, which is far stronger. Report the regression denominator explicitly (cf. SAGE note item 3) and per-family breakdowns, so the comparison with this paper's per-metric δ_reg = 0.05 rule is clear.
7. **Bloat / context cost as a mechanism for H5. *No ADR — reporting.*** RQ4's decline came with input tokens per turn rising from 1.50× to 2.52× Baseline. EXP-0001 already treats prompt overhead as part of the arm B treatment. Report per-arm expression tokens alongside quartile-of-execution performance, so a B < C result can be split into "harmful traits" vs "context dilution".
8. **Wording.** Do not present any of the following as HeritAgent's contribution: runtime admission control of self-authored updates; target-improvement + non-regression admission; provisional-vs-persistent authority tiers; corpus-level re-testing of an accumulated set; per-promotion evidence records; or keeping gate and evaluator outside the agent's tool surface. Cite "Self-Healing Harness (Tayebati et al., arXiv:2609.24130)". When contrasting, say "per-candidate runtime admission into one agent's live rule set within a single deployment stream, without held-out or cross-generation evaluation".

## Suggested ledger update (§2.10.18; pending cross-review)

- Mark the entry deep-read (full text). Lower the threat to the chain claim from MEDIUM to LOW.
- Chain coverage: 1 F · 2 P-weak (provisional/persistent authority tiers) · 3 P-weak (per-promotion evidence records; untyped NL rules) · 4 P (historical-baseline replay; ≤ 2 protected cases; ~66% of promotions via forward trial with no protected test; "not a certificate of causation") · 5 P · 6 P (frozen weights only).
- Re-word the 55% datum in §2.10.18 and §3 item 15 with its denominator and caveats (211/383 replay rejections = 26% of replay-decided verdicts; detected under ≤ 2 protected cases; no noise control).
- Add a §3 item: "set-level degradation despite individually admitted updates, and the inability of per-case corpus guards to detect it" (this paper, RQ4).
- Add to the naming hazards: "B′" (this paper: immediate admission + guard).
- Add to §5: B′ option (f); the joint-genome check; the placebo-trait calibration.
