# EXP-0001 pre-registration fields: DRAFT proposal for issue #6

**Status.** This is a draft from an independent third-agent researcher (Claude subagent), written 2026-10-12. It is read-only work. Nothing has been edited in the repo and nothing has been posted to GitHub.

**Who decides.** These predictions belong to the research team. The human owner and the Research/Critic agent can accept, edit or reject each field. No value here is decided, and no locked element is changed by this note.

**What this answers.** Issue #6 asks for the four open fields of `experiments/manifests/EXP-0001-design.md` §8 to be filled. It then asks for `preregistration.expected_results_filled` to be set to `true`.

**Sources.**
- Repo: `main` @ `0ce0a36` at the start of this session. The PR #28 branch `origin/issue-12-cig-gate` was read at head `c6ce4c1`; the brief named `4ca5517`, but the branch has moved since. ADR-0002 Amendment A1 is unchanged between those heads.
- **During this session the local `main` advanced to `6014bc7` ("Issue #12: Basic CIG 4-stage gate").** Its tree is identical to `c6ce4c1` (`git diff origin/issue-12-cig-gate 6014bc7` is empty). So PR #28's content, including Amendment A1, is now on main, and the code citations below hold for both.
- Sibling notes (same directory, other subagents, this round):
  - `ALFWorld-baselines.md` ("BL")
  - `EXP-0001-power-analysis.md` ("PA")
  - `EXP-0001-validity-review.md` ("VR")
- New scripts for this note, run with `venv314/bin/python -I`:
  - `scripts/exp0001_compute_estimate.py` → `.out`
  - `scripts/exp0001_prereg_stage_counts.py` → `.out`

> **Verification status (independent adversarial verification, 2026-10-13; full record in `research/methodology/VERIFICATION-2026-10-13.md`).**
> - **Confirmed:** the regression binomials (C3: neutral trait 40 % at f = 0.05, 8 % at f = 0.10, symmetric noise), the out-of-scope tolerance being void at n = 10 (C4), the zero-noise false-reject rates (C2), the replay rule (R1), and the interaction worst case of 2,700 / 8,526 episodes (V8).
> - **Confirmed with correction:** the stage-2 pass rule (C1: it extends to b ≥ 3, five compositions depend on ordering, not three, and it is more lenient than an exact sign test); the H2 T_eval MDE (C5: about 7–10 pp with informative seeds, 12.7–16.9 pp with redundant seeds under symmetric noise); the H2 regression power (C6: 12–35 % holds only at ρ = 1, it is 30–98 % at ρ = 0); the null probability of the ≥ 2/3-seed rule (C7: 0.44–0.47; not cited in this draft). VR findings V1–V7 hold but were overstated (V1 stays P0; V2–V7 are P1).
> - **Refuted:** PA's assumption that candidates are drawn independently in T2 (R8). Means are unaffected, but P(arm C empty) is understated: the optimistic cell is about 33 %, not 25 %.
> - Passages changed as a result are tagged "[corrected after independent verification 2026-10-13]". Everything else is unchanged.

**Evidence labels.**
- **V**: I read it in a repo file, a paper text or a script output during this session. "V(BL)", "V(PA)" and "V(VR)" mean the sibling note marked it V and I did not re-verify it.
- **M**: output of an explicit model with stated assumptions. It is not data.
- **U**: unverified, or my judgement.
- **S**: a subjective prior proposed for the team to adopt or change.

SPEC §34 applies to the wording throughout. "Causal" is used only for the gate's controlled interventions, and "better" only as a pre-stated hypothesis.

---

## 0. Summary

1. **The fields cannot be filled credibly for the design as currently coded.** Four problems from the sibling validity review (VR), all confirmed here in code, would make the predictions describe a different experiment:
   - **Mining gets no input.** The runtime is fixed at temperature 0 with one episode per task, and the miner needs a success and a failure of the *same* task. Under the protocol as specified, no same-instance pair can arise, so K ≈ 0. Pairs could appear only by accident. One route is different game instances that share identical task text: the pair check accepts these, because provenance carries no instance id. The other is pairing across the 3 mining seeds, which at temperature 0 gives identical trajectories anyway. The pairing policy was explicitly deferred to #13 but never pre-registered (VR finding 1, P0). [corrected after independent verification 2026-10-13]
   - **8-step cap.** The G0 planner caps every episode at 8 loop steps (`runtime/seed_g0.py:17` `"max_plan_steps": 8`; `runtime/loop.py:151`). A Final Answer or a rejected tool call also uses a step, so an episode gets *at most* 8 environment actions. The planner schema caps `max_plan_steps` at 16, so a 50-step protocol needs a schema change as well as a new planner gene. The manifest has no numeric step budget. The cap is unregistered but not hidden: it is logged in `run_started`/`budget_exhausted` and is identical across arms (VR finding 4, P1). [corrected after independent verification 2026-10-13]
   - **T_reg cannot be filled.** It needs at least 40 of 60 T_mine tasks solved by G0, which is ≥ 66.7 % success. Every anchor is between 15 and 46 % (BL).
   - **No ALFWorld adapter exists yet** at `0ce0a36` or `c6ce4c1`. G0's actual prompt rendering and step semantics are therefore undetermined, and BL shows these dominate the success level.

   **Proposal:** fill §8 in two steps. Step (a): now, commit the structure and the conditional, qualitative predictions below. Step (b): after the pending ADRs and a calibration pilot on a train pool disjoint from all EXP-0001 splits, freeze the numbers before the first T_mine episode. Step (b) is legitimate pre-registration only if the plan itself, "pilot then freeze", is recorded before the pilot runs.
2. **H2 (proposed, S/M): not supported under the locked rule, P(support) ≈ 0.15 (range 0.05–0.30).**
   - The modal outcome is arm C = G0 (empty) in ≥ 2 of 3 seeds. P ≈ 0.8–0.99 from PA's operating characteristics. PA's independent-draw model, if anything, understates P(empty) (R8). [corrected after independent verification 2026-10-13]
   - T_eval C − B: point estimate ≈ 0 pp, 80 % interval −10 to +8 pp, direction not predicted.
   - Regression: reg(C) ≈ 0–3 pp and reg(B) ≈ 5–20 pp, so the direction C < B is likely (P ≈ 0.85). Significance is unlikely only if mining seeds are largely redundant (ρ ≈ 1). With independent seeds and reg(C) ≈ 0, the regression half reaches ≈ 80 % power already at reg(B) ≈ 5–7.5 pp (C6). [corrected after independent verification 2026-10-13]
   - If H2 is "supported" while C is empty, that is a gate-as-veto finding, not evidence of causal selection.
3. **H1 (proposed): not supported, P(support) ≈ 0.10 (range 0.03–0.25).** C − A ≈ 0 pp, and exactly 0 in any seed where C is empty. 80 % interval −3 to +6 pp.
4. **Gate stage counts (proposed, per seed, out of K = 10).** These are marginal counts; all stages run for all candidates, with no short-circuit in `inheritance/gate.py`.

   | Stage | Expected passes |
   |---|---|
   | Replay | ≈ 0.5 (0–2) |
   | Ablation | ≈ 1 (0–3) |
   | Generalization, in-scope | ≈ 7 (6–9) |
   | Generalization, out-of-scope | ≈ 7 (6–9) |
   | Regression | ≈ 3 (0–8; set mostly by the prompt-perturbation rate f) |
   | Overhead | ≈ 10 (U) |
   | Interaction | not triggered |
   | **Admitted** | **≈ 0.1–0.3 per seed (0–1); 0–2 across the 3 seeds** |

5. **Compute (proposed): 1 × H100-80GB (A100-80GB acceptable), one offline vLLM engine, sequential batch size 1. Wall-clock cap 120 h and 120 GPU-h for the primary run, plus 12 h for the pilot.**
   - With a 50-step budget, the planning model gives 40–60 h wall-clock on an H100 for 5.8–6.7k episodes (233–269k calls).
   - The locked worst case of 8.5k episodes takes up to 76 h on an H100 and 107 h on an A100.
   - With the 8-step cap as coded, it takes 15–24 h.
   - Every throughput constant is an assumption (U) to be replaced by pilot measurements.
6. **The YAML changes are listed exactly in §7.** Only Y1 is "filling". Items Y2 and Y3 change locked text and need an ADR.

---

## 1. Preconditions: what must be settled before the numbers mean anything

| # | Precondition | Evidence | Effect on the fields | ADR? |
|---|---|---|---|---|
| P1 | **Lifetime/mining protocol**: number of rollouts per T_mine task, lifetime temperature, pairing rule | VR §2.1 (V(VR)); `traits/miner/inputs.py:3-4` requires same-task success + failure; `runtime/loop.py:174` hard-codes temperature 0 | As specified, K ≈ 0 and every field is void. Any pairs that do appear are accidental: different instances with identical task text, which the code accepts because provenance has no instance id. Such pairs are few and confounded by scene layout. The ADR must also fix the pairing unit (same instance or same family) and whether the pair check carries instance identity (V1). [corrected after independent verification 2026-10-13] With k stochastic rollouts the mining cost rises; this note budgets k = 4 as a sensitivity. | **requires ADR** |
| P2 | **G0 step budget**: `max_plan_steps`, `Budgets.max_steps`, `max_total_retries`, `max_tokens_per_request`, `max_total_tokens` | `runtime/seed_g0.py:17` (V); manifest has no budget field (V) | With ≤ 8 actions, G0 success is far below every anchor (§2.2). Calls per episode drop from the manifest's "~30" to ≤ ~9. | **requires ADR** (new planner gene version, ADR-0001 D6) |
| P3 | **T_reg definition and selection rule** (it is also the H2 outcome set; VR finding 3) | design §2; BL finding 4: P(≥ 40/60 solved \| p = 0.45) = 0.0006 | Without it, the regression stage and half of H2 cannot be run | **requires ADR** |
| P4 | **H1/H2 estimator across seeds** (seed-averaged per-task clusters, per-seed CIs, or pooled rows) | PA R1 | Changes the MDE by up to about 2×, and with it the predicted P(support) | **requires ADR** |
| P5 | **How arm B is represented** (germline origin; whether `procedure[]` is stripped) | VR finding 5 | Decides whether B and C differ only in which skills are selected. As coded, every arm uses one rendering path (`project_skill`, `procedure[]` kept), and `project_skillrl` has no runtime caller, so B and C differ only in selection unless #13 wires the stripped projection into B. genome/0.1 can hold arm-B genes only under a false origin (`seed`, or `assimilation` citing any CIG record). It has no honest label for ungated genes (V5, P1). [corrected after independent verification 2026-10-13] | **requires ADR** |
| P6 | **ALFWorld adapter**: tool catalog, success signal, invalid-action semantics (`ok=False`?), prompt rendering | `runtime/loop.py:5`: "No benchmark logic, no ALFWorld code" (V); no adapter on `c6ce4c1` (V) | G0 base rate. BL shows rendering alone moves Qwen2.5-7B from 83.6 to 1.5 on identical content (ExpRAG, V(BL)). | builder; prompt template must be pinned before step (b) |
| P7 | **Calibration pilot** on train games disjoint from every EXP-0001 split. Measures G0 success p0, the placebo-prompt flip rate f, the shape of the noise, and backend determinism. | VR §2.2 recommendation; PA R8 | Turns the M/S values below into calibrated values | the pilot itself: no; acting on it: ADR |

**Proposed two-step procedure (addition; owner and critic to accept).**
- **Step (a), now.** Commit §8 with the conditional statements in §3–§6, the pilot protocol, and an explicit rule: "numeric expectations are frozen after the pilot and before any T_mine episode, and the pilot never touches T_mine, T_gate, T_reg or T_eval."
- **Step (b).** After P1–P6 are resolved by ADR and the pilot has run, re-run `scripts/exp0001_power_sim.py --p0 <pilot p0> --f <pilot f> --model <shape>`. Replace the ranges below with the calibrated ones, commit, and then set `expected_results_filled: true`.
- **Why this is not post-hoc.** The pilot sees no EXP-0001 data, and the freeze happens before any EXP-0001 episode.

---

## 2. Empirical anchors used

### 2.1 Qwen2.5-7B-Instruct on ALFWorld without training

| Anchor | Value | Protocol | Source | Status |
|---|---|---|---|---|
| Zero-shot, static action-template list, greedy | 29.9 % (40/134) | valid_unseen 134, temp 0 | ExpRAG arXiv 2603.18272, Table 17 / App. C.2 | V(BL) |
| Base model, no memory, strict 7B protocol | 45.5 % (61/134) | valid_unseen 134, temp 0, step limit 50, 16,384-token context | SAGA arXiv 2610.06964, Tables 2/6/12 | V(BL) |
| SAGA trajectory-only memory, rounds R0–R3 | 45.5 / 46.3 / 57.5 / 46.3 | same 134 games; memory content differs | SAGA appendix | V(BL) |
| SAGA complete-hierarchy memory, rounds R0–R3 | 45.5 / 48.5 / 54.5 / 59.4 | same | SAGA appendix | V(BL) |
| Retrieved expert trajectories (K = 4), and the same content in "Textual" rendering | 83.6 % vs 1.5 % | valid_unseen, greedy | ExpRAG App. C.2, Table 17 | V(BL) |
| ReAct / Reflexion / ExpeL / Mem0 / SimpleMem / MemP | 31.2 / 42.7 / 46.3 / 33.6 / 29.7 / 41.4 | **split not stated** (verl-agent default is valid_seen, temp 0.4) | GiGPO 2505.10978 Table 1; SkillRL 2602.08234 Table 1 | numbers V(BL); split U |
| "simply providing skills to an unchanged model yields limited benefit" | qualitative | — | SkillRL, citing its ref [12] | V(BL) |
| Serving-schedule effect | SAGA reports its no-memory baseline (concurrency 32) vs evolution runs (concurrency 8) as a potential implementation confound | — | SAGA | V(BL) |

### 2.2 Step-length anchors (new, for P2)

| Anchor | Value | Source | Status |
|---|---|---|---|
| ALFWorld HandCodedTWAgent expert trajectories (78, training split) | mean **14.2 actions (σ = 5.8)** | arXiv 2511.21730, §3.2.1 "Corpus Statistics" | V (PDF text read this session) |
| 1,200 ALFWorld expert trajectories | "range from **9 to 49 steps**, with an average of **20.1**" | arXiv 2605.09330, dataset appendix | V (PDF text read this session) |

**Reading (U).** An 8-step cap (at most 8 environment actions) is below the shortest trajectory in the second corpus. Under a normal approximation it is above only about 14 % of the hand-coded expert's trajectories in the first. A 7B ReAct agent that sees only observations, not its own previous actions (`runtime/policies/planner_react.py:65-67`, V), will be less efficient than the expert. **Under the cap as coded, I expect G0 success of ≤ 15 % (U).** That is too low for T_reg, and it would push most families to a floor. This is the main reason the numbers below assume P2 is resolved with a budget near the 50-step protocol. That limit is stated for the SAGA 45.5 % anchor and for verl-agent / GiGPO; the ExpRAG 29.9 % row states no step limit (V4). That choice is the team's, and needs an ADR. [corrected after independent verification 2026-10-13]

### 2.3 Operating characteristics of the locked gate and statistics

These come from PA, model-based (M):
- **Stage 2.** Needs net ≥ 4 out of 20, i.e. ≥ +20 pp observed. That means ≥ 4 rescues with 0 breaks, ≥ 7 with 1, ≥ 9 with 2, and ≥ 11 / 13 / 14 with 3 / 4 / 5 breaks; 6 or more breaks never pass. Five compositions depend on task ordering: (6,1) ≈ 2 %, (8,2) ≈ 1 %, (10,3) ≈ 32 %, (12,4) ≈ 98 % and (13,5) ≈ 0.1 %. The rule is more lenient than an exact one-sided sign test, which would need 5–6 rescues with 0 breaks. The strictness comes from n = 20, not from the bootstrap (C1). [corrected after independent verification 2026-10-13]
- **Replay.** Needs ≥ 2 net rescues out of 5.
- **Regression.** "≤ 1 failure of 40" passes a neutral trait at 40 % (f = 0.05) to 8 % (f = 0.10) under symmetric noise, and at 74 % / 40 % under mean-neutral noise.
- **Arm C composition.** E[admitted] per seed is 0.02–1.3, and P(arm C empty) is 25–98 % under PA's independent per-candidate draws. The real design shares the ablation, replay/generalization, interaction and T_reg pools across all K = 10 candidates. With shared instances the means are unchanged, but the lower end of P(arm C empty) is about 33 % (optimistic cell: 24.4 → 33.1 %), so PA's P(empty) is if anything understated (R8). [corrected after independent verification 2026-10-13]
- **H2 power.** The realized MDE on T_eval is about 7–10 pp with informative seeds (7.3 / 8.6 / 9.6 pp for f = 0.1 / 0.2 / 0.3). With redundant seeds it is 12.7–16.9 pp under symmetric noise and 9.9–13.8 pp under the redraw model. It exceeds the pre-registered +5 pp in every variant tried (C5). The regression half is under-powered when both arms regress ≤ 10 pp only if seeds are redundant (ρ = 1: 12–35 %). With independent seeds it has 30–98 % power, and it reaches ≈ 80 % once reg(C) ≈ 0 and reg(B) ≥ 5–7.5 pp (C6). reg(C) ≈ 0 is partly guaranteed by selection, because each arm-C gene already passed stage 4 on the same 40 T_reg tasks (C6, V3). [corrected after independent verification 2026-10-13]
- **H1 identifiability.** H1 is not identifiable as trait value without a placebo-prompt control.

---

## 3. Field 1: Expected result, H2 (direction and rough magnitude)

**Proposed value (S, informed by M).**
> H2 is expected **not** to be supported under the locked criterion. Subjective P(support) ≈ 0.15, plausible range 0.05–0.30.
> - The expected arm-C genome is G0 with 0 admitted genes in ≥ 2 of 3 mining seeds.
> - Expected T_eval contrast: C − B ≈ 0 pp, 80 % interval −10 to +8 pp. The direction is **not** predicted: it is set by whether the 10 unvalidated skills in arm B help or hurt on balance.
> - Expected T_reg regression: reg(C) ≈ 0–3 pp, and exactly 0 in any seed where C is empty, assuming backend determinism. reg(B) ≈ 5–20 pp. The direction reg(C) < reg(B) is likely (P ≈ 0.85). If seeds are redundant (ρ ≈ 1), the CI on the difference excludes 0 only if reg(B) ≳ 15–20 pp (P ≈ 0.3–0.5). With independent seeds and reg(C) ≈ 0, it does so with ≈ 80 % power already at reg(B) ≈ 5–7.5 pp (C6), so this P is a ρ ≈ 1 value, to be recalibrated at step (b). [corrected after independent verification 2026-10-13]
> - If H2 meets the criterion while arm C is empty in ≥ 2/3 seeds, it is reported as **"gate-as-veto: G0 > G0 + unvalidated skills"**, not as support for causal selection of traits.

**Reasoning.**
1. *Arm C composition dominates.* Across every candidate mixture and noise level in PA T2, E[admitted] ≤ 1.3 per seed. For the baseline mixture (20/50/30, helpful = +20 pp) it is 0.07–0.30, which gives P(empty per seed) = 0.74–0.93. Treating seeds as independent, that means P(empty in ≥ 2/3 seeds) = 0.83–0.99 (`exp0001_prereg_stage_counts.out`, M). PA draws candidates independently, but the real gate pools are shared across candidates. With shared instances, P(empty per seed) rises slightly (e.g. 74.8 → 75.6 % at +20 pp), so these values are if anything understated (R8). The result is model-dependent, though. Under the validity verifier's consistent-noise model, P(|C| = 0) per seed spans about 0.2–0.75, depending on the unmeasured flip rate f. An empty C is therefore the modal outcome under PA's model, and a plausible but not certain one under others (V2, M). [corrected after independent verification 2026-10-13] So C − B ≈ A − B: H2 mostly asks whether 10 unvalidated skills hurt G0.
2. *Size of A − B.* The anchors span a wide range:
   - Validated memory from the agent's own experience helps Qwen2.5-7B by +0.8 to +13.9 pp on these 134 games, but not monotonically (SAGA, V(BL)).
   - Generic memory baselines sit within about ±3 pp of ReAct (Mem0 +2.4, SimpleMem −1.5), on an unstated split (U).
   - The prompt rendering of the same content can cost tens of points (ExpRAG: 29.9 → 1.5 at the extreme, V(BL)).
   - SkillRL states that skills given to an unchanged model bring limited benefit (V(BL)).
   - "Agent Skills Can Be Harmful" (arXiv 2608.11888, repo novelty ledger, V) documents regressions from relevant-looking skills.
   - Taken together: a central value near 0 with a heavier lower tail for B, i.e. C − B centred near 0 with a longer positive tail.
3. *Power.* Even for a true realized C − B of +10 pp, the T_eval half has 44–90 % power depending on the seed-sharing fraction ρ and the arm-level noise f_arm (PA T3). The regression half reaches ≥ 80 % only if reg(B) ≥ 20–30 pp while reg(C) ≤ 5 pp, but this holds only for redundant seeds (ρ = 1). With independent seeds it reaches 80 % once reg(C) ≈ 0 and reg(B) ≥ 5–7.5 pp. PA's prose mislabelled ρ, while its T4 cells are correct (C6). [corrected after independent verification 2026-10-13] The joint criterion is the product of the two halves.
4. *Why reg(C) ≈ 0.* An empty C reruns G0 on tasks G0 solved, at temperature 0, so it should reproduce those successes exactly. Any nonzero reg(C) in an empty-C seed would therefore be a determinism failure. This assumes P3 defines "G0-solved" at evaluation settings, as VR recommends. For a non-empty C, the ≤ 3 pp bound is not guaranteed. Stage 4 bounds each trait *alone* on T_reg, not the multi-trait composite that G1 runs. The bound is tight only for a single admitted gene whose G1 prompt reproduces the gate overlay. Because the same 40 tasks select arm C's genes, the regression leg of H2 is not a held-out test of the gate (V3, P1). [corrected after independent verification 2026-10-13]

**Uncertainty range.** P(support) 0.05–0.30. T_eval C − B, 80 % interval: −10 to +8 pp. reg(B): 3–30 pp. All of this narrows after the pilot (step b) once p0 and f are measured.

**What would count as surprising (pre-declared).**
- Arm C is non-empty in ≥ 2/3 seeds with ≥ 2 genes in any seed. That would contradict PA's operating characteristics and suggest f ≈ 0 with large per-trait effects.
- C − B ≥ +10 pp with the CI excluding 0 **and** C non-empty in ≥ 2/3 seeds. This is the strong, intended form of H2.
- C − B ≤ −5 pp with the CI excluding 0. B beats C: unvalidated skills help and the gate removed value, i.e. the gate is too strict. This would be reported as a finding.
- |B − A| ≥ 15 pp in either direction, which is large relative to the memory anchors.
- reg(C) > 0 in a seed where C is empty, which signals backend nondeterminism. That is a protocol flag, not a result.

---

## 4. Field 2: Expected result, H1

**Proposed value (S/M).**
> H1 is expected **not** to be supported. Subjective P(support) ≈ 0.10, range 0.03–0.25.
> - C − A ≈ 0 pp, 80 % interval −3 to +6 pp.
> - C − A is **exactly 0 by construction** in every seed where arm C is empty (G1(C) = G0 = G1(A), deterministic). An observed nonzero per-task difference in such a seed is a determinism failure, not an effect.
> - If C holds one gene, the expected realized effect on T_eval is small (≤ +5 pp). The gate selects a gene by its contribution on a 20-task pool, and that contribution regresses toward the mean on 134 unseen tasks.

**Reasoning.**
- Arm C composition is as in §3.
- PA T6 under mean-neutral noise: a realized +5 pp gives 46 % power at ρ = 1, and a realized +10 pp gives 70–92 %.
- Under symmetric noise with p0 < 0.5, the zero-effect "support" rate is 10–43 %. The reason is that arm A carries no added prompt, so H1 also picks up the prompt-perturbation bias. The proposed P(support) of 0.10 includes this confound.
- PA R4 (a placebo arm or probe) is the fix. It **requires ADR** and is not assumed here.

**What would count as surprising.**
- H1 supported with CI lower bound > +3 pp: a strongly helpful gene was admitted and generalized to unseen rooms.
- C − A < 0 with the CI excluding 0: an admitted gene hurts on T_eval, so the gate admitted a harmful trait (PA: false-pass rate ≈ 0 under every model).
- Any nonzero C − A in an empty-C seed, which is a determinism flag.

---

## 5. Field 3: Expected number of candidates passing each gate stage

**Proposed value (M → S).** Marginal pass counts per mining seed, out of K = 10. All stages run for all candidates (`inheritance/gate.py` docstring: "A scientifically valid stage FAIL still produces immutable child evidence and later primary stages continue", V), so each marginal count is observable and can be checked after the run.

| Stage | Proposed expected passes / seed (central; 80 % range) | Model values (PA T1, p0 = 0.5, symmetric noise; mixtures 20/50/30 · 50/30/20 · 10/60/30; f = 0 / 0.05 / 0.10) | Main driver |
|---|---|---|---|
| Mined candidates K_realized | 10 (conditional on P1) | — | as coded: 0 (VR) |
| S1 replay (≥ 3/5 and +2/5, expression in every success) | **0.5** (0–2) | 0.50 / 0.52 / 0.59 · 1.26 / 1.24 / 1.25 · 0.25 / 0.28 / 0.38 | 5 instances resolve only ≥ 40 pp discordance |
| S2 ablation (LB > 0 ⇒ net ≥ 4 of 20: ≥ 4 rescues / 0 breaks, ≥ 7 / 1, ≥ 9 / 2, …; never with ≥ 6 breaks) | **1** (0–3) | 1.18 / 0.82 / 0.61 · 2.94 / 2.04 / 1.45 · 0.59 / 0.41 / 0.33 | integer rule; specialists are diluted. Under the proportional allocation, a Look-, Heat- or Cool-only specialist (2–3 in-scope tasks) cannot pass. A Pick-, Clean- or Pick2-only specialist passes only by fixing all 4 in-family tasks with 0 losses (C1, V6). [corrected after independent verification 2026-10-13] |
| S3 in-scope (Δ ≥ 0, n = 10) | **7** (6–9) | 8.05 / 7.18 / 6.74 · 8.70 / 8.05 / 7.64 · 8.05 / 7.03 / 6.54 | non-inferiority; a no-op passes |
| S3 out-of-scope (Δ ≥ −0.05 ⇒ net ≥ 0, n = 10) | **7** (6–9) | same as in-scope for generalists; higher for true specialists | tolerance void at n = 10 |
| S4A regression (≤ 1 failure of 40) | **3** (0–8) | 7.00 / 2.79 / 0.56 · 8.00 / 3.19 / 0.64 · 7.00 / 2.79 / 0.56 | the per-task flip rate f, not the trait's value |
| S4B overhead (≤ 512 tokens) | **≈ 10** (U) | not modelled | miner output length (U) |
| S4C interaction (A1; needs ≥ 2 eligible) | **not triggered** in ≥ 2/3 seeds | E[eliminated] = 0.00 in every PA T2 cell | |
| **Admitted (all stages)** | **0.1–0.3 per seed (0–1); 0–2 total across 3 seeds** | 0.30 / 0.07 / 0.01 · 0.74 / 0.18 / 0.03 · 0.15 / 0.04 / 0.01 | conjunction of S1, S2 and S4A |
| P(arm C empty) per seed | **0.75–0.95** | 0.74 (f = 0) to 0.93 (f = 0.05), baseline mixture, helpful = +20 pp (PA T2). These assume independent per-candidate draws. Shared gate pools make P(empty) if anything higher (e.g. 74.8 → 75.6 % at +20 pp, f = 0; R8). [corrected after independent verification 2026-10-13] | |

**Notes.**
- *Noise model.* Under PA's mean-neutral "redraw" model, the regression row rises (a neutral trait at f = 0.05 passes 74 %), so S4A ≈ 5–7 and admitted ≈ 0.25–1.2 per seed (PA §4.9). The pilot's placebo probe decides which model applies (P7).
- *Base rate.* With p0 = 0.3 instead of 0.5, the counts move by ≤ 0.1 except for replay, which drops slightly (PA T1).
- *Expected effect mixture.* The mixtures (helpful/neutral/harmful) and the helpful effect of +20 pp are PA's modelling assumptions (A10), not data. No published per-skill effect distribution exists for self-mined ALFWorld skills on a 7B model (U). This field is therefore the least anchored of the four.

**What would count as surprising.**
- ≥ 3 admitted in any seed.
- Replay ≥ 4/10 in a seed.
- Regression ≥ 9/10 in a seed. This would mean temperature-0 trajectories barely change under added skill text, which is informative in itself.
- Any admitted candidate whose stage-2 point estimate is < +0.20. That is impossible by the integer rule (the smallest passing net is 4/20 = +0.20, e.g. 4 rescues / 0 breaks; C1), so it would indicate an implementation error. [corrected after independent verification 2026-10-13]
- In-scope or out-of-scope pass counts ≤ 4/10. That would mean most skills are actively harmful.

---

## 6. Field 4: Compute allocation and wall-clock cap

### 6.1 Episode counts (V from design §7 and ADR-0002/A1; arithmetic checked in the script)

| Item | Episodes | Note |
|---|---|---|
| Mining (as locked) | 180 | 3 seeds × 60. **Sensitivity:** k = 4 stochastic rollouts per task plus 1 deterministic G0 pass gives 780 (P1, ADR) |
| CIG | 3,900 | 3 × 10 × 130 (replay 10, ablation 40, generalization 40, regression 40 with-trait only; the baseline is not rerun) |
| Final evaluation | 1,746 | 3 arms × (134 + 40 + 20) × 3 seeds. Arm A's 3 copies are identical deterministic runs. Proposal: keep them as a free determinism check, not as evidence |
| Interaction, expected | ≈ 0 | needs ≥ 2 eligible (PA T2) |
| Interaction, worst case | **2,700** | A1's own arithmetic is 900 **per seed** (45 × 10 × 2). Design §7 and YAML say "≤ ~900" in total: an erratum, VR I5 |
| Pilot (proposed) | 480 | G0 + 3 length-matched placebo texts × 120 disjoint train games |
| **Totals** | 5,826 expected · 6,726 manifest upper · **8,526 locked worst** · 9,606 planning ceiling | |

### 6.2 Per-call model (assumptions, all U; the pilot measures them)

- **Prompt.** System prompt with format instructions and tool catalog ≈ 600 tokens. Task plus initial room description ≈ 200. Each step appends about 40 tokens; only observations enter the history (`planner_react.py:65`, V). Step tail ≈ 30.
- **Skills.** About 200 tokens per rendered skill (cap τ_k = 512). Arm B carries 10 skills, CIG with-trait runs carry 1, interaction runs carry 2.
- **Completion.** About 60 tokens per call (cap 512, `Budgets.max_tokens_per_request`, V).
- **Calls per episode, three scenarios.**
  - S8, as coded: ≈ 8.5 (≤ 8 steps plus parse retries).
  - S30: the manifest's "~30".
  - S50, 50-step budget with p0 ≈ 0.4: ≈ 40 (0.4 × 18 + 0.6 × 50, plus retries).
- **Hardware, batch size 1.** H100-80GB: about 7 ms per output token, about 10k prompt tokens/s, 30 ms overhead per call. A100-80GB: about 11 ms, 7k tok/s, 40 ms.
- **Host side.** About 2 s per ALFWorld reset, 30 ms per env step, and about 3 s mean retry back-off sleep per episode. `retry_backoff` sleeps 1/2/4 s in real time (`runtime/policies/retry_backoff.py:14,26`, V).
- **Mining teacher calls are not included.** There is one call per pair with up to 2,048 output tokens (`traits/miner/miner.py:84`, V). At tens of pairs per seed this is ≤ about 0.5 h. It is negligible unless the teacher is a different, larger model (VR finding 12).
- **Serving mode as coded.** `VLLMAdapter.generate` sends one `LLM.chat` request per call (`runtime/model_adapters/vllm_backend.py:144-148`, V). With sequential episodes this means batch size 1, latency-bound.

### 6.3 Results (M; `scripts/exp0001_compute_estimate.out`)

| Episode scenario | Step scenario | LLM calls | Prompt tokens (M) | Wall h, H100, seq (no prefix cache / prefix cache) | Wall h, A100, seq (no prefix cache / prefix cache) | Wall h, H100, 32 concurrent |
|---|---|---|---|---|---|---|
| 5,826 expected | S8 as coded | 50k | 63 | 16 / 15 | 21 / 19 | 0.9 |
| 5,826 expected | S30 | 175k | 298 | 40 / 32 | 55 / 44 | 2.8 |
| 5,826 expected | S50 | 233k | 444 | **52 / 40** | 73 / 56 | 3.8 |
| 6,726 manifest upper | S50 | 269k | 517 | 60 / 46 | 84 / 65 | 4.4 |
| 8,526 locked worst | S50 | 341k | 662 | **76 / 58** | **107 / 82** | 5.5 |
| 9,606 ceiling | S50 | 384k | 731 | 85 / 66 | 120 / 93 | 6.2 |

**Readings.**
- **Call counts.** The manifest's "~180–200k calls" matches S30. S50 gives 233–341k calls, and the 8-step code gives only 50–72k.
- **Sequential mode fits "a few days" on one GPU.** The concurrent column assumes about 12× aggregate speed-up at 32 parallel episodes (U). It would fit in an afternoon, but see §6.4.
- **Context fit.** At step 50, arm B with 10 skills at 200 tokens needs about 4.8k prompt tokens, which fits `max_model_len` 8192 minus 512. At the τ_k cap (10 × 512) it needs about 7.9k tokens, which does **not** fit 7,680. Proposal: the pilot logs arm-B-like prompt lengths, and the builder fails closed on overflow rather than truncating. Possibly also pre-register `max_model_len` (it is part of the serving record).

### 6.4 Proposed compute allocation and cap

> **Allocation.**
> - **Hardware and engine.** One node, 1 × H100-80GB (A100-80GB acceptable; record which). One offline vLLM engine: pinned version, bf16, TP = 1, `max_model_len` 8192, prefix caching setting pinned and recorded. Episodes are **sequential at batch size 1** for all gate stages and for final evaluation.
> - **Expected cost.** 40–60 GPU-h (S50) or 15–25 GPU-h (S8), with 2–4 CPU cores for ALFWorld. Pilot: ≤ 12 h.
>
> **Caps.**
> - **Primary run.** Hard caps of 120 h wall-clock and 120 GPU-h, measured from the first T_mine episode to the last T_eval/T_reg episode.
> - **Soft caps per phase.** Mining ≤ 8 h, CIG ≤ 72 h, final evaluation ≤ 36 h.
> - **Re-plan trigger.** Before the run, the pilot's measured seconds per episode × planned episodes must be ≤ 0.8 × the cap (96 h). If not, the run does not start, and hardware, serving mode or the cap changes only by ADR.
> - **On breach.** Halt. Record a protocol deviation on #6/#13. Make no H1/H2 claim from partial data. Extend only by an owner-approved ADR note. Episodes already completed are kept; they are deterministic, so extending the run does not change them.

**Reasoning.**
- 120 h covers the locked worst case under S50 on an H100 with about 1.6× margin, and on an A100 just barely (107 h).
- Sequential batch size 1 is proposed because the determinism policy (design §4, §6.4) treats temperature-0 executions as reproducible. Batched or concurrent serving can change greedy outputs through batch composition; SAGA flags serving concurrency as a confound (V(BL)), and ExpRAG seeds vary even at greedy decoding (V(BL)).
- vLLM offers batch-invariant and other determinism options in recent versions. I know this from memory and did not verify it here. If the team wants the roughly 10× speed-up of concurrent serving, it should be adopted only after the pilot shows identical per-task outcomes between sequential and concurrent runs on the same episodes. Changing the serving mode **requires ADR**.
- Prefix caching should likewise be validated (cold-cache vs warm-cache rerun of 20 episodes), because it can change numerics (U).

**Small no-ADR wall-clock saving (builder).** `run_agent` accepts an injectable `sleep` (`runtime/loop.py:142`, V). For a local deterministic simulator, a runner-level no-op sleep removes about 2–8 h of idle back-off over 6–9k episodes without changing any outcome. Retrying a deterministic ALFWorld tool failure reproduces the same failure (U, depends on the adapter's `ok` semantics). The setting must be recorded in the run log. It does not change the genome: `base_delay_s` is not in the gene payload (`retry_backoff.py:14`, V).

---

## 7. Exact YAML changes after owner approval (do not apply before)

### Y1: filling the pre-registered fields (no ADR; must precede run start and be critic-reviewed)

```yaml
experiment:
  preregistration:
    locked: true
    expected_results_filled: true            # flip ONLY at step (b), after pilot + freeze
    expected_results_ref: "experiments/manifests/EXP-0001-design.md#8-pre-registration-fields-to-fill-before-run-start"
    expected_results_commit: <sha-of-commit-filling-section-8>
    filled_at: <YYYY-MM-DD>
    reviewed_by: [owner, research-critic]
    calibration_pilot:                        # only if the two-step procedure (section 1) is accepted
      report: <path>                          # pilot report, content-hashed
      sha256: <hash>
      pool: train-games-disjoint-from-all-EXP-0001-splits
      measured: {g0_success: <p0>, placebo_flip_rate: <f>, noise_shape: <symmetric|redraw|other>, determinism_repeat_discordance: <n/N>}
      power_sim_rerun: {script: exp0001_power_sim.py, args: "--p0 <p0> --f <f> --model <shape>", output_sha256: <hash>}
    expected_results:                         # values = whatever section 8 finally states (proposals in sections 3-5 of this draft)
      g0_success_T_eval: {point: <..>, range: [<..>, <..>]}
      h2:
        p_supported: <..>
        teval_c_minus_b_pp: {point: <..>, interval80: [<..>, <..>]}
        reg_c_pp: {point: <..>, range: [<..>, <..>]}
        reg_b_pp: {point: <..>, range: [<..>, <..>]}
        arm_c_empty_in_ge_2_of_3_seeds: {p: <..>}
        degenerate_case_label: "gate-as-veto (C = G0) if supported with empty C"
      h1:
        p_supported: <..>
        c_minus_a_pp: {point: <..>, interval80: [<..>, <..>]}
        note: "exactly 0 by construction in seeds with empty C"
      gate_stage_passes_per_seed:             # marginal counts of K = 10
        replay: {point: <..>, range: [<..>, <..>]}
        ablation: {point: <..>, range: [<..>, <..>]}
        generalization_in_scope: {point: <..>, range: [<..>, <..>]}
        generalization_out_of_scope: {point: <..>, range: [<..>, <..>]}
        regression: {point: <..>, range: [<..>, <..>]}
        overhead: {point: <..>, range: [<..>, <..>]}
        interaction_triggered: {p: <..>}
        admitted: {point: <..>, range: [<..>, <..>]}
      surprising_if: [<list copied from section 8>]

compute:                                      # new top-level block (design section 8 "compute allocation")
  hardware: {gpu: <H100-80GB|A100-80GB>, count: 1, node: single}
  serving:
    engine: vllm-offline
    vllm_version: pin-at-run-time
    dtype: bfloat16
    tensor_parallel: 1
    max_model_len: 8192
    prefix_caching: <on|off>                  # pinned; validated in pilot
    mode: sequential-batch-1                  # changing to concurrent requires ADR + pilot equivalence check
  expected_gpu_hours: <..>                    # e.g. 40-60 (S50) from pilot-calibrated projection
  caps:
    wall_clock_hours: 120
    gpu_hours: 120
    pilot_wall_clock_hours: 12
    phase_soft_caps_hours: {mining: 8, cig: 72, final_eval: 36}
  replan_trigger: "pilot-projected time > 0.8 x cap -> do not start; change only via ADR"
  on_cap_breach: halt-record-deviation-no-partial-H1-H2-claim-extension-only-by-owner-ADR
```

### Y2: corrections to locked text (**requires ADR**, at minimum an ADR-0002 erratum note)

```yaml
budget:
  episodes_est: <recomputed>                  # was 6000
  breakdown: "mining <180 | per lifetime ADR> + CIG 3900 + final eval 1746 (arm A x3 = determinism check) + interaction expected ~0 / worst 2700 (900 per seed x 3)"
  note: "<calls per episode per step-budget ADR>; ~50-340k local-7B calls depending on step budget"
experiment:
  status: <non-stale status string>           # VR I18
```

### Y3: new locked elements that the predictions depend on (**each requires ADR**, P1–P5)

```yaml
model:
  lifetime_temperature: <..>                  # P1
  lifetime_rollouts_per_task: <k>             # P1
  mining_pairing_rule: <..>                   # P1
runtime_budgets:                              # P2
  planner_max_plan_steps: <..>                # new planner gene version if != 8
  max_steps: <..>
  max_total_retries: <..>
  max_tokens_per_request: 512
  max_total_tokens: null
environment:
  splits:
    T_reg: {selection_rule: <..>, solved_definition: "G0 at eval settings (temp 0)", hash: <..>}   # P3
evaluation:
  statistics:
    seed_pooling: <seed-averaged-task-cluster | per-seed | pooled-rows>                       # P4
    h2_non_degeneracy: <e.g. "C non-empty in >= 2/3 seeds", if adopted>                     # VR 2.2
arms:
  B-unrestricted: {representation: <..>, projection: <full|skillrl-stripped>}                # P5
```

### Proposed §8 replacement text (design.md; for the owner and critic to edit)

```markdown
## 8. Pre-registration fields (filled <date>, commit <sha>; frozen before any T_mine episode)

- [x] Expected result, H2: NOT supported under the locked rule (P ≈ <0.15>). Arm C expected
  empty in ≥ 2/3 seeds (P ≈ <0.8-0.99>; if anything understated under shared gate pools). T_eval
  C−B ≈ <0> pp (80% interval <−10, +8>), direction not predicted; reg(C) ≈ <0-3> pp vs
  reg(B) ≈ <5-20> pp (direction C<B likely; significance unlikely if seeds are redundant, ρ ≈ 1).
  If supported with C empty → reported as "gate-as-veto", not causal-selection evidence.
  [corrected after independent verification 2026-10-13]
  Surprising: C non-empty ≥ 2/3 seeds with ≥2 genes; C−B ≥ +10 pp with CI>0 and C non-empty;
  C−B ≤ −5 pp with CI<0; |B−A| ≥ 15 pp; reg(C) > 0 in an empty-C seed (determinism flag).
- [x] Expected result, H1: NOT supported (P ≈ <0.10>); C−A ≈ <0> pp (80% <−3, +6>); exactly 0
  in empty-C seeds. Surprising: LB > +3 pp; CI < 0; any nonzero C−A in an empty-C seed.
- [x] Expected candidates passing each stage (per seed, of K=10; marginal): replay <0.5 (0-2)>,
  ablation <1 (0-3)>, in-scope <7 (6-9)>, out-of-scope <7 (6-9)>, regression <3 (0-8)>,
  overhead <≈10>, interaction not triggered, admitted <0.1-0.3 (0-1)>; 0-2 across 3 seeds.
  Surprising: ≥3 admitted in a seed; replay ≥4; regression ≥9; in/out-scope ≤4.
- [x] Compute: 1× <H100-80GB>, one offline vLLM engine, sequential batch-1, bf16, max_model_len
  8192; expected <40-60> GPU-h; caps 120 h wall / 120 GPU-h (+12 h pilot); phase soft caps
  mining 8 h / CIG 72 h / final 36 h; re-plan if pilot projection > 96 h; breach → halt, deviation
  record, no partial H1/H2 claim.
```

---

## 8. Additional pre-registration items recommended (not in §8 today)

| Item | Proposal | Why | ADR? |
|---|---|---|---|
| Expected G0 (arm A) success on T_eval | ≈ 30 % (15–46 %) under a ~50-step budget; ≤ 15 % under the 8-step cap (U) | Anchors 29.9 / 45.5 (V(BL)). The G0 format sees no action history, which suggests the lower end (U) | no (an added prediction) |
| Expected determinism | 0 discordant per-task outcomes across arm A's 3 identical copies (194 × 3), and across the pilot's repeat runs | Free check of the determinism policy | no |
| Degenerate-case declarations | Pre-declare "C = A" (gate admits nothing) next to the existing "B ≈ C" (§6.5) | PA R5, VR 2.2 | declaring: no; making it a decision condition (non-degeneracy): **requires ADR** |
| Per-family floor reporting | Report per-family T_eval success per arm; families with arm-A success ≤ 5 % are flagged as floor-limited | The 8-step cap and the anchors show floors in Heat/Cool/Pick2 (VR, BL) | no |
| `budget_exhausted` rate per arm | Report as a diagnostic | Arm B's longer prompts interact with token and step budgets (VR) | no |

---

## 9. Items in this draft that would change locked or pre-registered elements (all **requires ADR**, none decided)

1. Lifetime/mining protocol (P1). This also changes the mining budget.
2. G0 step budget and `Budgets` values (P2). This means a new planner gene version if `max_plan_steps` ≠ 8.
3. T_reg selection rule and definition of "solved" (P3).
4. Seed-pooling estimator for H1/H2 (P4).
5. Arm-B representation and projection (P5).
6. The erratum on interaction worst-case episodes (2,700, not 900) and the re-estimated `episodes_est` (Y2).
7. Any H2 non-degeneracy condition.
8. Moving from sequential to concurrent serving, if chosen for wall-clock reasons.
9. Any change to stage-2 n, τ_r, the replay rule or the H2 +5 pp threshold that the pilot may motivate (PA R2, R3, R9). Per VR, these must be revisited **before** the run, never after.

## 10. Limitations of this draft

- **The proposed P(support) values are subjective priors (S).** They combine PA's model outputs with anchor-based judgement. They should be replaced or endorsed by the team, not cited as estimates.
- **The compute model is built only from assumed constants.** The tokens per call, decode/prefill rates, ALFWorld reset time and concurrency speed-up are all assumptions (U). The pilot must measure them; the wall-clock cap is what matters, not the point estimate.
- **The stage-count values are copied from PA T1.** They assume the symmetric noise model at p0 = 0.5 with PA's candidate mixtures and generalist traits. Specialists pass stage 2 less often (PA §4.5). Single-family Look, Heat or Cool specialists cannot pass it at all under the proportional 20-task allocation. That makes arm C effectively "general or high-effect traits only" (V6, P1). [corrected after independent verification 2026-10-13]
- **The step-length reasoning for the 8-step cap is mine (U).** The two cited expert-length statistics are V, but they count steps somewhat differently from each other and possibly from the future adapter.
- **PR #28 is merged at `6014bc7`.** Gate behaviour (no short-circuit; interaction per seed) was read at `c6ce4c1`. The verification diffed through `6014bc7` and found no threshold, pass rule, bootstrap convention or elimination logic changed (R12). [corrected after independent verification 2026-10-13]
