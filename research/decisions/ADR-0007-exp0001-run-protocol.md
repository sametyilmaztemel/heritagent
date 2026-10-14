# ADR-0007 — EXP-0001 run protocol: lifetime, pairs, splits, pilot, T_reg and transfer subset

Date: 2026-10-14 · Status: **E1–E4 ACCEPTED by the owner** (Samet, 2026-10-14, explicit answers in chat, each on the recommended option). **E5 PROPOSED by builder-2, pending the owner.** **Codex cross-review requested (SPEC §32).** Artifacts: `evaluation/exp0001/` (`protocol.py`, `pilot.py`, `provenance.py`, `ledger.py`, the lifetime, pairing and T_reg modules), `experiments/manifests/EXP-0001-split-plan.json`, `EXP-0001-splits.json`, `EXP-0001-pilot.json`, the manifest's `run_protocol` block · Origin: issue #13 Phase A handoff (the Critic's), which lists the pre-run artifacts and policies this ADR fixes; the owner reassigned the rest of Phase A to builder-2 on 2026-10-14.

**Problem.**
- **No same-task pairs.** The miner takes `MiningPair`s: a verified success and a verified failure of the *same* task (`traits/miner/inputs.py`). The design's lifetime runs each of the 60 T_mine tasks once per seed (design §7: "Mining: 3 seeds × 60 tasks = 180 episodes"), so no task has two trajectories in a seed and no pair can form. The README already lists "the lifetime/mining-pair protocol" among the design decisions open before #13.
- **Split inputs.** The split builder applies the locked rules, but the per-family counts and the seed are pre-registration inputs (`evaluation/benchmarks/alfworld/splits.py`). No `SplitPlan` existed.
- **Pilot.** ADR-0003 requires a calibration pilot on a disjoint train pool. No pilot selection existed.
- **T_reg.** "Subset of T_mine that G0 solved" does not say which lifetime episodes count, or which 40 to take. ADR-0006 D8 requires one set shared by every seed.
- **Transfer subset.** Design §4 names "a 20-task T_gate transfer subset" for adaptation cost, but no pool defines it.
- **Per-episode budgets.** Only `max_steps = 50` is locked (ADR-0003 D4). The retry and token budgets of `runtime.loop.Budgets` are not.

**Decision.**
- **E1 G0 lifetime (owner).**
  - In each seed's lifetime, G0 runs every T_mine task 3 times at the locked mining temperature 0.7, in game_id order, attempts 0, 1, 2.
  - A task with at least one success and at least one failure gives one mining pair: its first success and its first failure, in attempt order. A task that never fails, or never succeeds, gives none.
  - Each attempt's sampling seed is fixed by rule (the manifest requires an explicit stochastic source): `int(sha256("heritagent-attempt|<mining_seed>|<task_id>|<attempt>")[:8], 16) mod 2**31`.
- **E2 Split plan and pilot (owner).**
  - `SplitPlan` seed 0, with counts from the real ALFWorld inventory:
    - train: 3553 games accepted by the loader rules; valid_unseen: 134, the full T_eval;
    - t_mine 60, proportional to availability with a floor of 8 per family (9/11/10/10/9/11);
    - ablation 20 and interaction 10, proportional;
    - replay bank 4 per family; generalization bank 10 per family.
  - No feasibility fallback is needed: the smallest family has 308 accepted games.
  - The calibration pilot: 30 train games, 5 per family, drawn with seed 0 by the split builder's keyed-hash rule (pool name `calibration_pilot`) from the games left after every EXP-0001 pool. Its measures and execution plan are frozen in its manifest; it tunes nothing.
- **E3 T_reg (owner).**
  - Eligible tasks are the T_mine tasks G0 solved in at least one lifetime episode of any seed.
  - The 40 eligible tasks with the smallest `sha256("heritagent-treg|0|<task_id>")` form T_reg, one set for every seed (ADR-0006 D8).
  - Each task's baseline is its earliest success, in seed order and then attempt order.
  - Fewer than 40 eligible tasks fails closed; changing the rule then needs a pre-execution amendment.
- **E4 Transfer subset (owner).** The adaptation-cost transfer subset is the 20-task ablation pool. The gate used these tasks for arm C's selection, so the measure can favour C. It is descriptive only (ADR-0006 D6).
- **E5 Per-episode budgets and the cap rule (builder-2 proposal, pending the owner).**
  - `Budgets(max_steps=50, max_total_retries=10, max_tokens_per_request=512)`, identical in every arm and stage.
  - The cap rule. Before each episode, the run stops and is marked incomplete if:
    - one more episode would exceed `episodes_max`;
    - the model calls so far plus an episode's worst case (`max_steps + max_total_retries`) would exceed `model_calls_max`;
    - the wall clock since run start exceeds `wall_clock_hours_max`.

    A resumed run reads its ledger back and never runs a recorded episode twice.

**Reason.**
- **E1:** three attempts give same-task pairs without changing the miner. At a G0 success rate near 30 %, about 1 − 0.3³ − 0.7³ ≈ 63 % of tasks yield a pair (status M), so about 35–40 pairs per seed. The seed-derived sampling seeds keep mining reproducible.
- **E2:** seed 0 matches the stage-2 convention. The counts follow from the data by the existing helper, so they can be recomputed from the committed manifest.
- **E3:** "solved in any lifetime" is the most feasible reading of "G0 solved". With 9 attempts per task, almost every T_mine task qualifies near p ≈ 0.3. The keyed-hash draw is interpreter-independent, like the split draw.
- **E4:** it is literally a 20-task T_gate subset, already frozen and stratified.
- **E5:** these are the loop's own defaults for tokens, with a retry budget large enough for the G0 retry gene (3 retries per call). The worst-case call count makes the cap check conservative: no episode can start that could push the run past the cap.
- All choices were made before any EXP-0001 or pilot episode. None was chosen for its effect on an outcome.

**Research implication.**
- **E1:** the lifetime becomes 540 episodes (+360 over design §7's 180). This adds about 10–20k model calls to the cap pressure noted in the takeover note §5.12 (status M).
- **E3:** T_reg's "solved" comes from temperature-0.7 episodes, while G1 evaluation runs at temperature 0. Even arm A can therefore fail some T_reg tasks. The H2 regression contrast reg(C) − reg(B) is unaffected (ADR-0006 D4).
- No novelty or causal claim.

**Experiment needed.** The calibration pilot (ADR-0003) runs under this protocol in Phase B, after the Critic clears Phase A.

**Compatibility impact.**
- New modules, artifacts and tests. No existing behaviour changes.
- The locked manifest gains a `run_protocol` block that points at the committed artifacts and their hashes.
