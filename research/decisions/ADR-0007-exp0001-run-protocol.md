# ADR-0007 — EXP-0001 run protocol: lifetime, pairs, splits, pilot, T_reg and transfer subset

Date: 2026-10-14 · Status: **ACCEPTED by the owner** (Samet, 2026-10-14, explicit answers in chat, each on the recommended option: E1–E4 first, then E5–E8 on builder-2's proposals; for E6's all-family case the owner chose a vacuous out-of-scope stage over failing closed). **Amended by ADR-0008**: E5's wall-clock clause counts active run time (the owner's decision, 2026-10-14, ADR-0008 D2), and its cap checks count the attempts in flight (builder-2's mechanism under ADR-0008 D3). **Cross-review requested (SPEC §32)**, in particular for E6, which changes the stage-3 evaluator (`inheritance/generalization.py`); since the owner removed Codex on 2026-10-14 it is builder-2's (takeover note §5.17). Artifacts: `evaluation/exp0001/` (`protocol.py`, `pilot.py`, `provenance.py`, `ledger.py`, `episodes.py`, `lifetime.py`, `allocations.py`, `gate_runner.py`, `arms.py`, `pipeline.py`, `backends.py`), `experiments/manifests/EXP-0001-split-plan.json`, `EXP-0001-splits.json`, `EXP-0001-pilot.json`, the manifest's `run_protocol` block · Origin: issue #13 Phase A handoff (the Critic's), which lists the pre-run artifacts and policies this ADR fixes; the owner reassigned the rest of Phase A to builder-2 on 2026-10-14.

**Problem.**
- **No same-task pairs.** The miner takes `MiningPair`s: a verified success and a verified failure of the *same* task (`traits/miner/inputs.py`). The design's lifetime runs each of the 60 T_mine tasks once per seed (design §7: "Mining: 3 seeds × 60 tasks = 180 episodes"), so no task has two trajectories in a seed and no pair can form. The README already lists "the lifetime/mining-pair protocol" among the design decisions open before #13.
- **Split inputs.** The split builder applies the locked rules, but the per-family counts and the seed are pre-registration inputs (`evaluation/benchmarks/alfworld/splits.py`). No `SplitPlan` existed.
- **Pilot.** ADR-0003 requires a calibration pilot on a disjoint train pool. No pilot selection existed.
- **T_reg.** "Subset of T_mine that G0 solved" does not say which lifetime episodes count, or which 40 to take. ADR-0006 D8 requires one set shared by every seed.
- **Transfer subset.** Design §4 names "a 20-task T_gate transfer subset" for adaptation cost, but no pool defines it.
- **Per-episode budgets.** Only `max_steps = 50` is locked (ADR-0003 D4). The retry and token budgets of `runtime.loop.Budgets` are not.
- **Stage-3 allocation.** The design draws 10 in-scope games from `generalization_bank[f]` over the candidate's applicability families and 10 out-of-scope games from the other families. It does not say how 10 games split over several families, which games of a family are taken, or what happens when the applicability names all six families (the miner allows any non-empty subset).
- **Arm genomes.** The manifest says "G0 + all frozen candidates" (B) and "G0 + validated candidates" (C). It does not say how the added skills are recorded in the G1 genome.
- **Run order.** T_reg (E3) needs every seed's lifetime, and the gate's regression stage needs T_reg, so the stage order is not free.

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
  - Fewer than 40 eligible tasks fails closed. A shortfall is known only after the 540 lifetime episodes, when the manifest's immutability clause already applies, so a changed rule then needs a new experiment revision.
- **E4 Transfer subset (owner).** The adaptation-cost transfer subset is the 20-task ablation pool. The gate used these tasks for arm C's selection, so the measure can favour C. It is descriptive only (ADR-0006 D6).
- **E5 Per-episode budgets and the cap rule (owner, on builder-2's proposal).**
  - `Budgets(max_steps=50, max_total_retries=10, max_tokens_per_request=512)`, identical in every arm and stage.
  - The cap rule. Before each episode, the run stops and is marked incomplete if:
    - one more episode would exceed `episodes_max`;
    - the model calls so far plus an episode's worst case (`max_steps + max_total_retries`) would exceed `model_calls_max`;
    - the wall clock since run start exceeds `wall_clock_hours_max`.

    A resumed run reads its ledger back and never runs a recorded episode twice.
  - The miner's teacher calls (one per mining pair) count against `model_calls_max` too; they are not episodes.
  - Every admission is recorded before it runs, with the calls it reserves. An episode or a mining batch that a crash leaves unfinished is charged its whole reservation, so calls made before a crash still count (conservatively), and the number of aborted attempts is reported.
  - The caps in force are the ones the frozen run-start record carries; a run cannot be resumed under other caps.
- **E6 Stage-3 allocation (owner, on builder-2's proposal).**
  - In-scope: the 10 games split as evenly as possible over the applicability families; the remainder goes one game each to the first families in name order (2 families: 5 + 5; 3: 4 + 3 + 3).
  - Out-of-scope: the same split of 10 over the other families.
  - Per family, the games with the smallest `sha256("heritagent-gen|0|<family>|<game_id>")`. Every candidate that needs n games of a family gets the same n (cross-trait reuse, logged in the CIG records).
  - A candidate whose applicability names all six families leaves no out-of-scope family. Its out-of-scope stage is then **vacuous**: no episode runs and the stage passes, with the reason in its child record. Its 10 in-scope games cover every family (2, 2, 2, 2, 1, 1 by the split above). The pooled in-scope delta (Δ ≥ 0 over those 10 games) is then its only harm check; no per-family or out-of-scope threshold is applied, and its child record and aggregate say so. The evaluator accepts an empty out-of-scope allocation only with the family universe the allocation carries (EXP-0001 passes ALFWorld's six families), and only when the stored artifact's applicability names all of it.
  - Replay follows from the design and is recorded here: the source task is the T_mine task of the candidate's source trajectory (its mining pair's success), and the 4 variants are the 4 games of `replay_bank[<that task's family>]`.
- **E7 Arm genomes (owner, on builder-2's proposal).**
  - G1 is generation 1 with parent G0. The added skills follow the frozen rank order.
  - Each added skill is a germline ref with `origin: assimilation`, its source trajectory and the aggregate CIG record of its gate run. Arm B's skills carry their record too, whatever its verdict, so every assimilated gene traces to its evidence.
  - Origin and provenance are not part of the genome id (ADR-0001 A2): B and C genomes with the same skills share one id.
  - An arm that adds no skill (the gate promoted nothing, or the miner proposed nothing) is G0 itself, not a generation-1 record that names its own id as parent.
- **E8 Run order (owner, on builder-2's proposal).**
  - All three lifetimes, then T_reg, then per seed the miner, the gate and the arm genomes, then the final evaluation and the final statistics. T_eval is touched only in the final evaluation.
  - The final evaluation runs every arm at temperature 0 with no sampling seed (arm A once, ADR-0006 D7). The gate keeps its locked settings (seed = the mining seed).
  - Every stage result is a write-once artifact in the run directory; a crash resumes from the ledger and the artifacts. T_reg and the arm genomes are re-derived on every resume and must equal their frozen artifacts.
  - The gate's decisions are written to its artifact before the somatic journal takes them, and on completion and on a protocol stop the report is written before the ledger's terminal record, so a crash between the two is finished on resume. A hard cap writes its terminal record first; a crash before its report leaves a report rebuilt from the ledger, without the stage or the partial statistics. A crash inside an episode or a mining batch re-runs that unit; its partial calls stay charged (E5).
  - One process writes a run directory at a time (an exclusive lock).
  - The run binds to its run-start record: the split and pilot manifests must verify from their hashes and be the committed ones, and the agent model, the teacher, the token counter's tokenizer and the caps must be the ones the record names.
  - CIG ids are `CIG-<seed><rank + 1>` (two digits each), with `CIG-<seed>00` as the seed's batch id.

**Reason.**
- **E1:** three attempts give same-task pairs without changing the miner. At a G0 success rate near 30 %, about 1 − 0.3³ − 0.7³ ≈ 63 % of tasks yield a pair (status M), so about 35–40 pairs per seed. The seed-derived sampling seeds keep mining reproducible.
- **E2:** seed 0 matches the stage-2 convention. The counts follow from the data by the existing helper, so they can be recomputed from the committed manifest.
- **E3:** "solved in any lifetime" is the most feasible reading of "G0 solved". With 9 attempts per task, almost every T_mine task qualifies near p ≈ 0.3. The keyed-hash draw is interpreter-independent, like the split draw.
- **E4:** it is literally a 20-task T_gate subset, already frozen and stratified.
- **E5:** these are the loop's own defaults for tokens, with a retry budget large enough for the G0 retry gene (3 retries per call). The worst-case call count makes the cap check conservative: no episode can start that could push the run past the cap.
- **E6:** the even split treats every declared family alike. One keyed order per family keeps cross-candidate comparisons paired. For an all-family applicability, failing closed would end the run after its 540 lifetime episodes; eliminating the candidate would penalize the most general skills and could bias the comparison against arm C. The vacuous stage keeps a harm check through the pooled in-scope rule, weaker than a per-family check (see Research implication).
- **E7:** the CIG record on every assimilated gene keeps the audit chain of ADR-0001 R1 for both treated arms; the record's verdict shows which of B's genes the gate rejected.
- **E8:** the order is forced by E3 and the gate's regression stage. Once a seed's candidate set is frozen, its temperature-0.7 teacher calls are never made again, so a resumed run cannot mine a different set for that seed.
- All choices were made before any EXP-0001 or pilot episode. None was chosen for its effect on an outcome.

**Research implication.**
- **E1:** the lifetime becomes 540 episodes (+360 over design §7's 180). This adds about 10–20k model calls to the cap pressure noted in the takeover note §5.12 (status M).
- **E3:** T_reg's "solved" comes from temperature-0.7 episodes, while G1 evaluation runs at temperature 0. Even arm A can therefore fail some T_reg tasks. The H2 regression contrast reg(C) − reg(B) is unaffected (ADR-0006 D4).
- **E6:** an all-family candidate is checked for harm only by the pooled in-scope delta on 10 games across the six families (Δ ≥ 0, one or two games per family), not on 10 games outside its scope (Δ ≥ −0.05): a harm in one family can be offset by a gain in another. Its CIG record says so.
- **E3/E1:** the calibration pilot measures G0 at temperature 0 only, while T_reg eligibility and the pair yield depend on the temperature-0.7 success rate. Whether the pilot should also project those is an open question for the owner (it would change the frozen pilot manifest).
- **E8:** all three lifetimes run before any gate, so a stopped run may hold lifetime data for seeds whose gate never ran.
- No novelty or causal claim.

**Experiment needed.** The calibration pilot (ADR-0003) runs under this protocol in Phase B, after the Critic clears Phase A.

**Compatibility impact.**
- New modules, artifacts and tests.
- E6 changes one existing behaviour: `GeneralizationAllocation` accepts an empty out-of-scope allocation when it carries the family universe and the applicability names all of it, and the evaluator then runs 20 episodes instead of 40. Every other allocation, and its child records, are unchanged. For such a candidate it supersedes the locked `inheritance.stages.generalization.out_of_scope` line (10 tasks), which the manifest's `run_protocol` block notes; the stage line itself changes only through the Codex/owner path.
- The locked manifest gains a `run_protocol` block that points at the committed artifacts and their hashes.
