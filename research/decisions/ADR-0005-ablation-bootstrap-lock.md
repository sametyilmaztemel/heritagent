# ADR-0005 — The stage-2 ablation bootstrap is pre-registered: seed 0, task_id order

Date: 2026-10-14 · Status: **ACCEPTED by the owner** (Samet, 2026-10-14, explicit answer in chat: "Lock in the manifest: seed 0 + task_id order"). **Codex cross-review requested (SPEC §32).** **Amended by ADR-0009 D5** (owner, 2026-10-15): the method is the percentile paired cluster bootstrap at confidence 0.95, its generator Python's `random.Random(0)` with one `randrange(n)` per resampled cluster, and the gate recomputes the bound from the child record, both before any stage-3/4 episode and whenever a run reads its gate artifact back. Artifacts: `inheritance/ablation.py` (`BOOTSTRAP_SEED`, `BOOTSTRAP_CLUSTER_ORDER`, `AblationEvaluator`), `inheritance/gate.py` (`PRIMARY_THRESHOLDS`, the stage 1–2 evidence check), `experiments/manifests/EXP-0001-minimal-cig.yaml` (`inheritance.stages.ablation.bootstrap`), `tests/inheritance/test_bootstrap_lock.py` · Origin: audit finding F2 / P1-12; takeover note §5.7.

**Problem.**
- The stage-2 ablation verdict is "point estimate ≥ τ_c (0.05) and the 95 % paired cluster bootstrap lower bound > 0" over 20 paired task deltas (ADR-0002; manifest `inheritance.stages.ablation`).
- The locked text fixes the method and 10,000 resamples. It fixes neither the seed nor the order in which the deltas enter the resampling.
- **The seed was a free constructor argument** (`bootstrap_seed`, default 0). Since #31 a non-int seed is rejected, but any int was accepted.
- **The order was the allocation's order.** With a fixed seed the random index stream is fixed, so permuting the deltas changes which delta each index picks, and so the bound.
- Both choices can decide the verdict near the boundary. With seed 0, successes on 8 of 20 tasks with the trait and 3 without give a point estimate of 0.25. The allocation order then gives a lower bound of 0.00 (stage fails), and task_id order gives 0.05 (stage passes). Status M: computed by running the locked implementation, reproduced in `tests/inheritance/test_bootstrap_lock.py`.
- A free seed or order is a researcher degree of freedom after the data are in.

**Decision (owner).**
- **Seed.** The ablation bootstrap seed is pre-registered as **0**.
  - `AblationEvaluator` rejects any other value before any episode.
  - The constructor argument stays only for that check.
- **Order.** The paired deltas enter the bootstrap **ordered by `task_id`** (ascending, Python string order, i.e. Unicode code point order). Task ids are distinct strings by the allocation contract: `AblationAllocation` rejects a non-string id before any episode.
  - Episodes still run, and are still recorded, in allocation order.
  - The child record stores `paired_deltas` in task_id order, with `paired_delta_task_ids` naming each delta's task, so the bound recomputes from the record alone.
- **Resamples.** 10,000, unchanged (ADR-0002).
- **Recording.**
  - The child record's bootstrap block carries `seed`, `n_resamples` and `cluster_order: "task_id"`. (Extended by ADR-0009 D5: it also carries the `confidence`, 0.95, next to the `method`.)
  - The aggregate's `thresholds_used` carries `bootstrap_seed` and `bootstrap_cluster_order`.
  - The manifest's ablation stage carries `bootstrap: {seed: 0, resamples: 10000, cluster_order: task_id}`. (Extended by ADR-0009 D5: `method: percentile`, `confidence: 0.95`, the `generator` and `recomputed_by_gate: true`.)
  - The gate checks these values before any stage-3/4 episode: the child's and the evaluation's bootstrap blocks must carry seed 0, 10,000 resamples and `cluster_order: "task_id"`, and `paired_delta_task_ids` must be distinct strings in ascending order. So the aggregate's `thresholds_used` matches what the ablation evidence declares. The gate checks the declared values; it does not recompute the bound. (Superseded by ADR-0009 D5: the gate now recomputes the bound, the point estimate and the stage verdict from the record's deltas, and checks that the deltas' tasks are the allocation's.)
- **Scope.**
  - This covers the gate's primary ablation stage only.
  - Sensitivity analysis (`sensitivity_analysis`) stays analysis-only (ADR-0002: secondary recomputation on existing stage evidence). As `inheritance/ablation.py` documents, its seed stays free, since it never enters the primary gate. The evaluation object's convenience method uses the task_id-ordered deltas.
  - The final-evaluation statistics (H1/H2, #13) are not implemented yet and are not decided here. The same rule is the natural default, but #13 decides it.

**Reason.**
- Pre-registering both before any data removes the freedom to choose a seed or an order after seeing the verdict.
- task_id order makes the bound a function of the measured outcomes alone, independent of how the allocation was built or shuffled.
- Seed 0 was already the recorded default. Locking it changes no behaviour of a default-built evaluator except the delta order.
- The rule was fixed before any data. It was not chosen for its effect on any verdict: the example above happens to pass under task_id order, and its only role is to show that the order matters.

**Research implication.**
- None for the hypotheses. This removes an analysis degree of freedom from stage 2.
- No novelty or causal claim.

**Experiment needed.** None. No ablation has run yet.

**Compatibility impact.**
- A caller passing a seed other than 0 now fails closed. In the repo only tests did that, and they are updated.
- A default-built evaluator gives the same verdict whenever the allocation already lists the tasks in task_id order. Otherwise the bound may differ. No CIG record exists yet, so no stored evidence changes.
- The child record gains `cluster_order` (in its bootstrap block) and `paired_delta_task_ids`. The aggregate's `thresholds_used.values` gains `bootstrap_seed` and `bootstrap_cluster_order`. Both places are free-form, so the schemas are unchanged. `THRESHOLD_VERSION` stays `EXP-0001-CIG-M0-v1`, since no record has been produced under it.
- The locked manifest gains one line. The verdict rule text is unchanged.
