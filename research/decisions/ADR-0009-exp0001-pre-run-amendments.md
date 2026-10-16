# ADR-0009 — EXP-0001 pre-run amendments after the Critic cross-review

Date: 2026-10-15 · Status: **ACCEPTED by the owner** (Samet, 2026-10-15, explicit answers in chat, each on the recommended option: D1 "Eşleştirilmiş karşılaştırma", D2 "Bir kez oynat, paylaştır", D3 "Evet, ekle", D4–D7 "Önerildiği gibi uygula"; D8 "Sıkıştır, sığmazsa reddet", the same day, after the second review round of PR #69). Made before any pilot or EXP-0001 episode; the manifest's immutability clause has not started. Origin: builder-2's Critic cross-review of ADR-0003 to ADR-0007, the artifact-binding draft and the #13 Phase A clearance (workflow `wf_ba6185ce-d72` on `main` @ `67b3af7`; result posted on #13). Artifacts: D3, D4, D6 and D8 in PR #70 (`evaluation/exp0001/pilot.py`, `pilot_run.py`, `protocol.py`, `serving.py`, `cli.py`, `backends.py`, `pipeline.py`, `episodes.py`, `ledger.py`, `notebooks/EXP-0001-pilot.ipynb`, `runtime/loop.py`, `runtime/model_adapters/`, `runtime/tool_adapters/alfworld.py`, `traits/miner/` and `traits/schemas/mining-record-0.1.schema.json`, `experiments/manifests/EXP-0001-pilot.json`, the manifest's `model` and `run_protocol` lines); D1, D2 and D5's remaining notes in the PR that follows it (`inheritance/regression.py`, `gate.py`, `ablation.py`, `evaluation/exp0001/pipeline.py`, `evaluation/metrics/final_statistics.py`, the ADR-0005, ADR-0006 and ADR-0007 notes, the manifest's regression, ablation-bootstrap, out-of-scope and final-evaluation lines); D7 lands before the run.

**Problem.**
- **Stage 4A cannot be passed (blocker; two finders and two verifiers, reproduced on the real code).**
  - ADR-0007 E3 makes a T_mine task T_reg-eligible if G0 solved it in at least one of nine temperature-0.7 lifetime attempts.
  - Stage 4A then requires G0 + trait to solve ≥ 39 of the 40 at temperature 0, against a baseline assumed to be 100 % and never re-measured.
  - At G0's expected success near 30 %, a trait that changes nothing passes with probability ≈ 0 (status M, on the real `derive_t_reg` and `RegressionEvaluator`). Every candidate would be rejected, so arm C would equal G0 = A, H1 could not be supported, and H2 would reduce to A against B.
- **Identical genomes run twice.**
  - The final evaluation keys its episodes by arm. So an arm C that equals G0 re-runs arm A's 194 episodes, and a B equal to C re-runs C's.
  - ADR-0006 D7 shares cells only for arm A.
- **The pilot cannot see the lifetime.** It measures G0 only at temperature 0. The pair yield (E1) and T_reg's eligibility (E3) turn on the temperature-0.7 success, and an E3 shortfall would surface only after the 540 lifetime episodes.
- **The miner's prompt can overflow the window** (second review round of PR #69, measured on real ALFWorld 0.2.2 games with scripted replies; G0's own episode lengths are unmeasured, status M).
  - The teacher's prompt holds both trajectories of a pair in full, and every tool result carries the admissible-command list (ADR-0003 D2).
  - A 50-step failure renders to 12,000–26,000 tokens; with a success and the miner's 2,048 output tokens a pair can pass 32,768.
  - The server then answers HTTP 400, which no layer catches: the run would stop after the 540 lifetime episodes, and a resume would fail the same way.
- **No context window is pre-registered.**
  - The prompt carries the full history for up to 50 steps. With the offline adapter's 8192-token default, a long G0 episode would overflow at about step 26 (real tokenizer, synthetic episode; status M).
  - No rule says what an overflow does.
- **Smaller notes.**
  - ADR-0003 D8 promises a "data-tree hash" that the manifests do not carry.
  - `won` is read fail-open.
  - A2's stop forwarding and its backstop are untested.
  - The step budget has two literals.
  - ADR-0004 states pre-filter counts and says no manifest exists.
  - ADR-0005's check is declarative only.
  - ADR-0006 does not scope the H2 regression clause.
  - ADR-0007 E6 leaves two rules in the manifest for an all-family candidate.

**Current design.**
- Stage 4A: `drop = 1 − success_with/40 ≤ 0.03`, with-trait episodes only, the lifetime success as the baseline (`baseline_rerun: false`).
- The final evaluation runs every (arm, seed) under its own keys; arm A runs once (ADR-0006 D7).
- The frozen pilot (`pilot_sha256 6286d0fa…`) runs 37 episodes at temperature 0.
- No context window; the server's default was 32768 in ADR-0008, but nothing pinned it.
- The miner sends one teacher call per pair with the full evidence; the run charges one call per pair.

**Decision.**
- **D1 Stage 4A is paired (owner).**
  - Per mining seed, G0 without the trait plays every T_reg task once at the gate's locked settings: temperature 0, the mining seed, the ReAct stop and the request budget. That is 40 episodes per seed, shared by every candidate of the seed; the cross-trait reuse is logged.
  - A candidate's drop is `(success_without − success_with) / 40`, and it passes if `drop ≤ τ_r = 0.03`, in exact arithmetic. With 40 tasks that is a net loss of at most one task; a net gain passes.
  - The τ_r sensitivity grid uses the same paired drop.
  - T_reg's selection (E3) and its frozen baseline provenance stay as they are.
  - A gate that promotes nothing in a seed leaves arm C = G0 for that seed. The report says so, and the H1/H2 rules are computed unchanged: C − A is then exactly 0 and supports nothing.
- **D2 Identical genomes share their final-evaluation cells (owner).**
  - The final evaluation runs each distinct G1 genome once per task set and task, keyed by its genome id. Every (arm, seed) whose genome is that genome reads those cells.
  - The report records which arms shared which genome.
  - This extends ADR-0006 D7 from arm A to every arm and amends ADR-0007 E8.
- **D3 The pilot adds three temperature-0.7 attempts per game (owner).**
  - Every pilot game runs three more times at the mining temperature, with E1's attempt-seed rule under pilot seed 0, and the stop and budgets as in the run. That is 90 episodes, so the pilot has 127.
  - The pilot reports:
    - the per-attempt success;
    - the pair yield (games with both a success and a failure);
    - the share solved in any attempt;
    - the temperature-0 success among those solved.
  - From these it projects the lifetime's pairs per seed and a lower bound for T_reg's eligibility (status M).
  - It also reports the largest prompt per call against the window, and loop steps next to game steps.
  - For D8 it counts each temperature-0.7 pair's teacher prompt in both evidence forms with the model's tokenizer and chat template, and reports how many pairs fit in the full form, only without the lists, or in neither. No teacher call is made.
  - The pilot manifest's selection is unchanged; its measures and execution plan change. The new `pilot_sha256` is `4fc2b6fe…aa52`, pinned in `protocol.py` and the manifest's `run_protocol`.
  - The pilot still tunes nothing.
- **D4 The context window is 32,768 tokens, and an overflow is a failure (owner, as recommended).**
  - The window is Qwen2.5-7B-Instruct's native length. It is recorded in the protocol, the run-start record and the manifest's model block.
  - The server is started with it, and the pilot and the run refuse a server that reports another.
  - A model call fits when its prompt tokens plus its `max_tokens` are at most 32,768: that is the server's rule, so a ReAct call (512 tokens) fits with a prompt of up to 32,256 tokens. A call that does not fit ends the episode with status `budget_exhausted` (budget `context_window`): a failure, identical in every arm. The prompt is never truncated and the call never retried. The rejected call generated nothing and is not counted among the episode's model calls. [corrected after independent verification 2026-10-15: the request budget's share of the window, and the count, were not stated.]
- **D5 Wording and scope notes (owner, as recommended).**
  - ADR-0003 D8's data tree is the two pinned ALFWorld 0.2.2 archives plus every game's content hash (ADR-0008 D6).
  - The task text appears both as the task and as observation 0 in every prompt. This is part of the frozen G0 prompt, the same in every arm, and is not changed after the pilot.
  - The manifest's `eval_stop` applies to every ReAct call, the lifetime's included.
  - ADR-0004: the 3,553 train games and the smallest family of 308 are loader-filtered counts, the manifest exists, and no fallback was needed; the fallback is a split-level rule only.
  - Before the run:
    - ADR-0005 names the method, confidence and generator, and the gate recomputes the bound;
    - ADR-0006 states that the H2 regression clause is measured on the gate's own regression suite;
    - ADR-0007's stage-3 manifest line carries E6's all-family exception, and the report counts such candidates and labels the H3 preview as measured on the stage-2 selection pool.
- **D6 Engineering under ADR-0003 (builder-2).**
  - `won` fails closed when a game's infos lack a boolean `won`.
  - New tests: the offline backend forwards the ReAct stop, and a reply that runs on into an invented observation is reprompted.
  - The step budget has one source (`EPISODE_MAX_STEPS`).
- **D7 The artifact-binding draft is accepted as its option A, with the Critic's modifications (owner, as recommended).** It lands before the run, with its own ADR text and tests.
- **D8 A teacher prompt that does not fit is sent without the admissible-command lists, then rejected (owner).**
  - The miner sends a pair's evidence in full first.
  - If the server reports that the prompt and the miner's 2,048 output tokens do not fit the 32,768-token window, the same pair is sent once more in a second fixed form: the evidence with the admissible-command list removed from every tool result. Nothing else changes; the task text keeps its initial list (ADR-0003 A1).
  - If that does not fit either, the pair is rejected with reason `context_overflow` and recorded. No evidence is truncated and no form is sent twice.
  - The mining record names the form its call used (`evidence_form`), and the second form's evidence names itself, so its hash differs from the full form's. The full form's bytes are unchanged.
  - The cap check reserves two teacher calls per pair; the run records the calls actually sent, one per form tried. [corrected after independent verification 2026-10-16: it said the cap check charges two calls per pair.]

**Reason.**
- **D1.** It keeps τ_r = 0.03 and the 40-task T_reg. It measures regression the way every other stage measures a trait, with and without it at the same settings, and it costs 120 episodes, under 2 % of the episode cap. The alternatives were:
  - re-selecting T_reg at temperature 0, which needs a larger T_mine when G0's success is near 30 %;
  - a smaller T_reg, which leaves almost no tolerance.

  Both change locked pools or thresholds.
- **D2.** Duplicate deterministic executions are never independent evidence (design §4), and the sharing saves up to 582 episodes.
- **D3.** The E3 and E1 numbers are known before 540 episodes are spent.
- **D4.** No arm's outcome then depends on how an overflow is handled. A failure is the conservative reading of an episode that cannot continue.
- **D8.** It keeps every pair that can be mined at all, keeps the frozen evidence wherever it fits, and decides by the server's own count. A rejection alone would drop the long failures, which cluster in some families. Cutting the evidence to fit would make what the teacher sees depend on where the cut falls.
- **D5–D7.** They close the records' gaps the cross-review found, without changing any result-producing rule except where D1 to D4 say so.
- **Outcome-blind.** All choices were made before any pilot or EXP-0001 episode, and none for its effect on an outcome.

**Research implication.**
- **D1** changes what stage 4A measures. It is now harm relative to G0 at the evaluation condition, not survival against an assumed 100 %. Stage-4 attrition is therefore expected to come mostly from real harm and the overhead check, so the design's survival expectations are now rough guesses, as they always were.
- **D2.** Wherever genomes coincide, C − A and C − B are exactly 0. A seed in which the gate changed nothing therefore cannot count as support (ADR-0006 D3).
- **D3** adds 90 temperature-0.7 episodes to the pilot. Its new numbers inform the compute-cap decision and any later amendment; they tune nothing inside the pilot.
- **D4** can make a long episode a failure. The pilot's prompt maximum shows how close the run comes to the window.
- **D8.** The teacher sees less of the environment for a pair whose evidence is long, and a pair that fits in neither form yields no candidate. Both are the same in every arm, since the candidate set is mined once per seed. The pilot's count shows how often each happens (status M until the run).
- No novelty or causal claim. No result is called validated (SPEC §32).

**Experiment needed.** The calibration pilot under D3 and D4; the compute-cap decision follows it (takeover note §5.12).

**Compatibility impact.**
- **Pilot manifest.** It gets a new hash. Its 30 games are unchanged, and its measures list gains seven measures.
- **`EpisodeResult`** gains `loop_steps` and `exhausted_budget`.
- **The runtime loop** catches `ContextWindowExceeded`. Every other path is unchanged.
- **The miner (D8)** catches `ContextWindowExceeded`. The mining record gains `evidence_form` and the rejection reason `context_overflow` (schema 0.1, both additive). The cap check reserves two teacher calls per pair; the run records the calls actually sent, one per evidence form tried, so a form the server rejected counts (D8). The pilot's projection counts two per pair. [corrected after independent verification 2026-10-15: it said the run charges two calls per pair.]
- **Protocol.** It gains `context_window` and `context_overflow`, and the run-start record changes with it.
- **Stage 4A (D1)** changes the stage-4 child record (a baseline block and the paired drop) and `inheritance/regression.py`. It lands with D2 before the run.
- **ADR-0003, ADR-0004 and ADR-0008** carry pointers to this ADR since PR #70; ADR-0005, ADR-0006 and ADR-0007 carry theirs since the PR that lands D1, D2 and D5's remaining notes. [corrected after independent verification 2026-10-15: it said ADR-0003 to ADR-0007 carried pointers.] [corrected after independent verification 2026-10-16: it said ADR-0005 to ADR-0007 got theirs "in the next PR", which is now this one.]
