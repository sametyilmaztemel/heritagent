# ADR-0003 — ALFWorld `ToolEnvironment` adapter for EXP-0001

Date: 2026-10-13 · Status: **ACCEPTED by the owner** (Samet, 2026-10-13: "accept the recommendations"); **amendment A1** chosen by the owner the same day (explicit answer in chat); **amendment A2** chosen by the owner on 2026-10-14 (explicit answer in chat). **Amended by ADR-0009** (owner, 2026-10-15): a pre-registered context window of 32,768 tokens with an overflow that ends the episode as a failure (D4); D8's data tree is the two pinned ALFWorld 0.2.2 archives plus every game's content hash (ADR-0008 D6); the task text's repetition as observation 0 is part of the frozen G0 prompt (D5); `won` fails closed (D6). Cross-reviewed by builder-2 as Critic (2026-10-15, #13); Codex was removed on 2026-10-14. Proposal and alternatives: `research/proposals/ADR-draft-alfworld-tool-environment.md` · Artifacts: `runtime/loop.py` (`ToolObservation.terminal` / `retryable`, status `environment_terminal`), `trajectory/` (terminal kind), `runtime/expression/payload_schemas/planner_policy.schema.json`, and later `runtime/tool_adapters/alfworld.py`.

**Problem.** EXP-0001 runs on ALFWorld, but the repo has no adapter from ALFWorld's text games to the runtime's `ToolEnvironment` boundary. The adapter's protocol choices set G0's success level. Published Qwen2.5-7B-Instruct results span 14.8–45.5 % across protocols, and rendering alone moves 83.6 % to 1.5 %. G0's level in turn sets T_reg feasibility, the expected number of mining pairs and every power number. These choices are therefore pre-registration elements.

**Current design.**
- The catalog is static per episode. A tool failure is retried by the retry policy. An episode ends only on a `Final Answer` or when a budget runs out.
- `planner_react_v1` caps G0 at 8 steps, and the payload schema allows at most 16.
- Benchmark success comes from the episode runner, never from the runtime's `finished`.

**Decision (owner-accepted recommendations of the proposal).**
- **D1 Tool surface.** One tool, `act`, with schema `{"command": string}`. ALFWorld's native interface is free text.
- **D2 Observation.** ALFWorld's observation text, then `Admissible commands:` with the sorted `admissible_commands`, one per line. Identical in every arm and part of the frozen G0 prompt.
- **D3 Invalid commands.**
  - "Nothing happens." returns `ok=False` with `retryable=False`; anything else returns `ok=True`.
  - *Clarification found during implementation:* the loop retries every `ok=False` result with real backoff sleeps. A deterministic agent error would then be re-sent up to `max_retries` times, spending game steps and wall-clock for nothing. `ToolObservation.retryable` (default `True`) lets the environment mark such a failure. The loop then treats it like a rejected tool call: counted as a failure, not retried.
  - This keeps D3's intent (counters reflect real failures) without the retry side effect. **Codex review requested on this clarification specifically.**
- **D4 Step budget.**
  - Planner gene `planner_react_v2` with `max_plan_steps: 50`, and the planner payload-schema maximum raised from 16 to 64.
  - `Budgets.max_steps = 50`, identical in every arm and recorded in the manifest.
  - **Amends** the EXP-0001 manifest's `genome.initial` (`planner_react_v1 + retry_backoff_v1`) to `planner_react_v2 + retry_backoff_v1`. The YAML is updated in the same change as the G0 switch, so the manifest and the ADR never disagree on `main`.
- **D5 Episode end.** `ToolObservation.terminal` (default `False`). When it is set, the loop emits `environment_terminal` and returns that status. The trajectory schema and recorder accept it as a third terminal kind.
- **D6 Success.**
  - ALFWorld's `won` at episode end, never the model's text and never the runtime status.
  - The adapter exposes `won`, steps and the game id to the episode runner.
  - Family and game id are evaluation-only and never model-visible.
- **D7 Task text.** ALFWorld's initial observation minus the `-= Welcome to TextWorld, ALFRED! =-` banner. *As amended by A1:* rendered with the initial admissible commands exactly as D2 renders every later observation. Nothing else is added.
- **D8 Versions and determinism.**
  - `alfworld==0.4.2` and `textworld==1.7.0` in an optional `alfworld` extra.
  - `AlfredDemangler(shuffle=False)` and `AlfredInfos` on every split, no expert.
  - One TextWorld env per episode.
  - The data-tree hash recorded in the split manifest. [ADR-0009 D5: the tree is pinned by the two ALFWorld 0.2.2 archive hashes (ADR-0008 D6), and every game the run uses by its content hash.]

**Reason.**
- D1, D2, D4 and D7 follow the published protocols closest to EXP-0001's cited anchors (the SAGA and verl-agent step limit of 50; text actions).
- D3 and D5 keep the agent-observable counters (regulation) and step accounting meaningful.
- D6 keeps the #11 rule and the evaluation-only metadata rule.
- D8 removes known nondeterminism: object-name shuffling on train games.

**Research implication.**
- D1, D2, D4 and D7 fix G0's level and must stay frozen through the calibration pilot and the run. A change after the pilot is a new experiment id.
- Format collapse (rendering) is a distinct failure mode from the arm-B treatment, so the pilot includes a rendering smoke test.
- No novelty or causal claim: this is benchmark integration.

**Experiment needed.** The calibration pilot (P0-9) on a disjoint train pool under D1–D8 measures:
- G0 p0 and the invalid-command rate;
- steps to success and throughput;
- an arm-B rendering smoke test;
- determinism (the same game twice at temperature 0 gives identical trajectories).

**Compatibility impact.**
- D3 and D5 add two optional `ToolObservation` fields and one run status. Existing environments are unaffected; the trajectory schema 0.1 gains an additive status value.
- D4 raises the planner payload maximum (additive). The G0 genome switch to `planner_react_v2` changes G0's canonical `genome_id`. It lands after PR #29 (canonical `genome_id`, which also edits `runtime/seed_g0.py`) so the two do not conflict.
- D8 adds an optional dependency extra; the core install stays `jsonschema` only.

**Implementation status.**
1. Runtime support (D3 `retryable`, D5 `terminal`, the trajectory terminal kind, the D4 schema maximum): this change.
2. `runtime/tool_adapters/alfworld.py` (D1, D2, D3, D6, D7, D8), with tests against a scripted fake of TextWorld's single-game gym API (`reset() -> (obs, infos)`, `step(cmd) -> (obs, score, done, infos)`). Real-game integration is not yet run: no ALFWorld data in the build sandbox, and the integration test is skipped unless `ALFWORLD_GAMEFILE` is set. The open question on the initial admissible list is resolved by amendment A1.
3. The G0 switch to `planner_react_v2` (D4) and the manifest amendment: done after PR #29 merged. G0's planner is `planner_react_v2` with `max_plan_steps: 50`, and G0's canonical `genome_id` changed with the planner gene. The manifest's `genome.initial` names `planner_react_v2 + retry_backoff_v1`. The episode budget is declared as `runtime.seed_g0.EPISODE_MAX_STEPS` (50) and recorded as `environment.episode_max_steps: 50`. No code applies it yet: the EXP-0001 episode runner (#13) must build every arm's `Budgets` from it.
4. Amendment A2 (stop sequence): planner, loop, gate settings and manifest, in the change that records it.

**Amendment A1: the initial observation carries the admissible commands.** The owner chose this in chat on 2026-10-13 ("Evet, göster"). Codex cross-review is requested.
- *Problem.* As first written, D7 added nothing to the task text and D2 covered only step observations. The model therefore saw the first admissible-command list only after its first action, so that first action was blind.
- *Decision.* D2 applies to the initial state too. The task text is the initial observation minus the banner, then `Admissible commands:` and the sorted initial `admissible_commands`, rendered exactly like every later observation. It is the same in every arm and part of the frozen G0 prompt.
- *Reason.*
  - Every observation the model sees has the same form.
  - The proposal's D2 rationale (`research/proposals/ADR-draft-alfworld-tool-environment.md`) applies equally to the first action: showing the admissible commands is *expected* to reduce invalid commands. That expectation is unmeasured here. The calibration pilot reports the invalid-command rate under A1, but it has no comparison run without the list, so the expectation stays untested. The proposal also notes that admissible-action display is not stated for every anchor.
  - A blind first action is expected to add invalid commands that would measure the protocol rather than the agent.
- *Research implication.* Like D2, A1 sets G0's level, so it stays frozen through the calibration pilot and the run.
- *Compatibility.* `AlfworldEnvironment.start()` returns the rendered text, so callers that pass `start()` to `run_agent` need no change. Since the own review of #48, a game whose infos lack `admissible_commands` fails closed at `start()` and at every step. Without that check the model would see an empty command list.

**Amendment A2: generation stops before an observation line.** The owner chose this in chat on 2026-10-14 ("Evet, 'Observation:'"), after #55 raised the question. Codex cross-review is requested.
- *Problem.*
  - D1–D8 set no stop sequence, so a chat model can keep writing past its `Action Input` and invent the environment's reply (`Observation: …`) and further steps.
  - Since #55 such a reply is a mixed step: it fails closed, and the retry policy reprompts. A model that repeats the pattern spends the retry budget and can lose an episode it could have played. That would measure the protocol, not the agent.
- *Decision.*
  - Every ReAct model call stops at `"\nObservation:"`, the start of a line in which the model would begin an observation. The text before it is parsed as one step under the #55 rules, which are unchanged.
  - The stop sequence belongs to the planner's protocol: `runtime.policies.planner_react.STOP_SEQUENCES`.
  - `run_agent` uses it by default. It rejects explicit settings whose `stop` differs, as it does for `max_tokens` (#56).
  - The gate's locked evaluation settings (`inheritance.overlay.evaluation_settings`) carry it, so every CIG child records it in `eval_generation_settings`.
  - Every `model_called` event records it.
  - The EXP-0001 manifest records it as `model.eval_stop`.
  - It is the same in every arm and part of the frozen G0 protocol.
- *Reason.*
  - In a ReAct loop the environment, not the model, supplies observations. Stopping where the model starts one keeps each call to one step.
  - It is *expected* to reduce mixed replies and the retries they cost. That expectation is unmeasured here: the calibration pilot reports parse-failure and retry rates under A2, but it has no comparison run without the stop, so the expectation stays untested.
  - The #55 mixed-step check stays as a backstop for a backend that does not honour stop sequences.
- *Research implication.* Like D1, D2, D4 and D7, A2 sets G0's level, so it stays frozen through the calibration pilot and the run. No novelty or causal claim.
- *Compatibility.*
  - Callers that pass no settings need no change.
  - A caller that passes settings must pass the protocol stop.
  - The gate's stage 1–2 evidence check now requires the protocol stop in each replay and ablation child (it required an empty stop before). Every stage, regression included, runs with `evaluation_settings`.
  - The vLLM backend already forwards `stop`. The scripted test adapters return replies as if the backend had applied it.
  - No CIG record or trajectory exists yet, so no stored evidence changes.
