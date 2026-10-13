# ADR draft: ALFWorld `ToolEnvironment` adapter for EXP-0001

**Status: superseded by `research/decisions/ADR-0003-alfworld-tool-environment.md`** (the owner accepted the recommendations on 2026-10-13; Codex cross-review requested). Kept as the record of the alternatives. Originally: **PROPOSED, not adopted.** Drafted by Claude (second builder) on 2026-10-13. Every decision below is for Codex (Research/Critic) and the owner to accept, change or reject (SPEC §32). Claude implements the adopted version. Nothing in this file changes a locked element.

Format: SPEC §31 (Problem / Current design / Proposed change / Reason / Research implication / Experiment needed / Compatibility impact). The prior audit items this resolves are P0-2 (step budget) and P0-7 (ALFWorld adapter) in `research/reviews/2026-10-13-third-agent-research-synthesis.md` on branch `claude/keen-bohr-ku33ye`.

## Problem

EXP-0001 runs on ALFWorld (manifest `environment.benchmark`), but the repo has no ALFWorld code. `runtime/loop.py` says so explicitly ("no ALFWorld code"). Every live stage needs an adapter from ALFWorld's text games to the runtime's `ToolEnvironment` boundary: the G0 lifetime, the gate stages, the T_eval arms and the calibration pilot.

The adapter's choices are not neutral plumbing; they set G0's success level, and through it T_reg feasibility and the H2 power analysis. Published Qwen2.5-7B-Instruct results on the same weights span 14.8–45.5 %, depending on protocol. In ExpRAG Table 17, rendering the same retrieved content as "Textual" instead of "Chat JSON" drops the score from 83.6 % to 1.5 % (prior `ALFWorld-baselines.md`, A2 and A13). These choices must therefore be pre-registered before the pilot, not improvised in code.

## Current design

- **Tool boundary.** `ToolEnvironment` has `list_tools() -> list[ToolSpec]` and `call(name, arguments) -> ToolObservation(ok, content)`. The catalog is read once per episode and rendered into the ReAct prompt.
- **Validation.** Arguments are validated against each tool's JSON Schema before the environment sees them.
- **Termination.** An episode ends on a model `Final Answer` or when a budget runs out. The environment has no way to end it.
- **Step cap.** G0's planner gene `planner_react_v1` has `max_plan_steps: 8`, and the payload schema caps that value at 16. The loop uses `min(budgets.max_steps, max_plan_steps)`.
- **Success.** Benchmark success comes from the episode runner's outcome, never from the runtime's `finished` flag (#11 rule).
- **Splits.** Game instances and their family labels come from the split manifest (`evaluation/benchmarks/alfworld/splits.py`, PR #34). Family labels are evaluation-only.

## Proposed change

The proposal has eight decisions. For each, the alternatives are listed and **one is recommended**.

### D1. Tool surface
- **(a) Recommended:** one tool, `act`, with schema `{"command": {"type": "string"}}`. ALFWorld's native interface is a free-text command. The cited anchors (ReAct-style prompting, SAGA, ExpRAG) all act in text.
- (b) One tool per verb (`goto`, `take`, `put`, `open`, `close`, `toggle`, `clean`, `heat`, `cool`, `use`, `examine`, `look`, `inventory`) with object and receptacle arguments. This is more structured, but no anchor uses it, it adds a schema-induced format change, and it multiplies the tool catalog in the prompt.
- (c) `act` with an `enum` of admissible commands. Not possible: the catalog is static per episode, while admissible commands change every step.

### D2. What an observation contains
- (a) ALFWorld's observation text only.
- **(b) Recommended:** the observation text, then `Admissible commands:` with ALFWorld's `admissible_commands` list (sorted, one per line).

This is the largest single lever on G0's level. The verl-agent/GiGPO family shows admissible actions; it is not stated for every anchor. (b) is recommended because it removes most invalid-command failures, which otherwise dominate 7B behaviour. It must be the same in every arm and is part of the frozen G0 prompt.

### D3. Invalid commands and the runtime counters
ALFWorld answers an inadmissible command with "Nothing happens." and no error.
- **Recommended:** the adapter returns `ok=False` for "Nothing happens." and `ok=True` otherwise. The observable counters that regulation predicates read (`consecutive_failures`, `failing_tools`, `steps_without_progress`) then reflect real agent failures.
- Alternative: always return `ok=True`. The counters stay at zero for the whole episode, and regulated (non-constitutive) traits can never fire.

### D4. Step budget (resolves P0-2)
- **Recommended:**
  - a new planner gene version `planner_react_v2` with `max_plan_steps: 50`, and the payload-schema maximum raised from 16 to 64;
  - `Budgets.max_steps = 50`, identical in every arm and recorded in the manifest.

  50 matches the step limit of the SAGA and verl-agent anchors. Under 8 steps, T_reg (at least 40 of 60 solved) is very unlikely: the prior analysis puts P(≥ 40/60) at 0.0006 even at p = 0.45.
- Alternative: keep 8. The anchors become incomparable and the T_reg rule probably cannot be met (P0-3).

### D5. Episode termination
- **Recommended:** add `terminal: bool = False` to `ToolObservation`. The loop ends the episode with reason `environment_terminal` when the environment reports `done`. The default keeps every existing environment and test unchanged.
- Alternative (no runtime change): after `done`, the adapter answers every call with a fixed "The task is complete." message until the agent gives a `Final Answer` or the budget ends. This wastes steps, adds noise to step counts and token costs, and makes "steps to success" meaningless.

### D6. Success and episode record
- Benchmark success is ALFWorld's `won` flag at episode end. It is never inferred from the model's text and never taken from the runtime's `finished`.
- The adapter exposes `won`, the step count and the game id to the episode runner. The runner (#13) maps them into `EpisodeOutcome`. The family label and the game id stay out of every model-visible message (evaluation-only, manifest `runtime_input_policy`).

### D7. Task text
- The task string passed to `run_agent` is ALFWorld's initial observation after the `-= Welcome to TextWorld, ALFRED! =-` banner is stripped: the room description plus the "Your task is to: …" line. Nothing is added.
- The exact stripping rule is part of the frozen G0 prompt.

### D8. Versions and determinism
- Pin `alfworld==0.4.2` and `textworld==1.7.0` (the versions this draft was checked against) in a `[project.optional-dependencies] alfworld` extra.
- Record the hash of the downloaded data tree in the split manifest. The per-game content hashes already exist.
- Wrap the game with the same TextWorld wrappers alfworld uses (`AlfredDemangler`, `AlfredInfos`), without expert plans, and with `AlfredDemangler(shuffle=False)` on every split. alfworld turns object-name shuffling on for train games when the config sets `env.domain_randomization`, and EXP-0001 draws T_mine and T_gate from train, so the setting must be pinned off.
- One TextWorld env per episode, no batching. Games are deterministic, so the LLM is the only stochastic source.

## Reason

- D1, D2, D4 and D7 follow the published protocols closest to the ones EXP-0001 cites for its expected G0 level. A protocol of our own would make the anchors uninformative for the pilot.
- D3 and D5 keep the runtime's agent-observable counters and step accounting meaningful. Both the gate's regulation semantics and the expression-overhead and adaptation-cost metrics depend on those.
- D6 keeps the #11 rule ("benchmark success from the runner outcome") and the evaluation-only metadata rule.

## Research implication

- **G0 level.** D1, D2, D4 and D7 fix G0's success level, which in turn sets:
  - T_reg feasibility;
  - the expected number of mining pairs (P0-1);
  - every power number in the prior analysis.

  They must be frozen before the calibration pilot. Any later change is a new experiment id (design §2: "any change = new experiment id").
- **Arm comparisons.** D2 and D3 interact with arm B's longer prompts: format collapse is a distinct failure mode (A13). A rendering smoke test in the pilot is needed to keep it separate from the treatment effect.
- **Claims.** None of this is a novelty or causal claim. It is benchmark integration.

## Experiment needed

- Calibration pilot (P0-9) on a train pool disjoint from every EXP-0001 split, under the adopted D1–D8. It measures:
  - G0 success p0 and the invalid-command rate;
  - steps to success;
  - throughput;
  - a prompt-rendering smoke test of arm B's 10-skill prompt.
- A determinism check: the same game, the same G0 and temperature 0, run twice, give identical trajectories.

## Compatibility impact

- **D4:** new planner gene version and a payload-schema change. G0's genome changes, so its canonical `genome_id` changes too (computed since PR #29). This must land before any G0 lifetime run.
- **D5:** `ToolObservation` gains an optional field and the loop gains one termination reason. Existing environments and tests are unaffected by default.
- **D8:** a new optional dependency extra. The core install stays `jsonschema` only.
- **Code:** a new module `runtime/tool_adapters/alfworld.py` (SPEC §23), with tests against a scripted fake of the TextWorld env interface. A real-data integration test is skipped unless ALFWorld data is present.

## What Claude can build now, before the decisions

Only the parts that are the same under every option:
- the env wrapper that resets and steps one game and reads `won`, `admissible_commands` and `extra.gamefile`;
- the episode record of D6.

Nothing that encodes D1–D5 or D7 lands until Codex and the owner decide them.
