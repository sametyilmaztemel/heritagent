# ADR-0004 — When the EXP-0001 bank feasibility fallback applies, and how large it is

Date: 2026-10-13 · Status: **ACCEPTED by the owner** (Samet, 2026-10-13, in chat: "accept the recommendations" for Q1; Q2 chosen explicitly, "Evet, tamamı"). **Codex cross-review requested (SPEC §32).** Artifacts: `evaluation/benchmarks/alfworld/splits.py` (`build_split_manifest`, `_locked_floor_through`), `tests/evaluation/test_alfworld_splits.py` · History: PRs #42, #43, #44 and this change.

**Problem.**
- The locked split rules give the family-indexed banks per-family minima: replay at least 4 per family and generalization at least 10. A "pre-registered family-specific maximum" may replace a minimum **"if availability prevents the minimum"** (manifest `environment.splits.T_gate.evaluation_banks.feasibility_fallback`; design §2; ADR-0002 R1).
- The locked text defines neither when availability "prevents" the minimum nor what the "maximum" must be.
- Both choices decide which games enter the banks, so they are pre-registration elements. Leaving them to code review left two loopholes:
  - **Plan-made shortfalls.** A first implementation (#42) measured the games left at the bank draw. A plan could raise its own earlier counts (for example `replay_bank` 12 for a family with 35 games) and so waive the generalization minimum.
  - **Under-filled banks.** A plan could declare a maximum below the games available and leave a bank under-filled, even empty, for a family that still had games.

**Current design (before this ADR).**
- Since #43 the builder accepts a fallback only on a plan-independent per-family shortfall (Q1 below).
- Any maximum up to the available games is accepted, and `available_at_draw` is recorded.

**Decision (owner-accepted recommendations).**
- **Q1 — When the fallback applies.** A fallback for family `f` in bank `B` is accepted only if `f` has fewer loader-accepted train games than the sum of the locked per-family floors of every pool drawn up to and including `B`:
  - for the replay bank, `T_mine` 8 + replay 4 = **12**;
  - for the generalization bank, 8 + 4 + 10 = **22**.

  Ablation and interaction have no per-family floor, so they add nothing. Below this floor no plan could meet the minimum. At or above it, some plan meets it, so a shortfall at the draw would come from the plan's own allocation.
  - The rule is per family and **fails closed**. It ignores the fixed pool totals, so on a train set too small for every locked count at once (fewer than 174 games) it can reject a real shortfall.
  - That case is the owner's to decide, not the builder's to search. ALFWorld's train split is far larger: about 3,300 loader-accepted games, and its smallest family has about 300 games before filtering.
- **Q2 — How large the fallback is.** The pre-registered maximum must **equal** the games of that family still available when the bank is drawn. The bank takes every remaining game, neither fewer nor more. "Maximum" therefore means the most that availability allows, not any number below it.
- **Recording.** Each fallback is recorded in the split manifest's pool entry. The record holds `maximum`, `available_at_draw`, `accepted_in_train`, `locked_floor_through_bank` and `locked_minimum`, next to the plan's own `feasibility_fallback`.

**Reason.**
- Both rules remove plan discretion from a waiver of a locked minimum. The waiver is then a fact about the data, which anyone can recompute from the loader's per-family counts, not a choice hidden in the plan.
- Q1 is the strictest reading that stays independent of the plan.
- Q2 keeps a waived bank as large as the data allows. A smaller bank would make the stage-1 and stage-3 same-family and in-scope draws weaker than necessary.

**Research implication.**
- With ALFWorld's real train split, no family comes near the floors, so no fallback should be needed for EXP-0001. The rule matters only as a guard against a mis-specified or tampered plan.
- No novelty or causal claim.

**Experiment needed.** None. The split manifest built from the real data records the loader's per-family counts, which show whether any fallback was possible.

**Compatibility impact.**
- Plans without a fallback build byte-identical manifests.
- A plan whose fallback maximum differs from the games available now fails closed.
- No manifest has been built yet, because the build sandbox has no ALFWorld data.
- No change to the locked manifest text: this ADR interprets it.
