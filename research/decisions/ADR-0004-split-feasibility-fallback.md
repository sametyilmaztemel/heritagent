# ADR-0004 — When the EXP-0001 bank feasibility fallback applies, and how large it is

Date: 2026-10-13 · Status: **ACCEPTED by the owner** (Samet, 2026-10-13, in chat). Q1 was accepted as recommended ("accept the recommendations"). Q2 was chosen explicitly ("Evet, tamamı"), and so was its extension Q2b after the own review of #48 ("Kapat, veriye bağla"). **Codex cross-review requested (SPEC §32).** Artifacts: `evaluation/benchmarks/alfworld/splits.py` (`build_split_manifest`, `_locked_floor_through`, `_fallback_bank_size`), `tests/evaluation/test_alfworld_splits.py` · History: PRs #42, #43, #44, #48 and its follow-up.

**Problem.**
- The locked split rules give the family-indexed banks per-family minima: replay at least 4 per family, generalization at least 10. A "pre-registered family-specific maximum" may replace a minimum **"if availability prevents the minimum"** (manifest `environment.splits.T_gate.evaluation_banks.feasibility_fallback`; design §2; ADR-0002 R1).
- The locked text defines neither when availability "prevents" the minimum nor what the "maximum" must be.
- Both choices decide which games enter the banks, so they are pre-registration elements. Leaving them to code review left loopholes:
  - **Plan-made shortfalls.** A first implementation (#42) measured the games left at the bank draw. A plan could raise its own earlier counts (for example `replay_bank` 12 for a family with 35 games) and so waive the generalization minimum.
  - **Plan-sized banks.** A plan could declare a maximum below the games available. Even with the maximum tied to the games left at the draw, it could still shrink a waived bank by giving the family more than its floors in earlier pools. For example, a family with 21 games has room for a generalization bank of 9, but `T_mine` 10, ablation 4, interaction 2 and replay 4 would leave 1.

**Current design (before this ADR).**
- Since #43 the builder accepts a fallback only on a plan-independent per-family shortfall (Q1 below).
- Any maximum up to the available games is accepted, and `available_at_draw` is recorded.

**Decision (owner).**
- **Q1 — When the fallback applies.** A fallback for family `f` in bank `B` is accepted only if `f` has fewer loader-accepted train games than the sum of the locked per-family floors of every pool drawn up to and including `B`:
  - replay bank: `T_mine` 8 + replay 4 = **12**;
  - generalization bank: 8 + 4 + 10 = **22**.

  Ablation and interaction have no per-family floor, so they add nothing. Below this sum no plan could meet the minimum, while at or above it some plan meets it.
  - The rule is per family and **fails closed**. It ignores the fixed pool totals, so on a train set too small for every locked count at once (fewer than 174 games) it can reject a real shortfall. That case is the owner's to decide, not the builder's to search.
  - ALFWorld's train split is far larger. The prior analysis of the ALFWorld split table (`research/methodology/ALFWorld-baselines.md` on branch `claude/keen-bohr-ku33ye`) gives 3,553 train games, with a smallest family (`look_at_obj_in_light`) of 308. Both counts are before loader filtering. [corrected after independent verification 2026-10-15: the committed split manifest, built from the real data on 2026-10-14 (`manifest_sha256 2f74d188…ed9c`), gives 3,553 loader-accepted train games and accepted family counts 308–813; the smallest family has 308 against the floors 12 and 22, so no fallback was needed (ADR-0009 D5).]
  - SEEA-R1 reports a training set of 3,321 games; this loader has not verified that number. The split manifest records the loader's actual per-family counts.
- **Q2 / Q2b — How large the fallback bank is.**
  - A family with a fallback takes **only its locked floors from every earlier pool**: `T_mine` 8, ablation 0, interaction 0, and, before the generalization bank, replay 4 or its own replay fallback.
  - The pre-registered maximum must **equal** what that leaves:
    - replay bank: accepted − 8;
    - generalization bank: max(0, accepted − 12).
  - The bank therefore takes every game the family has left, and its size follows from the loader's count for that family alone.
  - Any other earlier count or maximum fails closed, before any draw. This decision covers only which plans build. When a plan has several defects, it does not fix which defect is reported first.
  - A family that the data leaves short of a bank's minimum (Q1) must have a fallback for that bank. When the plan passes validation and no fallback is pre-registered, the shortfall check names the fallback and its size. If the plan's own count for that bank is below the minimum, plan validation stops first and reports only that no fallback is pre-registered, since it has no data to size one.
  - A family with fewer games than the `T_mine` minimum (8) cannot be built at all, since `T_mine` has no fallback.
- **Recording.** Each fallback is recorded in the split manifest's pool entry, next to the plan's own `feasibility_fallback`. The entry holds:
  - `maximum`
  - `available_at_draw` (equal to `maximum` by construction)
  - `accepted_in_train`
  - `locked_floor_through_bank`
  - `locked_minimum`

**Reason.**
- Both the waiver (Q1) and the size of a waived bank (Q2 / Q2b) follow from the loader's per-family count and the locked floors. Anyone can recompute them from the manifest's `collections` summary, and the plan has no discretion over either.
- Q1 is the strictest reading that stays independent of the plan.
- Q2 / Q2b give a waived bank every game the data leaves it. A smaller bank would weaken the stage-1 same-family draws and the stage-3 in-scope draws more than necessary.
- Q2b alone constrains a family's ablation and interaction share, setting it to 0 even though those pools are stratified. It applies only to a family the data has already made too small for its floors.

**Research implication.**
- With ALFWorld's real train split no family comes near the floors, so no fallback should be needed for EXP-0001. The rules matter only as a guard against a mis-specified or tampered plan, and for any future benchmark with small families.
- No novelty or causal claim.

**Experiment needed.** None. The split manifest built from the real data records the loader's per-family counts, which show whether any fallback was possible.

**Compatibility impact.**
- Plans without a fallback build byte-identical manifests: no fallback code path runs for them. A known-answer test pins the fixture's `manifest_sha256`.
- A plan with a fallback must now satisfy Q2b, or it fails closed.
- No manifest has been built yet, because the build sandbox has no ALFWorld data. [corrected after independent verification 2026-10-15: the manifest now exists and needs no fallback. The fallback is a split-level rule only (ADR-0009 D5): the EXP-0001 stage allocators were written for the default banks. The replay allocator requires exactly 4 games per family; the generalization allocator fails only when a family's bank holds fewer games than a draw needs, so a short bank could pass the split and fail mid-run.]
- The locked manifest text is unchanged: this ADR interprets it.
