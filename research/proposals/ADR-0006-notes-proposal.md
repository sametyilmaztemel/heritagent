# Proposed notes for ADR-0006 — reading the final statistics

Date: 2026-10-16 · Status: **PROPOSED** (builder-2, for the owner's decision). Not part of ADR-0006. ADR-0009 D5 authorizes one note in ADR-0006, the scope of the H2 regression clause, and that note is in ADR-0006 D4. The notes below go beyond D5's text. Builder-2's own review of the ADR-0009 D1/D2/D5 PR found that they had been added to ADR-0006 under D5's label without that basis, so they were moved here. Only the owner's decision moves any of them into ADR-0006.

None of them changes a threshold, a rule or a computation. They state, before any EXP-0001 data, how the pre-registered rule's outcomes are to be read. The code already behaves as they describe; three of them are visible in the results record, as noted.

## N1 — The intervals are conditional on the three mining seeds (under D1)

Only tasks are resampled, so each interval is conditional on the three mining seeds [11, 23, 41]. Seed-to-seed variability enters only through the 2/3 rule.

The record states this in every contrast block (`uncertainty`, `evaluation/metrics/final_statistics.py`).

## N2 — The percentile index convention (under D2)

The percentile indices are lower = int(α/2 · N) = 250 and upper = int((1 − α/2) · N) = 9,750 of the N = 10,000 sorted resample means (0-based). So the lower bound has 250 resamples below it and the upper bound 249 above it.

- "Lower bound > 0" is therefore one rank less conservative than a nearest-rank 2.5th percentile.
- "Upper bound < 0" is one rank more conservative.

This is what `inheritance.bootstrap.paired_cluster_bootstrap` does, and ADR-0006 D2 takes its order statistics. The note would close the convention in writing before any data.

## N3 — No multiplicity correction (under D2)

H1 and H2 are each tested at 95 %, as the locked design states them. There is no multiplicity correction across the two hypotheses or across H2's three clauses. A reader should weigh them as two pre-registered rules, not as a family-wise error rate.

## N4 — The regression clause is in sample (under D4)

ADR-0006 D4 now says the H2 regression clause is measured on T_reg, the suite that stage 4A already used to decide which traits arm C keeps. The proposed addition says what follows from that:

- the clause checks, in sample, the regression that the gate controls;
- it is not an out-of-sample estimate;
- it favours C over B by construction.

The record states this as the clause's `scope`.

## N5 — Detectability under the locked rule (Research implication)

Status M. Computed with the pinned implementation before any data.

- **The regression clause at n = 40.** Take C failing k fewer T_reg tasks than B, on the same tasks in every seed:
  - k ≥ 4 (10 pp) is needed to exclude 0;
  - k = 3 (7.5 pp) gives an upper bound of exactly 0, so the clause fails.

  On disjoint tasks per seed, k ≥ 2 (5 pp) passes.
- **The expected reduction.** The pre-registered expectation is 3–8 pp (1.2–3.2 tasks). In the same-tasks pattern, most of that range is below the floor.
- **The success clause at n = 134.** Take a net +7.46 pp made of 20 wins and 10 losses per seed: it fails (lower bound −0.0075). 15 wins and 5 losses pass. The expected central value is +7 pp.
- **What follows.** An outcome that matches the expectations can fail H2. This is conservative, not a bias, and no threshold changes.

The proposal reads a null on the regression clause as "below detectability at n = 40", not as evidence against the mechanism. That reading is an interpretation of a future result, which is why it needs the owner's decision.

## Options for the owner

- **Accept all.** Builder-2 adds N1–N5 to ADR-0006 as notes under the owner's decision of that date.
- **Accept some.** Builder-2 adds only the chosen ones.
- **Reject.** They stay here as a record. The code's record labels (`uncertainty`, `scope`) are descriptions and stay either way, unless the owner says otherwise.
