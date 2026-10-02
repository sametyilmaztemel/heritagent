# Registered hypotheses (H1–H6)

From SPEC §15, with operationalizations and the experiment each maps to. **Status: all untested.** Definitions are locked at design time (pre-registration style, SPEC §33); any post-hoc change requires a new ADR entry and must be disclosed in the paper.

| ID | Statement (from SPEC §15) | Operationalization (measurable form) | Primary experiment |
|---|---|---|---|
| H1 — Assimilation | Validated lifetime-acquired capabilities transferred into the functional genome increase descendant zero-shot performance | F(G1 assimilated) − F(G1 without) > 0 on T_eval, 95% CI excluding 0 | EXP-0001 arms C vs A |
| H2 — Causal Gate | Causal validation before inheritance beats automatic inheritance of every successful skill on generalization and regression | Arm C − arm B ≥ +5pp on T_eval AND regression rate(C) < regression rate(B) | EXP-0001 arms C vs B |
| H3 — Learning Cost Reduction | Assimilated traits reduce the action/token/evaluation budget descendants need to reacquire the same capability | Descendant adaptation cost (tokens, actions, episodes, retries to reach parity on T_gate) lower for C vs A/B | EXP-0002+ |
| H4 — Cross-Backbone Heritability | Part of the functional genome's advantage survives transfer to a different foundation model | Normalized gain of (G_backbone-A + genome) vs (G_backbone-A alone) > 0; report how much of original gain is retained | EXP-0003 |
| H5 — Stability | Controlled assimilation produces less genome bloat and fewer catastrophic behavioural regressions than unrestricted accumulation | Genome size (genes, tokens) C < B; regression rate C < B; count of "catastrophic" (>10pp) per-task drops C < B | EXP-0001 (bloat/regression); population runs later |
| H6 — Generational Improvement | Population fitness systematically exceeds the initial population over a generation range | Monotone-ish trend of population-level multi-objective fitness over ≥ N generations with CIs; not explainable by selection on noise (null: permutation of generation labels) | EXP-0004+ (after Milestone 0) |

## Discipline notes

- H2 is the load-bearing hypothesis (SPEC §17: the paper's most important comparison is unrestricted inheritance vs CIG-gated inheritance). If H2 fails, the project's core claim fails — Milestone 0 exists to test it cheaply before anything else is built.
- "Causal" wording (SPEC §34) is earned only where controlled interventions (replay/ablation) were actually run; elsewhere say "contribution estimate".
- Metrics used above are defined in SPEC §18 (Assimilation Gain, Inheritance Precision, Genome Efficiency, Descendant Adaptation Cost, Cross-Backbone Heritability, Regression Rate). EXP manifests must state which are computed and from which splits.
