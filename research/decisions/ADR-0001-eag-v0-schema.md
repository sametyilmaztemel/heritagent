# ADR-0001 — Propose EAG v0 schema (two-layer, typed, lifecycle-state design)

Date: 2026-10-02 · Status: PROPOSED (awaiting cross-review per SPEC §32) · Artifact: `architecture/eag-v0-schema.md`

**Problem.** Milestone 0 needs a concrete data structure for the heritable agent specification before any runtime exists (SPEC §6), but the schema must not hard-code Milestone-0 assumptions that block population/evolution features later (SPEC §27 vs §11–13).

**Current design.** None — specification only (SPEC §6 YAML sketch).

**Proposed change.** Adopt EAG v0.1 as a two-layer design: (1) a small typed genome manifest (slots: cognition/memory/execution policies + skills[] + declarative regulation) referencing (2) content-addressed gene artifacts in a trait registry (types: policy, skill, regulatory), with mandatory provenance (`origin`, `source_trajectory`, `cig_record`). Somatic candidates reuse the gene format plus a validation envelope — somatic/germline becomes a lifecycle-state transition, not a format change. Full JSON Schema in the artifact.

**Reason.** (a) Diffable small manifests make the Lineage Graph derivable from genomes rather than maintained separately; (b) same-container somatic/germline removes format conversion as a noise source in H2 comparisons; (c) content-addressed artifacts make assimilation auditable (which CIG record promoted which artifact); (d) typed config-first genes avoid DGM-style arbitrary-code genomes while remaining expressive enough for Milestone 0.

**Research implication.** The schema operationalizes SPEC §4's biology table: gene = typed mutable trait, regulatory gene = `express_when` predicate, germline = promoted registry entries. It also encodes the anti-overclaim discipline: `backbone_tested` is advisory, fitness is never stored in the genome (D7 — no Lamarckian shortcut).

**Experiment needed.** None for the schema itself; it is exercised end-to-end by EXP-0001. Schema changes discovered during EXP-0001 require a new ADR.

**Compatibility impact.** ModelAdapter (SPEC §21) consumes compiled genomes; Trait Miner must emit the skill-gene payload format; CIG must emit `cig/0.1` records referenced by assimilated genes. Deferred-but-not-precluded: organization/topology slots, diploidy, population fields, code-typed genes.

**Cross-review asks:** the four open questions at the end of the schema doc (regulation predicates scope, per-seed CIG records, backbone registry semantics, skill frontmatter format).
