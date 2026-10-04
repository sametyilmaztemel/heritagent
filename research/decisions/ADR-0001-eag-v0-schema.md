# ADR-0001 — Propose EAG v0 schema (two-layer, typed, lifecycle-state design)

Date: 2026-10-02 · Status: REVISED — critic round 1 applied, re-review requested (SPEC §32) · Artifact: `architecture/eag-v0-schema.md`

**Problem.** Milestone 0 needs a concrete data structure for the heritable agent specification before any runtime exists (SPEC §6), but the schema must not hard-code Milestone-0 assumptions that block population/evolution features later (SPEC §27 vs §11–13).

**Current design.** None — specification only (SPEC §6 YAML sketch).

**Proposed change.** Adopt EAG v0.1 as a two-layer design: (1) a small typed genome manifest (slots: cognition/memory/execution policies + skills[] + declarative regulation) referencing (2) content-addressed gene artifacts in a trait registry (types: policy, skill, regulatory), with mandatory provenance (`origin`, `source_trajectory`, `cig_record`). Somatic candidates reuse the gene format plus a validation envelope — somatic/germline becomes a lifecycle-state transition, not a format change. Full JSON Schema in the artifact.

**Reason.** (a) Diffable small manifests make the Lineage Graph derivable from genomes rather than maintained separately; (b) same-container somatic/germline removes format conversion as a noise source in H2 comparisons; (c) content-addressed artifacts make assimilation auditable (which CIG record promoted which artifact); (d) typed config-first genes avoid DGM-style arbitrary-code genomes while remaining expressive enough for Milestone 0.

**Research implication.** The schema operationalizes SPEC §4's biology table: gene = typed mutable trait, regulatory gene = `express_when` predicate, germline = promoted registry entries. It also encodes the anti-overclaim discipline: evaluation metadata (fitness, backbone evidence) lives outside the genome (D7 — no Lamarckian shortcut, no implied claims in the genotype).

**Experiment needed.** None for the schema itself; it is exercised end-to-end by EXP-0001. Schema changes discovered during EXP-0001 require a new ADR.

**Compatibility impact.** ModelAdapter (SPEC §21) consumes compiled genomes; Trait Miner must emit the skill-gene payload format (SkillRL-compatible superset with `procedure[]` step IDs); CIG must emit two-level `cig/0.1` records (aggregate + immutable per-seed children) referenced by assimilated genes; backbone evidence goes to the BackboneEvaluationRegistry, not the genome. Deferred-but-not-precluded: organization/topology slots, diploidy, population fields, code-typed genes.

## Review resolution — critic round 1 (2026-10-02, PR #1 / issue #3)

All four requested changes applied; schema artifact updated in the same revision:

| # | Critic requirement | Resolution |
|---|---|---|
| R1 | Remove `task_type` from v0 runtime regulation predicates (oracle-label leakage risk) | Removed from predicate metric enum; regulation vocabulary is agent-observable counters only; specialization moved to trait-level `when_to_apply` applicability metadata (D5) |
| R2 | Two-level CIG evidence: aggregate verdict + immutable per-seed child records | Adopted in `cig/0.1`: `CIG-0042` → `CIG-0042/S11`, `/S23`, `/S41`; children append-only; aggregate references children (§4 of the artifact) |
| R3 | Remove `backbone_tested` from genome manifest; separate BackboneEvaluationRegistry keyed `(genome_id, model_hash, benchmark, seed)` | Property removed from the JSON Schema; registry specified in D7 |
| R4 | SkillRL-compatible superset payload with stable `procedure[]` step IDs + projection adapter | Adopted in D1: `name`, `principle`, `when_to_apply` (SkillRL names) + stable step IDs; M0 scores whole skills; projection adapter emits strict SkillRL skills for the baseline arm |

## Review resolution — critic round 2 (2026-10-02, PR #1 / issue #3)

| # | Critic requirement | Resolution |
|---|---|---|
| R1 | Enforce design invariants in JSON Schema: `artifact` + `provenance` required; assimilation requires source trajectory + CIG record + born generation | `geneRef.required` = [gene_id, version, type, **artifact**, origin, **provenance**]; `provenance.required` = [**born_generation**] for all origins (lineage auditability); `if origin == "assimilation"` → require `source_trajectory` + `cig_record` (draft 2020-12 `if/then`). Example instance updated and validates against the tightened schema. Implementation issue #4 must reject violations via validator tests |
| R2 | Frozen machine-readable applicability for gate sampling, oracle-free at runtime (coordinated with PR #2 round 2) | `applicability.task_families[]` added to the skill payload frontmatter (D1): produced at extraction, frozen before gate stage 1, **evaluation-only** (gate sampling/audit); never a runtime regulation predicate or agent-visible oracle; agent-facing scope remains free-form `when_to_apply`; stripped by the SkillRL projection adapter |

## Review resolution — final round (2026-10-02, PR #1)

| # | Critic requirement | Resolution |
|---|---|---|
| R1 | Enforce real content-addressed artifact URI grammar + `sha256:<64-hex>` in JSON Schema; real 64-hex examples | D8 added: canonical v0 grammar `registry://<kind>/<gene_id>@<version>/sha256:<64-hex>`, `kind ∈ {policies, skills, regulators}`; `artifact.pattern` enforces the full grammar including the 64-hex digest (portable pattern, no custom format checker). Example instance uses genuine 64-hex SHA-256 digests. The "artifact-addressed" weakening option was rejected — content addressing stands. Validator tests to additionally check name↔gene_id and digest↔artifact equality (noted for issue #4) |

## Review resolution — implementation review (PR #21, 2026-10-02)

Code-review blocking fixes applied to the implementation and schema artifacts (schema doc updated in the same commit to stay in sync):

| # | Blocking group | Resolution |
|---|---|---|
| 1 | Typed slot semantics | Genome schema: cognition/memory/execution slots → `policyRef` (`type` const `policy`), `skills[]` → `skillRef` (const `skill`); negative tests for wrong-type-in-slot; `project_for_runtime()` runs `load_genome` (incl. registry digest/binding checks) at its own boundary |
| 2 | Gene identity/version immutability | Registry keeps a `(kind, gene_id, version) -> digest` binding index; rebinding an identity to different content → `RegistryIntegrityError` (D6: version increment required); hand-crafted URIs caught by `verify_binding`; `diff_genomes()` includes artifact URI in the inheritance test — same id/version with a different artifact is a `rebinds` mutation, never inheritance |
| 3 | CIG evidence integrity | `child_records` `uniqueItems`; `add_child` rejects children not declared in the aggregate; `verify()` flags orphan/undeclared children and child-id/seed incoherence (`/S<seed>` digits must match the `seed` field, enforced at insert and in verify) |

Non-blocking follow-ups recorded: deeper payload-shape validation with #7/#9; canonical `genome_id` hashing tracked in issue #22 (blocks #13, not #4).

## Final hardening (PR #21 re-review, 2026-10-02)

| # | Requirement | Resolution |
|---|---|---|
| 1 | Record stores must not retain/return externally mutable shared dict references | `CigRecordStore.add_aggregate/add_child` and `SomaticStore.add` deep-copy on insertion; `aggregates` / `children` / `all()` return deep copies. Tests: input mutation after add leaves stored state unchanged; accessor mutation leaves stored state unchanged |
| 2 | Registry-attached validation must reject unbound identity/version URIs | `verify_binding()` rejects an unbound `(kind, gene_id, version)` even when its digest exists under another identity; bound-digest mismatch rejection unchanged; content sharing valid only with explicit `put()` bindings per identity. Tests: forged identity B → A's digest rejected as unbound (registry-level, loader-level, `check_ref`) |

## Amendment A1 — somatic-only `origin="acquired"` (2026-10-04, issue #9)

The Trait Miner (#9) emits lifetime-mined skills that are neither evolutionary
mutations nor already-assimilated genes. Labelling them `mutation` would be
wrong and would blur the somatic/germline lifecycle.

**Decision.**
- `somatic/0.1` geneRef gains `origin: "acquired"`;
- `origin="acquired"` requires `provenance.source_trajectory` + `born_generation`
  (no `cig_record` — that belongs to assimilation only);
- an acquired trait may remain somatic through `candidate|rejected|validated`;
- later VTA/assimilation creates the germline ref with `origin="assimilation"`;
- the germline `genome/0.1` schema still rejects `acquired`.

**Consequences.** Mined skills are explicitly marked as unvalidated lifetime
acquisitions end-to-end; a later somatic→germline promotion is visible in
provenance as `acquired → assimilation` with the CIG record attached.
