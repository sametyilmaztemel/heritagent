# EAG v0 — Executable Agent Genome, schema proposal

**Status: REVISED after critic round 1 (2026-10-02) — re-review requested.** Originally proposed 2026-10-02 (builder role); revised same day per Research/Critic review on PR #1 / issue #3 — see §7 "Review resolution". Decision recorded in `research/decisions/ADR-0001-eag-v0-schema.md`.

## 1. Scope and goals

v0 exists to answer SPEC §27's question only: *when an agent acquires a reusable trait during a task, can a Causal Inheritance Gate reliably decide whether the trait deserves inheritance, and does that choice help descendants?* Therefore v0 covers: single agent, single lineage, two generations (Milestone 0), skill- and policy-type genes, minimal regulation. Everything else (population, niches, recombination, diploidy) is deferred but must not be precluded by the schema.

Design goals:

1. **Typed and versioned** — every gene has an ID, version, type, and provenance (SPEC §6).
2. **Genome = composition, genes = artifacts.** The genome references content-addressed executable artifacts in a trait registry; the genome itself is a small typed manifest. This keeps genomes diffable (lineage edges), keeps the germline auditable, and lets the somatic store reuse the exact same container format.
3. **Somatic/germline is a lifecycle state, not a format.** A candidate trait in the Somatic Trait Store is the same object type as a germline gene, plus a validation envelope. Assimilation (SPEC §10) becomes a metadata transition + registry promotion — no format conversion, no silent rewrites.
4. **Model-independent by construction, not by assumption** — genome contains no backbone-specific fields; the expression layer compiles it per ModelAdapter (SPEC §21). Whether it is *also* model-independent in effect is H4, an experiment.
5. **Regulation is declarative and cheap** — v0 predicates over runtime counters only (no learned regulators).

## 2. Core decisions

- **D1 — Two-layer design.** `Genome` (this schema) references `Gene` artifacts stored in a trait registry keyed by content hash. Gene payload formats per type:
  - `policy`: JSON config (e.g. planner params, retry budget, retrieval top-k) + optional prompt-template file reference.
  - `skill`: markdown skill doc with structured frontmatter that is a **SkillRL-compatible superset**: `name`, `principle`, `when_to_apply` (SkillRL field names, so baselines consume them unchanged), `applicability.task_families[]` — a frozen, **evaluation-only** structured scope field produced at extraction and used solely by gate sampling/audit (never a runtime regulation predicate or agent-visible oracle; the agent-facing scope remains the free-form `when_to_apply` text) — plus a `procedure[]` list whose entries carry **stable step IDs** (e.g. `S1`, `S2`). M0 gates and scores the skill as a whole; stable step IDs are reserved so future SkillShapley-style step attribution needs no re-extraction. A projection adapter strips `procedure[]` and `applicability` (mapping remaining fields 1:1) to yield a strict SkillRL skill for the baseline arm.
  - `regulatory`: a named predicate (v0: counter-based; see D5).
- **D2 — Gene slots.** v0 slots: `cognition.planner`, `cognition.verifier`, `cognition.reflection`, `memory.retrieval`, `memory.compression`, `execution.tool_selector`, `execution.retry_policy`, `execution.error_recovery`, plus `skills[]` (open list). Each slot is optional; absence = framework default (explicitly recorded, not implicit).
- **D3 — Provenance is mandatory** (`origin`: seed | mutation | recombination | assimilation; `source_trajectory`; `cig_record` for assimilated genes; `born_generation`). This is what makes the Lineage Graph (SPEC §13) derivable rather than separately maintained. **The JSON Schema enforces these invariants mechanically** (§3): `artifact` and `provenance` are required on every gene reference, `born_generation` is required in all provenance, and assimilation additionally requires `source_trajectory` + `cig_record`.
- **D4 — Somatic envelope.** Somatic candidates: same geneRef + `validation: {state: candidate|rejected|validated, gate_reports: [...]}`. Each gate report references the **aggregate CIG record** for the candidate; per-seed/per-run raw measurements live in that record's immutable children (see §4).
- **D5 — Regulation v0**: `express_when` map from gene_id → predicate over **agent-observable runtime counters only** (`consecutive_failures`, `failing_tools`, `repeated_tool_error`, `steps_without_progress`). Environment-label predicates (e.g. `task_type`) are excluded from the runtime regulation vocabulary: they would leak oracle task labels into gene expression. Trait specialization is expressed as applicability metadata in the trait itself (`when_to_apply` in skill payloads), which is evaluated evidence-side, not as a genotype-level environment predicate. Ablation tooling MUST evaluate regulated genes under their conditions (SkillShapley lesson: unconditional ablation misvalues conditionally-expressed genes).
- **D6 — Versioning.** `schema_version` for the manifest format; `gene.version` integer, monotonically increased on artifact change; `genome_id` = `G-<content-hash-prefix>`; lineage edges store genome-id pairs + typed diffs (SPEC §13 JSON).
- **D7 — Evaluation metadata is NOT stored in the genome.** Fitness vectors live in the evaluation store keyed by (genome_id, env, seed); backbone evidence lives in a separate **BackboneEvaluationRegistry** keyed by `(genome_id, model_hash, benchmark, seed)`. Genomes stay pure specification; no Lamarckian shortcut of baking scores into heritable state, and no implied "tested-on" claims inside the genotype (SPEC §34 anti-overclaim).

## 3. JSON Schema (draft 2020-12)

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://heritagent.dev/schemas/eag/0.1/genome.json",
  "title": "Executable Agent Genome v0.1",
  "type": "object",
  "required": ["genome_id", "schema_version", "generation", "lineage_id", "parent", "genes"],
  "properties": {
    "genome_id":         {"type": "string", "pattern": "^G-[a-z0-9]{8,}$"},
    "schema_version":    {"const": "0.1"},
    "generation":        {"type": "integer", "minimum": 0},
    "lineage_id":        {"type": "string"},
    "parent":            {"type": ["string", "null"],
                          "description": "parent genome_id; null for founders"},
    "genes": {
      "type": "object",
      "properties": {
        "cognition": {
          "type": "object",
          "properties": {
            "planner":    {"$ref": "#/$defs/geneRef"},
            "verifier":   {"$ref": "#/$defs/geneRef"},
            "reflection": {"$ref": "#/$defs/geneRef"}
          },
          "additionalProperties": false
        },
        "memory": {
          "type": "object",
          "properties": {
            "retrieval":   {"$ref": "#/$defs/geneRef"},
            "compression": {"$ref": "#/$defs/geneRef"}
          },
          "additionalProperties": false
        },
        "execution": {
          "type": "object",
          "properties": {
            "tool_selector":   {"$ref": "#/$defs/geneRef"},
            "retry_policy":    {"$ref": "#/$defs/geneRef"},
            "error_recovery":  {"$ref": "#/$defs/geneRef"}
          },
          "additionalProperties": false
        },
        "skills": {"type": "array", "items": {"$ref": "#/$defs/geneRef"}}
      },
      "additionalProperties": false
    },
    "regulation": {
      "type": "object",
      "description": "gene_id -> activation condition; genes absent from this map are constitutively expressed",
      "additionalProperties": {"$ref": "#/$defs/condition"}
    }
  },
  "additionalProperties": false,

  "$defs": {
    "geneRef": {
      "type": "object",
      "required": ["gene_id", "version", "type", "artifact", "origin", "provenance"],
      "properties": {
        "gene_id":  {"type": "string", "pattern": "^[a-z0-9_]+(_v[0-9]+)?$"},
        "version":  {"type": "integer", "minimum": 1},
        "type":     {"enum": ["policy", "skill", "regulatory"]},
        "artifact": {"type": "string",
                     "description": "content-addressed URI into the trait registry, e.g. registry://traits/sha256:..."},
        "origin":   {"enum": ["seed", "mutation", "recombination", "assimilation"]},
        "provenance": {
          "type": "object",
          "required": ["born_generation"],
          "properties": {
            "source_trajectory":   {"type": "string"},
            "cig_record":          {"type": "string", "description": "aggregate CIG record id; required iff origin=assimilation"},
            "born_generation":     {"type": "integer", "minimum": 0,
                                    "description": "required for all origins (lineage auditability)"},
            "notes":               {"type": "string"}
          }
        }
      },
      "if": {
        "properties": {"origin": {"const": "assimilation"}},
        "required": ["origin"]
      },
      "then": {
        "properties": {
          "provenance": {
            "required": ["born_generation", "source_trajectory", "cig_record"]
          }
        }
      }
    },
    "condition": {
      "type": "object",
      "properties": {
        "express_when": {
          "type": "object",
          "properties": {
            "all_of": {"type": "array", "items": {"$ref": "#/$defs/predicate"}, "minItems": 1},
            "any_of": {"type": "array", "items": {"$ref": "#/$defs/predicate"}, "minItems": 1}
          },
          "additionalProperties": false
        }
      },
      "additionalProperties": false
    },
    "predicate": {
      "type": "object",
      "required": ["metric", "op", "value"],
      "properties": {
        "metric": {"enum": ["consecutive_failures", "failing_tools", "repeated_tool_error",
                            "steps_without_progress"],
                   "description": "agent-observable runtime counters only; environment labels (task_type etc.) are excluded by design"},
        "op":     {"enum": [">=", ">", "==", "<", "<="]},
        "value":  {"type": "number"}
      },
      "additionalProperties": false
    }
  }
}
```

## 4. Companion schemas (same package, stubs)

**Schema-enforced invariants (draft 2020-12):** every gene reference must be content-addressed (`artifact` required) and carry provenance (`provenance` required, with `born_generation` mandatory for **all** origins — lineage auditability); `origin: "assimilation"` additionally requires `provenance.source_trajectory` and `provenance.cig_record` (via `if/then`). These turn ADR-0001's prose invariants into machine-checkable constraints; validators (issue #4) must reject violations.

**Somatic trait envelope** (`somatic/0.1`): `{"candidate": <geneRef>, "validation": {"state": "candidate|rejected|validated", "gate_reports": [<CIGRecordId>]}}`.

**CIG record** (`cig/0.1`) — **two-level evidence structure**: one **aggregate record** per candidate carrying the verdict, plus **immutable per-seed/per-run child records** with the raw stage measurements. The aggregate references its children (e.g. `CIG-0042` → `CIG-0042/S11`, `CIG-0042/S23`, `CIG-0042/S41`); children are append-only and never edited. This keeps the genome and somatic envelope free of evaluation detail while preserving full auditability and per-seed noise analysis.

```json
{
  "cig_id": "CIG-0042",
  "candidate_gene_id": "look_before_heat_v1",
  "genome_id": "G-0f1e2d3c",
  "child_records": ["CIG-0042/S11", "CIG-0042/S23", "CIG-0042/S41"],
  "stages": {
    "replay":                {"measurements_ref": "children", "threshold": "...", "pass": true},
    "ablation":              {"measurements_ref": "children", "threshold": "...", "pass": true},
    "generalization_in_scope":  {"measurements_ref": "children", "threshold": "...", "pass": true},
    "generalization_out_of_scope": {"measurements_ref": "children", "threshold": "...", "pass": true},
    "interaction_regression":{"measurements_ref": "children", "threshold": "...", "pass": true}
  },
  "verdict": "reject|promote",
  "thresholds_used": {"version": "exp-0001-v2", "values": {}},
  "artifacts": {}
}
```

Written before thresholds are re-tuned (SPEC §33 pre-registration applies at the trait level too).

## 5. Example instance (v0)

```json
{
  "genome_id": "G-a1b2c3d4",
  "schema_version": "0.1",
  "generation": 1,
  "lineage_id": "L-alpha",
  "parent": "G-0f1e2d3c",
  "genes": {
    "cognition": {
      "planner": {"gene_id": "planner_react_v1", "version": 1, "type": "policy", "origin": "seed",
                  "artifact": "registry://policies/planner_react_v1@1/sha256:9f2c81aa",
                  "provenance": {"born_generation": 0}}
    },
    "execution": {
      "retry_policy": {"gene_id": "retry_backoff_v1", "version": 1, "type": "policy", "origin": "seed",
                       "artifact": "registry://policies/retry_backoff_v1@1/sha256:41ab77d2",
                       "provenance": {"born_generation": 0}}
    },
    "skills": [
      {"gene_id": "look_before_heat_v1", "version": 1, "type": "skill",
       "origin": "assimilation",
       "artifact": "registry://skills/look_before_heat_v1@1/sha256:c7d03e5f",
       "provenance": {"source_trajectory": "T-00184", "cig_record": "CIG-0007", "born_generation": 1}}
    ]
  },
  "regulation": {
    "look_before_heat_v1": {"express_when": {"all_of": [{"metric": "consecutive_failures", "op": ">=", "value": 2}]}}
  }
}
```

## 6. Deliberate omissions (v0)

- No `organization` section (multi-agent topology) — v0 is single-agent; slot added in v1.
- No diploidy/dominance/recessive machinery (Genomebook territory; adds parameters without Milestone-0 value).
- No learned regulators, no nested genomes, no cross-genome references.
- No population fields — population state lives in the Evolution Engine store, keyed by genome_id.
- Gene payloads for `policy` are config+prompt refs only; v0 does not allow arbitrary code genes (DGM-style) — typed configs first, code genes reconsidered only if expressiveness blocks Milestone 0.

## 7. Review resolution (critic round 1, 2026-10-02)

All four cross-review questions from PR #1 / issue #3 are resolved as follows; decisions recorded in ADR-0001.

1. **Regulation predicates (task_type):** removed from the runtime regulation vocabulary. v0 regulation uses agent-observable counters only; specialization is trait applicability metadata (`when_to_apply`), evaluated evidence-side — not a genotype-level environment predicate (D5). No oracle-label leakage into gene expression.
2. **CIG evidence granularity:** two-level records adopted — aggregate candidate verdict + immutable per-seed/per-run children (`CIG-0042/S11`, ...) referenced by the aggregate (§4). Genome stays free of evaluation detail; per-seed noise analysis remains first-class.
3. **backbone_tested:** removed from the heritable genome manifest. Backbone evidence lives in a separate BackboneEvaluationRegistry keyed `(genome_id, model_hash, benchmark, seed)` (D7) — evaluation metadata, not genotype, and no implied claims inside the genome.
4. **Skill payload format:** SkillRL-compatible superset — `name`, `principle`, `when_to_apply` (SkillRL names) + stable `procedure[]` step IDs. M0 scores whole skills; step IDs reserve SkillShapley-style attribution without re-extraction; a projection adapter emits strict SkillRL skills for the baseline arm (D1).

### Round 2 (2026-10-02)

1. **Invariants enforced in the JSON Schema:** `geneRef.required` now includes `artifact` and `provenance`; `provenance.required` includes `born_generation` for all origins; an `if/then` clause requires `source_trajectory` + `cig_record` whenever `origin == "assimilation"`. The example instance was updated and validates against the tightened schema (§3, §4 note).
2. **Machine-readable applicability (coordinated with PR #2 round 2, definition lives in this artifact):** skill payloads gain `applicability.task_families[]` — produced at extraction, frozen before gate stage 1, **evaluation-only** (gate sampling/audit); never a runtime regulation predicate or agent-visible oracle. The projection adapter strips it along with `procedure[]` (D1).
