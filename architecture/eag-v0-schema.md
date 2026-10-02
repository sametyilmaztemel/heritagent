# EAG v0 — Executable Agent Genome, schema proposal

**Status: PROPOSAL — pending cross-review** (SPEC §31/§32: builder proposes, research role critiques; no implementation until reviewed). Author: builder-role session, 2026-10-02. Decision recorded in `research/decisions/ADR-0001-eag-v0-schema.md`.

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
  - `skill`: markdown skill doc with structured frontmatter (`name`, `purpose`, `when_to_apply`, `procedure`) — Voyager/SkillRL-compatible so baselines can consume it unchanged.
  - `regulatory`: a named predicate (v0: counter-based; see D5).
- **D2 — Gene slots.** v0 slots: `cognition.planner`, `cognition.verifier`, `cognition.reflection`, `memory.retrieval`, `memory.compression`, `execution.tool_selector`, `execution.retry_policy`, `execution.error_recovery`, plus `skills[]` (open list). Each slot is optional; absence = framework default (explicitly recorded, not implicit).
- **D3 — Provenance is mandatory** (`origin`: seed | mutation | recombination | assimilation; `source_trajectory`; `cig_record` for assimilated genes; `born_generation`). This is what makes the Lineage Graph (SPEC §13) derivable rather than separately maintained.
- **D4 — Somatic envelope.** Somatic candidates: same geneRef + `validation: {state: candidate|rejected|validated, gate_reports: [...]}`. The gate reports are CIG stage outputs (see §4).
- **D5 — Regulation v0**: `express_when` map from gene_id → predicate over runtime counters (`consecutive_failures`, `failing_tools`, `repeated_tool_error`, `steps_without_progress`, `task_type`). Ablation tooling MUST evaluate regulated genes under their conditions (SkillShapley lesson: unconditional ablation misvalues conditionally-expressed genes).
- **D6 — Versioning.** `schema_version` for the manifest format; `gene.version` integer, monotonically increased on artifact change; `genome_id` = `G-<content-hash-prefix>`; lineage edges store genome-id pairs + typed diffs (SPEC §13 JSON).
- **D7 — Fitness is NOT stored in the genome.** Fitness vectors live in the evaluation store keyed by (genome_id, env, seed). Genomes stay pure specification; no Lamarckian shortcut of baking scores into heritable state.

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
    "backbone_tested":   {"type": "array", "items": {"type": "string"},
                          "description": "advisory only; informational, not a constraint"},
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
      "required": ["gene_id", "version", "type", "origin"],
      "properties": {
        "gene_id":  {"type": "string", "pattern": "^[a-z0-9_]+(_v[0-9]+)?$"},
        "version":  {"type": "integer", "minimum": 1},
        "type":     {"enum": ["policy", "skill", "regulatory"]},
        "artifact": {"type": "string",
                     "description": "content-addressed URI into the trait registry, e.g. registry://traits/sha256:..."},
        "origin":   {"enum": ["seed", "mutation", "recombination", "assimilation"]},
        "provenance": {
          "type": "object",
          "properties": {
            "source_trajectory":   {"type": "string"},
            "cig_record":          {"type": "string", "description": "CIG record id; required iff origin=assimilation"},
            "born_generation":     {"type": "integer", "minimum": 0},
            "notes":               {"type": "string"}
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
                            "steps_without_progress", "task_type"]},
        "op":     {"enum": [">=", ">", "==", "<", "<="]},
        "value":  {"type": ["number", "string"]}
      },
      "additionalProperties": false
    }
  }
}
```

## 4. Companion schemas (same package, stubs)

**Somatic trait envelope** (`somatic/0.1`): `{"candidate": <geneRef>, "validation": {"state": "candidate|rejected|validated", "gate_reports": [<CIGRecordId>]}}`.

**CIG record** (`cig/0.1`): one per candidate evaluation — `{"cig_id", "candidate_gene_id", "genome_id", "stages": {"replay": {...measurements, threshold, pass}, "ablation": {...}, "generalization": {...}, "interaction_regression": {...}}, "verdict": "reject|promote", "thresholds_used": {...}, "artifacts": {...}}`. Written before thresholds are re-tuned (SPEC §33 pre-registration applies at the trait level too).

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
      "planner": {"gene_id": "planner_react_v1", "version": 1, "type": "policy", "origin": "seed"}
    },
    "execution": {
      "retry_policy": {"gene_id": "retry_backoff_v1", "version": 1, "type": "policy", "origin": "seed"}
    },
    "skills": [
      {"gene_id": "look_before_heat_v1", "version": 1, "type": "skill",
       "origin": "assimilation",
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

## 7. Open questions for cross-review (research role)

1. Should `regulation` predicates include task-type matching in v0, or is `consecutive_failures`-style counters enough for ALFWorld-class environments? (Concern: task_type predicates could let traits overfit the mining split.)
2. Is one CIG record per candidate sufficient, or do we need per-seed records (noise audit) as first-class?
3. `backbone_tested` as advisory array vs. strict registry of evaluated backbones — which avoids implied claims (SPEC §34)?
4. Skill payload: adopt SkillRL-style `{name, principle, when_to_apply}` frontmatter verbatim for baseline comparability, or extend with `procedure` steps (SkillShapley needs step-level structure if we reuse removal-based attribution)?
