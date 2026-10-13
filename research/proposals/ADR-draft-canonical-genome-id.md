# ADR-00XX: Canonical `genome_id` hashing (genotype identity), resolving issue #22

Date: 2026-10-12. **Status: PROPOSED.** This is a draft by Claude, a third agent working as an independent researcher. It has not been reviewed. It needs Research/Critic cross-review (SPEC §32) and an owner decision. It is not an accepted decision, and nothing in it is implemented.
Numbering: left as `ADR-00XX` on purpose, so that it does not collide with the Builder's ADR numbering. The owner or Builder assigns the number on adoption.
Scope: issue #22, which blocks #13 (EXP-0001 execution) but not #4. It amends ADR-0001 D6 and touches D3/D8 and the lineage-edge contract (SPEC §13, issue #16).
Evidence base: repo `main` @ `6014bc7` (PR #28 / issue #12 merged), read-only. The draft was first written against `0ce0a36` plus a local PR #28 ref; every code reference below was re-checked against `6014bc7` in review, and line numbers are as of that commit. Every number in §6 was computed in a scratchpad prototype, not in repo code, and was independently recomputed in review. Unverified items are marked **[unverified]**.

Format follows SPEC §31: Problem, Current design, Proposed change, Reason, Research implication, Experiment needed, Compatibility impact.

---

## 1. Problem

ADR-0001 D6 says `genome_id = G-<content-hash-prefix>`, but it does not define:

- the hashed content,
- the serialization,
- the algorithm,
- the prefix length,
- how the id is validated.

Today `genome_id` is a free string checked only by `^G-[a-z0-9]{8,}$` (`genome/schema/eag-0.1.genome.schema.json`, and the same pattern in `eag-0.1.cig.schema.json`). Nothing computes it or checks it. Before #13 creates G1 genomes for arms A, B and C, and before M1 population evolution (#16 lineage graph, evaluation store keyed by `(genome_id, env, seed)`, BackboneEvaluationRegistry keyed by `(genome_id, model_hash, benchmark, seed)` per ADR-0001 D7), the project needs the following:

1. The same heritable content always yields the same id.
2. Any heritable change yields a new id.
3. A claimed id that does not match its content is rejected.

Issue #22 says explicitly: "Do not silently invent the rule in code; record the exact identity semantics in an ADR first."

## 2. Current design (what exists in-repo, verified by reading the code)

| Element | Location | Behaviour today |
|---|---|---|
| `genome_id` field | `genome/schema/eag-0.1.genome.schema.json` | Required. Pattern `^G-[a-z0-9]{8,}$`. Never computed or verified. `parent` is `["string","null"]` with **no pattern** ("parent genome_id; null for founders"); `lineage_id`, `provenance.source_trajectory` and `provenance.cig_record` are also unconstrained strings. |
| G0 seed id | `runtime/seed_g0.py` | Hard-coded constant `G0_GENOME_ID = "G-0f1e2d3c"`, a placeholder that is not a hash. |
| Artifact content addressing | `genome/validation/registry.py::canonical_bytes` | dict → `json.dumps(sort_keys=True, separators=(",",":"), ensure_ascii=False)`. str → UTF-8. bytes → verbatim. SHA-256, full 64 hex in the URI `registry://<kind>/<gene_id>@<version>/sha256:<64-hex>` (D8). **Missing `allow_nan=False`** (see §9 O1). |
| Strict canonical JSON (reusable) | `trajectory/storage/canonical.py::canonical_json` | `json.dumps(sort_keys=True, separators=(",",":"), ensure_ascii=False, allow_nan=False)`. The trajectory digest uses a *domain-tagged* object `{"kind": "heritagent-trajectory", "schema_version": "0.1", ...}`. |
| Precedent for reuse | `somatic/journal.py` | Imports `canonical_json`/`sha256_hex` from `trajectory.storage.canonical`. Record digest over a locked, domain-documented object that excludes its own digest. |
| Lineage edge | `genome/validation/diff.py::diff_genomes` | Raises `ValueError` if `parent.genome_id == child.genome_id`. Inheritance is decided on `(gene_id, version, artifact)`; regulation is compared with raw `!=`. |
| Skill order at runtime | `runtime/expression/__init__.py::compile_runtime_config`, `runtime/skills.py::skill_block` | Skills are compiled **in `genes.skills[]` list order** and rendered into the planner prompt in that order. List order therefore changes the phenotype (the LLM prompt). |
| CIG overlay | `inheritance/overlay.py:107/116`; `inheritance/interaction.py:272/276`; `inheritance/gate.py:741` | Both `with_trait` and `without_trait` `RuntimeConfig`s carry `base_config.genome_id`; the stage-4 pair configs (`both`/`neither`) do the same, and the CIG aggregate record's `genome_id` is the base config's id. `runtime/loop.py:152` logs `run_started genome_id=config.genome_id`, so with-trait and without-trait episodes are logged under the same genome id. The overlay appends the target skill **last**; the stage-4 `both` config appends the two pair genes after the base skills in `sorted(gene_id)` order. |

## 3. Proposed change: summary

1. **Two identities, not one.** `genome_id` becomes a **genotype identity**: a hash over the heritable, phenotype-determining content only. Lineage position and per-gene provenance are excluded. A second computed value, `manifest_sha256`, covers the entire document except `genome_id`. It is used for document integrity and for lineage-store records. It is **not** stored in the genome, so the schema does not grow.
2. **Canonical JSON reuses** `trajectory.storage.canonical.canonical_json`, following the `somatic/journal.py` precedent. On top of it:
   - a domain tag,
   - a numeric normalization rule,
   - a set-ordering rule for `skills[]` and predicate lists,
   - an ASCII-only assertion.

   RFC 8785 (JCS) is considered and not adopted (§5.4).
3. **Format:** `genome_id = "G-" + sha256(canonical_json(domain).encode("utf-8"))[:16]` (i.e. `trajectory.storage.canonical.canonical_bytes`, **not** the different `genome.validation.registry.canonical_bytes`, which lacks `allow_nan=False` and passes `str`/`bytes` through verbatim), which is 16 lowercase hex characters (64 bits). The full 64-hex digest is kept wherever ids are stored (genome store, lineage store, run manifests), and inserts fail closed on prefix collisions.
4. **API:** a new module `genome/identity.py` with `genome_identity_domain()`, `compute_genome_id()`, `validate_genome_id()`, `genome_digest()`, `manifest_sha256()` and `with_genome_id()`. `load_genome()` calls `validate_genome_id()` unconditionally (fail closed).
5. **Expression parity:** `compile_runtime_config()` renders skills in the same canonical order used for the id. Equal ids therefore imply identical prompts. This changes runtime behaviour and must land before any EXP-0001 run (§8).
6. **Lineage edges:** `diff_genomes()` replaces the "ids must differ" guard with an **id/diff consistency** check, so that a clonal edge (arm A: G1 has the same genotype as G0) is representable.

## 4. Canonical serialization domain (which fields participate)

### 4.1 Domain object (identity rule "1")

```text
genome_digest = sha256( canonical_json( {
  "kind":           "heritagent-genome-id",     # domain separation from trajectory/journal digests
  "id_rule":        "1",                        # bump => all ids change; requires a new ADR
  "schema_version": <genome.schema_version>,    # "0.1"
  "slots": { "<section>.<slot>": GENE_CORE, ... },   # flattened; absent/empty sections contribute nothing
  "skills": [ GENE_CORE, ... ],                 # sorted by (gene_id, version); see 5.2; key ALWAYS present ([] when absent/empty)
  "regulation": { "<gene_id>": CANON_CONDITION, ... } # key ALWAYS present; absent == {}
} ).encode("utf-8") )

GENE_CORE       = {"gene_id", "version", "type", "artifact"}  # exactly these four keys
CANON_CONDITION = {"express_when": {"all_of": [PRED...], "any_of": [PRED...]}}  # empty lists omitted; PRED lists sorted (5.2)
PRED            = {"metric", "op", "value"}   # value numerically normalized (5.3)
```

### 4.2 Participation table

| Field | Participates? | Justification |
|---|---|---|
| `genome_id` | **No, excluded** | It cannot hash itself. `compute_genome_id()` must not even read it (invariant I2). |
| `schema_version` | Yes | A manifest-format change is a different genotype encoding. It keeps ids from colliding across schema versions. |
| Slot path (`cognition.planner`, ...) | Yes | The same policy in a different slot is a different phenotype. |
| `gene_id`, `version`, `type`, `artifact` | Yes | `artifact` carries the 64-hex content digest (D8), so payload changes propagate. `gene_id`/`version`/`type` are redundant with the URI after loader checks, but are kept explicitly so the rule does not depend on URI parsing. |
| `skills[]` membership | Yes, as a **set** | See 5.2. Duplicate `gene_id`s are already rejected at compile time and must also be rejected by `load_genome` before hashing (new check). |
| `regulation` | Yes (full canonical condition) | Changes when a gene expresses, so it is heritable phenotype (D5). |
| `origin` (per gene) | **No** | Describes *how* a gene arrived (seed / mutation / assimilation), not *what* it does. It is excluded so that identical content reached by different routes collides (see 4.3). |
| `provenance.*` (`born_generation`, `source_trajectory`, `cig_record`, `notes`) | **No** | Lineage and audit metadata. `born_generation` differs by lineage depth for the same gene. `notes`, `source_trajectory` and `cig_record` are unconstrained strings (non-ASCII-capable), as are the top-level `lineage_id` and `parent`. It stays integrity-protected by `manifest_sha256`. |
| `generation` | **No** | Lineage position. See 4.3. |
| `parent` | **No** | Lineage position. See 4.3. |
| `lineage_id` | **No** | Lineage label. |

### 4.3 Decision on lineage metadata (`parent`, `generation`, `lineage_id`, provenance)

**Proposed: lineage metadata does NOT participate in `genome_id`.** Identical heritable content reached by different lineages **collides on purpose**: same genotype, same id.

Reasons:

- **Issue #22's own invariant.** "same heritable content must reproduce the same id" cannot hold if `parent` or `generation` is in the domain. Every child would have a fresh id even with zero heritable change.
- **SPEC §13 already separates agent nodes from genotypes.** Lineage-graph nodes are agents (`G0:A0`, `G1:A3`; edges `"parent": "A9" → "child": "A21"`), not genome hashes. Lineage position belongs to the node, not to the genotype id.
- **Convergent evolution must be detectable in M1.** With content-only ids, "two lineages independently assimilated the same skill set" becomes an id equality query. With lineage-inclusive ids it is invisible without re-deriving content.
- **The evaluation registries are keyed by `genome_id` (D7).** Phenotype evidence (fitness, backbone transfer) belongs to the genotype. Re-measuring an identical genotype reached by another lineage should *add samples to the same key*, not create a new, apparently independent genotype. This keeps genotype-level evidence counts honest (SPEC §34). Records must still carry run/arm/lineage context (see the hazards below). **Caveat (review):** under the locked EXP-0001 `determinism_policy` (temp 0, `no_pseudo_replication: true`: "duplicate deterministic runs are never independent evidence"), re-running an identical genotype on the same task instance is *not* a new sample. Pooling under one key is only valid across distinct task instances or a pre-registered stochastic source, and the store must deduplicate on `(genome_id, task instance, condition)` rather than count repeats.
- **Population dedup and caching (M1).** The population manager can recognise re-generated duplicates instead of spending evaluation budget on them.

What the lineage graph needs instead (input to #16, not decided here):

- **Node key:** an agent/node id (SPEC §13 `A<n>`) or the tuple `(lineage_id, generation, genome_id)`, plus `manifest_sha256` of the exact document.
- **Edge key:** `(parent_node, child_node)`, with `parent_genome_id` and `child_genome_id` stored as attributes.
- Ancestry tamper-evidence, if wanted, comes from a hash-chained lineage store (as in `somatic/journal.py`), not from mixing ancestry into the genotype id.

**Rejected alternative L1: lineage-inclusive id**, a Merkle-style id where the hash includes `parent` and `generation`.

- Pros: every node has a unique id. The `diff_genomes` guard stays valid. The id commits to ancestry.
- Cons: it violates the issue's "same content → same id" requirement across lineages. It hides convergence. It fragments genotype-level evidence across ids. Arm A's G1 would get a new id despite zero heritable change, which overstates that a "new genome" was evaluated.
- It is listed for the critic: the right choice depends on whether the evaluation store wants genotype-keyed or individual-keyed evidence. This draft argues genotype-keyed, plus a separate node key.

**Hazards introduced by the proposal (must be addressed, not ignored):**

- **H-a. Arm A's G1 equals G0's id.** Arm C with zero promoted candidates also equals G0's id. Arm B equals arm C's id if all K=10 candidates pass. Consequence: EXP-0001 bookkeeping **must not** use `genome_id` as the arm key. Evaluation records must be keyed by at least `(arm, mining_seed, genome_id, env, seed)`. This is a requirement on #13 and may touch the EXP-0001 manifest's result schema. If the manifest's locked fields are affected, that **requires ADR / pre-registration amendment review**. **[unverified]**: I did not find a locked result-key schema in `experiments/manifests/EXP-0001-minimal-cig.yaml`.
- **H-b. `diff_genomes()` currently raises when ids are equal** (`diff.py:28`). It must be replaced by the consistency check in §7 (I8), which also makes clonal edges (arm A) expressible.
- **H-c. Provenance is no longer committed by the id.** The CIG evidence trail (`cig_record`) is protected by `manifest_sha256`, which every run manifest and lineage record must store next to `genome_id`.
- **H-d (added in review). The genome-level `parent` pointer becomes ambiguous.** The schema defines `parent` as a *genome_id* ("parent genome_id; null for founders"). With genotype-only ids: a clonal child (arm A's G1, if it is a new document with `generation: 1`) has `parent == genome_id` (a self-pointer); a reversion mutation (G0 → G1 → G2 whose content equals G0) makes the id-level ancestry cyclic; and convergent genotypes merge distinct lineage nodes into one id with several `parent` values. ADR-0001 D3 says provenance makes the lineage graph "derivable rather than separately maintained", issue #16 says "derivable from genome provenance (ADR-0001 D3) but stored first-class", and ADR-0002 says EXP-0001 has "no lineage graph beyond parent pointers". Under this proposal the lineage graph is **not** derivable from genome documents alone, because the node key (§4.3 above) is not in the genome. The critic must decide among: (i) accept this and document `parent` as "parent genotype id" (lineage derivable only from the #16 store); (ii) add a node-level parent reference (e.g. the parent's `manifest_sha256`) to the schema, which is a schema change; (iii) reject genotype-only ids (L1). This draft does not decide it. It is part of Q0.

## 5. Canonical JSON rules

### 5.1 Base canonicalizer: reuse `trajectory.storage.canonical.canonical_json`

- UTF-8 output.
- `sort_keys=True` (object keys ordered by Python `str` ordering, i.e. code-point order).
- Compact separators `(",", ":")`.
- `ensure_ascii=False`.
- `allow_nan=False` (NaN/Infinity rejected).

This is the canonicalizer already used, as a locked rule, for trajectory digests and the somatic journal chain. A third convention is not introduced.

Layering note: `genome/` importing from `trajectory/` is a dependency smell, but it has precedent (`somatic/journal.py`). A follow-up may move `canonical_json`/`sha256_hex` into a neutral module, for example `heritagent_common/canonical.py`, re-exported from both places with **byte-identical output** (a golden-vector test guards this). This draft does not require the move.

### 5.2 Ordering rules (application-level; applied before `canonical_json`)

- **Objects:** key order is irrelevant by construction (`sort_keys`).
- **`skills[]` is a set.**
  - Canonical order: ascending `(gene_id, version)`. `gene_id` is ASCII under the schema pattern, so Python ordering is unambiguous.
  - Rationale: SPEC §6 / ADR-0001 D2 treat `skills[]` as an open list of genes with no positional semantics, and duplicate `gene_id`s are already rejected.
  - **Parity requirement:** the runtime currently renders skills in list order (§2), so permutation-equal genomes would have *equal ids but different prompts*. LLMs are position-sensitive, so this would break "same id ⇒ same phenotype".
  - This ADR therefore also requires `compile_runtime_config()` to emit `RuntimeConfig.skills` in the same canonical `(gene_id, version)` order, ignoring document order.
  - Rejected alternative S1: an order-sensitive id, where `skills[]` is hashed as an ordered list and builders are required to emit a canonical order. It is simpler for the id, but it leaves "ordering independence" (an issue #22 test item) false and makes prompt order an undocumented heritable trait.
  - Consequence for EXP-0001: the arm B/C prompt order becomes gene_id-lexical rather than miner-rank order (`ranking: [miner-estimated-generality, discovery-order]` in the manifest governs **selection**, which this does not change). Because this changes what the model sees, it must land before any EXP-0001 run, and it is flagged **requires ADR** (this one) **plus critic sign-off that no pre-registered element fixes prompt order**. **[unverified]**: I found no such locked element in the manifest.
- **Predicate lists** (`express_when.all_of`, `express_when.any_of`) are commutative in `runtime/regulation.py::Condition.evaluate`, which is pure with no short-circuit side effects. They are sorted by the `canonical_json` string of each normalized predicate.
  - **Duplicate predicates are rejected** by `load_genome`, not deduplicated silently. This fails closed and avoids silent rewrites, consistent with ADR-0001 design goal 3.
- **Empty forms:**
  - Absent sections, `{}` sections, absent `skills`, `[]` skills, absent `regulation` and `{}` regulation all map to the same canonical form, because they are semantically identical (D2: absence = framework default).
  - A regulation entry whose condition is empty (`{}` or `{"express_when": {}}`) is semantically constitutive, equal to having no entry. **Proposed: reject it in `load_genome`** (fail closed) rather than normalize it away. **Open question Q2** for the critic.

### 5.3 Number rules

- Integers (`version`) serialize as JSON integers. `bool` is rejected wherever a number is expected (in Python, `bool` is a subclass of `int`). JSON Schema `integer`/`number` already rejects booleans in python-jsonschema, so this is defense in depth.
- **Integral floats in `version` (verified in review):** python-jsonschema's `integer` type **accepts** `1.0`, and `load_genome` compares `int(uri_version) != ref["version"]`, which is `False` for `1 == 1.0`. A genome with `"version": 1.0` therefore passes `load_genome` and `compile_runtime_config` today, but `canonical_json` would write `1.0`, so the same heritable content would get a different id. The identity rule must either reject `type(version) is not int` in `load_genome` (fail closed, proposed) or normalize it like predicate values.
- **Predicate `value` (schema `type: number`):**
  - `Predicate.evaluate` compares via `float()`, so `2` and `2.0` are the same phenotype, but `canonical_json` writes `2` and `2.0` differently. (Verified in the scratchpad prototype: raw JSON differs, normalized ids are equal.)
  - Rule: a finite float with `v.is_integer()` is converted to `int` (this also maps `-0.0` to `0`). Any other finite float is emitted with Python's shortest round-trip `repr` (the `json.dumps` default). Non-finite values are rejected (`allow_nan=False`).
  - Proposed validator bound: `|value| ≤ 2**53` so that the int/float equivalence is exact. **Open question Q3**: whether to restrict v0 predicate values to integers outright. All four v0 metrics are counters.

### 5.4 Unicode rules

- Under the proposed domain every string is ASCII by schema construction:
  - `gene_id` pattern `^[a-z0-9_]+(_v[0-9]+)?$`;
  - the artifact URI grammar;
  - metric and op enums;
  - `schema_version` const;
  - fixed slot paths;
  - regulation keys, which are validated to reference existing gene_ids.
- All unconstrained string fields (`provenance.notes`, `provenance.source_trajectory`, `provenance.cig_record`, `lineage_id`, `parent`) are excluded.
- No Unicode normalization (NFC/NFD) is therefore needed. `compute_genome_id()` **asserts the canonical bytes are ASCII** and raises otherwise, so a future schema change that admits Unicode into the domain cannot silently produce platform-dependent ids without a new ADR.

### 5.5 Why not RFC 8785 (JCS)

- For this ASCII + integer domain, the output of `canonical_json` coincides with JCS:
  - JCS sorts keys by UTF-16 code units, which matches code-point order for ASCII;
  - integers serialize identically.
- The difference is only in formatting non-integral floats with exponents (e.g. Python `1e-07` vs ECMAScript `1e-7`). §5.3 confines and normalizes floats.
- Adopting JCS would add a third canonicalizer and possibly a dependency, for no gain inside the domain.
- If future identity domains include arbitrary floats or non-ASCII strings, JCS should be reconsidered in that ADR.

## 6. Hash algorithm, prefix length, collision estimate

- **Algorithm:** SHA-256 (`hashlib`), the same as D8 artifact addressing, trajectory digests and the journal chain.
- **Id format:** `G-` + the first **16** lowercase hex characters (64 bits) of the digest. Tighten the schema pattern (genome and cig schemas) from `^G-[a-z0-9]{8,}$` to `^G-[0-9a-f]{16}$`. This is a schema change, and therefore part of this ADR.

**Collision estimate.** The birthday approximation for accidental collisions among `n` distinct genotypes, with `b`-bit prefixes, is `p ≈ 1 − exp(−n(n−1)/2^(b+1))`. The values below were computed in the scratchpad.

| prefix | n = 128 | n = 10^3 | n = 10^4 | n = 10^5 | n = 10^6 | n = 10^7 |
|---|---|---|---|---|---|---|
| 32 bits (8 hex; current examples) | 1.9e-6 | 1.2e-4 | **1.2e-2** | **0.69** | ≈1 | ≈1 |
| 48 bits (12 hex) | 2.9e-11 | 1.8e-9 | 1.8e-7 | 1.8e-5 | 1.8e-3 | 0.16 |
| **64 bits (16 hex; proposed)** | 4.4e-16 | 2.7e-14 | 2.7e-12 | 2.7e-10 | **2.7e-8** | 2.7e-6 |
| 96 bits (24 hex) | 1.0e-25 | 6.3e-24 | 6.3e-22 | 6.3e-20 | 6.3e-18 | 6.3e-16 |

- **Expected sizes [unverified, assumption].** SPEC §24's illustrative manifest uses `population.size: 16`, `generation: 8`, so about 128 genotypes per run. Across an M1 research program (many runs, seeds and ablations, plus mutation candidates that are evaluated but not selected), this draft assumes a working upper bound of 10^5 to 10^6 distinct genotypes. No SPEC number fixes this.
- At 10^6 genotypes:
  - 8 hex gives a near-certain collision, so the current 8-hex placeholders are unsafe beyond M0;
  - 16 hex gives about 3e-8.
- **Threat model:** accidental collisions only. LLM-guided mutation is not an adversary. 64 bits is **not** collision-resistant against a deliberate attacker (about 2^32 work); if adversarial genomes ever enter the threat model, store and compare the full 64-hex digest. As defense in depth now:
  - `genome_digest()` (full 64 hex) is stored next to every persisted `genome_id` (genome store, lineage store, run manifest);
  - any store insert where an existing `genome_id` maps to a different full digest raises `GenomeIdCollisionError` (fail closed).
- **Open question Q1:** 16 hex vs 20/24 hex. 16 is proposed as the shortest hex length that stays below 1e-7 at 10^6 (2.7e-8; 15 hex gives 4.3e-7) while staying readable in logs and fixtures. (Corrected in review: at 10^7, 16 hex gives 2.7e-6, **above** 1e-6; staying below 1e-6 at 10^7 needs 17 hex, 1.7e-7.)

## 7. API and invariants

### 7.1 Proposed API (new module `genome/identity.py`; draft signatures, not implemented)

```python
GENOME_ID_RULE = "1"          # bump only via new ADR
GENOME_ID_HEX_LEN = 16
GENOME_ID_PATTERN = re.compile(r"^G-[0-9a-f]{16}$")

class GenomeIdentityError(GenomeValidationError): ...       # claimed id mismatch / malformed
class GenomeIdCollisionError(RegistryIntegrityError): ...   # same prefix, different full digest (stores)

def genome_identity_domain(genome: dict) -> dict:
    """Pure projection to the §4.1 domain object. Never reads genome['genome_id'],
    'generation', 'parent', 'lineage_id', 'origin' or 'provenance'. Applies §5.2
    ordering and §5.3 numeric normalization; raises on bool/non-finite/duplicate
    skills/duplicate predicates/empty conditions. Requires a structurally valid genome."""

def canonical_genome_bytes(genome: dict) -> bytes:   # canonical_json(domain).encode("utf-8"); asserts ASCII
def genome_digest(genome: dict) -> str:              # full 64-hex sha256
def compute_genome_id(genome: dict) -> str:          # "G-" + genome_digest(genome)[:16]
def validate_genome_id(genome: dict) -> str:
    """Raise GenomeIdentityError unless GENOME_ID_PATTERN matches and
    genome['genome_id'] == compute_genome_id(genome). Returns the id."""
def with_genome_id(genome: dict) -> dict:            # deep copy with the computed id set; the ONLY
                                                     # sanctioned way builders/mutation/VTA assign ids
def manifest_sha256(genome: dict) -> str:            # sha256(canonical_json({"kind":"heritagent-genome-manifest",
                                                     #   "id_rule":"1","document":<genome minus genome_id>}))
```

Integration points (proposed):

- `load_genome()` calls `validate_genome_id()` after the structural and semantic checks. This is unconditional; there is no opt-out flag.
- `project_for_runtime()` and `compile_runtime_config()` inherit it through `load_genome`.
- `diff_genomes()` uses `genome_identity_domain()` for its regulation comparison, so `2` vs `2.0` is not reported as `changed`.
- The CIG overlay (`inheritance/overlay.py`) should label the with-trait config with the computed id of the overlay genotype, or with an explicit `overlay_of=<base id>` marker, rather than the base id (§9 O2).

### 7.2 Invariants (each maps to tests in §10)

| ID | Invariant |
|---|---|
| I1 Determinism | Same document gives the same id across processes, `PYTHONHASHSEED` values and supported CPython versions (≥3.10). |
| I2 Self-exclusion | Changing only `genome_id` never changes `compute_genome_id()`. |
| I3 Lineage independence | Changing only `generation`, `parent`, `lineage_id`, any `origin`, or any `provenance.*` leaves the id unchanged. `manifest_sha256` changes. |
| I4 Sensitivity | Each single change gives a new id: slot occupancy (add/remove/move), `gene_id`, `version`, `type`, `artifact` digest, skill-set membership, regulation (add/remove/metric/op/value/all_of↔any_of), `schema_version`, `id_rule`. |
| I5 Ordering independence | Object key order, `skills[]` permutations and predicate-list permutations give the same id. |
| I6 Numeric equivalence | Integral float and int predicate values give the same id. NaN/Inf/bool are rejected. |
| I7 Claimed-id validation | Pattern-invalid ids (8-hex legacy, uppercase, wrong length, non-hex) and pattern-valid but mismatched ids (including a copied parent id) are rejected by `load_genome`. |
| I8 Diff consistency | For validated genomes, `parent.genome_id == child.genome_id` holds **iff** `diff_genomes` reports no heritable mutation, assimilation or removal. A violation in either direction raises an integrity error. This replaces the current "ids must differ" guard and allows explicit clonal edges. |
| I9 Phenotype parity | Two genomes with equal ids compile, against the same registry, to `RuntimeConfig`s that are identical except for `genome_id` (same policies, same skills in the same order, same regulation). |
| I10 Operator discipline | Mutation, VTA/assimilation and seed builders assign ids only via `with_genome_id()`. A child whose heritable content changed always gets a new id (I4), and a hand-written id fails I7. |

## 8. Migration

There is no EXP-0001 run yet, so migration is a fixture and constants change with no data rewrite. **[unverified]**: no run directories or recorded `.jsonl` artifacts are tracked in the repo; any untracked local runs bound to placeholder ids would be pre-migration and are not valid EXP-0001 evidence anyway.

1. **`runtime/seed_g0.py`**
   - The existing `build_g0_seed_genome(registry)` embeds the constant `G0_GENOME_ID = "G-0f1e2d3c"`; it should instead return `with_genome_id(...)` of the document it builds. (`G0_GENOME_ID` is not imported anywhere else in the repo at `6014bc7`.)
   - Under rule "1" as drafted (with the `skills: []` and `regulation: {}` keys present in the domain; omitting either changes the value), with the current payloads `{"style":"react","max_plan_steps":8}` and `{"max_retries":3,"backoff":"exponential"}`, the prototype computed **`G-ab49fa967c969a10`**. The full digest is `ab49fa967c969a104b9a5d8b42e17d7fd61f1e44df88656cb90cfc88482b60fe`, and the artifacts are `planner_react_v1@1/sha256:bfa1661e…` and `retry_backoff_v1@1/sha256:6ff872ca…`.
   - This value is **illustrative**: it is from a scratchpad prototype, and any rule change in review will change it.
   - Keep the `G0_GENOME_ID` symbol only as a computed module-level value, or remove it.
2. **Golden vector:** one test pins the G0 domain bytes and id, so accidental drift in the rule or canonicalizer fails loudly. Every other test computes ids instead of hard-coding them.
3. **Test fixtures using placeholder ids:**
   - `tests/conftest.py` (`minimal_genome` `G-deadbeef`, doc-example `G-a1b2c3d4`, `G-0f1e2d3c`), `tests/genome/test_diff.py` (`G-11111111`, `G-99999999`, `G-22222222`), `tests/genome/test_projection.py`, `tests/genome/test_records.py`, `tests/miner/test_acquired_origin.py`, `tests/runtime/test_expression.py`.
   - Fixtures that pass through `load_genome` must use `with_genome_id()`.
   - `test_diff.py` children must be built by mutation plus `with_genome_id()`, not by assigning literal ids.
4. **Trajectory and miner fixtures** (`TrajectoryContext(genome_id="G-0f1e2d3c")` in `tests/trajectory/*`, `tests/miner/*`): the trajectory schema only requires a string, so these are not forced to change. Proposed: switch them to the computed G0 id for consistency, so that fixtures never carry an id that `load_genome` would reject.
5. **`architecture/eag-v0-schema.md`**
   - The §3 schema snippet (pattern), §4 CIG example `genome_id` and §5 example instance (`G-a1b2c3d4`, `parent: G-0f1e2d3c`) must be updated, because `tests/conftest.py::doc_example_genome()` loads the doc example through the validator.
   - The example uses illustrative artifact digests, so its id is computed from them, and its `parent` must be a pattern-valid 16-hex id.
   - Update D6 text with a pointer to this ADR. The doc and JSON schema are kept in sync, as in past rounds.
6. **Schemas:** `eag-0.1.genome.schema.json` and `eag-0.1.cig.schema.json` get `genome_id` patterns `^G-[0-9a-f]{16}$`; the genome schema's `parent` (which **has no pattern today**) gains the same pattern and stays nullable. The cig schema has no `parent` genome field. `schema_version` stays `"0.1"`. **Open question Q4**: whether tightening the pattern warrants `0.2`. The draft argues no, because no persisted 0.1 genomes exist outside fixtures.
7. **Expression parity:** canonical skill order in `compile_runtime_config()` (§5.2). It must land in the same change set, before EXP-0001.
8. **CIG (PR #28, merged in `6014bc7`):** CIG aggregate records carry `genome_id` of the base config (`inheritance/gate.py:741`). After migration this is the computed G0 id; no structural change is needed. Merge-order coordination is no longer needed because #28 is merged. Gate tests that build G0 via `g0_runtime_config()` (`tests/gate/*`, `tests/inheritance/*`) pick up the computed id automatically.

## 9. Observations for cross-review (outside the strict #22 scope)

- **O1:** `genome/validation/registry.py::canonical_bytes` omits `allow_nan=False`, unlike `trajectory.storage.canonical`. A policy payload containing NaN would be content-addressed as non-standard JSON (`NaN`). Suggest aligning it by reusing `canonical_json`. Check that existing artifact digests are unchanged; they should be, since no stored payload is expected to contain non-finite floats **[unverified]**.
- **O2:** CIG overlay episodes log the base genome's id for a with-trait phenotype that differs from the base genotype (`overlay.py:107/116` and the stage-4 pair configs in `interaction.py:272/276` → `loop.py:152`). Evidence linkage relies on other fields (`target_gene_id`, condition). Under this ADR, with-trait runs should carry the overlay genotype's computed id or an explicit overlay marker, to avoid mislabelled trajectories in the lineage/evaluation stores. This is an observation, not a confirmed bug in CIG verdicts.
- **O3:** the CIG overlay appends the target skill **last**, while G1 compiled under §5.2 orders skills by `gene_id`. CIG-measured effects (with the trait at the final prompt position) may not transfer exactly to G1's prompt (trait at a lexical position). This is a possible position confound between gate evidence and arm-C evaluation. **Scope note (review):** when the gate base config is G0 (skills empty), as in the gate tests (`g0_runtime_config()`) and as the EXP-0001 design implies **[unverified for the #13 wiring]**, the stage 1–3 with-trait config holds exactly one skill, so canonical ordering inside the overlay would change nothing there. The stage-4 `both` config already orders the pair lexically. The substantive gap is single-skill gate context vs G1's multi-skill prompt, which ordering alone does not fix. The Research/Critic should decide whether the overlay should also use canonical order (recommended for parity) **[requires ADR / ADR-0002 consideration if the gate protocol is locked]**.

## 10. Test plan (mirrors issue #22's list, then extends it)

These tests would live in `tests/genome/test_identity.py` unless noted. No new dependencies: `itertools.permutations` and seeded `random.Random` shuffles instead of `hypothesis`, which is not in `pyproject.toml`.

**Issue #22 items:**

1. **Ordering independence (I5):**
   - shuffle object key order at every nesting level (rebuild dicts in random order, 50 seeded shuffles);
   - all permutations of a 4-skill genome;
   - all permutations of `all_of`/`any_of` predicate lists;
   - all of these give the same id.
2. **Changed artifact gives a changed id (I4):**
   - re-`put()` a payload as `version+1` with one field changed, then swap the ref; the id changes;
   - alter only the URI digest (fixture with fake registry); the id changes;
   - table-driven single-field mutations for `gene_id`, `version`, `type`, slot move, skill add/remove, `schema_version`.
3. **Changed regulation gives a changed id (I4):** add or remove a rule; change metric, op or value; move a predicate from `all_of` to `any_of`.
4. **Invalid claimed id rejected (I7):**
   - the legacy `G-0f1e2d3c`;
   - an uppercase hex id;
   - 15 or 17 hex characters;
   - non-hex `G-zzzz…`;
   - a correct-format but wrong id;
   - the child carrying the parent's id after a heritable change;
   - `load_genome` raises `GenomeIdentityError` (a subclass of `GenomeValidationError`).
5. **Same heritable content reproduces the same id (I1, I3):**
   - recompute in subprocesses with `PYTHONHASHSEED` ∈ {0, 1, random};
   - vary only `generation`, `parent`, `lineage_id`, `origin`, `provenance.*`: the id is unchanged and `manifest_sha256` changes.
6. **Mutation/assimilation gives a new id (I10):** build G1 from G0 by adding an assimilated skill via `with_genome_id()`; the id differs, it validates, and `diff_genomes` reports `assimilated_traits=[gene]`.

**Extensions:**

7. **Self-exclusion (I2):** set `genome_id` to arbitrary strings; `compute_genome_id` is unchanged.
8. **Numeric normalization (I6):** `2` vs `2.0` vs `-0.0`/`0` give equal ids; `NaN`, `inf` and `True` are rejected. A gene `"version": 1.0` is rejected by `load_genome` (or normalized, per §5.3); today it is accepted.
9. **Empty-form equivalence:** absent vs `{}` sections, absent vs `[]` skills, absent vs `{}` regulation give equal ids. An empty condition entry is rejected (pending Q2).
10. **Duplicates rejected:** a duplicate skill `gene_id` and a duplicate predicate both raise.
11. **ASCII guard:** a monkeypatched domain containing a non-ASCII string raises.
12. **Golden vector:** the pinned G0 canonical bytes and id (§8.2).
13. **Diff consistency (I8):**
    - clonal edge (same genotype, `generation` 0→1): no raise, empty mutations;
    - a forged equal id with a heritable change raises;
    - different ids with an empty heritable diff raise;
    - `2` vs `2.0` regulation is not reported as `changed`.
14. **Phenotype parity (I9),** in `tests/runtime/`: two skill-permuted genomes give identical `RuntimeConfig.skills` order and identical `skill_block()` text.
15. **Collision handling:** monkeypatch `GENOME_ID_HEX_LEN = 1` to force a prefix collision; the store insert raises `GenomeIdCollisionError`.
16. **Schema sync:** the `test_schema_artifacts.py` extension asserts that the genome and cig `genome_id` patterns equal `^G-[0-9a-f]{16}$`, and that the doc-example genome validates with its computed id.
17. **Canonicalizer parity:** genome identity uses `trajectory.storage.canonical.canonical_json` byte-for-byte (a guard for any future move to a neutral module).

## 11. Reason (summary)

- It operationalizes D6 and is consistent with D8. It is **in tension with D3** (lineage no longer derivable from genome documents alone; H-d) and needs a D7 key-context decision for EXP-0001 run records (H-a). Both are flagged for the critic, not resolved here.
- It satisfies issue #22's invariants literally, including "same heritable content → same id".
- It matches SPEC §13's agent/genotype separation.
- It reuses the repo's locked canonicalizer convention instead of inventing a new one.
- It closes a real phenotype ambiguity: skill order affects the prompt but is not captured by an order-insensitive id unless expression is canonicalized too.

## 12. Research implication

- Genotype-keyed evidence becomes well defined: the same genotype means the same key, and measurements on *distinct task instances* pool. Repeated deterministic executions on the same instance do not add evidence (EXP-0001 `no_pseudo_replication`).
- This requires analyses to separate *genotype* evidence from *individual or lineage* evidence explicitly (SPEC §34: do not count re-measurements of an identical genotype as independent genomes).
- For EXP-0001, arms that collapse to the same genotype (A = C when there are 0 promotions; B = C when all 10 pass) become **visible by id**. This is a useful diagnostic for H2 interpretation. It is not a result, and no claim is made here.

## 13. Experiment needed

- None to adopt the rule.
- The §5.2 expression-order change alters runtime prompts. If adopted, a small sanity check is recommended before EXP-0001: G0 and the arm configs compile and run end-to-end. This is not an efficacy experiment.
- Possible position-sensitivity of skills (O3) is an empirical question for the Research/Critic to decide whether to pre-register **[requires ADR if added to EXP-0001]**.

## 14. Compatibility impact

| Component | Impact |
|---|---|
| Schemas (genome, cig) + architecture doc | Pattern tightening; examples recomputed (§8). |
| `genome/validation/loader.py` | `validate_genome_id` is now mandatory; new duplicate and empty-condition checks. |
| `genome/validation/diff.py` | Guard replaced by I8; normalized regulation comparison. |
| `runtime/expression` | Canonical skill order (behaviour change before EXP-0001). |
| `runtime/seed_g0.py` | Computed id. |
| `inheritance/overlay.py`, `inheritance/interaction.py` (PR #28, merged) | Labelling of overlay and pair configs (O2), possibly skill order (O3). |
| #13 EXP-0001 bookkeeping | Arm results must not be keyed by `genome_id` alone (H-a). |
| #16 lineage store | Node key ≠ `genome_id`; stores `genome_id`, `genome_digest` and `manifest_sha256`. |
| Evaluation store / BackboneEvaluationRegistry | D7 key form unchanged for genotype-level records; must store the full digest for collision checks. EXP-0001 run-level records need arm/mining-seed context that the D7 key `(genome_id, env, seed)` does not carry (H-a): critic decision. |

## 15. Open questions for Research/Critic and the owner

- **Q0 (central):** Is the genotype-only identity (§4.3) acceptable, or does the critic prefer a lineage-inclusive id (L1)? Everything in H-a, H-b, H-d and I8 follows from this choice.
- **Q1:** Prefix length: 16 hex (proposed) vs 20/24.
- **Q2:** Empty regulation conditions: reject (proposed) or normalize away.
- **Q3:** Restrict v0 predicate values to integers.
- **Q4:** Does the `genome_id` pattern tightening require `schema_version` "0.2"? Proposed: no.
- **Q5:** Should the canonical skill order also apply inside the CIG overlay (O3)? This may touch the ADR-0002 gate protocol, so it **requires ADR** if the protocol is locked.
- **Q6:** Should `origin` participate after all? A seed gene and an assimilated gene with identical artifacts currently collide. This draft argues they should (same phenotype), but the critic may want `origin` in the id to keep CIG-validated and unvalidated genotypes apart. Proposed alternative: keep that distinction in `manifest_sha256` and the run keys.

*Nothing in this draft is decided. Adoption requires Research/Critic cross-review (SPEC §32) and owner approval. Locked or pre-registered elements potentially affected (EXP-0001 result keys H-a, prompt order §5.2/O3, ADR-0002 gate protocol Q5) are flagged and must not be changed on the basis of this draft alone.*
