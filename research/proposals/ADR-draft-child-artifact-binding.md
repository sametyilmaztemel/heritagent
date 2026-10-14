# ADR draft — Every CIG child record names the artifact it measured

Date: 2026-10-14 · Status: **PROPOSED** (builder-2 draft at the owner's request, 2026-10-14). Not accepted; the code changes only after the owner accepts it and Codex cross-reviews it (SPEC §32). Origin: takeover note §5.8 (the residual of the stage 1–2 artifact-binding finding).

**Problem.**
- A gate run promotes or rejects one *artifact*. The somatic store keys candidates by `(gene_id, version)`, and registry artifacts are content-addressed (`registry://skills/<gene_id>@<version>/sha256:<digest>`).
- Since #31 the aggregate record names the evaluated artifact (`artifacts.candidate`: gene_id, version, artifact URI). Since #35 the commit also refuses a stored candidate whose artifact differs from the evaluated one.
- The five child records name only `candidate_gene_id`:
  - replay;
  - ablation;
  - generalization in-scope;
  - generalization out-of-scope;
  - interaction_regression.
- Stages 1–2 are produced outside the gate. #11's `ReplayEvaluator` and `AblationEvaluator` run first, and the gate receives their evaluations as `GateCandidateInput`. The gate's `_consume_stage12_evidence` checks the child id, parent, seed, stage, `candidate_gene_id`, `stage_pass`, the generation settings and (since ADR-0005) the ablation bootstrap. It cannot check *which version* the children measured.
- So stage 1–2 evidence for `look_v1@1` can be combined with a gate run on `look_v1@2`. The aggregate would then name version 2, although stages 1–2 measured version 1, and nothing in the records could reveal it.
- The interaction pair evidence (`canonical_evidence.gene_ids`) has the same gap: it names gene ids without versions.

**Current design.**
- Child `measurements` is a free-form object in the CIG schema ("append-only raw measurements"), and so is the aggregate's `artifacts`.
- The store already enforces part of a binding outside the schema. `CigRecordStore._candidate_binding_issue` requires every child's `measurements.candidate_gene_id` to equal the aggregate's. It runs in `add_bundle`, which the gate's commit uses, and in `verify()` over stored state for M0-shaped aggregates. It never reads `aggregate.artifacts.candidate`, so a child's version or artifact is not compared with anything.
- The gate always writes `aggregate.artifacts.candidate`, but the schema leaves `artifacts` optional, and the store accepts an aggregate without it.
- The #11 evaluators receive the full candidate envelope (`candidate_envelope`, whose `candidate` holds gene_id, version and artifact). They pass it to `build_evaluation_configs`, but they record only the gene id.
- The stage-3 children come from the `GeneralizationEvaluator` that the gate runs. The gate builds the interaction_regression child itself, in `_build_stage4_child`, which receives the gene id only. `RegressionEvaluator` and `InteractionEvaluator` emit no child record.

**Options.**
- **A. Record, check at the gate, and check in the store; no schema change (recommended).**
  - **Record.** Every child carries `candidate_version` and `candidate_artifact` in its `measurements`, next to `candidate_gene_id`, taken from the evaluated envelope: the #11 evaluators and `GeneralizationEvaluator` write them into their children; `_build_stage4_child` receives the evaluated candidate and writes them into the interaction_regression child. The interaction evidence records `{gene_id, version, artifact}` for both genes.
  - **Gate check, before any episode.** `_consume_stage12_evidence` refuses the batch if a stage 1–2 child's `(gene_id, version, artifact)` differs from the candidate envelope it evaluates. This is the only check that fails before any stage-3 episode is spent.
  - **Store check, at commit and on stored state.** `_candidate_binding_issue` also compares each child's `candidate_version` and `candidate_artifact` with `aggregate.artifacts.candidate`, and, in the interaction_regression child, the candidate's own entry in each pair evidence. It runs in `add_bundle` and, for M0-shaped aggregates, in `verify()`, independently of which code produced the records.
  - **Both copies of a pair evidence are linked.** Each pair evidence is embedded in both candidates' interaction_regression children, and its `evidence_sha256` is the digest of its `canonical_evidence`. The store requires two things:
    - each copy matches its own digest;
    - for every pair evidence in a child of candidate g that names partner p, an interaction_regression child of an M0-shaped aggregate for p carries a pair evidence with the same `evidence_sha256`.

    Equal digests mean equal evidence, and the own-entry check runs on M0-shaped aggregates, so the partner's entry in g's copy is then attested by p's own-entry check. A copy that names another version of the partner, even with a recomputed digest, has no linked copy and is refused.
  - **Where the link is checked.** It spans two bundles, so `add_bundle`, which sees one bundle, cannot check it. It runs in `verify()`, and as an explicit store-wide check on the staged store after the gate's commit has added every bundle (today `_commit_results` adds the bundles one by one and runs no store-wide check). A caller that adds a single bundle directly gets only the digest check.
  - **Fail closed on a missing or incoherent reference.** An M0 bundle is refused, and `verify()` flags it, in three cases:
    - its aggregate lacks a complete `artifacts.candidate` (gene_id, version and artifact);
    - the reference is not coherent: the artifact URI (parsed with `parse_uri`) must name `artifacts.candidate.gene_id`, which must equal the aggregate's `candidate_gene_id`, and must carry `artifacts.candidate.version`, a strict int (not a bool). The same applies to each pair-evidence entry;
    - a pair evidence in its interaction_regression child has no entry for the candidate.

    Otherwise the store check would pass any version whenever the reference or the entry is absent, or whenever a version label disagrees with the content-addressed URI.
  - **Scope of the store checks.** They check that the stored records are consistent with each other and with the artifact URI (its name and version; the digest itself is not recomputed). They do not authenticate the records: a writer who forges a complete, self-consistent set of bundles is not caught. That would need signed records, which is outside this ADR.
- **B. A and make the fields schema-required.**
  - The CIG child schema requires `candidate_gene_id`, `candidate_version` and `candidate_artifact` in `measurements` for the five gate stages, and the aggregate schema requires `artifacts.candidate`.
  - This adds presence only: a schema sees one document and cannot compare a child with its aggregate, so the match itself is still checked by the gate and the store. It changes the locked ADR-0001 schema and makes `measurements` partly structured.
- **C. Bind by digest.**
  - Each child records the SHA-256 of the evaluated with-trait configuration: the base genome, the candidate artifact and the target regulation.
  - Broader than A: it also pins the base genome and the target regulation, which the artifact URI does not (see the question on scope below). Harder to audit by eye, and for the artifact alone it repeats what the URI already pins.

**Recommendation.** A.
- It closes the version gap where the evidence is consumed (the gate, before any episode) and where it is kept (the store), with the artifact URI that the aggregate and the somatic store already use.
- It needs no schema change. B adds only presence checks.

**Scope note: base genome and target regulation.**
- The evaluated unit is the base genome, the candidate artifact and the candidate's target regulation (`build_evaluation_configs`). A binds the artifact only.
- No stage 1–2 child records the base genome or the regulation, and the gate does not compare them. The own review of this draft reproduced both cases: stage 1–2 evidence produced under one base genome, or without a regulation, was accepted by a gate run under another base genome, or with an `express_when` condition, and the aggregate named the gate's genome.
- This is the same kind of attribution gap. It is left as a question rather than folded into A, because it widens the binding beyond §5.8.

**Questions for the owner and Codex.**
1. A or B (or C)?
2. Should the interaction evidence carry versions and artifacts too? Recommended: yes. Without them the pair evidence cannot attest which artifacts interacted.
3. Is stage 1–2 evidence produced for a different version an error (fail closed, recommended), or a warning?
4. Should the binding also cover the base genome (`genome_id`) and the target regulation (A plus two fields, or C), or stay artifact-only and record those as an accepted residual?

**Research implication.**
- None for the hypotheses. Under A, each CIG verdict becomes traceable to the exact artifact every stage measured (ADR-0001 reason (c)); the base genome and the regulation are traceable only if question 4 says so.
- No novelty or causal claim.

**Experiment needed.** None. No CIG record exists yet.

**Compatibility impact (if A is accepted).**
- Child `measurements` gain two fields, and the interaction evidence records versions and artifacts. Whether `canonical_evidence.gene_ids` keeps its shape (with a new field beside it) or changes is part of the decision, because `evidence_sha256` is computed over the canonical evidence.
- The gate gains one pre-episode check, and one store-wide check on the staged store before it exposes the commit. The store gains: the per-child comparison with `aggregate.artifacts.candidate`; the digest and partner-link rules for pair evidence (in `verify()` and in that store-wide check); and the refusal of an M0 bundle without a complete reference.
- No stored record changes, since none exists. The `measurements` schema stays free-form.
- `tests/gate/test_gate_atomic_commit.py::test_aggregate_artifacts_carry_the_evaluated_version` pairs stage 1–2 evidence built for version 1 with a gate run on version 2, which is exactly what A refuses. Its fixture must build version-2 evidence.
- Test fixtures that hand-build records change: `tests/gate/helpers.py` (`coherent_stage11_children`), and `tests/genome/test_records.py` (`m0_children`, and `aggregate_record`, which today carries an empty `artifacts`).
- The assertions that pin the interaction evidence's `gene_ids` (`tests/gate/test_gate_orchestrator.py`, `tests/inheritance/test_interaction.py`) change only if `gene_ids` changes shape; they stay valid if the versions and artifacts are added beside it.
