# ADR-0010 — Every CIG child record names the artifact and the base genome it measured

Date: 2026-10-15 · Status: **ACCEPTED by the owner** (Samet, 2026-10-15). The owner accepted it in chat as ADR-0009 D7: the draft's option A with the Critic's modifications. The draft's four questions were answered as builder-2's Critic cross-review recommended (the draft recommended answers to the first three and left the fourth open):
- A, not B or C;
- the interaction evidence carries versions and artifacts too, beside its gene ids;
- evidence for another version fails closed;
- the base genome is bound, the target regulation is not (the cross-review's answer to the fourth).

Made before any pilot or EXP-0001 episode; no CIG record exists. Origin: takeover note §5.8; the draft `research/proposals/ADR-draft-child-artifact-binding.md` (2026-10-14), which this ADR supersedes; builder-2's Critic cross-review of it (workflow `wf_ba6185ce-d72`, posted on #13). Artifacts: `inheritance/overlay.py`, `replay.py`, `ablation.py`, `generalization.py`, `interaction.py`, `gate.py`, `genome/validation/records.py`, `evaluation/exp0001/pipeline.py`, `architecture/eag-v0-schema.md` (its ADR-0010 section), the tests that build records by hand.

**Problem.**
- A gate run promotes or rejects one *artifact*. The somatic store keys candidates by `(gene_id, version)`, and registry artifacts are content-addressed (`registry://skills/<gene_id>@<version>/sha256:<digest>`). Since #31 the aggregate record names the evaluated artifact (`artifacts.candidate`).
- The five child records name only `candidate_gene_id`: replay, ablation, generalization in-scope and out-of-scope, and interaction_regression.
- Stages 1–2 run outside the gate, and the gate receives their evaluations. So stage 1–2 evidence for `look_v1@1` can be combined with a gate run on `look_v1@2`. The aggregate would name version 2 although stages 1–2 measured version 1, and nothing in the records would show it.
- The interaction pair evidence names gene ids without versions.
- The evaluated unit is the base genome, the candidate artifact and the candidate's target regulation. No child records the base genome either. The draft's own review showed that stage 1–2 evidence produced under one base genome was accepted by a gate run under another.

**Current design.**
- Child `measurements` and the aggregate's `artifacts` are free-form in the CIG schema (ADR-0001).
- `CigRecordStore._candidate_binding_issue` requires each child's `candidate_gene_id` to equal the aggregate's, in `add_bundle` and in `verify()`. It never reads `aggregate.artifacts.candidate`.
- The gate writes `artifacts.candidate`, but the store accepts an aggregate without it.

**Decision (owner).**
- **B1 Record.**
  - Every child carries `candidate_version` (a strict int), `candidate_artifact` (the registry URI) and `base_genome_id` in its `measurements`, beside `candidate_gene_id`.
  - The values come from the validated candidate and the base runtime configuration, not from the raw envelope. The vacuous out-of-scope child (ADR-0007 E6) carries them too.
  - The aggregate's `artifacts` gains `base_genome: {genome_id}` beside `candidate`.
- **B2 Interaction evidence.** Each pair evidence keeps `gene_ids` and adds `artifacts`: one `{gene_id, version, artifact}` per gene, in `gene_ids` order. Its `evidence_sha256` covers them.
- **B3 Gate check before any episode.** The gate refuses the batch when a stage 1–2 child names another `(gene_id, version, artifact)` than the candidate it evaluates, or another base genome than the gate's. A child that lacks a field is a mismatch. It is the only check of the stage 1–2 children's binding that fails before a stage-3 episode is spent: it saves the stage 3–4 episodes, not the stage 1–2 episodes already run.
- **B4 Store check at commit and on stored state.**
  - An M0 bundle needs a complete and coherent `artifacts.candidate`:
    - the URI parses;
    - its name is `artifacts.candidate.gene_id`, which equals the aggregate's `candidate_gene_id`;
    - its version, read as an int from the URI's string, equals `artifacts.candidate.version`, a strict int.
  - The bundle also needs `artifacts.base_genome.genome_id`, equal to the aggregate's own `genome_id` (B1 takes both from the base configuration).
  - Every child's four fields must equal those references.
  - In the interaction_regression child, each pair evidence must:
    - match its own digest;
    - name two distinct genes, the candidate and the partner its `other_gene_id` names;
    - carry an entry for the candidate equal to `artifacts.candidate`;
    - have every entry coherent in the same way.
  - Both copies of a pair evidence are linked: for every pair evidence in a child of candidate g that names partner p, an interaction_regression child of an M0 aggregate for p carries a pair evidence with the same `evidence_sha256`. This check spans two bundles. It therefore runs in `verify()`, and in the gate as a store-wide check on the staged store, once every bundle of the batch is added and before any live store changes.
- **B5 Not bound.** The target regulation is not recorded or compared (owner, Q4, as the cross-review recommended); it stays a recorded residual. A regulated candidate would first have to become part of the candidate, pinned by its artifact URI, in its own ADR.
- **B6 Scope.** The store checks prove that the records are consistent with each other and with the artifact URI (name and version; the digest is not recomputed). They do not authenticate the records: a forged, self-consistent set of bundles is not caught.
- **B7 Compatibility.** `THRESHOLD_VERSION` stays `EXP-0001-CIG-M0-v1` (as in ADR-0005: free-form fields, no record yet). `architecture/eag-v0-schema.md` documents `artifacts.candidate`, `artifacts.base_genome` and the child fields.

**Reason.**
- **B1–B4** close the version gap where the evidence is consumed (the gate, before any episode) and where it is kept (the store), with the URI the aggregate and the somatic store already use. They need no schema change.
- **Binding the base genome** costs one field and closes the second gap the draft's review reproduced: the aggregate asserts a `genome_id` that no child attested.
- **The regulation** is not bound, for the cross-review's reasons: it is a parameter the caller passes beside the candidate, not part of the candidate or its artifact; no EXP-0001 path passes one, so the gate runs every candidate constitutively; and the aggregate asserts nothing about it.

**Research implication.**
- None for the hypotheses.
- Each CIG verdict becomes traceable to the artifact URI and the base genome each stage recorded. The records are checked for mutual consistency, not authenticated.
- No novelty or causal claim.

**Experiment needed.** None. No CIG record exists.

**Compatibility impact.**
- **Records.** Child `measurements` gain three fields. The aggregate's `artifacts` gains `base_genome`. Pair evidence gains `artifacts`, so its `evidence_sha256` changes for the same outcomes. The CIG schema is unchanged, and no stored record exists.
- **Gate.** It gains one pre-episode check, and a store-wide check on its staged store before any live store changes.
- **Run.** A resumed EXP-0001 run re-verifies each seed's stored `gate.json` records as a record store (B4 on stored state) before the somatic journal takes their decisions, and stops if they do not verify. This adds refusals only.
- **Store.** It gains:
  - the per-child comparison;
  - the coherence of `artifacts.candidate`;
  - the pair-evidence digest, own-entry and partner-link rules;
  - the refusal of an M0 bundle without complete references.
- **Tests.** The fixtures that build records by hand change: `tests/gate/helpers.py`, and `tests/genome/test_records.py` (`m0_children`, and `aggregate_record`, whose `artifacts` was empty). `test_aggregate_artifacts_carry_the_evaluated_version` builds version-2 evidence for its version-2 gate run.
