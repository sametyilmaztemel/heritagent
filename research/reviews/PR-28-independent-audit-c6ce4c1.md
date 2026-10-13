# Independent audit: PR #28 (head c6ce4c1) and merged M0 code

## 1. Header

- Reviewer: Claude (third agent, independent of Builder/Zcode and Critic/Codex)
- Date: 2026-10-12
- Pinned commits: PR #28 head `c6ce4c1`; main `0ce0a36`
- Environment: Python 3.14.6 and Python 3.13 (main also run on 3.10, 3.11, 3.12 for portability)
- Method: read-only. Code reading at pinned commits, the full test suite, targeted repro scripts per claim (each re-run by a second verification pass), mutation runs on scratch copies, an exhaustive enumeration of the n=20 bootstrap rule, and a portability matrix. Nothing was posted to GitHub. The repo and PR worktree were not modified.
- Note: the PR head moved 4ca5517 -> 7886852 -> c6ce4c1 during this audit. This report is pinned to c6ce4c1. Any result for an older head is labelled as such.
- Wording: per SPEC §34, "confirmed" below means "reproduced at the pinned commit". It does not mean causal or novel claims.

## 0. Post-merge status (main 6014bc7, 2026-10-13)

- PR #28 was merged at head `628edd2` after the Research/Critic (Codex) cleared L6/L7. The section-2 recommendation "do not merge at c6ce4c1" is therefore superseded. It is kept below only as the record of the audit at c6ce4c1.
- Method of this update: read-only re-check at main `6014bc7` (repo not modified). Each earlier repro was re-run unmodified with `PYTHONPATH=.` under Python 3.14, and the source was re-read. Baseline suite at main: 443 passed.
- The c6ce4c1..6014bc7 merge delta touched only `genome/validation/records.py`, `inheritance/gate.py` (the `eval_generation_settings` strictness check, `import math`) and two test files. `inheritance/bootstrap.py`, `ablation.py`, `runner.py`, `regression.py` are unchanged, so no earlier finding was fixed by the merge except the string and bool cases of L6/L7.
- Wording per SPEC section 34: "still present" means "reproduced at 6014bc7". Nothing here is a causal or novelty claim.

### 0.1 Status of every confirmed finding at main

| Finding | Sev | Status at main | Location at main |
|---|---|---|---|
| L6/L7 loose #11 preflight (stage_pass, seed, settings types) | minor now | changed: string/bool cases fixed (`'true'`, `'false'`, `'11'` give EvidenceCoherenceError with 0 runner calls). Int 1, float 1.0 stage_pass and float seed 11.0 still pass the preflight (no-store mode returns promote; with stores add_bundle rejects only after 80 episodes) | `inheritance/gate.py:574` (`child['seed'] != seed`), `:582` (`evaluation.passed != m.get('stage_pass')`) |
| tau_r=0.05 sensitivity cell float-fragile (38/40 gives 0.050000000000000044) | major | still present; primary rule unaffected | `inheritance/regression.py:231`; `inheritance/gate.py:437`, `:457`, persisted `:673` |
| Somatic terminal decision not bound to evaluated artifact; aggregate `artifacts` is `{}` | major | still present (version now derived from evaluated envelope, digest still unbound) | `inheritance/gate.py:755-785` (state check `:771`, decide `:780`), `:749` |
| Chain resolution: eliminated candidate still eliminates; no `eliminated_by` persisted | minor | still present | `inheritance/gate.py:513-541`, `:654`, `:657-696`, `:366` |
| Late setup errors after episodes spent (token_counter, interaction_allocation, existing aggregate id) | minor | still present (80, 160, 80 runner calls) | `inheritance/gate.py:281`, `:315`, commit prevalidation in `_commit_results` (`:755+`) |
| Overhead leakage guard substring-matches English words, aborts batch after episodes | minor | still present | `inheritance/regression.py:295-300` |
| Ablation delta/lower_bound not bound to durable #11 child | minor | still present (hardening; needs forged evaluation) | `inheritance/gate.py:542-632`; consumed `:337`, `:416-417` |
| Batch commit not atomic (k-th decide() failure) | minor | still present | `inheritance/gate.py:755-780` |
| Stage 4C ignores per-candidate mining seeds (mixed-seed child accepted) | minor | still present | `inheritance/gate.py:322-325`, `:504` |
| sensitivity_recomputation reads aliased or stale run state | minor | still present | `inheritance/gate.py:239`, `:369`, `:407`, `:410` |
| tau_c=0.05 never binding at n=20 (0 of 231 decisions change; design property, not a code bug) | minor | still present | `inheritance/bootstrap.py:66-70`; `inheritance/ablation.py:15,142-175,203` |
| Unlocked bootstrap seed flips primary ablation verdict (61/200 seeds pass at 10/3/7) | minor | still present | `inheritance/ablation.py:42`, `:188`, `:204`, `:269` |
| Bootstrap inputs unvalidated (seed=None non-deterministic; NaN, out-of-range, bool deltas) | minor | still present | `inheritance/bootstrap.py:35-48`; `inheritance/ablation.py:188,204,269` |
| validate_outcome ignores status (adapter_error counted as outcome) | minor | still present | `inheritance/runner.py:184-223`, `:164` |

### 0.2 Second-pass results: mutation and test-adequacy leads

All mutants ran against a scratch copy of main; baseline 443 passed. Code at main is correct in M1, M4, M5, T1, T2 (gaps are in tests).

| Lead | Verdict | Result |
|---|---|---|
| M1 contribution ignored in `_resolve_harmful_pairs` (`inheritance/gate.py:527`) | confirmed, minor | mutant gives 443 passed; verdicts flip (a promote/b reject becomes a reject/b promote) when the higher contribution sits on the later rank. All harmful-pair tests (`tests/gate/test_gate_orchestrator.py:214-252`, `tests/gate/test_gate_sensitivity.py:87`) put it on the earlier rank |
| M4 tau_k overhead gate removed (`inheritance/gate.py:308-309`, `:660`) | confirmed, minor | three mutants, all 443 passed; a 600-token counter flips reject to promote. No gate test uses an over-threshold counter |
| M5 generalization removed from eligibility (`inheritance/gate.py:307`) | confirmed, minor | 443 passed; an ineligible candidate then eliminates an eligible one. No test sets `gen_in_with=False` or `gen_out_with=False` |
| T1 ablation child preflight (`inheritance/gate.py:554-555`) | confirmed, minor | dropping it gives 443 passed; a corrupted ablation child (temperature 0.7) is accepted and 80 episodes run. Preflight tests corrupt only the replay child |
| T2 `target_regulation` dropped (`inheritance/gate.py:273`, `:290`, `:322`) | confirmed, minor | four mutants, all 443 passed; spy shows regulated candidates run constitutively. Only evaluator unit tests cover it |
| T3 vacuous tests | confirmed, minor | `tests/gate/test_gate_atomic_commit.py:143` asserts on untouched live stores (a child-level non-atomic `add_bundle` mutant at `genome/validation/records.py:213` passes 443/443 and leaves 5 orphan children). `tests/gate/test_gate_orchestrator.py:327` `is not None` is always true (dict literal) |
| V1 `CigRecordStore.verify()` gaps | confirmed, minor | keyed on dict keys not `cig_id`, never re-runs `load_cig`, truthiness for stage pass (`genome/validation/records.py:255,286`, `_verdict_issue`); needs white-box corruption to trigger |
| V2 false "vacuous" reason on a non-eligible candidate | confirmed, minor | `inheritance/gate.py:665-690`, `:92-94`; 3 candidates, c fails replay: c's committed child says "fewer than two preliminary-eligible" while pair (a,b) exists. Verdicts unaffected |
| L67 residual loose comparisons | confirmed, minor | see L6/L7 row above; int/float cases still pass preflight |

### 0.3 Proposed follow-up issues (for the owner/Codex to accept or reject)

Blocking before #13 (they affect the evidence or claims that #13 will consume):
- Fix tau_r sensitivity float boundary. Compute drop with integer or Fraction arithmetic (`inheritance/regression.py:231`, `inheritance/gate.py:437,457`) and persist successes/total. Otherwise the tau_r=0.05 cell equals tau_r=0.03.
- Bind the somatic decision to the evaluated artifact. In commit prevalidation, require the stored candidate artifact to equal the evaluated envelope artifact, and record version and URI/digest in the aggregate (`inheritance/gate.py:749,771`).
- Lock and preregister the bootstrap seed and task order, and validate bootstrap inputs. Seed becomes a module constant recorded in the child; reject None, bool, NaN and out-of-range deltas (`inheritance/ablation.py:188`, `inheritance/bootstrap.py:35-48`).
- Define the terminal-status policy. `validate_outcome` raises unless status is `finished`, or such episodes are excluded under a minimum-evaluable-pairs rule (`inheritance/runner.py:184`).
- Document tau_c non-binding at n=20 (SPEC section 34 note, EXP-0001 design) so the tau_c sensitivity grid is not presented as informative.

Hardening (not blocking):
- Strict type checks in the #11 preflight for stage_pass and seed (`inheritance/gate.py:574,582`), with a parametrized replay/ablation test asserting 0 runner calls.
- Fail-fast setup checks at the start of `run()` for token_counter, interaction_allocation (len(inputs) >= 2) and existing aggregate ids (`inheritance/gate.py:281,315`).
- Make batch commit atomic, or downgrade the docstring (`inheritance/gate.py:755-780`); make retry idempotent.
- Chain-resolution semantics: decide, document, and test N>=3; persist `eliminated_by` in the child (`inheritance/gate.py:513-541,654`).
- Stage 4C per-candidate seed rule and `verify()` check child seed equals interaction locked seed (`inheritance/gate.py:322-325`).
- Snapshot run state for `sensitivity_recomputation` and clear it on abort (`inheritance/gate.py:369,407`).
- Replace the overhead substring guard with a structural check (`inheritance/regression.py:295-300`).
- Bind ablation delta and lower_bound to the durable child (`inheritance/gate.py:542`).
- Distinct non-vacuous reason for non-eligible candidates (V2, `inheritance/gate.py:665-690`).
- Test gaps: M1 inverted-rank harmful case; M4 over-threshold counter; M5 failing generalization; T1 ablation preflight; T2 regulation spy; T3 fix the two vacuous tests.
- Harden `CigRecordStore.verify()`: re-run `load_cig`, key checks on record `cig_id`, strict bool (V1, `genome/validation/records.py:241-292`).

## 2. Executive summary

- Historical merge-gate opinion at c6ce4c1 (superseded, see section 0): L6 and L7 were open and reproduced end to end, and the Critic's review also held them open. PR #28 was later merged at 628edd2 after Codex cleared them.
- L1 is partially open (the boolean-type clause, which is the same defect as L6). L2, L3, L4, L5 are closed. G1-G9 still hold (no regression).
- L6/L7 repros: a replay child with temperature=False, max_tokens=64.0, seed=11.0, an extra key, or stage_pass=1 is accepted. 80 stage-3/4 runner calls are spent and the values are committed to the live store with verify() == [].
- The same looseness exists at two sites the Critic did not name: gate.py:581 (stage_pass compare) and gate.py:573 (child seed compare).
- New confirmed PR #28 findings: 2 major (tau_r float boundary, somatic artifact binding) and 7 minor. Details are in section 4.
- The tau_r finding is the most consequential for the research claim: a third of the pre-registered tau_r sensitivity grid is degenerate. The primary rule is unaffected.
- New confirmed main (M0) findings: 4, all rated minor after verification. The tau_c grid is vacuous at n=20 (a prereg-design note, not a code bug).
- 4 candidate findings were refuted or downgraded to non-defects (section 5).
- Tests at c6ce4c1: 438 passed on 3.14.6. On 3.13: 435 passed and 1 collection error (the same NameError as main). There is no CI on any Python version.
- Mutation testing shows three verdict-flipping mutants survive all 438 tests (section 4.3). This is reported by the lens, without a second-pass verdict.

## 3. Latest Critic items and G1-G9 regression check

| Item | Independent verdict | Key evidence |
|---|---|---|
| L1 add_bundle/verify parity (stage_pass, candidate binding) | partially open | `genome/validation/records.py:125,130,217,248,259`; `tests/genome/test_records.py:254` |
| L2 #11 settings provenance, exact values | closed (value terms) | `inheritance/gate.py:585,593,599,262`; `tests/gate/test_gate_orchestrator.py:415` |
| L3 durable interaction evidence | closed | `inheritance/interaction.py:87,217,228`; `inheritance/gate.py:622` |
| L4 sensitivity not_identifiable only for missing pairs | closed | `inheritance/gate.py:450,462,469,333`; `tests/gate/test_gate_sensitivity.py:118,168` |
| L5 TokenCounter actual-count validation | closed | `inheritance/regression.py:285,289`; `inheritance/gate.py:293` |
| L6 stage_pass not type-checked as boolean | still open | `genome/validation/records.py:130`; `inheritance/gate.py:581` |
| L7 #11 settings check value-only | still open | `inheritance/gate.py:599,573` |
| G1-G9 regression check | all hold at c6ce4c1 | `inheritance/gate.py:736,251,554,645,621,231`; `genome/validation/records.py:148` |

### L1 (partially open)
- Closed: add_bundle rejects a missing stage_pass, a stage_pass that contradicts the aggregate, and a wrong or missing candidate_gene_id. verify() flags all four after white-box corruption of a stored child. The fixtures now include stage_pass and candidate_gene_id, so the old permissive path is gone.
- Open: the "boolean" requirement. `records.py:130` coerces with `bool()`. This is L6.
- Critic accuracy: the c6ce4c1 review statement ("presence + verify candidate binding implemented") is accurate. verify() checks candidate binding only for M0-shaped aggregates (records.py:264), which matches the L1 wording.

### L2 (closed)
- The loop at `gate.py:599-603` covers both replay and ablation children. Preflight runs at step 2 (gate.py:262), before any stage-3/4 evaluator is built.
- Repro: temperature 0.7, max_tokens 32, seed 99, stop ['x'] and missing settings each give EvidenceCoherenceError with 0 new runner calls, on both children.
- Critic accuracy: accurate. The PR test corrupts only the replay child, so the ablation path rests on inspection and my repro. The residual weakness (value-only `==`) is L7.

### L3 (closed)
- In a real two-candidate run, both stage-4 children (CIG-0007, CIG-0008) embed byte-identical canonical_evidence. evidence_sha256 equals sha256(canonical_json(record)). The record holds per-task task_id/family/bank_id/payload_sha256 and 20 paired outcomes with expressed_gene_ids.
- Residuals, not blocking: `InteractionAllocation.__post_init__` accepts bank_id=None and family='' (recorded as null/empty). Serialisation happens only after the 20 episodes (interaction.py:205), so an unserialisable payload fails after runner calls, still fail-closed.
- Critic accuracy: accurate. The persisted evidence is task identity plus a payload digest, not the payload itself, so "reconstructable" slightly overstates. Observation only: the "both" arm in the fixture expresses only gene 'a', and the code does not check that both target genes were expressed in that arm. Neither the Critic nor the handoff requires that check.

### L4 (closed)
- The old `newly = hyp_eligible - run.eligible` rule is gone. Each cell builds all pairs inside its hypothetical eligible set (gate.py:462) and is not_identifiable only when some pair is missing, listing them in `missing_interaction_pairs`.
- Repro A: a drops out while b is newly eligible gives an identifiable cell with promote ['b']. Repro B: three candidates give missing pairs exactly [['a','c'],['b','c']].
- Critic accuracy: accurate. Tests do not cover repro A or B, but behaviour is correct. Identifiable cells omit the missing-pairs key instead of setting [].

### L5 (closed)
- Check at `regression.py:289` runs before `passed` is computed. Through the full orchestrator, 1.5, 3.0, -1, True, False, None and '12' all raise RunnerContractError. Afterwards somatic state is still candidate and the CIG store is empty. An int subclass is accepted, which the contract allows.
- Critic accuracy: accurate. The PR test is input-sensitive but calls the function directly, not via the gate. Residuals: measure_expression_overhead relies on the constructor for the identity check, and the regression episodes have already run when a bad count aborts. Nothing is persisted.

### L6 (still open, broader than described)
- With aggregate pass True, add_bundle accepts and verify() returns [] for stage_pass = 1, 'true', 'false', [0], 0.5, {'x':1}. The string 'false' is accepted for a passing stage, which inverts its meaning.
- With aggregate pass False, the falsy values 0, '', None, [] are accepted.
- Critic imprecision: (a) the Critic mentions only truthy values. (b) The same looseness is upstream at `gate.py:581`, so a fix confined to records.py leaves preflight accepting stage_pass=1.
- The schema leaves child measurements untyped (`genome/schema/eag-0.1.cig.schema.json:71`). No test covers non-boolean stage_pass.
- Repro: `repro_records.py` and `repro_chain.py` under `research/reviews/repro/claims-latest-1-3+6-7/`. Result: live stored replay stage_pass 1, verify() == [].

### L7 (still open)
- Each of these passes preflight and the run makes 80 stage-3/4 calls: temperature=False, temperature=0 (int), temperature=-0.0, max_tokens=64.0, seed=11.0, extra key top_p=0.9. Same for replay and ablation children.
- Already rejected: a missing key (.get gives None), stop=() tuple, a non-dict settings value.
- Critic imprecision: (1) "missing field fails" is already met, so only the extra-key half is open. (2) "numeric finite zero" would still allow int 0 and -0.0, so the requirement should say float 0.0. (3) Not flagged by the Critic: the child-level seed check at `gate.py:573` has the same `==` looseness. A float child seed passes preflight and fails only in add_bundle after 80 runner calls (RecordConsistencyError, incoherent seeds ['11','11.0']), breaking the fail-before-any-stage-3/4 contract.

### G1-G9 (all hold)
- G1 version-2 envelope commits to (a,2), (a,1) stays candidate. G2 duplicate parents rejected with 0 calls. G3 add_bundle/verify invariants match, with white-box parity tests. G4 wrong ablation id, seed 99 and load_cig failures rejected with 0 calls. G5 stage-4 children pass load_cig without stores, per-candidate seeds persisted. G6 both-arm skills in canonical order, regulation preserved. G7 canonical evidence identical in both children and in the live store. G8 per-cell composition computed (see L4). G9 bad counter rejected at construction, vacuous reason constant used.
- The c6ce4c1 diff only adds checks and removes no earlier guard. The real #11 producers write `stop` as a list (replay.py:298-302, ablation.py:322-326), so the exact `stop == []` check does not break real evidence.
- Critic imprecision: G5's "no load_cig without stores" described the old state. G9's "validate_token_counter is not called" is no longer true, but a None counter is still caught late (see 4.1 item 6).

## 4. New confirmed findings

Only findings with verdict.real=true after a second verification pass. Severity is the verifier's assessment, which in several cases is lower than the original claim.

### 4.1 PR #28 (c6ce4c1)

**1. MAJOR: sensitivity tau_r=0.05 cell is float-fragile** (`inheritance/regression.py:231`; compared at `inheritance/gate.py:436,456`)
- Failure: `1.0 - 38/40 = 0.050000000000000044 > 0.05`, so a candidate with exact regression 2/40 = 0.05 is rejected in the tau_r=0.05 cell. Over all 41 T_reg outcomes, the tau_r=0.05 cell equals the tau_r=0.03 cell. The persisted stage-4 child also stores the 0.050000000000000044 value. The primary rule is unaffected (39/40 passes, 38/40 is correctly rejected at 0.03).
- Spec: the inclusive `<=` comes from EXP-0001-design.md:55 and the manifest. SPEC.md:400 writes strict `<`, but at 38/40 the grid is wrong even under strict `<`.
- Repro: `cd pr28c && PYTHONPATH=. ../venv314/bin/python <scratch>/out/repro/ (copy: research/reviews/repro/)c6-decision-correctness/r1_tau_r_float_boundary.py`
  ```
  primary verdict: reject
  persisted regression_drop: 0.050000000000000044  exact = 2/40 = 0.05
  grid regression flags: {'tau_r=0.02': False, 'tau_r=0.03': False, 'tau_r=0.05': False}
  exact rational check 1 - 38/40 <= 0.05: True
  ```
- Fix: use integer or Fraction arithmetic (`failures = 40 - successes` against `floor(tau*40)`), and persist successes and total.
- Side effect: once fixed, the not_identifiable branch becomes reachable from real data (see refuted item R2).

**2. MAJOR: somatic decision is not bound to the evaluated artifact** (`inheritance/gate.py:743`, `_commit_results`)
- Failure: the commit only checks `somatic_store.state(...) == "candidate"`. It never compares `somatic_store.candidate_ref(gene_id, version)` with the evaluated `candidate_envelope`. The issue #12 handoff section 13 requires "same gene/version/artifact as the evaluated candidate".
- Scenario: the gate registry has a@1 = P1 (evaluated). The somatic store is bound to a different registry where a@1 = P2. The run promotes and marks P2 as validated with gate_reports [CIG-0007].
- Limits: needs a wiring mismatch between registry and store, so the likelihood is low. The aggregate not carrying a version or digest is not itself a defect; the missing commit-time guard is.
- Repro: `r3_somatic_decision_not_bound_to_evaluated_artifact.py` (`research/reviews/repro/c6-state-contracts-tests/`)
  ```
  evaluated artifact : registry://skills/a@1/sha256:3f10fe97...
  validated artifact : registry://skills/a@1/sha256:dde5a723... {'state': 'validated', 'gate_reports': ['CIG-0007']}
  ```
- Fix: before decide(), compare candidate_ref against the evaluated envelope ref and fail closed. Optionally check that the store's registry is the gate's registry in `__init__`.

**3. MINOR: L6 and L7 (open Critic items)**: see section 3. These are the blocking pair.

**4. MINOR: ablation numbers consumed by the gate are not bound to the durable #11 child** (`inheritance/gate.py:336,416`)
- `delta_success` and `bootstrap['lower_bound']` are never compared with the child's `point_estimate_delta_success` and lower_bound. A forged AblationEvaluation (`_delta_success=0.95` vs child 0.1) flips an elimination and commits with verify() clean.
- Verdict status: one verification pass rated this real but minor hardening. A separate pass classed it as a trusted-boundary hardening gap and not a defect (R1 in section 5), because the only producer writes both values from the same bootstrap object and the handoff does not list this check. Treat as optional defense-in-depth.
- Repro: `r1_ablation_numbers_not_bound_to_child.py`.
- Fix: assert equality of delta, lower bound and passed against the child before consuming them.

**5. MINOR: batch commit is not atomic when a later decide() fails** (`inheritance/gate.py:750`)
- The docstring promises "Fail-atomic BATCH commit. Any failure leaves both live stores untouched". The loop does decide() then add_bundle() per candidate with no rollback. With ENOSPC on b's journal append: live CIG holds CIG-0007 plus 5 children, somatic a validated, b candidate. A retry re-runs 180 episodes and fails with "aggregate 'CIG-0007' already exists".
- Why minor: the handoff scopes atomicity to the in-process commit and mandates "somatic first, then expose CIG". It also forbids changing a terminal decision afterwards. The valid description is that the docstring overclaims (SPEC §34).
- Repro: `r2_partial_batch_commit_on_decide_fault.py`.
- Fix: soften the docstring to state what is guaranteed, and document that journal faults can leave a partial commit.

**6. MINOR: setup errors detected only after episodes are spent** (`inheritance/gate.py:280,736`)
- A None token_counter is raised after 80 runner calls. A pre-existing aggregate id in the live CigRecordStore is raised after 80 new runner calls. The GateInputError docstring says "before any episode".
- Partly not a defect: the missing interaction_allocation (gate.py:314-317) needs eligibility known only after stages 3/4, so a blanket `len(inputs) >= 2` pre-check would over-reject. Keep that late.
- Repro: `r4_late_setup_failures.py`, `r7_setup_errors_after_episodes.py`.
- Fix: check the counter and existing parent ids at the start of run().

**7. MINOR: overhead leakage guard matches English substrings** (`inheritance/regression.py:295`)
- A principle such as "Check the provenance label first." raises RunnerContractError after 160 runner calls and aborts the whole batch, including the valid candidate. Projection already removes the fields structurally.
- Why minor: handoff section 16 lists the no-evaluation-only-metadata test as blocking, and tests/inheritance/test_regression_overhead.py:160-168 asserts the substring behaviour. It is a design trade-off, not a spec violation.
- Repro: `r8_overhead_substring_false_positive.py`.
- Fix: check the projected structure instead of substrings, or run the check before any episodes.

**8. MINOR: stage 4C uses the batch seed, not per-candidate seeds** (`inheritance/gate.py:324`)
- A stage-4 child with seed 23 embeds interaction locked_settings.seed 11. verify() is clean. GateOutcome.mining_seed reports only the batch default.
- Why minor: no document specifies a pair seed. A pair shared by two aggregates has no single correct seed, the recorded value is truthful, and at temperature 0 the seed does not change outcomes. The residual inconsistency is real when all candidates carry the same per-candidate seed and the batch default differs (the constructor default is 0).
- Repro: `r4_interaction_ignores_per_candidate_seed.py`.
- Fix: record the rule, or fail closed when candidate seeds differ from the pair seed.

**9. MINOR: stage-4 pair evidence is incomplete, but chain semantics are not a defect** (`inheritance/gate.py:520,626`)
- Verified: `retained` is stored per candidate, not per pair, so b's child marks (b,c) retained=False although b won that pair, and a non-harmful pair can carry retained=False. `eliminated_by` is never written to a child, and for several losses it is just the last pair key in sorted order. Handoff section 10 asks for harmful-pair resolutions to be recorded.
- Not a defect (downgraded from major): for a>b>c with harmful (a,b) and (b,c), the gate eliminates b and c and promotes a. That follows the written "resolve all harmful pairs simultaneously" rule (handoff section 7 and 14, ADR-0002 A1). A cascade-aware policy that keeps {a, c} would be a different, unstated policy. This is a design question for the owner, relevant at K=10, and no N>=3 chain test exists.
- Repro: `r2_chain_elimination.py`.
- Fix: store per-pair winner/loser and eliminated_by in the child. Add an N>=3 chain test that documents the chosen policy.

**10. MINOR: sensitivity_recomputation reads aliased or stale run state** (`inheritance/gate.py:368,406-417`)
- `_LastRun` keeps the caller's inputs list by reference. After `inputs.reverse()` the replay hypothetical_pass for a flips from True to False. After an aborted second run, sensitivity answers from the first run with no staleness signal. `_last` is also assigned before the commit.
- Repro: `r8_sensitivity_reads_stale_or_aliased_run.py`.
- Fix: snapshot with `tuple(inputs)`, clear `_last` at the start of run(), and stamp a run identity into the grid.

### 4.2 Merged main (M0 statistics, 0ce0a36)

**M1. MINOR (design/prereg note): tau_c is never binding at n=20** (`inheritance/bootstrap.py:66`, `inheritance/ablation.py:268-271`)
- Over all 231 compositions of d_i in {-1,0,+1}, 54 pass for every tau_c in [0, 0.20]. The smallest passing result is +4/20 with no harms (point 0.20, LB 0.05). The pre-registered grid {0.03, 0.05, 0.08} changes 0/231 decisions. Seeds 0-4 give the same minimum.
- This is a property of the locked prereg, not a code defect. Do not retune. Add a post-hoc, labelled note that at n=20 the stage-2 rule reduces to LB>0 and the tau_c grid is vacuous for stage 2. Do not describe the tau_c sensitivity analysis as supporting robustness (SPEC §34).
- Repro: `cd /home/user/heritagent && PYTHONPATH=. ../venv314/bin/python research/reviews/repro/verify-m0-1/v1.py` (under the scratchpad).

**M2. MINOR (prereg hygiene): bootstrap seed and task order are not locked** (`inheritance/ablation.py:188,204`)
- For pos=10/neg=3 the locked rule passes for 61/200 seeds. Through the evaluator, bootstrap_seed=1 gives LB 0.05 and PASS and the default seed 0 gives LB 0.00 and FAIL, both schema-valid with the same id. The default is BOOTSTRAP_SEED = 0 (ablation.py:42) and it is recorded in the child. No non-test caller passes another seed.
- The task-order variant (4/8 orderings pass) was not separately re-run by the verifier.
- Fix: pin the seed and a canonical order in the manifest or ADR-0002, or compute the exact distribution.

**M3. MINOR: bootstrap inputs not validated** (`inheritance/bootstrap.py:48`)
- seed=None seeds from OS entropy (8 runs gave T,T,F,F,F,T,F,F). NaN and out-of-range deltas are accepted via the public sensitivity_analysis (analysis only, no verdict). `runner.py:51` already validates mining_seed but not bootstrap_seed.
- Fix: require non-bool int bootstrap_seed and finite deltas in [-1, 1].

**M4. MINOR: validate_outcome ignores status** (`inheritance/runner.py:184`)
- Six control-arm episodes with status='adapter_error', success=False give point 0.3, LB 0.1, stage_pass=True for a trait with no effect. Reachability is hypothetical (no real adapter exists) and the prereg defines no error policy.
- Fix: add a status allowlist (finished, budget_exhausted), or add a prereg-level error policy before a real runner lands.

### 4.3 Additional observations without an independent verdict pass
These came from the overlapping lenses. They have not been second-pass verified, so treat them as leads.
- Test adequacy (reported as major by its lens): three verdict-flipping mutations leave all 438 tests passing. M1 ignores ablation contribution in harmful-pair resolution (`gate.py:526`). M4 removes the tau_k overhead gate (`gate.py:632`). M5 removes generalization from the preliminary eligible set (`gate.py:306`). Every harmful-pair test puts the higher contribution on the earlier rank. Scripts: `research/reviews/repro/c6-state-contracts-tests/run_mutations.sh`.
- Further test gaps: the ablation child is never corrupted in the preflight tests (dropping the whole ablation loop passes 438). Passing target_regulation=None to the generalization/regression/interaction evaluators passes. test_atomic_bundle_failure_leaves_stores_unchanged (test_gate_atomic_commit.py:143) and an `is not None` assertion at test_gate_orchestrator.py:327 are vacuous.
- verify() parity: checks dict keys, not record['cig_id']. It never re-runs load_cig and uses truthiness for passes (`genome/validation/records.py:105,130,235`).
- A false vacuous reason is written into the stage-4 child of a non-eligible candidate when two or more others were eligible (`gate.py:628`).
- target_regulation_by_gene is an unrecorded side input to stages 3/4 and is not bound to the #11 evidence (`gate.py:237`).

## 5. Refuted or unconfirmed candidates (do not re-raise)

- R1: "Ablation delta/LB not cross-checked against the child" as a major evidence-integrity bug. The only producer writes both from one bootstrap object, and reaching the state needs a hand-forged frozen object. #12 handoff §2 does not list it. At most optional hardening (see 4.1 item 4).
- R2: "Locked thresholds collapse at the real n (out-of-scope -0.05 equals >=0, tau_c 0.03 equals 0.05)". True arithmetic, but thresholds are locked by handoff §4 and ADR-0002. The "not_identifiable unreachable" part duplicates the tau_r float finding. Only a reporting note is warranted.
- R3: "Interaction uses the batch seed instead of candidate seeds" as a spec violation. No spec requires a pair seed, and the pair is shared evidence. Kept only as the minor provenance item 8.
- R4: "Replay expression check is presence-only for constitutive traits". Real mechanically, but the prereg defines "fire" by regulation. Behavioural contribution is tested by replay delta and ablation. Do not write "the skill was used" for expression_check.passed.
- R5: "Ablation allocation neither checks nor records task provenance". Not required by the prereg, and a label check would be self-referential bookkeeping. Optional audit-trail improvement only. "Contamination guard not enforced" would overclaim.

## 6. Test results

- PR head c6ce4c1, Python 3.14.6: 438 passed.
- PR head c6ce4c1, Python 3.13: 435 passed + 1 collection error (`tests/somatic/test_store_replay_parity.py:63` NameError: Path, same as main).
- Portability matrix (passed/failed/errors):

| Python | main 0ce0a36 | PR #28 at older head 4ca5517 (not c6ce4c1) |
|---|---|---|
| 3.10.20 | 365 / 0 / 1 | 402 / 0 / 1 |
| 3.11.17 | 365 / 0 / 1 | 402 / 0 / 1 |
| 3.12.3 | 365 / 0 / 1 | 402 / 0 / 1 |
| 3.13.16 | 365 / 0 / 1 | 402 / 0 / 1 |
| 3.14.6 | 368 / 0 / 0 | 405 / 0 / 0 |

  The PR column was measured at 4ca5517 and was not re-run across all versions at c6ce4c1. The c6ce4c1 numbers above are the two Python runs.
- Root cause of the error: the file annotates with `Path` but never imports pathlib.Path. Python 3.14 masks it through PEP 649 deferred annotations. Minimal fix: add `from pathlib import Path` after `import json` (line 6). Verified on scratch copies: main 368 passed and PR 405 passed (PR checked on 3.10, 3.13, 3.14 only).
- Latent annotation issues, masked by `from __future__ import annotations`: `inheritance/ablation.py:54` and `replay.py:59,60,96` (EvaluationTask), `inheritance/gate.py:199` (GateEpisodeRunner, PR only), `tests/miner/conftest.py:76` ('MiningPair').
- CI attestation: none. No `.github/` directory on main or the PR worktree. list_workflows returns 0, and the PR has 0 check runs. Issue #18 (CI) is open and unassigned. The suite is not gated on any Python version.
- M0 statistics audit: `research/reviews/M0-statistics-audit.md`. Inheritance tests 56/56 on 3.14. The bootstrap matches the locked prereg and is deterministic across 3.11-3.14. The replay rule agrees with an independent reading in 2048/2048 cases.
- Portability detail: `research/reviews/test-portability.md`.

## 7. Suggested next patch scope

### Blocking for #12 merge
1. Close L6: require `isinstance(x, bool)` for stage_pass at `genome/validation/records.py:130` (add_bundle and verify) and at `inheritance/gate.py:581`. Add tests for 1, 'true', 'false', 0, None.
2. Close L7: require an exact key set for eval_generation_settings (reject extras). Require type-exact values at `gate.py:599`: float 0.0 temperature, int non-bool max_tokens and seed, list stop. Apply the same type check to the child-level seed at `gate.py:573`. Corrupt the ablation child in the tests too.
3. Bind the somatic commit to the evaluated artifact (`gate.py:743`), as the handoff requires ("same gene/version/artifact"). The Critic should confirm whether this is a merge gate. I rate it major but low-likelihood.
4. Fix the tau_r float comparison (`regression.py:231`, `gate.py:436,456`) with integer or Fraction arithmetic. The primary decision is unaffected, so the Critic may accept this as follow-up if the sensitivity grid is not yet reported. It must be fixed before the grid is reported (SPEC §34).

### Non-blocking / follow-up issue
- Add the missing-import fix in `tests/somatic/test_store_replay_parity.py` and the three latent annotation imports. Wire a minimal CI over Python 3.10-3.14 (issue #18).
- Add tests for the three surviving mutants, the ablation preflight, regulation wiring, an N>=3 chain, and the two vacuous assertions.
- Persist per-pair resolution and eliminated_by in the stage-4 child. The owner should decide the chain policy (simultaneous vs cascade).
- Soften the batch-commit atomicity docstring. Move the counter and existing-parent-id checks to the start of run().
- Replace the substring overhead guard with a structural check. Define the pair-seed rule.
- Snapshot inputs and clear `_last` in the sensitivity API.
- Optional hardening: bind ablation delta/LB to the child. Reject allocations with null provenance. Record target regulation. Make verify() check record ids and re-run load_cig.
- M0 (prereg notes, no code retune): record that tau_c is vacuous at n=20. Pin the bootstrap seed and task order, or compute the exact distribution. Validate bootstrap_seed and delta finiteness. Add a status policy before a real runner lands.
