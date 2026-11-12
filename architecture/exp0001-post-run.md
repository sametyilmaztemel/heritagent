# EXP-0001 after the run: procedure (PROPOSED)

**Status: PROPOSED.** A draft by builder-2 for the owner, written before any EXP-0001 episode has run. It is not in force, changes no protocol value, and is not part of any ADR. Two points are the owner's decision (takeover note §5, item 25): where the run's records are archived, and who writes the post-run review. Status keys as in `CLAUDE.md`.

It covers SPEC §33's post-run review, the SPEC §24 record, and the archive of the run's files. The run itself is in `architecture/exp0001-colab-operations.md`.

## 1. Who does what after the last cell

1. **The owner** runs the run notebook to its end. Its last cells copy `report.json` to `heritagent/runs/<RUN_ID>/report.json` on Drive, draw the figures into `heritagent/runs/<RUN_ID>/figures/`, and release the runtime (R, `notebooks/EXP-0001-run.ipynb`). The mirror stays in `heritagent/runs/<RUN_ID>/mirror/`. The owner sends `report.json`, as the notebook's last cell asks; builder-2 reads the Drive copies itself only when a Drive connector is attached (U).
2. **The archive** (section 2) is made before anything else is done with the results, by the owner or by builder-2 with the owner's files.
3. **builder-2** checks the SPEC §24 fields (section 3) against the archived files, and fills sections B to F of the review (section 4), whose section A was filled before the run.
4. **builder-2, in its Critic role,** has the filled review checked on a model different from the author's (`CLAUDE.md`, Critic duties) and posts the review and that check on issue #13.
5. **The owner** decides: accepts the review, signs it, and decides any change to the hypotheses file or the novelty ledger (section 5) and whether #13 closes. Whether M0's minimum criterion (SPEC §29) is met is the owner's judgement on the review, not a field of the report.

## 2. What is archived, and how its hashes are recorded

**The files** (R, the run's code at this commit):
- the run directory as the mirror's last whole snapshot holds it: `run-start.json`, `ledger.jsonl`, `report.json`, `t_reg.json`, `seed-<s>/` for each mining seed (`candidates.json` with the mining records and rejections, `somatic.jsonl`, `gate.json`, `arms.json`), `registry/` and `trajectories/`;
- the figures directory (`figures.json`, `captions.md`, each figure's PDF, PNG and CSV);
- the run rehearsal's record the run started with (`heritagent/rehearsal/run-passed-<COMMIT>-<model>.json`) and the model revision file (`heritagent/model-revision-<model>.txt`).

**The hashes.** The last snapshot's manifest (`snapshot-<seq>.json` in the mirror) already lists the sha256 and size of every file of the run directory, and every reader checks them (V, `evaluation/exp0001/mirror.py`). The record kept in the repository is therefore small: the run id, the final snapshot's number and the sha256 of its manifest file, the sha256 of `report.json` and `run-start.json` (equal to the manifest's entries), and the sha256 of each figures file and of the rehearsal record (computed with `sha256sum`). Its place in the repository is part of the owner's decision; it is not in `experiments/manifests/`, which is locked.

**Where the files go** (the owner's decision A5; builder-2's recommendation is option b, S):
- (a) **Drive.** The Drive folder `heritagent/runs/<RUN_ID>/` is kept as it is, and the hash record goes into the repository. It costs nothing more, but the files stay in one account, and the mirror holds compressed blobs that need this repository's code to read (`figures --mirror-dir`, a restore).
- (b) **A GitHub Release.** The files above, restored from the last snapshot into a plain directory (the `restore` function of `evaluation/exp0001/mirror.py`, at the run's commit) and packed as one archive, are attached to a release of the repository's commit; the archive's sha256 goes into the hash record. A restore appends a `restore` note to the ledger it rebuilds, so the archive's `ledger.jsonl` differs from the snapshot's by that last line; the record names both hashes.

**The calibration pilot's report** is the earliest record this rule applies to. `pilot-report.json` is on the owner's Drive, in the pilot's folder under `heritagent/pilot/` (named by its commit and model), and its numbers are in issue #13; it is committed nowhere (R). Its archive and hash record follow the same option.

## 3. The SPEC §24 record: where each field is

SPEC §24: results are never reported without the seed, the model revision, the genome revision, the dataset split and the code commit. In EXP-0001 they are in these keys (V, `evaluation/exp0001/provenance.py`, `pipeline.py`, `evaluation/metrics/final_statistics.py` at this commit):

| SPEC §24 field | `run-start.json` | `report.json` |
|---|---|---|
| seeds | `protocol.mining_seeds` | `statistics.mining_seeds`; `statistics.seed_consistency.per_seed` |
| model revision | `model.model_id`, `model.revision`, `model.tokenizer`, `model.backend_version`; `teacher.*`; `weights.sha256` | — |
| genome revision | `g0.genome_id`, `g0.genome_sha256` | `final_genomes` (each genome id and the arms that read it); per seed in `seed-<s>/arms.json` |
| dataset split | `splits.manifest_sha256`, `splits.pools`; `pilot.pilot_sha256`; `benefit_bank.bank_sha256` | `statistics.t_eval.task_count`, `statistics.t_reg.task_ids` (the draw in `t_reg.json`) |
| code commit | `code.commit` (`code.dirty` is always false) | — |

`run-start.json` also holds its own `record_sha256`, the protocol, the gate thresholds, the miner's prompt and schema hashes and the installed software (`environment`). `report.json` holds the outcome: `status` (`complete` or `incomplete`), `stage`, `reason`, the counts, `ledger_notes` (the restores and repairs up to the report; the ledger holds any later ones), `gates`, `statistics` (partial or null for an incomplete run) and the descriptive measures. A report of the run is quoted only together with the `run-start.json` of the same directory.

## 4. The post-run review template

One file per run, posted on issue #13. Section A is filled and posted **before the run starts**, with the design's expectations quoted verbatim; sections B to F are filled after it. The headings are SPEC §33's six.

> **A. Expected result** (filled before the run; quoted from `experiments/manifests/EXP-0001-design.md` §8, locked 2026-10-14):
>
> The values below are EXPECTATIONS — directional forecasts recorded for post-run review against SPEC §33. They are **not** success thresholds and must never be used to retune anything mid-run.
>
> - [x] **Expected result, H2 (direction + rough magnitude):** CIG-gated arm C > unrestricted arm B on held-out `T_eval`, with lower regression. Expected `C−B` held-out success: approximately **+5 to +10 percentage points**, central expectation approximately **+7pp**. Regression rate for C expected approximately **3–8pp lower** than B. These are expectations, NOT success thresholds.
> - [x] **Expected result, H1:** arm C shows positive held-out improvement over the no-inheritance arm A. Expected `C−A`: approximately **+5 to +15 percentage points**, central expectation approximately **+10pp**. Expectation, NOT a success threshold — the H1 support criterion stays `C−A > 0` with bootstrap 95% CI excluding 0.
> - [x] **Expected number of candidates passing each gate stage** — a **rough expectation per mining seed, not a quota** (frozen candidate set: K = 10 per seed):
>   - Stage 1 — Replay: ~**6 / 10**;
>   - Stage 2 — Ablation: ~**4 / 10**;
>   - Stage 3 — Generalization (combined): ~**4 / 10** (audit detail: in-scope pass ~5–6/10; out-of-scope harmlessness ~7–9/10; combined stage-3 pass ~4/10);
>   - Stage 4 — Interaction + regression + overhead: ~**3 / 10** (most candidates are expected to pass the token-overhead check alone; attrition is primarily expected from the 40-task regression stage and occasional harmful-pair interaction);
>   - Final promoted into arm C: ≈ **3 / 10**.
>
>   *Dated note (ADR-0012 D11, owner, 2026-10-19):* these expectations stay as recorded, written for Qwen2.5-7B and the locked gate. ADR-0012's projections for the recalibrated gate (M, not decision rules; per seed 3 traits at +15 pp in all families, 2 at +25 pp in two, 5 with no effect; δ = 0 / 0.05 / 0.18): helpful traits promoted 4.71 / 4.02 / 2.37 of 5; no-effect traits promoted 0 / 0.0034 / 0.0115 of 15; arm C empty in a seed ≈ 1e-6 / 3e-4 / 0.049.
>
> The decision rules (the same section, quoting the design's §5):
>
> - H2 supported iff: `C−B ≥ +5pp` on T_eval, bootstrap 95% CI excluding 0, AND regression rate(C) < regression rate(B) with the regression-difference CI excluding 0, AND the C−B direction holds in ≥ 2/3 mining seeds;
> - H1 supported iff: `C−A > 0` with bootstrap 95% CI excluding 0.

**B. Actual result.** The run's `status`, `stage` and `reason`; the counts against the caps; per seed, the candidates that passed each stage and the arm-C composition (`gate.json`, `arms.json`); the point estimates of C−B, C−A and reg(C)−reg(B) (`statistics`). Each figure with its status key and the file it comes from.

**C. Statistical significance.** The pre-registered rule outcomes as the report states them (`statistics.h2`, `statistics.h1` with their clauses, the bootstrap intervals, `statistics.seed_consistency`), quoted with the report's `claim_status`. Nothing is recomputed here with another rule; the sensitivity grid is secondary and descriptive (the design's §5).

**D. Unexpected behaviour.** Anything the run did that the design and the ADRs did not foresee: restores and repairs (`ledger_notes`) and the work they replayed, refused requests, context overflows and reduced evidence forms (ADR-0009 D8), malformed teacher replies and their `finish_reason`, teacher replies with more than five proposals (counted from each mining record's raw reply; ADR-0011 note N1), episodes that used all 50 steps (status `environment_terminal`, `won` false, 50 steps), episodes that ended on retries (the event names the budget `max_total_retries` in both cases: 11 retries is the run's total budget, fewer is the retry gene's cap of 3 per call), snapshots a restore passed over (the restore notes' `passed_over`; the accounting table shows only the restored snapshot's number), sessions the mirror paused (from the run cell's output: the run's records do not say why a session ended), the backend-noise measures (`cell_disagreement`, ADR-0012 D8).

**E. Confounders.** The design's §6 threats, each with what the records show; the owner's decisions of 2026-11-12 (takeover note §5, items 21 and 22): the G0 prompt's history was kept as the observations only (ADR-0003 note N1; the design's §6 item 7), and the teacher's per-call request became at most 5 proposals, with K = 10 and the 2,048-token budget kept (ADR-0011 note N1; the mining records show a reply with more than five proposals or a reply cut at the budget); differences from the pilot's machine or software (`run-start.json` `environment`). The review also states these limits of what the three arms can show, each as unmeasured (the Critic's triage of 2026-11-12; takeover note §5):
- **What C−B separates.** The experiment has no arm with another filter and no size-matched random subset. A difference between C and B therefore does not separate this gate from any other selection of fewer skills, or from a shorter prompt (the design's §3 counts arm B's larger prompt as part of the treatment).
- **The H2 regression clause is in sample.** It is measured on T_reg, the suite stage 4A used to choose arm C's traits (ADR-0006 D4), and T_reg is drawn from T_mine. The report's `h2.regression_clause.scope` names the suite; the reading is in the PROPOSED notes N4 and N5 (`research/proposals/ADR-0006-notes-proposal.md`) until the owner decides them.
- **Arm C equal to G0.** In a seed where the gate promoted nothing, C−B is A−B. The H2 rule outcome then compares no inheritance with unrestricted inheritance, with no gated trait in arm C (`final_genomes` shows which arms share a genome; ADR-0012 gives the probability, M).
- **What C−A mixes.** A skill text changes the prompt as well as its content, and there is no placebo or length-matched control (ADR-0012 Q9 declined a placebo candidate for the gate).
- **Not adopted from the earlier advisory list** (2026-10-13): a joint check of G0 with all admitted traits before G1 (stage 4C checks pairs), a per-candidate held-out contribution and its rank correlation with stage 2, a family- or scene-blocked bootstrap as a secondary analysis, a random-accept control matched on the acceptance rate.

**F. Follow-up.** Proposed next steps, each as an issue or an ADR draft for the owner; any locked change it would need is named as such.

## 5. What never happens at this step

- No result is called validated (SPEC §32). The report's `claim_status` says the same: "pre-registered rule outcome only; not a validated result (SPEC section 32) and not a causal claim (SPEC section 34)". The review uses no word SPEC §34 reserves for evidence beyond what the rule outcomes support, and a null or negative outcome is recorded as a result (the design's §8).
- The expectations of section A are never used to retune anything, and no threshold, seed, cap or allocation changes under the EXP-0001 identity (the design's §8, "Pre-registration immutability").
- `research/hypotheses/` and `research/novelty-ledger.md` change only by the owner's decision, through a reviewed PR; builder-2 drafts the text.
- The run's records are never edited. A correction is a new record with its own hash.
