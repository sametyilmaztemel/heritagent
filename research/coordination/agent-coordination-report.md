# HeritAgent inter-agent coordination health report

Author: independent researcher subagent (third agent, Claude). Read-only analysis. Nothing was posted to GitHub and /home/user/heritagent was not modified.
Data cut: ledger snapshot (173 comments, issues/PRs 1-28) plus a live read at 2026-10-12 ~21:40Z that found 3 newer comments (see section 0). All times UTC.

Script: `scripts/coordination_metrics.py` (outputs `metrics.json`, `metrics_tables.md`). Raw JSON: `raw/`. Ledger: `comms-ledger.jsonl`.

## 0. Data provenance, freshness, limits

- All 173 ledger comments (and every GitHub comment I read live) come from ONE login, `sametyilmaztemel`. Role (builder / critic) is a content heuristic: builder = "## Builder ..." headers with commit SHA + test counts; critic = "## Research/Critic ..." headers with blocker lists. Agent identity (Zcode / Codex) is NOT recoverable from GitHub data (section 8).
- **Ledger is stale by 3 comments.** Live read found: builder report on PR #28 at 21:32:48Z (head `c6ce4c1`, "438 passed", local pytest, re-review requested); critic re-review on PR #28 at 21:35:14Z (head `c6ce4c1`: gate still CLOSED on 2 type-strict blockers); #20 coordination note at 21:35:19Z. PR #28 head is therefore `c6ce4c1` (not 4ca5517 / 7886852). The mirrored issue #12 copy of the 21:35 critic review was not fetched (unverified). `coordination_metrics.py` merges the first two as SUPPLEMENT rows.
- Zero comment edits in the snapshot (updated_at == created_at for all 173). Deleted comments would be invisible; see PR #26 anomaly (section 5).
- PR open/merge times come from live `pull_request_read` calls. All merges are `merged_by: sametyilmaztemel`.
- Category tags in section 4 are my manual reading of the review headings (all critic reviews were read). The regex category counts in `metrics.json` are rough and should not be quoted.
- Latency stats are wall-clock and include nights/weekend (2026-10-10 is a Saturday). They measure the agent+human relay system, not model speed.

## 1. Executive summary

1. **The Builder<->Critic loop works and converges per PR.** 8 of 8 merged PRs needed critic-gated rework (0 first-pass merges); median 3 blocking rounds (21 total incl. 1 invisible, 20 visible), then a explicit "FINAL CLEAR" and merge within 3-22 s. Items per round fall strictly in 8 of 8 merged PRs (e.g. #27: 6 -> 3 -> 2; #25: 6 -> 3).
2. **PR #28 (issue #12, CIG stage 3/4) is the outlier and the live risk.** Largest PR (3,805 additions), 5 critic rounds so far with items 8 -> 10 -> 9 -> 5 -> 2, open >5.8 h, merge gate CLOSED on two type-strict items at head `c6ce4c1`. Single invariants have now been re-opened 4-5 times at ever finer strictness (`stage_pass`: "enforced" -> "present" -> "boolean"; settings: "present" -> "value" -> "typed"). This is closer to convergence than creep, but acceptance criteria are not frozen up front, so "done" is defined retroactively.
3. **Dominant Builder blind spot: written/recorded-but-not-enforced, and write-path vs replay/verify-path parity.** Recurs in 7 of 9 PRs (schema invariants in prose only, `max_plan_steps` compiled but ignored, `finalize()` not running the schema check, `add_bundle()` not enforcing what its docstring says, settings recorded but not validated). Second: "frozen" objects that are still mutable (6 PRs).
4. **Protocol drift against issue #20 and SPEC §31:** 31% of all ledger comments (54/173) are status mirrors on #20 despite its rule "technical discussion stays on the PR/issue"; most events are posted 2-3 times; PR #26's first critic review exists on no GitHub thread (the builder answered "four blocking groups" with no visible review), i.e. repo-as-truth was bypassed once; in PR #28 builder reports dropped the explicit "unresolved questions" line (1 of 4 reports has it vs 100% before).
5. **Structural risks (ranked):** (a) no CI anywhere, every test count is builder-reported local pytest on Python 3.14.5 only (pyproject says `>=3.10`); (b) one identity, so no formal REQUEST_CHANGES and the "clear" comment and the merge land within seconds from the same account, no separate human or independent confirmation step is visible; (c) the critic also acts as work scheduler and author of acceptance criteria (19 KB handoff for #12), so builder is graded against the critic's own checklist; (d) the pre-registered EXP-0001 manifest was edited after lock (PR #21) and ADR-0002 Amendment A1 (PR #28) was not mirrored into the machine-readable manifest.
6. **Third-agent entry (section 9):** join as an advisory, non-gating reader; one tagged comment header; findings go to the Critic via a dedicated thread, never to the Builder and never into the gate language; no pushes to existing branches; no edits to locked ADR/manifest/SPEC files. Zero extra comments on PR threads during an active fix cycle.

## 2. Timeline (UTC, compact)

| When | Event |
|---|---|
| 10-02 18:24 | PR #1 (ADR-0001 EAG) and #2 (ADR-0002 EXP-0001 design) opened; issue #20 coordination thread created 18:38 |
| 10-02 18:38-19:16 | Critic R1 (4 / 5 blockers), R2 (2 / 3), R3 (1 / 1), FINAL CLEAR 19:16; PR #1 and #2 merged 19:16:40 / :53 |
| 10-02 19:17 | Critic posts "#4 START" handoff (critic as scheduler) |
| 10-02 19:45-21:01 | PR #21 (EAG schema, issue #4): 2 rounds, merged 21:01:36 |
| 10-02 21:02 -> 10-03 16:40 | "#7 START"; PR #23 (runtime): 2 rounds (4, 3), merged 10-03 16:40 |
| 10-03 16:42 -> 10-04 10:39 | "#8 START"; PR #24 (trajectory recorder): 3 rounds (4, 3, 2), merged 10-04 10:39 |
| 10-04 10:46 -> 10-07 16:21 | "#9 START"; PR #25 (trait miner): 2 rounds (6, 3); 49.5 h critic gap before FINAL CLEAR |
| 10-07 16:23 -> 10-08 12:33 | "#10 START"; PR #26 (somatic store): 1 invisible + 2 visible rounds, merged 10-08 12:33 |
| 10-08 12:37 -> 10-12 14:20 | "#11 START"; PR #27 (replay/ablation): 3 rounds (6, 3, 2), 50 h builder gap over the weekend, merged 10-12 14:20 |
| 10-12 14:26 | "#12 START" (19 KB handoff incl. ADR-0002 amendment instruction) |
| 10-12 15:47 | PR #28 opened, 406 tests reported |
| 10-12 16:08 / 16:23 / 16:26 | Critic R1 (8 blockers) -> Builder partial ("what remains" list) -> Critic residual checklist (10) |
| 10-12 18:29 / 18:47 | Builder `4ca5517` (405 tests, down 1) -> Critic: generalization closed, 9 remain |
| 10-12 21:14 / 21:18 | Builder `7886852` (428) -> Critic: 5 narrow blockers |
| 10-12 21:32 / 21:35 | Builder `c6ce4c1` (438) -> Critic: 2 type-strict blockers; gate CLOSED (state at time of analysis) |

## 3. Metrics

Per PR (script output; rounds = critic blocking comments on the PR thread; tests = builder-reported, local, not CI-attested; handoff count is first value).

| PR | issue | open->merge h | critic blocking rounds | items / round | fix latency min (critic->builder) | critic latency min (builder->critic) | tests trajectory | FINAL CLEAR -> merge s |
|---|---|---|---|---|---|---|---|---|
| #1 | #3 | 0.87 | 3 | 4, 2, 1 | 6.1, 13.8, 8.2 | 14.1, 5.7, 2.3, 1.8 | n/a (docs) | 10 |
| #2 | #5 | 0.86 | 3 | 5, 3, 1 | 6.3, 13.8, 8.2 | 13.7, 5.5, 2.3, 1.7 | n/a (docs) | 22 |
| #21 | #4 | 1.27 | 2 | 3, 2 | 16.0, 9.9 | 25.1, 8.3, 17.1 | 61, 77, 85 | 3 |
| #23 | #7 | 19.11 | 2 | 4, 3 | 61.2, 357.0 | 19.0, 705.0, 4.0 | 127, 137, 143 | 3 |
| #24 | #8 | 17.43 | 3 | 4, 3, 2 | 14.2, 16.8, 289.1 | 10.2, 30.4, 15.3, 669.6 | 178, 188, 194, 199 | 3 |
| #25 | #9 | 70.46 | 2 | 6, 3 | 1062.3, 29.6 | 157.2, 8.0, 2970.6 | 236, 243, 248 | 3 |
| #26 | #10 | 19.75 | 2 visible + 1 inferred hidden | 3, 1 (+hidden R1 ~4) | 19.1, 6.3 | 1123.6, 6.3, 2.8 | 283, 298, 307, 312 | 9 |
| #27 | #11 | 97.28 | 3 | 6, 3, 2 | 417.6, 648.6, 3011.6 | 8.5, 213.8, 1534.3, 2.4 | 348, 360, 365, 368 | 8 |
| #28 (open) | #12 | open 5.9+ | 5 so far | 8, 10, 9, 5, 2 | 14.6, 122.8, 147.3, 14.3 | 21.4, 3.2, 17.9, 3.7, 2.4 | 406, 406, 405, 428, 438 | n/a |

Notes: PR #26 critic latency excludes PR-open -> first visible critic comment (round-1 review is invisible). "Items/round" for unnumbered reviews were hand-counted (PR #1/#2 R3 = 1; PR #26 R2 = 3, R3 = 1). Ledger `blocker_groups` agrees where it is non-null.

Overall (merged PRs, n = 8; latencies pooled across all 9 PRs):

| Metric | Value |
|---|---|
| Blocking rounds per merged PR | median 3 (range 2-3 visible, 3 with hidden); 21 total |
| Merged with zero blocking rounds | 0 of 8 |
| Builder fix latency (critic -> next builder report) | n = 24, median 16.4 min, mean 263 min; 62% <= 30 min; 3 gaps > 8 h (max 50.2 h) |
| Critic latency (builder report -> critic verdict) | n = 33, median 10.2 min, mean 231 min; 76% <= 30 min; 5 gaps > 8 h (max 49.5 h) |
| Open -> merge | median 18.3 h, range 0.86-97.3 h (docs PRs < 1 h; code PRs 17-97 h) |
| FINAL CLEAR -> merge | 3-22 s (median 3 s) |
| Critic START handoff after previous merge | 0.5-6 min |
| Test count at merge | 85, 143, 199, 248, 312, 368 (monotone up; one decrease, 406 -> 405 in PR #28 at `4ca5517`, not explained in the comment) |
| Ledger volume by thread | PR threads 56 (32%), issue threads 63 (36%), #20 54 (31%) |
| Builder report carries SHA / "unresolved" / re-review request (PR threads) | SHA 100%; "unresolved" explicit in 21 of 21 reports on PR #1-#27 but 1 of 4 on PR #28; re-review request 100% |

Latency distribution is bimodal (<=30 min or >8 h) which fits a human-relayed loop (owner triggers each agent turn); unverified hypothesis, see section 8 for how to test it. Size vs rounds: PR additions 281, 277, 1988, 2437, 2340, 2292, 1859, 2370, 3805; the largest PR has the most rounds (n = 9, qualitative only).

## 4. Recurring blocker categories (manual reading of all critic reviews)

| Category | PRs where it recurred (round) | What it suggests |
|---|---|---|
| **Declared / recorded but not enforced** (prose or docstring says X; code or schema does not) | #1 R2/R3 (schema invariants only in prose), #21 R1 (typed slots), #23 R1 (`max_plan_steps` compiled but ignored; tool boundary not wired to planner), #24 R1/R2 (schema not run in `finalize()`; "strict JSON" not strict), #27 R1/R2 (temperature recorded not enforced), #28 R1 (`add_bundle` lacks coherence its docstring claims), #26 R3 (store can run without registry) | Builder writes the contract then implements a subset; no self-audit of "every sentence in the spec maps to a failing test" |
| **Write-path vs replay/verify-path parity** | #21 R1 (aggregate<->child), #26 R1/R2 (replay vs live write, stale append guard), #28 (add_bundle vs `verify()`, 4 rounds) | Builder hardens the path it just wrote and not the mirror path; dual implementations drift |
| **Immutability / aliasing ("frozen" but mutable)** | #21 R2 (shared dict refs), #23 R1 (events not immutable), #24 R3 (context not snapshotted), #25 R1 (`FrozenCandidateSet`), #27 R1/R2/R3 (allocations; 3 rounds), #26 (defensive copies) | Recurring since PR #21 despite being flagged each time; not learned as a checklist item |
| **Atomicity / TOCTOU / fail-closed ordering** | #24 R1/R2 (append to existing file; check-then-open), #25 R2 (persistence not fail-atomic), #26 R2 (stale writer), #28 R1-R3 (live CIG store never populated; commit ordering) | Multi-store commit and file lifecycle are a repeated weak area, expect more in #13 (orchestration) |
| **Provenance / durable audit evidence** | #24 R1 (id coupled to digest), #25 R1/R2 (mining record cannot reconstruct prompt), #27 R1 (child evidence without trajectory refs), #28 R1-R3 (evidence hash only, not reconstructable) | Evidence-integrity is the critic's strongest theme; builder tends to store a hash, not the evidence |
| **Determinism / settings / reproducibility** | #2 R2 (repeats at temp 0 = pseudo-replication), #23 R2 (timestamps, token surface), #24 R1 (unknown tokens as 0), #27 R1/R2, #28 (settings value, type) | Builder records settings but treats them as logging, not as locked inputs |
| **Hashing / identity** | #21 R1/R2 (version identity rebind; unbound identity reusing a digest), #24 R1, #25 R1 (record identity), #28 (unique aggregate ids) | Identity/hash domains need an explicit spec table per artifact; currently decided ad hoc per PR |
| **Pre-registration / locked-design adherence** | #2 (sensitivity plan, split), #27 R2/R3 (protocol constants retunable; `evaluate_analysis()` reruns episodes, violating the locked design), #28 (sensitivity stage separation) | Builder sometimes implements convenient behavior that contradicts the locked design |
| **Leakage / oracle labels** | #1 R1 (`task_type` predicates, `backbone_tested`) | Caught at design time; no recurrence in code (good) |
| **Type strictness** | #28 R4/R5 only | New, late-stage; a symptom of the "present -> value -> type" escalation, not of Builder carelessness alone |

Reading: the Builder implements the architecture well and tests the happy path plus listed boundaries; it systematically under-covers (i) enforcement at every boundary, (ii) mirror paths, (iii) object lifetime (immutability). These are testable with generic property tests (mutate input after construct; corrupt on disk then reopen; run `verify()` against every `add_*` acceptance case), which the Critic currently supplies one review at a time.

## 5. Scope stability: convergence or scope creep?

Evidence for convergence: items per round are strictly decreasing in all 8 merged PRs (#1, #2, #21, #23, #24, #25, #26, #27; #26 counting the builder-reported 4 groups of the invisible R1); critic review length shrinks in later rounds for those PRs (e.g. #27: 6.7k -> 5.3k -> 3.8k chars); every PR terminated in a FINAL CLEAR; later blockers are mostly carry-overs (share of later-round items containing "still/remain/not yet": 0.0-0.5 on #1/#2/#23, 0.67-1.0 on #21/#24/#25/#27, 0.75 on #28; heuristic).

Evidence for expansion:
- **Acceptance bar moves after the handoff.** The #12 handoff (critic, 14:26Z) required: child "settings/provenance are present" and `result.passed == child.measurements.stage_pass`. The critic's 21:18 review required settings VALUES; the 21:35 review required exact TYPES (bool rejected as 0.0, `64.0` rejected as 64) and `stage_pass` as an actual `bool`. Each step is defensible fail-closed engineering, but none was in the handoff text. (Quote check: the handoff text contains no "boolean" and "present" for settings.)
- **PR #28 grew in round 2** (8 -> 10) because the 16:26 "residual checklist" re-split items and promoted the earlier "smaller notes" (TokenCounter call, vacuous reason, test assertions) into blockers. Round 4 then turned the "validate_token_counter exists but is not called" item into "validate the actual returned value", i.e. a deeper version of the same item.
- **Non-blocking notes that "land with this patch"** appear in 7 of 9 PRs (#21 1, #23 1, #24 3, #25 2, #26 1, #27 3, #28 2); they are not formally blockers but the builder applies them, so they widen the effective scope.
- **Critic authors both the spec and the grade.** Handoffs set explicit scope and exclusions, then the same agent judges. No evidence of an independent acceptance-test list that the Builder could run beforehand.
- **A critic-gated ADR amendment inside the implementation PR:** #12's handoff told the builder to add ADR-0002 Amendment A1 (both-traits-vs-neither interaction contrast) "before implementation"; it landed in the same PR (commit order not verifiable from comments).

Assessment: convergent per round, but with a repeated "same invariant, finer strictness" chain (PR #28 `stage_pass`: 5 rounds; settings: 4 rounds; PR #27 immutability: 3 rounds). This is not scope creep in the sense of new features, but it does cause avoidable cycles and makes "done" depend on the critic's next idea. Mitigation (below): freeze a per-PR acceptance checklist at R1 with explicit type/value strictness language, and route additional ideas to follow-up issues.

**PR #26 anomaly:** at 17:15Z the builder replied "Four blocking groups applied" with commit `a59fe56` while no critic review of PR #26 exists on #26, #10 or #20 (first visible critic comment is 11:58Z next day, a "re-review"). Either a review was delivered outside GitHub (chat relay, contradicting #20 rule 1) or a comment was deleted. Unresolved; the owner can answer in one line.

## 6. Protocol adherence (SPEC §31-§33, issue #20)

| Rule | Observation | Status |
|---|---|---|
| Repo as truth, not chat (§31, #20 r1) | PR #26 R1 review invisible on GitHub; the builder's "four groups" are only reconstructable from the builder's own restatement | One breach (unresolved) |
| ADR before research-relevant change (§31, #20 r2) | ADR-0001 Amendment A1 (`origin="acquired"`, PR #25) and ADR-0002 Amendment A1 (PR #28) were written inside the implementation PR, at critic instruction; accepted by critic. Process works but the ADR is not independently reviewed before code exists | Followed in form; not "before change" in time |
| Pre-registered manifest before experiment; no changes after lock (§33) | EXP-0001 manifest says `preregistration.locked: true`, but `expected_results_filled: false`, status `design-revised-re-review-requested`, and `hash: commit-at-lock` placeholders; PR #21 edited it after the lock commit (7220f6e -> 8781c40, `applies_to` key rename, "terminology only"); PR #28's Amendment A1 changes interaction operationalization (original design: "pairwise co-expression", manifest `interaction.runs_per_condition: 1`) but the manifest was not updated | Gap: lock is declared, not enforced (no tag / hash); manifest and ADR may drift |
| PRs marked review-needed not merged until critic resolves (#20 r4) | 8 of 8 merged only after FINAL CLEAR (merge 3-22 s later) | Followed |
| Builder replies with what changed / SHA / unresolved / re-review (#20 r5) | SHA and re-review request 100%; explicit "unresolved questions" 21/21 until PR #27, then 1/4 on PR #28 (`4ca5517`, `7886852`, and 16:23 reports lack it; `c6ce4c1` has "none") | Degrading |
| Critic re-reviews and clears gate or records remaining changes (#20 r6) | Always; critic states "Reviewed exact head X, not the builder report" and notes "no CI/check; local only" in later rounds | Followed (strong) |
| No self-validation (§32) | Builder test counts are self-reported; builder self-describes fixes "applied in full" (e.g. PR #21, #23) and the next review frequently finds residuals; PR #28 16:23 report openly listed "what remains" (good). Critic is the only validator and also the work scheduler | Partially met; no independent verification of tests |
| Technical discussion on PR/issue; #20 only for cross-cutting (#20 footer) | 54/173 comments on #20 are per-PR status mirrors; each event is posted up to 3 times (PR, issue, #20) | Drift; inflates noise, risks divergence between copies |
| Human owner is final decision maker (#20 roles) | No visible human comment distinguishable from agent comments; "merge gate opened" decisions are made by the critic text, and merges follow within seconds from the same account | Unclear, depends on who runs the merge command |
| SPEC §34 anti-overclaim | PR bodies repeatedly carry explicit "not a novelty claim / not H2 / no inheritance" boundary statements; critic checks | Followed |

## 7. Structural risks

| # | Risk | Evidence | Severity |
|---|---|---|---|
| S1 | **No CI, no attested test counts.** Every count (up to 438) is builder-reported local pytest; no `.github/`, no check runs, issue #18 pending | PR #28 critic notes it three times | High: the gate rests on unverifiable numbers and on the 3.14.5-only run |
| S2 | **Python pinned in practice to 3.14.5** while `pyproject.toml` declares `requires-python >=3.10`; every PR reports "Python 3.14.5, jsonschema 4.26.0, pytest 9.1.1" | PR bodies | Medium: untested on declared minimum; an experiment run elsewhere (GPU node, vLLM) may use another interpreter. Scratchpad has venv310/311/312/314, so an independent replay is cheap |
| S3 | **Single GitHub identity** | #20 body acknowledges; no formal REQUEST_CHANGES, branch protection cannot require a distinct reviewer; critic "clear" and the merge are one account | High for governance, Medium for current work: with 3 agents the attribution problem triples |
| S4 | **Merge coupled to the critic's clear comment** (3-22 s) | all merged PRs | Medium: no pause for owner or independent check; a mistaken clear merges instantly |
| S5 | **Critic is a single point of failure and scheduler** (START handoffs 0.5-6 min after merge; 19 KB acceptance spec) | ledger | Medium: critic blind spots become project blind spots; independent verification of its own acceptance lists is absent |
| S6 | **Relay latency tails** (gaps up to 50 h, weekend) | metrics | Low/Medium: delays, no integrity risk; but suggests human-in-the-loop dispatch |
| S7 | **Comment fan-out** (PR + issue + #20) | 173 comments, of which 56 are PR-thread events (the rest mirror them) | Low/Medium: a third agent posting would worsen it |
| S8 | **Pre-reg lock not cryptographic** (`hash: commit-at-lock` placeholders; manifest/ADR drift) | manifest, PR #21, PR #28 | Medium (H2 is load-bearing): any later dispute about "what was pre-registered" has no immutable anchor |
| S9 | **Squash merges** make branch SHAs quoted in comments non-ancestors of main (e.g. `9f9e4b7` vs merged `0ce0a36`) | git log | Low: experiment manifests must record the main SHA, not the branch SHA |
| S10 | **PR #26 invisible review** | section 5 | Low now, but a repo-as-truth breach |

## 8. Agent identity (Zcode / Codex): mapping UNCONFIRMED

I do not state which of Zcode and Codex is Builder or Critic. GitHub data cannot say: one login, no commit trailers (`git log --all` shows only `Samet Yilmaz Temel <sametyilmaztemel@users.noreply.github.com>` for branch commits and the GitHub squash committer for merges), no `Co-Authored-By` agent names, no `AGENTS.md` / `CLAUDE.md` / `.claude` files in the checkout.

Signals that could confirm the mapping (none used to guess here):
1. Owner statement in #20, or a one-line `agent:` tag the owner adds to each agent's standing instructions.
2. Local session logs / working-directory histories of Zcode and Codex on the owner's machine: compare tool-call timestamps with comment `created_at` (builder commits appear ~1-3 min before the builder comment: e.g. `c6ce4c1` committed 21:32:24Z, comment 21:32:48Z).
3. Local `git config` / reflog on the machine that authored the branch commits (builder) vs the one that only posts reviews; branch-commit author email lacks the numeric `115784698+` prefix that GitHub-web merges carry.
4. The unexplained PR #26 R1: check whether the critic agent's session log contains a review for PR #26 between 16:48Z and 17:15Z on 10-07.
5. Style fingerprints (builder: "### 1) ..." headings with test counts, `Commit:` line; critic: "Reviewed exact head", "Merge gate: CLOSED only on ...", "Keep the patch minimal"). These distinguish roles, not products.
6. API/audit logs of the token or app grants (if separate tokens were issued), GitHub audit log "user-agent" for the account (may distinguish tool vs MCP client).
7. The two "owner_or_unclear" ledger rows (#20, 10-12 18:47 and 21:18) read like critic status notes; unconfirmed.

## 9. Recommendations for a THIRD agent (Claude) joining without disrupting the loop

Principles: advisory only; never inside the gate language; findings reach the Builder only via the Critic; zero extra comments on a PR during an active fix round.

A. Identity and headers (needed because all comments share one login)
- First line of every comment: `**[Claude / independent-research, advisory, non-gating]** target: critic | owner | builder-info | all | ref: PR#N @<sha> | type: finding | question | verification | note`.
- Never use these strings: "FINAL CLEAR", "merge gate", "REQUEST CHANGES", "blocking", "Research/Critic", "Builder". Say "advisory finding" and "suggest the Critic consider".
- One top-level comment per PR per cycle at most; link to detail, do not paste it.

B. Where to post
- One short "third agent joining" note in #20 (owner approves wording; no per-PR status in #20). Everything else goes into ONE new issue "Independent research notes (third agent)" with label `third-agent`, one comment per topic. Cross-link PR numbers; do not mirror to PR threads, issue threads and #20.
- On a PR thread: only after the Critic's FINAL CLEAR (or when asked), and only a link to the notes issue. Never between a Critic review and the matching Builder report.
- Hand findings to the Critic: address the finding to the critic role, include a reproducible check (command, expected vs actual) and the exact head SHA; the Critic decides whether it becomes a blocker. Builder-facing text is out of scope until the Critic adopts it.

C. What not to touch
- Do not push to, rebase, or comment as a reviewer on `issue-*` branches; do not merge, close, label, or re-assign anything; do not edit SPEC.md, ADR-0001/0002, EXP-0001 manifests, `research/hypotheses/` or the novelty ledger directly. Propose changes as text in the notes issue, and any change to a locked or pre-registered element goes through an ADR (requires ADR).
- Work on a separate branch `claude/<topic>` and open PRs only when the owner asks; use a distinct directory (e.g. `research/third-agent/`) for notes.
- Read-only tooling: replay tests from an exported tree (`git archive`) into a scratch directory with `-p no:cacheprovider`, not in the checkout.

D. High-value, low-friction contributions (ordered)
1. Independent test replay at the current PR #28 head and at main (Python 3.10/3.12/3.14), report pass/fail per interpreter as a note to the Critic (closes S1/S2 evidence gap without changing process).
2. A generic "boundary parity" property-test checklist (mutate-after-construct, corrupt-on-disk-then-reopen, `verify()` vs `add_*` acceptance parity, bool-vs-int/float typing) for the Critic to attach to the next handoff; targets the top 3 recurring categories in section 4.
3. Manifest/ADR drift audit: diff ADR-0002 Amendment A1 vs `EXP-0001-minimal-cig.yaml` (`interaction`) and list fields that disagree.
4. Pre-registration lock audit before any EXP-0001 run: `expected_results_filled`, `hash: commit-at-lock` placeholders, status field.
5. Literature/novelty cross-check on request only, and only against the cited anchors; mark unverifiable references as unverified (SPEC §34).
6. Read #20, the open PR thread, touched ADRs and the novelty ledger at session start (per #20 "Agent instruction"), then post nothing unless a finding passes: reproducible, head-pinned, not already in a critic review.

E. Process suggestions for the owner (each is small; those that alter protocol are flagged)
- Single-copy posting: one thread per event, #20 only for cross-cutting changes (reduces 173 -> ~60 comments). (Process change, no ADR needed; SPEC §31 does not specify thread mirroring.)
- Freeze a per-PR acceptance checklist at critic R1 with explicit type/value strictness and a "new ideas go to follow-ups" rule; cap rounds (e.g. escalate to the owner after round 3).
- Add the one-line `unresolved questions:` back to every builder report (restores #20 rule 5).
- Minimal GitHub Actions workflow running pytest on Python 3.10-3.14 and writing the result to the PR check (issue #18); until then label test counts "builder-reported".
- Ask the owner to confirm the PR #26 R1 channel and to state who may press "merge".
- Consider a separate reviewer identity (bot account or GitHub App) so REQUEST_CHANGES and branch protection become possible (S3).
- Introduce a three-agent protocol note: SPEC §31/§32 describe two agents; adding a third role changes that text (requires ADR, process ADR, not a research-claim ADR).

## 10. What would change these conclusions

- Owner confirmation of the PR #26 R1 channel and of who triggers merges.
- Output of an independent test run at `c6ce4c1` / main (if tests fail at an interpreter, S1/S2 severity rises).
- Whether the Critic publishes a frozen acceptance checklist for PR #28 after round 5; if the next round again finds new type/value tightening, treat as creep.
- Live state after 21:40Z: the PR #28 builder reply to the 21:35 critic review and any #12-thread mirror (not fetched).
