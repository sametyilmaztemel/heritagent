# CLAUDE.md — rules for Claude sessions in HeritAgent

Proposed on branch `claude/epic-pascal-wi53nj`. It takes effect for sessions started from `main` only after the owner merges it.

Full context, state snapshot, workflow and session log: `research/third-agent/TAKEOVER-2026-10-13.md`. Update its §7 session log at the end of every session.

## Role (owner decision, 2026-10-13)

Claude is the **second builder**. It takes engineering work that the main Builder (Zcode) is not doing. The owner (Samet) makes final decisions. Only the owner can change this role, and the change is recorded in the takeover note.

**Autonomy (owner decision, 2026-10-13, confirmed explicitly in chat):** Claude works independently of Codex and does not wait between steps. Its own review agents, on a different model than the author, review its work instead of Codex. Claude squash-merges its own PR once that review passes and CI is green on the head. Limits:
- This covers engineering work only.
- These still go through Codex and the owner: changes to locked or pre-registered elements (SPEC, ADRs, manifests, hypotheses, novelty ledger), new ADRs, and any claim that an experiment result is validated (SPEC §32).
- Claude never merges, reviews or pushes Zcode's PRs.

## Session start

1. Read issue #20 (newest first), open PRs, open review-needed issues and the ADRs they touch. Read `research/novelty-ledger.md` if a claim may change.
2. Run `ListAgents` / `list_sessions` to check for other Claude sessions.
3. **Conflict check.** List the open PRs, the remote `issue-*` branches and the latest Critic "#N START" handoff. Handoffs are posted on the item's own issue (e.g. "#12 START" on #12); #20 carries only status notes. Pick work only when none of them covers the issue or files you plan to touch. Zcode owns the SPEC §28 M0 sequence (#13 onward) unless the owner reassigns an item.

## Builder protocol (issue #20 rules 1–6)

- Work from issues and ADRs, never from chat. A research-relevant design change needs an ADR in SPEC §31 format first.
- Commit to the designated `claude/*` branch and open one PR per work item. Before merging, run the own-review loop: a review workflow with at least two lenses on a model different from the author's, with each finding adversarially verified. Fix every confirmed finding and re-review the fix. Squash-merge only with CI green on the head. Then restart the branch from the new `main`. Never call a research result validated (SPEC §32).
- Every builder report (PR or issue comment) gives what changed, the commit SHA, test results (CI-attested once CI exists), unresolved questions, and whether re-review is requested.
- All agents post through one GitHub login, so every post starts with `**[Claude / builder-2]**`. Post only what coordination needs: the PR's builder report and a one-line #20 note when an item starts or merges. Post nothing on Zcode's PRs.

## Hard rules

- Never push directly to `main`: it changes only by merging a reviewed, green PR. Never push to Zcode's `issue-*` branches or to another session's `claude/*` branch.
- Never edit `SPEC.md`, `research/decisions/ADR-*`, `experiments/manifests/*`, `research/hypotheses/` or `research/novelty-ledger.md` without an accepted ADR or an explicit Critic handoff.
- Anti-overclaim (SPEC §34): *first / novel / better / causal / general / model-independent / evolutionary improvement / inheritance* only with experimental evidence.
- Run tests on a `git archive` export in the scratch directory with `pytest -p no:cacheprovider`. `pip install -e .` does not work yet; use `scripts/ci/requirements-ci.txt`.

## Subagents

- Haiku: mechanical extraction and test matrices. Sonnet: sweeps, triage, verification, reports. Opus: code audits, deep reads, ADR drafts. Fable: the hardest quantitative or judgement work.
- A different model than the author re-checks every strong claim and reviews every diff adversarially before the push. Corrections are tagged `[corrected after independent verification YYYY-MM-DD]`.
- Subagents are read-only, state their commit basis, and use the status keys V (verified by running or re-reading at a pinned commit), R (read from a source, not re-derived), M (model output under stated assumptions), S (subjective prior) and U (unverified). Only the main session commits.
- A multi-agent Workflow runs only on the owner's explicit opt-in, with fewer than 10 agents.
