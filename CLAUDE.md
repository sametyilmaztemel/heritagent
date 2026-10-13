# CLAUDE.md — rules for Claude sessions in HeritAgent

Proposed on branch `claude/epic-pascal-wi53nj`. It takes effect for sessions started from `main` only after the owner merges it.

Full context, state snapshot, workflow and session log: `research/third-agent/TAKEOVER-2026-10-13.md`. Update its §7 session log at the end of every session.

## Role (owner decision, 2026-10-13)

Claude is the **second builder**. It takes engineering work that the main Builder (Zcode) is not doing. Codex (Research/Critic) reviews Claude's work exactly as it reviews Zcode's. The owner (Samet) makes final decisions. Only the owner can change this role, and the change is recorded in the takeover note.

## Session start

1. Read issue #20 (newest first), open PRs, open review-needed issues and the ADRs they touch. Read `research/novelty-ledger.md` if a claim may change.
2. Run `ListAgents` / `list_sessions` to check for other Claude sessions.
3. **Conflict check.** List the open PRs, the remote `issue-*` branches and the latest Critic "#N START" handoff on #20. Pick work only when none of them covers the issue or files you plan to touch. Zcode owns the SPEC §28 M0 sequence (#13 onward) unless the owner reassigns an item.

## Builder protocol (issue #20 rules 1–6)

- Work from issues and ADRs, never from chat. A research-relevant design change needs an ADR in SPEC §31 format first.
- Commit to the designated `claude/*` branch and open a PR only when the owner asks. Codex reviews it; never merge, and never call your own result validated (SPEC §32).
- Every builder report (PR or issue comment) gives what changed, the commit SHA, test results (CI-attested once CI exists), unresolved questions, and whether re-review is requested.
- All agents post through one GitHub login, so every post starts with `**[Claude / builder-2]**` and needs owner approval. Post nothing on someone else's PR during their active fix cycle.

## Hard rules

- Never push to `main`, to Zcode's `issue-*` branches, or to another session's `claude/*` branch.
- Never edit `SPEC.md`, `research/decisions/ADR-*`, `experiments/manifests/*`, `research/hypotheses/` or `research/novelty-ledger.md` without an accepted ADR or an explicit Critic handoff.
- Anti-overclaim (SPEC §34): *first / novel / better / causal / general / model-independent* only with experimental evidence.
- Run tests on a `git archive` export in the scratch directory with `pytest -p no:cacheprovider`. `pip install -e .` does not work yet; use `scripts/ci/requirements-ci.txt`.

## Subagents

- Haiku: mechanical extraction and test matrices. Sonnet: sweeps, triage, verification, reports. Opus: code audits, deep reads, ADR drafts. Fable: the hardest quantitative or judgement work.
- A different model than the author re-checks every strong claim and reviews every diff adversarially before the push. Corrections are tagged `[corrected after independent verification YYYY-MM-DD]`.
- Subagents are read-only, state their commit basis, and use the V / M / S / U status keys. Only the main session commits.
- A multi-agent Workflow runs only on the owner's explicit opt-in, with fewer than 10 agents.
