# Repro scripts for the third-agent audit and verification (2026-10-12/13)

Provenance copies of the scripts and saved outputs behind the claims in:

- `research/reviews/PR-28-independent-audit-c6ce4c1.md` (incl. §0 post-merge status at `main` 6014bc7)
- `research/reviews/M0-statistics-audit.md`
- `research/methodology/VERIFICATION-2026-10-13.md`

`<scratch>` in the scripts stands for the session scratchpad. It contained detached worktrees
(`pr28` = 4ca5517, `pr28c` = c6ce4c1, `pr28d` = 628edd2) and the virtualenvs used to run everything
(`venv314` = CPython 3.14.6; `venv`/`venv310`–`venv312` = 3.13/3.10–3.12; `jsonschema` 4.26.0,
`pytest` 9.1.1, `numpy`, `scipy`). To re-run a script, check out the named commit, run from the repo
root with `PYTHONPATH=.`, and point the interpreter at a 3.14 environment. Mutation-test tree copies
(`mut/`, `delta-mut/`) were not committed; `delta-mut-scripts/` and
`c6-state-contracts-tests/run_mutations.sh` recreate them.

| Directory | What it checks |
|---|---|
| `claims-1`…`claims-3`, `claims-latest-*` | Research/Critic residual items on PR #28 (4ca5517, then c6ce4c1) |
| `c6-decision-correctness`, `c6-state-contracts-tests` | Fresh-eyes review findings at c6ce4c1 |
| `verify-*` | Independent second-pass verification of individual findings |
| `delta`, `delta-mut-scripts` | Re-verification at merged `main` 6014bc7 and mutation-testing leads |
| `m0-stats` | Merged M0 statistics code (bootstrap, ablation, replay) |
| `verify-power` | Independent recomputation of the EXP-0001 power-analysis headline numbers |
| `refute-V*` | Skeptic checks of the EXP-0001 validity-review findings |

Nothing here changes repository code. All scripts are read-only against the checkout they run in.
