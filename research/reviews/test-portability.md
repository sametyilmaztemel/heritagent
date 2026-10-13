# Test-environment portability and CI-attestation audit

Scope: main at /home/user/heritagent (HEAD 0ce0a36) and PR #28 head at scratchpad/pr28 (branch claude/keen-bohr-ku33ye, HEAD 4ca5517). Both trees were left unmodified (git status clean; PYTHONDONTWRITEBYTECODE=1; -p no:cacheprovider). Fix validation ran on scratch copies (git archive / cp), not on the trees.

## 1. Test matrix (python -m pytest -q --continue-on-collection-errors -p no:cacheprovider)

| Python | main (0ce0a36) | PR #28 (4ca5517) | Log |
|---|---|---|---|
| 3.10.20 | 365 passed, 1 collection error | 402 passed, 1 collection error | out/research/portability/venv310-{main,pr28}.log |
| 3.11.17 | 365 passed, 1 collection error | 402 passed, 1 collection error | venv311-{main,pr28}.log |
| 3.12.3 | 365 passed, 1 collection error | 402 passed, 1 collection error | venv312-{main,pr28}.log |
| 3.13.16 | 365 passed, 1 collection error | 402 passed, 1 collection error | venv-{main,pr28}.log |
| 3.14.6 | 368 passed, 0 errors | 405 passed, 0 errors | venv314-{main,pr28}.log |

The 3 extra passes on 3.14 are the 3 tests in tests/somatic/test_store_replay_parity.py, which is not collected below 3.14. The collection error is the only failure on every version.

Venvs (all jsonschema 4.26.0, pytest 9.1.1): scratchpad/venv310, venv311, venv312 (created with uv), venv (3.13.16), venv314 (3.14.6).

## 2. Root cause of the only failure

- File: tests/somatic/test_store_replay_parity.py:63 `def re_sign_with_edit(path: Path, edit):`
- The file never imports `Path`. The import block (lines 6-13) has only `json`, `pytest`, `genome.validation.errors`, `somatic.journal`, `somatic.store`, `tests.somatic.conftest`.
- Under 3.14, PEP 649 defers annotation evaluation, so the missing name is never looked up and the file passes. Under 3.10-3.13 the annotation is evaluated at def time and raises `NameError: name 'Path' is not defined`.
- Identical on PR head: tests/somatic/test_store_replay_parity.py:63 (line 6 `import json`).

Minimal fix: add `from pathlib import Path` after `import json` (line 6 in both trees).

Validation on scratch copies with that one line added:
- main: 368 passed on 3.10, 3.11, 3.12, 3.13, 3.14.
- PR #28: 405 passed on 3.10, 3.13, 3.14. The 3.11 and 3.12 runs were not repeated on the PR copy, but the fix is the same single line in an identical file.

## 3. Other annotation-only undefined names (ruff F821, --select F821, target py310, --no-cache)

These are not failing today because each file has `from __future__ import annotations`, so the annotations are stored as strings and never evaluated. They break any caller of `typing.get_type_hints` (or other runtime annotation introspection) and are latent defects:

| File:line (main) | Name | Status on PR #28 | Note |
|---|---|---|---|
| inheritance/ablation.py:54 | EvaluationTask | same | `from __future__` at line 21; EvaluationTask is defined in inheritance/runner.py:88 and not imported here |
| inheritance/replay.py:59, 60, 96 | EvaluationTask | same | `from __future__` at line 20; same missing import |
| inheritance/gate.py:199 (PR only) | GateEpisodeRunner | new in PR | `from __future__` at line 30; GateEpisodeRunner is a Protocol in inheritance/runner.py:172, not imported in gate.py; PR also imports it in inheritance/ablation.py:34, replay.py:37, __init__.py:35 |
| tests/miner/conftest.py:76 | MiningPair (string annotation "MiningPair") | same | Defined at traits/miner/inputs.py:46; string annotation, no runtime impact unless introspected |
| tests/somatic/test_store_replay_parity.py:63 | Path | same | The only one that fails at import (section 2) |

Ruff also reported about 50 unused-import (F401) and redefinition (F811) findings in tests and source. These are lint noise and do not affect runtime. They are listed in the ruff output but not repeated here.

Fix suggestion for the latent set: add the missing names under `if TYPE_CHECKING:` (or import them directly) in inheritance/ablation.py, inheritance/replay.py, inheritance/gate.py, and tests/miner/conftest.py.

## 4. Newer-Python syntax and stdlib APIs

- A regex scan over all 92 .py files found no PEP 695 generics or `type` statements, no `except*`/ExceptionGroup/TaskGroup, no StrEnum/Self/TypeIs/ReadOnly/override, no datetime.UTC, no itertools.batched, no Path.walk, no warnings.deprecated, no copy.replace, no tomllib, no t-strings, no annotationlib, and no compression.* modules.
- Parse check with ast.parse(feature_version=(3,10)) across all .py files in both trees: 0 failures.
- The 3.10 interpreter runs the full suite with only the one collection error above, so no other 3.11+ runtime API is exercised by tests.
- Pyproject: `requires-python = ">=3.10"`, `dependencies = ["jsonschema>=4.18"]`, `dev = ["pytest>=8"]`, `pythonpath = ["."]`. The claim is supported for 3.10-3.14 by the runs above, with the one-line fix.

Caveat: the regex scan is a heuristic. The 3.10 run is the stronger evidence.

## 5. CI attestation

- No `.github/` directory on main or on the PR #28 worktree (checked with ls).
- GitHub Actions `list_workflows` for sametyilmaztemel/heritagent returns total_count 0. No workflows exist on the repo.
- PR #28 `get_check_runs` returns total_count 0. No check runs are attached to the PR head.
- Issue #18 ("CI: EAG/CIG schema validation + markdown lint", open, labels engineering/infra, created 2026-10-02, no linked PRs) is the tracker. It asks for: schema validation of example genomes and fixtures against draft 2020-12 JSON Schemas once #4 lands, yaml lint for experiment manifests, and a markdown link check for research/ docs.
- Consequence: no CI gate currently runs the test suite on any Python version. The 3.13 failure is only detected by a manual run. Any "tests pass" claim in PR text is unattested.

## 6. Recommended CI matrix

Minimum viable workflow (new file, e.g. .github/workflows/tests.yml, to be added under issue #18):

- Trigger: pull_request and push to main.
- Matrix: python-version ["3.10", "3.11", "3.12", "3.13", "3.14"], on ubuntu-latest. Optionally one macOS entry on 3.12.
- Steps: checkout; setup-python with the matrix version; `pip install -e ".[dev]"`; `python -m pytest -q` (drop --continue-on-collection-errors in CI so collection errors fail the job).
- Required status check: the full matrix, so a 3.13-only break blocks merge.
- Separate job for schema/yaml/markdown lint from issue #18.
- Until the one-line fix lands, the 3.10-3.13 legs will fail. Either land the fix first, or mark those legs allowed-to-fail for one cycle and track it.
- Add ruff with `--select F821` as a gate so new annotation-only undefined names fail the build. Consider `from __future__` consistency as a separate lint.

## 7. Recommendations (ordered)

1. Add `from pathlib import Path` to tests/somatic/test_store_replay_parity.py (both main and PR #28). Verified: 368 (main) / 405 (PR) passed on 3.10-3.14 with this change.
2. Fix the latent undefined annotation names in inheritance/ablation.py, inheritance/replay.py, inheritance/gate.py (PR #28 only) and tests/miner/conftest.py.
3. Add the CI workflow above under issue #18, so the matrix becomes a required check. Attest test results in PR bodies only against a CI run, not a local run.
4. Re-run this matrix after the fix lands on both branches, and record the exact interpreter patch versions.

## Limits

- Only one PR head (#28) and main were checked. Other open PRs were not inspected.
- The 3.11 and 3.12 runs on the PR-fix copy were not repeated; see section 2.
- Coordination traffic on issue #20 was not analyzed in this audit. The copy of the thread is at scratchpad/ctx/issue20-thread.txt.
