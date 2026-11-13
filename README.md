# HeritAgent

> HeritAgent investigates whether capabilities learned by an LLM agent during its lifetime can be causally validated, assimilated into an explicit executable genome, and inherited by future generations to produce measurable cross-generational improvement without modifying the underlying foundation model. — SPEC §38

A research framework for **causally-gated assimilation of acquired skills into functional genomes** for evolvable LLM agents, designed to be backbone-agnostic (untested: H4). Full specification: [SPEC.md](SPEC.md) (source document, v0.1).

## Status (2026-11-12) — Milestone 0: gate pipeline and EXP-0001 orchestration implemented, EXP-0001 not yet run

Phase 0 (SPEC §40, research grounding, 2026-10-02) is complete: related-work notes in `research/related-work/`, the novelty ledger, and the cross-reviewed ADR-0001 (EAG v0.1 schema) and ADR-0002 (EXP-0001 pre-registered design).

Implementation follows the SPEC §28 sequence. Merged (each row's review is on its PR; the rows up to #12 after the Research/Critic review, #13 Phase B without a Critic clearance yet):

| SPEC §28 step | Issue | Code |
|---|---|---|
| EAG schema | #4 | `genome/` |
| single-agent runtime + ModelAdapter | #7 | `runtime/` |
| trajectory recorder | #8 | `trajectory/` |
| trait miner | #9 | `traits/` |
| somatic trait store | #10 | `somatic/` |
| replay + ablation evaluators | #11 | `inheritance/` |
| basic CIG (4-stage gate) | #12 | `inheritance/gate.py` |
| canonical `genome_id` hashing | #22 | `genome/validation/identity.py` (PR #29, `6c1e096`) |
| EXP-0001 pre-registration fields | #6 | `experiments/manifests/EXP-0001-design.md` §8, locked 2026-10-14 (`18f9bec`) |
| EXP-0001 orchestration, pre-run artifacts, run protocol (#13 Phase A) | #13 | `evaluation/exp0001/`, `evaluation/metrics/` (PR #68; ADR-0006, ADR-0007) |
| EXP-0001 on one Colab A100: CLI, ledger, mirror, pilot, rehearsals, notebooks (#13 Phase B) | #13 | `evaluation/exp0001/cli.py`, `notebooks/`, `environments/exp0001-colab/` (PRs #69 to #82; ADR-0008 to ADR-0012) |

Not done yet (#13, the SPEC §28 step "single-lineage inheritance" with its experiment, EXP-0001):
- the run rehearsal on a CPU runtime (`notebooks/EXP-0001-run-rehearsal.ipynb`, the owner's step);
- the EXP-0001 run (`notebooks/EXP-0001-run.ipynb`, A100 80 GB);
- the post-run review (SPEC §33; draft procedure: `architecture/exp0001-post-run.md`, PROPOSED);
- closing #13.

The ALFWorld adapter (`runtime/tool_adapters/alfworld.py`) and the EXP-0001 orchestration (`evaluation/exp0001/`, run on Colab through `notebooks/`) exist. The calibration pilot ran on 2026-10-16 (127 episodes, see below); no EXP-0001 episode has run, so nothing here supports a research claim (SPEC §34).

## Repository rules (SPEC §31–34)

- **The repo is the source of truth** — not chat history. Decisions go in `research/decisions/` (ADR format), claims auditing in `research/novelty-ledger.md`, hypotheses in `research/hypotheses/`, experiment definitions in `experiments/manifests/`.
- **Anti-overclaim:** *first / novel / better / causal / model-independent* only with experimental evidence; otherwise "contribution estimate" / "association" (SPEC §34).
- **Experiments:** pre-registration style review before the run, post-run review after (SPEC §33); results reported with seed, model revision, genome revision, split, code commit (SPEC §24).
- Two-agent collaboration: builder ↔ research/critic roles cross-review critical decisions; no agent validates its own critical result alone (SPEC §32).

## Layout

```text
heritagent/
├── SPEC.md                      # master specification (v0.1)
├── CLAUDE.md                    # rules for Claude sessions (builder-2)
├── .mcp.json                    # colab-mcp, pinned, for a Claude session on the owner's computer
├── architecture/                # EAG schema notes; EXP-0001 Colab operations and post-run procedure
├── experiments/manifests/       # pre-registered experiment designs (locked)
├── genome/                      # EAG v0.1 JSON Schemas, loader, registry, records, diff
├── runtime/                     # single-agent loop, ModelAdapter (vLLM / scripted), policies, ALFWorld adapter
├── trajectory/                  # trajectory recorder, JSONL storage, normalizer
├── traits/                      # Trait Miner v0 (differential skill distillation)
├── somatic/                     # persistent hash-chained somatic trait store
├── inheritance/                 # CIG: replay, ablation, generalization, regression, gate
├── evaluation/benchmarks/       # ALFWorld split-manifest builder (EXP-0001 splits)
├── evaluation/exp0001/          # EXP-0001 run: ledger, lifetimes, gate, arms, pilot, mirror, rehearsals, figures, CLI
├── evaluation/metrics/          # EXP-0001 final statistics
├── notebooks/                   # Colab notebooks: CPU rehearsals, calibration pilot, EXP-0001 run, figures
├── environments/exp0001-colab/  # the locked environment the notebooks install (uv, Python, hashed packages)
├── tests/                       # pytest suite (tests/fixtures/alfworld: one ALFWorld game, MIT)
├── scripts/ci/                  # CI checks (schemas, manifests and their code binding, links, wheel smoke test)
├── scripts/exp0001/             # the run rehearsal's script builder, the Colab environment lock script
└── research/
    ├── novelty-ledger.md        # living novelty ledger (SPEC §30)
    ├── related-work/            # per-paper detailed notes
    ├── hypotheses/              # H1–H6, locked operationalizations
    ├── decisions/               # ADRs (SPEC §31 format)
    ├── proposals/               # drafts for the owner (PROPOSED or superseded)
    └── third-agent/             # builder-2's takeover note: current state, open decisions, session log
```

Later SPEC §23 components (evolution/, lineage/, the rest of evaluation/) are created when their milestones start (SPEC §27).

## Development

```bash
python -m pip install -e ".[dev]"     # Python 3.10–3.14; installs pytest, matplotlib and pyyaml
python -m pytest -q
python scripts/ci/check_artifacts.py  # needs: pip install -r scripts/ci/requirements-ci.txt
```

The ALFWorld adapter's real-game test (`tests/runtime/test_alfworld_adapter.py`) skips without the alfworld extra. To run it, install `python -m pip install -e ".[dev,alfworld]"` (it builds `jericho` and `fast-downward-textworld` from source) and run that file. It plays the committed game `tests/fixtures/alfworld/game.tw-pddl` unless `ALFWORLD_GAMEFILE` names another.

GitHub Actions (`.github/workflows/ci.yml`) runs on every pull request, on pushes to `main` and by hand:
- `pytest` on Python 3.10–3.14 with `-e ".[dev]"`, skip reasons shown (`-rs`); 30-minute limit;
- `ALFWorld adapter on a real game` (Python 3.13, `.[dev,alfworld]`): the real-game test against the committed game, failing if it skips; 20 minutes;
- an installed-wheel smoke test; 10 minutes;
- `schemas, manifests, links`: `check_artifacts.py` (schemas, manifests, the locked EXP-0001 manifest values against the code constants, relative links), `yamllint --strict` and `ruff check --select F821`; 10 minutes.

A newer push to a pull request cancels that pull request's older run; every other run has a concurrency group of its own. A test that runs longer than 5 minutes prints every thread's stack and goes on (`faulthandler_timeout`). A weekly job (`link-rot.yml`) checks the external citations in `research/`. Test counts should come from CI, not from local runs.

## GitHub workflow

Repo: https://github.com/sametyilmaztemel/heritagent (public). Tracking:

- **PRs** are the cross-review mechanism (SPEC §32): ADR/design artifacts land via PR and merge only after review questions are resolved and answers recorded in the ADR.
- **Agents:** Zcode (Builder) follows the protocol in issue #20. Claude is a second builder for work outside Zcode's M0 sequence. Since 2026-10-14 the owner has removed Codex from the process, and Claude holds the Critic's duties through different-model review workflows. Its rules are in `CLAUDE.md`: its own different-model review agents review its PRs, and changes to locked or pre-registered elements are the owner's decision.
- **Milestones** map to SPEC §29 + §28: `M0: Minimal HeritAgent` (schema → runtime → recorder → miner → somatic store → evaluators → CIG → single-lineage inheritance → EXP-0001), `M1: Population Evolution`, `M2: Cross-Backbone & Paper`.
- **Issues** carry the SPEC §28 development sequence in order; dependencies are stated in issue bodies. Labels: `research` / `engineering` / `review-needed` / `documentation` / `infra`.
- Experiment runs must record model weights hash, split hashes, and code commit (SPEC §24). For EXP-0001 the code enforces it: `cli run` writes the run-start record (`run-start.json`: a clean 40-hex commit, the model, revision and tokenizer, the weights' sha256, the split, pilot and benefit-bank manifest hashes, the miner's prompt and schema hashes, G0's genome id, the protocol) before any episode, refuses a dirty checkout, and on a resume refuses another commit, other weights, another benefit bank or other software (`evaluation/exp0001/provenance.py`, `cli.py`). `run --rehearsal-passed` refuses to start without a passed run rehearsal of the same commit, model, revision, tokenizer, Python, pinned software, protocol and worker count.

## Suggested next step

#13 Phase A (the EXP-0001 orchestration, pre-run artifacts and run protocol, ADR-0006 and ADR-0007) and Phase B (the Colab execution path, ADR-0008 to ADR-0012) are merged. The run executes on one Colab A100 80 GB with parallel episodes (ADR-0008); `notebooks/EXP-0001-pilot.ipynb` and `notebooks/EXP-0001-run.ipynb` drive it through `python -m evaluation.exp0001.cli`. The pilot notebook stops before the install unless `notebooks/EXP-0001-rehearsal.ipynb` passed on a CPU runtime for the same commit and model. The run notebook stops before the install without a passed `notebooks/EXP-0001-run-rehearsal.ipynb` (the whole run against a stand-in server, at the locked protocol) for the commit and model, and before the model download unless that record covers this machine's Python, software, revision and worker count; `run` itself checks the same. The pilot, run and rehearsal notebooks install one locked environment (`environments/exp0001-colab/`: a pinned Python and every package by version and hash, not Colab's own) and check it with `cli env-check` before the data, the model or any command that records. The model is Qwen3.8-27B, served text-only with thinking off (ADR-0011). The calibration pilot ran on 2026-10-16 (127 episodes); its ADR-0011 D6 triggers fired and led to ADR-0012, the gate's recalibration, accepted on 2026-10-19 and implemented. No EXP-0001 episode has run. Operations: `architecture/exp0001-colab-operations.md`.

The run notebook's first cell lists four preconditions. Their status (R, the takeover note):
1. the run rehearsal passed on a CPU runtime for the run's commit, model and worker count: **not yet** (the owner's step, about 1.5–3 hours with `WORKERS = 16`, M);
2. the pilot ran, its report was read and the ADR-0011 D6 decision made: done (ADR-0012, owner, 2026-10-19);
3. the compute cap decided: done (ADR-0012 D9);
4. the Critic's clearance: given on 2026-11-12 (builder-2's review workflows; posted on #13 with the commit it covers). A later change to the run path needs its own review.

The two owner decisions to settle before the run were made on 2026-11-12: the G0 prompt's history stays as it is (ADR-0003 note N1), and the teacher is asked for at most 5 proposals per call (ADR-0011 note N1). The second changes the run protocol's hash, and the run rehearsal must be of the run's commit. On 2026-11-13 the owner decided three more, each recorded before any EXP-0001 episode: a threat line on the missing control arm (the design's §6 item 8), ADR-0006's notes N4 and N5 on how the H2 regression clause is read, and the rule for the threshold-sensitivity grid's T_eval outcomes (a note under ADR-0012 D7: a table derived after the run). None changes the run's code or protocol. The other open owner decisions are listed in `research/third-agent/TAKEOVER-2026-10-13.md` §5. Next: the run rehearsal, then the EXP-0001 run.
