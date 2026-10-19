# HeritAgent

> HeritAgent investigates whether capabilities learned by an LLM agent during its lifetime can be causally validated, assimilated into an explicit executable genome, and inherited by future generations to produce measurable cross-generational improvement without modifying the underlying foundation model. — SPEC §38

Model-independent research framework for **causally-gated assimilation of acquired skills into functional genomes** for evolvable LLM agents. Full specification: [SPEC.md](SPEC.md) (source document, v0.1).

## Status (2026-10-15) — Milestone 0: gate pipeline and EXP-0001 orchestration implemented, EXP-0001 not yet run

Phase 0 (SPEC §40, research grounding, 2026-10-02) is complete: related-work notes in `research/related-work/`, the novelty ledger, and the cross-reviewed ADR-0001 (EAG v0.1 schema) and ADR-0002 (EXP-0001 pre-registered design).

Implementation follows the SPEC §28 sequence. Merged after Research/Critic review:

| SPEC §28 step | Issue | Code |
|---|---|---|
| EAG schema | #4 | `genome/` |
| single-agent runtime + ModelAdapter | #7 | `runtime/` |
| trajectory recorder | #8 | `trajectory/` |
| trait miner | #9 | `traits/` |
| somatic trait store | #10 | `somatic/` |
| replay + ablation evaluators | #11 | `inheritance/` |
| basic CIG (4-stage gate) | #12 | `inheritance/gate.py` |

Not done yet:
- canonical `genome_id` hashing (#22);
- single-lineage inheritance and the EXP-0001 run (#13);
- filling the pre-registration fields (#6).

The ALFWorld adapter (`runtime/tool_adapters/alfworld.py`) and the EXP-0001 orchestration (`evaluation/exp0001/`, run on Colab through `notebooks/`) exist. No pilot or EXP-0001 episode has run, so nothing here supports a research claim (SPEC §34).

## Repository rules (SPEC §31–34)

- **The repo is the source of truth** — not chat history. Decisions go in `research/decisions/` (ADR format), claims auditing in `research/novelty-ledger.md`, hypotheses in `research/hypotheses/`, experiment definitions in `experiments/manifests/`.
- **Anti-overclaim:** *first / novel / better / causal / model-independent* only with experimental evidence; otherwise "contribution estimate" / "association" (SPEC §34).
- **Experiments:** pre-registration style review before the run, post-run review after (SPEC §33); results reported with seed, model revision, genome revision, split, code commit (SPEC §24).
- Two-agent collaboration: builder ↔ research/critic roles cross-review critical decisions; no agent validates its own critical result alone (SPEC §32).

## Layout

```text
heritagent/
├── SPEC.md                      # master specification (v0.1)
├── architecture/                # EAG schema proposals
├── experiments/manifests/       # pre-registered experiment designs
├── genome/                      # EAG v0.1 JSON Schemas, loader, registry, records, diff
├── runtime/                     # single-agent loop, ModelAdapter (vLLM / scripted), policies
├── trajectory/                  # trajectory recorder, JSONL storage, normalizer
├── traits/                      # Trait Miner v0 (differential skill distillation)
├── somatic/                     # persistent hash-chained somatic trait store
├── inheritance/                 # CIG: replay, ablation, generalization, regression, gate
├── evaluation/benchmarks/       # ALFWorld split-manifest builder (EXP-0001 splits)
├── evaluation/exp0001/          # EXP-0001 run: ledger, lifetimes, gate, arms, pilot, CLI
├── evaluation/metrics/          # EXP-0001 final statistics
├── notebooks/                   # Colab notebooks: CPU rehearsal, calibration pilot and EXP-0001 run
├── tests/                       # pytest suite
├── scripts/ci/                  # CI checks (schemas, manifests, links, wheel smoke test)
└── research/
    ├── novelty-ledger.md        # living novelty ledger (SPEC §30)
    ├── related-work/            # per-paper detailed notes
    ├── hypotheses/              # H1–H6, locked operationalizations
    └── decisions/               # ADRs (SPEC §31 format)
```

Later SPEC §23 components (evolution/, lineage/, the rest of evaluation/) are created when their milestones start (SPEC §27).

## Development

```bash
python -m pip install -e ".[dev]"     # Python 3.10–3.14
python -m pytest -q
python scripts/ci/check_artifacts.py  # needs: pip install -r scripts/ci/requirements-ci.txt
```

GitHub Actions (`.github/workflows/ci.yml`) runs these on every pull request and on pushes to `main`: pytest on Python 3.10–3.14, an installed-wheel smoke test, the schema / manifest / relative-link checks, yamllint and ruff F821. A weekly job (`link-rot.yml`) checks the external citations in `research/`. Test counts should come from CI, not from local runs.

## GitHub workflow

Repo: https://github.com/sametyilmaztemel/heritagent (public). Tracking:

- **PRs** are the cross-review mechanism (SPEC §32): ADR/design artifacts land via PR and merge only after review questions are resolved and answers recorded in the ADR.
- **Agents:** Zcode (Builder) follows the protocol in issue #20. Claude is a second builder for work outside Zcode's M0 sequence. Since 2026-10-14 the owner has removed Codex from the process, and Claude holds the Critic's duties through different-model review workflows. Its rules are in `CLAUDE.md`: its own different-model review agents review its PRs, and changes to locked or pre-registered elements are the owner's decision.
- **Milestones** map to SPEC §29 + §28: `M0: Minimal HeritAgent` (schema → runtime → recorder → miner → somatic store → evaluators → CIG → single-lineage inheritance → EXP-0001), `M1: Population Evolution`, `M2: Cross-Backbone & Paper`.
- **Issues** carry the SPEC §28 development sequence in order; dependencies are stated in issue bodies. Labels: `research` / `engineering` / `review-needed` / `documentation` / `infra`.
- Experiment runs must record model weights hash, split hashes, and code commit (SPEC §24) — enforced by checklist in the EXP-0001 execution issue.

## Suggested next step

#13 Phase A (the EXP-0001 orchestration, pre-run artifacts and run protocol, ADR-0006 and ADR-0007) is merged. The run executes on one Colab A100 80 GB with parallel episodes (ADR-0008); `notebooks/EXP-0001-pilot.ipynb` and `notebooks/EXP-0001-run.ipynb` drive it through `python -m evaluation.exp0001.cli`. The pilot notebook stops before the install unless `notebooks/EXP-0001-rehearsal.ipynb` passed on a CPU runtime for the same commit and model; the run notebook, and `run` itself, unless `notebooks/EXP-0001-run-rehearsal.ipynb` (the whole run against a stand-in server, at the locked protocol) passed for the same commit, model, software and worker count. The model is Qwen3.8-27B, served text-only with thinking off (ADR-0011). No EXP-0001 or pilot episode has run. Next: the rehearsal on a CPU runtime, then the calibration pilot, then the owner's decision if G0 is above 60 % or the temperature-0 repeats differ (ADR-0011 D6), then the compute-cap decision, then the EXP-0001 run.
