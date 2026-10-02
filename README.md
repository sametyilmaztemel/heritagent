# HeritAgent

> HeritAgent investigates whether capabilities learned by an LLM agent during its lifetime can be causally validated, assimilated into an explicit executable genome, and inherited by future generations to produce measurable cross-generational improvement without modifying the underlying foundation model. — SPEC §38

Model-independent research framework for **causally-gated assimilation of acquired skills into functional genomes** for evolvable LLM agents. Full specification: [SPEC.md](SPEC.md) (source document, v0.1).

## Status (2026-10-02) — Phase 0: research grounding complete, implementation NOT started

Per SPEC §40, the first task was literature, not code. Completed this round:

1. ✅ Reviewed the seven works named in SPEC §40 → `research/related-work/` (AgentSquare, Darwin Gödel Machine, Mendel Gödel Machine, Genomebook, SkillRL, ShapleyFlow+CapaBench, SkillShapley) + a 2023–2026 sweep (`adjacent-sweep.md`: ASSAY, Skill-α, AgentSpawn, LOGOS, …).
2. ✅ Novelty ledger created and updated → `research/novelty-ledger.md` (claim position, do-not-claim list, naming hazard: CapaBench renamed itself "ShapleyFlow").
3. ✅ EAG v0 schema proposed → `architecture/eag-v0-schema.md` (ADR-0001, pending cross-review).
4. ✅ Minimal CIG experiment designed and pre-registered → `experiments/manifests/EXP-0001-*` (ADR-0002, pending cross-review).
5. ⬜ Cross-review by research/critic role (SPEC §32) → then implementation begins with the SPEC §28 sequence (EAG schema → runtime → trajectory → miner → CIG → single-lineage inheritance).

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
└── research/
    ├── novelty-ledger.md        # living novelty ledger (SPEC §30)
    ├── related-work/            # per-paper detailed notes
    ├── hypotheses/              # H1–H6, locked operationalizations
    └── decisions/               # ADRs (SPEC §31 format)
```

The full code skeleton (SPEC §23: genome/, runtime/, trajectory/, traits/, inheritance/, evolution/, lineage/, evaluation/) is created when implementation starts — deliberately not before (SPEC §27).

## GitHub workflow

Repo: https://github.com/sametyilmaztemel/heritagent (public). Tracking:

- **PRs** are the cross-review mechanism (SPEC §32): ADR/design artifacts land via PR and merge only after review questions are resolved and answers recorded in the ADR.
- **Milestones** map to SPEC §29 + §28: `M0: Minimal HeritAgent` (schema → runtime → recorder → miner → somatic store → evaluators → CIG → single-lineage inheritance → EXP-0001), `M1: Population Evolution`, `M2: Cross-Backbone & Paper`.
- **Issues** carry the SPEC §28 development sequence in order; dependencies are stated in issue bodies. Labels: `research` / `engineering` / `review-needed` / `documentation` / `infra`.
- Experiment runs must record model weights hash, split hashes, and code commit (SPEC §24) — enforced by checklist in the EXP-0001 execution issue.

## Suggested next step

Research/critic review of the two open PRs (ADR-0001, ADR-0002); in parallel, #4 (EAG schema module) can start once ADR-0001 questions are resolved.
