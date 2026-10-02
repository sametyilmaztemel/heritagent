# Mendel Gödel Machine (MGM) — detailed notes

**Paper:** *Mendel Gödel Machine: Recursive Self-Improving Coding Agents via Comparative Evolution* — Changzhi Liu (UESTC; eq. contrib.), Yilun Liu (LMU Munich/MCML; eq. contrib.), Sikuan Yan, Volker Tresp, Yunpu Ma (LMU/MCML). arXiv:2608.07645, submitted 7 Aug 2026. **Not** a Sakana/Clune work. Project page: https://reallcz.github.io/MGM/ ; code: https://github.com/RealLcz/MGM. Reviewed 2026-10-02.

## Mechanism

- Keeps DGM's archive-based tree (source code = genotype; task outcomes = phenotype); Thompson sampling over Beta posteriors at node and clade level.
- Two Mendelian operators beyond clonal mutation:
  - **Reaction-norm mutation (Φ_RM):** agent edits itself conditioned on trajectories on *multiple* tasks simultaneously (one target failure + ≥1 reference trajectory, gated by an m_RM task-count threshold). Must identify a recurring/contrastive behavioral pattern and write a *general* fix — separating genotype-level defects from task-specific accidents ("reaction norm" = genotype's performance profile across environments).
  - **Cross-lineage hybridization (Φ_CH):** when agents from different lineages share a failed task, the failing agent is edited using the reference agent's trajectory on that same task as contrastive evidence. No code splicing — a "transferable behavioral trait" is extracted and adapted. Children attach to the failing agent's lineage (or the higher-utility one if both failed).
- **"Controlled inheritance" in their framing** = heritable effects isolated via controlled comparisons (same genotype across environments [RM]; different genotypes in same environment [CH]) — i.e., Mendel's experimental design used to **diagnose** defects, *not* a heritability gate.
- Global failed-task pool with boosted sampling (β_fail) increases cross-lineage task overlap. Proposition 1 (under additive fitness landscape): comparative operators have strictly higher fix probability than clonal mutation.

## Results (backbone Qwen3.6-35B-A3B; evolved on 60-task subsets; baseline HGM)

- SWE-bench Verified-60: HGM 73.3% → **MGM 78.3%** (+10.0 over HGM ancestor lineage).
- Polyglot-60: HGM 77.9% → **MGM 93.2%**; full Polyglot-225: 93.3% (claimed > GPT-5 with ~117× fewer params).
- Zero-shot transfer (evolved on Polyglot): SWE-bench Pro 16.7% → 26.7% (HGM regressed to 13.3%); Multilingual 41.7% → 55.0%.
- Cross-model transfer (scaffold frozen, backbone → DeepSeek): 70.8% avg vs HGM 65.0%; Polyglot scaffold on DeepSeek-V4-Pro: 96.89%.
- Ablations: removing Φ_RM or Φ_CH degrades (Φ_CH more critical); token costs per operator comparable.

## Overlap vs HeritAgent chain

| Element | Coverage |
|---|---|
| 1. Lifetime-trajectory mining | Partial — edits conditioned on logged trajectories (comparative evidence); acquisition is a code edit, not mined capabilities |
| 2. Somatic/germline separation | No — trajectories are diagnostic evidence only, never heritable state |
| 3. Executable genome | Partial — executable, versioned (archive tree) but untyped raw code |
| 4. Causal Inheritance Gate | **No — clearest gap.** Children enter the archive on `c.is_valid()` (syntactic/executability); inheritance is automatic upon a valid edit; usefulness established only statistically, post hoc, population-level. Authors concede edits aren't guaranteed "correct, general, or maintainable" |
| 5. Gated assimilation | No — assimilation immediate |
| 6. Multi-generational, frozen weights | Yes — only scaffolds evolve |

## Limitations (stated)

Expensive repository-level evaluation (few seeds/sweeps); history dependence (RM needs multiple trajectories per agent, CH needs cross-lineage task overlap — sparse archives degenerate to clonal baseline); no edit-quality guarantee; additive fitness surrogate misses real scaffold complexity; coding-scaffolds-on-public-SE-benchmarks scope (contamination risk; evolved agents unaudited for security).

## Consequences for HeritAgent

- Their "controlled inheritance" phrasing overlaps ours linguistically — in writing, always gloss theirs as *controlled diagnosis* and ours as *controlled gating* to avoid reviewer confusion.
- Reaction-norm insight → CIG replay stage should validate candidates across **multiple** environments, not just the source task (already implied by SPEC §9; now literature-grounded).
- Cross-model transfer protocol (frozen scaffold, swapped backbone) is the template for our H4 cross-backbone experiment (SPEC §19).
