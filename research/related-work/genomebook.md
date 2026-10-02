# Genomebook — detailed notes

**Paper:** *Genomebook: Mendelian inheritance of behavioural traits in large language model agents across eight generations* — Manuel Corpas (sole author, University of Westminster). bioRxiv preprint 10.64898/2026.03.22.713494, posted 2026-03-24, 21 pp., not peer-reviewed. No arXiv version. Companion docs site (ClawBio, reportedly under revision at Nature Machine Intelligence) describes extended free simulations to 100 generations on GPT-5.4-mini / Claude Haiku 4.5: https://docs.clawbio.ai/research/genomebook/. Reviewed 2026-10-02.

## Mechanism

- Genome encodes **behavioural/personality trait descriptors, not executable capabilities**: 26 traits (9 cognitive, 9 personality, 8 physical — e.g. "leadership drive", "obsessive focus", "mathematical intuition") mapped to 60 diploid loci across 22 autosomes + sex chromosomes.
- Each trait continuous 0–1; per-locus effect size 0.15–0.50; dominance modes: additive (0/0.5/1.0 per alt allele), dominant (≥1 alt allele), recessive (homozygous). Trait score = effect-size-weighted normalized sum.
- Founders: hand-authored SOUL.md profiles (20 historical figures) discretised into diploid genotypes (DNA.md) via fixed thresholds.
- **Phenotype expression = prompt injection**: SOUL.md + DNA.md placed in the Claude Sonnet system prompt; weights frozen (no training).
- Breeding: pairwise compatibility scoring (expected offspring heterozygosity, trait complementarity, shared recessive disease risk); top pairs → 3 offspring each via Mendelian segregation; mutation 0.1%/locus/generation (3× at hotspots); 20 synthetic disease conditions (penetrance 0.40–0.95, fitness costs −0.25..+0.05). Selection **centrally imposed by the compatibility function**, not emergent.
- Lineage analysis: parentage naming, parent-of-origin allele tracing, PCA of founder lineages.

## Results

Single run: 8 generations, 20 founders, 626 agents, 792 posts on "Moltbook" (simulated social network; ~$70/run). **No external benchmark or task environment** — only the social simulation; no heritability statistics. Leadership drive 0.525→0.710; obsessive focus 0.775→0.601; longevity 0.463→0.209; hyperfocus syndrome hit 106 agents by gen 5; vocabulary diversity 0.42→0.19; parent–offspring topic inheritance 83%. Genetics-only replications (20 runs × 3 conditions, no LLM): standard selection 0.586→0.640 (95% CI 0.492–0.756) vs non-genetic baseline flat ~0.5.

## Overlap vs HeritAgent chain

| Element | Coverage |
|---|---|
| 1. Lifetime-acquired capabilities | **No** — traits hand-authored at founding; nothing mined from experience; no task trajectories |
| 2. Somatic/germline separation | No — no runtime learning layer at all |
| 3. Executable functional genome | **No** — genotype parameterizes personality descriptors injected as prompt text; no planner/memory/tool policies, skills, verification logic, or typed spec |
| 4. Causal Inheritance Gate | Weak partial — ablations (random mating, non-genetic baseline, no-DNA.md) are post-hoc experimental controls, not per-trait gates; authors concede patterns "may reflect prompt conditioning rather than purely genetic causation" |
| 5. Assimilation of validated traits | No — heritability by design (segregation), not earned |
| 6. Multi-generational inheritance, frozen weights | **Yes** — the paper's core |

## Limitations (stated)

Single-run behavioural results; missing intermediate baselines; no component ablations; arbitrary thresholds/effect sizes; centrally imposed selection; small population (selection confounded with drift); single-LLM dependency; scalability.

## Consequences for HeritAgent

- The cleanest foil for the paper's key contrast: **behavioural genome vs functional genome**. Genomebook inherits *who the agent is*; HeritAgent inherits *what the agent can do*. Cite early in Related Work (SPEC §3 framing confirmed).
- Their "prompt conditioning vs genetic causation" concession strengthens our argument that inheritance claims require controlled intervention — exactly what CIG operationalizes.
- Spec §3's figures verified: 26 traits, 60 diploid loci, 8 generations.
