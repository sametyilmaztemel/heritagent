# The Evolution of Sex for AI (Gilestro 2026): detailed notes

**Paper:** *The evolution of sex for artificial intelligence: a population-genetic framework for multigenerational model populations* by Giorgio F. Gilestro (Dept. of Life Sciences, Imperial College London). Single author. arXiv:2609.18560v1, submitted 16 Sep 2026, primary category cs.LG. The sweep also listed cross-lists cs.NE and q-bio.PE; I could not confirm those from the abs page. Code and artifacts are reported at git.lab.gilest.ro/giorgio/MachineSex; I did not open that link. Reviewed 2026-10-12.
**Read depth:** full main text (22-page arXiv PDF including Methods and references). The Supplementary Information (Table S2, Texts S1–S2, Figs. S1–S15, SI Methods) is not in the arXiv PDF and was not read. Statements below that depend on the SI are marked as such.

## Mechanism

- **Framing.** The paper maps multigenerational *model* populations (fine-tunes, merges, models retrained on model output) onto population genetics, and tests the mapping at three tiers:
  - an exact Wright–Fisher inheritance model (NumPy);
  - trained networks (RNN/MLP/VAE generators on synthetic oracles, and a conv-VAE on MNIST);
  - LoRA specialists on frozen Qwen2.5-Instruct bases (0.5B, 1.5B, 7B).

  The paper's own glossary pairs each biological term with an operation:

  | Population-genetics term | Model-population operation |
  |---|---|
  | item / allele | capability, fact or behaviour mode |
  | genetic drift | model collapse |
  | immigration | "grounding", i.e. verified real data added each generation |
  | recombination / sex | weight-space merging, or routing |
  | blending inheritance | weight averaging |
  | Fisher–Muller effect | merged specialists beating every parent |
  | Bateson–Dobzhansky–Muller incompatibility | merge failure from conflicting answer conventions |
  | seed bank | merging with one's own ancestor |
  | recombination modifier | declinable merge |
  | selection on fitness | "verifier-anchored selection" |

- **Multigenerational LLM population (the part that matters for us).**
  - **Setup:** 3 lineages × 6 generations × 3 training seeds. Every lineage starts from one frozen Qwen2.5-1.5B base.
  - **Skill acquisition:** each generation, every lineage learns one new skill by supervised LoRA training (rank 16; 300 new plus 150 replay examples; 3 epochs). The six skills come from public datasets (MNLI, ARC-Easy, HellaSwag, SQuAD, BoolQ, WinoGrande), taken in Latin-square order. Each skill is scored by its own exact-match or execution verifier.
  - **Inheritance:** a child starts from its parent's adapter **as it stands** and keeps training it. The author explicitly calls this inheritance of acquired characters, "that Lamarck proposed and biology rejected, and that a weight file makes trivial".
  - **Replay:** examples from the lineage's earlier skills are replayed each generation (framed as grounding).
- **Merge arms:**
  - never merge;
  - obligate merge with a contemporary from another lineage (with verified or self-generated replay);
  - merge with one's own ancestor from three generations back;
  - **declinable merge**: the unchanged parent is scored as one candidate beside every merge (weights {0.5/0.5, 0.3/0.7, 0.7/0.3}, chosen on 20 validation items per family, reported on 60 held-out items) and wins if no merge beats it;
  - a forced-stop control: obligate merging through generation 2, then never;
  - a supplementary truncation-selection arm: the lowest lineage is re-founded from the highest (Fig. S15; SI).
- **Other operators:** routing among intact specialists; "directed recombination", in which Dirichlet-sampled offspring are screened on held-out validation; and a pre-merge **functional-disagreement probe**, in which both parents answer the same probes and the share of differing answers predicts merge damage.

## Results (main text; per-seed detail is in SI Table S2, not read)

- **Grounding is a count, not a fraction.** In the inheritance model, about 10 real samples per generation keep ≥95% diversity whatever the training-set size (g ≈ 0.05 at n = 200). The conv-VAE needed about 10% real data (30 → 1 modes without it, 30/30 kept with it).
- **Blending conservation law.** Averaging N parents leaves a rare item's expected mass and survival unchanged. A union operator raises retention.
- **LLM merges, easy tasks (Qwen2.5-0.5B, 5 seeds).** Both merges beat the best specialist on the weakest task family (p < 1e-4). TIES beats it overall (p = 0.006).
- **LLM merges, hard tasks (Qwen2.5-7B, 3 seeds).** Averaging only matches the best specialist (p = 0.96). Routing beats averaging in every seed (p = 0.007).
- **Six-generation population:**
  - Obligate merging collapsed from 0.65 to 0.27, starting when partner complementarity fell below 0.8.
  - The declinable merge finished at 0.792, level with never-merge (0.796), the forced-stop control (0.793) and one model taught the whole syllabus (0.80).
  - In the declinable arm, the share of declined merges rose from 0.44 to 1.00. Declines tracked generation (partial ρ = 0.31; 0.45 in the conflict-placement curricula), not complementarity (ρ = −0.07) or conflict (ρ = −0.09).
  - Own-ancestor merge beat contemporary merge in every seed (0.66 vs 0.27).
  - Truncation selection raised early means but not final levels. The author's summary: recombination and selection "each reach it sooner without raising it".
- **Reproductive isolation.** Isolation appears only with conflicting conventions (e.g. yes/no vs 1/2). It does not appear with divergence or long specialisation: 6.4× base training in MLPs, 1–12 epochs in LoRA, and the merge improved from 0.76 to 0.95.
- **Merge-damage prediction.** On 39 pairs (13 conditions × 3 seeds), functional disagreement correlates with the merge penalty at ρ ≈ +0.45 (CI excludes 0; ≈ 0.35–0.40 leaving one condition out). LoRA cosine and distance show no association once a compatible-overlap control is added; cosine falls from +0.60 to +0.03.
- **Four-arm NK-landscape ablation** (inheritance model only; 60 agents, 12 loci, 80 generations): removing grounding, recombination or diversity each causes a different failure.

## Overlap vs HeritAgent chain

| Element | Coverage |
|---|---|
| 1. Lifetime acquisition | **Partial.** Each lineage acquires one new skill per generation, and what is acquired in a lifetime is inherited (explicitly Lamarckian). But acquisition is curriculum supervised fine-tuning on public labelled datasets. Nothing is mined from the agent's own trajectories or experience, and there is no agent loop (no tools, no environment interaction). |
| 2. Somatic/germline separation | **Absent, and the design is the opposite.** The trained adapter is both the working state and the inherited state, passed on "as it stands". No separate heritable store and no Weismann barrier. The paper is a clean Lamarckian contrast to our Weismannian design. |
| 3. Typed executable genome | **Absent.** The inherited unit is LoRA weight deltas. "Locus" and "allele" are used conceptually for answer conventions ("the unit of inheritance in a model population is therefore a convention"). The NK tier's 12-locus genotypes are abstract vectors. Nothing is typed, versioned at trait level, or executable as policies/skills/regulation. |
| 4. Causal validation gate | **Partial (weak).** The declinable merge accepts a recombination event only if the merged child beats the unchanged parent on held-out validation items. That is an interventional, regression-protecting accept/reject test, but only at the level of the whole merge event and on aggregate accuracy. There is no per-trait ablation or attribution, and no replay → ablation → attribution chain. The lifetime-acquired skill itself is never gated: every child inherits the full adapter automatically. The paper's own forced-stop control shows the gate's outcome is "explained by when it stopped and not by which merges it chose". |
| 5. Selective assimilation | **Partial.** There is selection over recombination (declinable merge, screened offspring, routing over averaging) and between lineages (truncation; SI). There is no selection of individual acquired traits into a heritable genome. Assimilation of a lineage's own new skill is total and automatic. |
| 6. Multi-generational, frozen weights | **Partial (strong).** Six generations, 3 lineages, and the foundation base stays frozen throughout. But the heritable artifact *is* trained weights (the adapter), so this is "frozen foundation, inherited weight deltas", not inheritance without weight change. Scale is small: one 1.5B base, 3 seeds, 6 generations, nothing varied for base model or replay. |

**Threat level: medium for the framing, low for the core claim.** This is the clearest recent example of biology-framed, multigenerational inheritance of acquired skills on a frozen foundation model, with matching vocabulary (Lamarckian, drift, Fisher–Muller, seed bank, recombination modifier, verifier-anchored selection). Any HeritAgent sentence implying that this framing or setup is new is now contradicted. It does not touch somatic/germline separation, the typed executable genome, or trait-level causal gating of assimilation. Those differentiators stand, and the contrast is sharper because this paper is explicitly Lamarckian.

## Limitations (stated or evident)

Stated by the author:
- one base model, six generations, replay throughout, and no differential reproduction in the main runs ("none of which was varied");
- three seeds, so some every-seed differences are not significant;
- the merge-predictor rows are not independent (condition-clustered inference);
- adapter age and skill count are confounded by construction;
- epistasis between real skills cannot yet be measured;
- the fitness function / selective environment is called "the open problem".

Evident:
- tasks are labelled NLP benchmarks, not agentic environments;
- the "skills" are dataset-level task families, not reusable procedures;
- the SI, which holds most per-seed numbers and the Fig. S15 selection arm, was not inspected for this note.

## Consequences for HeritAgent

- **Writing.** Cite as the Lamarckian / weight-delta counterpart.
  - Wording: "Gilestro (2026) inherits the whole acquired adapter across generations on a frozen base; HeritAgent separates somatic from germline state and assimilates only gated, typed traits."
  - Do not claim biology-framed multigenerational skill inheritance on a frozen base, the Lamarckian framing, or verifier-anchored selection as ours.
  - Gloss their declinable merge as *validation-gated recombination* and our CIG as *trait-level causal gating of germline assimilation*.
- **Methodological threat to H1-type claims, which we must engage, not dismiss.**
  - (a) A single model taught the whole curriculum matched the population's ceiling (0.80), and inheritance, recombination and selection only changed speed. Our EXP design needs an "one agent taught everything" baseline and should report adaptation speed separately from final level.
  - (b) Their gate's benefit was fully explained by a timing-matched forced-stop control. Any CIG-vs-unrestricted comparison should add a non-selective, schedule-matched control (e.g. a random accept with the same acceptance rate per generation). This guards against the gate merely acting as an early stop on genome growth.
- **Obligate-merge collapse ≈ our arm B (unrestricted inheritance) failure mode.** It is useful supporting evidence that ungated accumulation can collapse, but in their data the collapse is driven by convention conflicts plus generation count, not by bloat.
- **Typed EAG design hint.** "Conventions as alleles at one locus" maps naturally onto typed trait interfaces. EAG merge/crossover could declare output conventions per trait type and flag same-locus conflicts before assimilation. Their functional-disagreement probe (ask both parents the same questions) is a cheap pre-assimilation compatibility check worth adapting and citing.
- **Replay budgets.** Specify regression-protection replay sets in absolute examples per trait, not as a fraction of the training mix.
