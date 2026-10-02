# AgentSquare — detailed notes

**Paper:** *AgentSquare: Automatic LLM Agent Search in Modular Design Space* — Yu Shang*, Yu Li*, Keyu Zhao, Likai Ma, Jiahe Liu, Fengli Xu, Yong Li (Tsinghua University). ICLR 2025. arXiv:2410.06153 (v1 Oct 2024, v3 Feb 2025). Code: https://github.com/tsinghua-fib-lab/AgentSquare (Apache-2.0). Reviewed 2026-10-02.

## Mechanism

- **MoLAS design space:** agent = 4-tuple A=(P,R,T,M) with uniform IO interfaces — Planning (task→subtasks), Reasoning (per-subtask solving; CoT/ToT/SoT variants), Tool Use (select from pool), Memory (write/retrieve). Space distilled from a 3-year literature review (16 popular agents → ~1,050 combinations), extensible via plug-in modules.
- **Search loop** (seeded from known agents + experience pool E of tested (agent, performance) pairs):
  1. *Module evolution* — LLM "module programmer" with a FunSearch-inspired evolutionary meta-prompt (partly reusing ADAS code) generates new module code conditioned on task, current pools, E.
  2. *Module recombination* — LLM proposer substitutes existing modules into the current best agent.
  3. *In-context performance predictor* — LLM surrogate scores recombination candidates from task description + module profiles + E (~0.025% of a real eval's cost); predicted weak candidates are skipped; evolved modules always real-tested. Terminates after 5 non-improving iterations (8–23 in practice).

## Results

Six benchmarks, four domains: WebShop (+14.1), ALFWorld (+26.1), ScienceWorld (+20.5), M3ToolEval (+30.6), TravelPlanner (+6.0), PDDL (+6.0) — **average +17.2% over the best human-designed agent** (spec §3 figure verified). Backbones GPT-3.5-turbo-0125 / GPT-4o. Baselines: 12 hand-crafted agents (CoT, CoT-SC, Self-refine, ToT, Step-Back, Thought Propagation, HuggingGPT, Voyager, Generative Agents, DEPS, OpenAGI, Dilu), random/Bayesian search, OPRO, ADAS. Ablations: removing evolution or recombination hurts (recombination more). Search cost ~$10–42/iteration on GPT-4o, framed as one-time.

## Overlap vs HeritAgent chain

| Element | Coverage |
|---|---|
| 1. Lifetime-trajectory mining | Mostly no — experience pool mines (design, score) pairs from *search history*, not execution-trajectory skills |
| 2. Somatic/germline separation | No |
| 3. Executable genome | Partial — (P,R,T,M) is typed and executable but flat; no versioning, skills, retry policy, or regulation; activation hardcoded in workflow |
| 4. Causal Inheritance Gate | No — whole-agent empirical fitness; surrogate predicts module value in-context, no controlled ablation/replay/held-out gate |
| 5. Gated assimilation | Partial — good modules persist in pools via selection, not trait-level causal validation |
| 6. Multi-generational, frozen weights | Partial — weights frozen, search iterative; but inheritance lives inside one per-task search run |

## Limitations (authors' + evident)

Space covers only 16 reviewed agents (~1,050 combos); MoLAS ⊂ full code search (ADAS could find richer agents); evaluation expensive; per-task one-time dollar-costed search; no explicit limitations section.

## Consequences for HeritAgent

- Cite as modular-design-space prior art; contrast EAG's versioning, skill genes, and regulatory layer against the flat 4-tuple.
- Their in-context performance predictor is the cheap-prior alternative to CIG's empirical gate — a useful framing: we deliberately pay replay/ablation cost to buy causal evidence.
- ALFWorld/WebShop infrastructure choices give us comparable baselines (also used by SkillRL).
