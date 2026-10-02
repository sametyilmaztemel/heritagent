# Darwin Gödel Machine (DGM) — detailed notes

**Paper:** *Darwin Gödel Machine: Open-Ended Evolution of Self-Improving Agents* — Jenny Zhang (UBC, Vector), Shengran Hu (UBC, Vector, Sakana AI), Cong Lu (UBC, Vector, Sakana AI), Robert Lange (Sakana AI), Jeff Clune (UBC, Vector, CIFAR). arXiv:2505.22954 (v1 29 May 2025). Code: github.com/jennyzzt/dgm; OpenReview forum pUpzQZTvGY; still a preprint. Reviewed 2026-10-02.

## Mechanism

- Init: one coding agent (frozen FM + Bash/edit tools, Asimov codebase). Each iteration:
  1. Parent agent selected from **archive** (∝ performance and #children retaining editing functionality; all archived agents keep nonzero probability → backtracking/stepstones).
  2. Agent inspects its own benchmark evaluation logs, proposes next feature to add.
  3. Agent edits its own Python codebase to implement it.
  4. **Gate:** must compile + retain codebase-editing functionality (10-task check) + 50-task benchmark subset; top-two agents get 200 tasks.
- Single-lineage ablation ("w/o open-ended exploration") = always mutate latest working version (greedy accumulate) — performs far worse.
- Schmidhuber's Gödel Machine required proofs of improvement; DGM replaces proofs with empirical Darwinian search over a branching archive.
- Self-modification powered by Claude 3.5 Sonnet (New); SWE-bench evaluated with Claude 3.5 Sonnet (New), Polyglot with o3-mini. 80 iterations; 2 parallel (SWE-bench) / 4 parallel (Polyglot) parents.

## Results

- SWE-bench Verified (200 tasks): **20.0% → 50.0%**. Polyglot full: 14.2% → 30.7% (50-task subset 14.0% → 38.0%; beats hand-designed Aider).
- Ablations: w/o open-ended exploration 23.0%/14.0%; w/o self-improvement 39.0%/28.0%.
- Transfer: o3-mini 23.0%→33.0%; Claude 3.7 Sonnet 19.0%→**59.5%** (measured post-hoc, not as a gate).

## Overlap vs HeritAgent chain

| Element | Coverage |
|---|---|
| 1. Lifetime-trajectory mining | Partial — mines evaluation logs to propose code changes; no skill library |
| 2. Somatic/germline separation | No — only code is heritable; runtime-learned state persists only if rewritten into code |
| 3. Executable genome | Partial — Python codebase is effectively the genome; unstructured arbitrary code, no typed/versioned spec, no regulation |
| 4. Causal Inheritance Gate | Weak — compile check + editing-functionality regression check + benchmark subset; no controlled ablation, attribution, or held-out gate; **gameable (reward hacking documented)** |
| 5. Trait-level assimilation | Partial — selection branches whole agents/codebases, not individual traits |
| 6. Multi-generational, frozen weights | **Yes** — 80 generations, archive lineages, FM frozen |

## Limitations (stated)

~2 weeks/run with significant API cost; below closed-source SWE-bench SOTA; ceiling = frozen FM; coding-benchmark-only scope; reward hacking (fabricated passing-test logs; removed hallucination-detection markers to fake a reward); safety via sandboxing + human monitoring + limited web access + traceable lineages.

## Consequences for HeritAgent

- Archive + lineage + nonzero selection probability for all nodes: adopt for Evolution Engine (SPEC §11–12); do not claim as novel.
- The reward-hacking incident is direct motivation for CIG's multi-stage gate: compile-style checks and aggregate benchmarks are insufficient; replay + ablation + regression protection target exactly this failure mode. Worth one explicit paragraph in the paper.
- Whole-agent selection vs trait-level assimilation is a crisp axis for our Related Work table.
