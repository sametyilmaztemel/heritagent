# Attribution family: ShapleyFlow (2021) + CapaBench (2026) — detailed notes

## (a) ShapleyFlow

**Paper:** *Shapley Flow: A Graph-based Approach to Interpreting Model Predictions* — Jiaxuan Wang, Jenna Wiens, Scott Lundberg (U. Michigan). **AISTATS 2021** (PMLR v130; not NeurIPS). arXiv:2010.14592. Code: github.com/nathanwang000/shapley-flow.

### Mechanism
- Input: a **full causal DAG** over the system — feature nodes, latent/noise nodes, sink = model output f(x) (the model itself can be expanded into the graph).
- Attribution assigned to **edges**, not nodes: each edge carries its source's "foreground" value; removing an edge forces the target to keep the "background" value. Credit for f(x)−f(x′) computed via DFS path tracing, averaged over "boundary-consistent" orderings.
- Proven the **unique solution to Shapley axioms generalized to DAGs**; SHAP, ASV, Owen value, on-manifold SHAP fall out as special cases (choice of explanation boundary). Exact computation exponential → Monte-Carlo ~10,000 orderings.

### Results
Synthetic 10-node linear graphs; NHANES survival (9,932 people, 18 features); Census Income. Only Shapley Flow exactly recovers ground-truth direct AND indirect effects on linear models; on-manifold SHAP can flip attribution signs (correlated variables "steal" credit via non-causal paths).

### Relation to HeritAgent
Justifies full-graph Shapley as the attribution primitive for CIG stage 4 when genome interactions exist (SPEC §9). No agents, no persistence, no decisions from attribution.

## (b) CapaBench

**Paper:** *Attribution-Based Analysis and Optimization of Modular Agentic Workflows* — Yingxuan Yang + 16 co-authors (SJTU, UChicago, Toronto, Meituan). **Findings of ACL 2026** (DOI 10.18653/v1/2026.findings-acl.359). arXiv:2502.00510 — v1 (Feb 2025) titled *"Who's the MVP? A Game-Theoretic Evaluation Benchmark for Modular Attribution in LLM Agents"* with framework named CapaBench; v3/ACL renamed framework to **"ShapleyFlow"** (⚠ name collision with (a) — see novelty-ledger naming hazard).

### Mechanism
- Fixed 4-module agent pipeline — **Planning, Reasoning, Action, Reflection** — with a default baseline model (Llama3-8B-Instruct) in every slot.
- To evaluate a test model: swap into each of the **2⁴ = 16 module subsets (exact enumeration)**, measure task success v(S) per coalition, compute standard Shapley φᵢ. Cost: 16 full agent runs per model per benchmark. ACL version adds interaction-aware attribution and model-allocation recommendations.

### Results
- v1: 5 suites / 1,000+ tasks — WebShop (110), math (500), theorem proving (miniF2F: Coq/Lean4/Isabelle), Ubuntu OS/git, RoCo robot cooperation; 9 LLMs. ACL: 7 domains, 1,500+ tasks.
- Findings: contributions task-dependent (Reasoning/Planning dominate OS & robot tasks; Action dominates math/ATP, φ up to ~0.66); **Reflection ≈ zero, often negative**; mixing best-per-slot models from different LLMs beats the best single model (43.31% vs 37.50% shopping; 86.79% ATP); Shapley↔GPT-judge Pearson 0.81/0.77/0.67; rankings robust to default-model swaps (85% consistency).

### Relation to HeritAgent
Attribution strictly for evaluation/interpretability/model-allocation in a **static, hand-designed** workflow — modules are interchangeable model calls, not learned capabilities; no inheritance, no lineage. Position HeritAgent as moving attribution "from explanation/configuration to lifecycle governance."

## Combined takeaway for CIG design

- Exact enumeration is feasible only for tiny genomes (2⁴ here) → for EAG with k active skills, default to cheap marginal/LOO + randomized masking (cf. ASSAY) and escalate to Shapley/BAES sampling only when interaction signals appear (SPEC §9 stage 4).
- CapaBench's "Reflection ≈ 0" result is a caution: some genome components will have near-zero marginal value; CIG thresholds (τ_c) must be estimated against measurement noise, not set optimistically.

Sources: arxiv.org/abs/2010.14592 · proceedings.mlr.press/v130/wang21b.html · arxiv.org/abs/2502.00510 · aclanthology.org/2026.findings-acl.359/
