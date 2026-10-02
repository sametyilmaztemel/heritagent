# Adjacent Works — Broad Sweep (2023–2026)

Round-1 sweep for works close to HeritAgent's chain (lifetime acquisition → somatic store → causal gate → genome assimilation → generational inheritance). Sources: arXiv/OpenReview/SSRN via structured queries ("Baldwin effect LLM agents", "genetic assimilation agents", "somatic germline LLM memory", "Lamarckian LLM", "heritable skills multi-generational", "self-evolving agents survey", "agent memory evolution across generations", "causal validation skill library", …). Performed 2026-10-02.

## Closest matches

### ASSAY — "Not All Skills Help: Measuring and Repairing Agent Skill Libraries"
arXiv:2606.15390 (Jun 2026). Measures each skill's **causal contribution via randomized masking**, restructures the library offline, suppresses harmful skills per task. This is the closest thing to CIG's measurement core — but it operates **within one lifetime**: no inheritance across agent generations, no genome, no somatic/germline distinction, no assimilation decision.
*Implication:* cite prominently; consider adopting randomized masking as the cheap ablation estimator in CIG stage 2. Our differentiation: the measurement drives an **inheritance decision**, not library curation.

### Skill-α — "Progressive Agent Skill Generation via Reinforcement Learning"
arXiv:2608.01678 (Aug 2026). Skills built incrementally; every edit scored by a **rollback reward** — counterfactual execution before/after the edit on an anchored query. A causal per-edit gate, but no generations, no genome, no biological-assimilation framing.
*Implication:* the "counterfactual before/after an edit" pattern is prior art for CIG replay/ablation semantics — cite; do not claim counterfactual evaluation itself.

### AgentSpawn
arXiv:2602.07072 (Feb 2026). Dynamic spawning of child agents with **automatic parent→offspring memory transfer** during spawning (selective slicing). Closest on literal generational inheritance of acquired state — no validation gate, no typed genome, no Baldwin framing.

### LOGOS — "A Living Logic for AI Agent Teams That Evolve With Humans"
arXiv:2607.10878 (Jul 2026). Versioned "agent packs" (agents, tools, knowledge, tests, permissions, policies); learned prompts/skills/memories remain "untrusted release candidates" until **held-out execution evidence** and authorization allow "promotion". A validation-before-assimilation gate plus a genome-like artifact — but governance/compliance-oriented, no trait-level causal attribution, no generational lifecycle.
*Implication:* cite as concurrent evidence that gating learned artifacts before assimilation matters; our gate is trait-causal and selection-driven, not authorization-driven.

## Relevant background

- **Voyager** (arXiv:2305.16291, 2023) — lifetime executable skill library, self-verification, frozen GPT-4. Canonical "acquire at runtime with frozen weights"; skills transfer across environments, not generations; verification = execution success, not causal ablation.
- **MUSE-Autoskill** (arXiv:2605.27366, May 2026) — full skill lifecycle (creation, memory, management, evaluation, refinement); cross-task and cross-framework skill transfer. No generations/genome/causal gate.
- **CASCADE** (arXiv:2512.23880, Dec 2025) — cumulative agentic skill creation via meta-skills and memory consolidation; shareable across agents. Cumulative, not multi-generational.
- **A Survey of Self-Evolving Agents** (arXiv:2507.21046; TMLR) — taxonomy: *what* evolves (model/memory/tools/prompts-workflows), *when* (intra- vs inter-test-time), *how* (reward signal, textual feedback, single/multi-agent). Nothing on somatic/germline separation, causal gates, or genome assimilation. Good Related Work scaffold.
- **OMEGA/RECLAIM — "Cultivating Machine Intelligence"** (arXiv:2605.25062, May 2026) — position paper with an explicit Baldwin-effect/genetic-assimilation section (lifetime Polya-Hebbian learning + across-generation variation/selection). Concept only, no implementation, not LLM-agent-specific.
- **"When Does Diversity Rescue Evolutionary Search of LLM-Based Agents?"** (SSRN ~2026) — evolves a six-gene "agent genome" of design components; genome as search-space parameterization, not lifetime-assimilated traits.
- **"Break It Down, Pass It On: Cross-Task Skill Transfer in LLM Agents"** (2025) — induces skills from completed tasks for later reuse.

## Sweep verdict

No single work combines lifetime acquisition → somatic store → causal gate → genome assimilation → generational inheritance as one pipeline. Pieces that are individually well covered (skill libraries, causal skill measurement, promotion gates, archives/generations, Baldwin rhetoric) are listed in `../novelty-ledger.md` §3 as do-not-claim items. Open ground: somatic/germline store separation (zero hits), typed executable functional genome as first-class inherited artifact, causal validation as a **prerequisite** for germline assimilation, and an implemented (not rhetorical) Baldwin/assimilation loop for LLM agents.

Caveat: one structured sweep; arXiv search coverage is imperfect (e.g., Workshop papers, non-indexed preprints). Re-run before submission (see ledger header).
