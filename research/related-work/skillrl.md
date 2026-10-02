# SkillRL — detailed notes

**Paper:** *SkillRL: Evolving Agents via Recursive Skill-Augmented Reinforcement Learning* — Peng Xia, Jianwen Chen, Hanyang Wang, Jiaqi Liu, Kaide Zeng, Yu Wang, Siwei Han, Yiyang Zhou, Xujiang Zhao, Haifeng Chen, Zeyu Zheng, Cihang Xie, Huaxiu Yao. arXiv:2602.08234 (v1 9 Feb 2026). Code org aiming-lab/SkillRL (Huaxiu Yao's lab); README states **accepted at NeurIPS 2026**. Reviewed 2026-10-02.

## Mechanism

- Base model Qwen2.5-7B-Instruct; teacher OpenAI o3.
- **Extraction:** rollouts collect success (T+) and failure (T−) trajectories; teacher does *differential distillation* — strategic patterns from successes, concise "failure lessons" from failures (~10–20× token compression vs raw storage).
- **SkillBank:** two-level hierarchy — *general skills* (always in context: precondition checks, goal tracking) and *task-specific skills* (embedding-similarity retrieval, threshold δ, Top-K=6). Each skill = {name, principle, when_to_apply}.
- **RL trains weights** (not just skill selection): GRPO with binary success rewards, KL-anchored to an SFT cold-start (~7,500 ALFWorld / 2,400 WebShop skill-augmented teacher demos; lr 1e-6, KL 0.01, 8×H100, ~30 h).
- **Recursive co-evolution:** at validation checkpoints (every 5 steps), weak categories (success < 0.4) trigger teacher-driven skill addition/refinement (≤3 new skills/update). Library grew 55 → 100 skills during training.
- Skills injected into the prompt each step; agent reasons in `<think>` tags citing skill IDs.

## Results

- ALFWorld: 89.9% (GRPO 77.6, RLOO 75.5, Reflexion 42.7, ExpeL 46.3, GPT-4o 48.0, Gemini-2.5-Pro 60.3).
- WebShop: 85.2 score / 72.7% success (GRPO 79.3/66.1).
- Search QA (7 tasks incl. HotpotQA, MuSiQue, Bamboogle): 47.1% avg (EvolveR 43.1, Search-R1 38.5; Bamboogle 73.8%).
- Ablations (ALFWorld/WebShop): no hierarchy 76.8/61.4; raw trajectories 61.7/50.2; no SFT cold-start 65.2/46.5; no dynamic evolution 84.4/70.3.

## Overlap vs HeritAgent chain

| Element | Coverage |
|---|---|
| 1. Trajectory-mined capabilities | **Yes** — success+failure differential distillation |
| 2. Somatic/germline separation | No — skills and weights are just training artifacts |
| 3. Executable genome | Primitive partials only — `when_to_apply` ≈ regulation, hierarchy ≈ typed organization; SkillBank is a prompt library, not a typed/versioned whole-agent spec |
| 4. Causal Inheritance Gate | No — persistence is heuristic (weak-category trigger, ≤3/update, Top-K manages bloat); no per-skill ablation, attribution, held-out gate, or regression tests |
| 5. Assimilation into heritable genome | No — skills stay prompt entries or get absorbed into weights via SFT/GRPO |
| 6. Multi-generational, frozen weights | No / contradicted — weights fine-tuned; "recursive evolution" is within one training run |

## Limitations (implicit — no explicit section)

Strong closed-source teacher dependence (o3) for distillation/SFT data; context-length ceiling on injected skills; skills alone transfer poorly without cold-start SFT (relevant negative evidence for pure frozen-weight skill transfer — HeritAgent must engage this in H4/H1 analysis); follow-up work (arXiv:2603.28716) notes privileged-teacher reliance.

## Consequences for HeritAgent

- Trait Miner v0: adapt their differential distillation (success/failure → skill with `when_to_apply`) — cite as adapted mechanism.
- Their persistence policy IS our **arm B (unrestricted inheritance)**: rate-limited growth + Top-K retrieval ≈ SkillRL's library dynamics without weight training. Direct comparability on ALFWorld/WebShop (same base model Qwen2.5-7B-Instruct is a deliberate choice for EXP-0001).
- Their "no cold-start SFT → skills transfer poorly" ablation is a threat to H4 we must measure, not assume away: expression layer quality matters when the genome crosses backbones.
