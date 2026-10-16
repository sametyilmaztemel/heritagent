# ADR-0011 — EXP-0001 runs on Qwen3.8-27B

Date: 2026-10-16 · Status: **ACCEPTED by the owner** (Samet, 2026-10-16, explicit answers in chat: the family, "qwen3.8 ailesinden bir üyeyle çalışalım", after the research below; then "Kabul, Qwen3.8-27B ile devam" to a question that named D2's determinism change, D7's compute, D6's 60 % rule, D4's message role and D5's kept values) · Origin: the owner asked for a current model family (2026-10-16, "araştır ve güncel model ailelerine geçelim") · Amends: ADR-0002 (the model and its reason (a); the determinism policy), ADR-0008 D4 (batch-invariant serving), ADR-0009 D4 (the window's reason), the manifest's model block and determinism lines · Research: workflow `wf_929b66f4-d0f` (four sweeps, a synthesis, two skeptics on another model), and builder-2's reads of the model's files and of vLLM 0.29.0's source.

Status keys: V verified at a primary source (model files from the Hugging Face API, vLLM 0.29.0's source tarball), R read from a secondary source, M arithmetic or reasoning, U unverified.

## Problem
EXP-0001 was designed on Qwen2.5-7B-Instruct (September 2024), chosen for SkillRL comparability, a single GPU and vLLM's batch-invariant mode. The owner wants a current model family.

## The family and its member
- Qwen3.8 is Alibaba's current open family (Qwen's GitHub news, 2026-08-14) [V]. Its members: Qwen3.8-27B (dense), Qwen3.8-2.4T-A95B, and Qwen3.8-Flash-Next (125B total, about 360 GB of bf16 files) [V]. Only the 27B fits one A100 80 GB.
- `Qwen/Qwen3.8-27B` [V, HF API and the repository's files, 2026-10-16]:
  - Apache-2.0, not gated; hub commit `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` today (the run pins the commit current at its first contact, as now).
  - 27.78B parameters in bf16, 55.56 GB of safetensors, a vision-language checkpoint (`Qwen3_5ForConditionalGeneration`).
  - 64 layers: 48 Gated-DeltaNet (linear attention) and 16 full attention (4 KV heads, head size 256).
  - Native context 262,144; vocabulary 248,320.
  - Thinking is on by default and is switched off through the chat template (`enable_thinking: false`).
- vLLM 0.29.0 (the pinned version) registers `Qwen3_5ForConditionalGeneration` and has `--language-model-only` (text only, vision tower not loaded) [V, source].

## Decisions
- **D1. The model.** G0 and the miner's teacher (the same model, as pre-registered) are Qwen3.8-27B in bf16, served text-only (`--language-model-only`). Only an A100 80 GB is admitted (ADR-0008 D1, unchanged). The 40 GB card cannot hold the weights: about 50.8 GiB without the vision tower [M].
- **D2. Determinism (amends ADR-0002's policy and ADR-0008 D4).** vLLM's batch-invariant mode refuses Gated-DeltaNet layers: `_cached_get_mamba_attn_backend` raises when `VLLM_BATCH_INVARIANT` is set and the GDN backend has no batch-invariant support [V, `vllm/v1/attention/selector.py` at 0.29.0; the same refusal is reported for Qwen3.8-27B on 0.28.0 in vLLM issue #42960, R]. So the server runs without batch-invariant kernels.
  - Kept: each (task, condition) is executed exactly once at the frozen settings, temperature 0 is greedy, the engine seed is 0, and repeated runs are never independent evidence.
  - Changed: a temperature-0 reply may now depend on what else is in the batch, so a repeat of the same request can differ. This backend variation is not a stochastic source of the design and is never averaged over. It is noise in every paired contrast (each gate stage, stage 4A's paired regression, the final arms).
  - Measured: the pilot's `temperature_0_repeat_determinism` measure (6 repeated episodes) is the only evidence of its size, and the report states it.
  - Gained: prefix caching, which batch invariance did not guarantee, and faster default kernels.
- **D3. Thinking off.** Every request carries `chat_template_kwargs: {"enable_thinking": false}`, and the server's default is the same (`--default-chat-template-kwargs`) [V, both exist in 0.29.0]. With them, the template prefills an empty `<think>\n\n</think>\n\n` and adds no reasoning instructions [V, `chat_template.jinja`]. The token counter renders prompts with the same arguments. The run-start record and the pilot report record them. G0's prompt text is unchanged (ADR-0003).
- **D4. The teacher's message.** Qwen3.8's template refuses a request without a user message ("No user query found in messages.") [V, `chat_template.jinja`]. The miner sends its teacher prompt as a single system message today, so its first call would fail after the 540 lifetime episodes. The teacher prompt is sent as one **user** message instead. Its text, template and hash are unchanged. The pilot's teacher-prompt counts render it the same way.
- **D5. Kept as locked.**
  - The 32,768-token window (ADR-0009 D4), served with `--max-model-len 32768`. Its reason is now the pre-registered protocol, no longer the model's native length.
  - 512 tokens per request, the miner's 2,048, τ_k = 512 tokens, the caps (96 active hours, 7,000 episodes, 225,000 calls), the budgets, seeds and splits.
  - The pilot manifest and its `pilot_sha256` (it names no model).
  - Every token number now counts Qwen3.8's tokens, a different unit from Qwen2.5's [M].
  - Neutral sampling (top_p 1, top_k 0, presence and repetition penalties off). Qwen recommends a presence penalty of 1.5 for its non-thinking mode [V, model card]. The protocol keeps one sampling rule for every model call, and the pilot measures parse failures and retries.
- **D6. Expectations and the pilot.** No ALFWorld result for Qwen3.8-27B was found [U]. Qwen3.6-27B, its predecessor at the same size, scores 61.7 % with no skills on ALFWorld (HDSO, 140 valid_seen tasks, 30 steps, sampled, three seeds) [V by a skeptic]. EXP-0001's design expected G0 near 30 % (design §8). A new calibration pilot runs on Qwen3.8-27B before anything else. If it measures G0 above 60 % at temperature 0, or its temperature-0 repeats differ, builder-2 reports and the owner decides how to go on (owner, 2026-10-16: "Bana sor").
- **D7. Compute.** Decoding reads about 3.6 times the weights per token that Qwen2.5-7B reads, so episodes take roughly 3–4 times as long at the same number of calls [M]. A stronger model may need fewer calls per episode [M]. The KV cache holds about 16 sequences of 16k tokens alongside the weights; vLLM queues the excess [M]. The compute-cap decision follows the new pilot's projection, as before (takeover note §5.12).
- **D8. Comparability.** ADR-0002's reason (a), the same base model as SkillRL, no longer holds. Arm B stays a frozen-weights reconstruction of SkillRL-style skill persistence, not a numeric replication. The publication states the model and this ADR's reasons: the owner's requirement for a current family, and the determinism trade-off of D2.

## Alternatives considered (research `wf_929b66f4-d0f`)
- **Qwen3-8B with thinking off.** On vLLM 0.29.0's batch-invariance tested list [V]; the most common frozen base of 2026 ALFWorld skill papers [R]; 47–53 % on valid_unseen in three of them [V by a skeptic]. It keeps D2's guarantee. The research recommended it; the owner chose the current family.
- **Qwen3-4B-Instruct-2507.** Non-thinking only; 38.8 % in the harness closest to EXP-0001 (SkillBrew) [V by a skeptic]. Not on the tested list in bf16.
- **Qwen3.5-9B.** A current family with the same GDN limitation as Qwen3.8.
- **Gemma 4 12B, gpt-oss-20b.** Gemma 4 12B has no ALFWorld evidence and is not on the tested list. gpt-oss-20b's reasoning channel cannot be switched off.
- **Keeping Qwen2.5-7B.**

## Research implication
H2 is still tested by C against B on the same frozen candidates. D2 adds backend noise to every paired contrast, and D6 may leave less headroom for skills. Both weaken detectability, and the report must disclose them (SPEC §34). A null result stays a finding (design §6.5).

## Experiment needed
The calibration pilot on Qwen3.8-27B (127 episodes, pilot manifest unchanged), after a CPU rehearsal with Qwen3.8's tokenizer (`EXP-0001-rehearsal.ipynb`). Then the owner's decision per D6, and the compute cap per D7.

## Compatibility impact
- **Locked, edited on acceptance:**
  - the manifest's `model.name` and determinism lines;
  - ADR-0002 and ADR-0008 D4 (pointers to this ADR);
  - ADR-0009 D4's reason;
  - the design's model line in §2, with a note that §8's G0 expectation was written for Qwen2.5-7B.
- **Engineering:**
  - `DEFAULT_MODEL`;
  - `server_command` (`--language-model-only`, `--default-chat-template-kwargs`, no batch-invariant environment);
  - the adapter's `chat_template_kwargs`;
  - the token counter's template arguments;
  - the miner's message role (D4);
  - the weights hash covering `chat_template.jinja`;
  - `preflight` with the real planner and teacher message shapes;
  - the notebooks' `MODEL`;
  - tests.
- **No new software pins** [M; the A100 confirms the server side].
