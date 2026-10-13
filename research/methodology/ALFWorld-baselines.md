# ALFWorld empirical anchors for EXP-0001 power analysis and pre-registration

Author: independent Claude researcher subagent. Date: 2026-10-12. Scope: read-only research (no repo edits, no GitHub writes).
Reads: `experiments/manifests/EXP-0001-design.md` and `EXP-0001-minimal-cig.yaml` (main @ 0ce0a36).
Status convention: **V** = number read by me in the cited source text (paper HTML/PDF, repo file) during this session. **U** = not read in a source, or only inferred; do not cite as fact. SPEC section 34 anti-overclaim applies. Items that would change a locked or pre-registered element are marked **requires ADR** and are observations only.

Scripts used: `research/methodology/scripts/power.py`, `research/methodology/scripts/checks.py` (venv314 python).

---

## 0. Findings that matter most (read this first)

1. **Published Qwen2.5-7B-Instruct "no training" ALFWorld numbers span 14.8% to 45.5%, and they are not on one protocol.** The 14.8 / 31.2 / 42.7 family (GiGPO, SkillRL, many copy-papers) does not state its eval split in the paper. The verl-agent code that produced them defaults to **valid_seen (`eval_in_distribution`, 140 games), 128 validation envs, validation temperature 0.4** (repo files read; the paper itself does not say). Numbers on the official **valid_unseen (134)** at temperature 0 with 50 steps exist from two independent 2026 papers: 29.9% zero-shot (ExpRAG paper) and 45.5% base (SAGA paper). A 30-point spread across prompt formats for the same weights means G0's prompt template, not the model, will dominate arm A's level.
2. **No published training-free, skill-gene-style number for Qwen2.5-7B-Instruct on valid_unseen exists that matches EXP-0001's arm B/C.** Closest: SAGA trajectory-only memory 45.5 -> 46.3 / 57.5 / 46.3 across rounds R0-R3 (non-monotone, single run each) and complete-hierarchy memory 45.5 / 48.5 / 54.5 / 59.4. SkillRL itself reports no training-free skill row and says (citing its ref [12]) that "simply providing skills to an unchanged model yields limited benefit".
3. **Power: n=134 paired tasks cannot support the pre-registered H2 criterion "C - B >= +5pp with CI excluding 0" unless the two arms disagree on very few tasks.** Cluster-bootstrap power to exclude 0 at a true +5pp is 0.45 / 0.25 / 0.16 / 0.13 for discordance rates 10 / 20 / 30 / 40 percent (section 5). A true +10pp needs discordance below about 25 percent for ~70 percent power. Observation only; changing thresholds **requires ADR**.
4. **T_reg feasibility problem.** The design defines T_reg as 40 tasks that G0 solved, drawn from the 60-task T_mine. That needs G0 success of at least 40/60 = 66.7% on T_mine. Every Qwen2.5-7B-Instruct anchor is 15-46%. At p = 0.45 the chance that G0 solves 40 or more of 60 is 0.0006 (binomial). Expected solved tasks at the anchored range: about 9 to 28. Observation; any fix (larger T_mine, smaller T_reg, different definition) **requires ADR**.
5. **Stage-2 gate is stricter than its stated tau_c = 0.05.** With 20 paired tasks and a percentile-bootstrap CI lower bound above 0, the pass region needs at least 4 wins and 0 losses (+20pp) or 7 wins and 1 loss (+30pp) (exact enumeration, section 5.3). With a 15-46% baseline, a single skill delivering +20pp is unlikely, so expect very few candidates to pass stage 2. Observation; **requires ADR** to change.
6. **Split sizes confirmed** (train 3553 / valid_seen 140 / valid_unseen 134, per-family counts below). Bank feasibility is not binding at the nominal counts: the tightest family (look_at_obj_in_light, 308 train games) has 14x the 22 per-family games needed for mine>=8 + replay>=4 + generalization>=10. **Caveat:** the TextWorld loader (`AlfredTWEnv`) filters games (movable/sliced paths, missing `game.tw-pddl`, `solvable` false/missing), and SEEA-R1 reports 3321 train games; per-family counts after filtering are unknown (U) and must be taken from the actual data at split-manifest build time.

---

## 1. ALFWorld split facts

Source for all counts: ALFWorld paper (Shridhar et al., ICLR 2021, arXiv 2010.03768), Table 1, read from the PDF text (V). Cross-checked against an independent 2026 paper (section 1.2).

### 1.1 Per-family counts (ALFWorld Table 1)

| Task family (ALFWorld / alias used by RL papers) | train | valid_seen | valid_unseen |
|---|---:|---:|---:|
| pick_and_place_simple ("Pick & Place", Pick) | 790 | 35 | 24 |
| look_at_obj_in_light ("Examine in Light", Look) | 308 | 13 | 18 |
| pick_clean_then_place_in_recep ("Clean & Place", Clean) | 650 | 27 | 31 |
| pick_heat_then_place_in_recep ("Heat & Place", Heat) | 459 | 16 | 23 |
| pick_cool_then_place_in_recep ("Cool & Place", Cool) | 533 | 25 | 21 |
| pick_two_obj_and_place ("Pick Two & Place", Pick2) | 813 | 24 | 17 |
| **All** | **3,553** | **140** | **134** |

Column sums verified by me: 790+308+650+459+533+813 = 3553; 35+13+27+16+25+24 = 140; 24+18+31+23+21+17 = 134. Total 3,827 = the "3,827 task instances" stated in the GiGPO paper (V).

### 1.2 Independent confirmations (V)

- ExpRAG paper (arXiv 2603.18272, Table 4): ALFWorld easy-type test = 73 (look 18 + clean 31 + pick_and_place 24) and hard-type test = 61 (cool 21 + heat 23 + pick2 17) = 134; easy train 1748, hard train 1805 = 3553.
- SAGA paper (arXiv 2610.06964, Table 6): valid_unseen 134 unique games, "Pick 24, Clean 31, Heat 23, Cool 21, Look 18, Pick2 17"; training pool listed as 2,432 games (Pick 790, Clean 650, Heat 459, Cool 533; Look and Pick2 withheld by that paper's own design).
- ALFWorld `alfred_tw_env.py` (V, raw GitHub): train/eval path selection by `train_eval` in {"train","eval_in_distribution","eval_out_of_distribution"}; verl-agent `config_tw.yaml` maps `eval_in_distribution` -> `json_2.1.1/valid_seen` and `eval_out_of_distribution` -> `json_2.1.1/valid_unseen`.

### 1.3 Caveat: the TextWorld loader filters games (affects T_mine/T_gate pool sizes)

`AlfredTWEnv.collect_game_files` (read in full, V) skips: paths containing `movable` or `Sliced`; games with no `game.tw-pddl`; games with no `solvable` key; games with `solvable == false`. It prints "Overall we have N games in split=..." (the actual N depends on the downloaded data). SEEA-R1 (arXiv 2506.21669, appendix F, V) states "a training set comprising 3321 games" with 140 seen / 134 unseen. I could not read per-family counts after filtering (U). The HeritAgent split manifest must record the loader-reported counts per family before run start. Even at 3321 the feasibility conclusion below holds with a wide margin.

### 1.4 Bank feasibility check against the design (arithmetic on the V counts)

Design needs: T_mine ~60 (min 8/family, stratified) + T_gate ~114 (ablation 20, interaction 10, replay_bank >=4/family = 24, generalization_bank >=10/family = 60), disjoint, all from train. Total train draw about 174 of 3,553 (4.9 percent).

| Family | train avail. | min need (8 + 4 + 10 = 22, excl. ablation/interaction share) | headroom |
|---|---:|---:|---:|
| Pick | 790 | 22 | 35.9x |
| Look | 308 | 22 | 14.0x |
| Clean | 650 | 22 | 29.5x |
| Heat | 459 | 22 | 20.9x |
| Cool | 533 | 22 | 24.2x |
| Pick2 | 813 | 22 | 37.0x |

Proportional T_mine quota for 60 tasks would be Pick 13.3, Look 5.2, Clean 11.0, Heat 7.8, Cool 9.0, Pick2 13.7; the per-family minimum of 8 binds for Look and Heat, leaving 12 tasks of free allocation. Conclusion: replay_bank >=4 and generalization_bank >=10 per family are feasible; the pre-registered family-specific maximum fallback should not be triggered at nominal counts. **Not checked / worth flagging as an observation:** train games share scenes (rooms) across families and instances; "disjoint instances" (design text) is weaker than "disjoint scenes". valid_unseen uses unseen rooms by construction (ALFWorld paper, V) while valid_seen and train share rooms. So T_eval measures room-generalization that T_gate cannot (T_gate shares rooms with T_mine). I did not verify the number of distinct train scenes (U).

Note also T_reg (subset of T_mine that G0 solved) is built from the exact tasks whose trajectories the miner saw; that makes the regression suite non-independent of the mined skills. Observation only; **requires ADR** if changed.

---

## 2. Published success rates for Qwen2.5-7B-Instruct (no weight training of the 7B policy unless stated)

All rows are copied from source text I read. "Split" states what the source says; where the source is silent I say so.

### 2.1 Table A: overall success, prompting / memory / skill methods on Qwen2.5-7B-Instruct

| # | Method | Success (%) | Split / protocol as stated | Source | V |
|---|---|---:|---|---|:--:|
| A1 | Qwen2.5-7B-Instruct, zero-shot, 134 games, static action-template prompt, greedy | **29.9** (29.85, = 40/134) | valid_unseen, 134; temp 0 (greedy); max steps not read | ExpRAG, arXiv 2603.18272, Table 17 / C.2 text | V |
| A2 | Qwen2.5-7B-Instruct "Base Model" (strict 7B protocol, token-prefix continuation, 16,384-token context) | **45.5** (=61/134) | valid_unseen, 134; temp 0, top-p 1, seed 42; step limit 50; no memory | SAGA, arXiv 2610.06964, Table 2 + Table 6 + Table 12 text | V |
| A3 | SAGA trajectory-only memory (L0/L1), rounds R0..R3 | 45.5 / 46.3 / **57.5** / 46.3 | same as A2; frozen memory built from train tasks each round | SAGA, appendix text | V |
| A4 | SAGA complete hierarchy, R0..R3 | 45.5 / 48.5 / 54.5 / **59.4** | same as A2; no L3 principle passed validation in the 7B config | SAGA, appendix text | V |
| A5 | Qwen2.5-7B-Instruct "Qwen2.5" prompt row | **14.8** | split not stated in the paper; verl-agent default eval is valid_seen (section 3) | SkillRL v2 Table 1; GiGPO Table 1 | V (number); split U |
| A6 | ReAct (Qwen2.5-7B) | **31.2** | split not stated; copied by SkillRL from GiGPO (SkillRL footnote *) | GiGPO Table 1; SkillRL Table 1 | V (number); split U |
| A7 | Reflexion (Qwen2.5-7B) | **42.7** | split not stated; copied by SkillRL from GiGPO | GiGPO Table 1; SkillRL Table 1 | V (number); split U |
| A8 | ExpeL | **46.3** | split not stated; SkillRL's own run; SAGA reprints it | SkillRL Table 1 | V (number); split U |
| A9 | MemP | **41.4** | split not stated | SkillRL Table 1 | V (number); split U |
| A10 | Mem0 | **33.6** | split not stated | SkillRL Table 1 | V (number); split U |
| A11 | SimpleMem | **29.7** | split not stated | SkillRL Table 1 | V (number); split U |
| A12 | ExpRAG (retrieval of expert scripted trajectories from train, K=4, static, index=all), training-free | **83.6** (83.58) | valid_unseen, 134; greedy; context-format sensitive (see A13) | ExpRAG Table C.2 text | V |
| A13 | ExpRAG format sensitivity (Qwen2.5-7B, K=4, same retrieved content) | Chat JSON 83.6; Agentic 81.3; Compact 79.1; **Textual 1.5** (K=2 Textual 14.9; K=5 Textual 0.0) | valid_unseen, all tasks | ExpRAG Table 17 | V |
| A14 | Qwen3.5-9B non-thinking ReAct (for backbone comparison, not Qwen2.5) | 33.0 (36/109) | "standard unseen evaluation split with 109 tasks" (not 134), temp 0 | ACCORD, arXiv 2606.16432, appendix A.4.3 | V |

Also read (RL-trained 7B, for scale only): SkillRL 89.9 (V), GRPO 77.6 (V), RLOO 75.5 (V), GiGPO w/ std 90.8 +/-1.3 and w/o std 90.2 +/-2.3 (V, GiGPO Table 1, 3 seeds; the verl-agent README shows 90.8 for 7B (V)), PPO 80.4 +/-2.7 (V). SkillRL ablations (RL-trained): w/o hierarchical 76.8, w/o skill library (raw trajectories) 61.7, w/o SFT 65.2, w/o dynamic evolution 84.4 (V). SkillRL context-length (V): raw-memory baseline ~1,450 tokens vs SkillRL <1,300; SkillBank initial 55 skills (12 general + 43 task-specific), growing to 100; K=6 task-specific retrieval plus always-on general skills.

### 2.2 Table B: per-family breakdown (columns as printed; order is Pick, Look, Clean, Heat, Cool, Pick2, All)

Table printed in SkillRL v2 Table 1 and GiGPO Table 1 (identical rows; V, read from raw HTML text, split not stated):

| Method (Qwen2.5-7B-Instruct) | Pick | Look | Clean | Heat | Cool | Pick2 | All |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen2.5 prompt | 33.4 | 21.6 | 19.3 | 6.9 | 2.8 | 3.2 | 14.8 |
| ReAct | 48.5 | 35.4 | 34.3 | 13.2 | 18.2 | 17.6 | 31.2 |
| Reflexion | 62.0 | 41.6 | 44.9 | 30.9 | 36.3 | 23.8 | 42.7 |
| Mem0 | 54.0 | 55.0 | 26.9 | 36.4 | 20.8 | 7.69 | 33.6 |
| ExpeL | 21.0 | 67.0 | 55.0 | 52.0 | 71.0 | 6.00 | 46.3 |
| MemP | 54.3 | 38.5 | 48.1 | 56.2 | 32.0 | 16.7 | 41.4 |
| SimpleMem | 64.5 | 33.3 | 20.0 | 12.5 | 33.3 | 3.84 | 29.7 |
| (RL, for scale) GRPO | 90.8 | 66.1 | 89.3 | 74.7 | 72.5 | 64.7 | 77.6 |
| (RL, for scale) SkillRL | 97.9 | 71.4 | 90.0 | 90.0 | 95.5 | 87.5 | 89.9 |

Valid_unseen per-family (SAGA Table 2 "Base Model", Qwen2.5-7B-Instruct per its text; columns reordered here to Pick, Look, Clean, Heat, Cool, Pick2; V as printed):

| Row | Pick (n=24) | Look (n=18) | Clean (n=31) | Heat (n=23) | Cool (n=21) | Pick2 (n=17) | All |
|---|---:|---:|---:|---:|---:|---:|---:|
| SAGA Table 2 "Base Model" | 75.0 | 18.9 | 57.4 | 13.0 | 60.0 | 36.5 | 45.5 |

Internal-consistency warning (my check, `checks.py`): 75.0 = 18/24 and 13.0 = 3/23 are integer-consistent, but Clean 57.4 (n=31), Cool 60.0 (n=21), Look 18.9 (n=18), Pick2 36.5 (n=17) are **not** a k/n for the stated family sizes (nearest 58.1, 61.9, 16.7, 35.3). The overall 45.5 = 61/134 is consistent. The per-family cells may be averaged or reprinted from another source; use with caution, and do not use them as per-family probabilities for exact power numbers (U).

Other per-family data read: ALFWorld paper BUTLER/Seq2Seq are not Qwen and are not reproduced here. ExpRAG gives easy (73-game) and hard (61-game) splits only, not six families.

### 2.3 What "seen vs unseen" means across sources (do not mix)

- ALFWorld: valid_seen = 140 games in rooms seen in training; valid_unseen = 134 games in unseen rooms (V, ALFWorld paper). Design's T_eval = valid_unseen.
- SAGA/Evolving-RL use "seen/unseen" for **task types** (four types seen, Look and Pick2 unseen), all evaluated on valid_unseen (V, SAGA Table 6). Different meaning.
- ExpRAG uses "easy/hard" task-type groups (V).
- I found no paper reporting Qwen2.5-7B-Instruct training-free **valid_seen vs valid_unseen** paired on one protocol (U). A search snippet mentions a SkillOS-curator paper with a 33.6 no-memory baseline on 140 valid_seen but I did not open it (U).

### 2.4 Items looked at and excluded

- MemSkill (arXiv 2602.02474) Table 4 ALF-Seen/ALF-Unseen numbers for ExpeL etc. use LLaMA-3.3-70B and Qwen3-Next-80B-A3B (V). Not 7B; excluded. (A search summary had suggested "Qwen" ExpeL 75.71/71.64; that is the 80B model.)
- SkillGraph (2605.12039) and MemHarness (2607.28272) reprint the same Qwen2.5-7B rows from SkillRL (V; MemHarness averages them differently, 14.5 / 27.9 / 39.9 "Avg."), so they are not independent evidence.
- ExpRAG's own Table 1 line "Reflexion 42.7 ... ExpRAG baseline 83.6" is labelled "valid-unseen" but its 42.7 and 33.6 entries are third-party reprints (footnoted e, b) of GiGPO/SkillRL numbers whose split is not stated; treat that label as the citing paper's claim, not a measured fact (V text; interpretation mine).
- SkillRevise (2606.01139) uses Qwen3-8B on 100 valid-seen/unseen tasks (71/100); not Qwen2.5.

---

## 3. Protocol evidence: what split did GiGPO / SkillRL's baselines likely use?

Read from `langfengQ/verl-agent` (raw GitHub, V):

- `verl/trainer/config/ppo_trainer.yaml`: `env.alfworld.eval_dataset: eval_in_distribution` (default).
- `agent_system/environments/env_package/alfworld/envs.py`: `eval_dataset = env_kwargs.get('eval_dataset', 'eval_in_distribution')`.
- `configs/config_tw.yaml`: `eval_id_data_path -> valid_seen`, `eval_ood_data_path -> valid_unseen`.
- `examples/gigpo_trainer/run_alfworld.sh`: `val_data_size=128`, `val_kwargs.temperature=0.4`, `do_sample=True`, `env.max_steps=50`.
- GiGPO paper appendix: "Each episode allows up to 50 environment steps"; ALFWorld validation temperature 0.4; WebShop QA validation temperature 0.0 (V).

Implication (inference, U): unless the authors overrode `eval_dataset`, the GiGPO/SkillRL-lineage numbers (14.8, 31.2, 42.7, 90.8, 89.9) were measured on valid_seen with 128 environments at temperature 0.4, not on the 134-game valid_unseen at temperature 0. Arithmetic hint only: 14.8% = 19/128 and 31.2% = 40/128, but 42.7 / 48.0 / 60.3 are not k/128, so I cannot confirm the n. Neither GiGPO nor SkillRL states the eval split or game count (V: SkillRL's only mention is "ALFWorld validation set"). Pre-registration should therefore **not** cite those numbers as valid_unseen anchors.

---

## 4. Variance and prompt-sensitivity evidence relevant to the determinism policy

- SAGA (V): "Because serving schedules may affect token-level execution" the no-memory baseline (online concurrency 32) vs evolution (concurrency 8) difference is reported as a potential implementation confound; temp 0, seed 42. Trajectory-only memory rounds on the same 134 games: 45.5 -> 46.3 -> 57.5 -> 46.3 (non-monotone by +/-11pp across rounds that differ only in the frozen memory content).
- ExpRAG (V): with greedy decoding, three seeds (2025/2026/2027) still vary "sampled environment instance (e.g., object placement)", retrieval tie-breaking and interaction history; median std 1.58pp (max 6.2pp) on ALFWorld across configurations (Ministral 3-8B).
- ExpRAG Table 17 (V): the same retrieved experience rendered as "Textual" instead of "Chat JSON" drops Qwen2.5-7B from 83.6 to 1.5 (K=4) and 0.0 (K=5). Prompt rendering can zero a 7B agent. Implication for arm B (all K=10 skills injected): longer prompts and formatting are part of the treatment per the design, but format collapse is a distinct failure mode worth a pre-run prompt-rendering smoke test (observation).
- The design's claim "ALFWorld deterministic + temp 0 => exactly one execution per (task, condition)" depends on vLLM producing identical token streams across runs; I found no source verifying that for vLLM batching (U), and SAGA's note points the other way. Not a recommendation to change the policy; a pre-registration risk to record.

---

## 5. Power-analysis anchors (my computation, `research/methodology/scripts/power.py`)

Assumptions: paired design on the same tasks; outcome per task in {C wins, B wins, tie}; the 95% cluster bootstrap over tasks of the mean paired difference; "power" = P(lower bound > 0 and point estimate >= 0). d = discordance rate = fraction of tasks where the two arms differ. Bootstrap is exact in distribution (multinomial resampling of the three outcome categories); simulation reps 1,000, B = 1,000 (Monte Carlo error about +/-0.03).

### 5.1 H2 primary: n = 134 (T_eval), pre-registered criterion C - B >= +5pp with CI excluding 0

| True delta (C - B) | d = 0.10 | d = 0.20 | d = 0.30 | d = 0.40 |
|---|---:|---:|---:|---:|
| +5.0pp | 0.45 | 0.25 | 0.16 | 0.13 |
| +7.5pp | 0.84 | 0.48 | 0.35 | 0.28 |
| +10pp | 1.00 | 0.77 | 0.56 | 0.46 |
| +15pp | n/a (delta > d) | 0.99 | 0.90 | 0.79 |

Normal-approximation minimum detectable effect at 80 percent power, two-sided 0.05, MDE = 2.8 * sqrt(d/n): d = 0.05 -> 5.4pp; 0.10 -> 7.6pp; 0.20 -> 10.8pp; 0.30 -> 13.2pp; 0.40 -> 15.3pp. Unpaired single-arm SE at n = 134: 3.1pp (p = 0.15), 4.0pp (p = 0.30), 4.3pp (p = 0.45).

Which d is realistic? When two arms differ only by the skill text in the prompt, many trajectories diverge. A rough empirical proxy is SAGA's R1 vs R2 vs R3 (+/-11pp moves in overall success with fixed model and games), which implies d of at least 0.1-0.3, i.e. the high-discordance rows (U; no per-game paired data were available to me). So plan for d around 0.2-0.4 unless pilot data show otherwise.

### 5.2 Other primary / secondary contrasts

- **Regression rate, n = 40 (T_reg), H2 second criterion "rate(C) < rate(B) with CI excluding 0 on the difference":** MDE at 80 percent power = 2.8*sqrt(d/40): d = 0.05 -> 9.9pp; 0.10 -> 14.0pp; 0.20 -> 19.8pp; 0.30 -> 24.2pp. A 3pp regression tolerance (tau_r) equals 1.2 tasks of 40; in task units 1/40 = 2.5pp passes, 2/40 = 5pp fails. And if T_reg is by definition G0-solved, the regression "rate" is a count of flips to failure; see feasibility of T_reg in finding 4.
- **Mining-seed layer (3 seeds, direction of C - B in >= 2/3 seeds):** with a true effect near zero the chance of >= 2/3 same-sign by luck is 0.5 (ignoring ties); a weak criterion by itself, but a conjunction with the n=134 CI. Not simulated further.

### 5.3 Stage-2 gate, n = 20 paired tasks (exact enumeration)

Rule as written: percentile cluster-bootstrap 95% CI lower bound > 0 AND point estimate >= +0.05. Pass table (b = tasks where with-trait wins, c = tasks where it loses):

| b \ c | 0 | 1 | 2 | 3 | 4 |
|---:|:--:|:--:|:--:|:--:|:--:|
| <=3 | fail | fail | fail | fail | fail |
| 4 | pass | fail | fail | fail | fail |
| 5, 6 | pass | fail | fail | fail | fail |
| 7, 8 | pass | pass | fail | fail | fail |

So tau_c = 0.05 (one task) is not binding; the CI rule is. Minimum pass: +20pp (4-0) or +30pp (7-1). If trait effects are a few tasks of 20, nearly all candidates are rejected; arm C would equal arm A in that case (the design already records "C ~ A" style outcomes as findings, threat 5, but H2 then compares an empty C to B). Observation; **requires ADR** to change.

### 5.4 Pass-rate anchors for pre-registration fields (section 8 of the design)

These are only anchors, not predictions of Qwen2.5-7B ReAct G0 behaviour on this harness:
- Arm A level on valid_unseen: 29.9 (A1) to 45.5 (A2) with the two protocols read directly; 14.8-31.2 if the GiGPO lineage was valid_seen at temp 0.4 (U). The design's prompt template (planner_react_v1 + retry_backoff_v1) is not any of these; it should be piloted, not assumed.
- H1 (C - A): published training-free skill/memory deltas on valid_unseen for Qwen2.5-7B: +12.0pp (SAGA trajectory-only R2 vs R0, non-monotone), +13.9pp (complete-hierarchy R3 vs R0), versus +53.7pp for retrieval of expert scripted trajectories (ExpRAG, not comparable to self-mined skills). These come from different protocols and single runs.
- SkillRL's own statement that unchanged models benefit little from skills (V) suggests a priori small H1/H2 effects for a frozen 7B; with the power table above, plan for an under-powered primary unless pilot d is low.

---

## 6. Backbone currency (observation, not a design change)

Date context: today is 2026-10-12.

**Is Qwen2.5-7B-Instruct still defensible for M0?** Yes, as a deliberate "comparability anchor", not as a state-of-the-art choice. Reasons (V where noted):
- The SkillRL line (SkillRL v2 revised 2026-10, SkillGraph, MemHarness, EvolveR, etc.) still uses Qwen2.5-7B-Instruct as the base (V in the three papers I opened), and the verl-agent/GiGPO tables report it, so EXP-0001's stated rationale (design section 2: "matches SkillRL's base model") holds.
- Open, local, vLLM-friendly, 32K-context; many reproducibility artifacts.

**Newer open ~7-9B instruct/post-trained models (as of Oct 2026, with sources I read):**
- Qwen3.5-9B (Hugging Face card, V): released Feb 2026 (card citation date), 9B (HF metadata 10B), post-trained with thinking (default) and non-thinking modes, Apache-2.0, 262,144-token native context; card lists BFCL-V4 66.1 and TAU2-Bench 79.1 for the 9B column.
- Qwen3-8B (older, Apr 2025 from my memory, U): used as ALFWorld backbone in SkillRevise (Qwen3-8B, V) and several RL checkpoints on Hugging Face (search snippets, U).
- Gemma 4 family (Google model card / release page, search result, partially V): E2B, E4B, 12B (added 2026-06-03 per release log), 26B-A4B MoE, 31B; Apache 2.0. Initial release dated 2026-03-31 (release log) or 2026-04-02 (arXiv text); sources disagree.
- Ministral 3 8B (`Ministral-3-8B-Instruct-2512`, cited with its HF id in ExpRAG appendix, V): used as a baseline there (zero-shot ALFWorld 4.5 on that paper's protocol, V).
- Others named in blog lists (Granite 4.1 8B, Llama variants, Phi, Olmo) are U; I did not verify them.

**Model-comparison anchors on valid_unseen (V):** Qwen2.5-7B zero-shot 29.9 vs Ministral 3-8B 4.5 vs Gemma 3-4B 0.8 vs Qwen2.5-7B-1M 5.2 (all ExpRAG, same protocol). Qwen3.5-9B non-thinking ReAct 33.0 on a 109-task unseen subset (ACCORD; not the 134 set). So Qwen2.5-7B-Instruct is not a weak ALFWorld baseline relative to newer 8-9B models under a plain prompt; newer models need not be better on this harness.

**Questions reviewers plausibly ask (anticipated; my framing):**
1. Why a 2024 model in late 2026? (answer: comparability with the SkillRL/GiGPO lineage; H4 backbone swap is planned in the SPEC, per the design rationale.)
2. Does the effect hold on a current model, for example Qwen3.5-9B non-thinking or Gemma 4 12B, where baseline success and prompt-following differ? Headroom changes with baseline (SAGA: base 45.5 vs Qwen3.8-Max 96.3 on the same split, V).
3. Is greedy/temperature-0 single execution adequate on vLLM, and was determinism measured? (Section 4.)
4. Which eval split and temperature are the published comparison numbers on? (Section 3.)
5. Is a thinking-mode model allowed, and how is token budget/prompt length handled for arm B?
6. Contamination: Qwen2.5-7B-Instruct and ALFWorld/ALFRED train data. A search summary of ExpRAG mentioned possible exposure of the standard 7B model to these tasks; I did not find that sentence in the paper text I read (U). Do not cite.

---

## 7. Not found / unverified (list for the orchestrator)

- A paper reporting Qwen2.5-7B-Instruct **ReAct on valid_unseen (134) at temperature 0 with per-family breakdown**: none read. Closest per-family: SAGA "Base Model" row (internally inconsistent for four of six families) and GiGPO/SkillRL rows (split unstated).
- Published training-free **skill-augmented** Qwen2.5-7B-Instruct numbers in SkillRL's tables: none (SkillRL's skill rows are RL-trained).
- Qwen2.5-7B ExpeL/Reflexion with original authors' protocols on 134 unseen: none; the 46.3 / 42.7 numbers are SkillRL/GiGPO reprints with unstated split.
- Per-family counts after TextWorld-loader filtering (3321 train claim): not verified.
- Paired per-game outcomes (needed to estimate real discordance d): not available; the power tables are conditional on assumed d.
- vLLM bitwise determinism for Qwen2.5-7B at temp 0: not verified.
- Release dates of Qwen3-8B and any non-Qwen/Gemma models: from memory or snippet only.
- ReAct original (Yao 2023) 134-game protocol numbers: not opened.

---

## 8. Sources (opened in this session)

- ALFWorld: Shridhar et al., arXiv 2010.03768 (PDF, Table 1 and Table 2). https://arxiv.org/abs/2010.03768
- ALFWorld loader: https://github.com/alfworld/alfworld/blob/master/alfworld/agents/environment/alfred_tw_env.py (raw)
- SkillRL v2: arXiv 2602.08234 (HTML, Table 1, Table 3, section 4.1, 4.3). https://arxiv.org/abs/2602.08234
- GiGPO / verl-agent: arXiv 2505.10978 (HTML, Table 1, appendix E.1); https://github.com/langfengQ/verl-agent (README, `examples/gigpo_trainer/run_alfworld.sh`, `verl/trainer/config/ppo_trainer.yaml`, `agent_system/environments/env_package/alfworld/{envs.py,configs/config_tw.yaml}`, `env_manager.py`)
- ExpRAG: arXiv 2603.18272 "Retrieval-Augmented LLM Agents: Learning to Learn from Experience" (Table 1, Table 4, C.1, C.2, Table 17)
- SAGA: arXiv 2610.06964 "Principles that Guide, Actions that Inform: Agent Evolution via Knowledge Abstraction" (Tables 2, 6, 7, 12 and appendix text)
- SEEA-R1: arXiv 2506.21669 (appendix F, 3321 train games)
- ACCORD: arXiv 2606.16432 (Table 2, appendix A.4.3)
- MemHarness: arXiv 2607.28272 (Table 1; reprints only). SkillGraph: arXiv 2605.12039. SkillRevise: arXiv 2606.01139 (Table 13, ALFWorld paragraph). MemSkill: arXiv 2602.02474 (Table 4). Harness-the-Memory: arXiv 2608.15008 (P8 definition, 134 valid_unseen).
- Qwen3.5-9B card: https://huggingface.co/Qwen/Qwen3.5-9B. Gemma 4: https://ai.google.dev/gemma/docs/core/model_card_4 and https://ai.google.dev/gemma/docs/releases (search results only; I did not open the pages).

Caveat on arXiv IDs 2605.x-2610.x: they were fetched successfully from arxiv.org HTML during this session; I cannot independently cross-check them against a second index, so treat very recent papers (especially 2610.06964, a freshly posted paper) as not yet peer-reviewed.
