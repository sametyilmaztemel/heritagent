"""EXP-0001 compute / wall-clock planning model (assumption-driven; NOT measured).

Every throughput/latency/token constant below is a planning ASSUMPTION
(unverified for this stack) to be replaced by pilot measurements.
Usage: python -I exp0001_compute_estimate.py
"""
import itertools, json

# ---------------- episode counts (from design §7 + ADR-0002/A1) ----------------
SEEDS, K = 3, 10
MINING = SEEDS * 60                                   # 180 (as locked; one G0 episode / task / seed)
CIG_PER_CAND = 5*2 + 20*2 + 20*2 + 40                 # replay, ablation, gen in+out, regression(with only) = 130
CIG = SEEDS * K * CIG_PER_CAND                        # 3,900
FINAL = 3 * (134 + 40 + 20) * SEEDS                   # 1,746 (arm A x3 duplicates included)
INTER_WORST = SEEDS * 45 * 10 * 2                     # 2,700 (A1 arithmetic is per seed)
INTER_WORST_CACHED_NEITHER = SEEDS * (45*10 + 10)     # 1,380 if 'neither' condition reused (not locked)
PILOT = 120 * 4                                       # recommended calibration pilot: G0 + 3 placebo texts x 120 disjoint train games
MINING_K4 = SEEDS * 60 * 4 + 60                       # sensitivity: k=4 stochastic rollouts/task + 1 deterministic G0 pass (lifetime-protocol ADR pending)

ep = {
    "locked_expected (interaction ~0)": MINING + CIG + FINAL,
    "locked_manifest_upper (interaction 900)": MINING + CIG + FINAL + 900,
    "locked_worst (interaction 2,700)": MINING + CIG + FINAL + INTER_WORST,
    "planning_ceiling (worst + pilot + k=4 mining)": MINING_K4 + CIG + FINAL + INTER_WORST + PILOT,
}

# ---------------- per-episode call/token model (ASSUMPTIONS) ----------------
SYS = 600      # system prompt: format + tool catalog (range 400-1000)
TASK0 = 200    # task + initial ALFWorld room description
OBS = 40       # tokens appended per step (observations only; actions are not in history)
TAIL = 30      # "Step k of at most N ..." tail
COMP = 60      # completion tokens per call (cap 512)
SKILL = 200    # tokens per rendered skill (cap tau_k = 512)

# share of episodes by number of skills in prompt (locked counts)
def skill_mix(total_ep, inter_ep=0):
    # CIG with-trait = 1 skill, without = 0; regression only with (1).
    cig_with = SEEDS*K*(5+20+20+40); cig_without = SEEDS*K*(5+20+20)
    armB = (134+40+20)*SEEDS                 # 10 skills
    armA = (134+40+20)*SEEDS                 # 0
    armC = (134+40+20)*SEEDS                 # ~0-1 skills (power analysis E[admitted] <= 1.3)
    mix = {0: MINING + cig_without + armA + armC*0.5, 1: cig_with + armC*0.5, 10: armB, 2: inter_ep}
    rest = total_ep - sum(mix.values())
    mix[0] += max(rest, 0)                   # pilot/mining extras: G0-like prompts
    return mix

def episode_tokens(steps, n_skills):
    prompt = sum(SYS + TASK0 + (k-1)*OBS + TAIL + n_skills*SKILL for k in range(1, steps+1))
    new_prompt_apc = TASK0 + SYS + n_skills*SKILL + (steps-1)*(OBS+TAIL) + TAIL  # with prefix caching
    return prompt, new_prompt_apc, steps*COMP

STEP_SCEN = {"S8 (as coded: max_plan_steps=8)": 8.5,     # <=8 steps + ~0.5 parse retries
             "S30 (manifest '~30 calls')": 30,
             "S50 (50-step budget, p0~0.4)": 40}          # 0.4*18 + 0.6*50 + ~2 retries

HW = {  # (TPOT s/token at bs=1, prefill tok/s at bs=1, per-call overhead s, concurrent speedup at N=32)
    "H100-80GB": (0.007, 10000, 0.03, 12),
    "A100-80GB": (0.011, 7000, 0.04, 10),
}
ENV_STEP_S, ENV_RESET_S, RETRY_IDLE_S = 0.03, 2.0, 3.0   # per step, per episode, mean backoff sleep per episode

out = {}
for (en, E), (sn, calls), (hn, (tpot, pre, ovh, spd)) in itertools.product(ep.items(), STEP_SCEN.items(), HW.items()):
    inter = INTER_WORST if "worst" in en or "ceiling" in en else (900 if "upper" in en else 0)
    mix = skill_mix(E, inter)
    steps = int(round(calls))
    tot_calls = tot_p = tot_p_apc = tot_c = 0
    for nsk, n in mix.items():
        p, papc, c = episode_tokens(steps, nsk)
        tot_calls += n*calls; tot_p += n*p*calls/steps; tot_p_apc += n*papc*calls/steps; tot_c += n*c*calls/steps
    gpu_seq = tot_p/pre + tot_c*tpot + tot_calls*ovh                 # no prefix caching (conservative)
    gpu_seq_apc = tot_p_apc/pre + tot_c*tpot + tot_calls*ovh
    host = E*(ENV_RESET_S + RETRY_IDLE_S) + tot_calls*ENV_STEP_S
    out[(en, sn, hn)] = dict(episodes=E, calls=tot_calls, prompt_tok=tot_p, compl_tok=tot_c,
                             wall_seq_h=(gpu_seq+host)/3600, wall_seq_apc_h=(gpu_seq_apc+host)/3600,
                             wall_conc_h=(gpu_seq/spd + host/32)/3600)

print("| episode scenario | step scenario | GPU | episodes | LLM calls | prompt tok (M) | compl tok (M) | wall h, sequential bs=1 (no APC) | wall h, sequential + prefix cache | wall h, 32 concurrent episodes |")
print("|---|---|---|---|---|---|---|---|---|---|")
for (en, sn, hn), r in out.items():
    print(f"| {en} | {sn} | {hn} | {r['episodes']:,} | {r['calls']/1e3:,.0f}k | {r['prompt_tok']/1e6:,.0f} | {r['compl_tok']/1e6:,.1f} | {r['wall_seq_h']:,.0f} | {r['wall_seq_apc_h']:,.0f} | {r['wall_conc_h']:,.1f} |")

# context-length check (max_model_len 8192 - max_tokens 512 = 7680 usable prompt)
for steps, nsk, sk in [(8,10,200),(50,10,200),(50,10,512),(50,1,512)]:
    p = SYS + TASK0 + (steps-1)*OBS + TAIL + nsk*sk
    print(f"context check: steps={steps} skills={nsk}x{sk} -> last prompt {p} tokens; fits 7680: {p<=7680}")
print(json.dumps({k: v for k, v in ep.items()}, indent=1))
