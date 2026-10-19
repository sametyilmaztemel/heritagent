"""A rehearsal of the EXP-0001 run without a GPU or a model (ADR-0012
"Experiment needed"; issue #13 Phase B): ``cli rehearse-run``.

The run's own code runs unchanged, at the locked protocol: ``Exp0001Run``
with its lifetimes, miner, six gate rounds per seed with round-scoped G0
cells and the pre-round reserve, the arms, the final evaluation and the
report; the real games through the run's environment factory (the benefit
bank's included); the worker processes; the ledger, trajectories, mirror and
restores; the model's own tokenizer. Only the model server is a stand-in
(``rehearsal.StandInServer``), whose replies are a pure function of what a
request carries: the game (the hash of its start text), the skills in the
prompt, the seed and the temperature. A committed script
(``run_rehearsal_script.json``, built offline by
``scripts/exp0001/build_run_rehearsal_script.py``) names which games each
scripted condition wins and the plans that win them, so the gate meets every
path of ADR-0012 D2-D7: stage 2 passing at looks 1, 2 and 3, meeting alpha
without a pass at look 4 and stopping for futility; out-of-scope pass, fail
and vacuous; 4A pass and fail; 4B fail; a harmful pair and two harmless
ones; an empty seed.

What the run must give is computed here, by ``expected_outcome``, from the
script and the rule texts alone (an interleave, the D4 decision with
``Fraction``, the T_reg draw, the pairing), never by calling the code under
test; ``checks`` compares the run directory, the ledger and the stand-in's
request log with it.

A rehearsal measures nothing about any model (SPEC §32): its run id, its
backend version and its weights are a rehearsal's, which a real run refuses.
No protocol value is scaled for it: the full protocol runs."""

from __future__ import annotations

import hashlib
import json
import math
import threading
from collections.abc import Callable, Iterable, Mapping, Sequence
from fractions import Fraction
from itertools import combinations
from pathlib import Path

RUN_REHEARSAL_RUN_ID = "EXP-0001-run-rehearsal"
RUN_REHEARSAL_PASSED = "run-rehearsal-passed.json"
RUN_REHEARSAL_PASSED_SCHEMA = "heritagent/exp0001-run-rehearsal-passed/0.1"
SCRIPT_SCHEMA = "heritagent/exp0001-run-rehearsal-script/0.1"
SCRIPT_PATH = Path(__file__).with_name("run_rehearsal_script.json")
# the run-start record's weights: a value no served model has
SENTINEL_WEIGHTS_SHA256 = hashlib.sha256(
    b"heritagent EXP-0001 run rehearsal: no weights").hexdigest()
SCOPE_FAMILY = "pick_two_obj_and_place"  # the cheapest family to open
LIFETIME_WINNERS = 45  # L: T_mine tasks that win once per seed (T_reg needs 40)
SEED_OFFSET = {11: 0, 23: 1, 41: 2}  # task i of L wins at attempt (i + offset) mod 3
SHORT_LOSS_STEPS = 3
# the faults: a blip the adapter's own retries absorb (the run's defaults, 2 s doubling: a
# request retried at 2 s and 6 s outlives a 5 s blip), and a pause in the final evaluation
ADAPTER_OPTIONS: dict = {}
BLIP_SECONDS = 5.0
PAUSE_AFTER_FINAL_REQUESTS = 300
DEATH_SEED, DEATH_ROUND = 11, 2  # the server dies at the second candidate's look-2 games
# the protocol's rules, written out from ADR-0012 D4-D6 and ADR-0002 (not imported
# from inheritance/: the oracle must not share the code it checks)
LOOKS = (48, 96, 144, 192)
ALPHAS = (Fraction(1, 2000), Fraction(1, 2000), Fraction(1, 1000), Fraction(3, 1000))
TAU_C = Fraction(1, 20)
FUTILITY_DIVISOR = 40
OUT_SCOPE_GAMES = 24
OUT_SCOPE_MAX_NET_LOSS = 1      # delta_out >= -0.05 over 24 games
T_REG_SIZE = 40
T_REG_MAX_NET_DROP = 1          # (without - with) / 40 <= 0.03
INTERACTION_GAMES = 10
HARMFUL_PAIR_DELTA = Fraction(-1, 20)
FINAL_EVALUATION_RESERVE = 7 * (134 + 40 + 20)
TEACHER_FORM_KEY = "form"       # the reduced evidence form names itself (ADR-0009 D8)
EVIDENCE_START = '{"failed_trajectory":'
OVERFLOW = "overflow (scripted)"


class RehearsalScriptError(ValueError):
    """The script does not fit the manifests, or a target cannot be met."""


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_script(path: Path | None = None) -> dict:
    return json.loads(Path(path or SCRIPT_PATH).read_text(encoding="utf-8"))


def script_sha256(script: Mapping) -> str:
    return hashlib.sha256(json.dumps(script, sort_keys=True, separators=(",", ":"))
                          .encode("utf-8")).hexdigest()


# -- the rules, from their texts ----------------------------------------------------------
def interleave(bank: Mapping, families: Sequence[str], total: int) -> list[str]:
    """ADR-0012 D4/D5: game j is the floor(j/m)-th game, in bank order, of
    the (j mod m)-th family in name order."""
    families = sorted(families)
    m = len(families)
    return [bank["families"][families[j % m]][j // m]["game_id"] for j in range(total)]


def sign_p(n_plus: int, n_minus: int) -> Fraction:
    n = n_plus + n_minus
    if n == 0:
        return Fraction(1)
    return Fraction(sum(math.comb(n, i) for i in range(n_plus, n + 1)), 2 ** n)


def stage2_path(pairs: Sequence[tuple[bool, bool]]) -> dict:
    """ADR-0012 D4 on (with-trait success, cell success) per allocation game:
    the looks played, the stop and whether some look met alpha."""
    looks = []
    for k, n_k in enumerate(LOOKS, start=1):
        first = pairs[:n_k]
        n_plus = sum(1 for w, c in first if w and not c)
        n_minus = sum(1 for w, c in first if c and not w)
        net = n_plus - n_minus
        point = Fraction(net, n_k)
        alpha = sign_p(n_plus, n_minus) <= ALPHAS[k - 1]
        looks.append({"look": k, "n_plus": n_plus, "n_minus": n_minus, "alpha_met": alpha})
        if alpha and point >= TAU_C:
            return {"looks": looks, "stop_look": k, "decision": "pass", "reason": "efficacy",
                    "contribution": point, "alpha_met": True}
        if k < len(LOOKS) and net <= n_k // FUTILITY_DIVISOR:
            return {"looks": looks, "stop_look": k, "decision": "fail", "reason": "futility",
                    "contribution": point, "alpha_met": any(x["alpha_met"] for x in looks)}
    return {"looks": looks, "stop_look": len(LOOKS), "decision": "fail",
            "reason": "no efficacy at the last look", "contribution": point,
            "alpha_met": any(x["alpha_met"] for x in looks)}


def t_reg_rank(task_id: str) -> str:
    return hashlib.sha256(f"heritagent-treg|0|{task_id}".encode("utf-8")).hexdigest()


def attempt_seed(mining_seed: int, task_id: str, attempt: int) -> int:
    """ADR-0007 E1's attempt seed, written out from its rule (protocol.py
    holds the run's copy; a unit test holds them equal)."""
    key = f"heritagent-attempt|{mining_seed}|{task_id}|{attempt}"
    return int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16) % (2 ** 31)


def pool_ids(splits: Mapping, pool: str) -> list[str]:
    return sorted(g["game_id"] for g in splits["pools"][pool]["games"])


def families_of(candidate: Mapping) -> list[str]:
    return sorted(candidate["applicability"]["task_families"])


SCRIPT_ONLY = ("wins", "loses", "overhead")  # a script candidate's keys no proposal carries


# -- the scripted world ------------------------------------------------------------------
class RunWorld:
    """Which scripted condition wins which game. Every outcome is a pure
    function of (kind, seed, game, skills): the stand-in and the oracle
    share it; it is the world the rehearsal plays in, not code under test."""

    def __init__(self, script: Mapping, splits: Mapping):
        self.script = script
        self.t_mine = pool_ids(splits, "t_mine")
        self.lifetime = list(script["lifetime"]["L"])
        self._lifetime_index = {task: i for i, task in enumerate(self.lifetime)}
        self.candidates = {int(s): list(cs) for s, cs in script["candidates"].items()}
        self.by_name = {c["name"]: (int(s), c) for s, cs in script["candidates"].items()
                        for c in cs}
        self.cells = {int(s): set(games) for s, games in script["cells"].items()}
        self.pair_losses = {frozenset(p["names"]): set(p["games"])
                            for p in script.get("pair_losses", ())}
        final = script["final"]
        self.final_wins = set(final["g0_wins"])
        self.final_extra = {n: set(g) for n, g in final.get("extra_wins", {}).items()}
        self.final_lost = {n: set(g) for n, g in final.get("losses", {}).items()}
        self.attempts = {attempt_seed(s, task, a): (s, task, a) for s in SEED_OFFSET
                         for task in self.t_mine for a in range(3)}

    def lifetime_win(self, seed: int, task: str, attempt: int) -> bool:
        i = self._lifetime_index.get(task)
        return i is not None and attempt == (i + SEED_OFFSET[seed]) % 3

    def gate_win(self, seed: int, game: str, skills: Iterable[str]) -> bool:
        skills = frozenset(skills)
        cell = game in self.cells.get(seed, set())
        if not skills:
            return cell
        if len(skills) == 1:
            (name,) = skills
            c = self.by_name[name][1]
            return game in c.get("wins", ()) or (cell and game not in c.get("loses", ()))
        return cell and game not in self.pair_losses.get(skills, set())

    def final_win(self, game: str, skills: Iterable[str]) -> bool:
        skills = set(skills)
        if any(game in self.final_extra.get(n, ()) for n in skills):
            return True
        return game in self.final_wins and not any(game in self.final_lost.get(n, ())
                                                   for n in skills)


# -- the expected run ------------------------------------------------------------------
def expected_outcome(script: Mapping, splits: Mapping, bank: Mapping) -> dict:
    """What the run must give in this world, from the rule texts: per seed the
    candidates' stage paths, verdicts and eliminations, each round's with-
    trait episodes, cells and cell readers, the arms, the final cells, the D8
    pair counts and the episode total."""
    world = RunWorld(script, splits)
    lifetime_wins = {s: [(task, a) for task in world.t_mine for a in range(3)
                         if world.lifetime_win(s, task, a)] for s in SEED_OFFSET}
    solved = sorted({task for wins in lifetime_wins.values() for task, _ in wins})
    if len(solved) < T_REG_SIZE:
        raise RehearsalScriptError(f"only {len(solved)} T_mine tasks are won in a lifetime")
    t_reg = sorted(solved, key=t_reg_rank)[:T_REG_SIZE]
    interaction = pool_ids(splits, "interaction")
    universe = sorted(bank["families"])
    seeds, rounds, cells = {}, {}, {}
    for s in SEED_OFFSET:
        # the miner's ranking: estimated generality, highest first (discovery order breaks
        # ties; the script gives none)
        cands = sorted(world.candidates.get(s, []), key=lambda c: -c["estimated_generality"])
        source = script["teacher"].get(str(s), {}).get("pair_task")
        per = {}
        round_with = {r: 0 for r in range(1, 7)}
        readers: dict[tuple[int, str], int] = {}

        def read(r, game, n=1):
            readers[(r, game)] = readers.get((r, game), 0) + n

        for c in cands:
            name = c["name"]
            families = families_of(c)
            allocation = interleave(bank, families, LOOKS[-1])
            pairs = [(world.gate_win(s, g, {name}), world.gate_win(s, g, ())) for g in allocation]
            path = stage2_path(pairs)
            round_with[1] += 1  # stage 1: the source task, with the trait only
            stage1_success = world.gate_win(s, source, {name})
            start = 0
            for look in path["looks"]:
                n_k = LOOKS[look["look"] - 1]
                round_with[look["look"]] += n_k - start
                for g in allocation[start:n_k]:
                    read(look["look"], g)
                start = n_k
            entry = {"name": name, "families": families, "stage1_success": stage1_success,
                     "stage2": path, "overhead_pass": c.get("overhead", "pass") == "pass"}
            if path["alpha_met"]:
                omitted = [f for f in universe if f not in families]
                if omitted:
                    out_games = interleave(bank, omitted, OUT_SCOPE_GAMES)
                    round_with[5] += len(out_games)
                    for g in out_games:
                        read(5, g)
                    net_loss = sum(1 for g in out_games if world.gate_win(s, g, ())
                                   and not world.gate_win(s, g, {name})) - \
                        sum(1 for g in out_games if world.gate_win(s, g, {name})
                            and not world.gate_win(s, g, ()))
                    entry["out_of_scope"] = {"vacuous": False, "net_loss": net_loss,
                                             "pass": net_loss <= OUT_SCOPE_MAX_NET_LOSS}
                else:
                    entry["out_of_scope"] = {"vacuous": True, "pass": True}
                round_with[5] += T_REG_SIZE
                drop = sum(1 for g in t_reg if world.gate_win(s, g, ())
                           and not world.gate_win(s, g, {name})) - \
                    sum(1 for g in t_reg if world.gate_win(s, g, {name})
                        and not world.gate_win(s, g, ()))
                entry["regression"] = {"net_drop": drop, "pass": drop <= T_REG_MAX_NET_DROP}
            else:
                entry["out_of_scope"] = {"not_run": True, "pass": False}
                entry["regression"] = {"not_run": True, "pass": False}
            entry["eligible"] = (path["decision"] == "pass" and entry["out_of_scope"]["pass"]
                                 and entry["regression"]["pass"] and entry["overhead_pass"])
            per[name] = entry
        if any(e["regression"].get("not_run") is not True for e in per.values()):
            for g in t_reg:  # the seed's one G0 run on T_reg (ADR-0009 D1), in R5
                read(5, g)
        eligible = [c["name"] for c in cands if per[c["name"]]["eligible"]]
        rank = {c["name"]: i for i, c in enumerate(cands)}
        harmful, pairs_played = [], []
        for a, b in combinations(eligible, 2):
            pairs_played.append((a, b))
            round_with[6] += INTERACTION_GAMES
            for g in interaction:
                read(6, g)
            delta = Fraction(sum(int(world.gate_win(s, g, {a, b})) - int(world.gate_win(s, g, ()))
                                 for g in interaction), INTERACTION_GAMES)
            if delta < HARMFUL_PAIR_DELTA:
                harmful.append((a, b))
        eliminated = {}
        for a, b in harmful:
            ca, cb = per[a]["stage2"]["contribution"], per[b]["stage2"]["contribution"]
            loser = (b if ca > cb else a) if ca != cb else (b if rank[a] < rank[b] else a)
            eliminated[loser] = b if loser == a else a
        for name, entry in per.items():
            entry["eliminated_by"] = eliminated.get(name)
            entry["verdict"] = "promote" if entry["eligible"] and name not in eliminated \
                else "reject"
        rounds[s] = {f"R{r}": {"with_trait": round_with[r],
                               "cells": len({g for (rr, g) in readers if rr == r}),
                               "cell_readers": {g: n for (rr, g), n in sorted(readers.items())
                                                if rr == r}}
                     for r in range(1, 7) if round_with[r]}
        for (r, g) in readers:
            cells[(s, r, g)] = world.gate_win(s, g, ())
        seeds[s] = {"candidates": [c["name"] for c in cands], "per_candidate": per,
                    "pairs": pairs_played, "harmful": harmful,
                    "promoted": [n for n in per if per[n]["verdict"] == "promote"],
                    "source_task": source}
    # D8 (descriptive): a seed's cells of one game in two rounds; seeds' earliest cells
    by_game: dict[tuple[int, str], dict[int, bool]] = {}
    for (s, r, g), win in cells.items():
        by_game.setdefault((s, g), {})[r] = win
    cross_round = [(v[a], v[b]) for v in by_game.values() for a, b in combinations(sorted(v), 2)]
    earliest: dict[str, dict[int, bool]] = {}
    for (s, g), v in by_game.items():
        earliest.setdefault(g, {})[s] = v[min(v)]
    cross_seed = [(v[a], v[b]) for v in earliest.values() for a, b in combinations(sorted(v), 2)]
    # arms: B all candidates, C the promoted; genomes that coincide share their cells
    genomes: dict[frozenset, list[str]] = {frozenset(): ["A/once"]}
    for s in SEED_OFFSET:
        for arm, names in (("B", seeds[s]["candidates"]), ("C", seeds[s]["promoted"])):
            genomes.setdefault(frozenset(names), []).append(f"{arm}/{s}")
    t_eval = pool_ids(splits, "t_eval")
    transfer = pool_ids(splits, "ablation")
    final = {}
    for skills, arms in genomes.items():
        final[tuple(sorted(arms))] = {
            "skills": sorted(skills),
            "t_eval_successes": sum(world.final_win(g, skills) for g in t_eval),
            "t_reg_successes": sum(world.final_win(g, skills) for g in t_reg),
            "transfer_successes": sum(world.final_win(g, skills) for g in transfer)}
    gate_episodes = sum(r["with_trait"] + r["cells"] for rs in rounds.values()
                        for r in rs.values())
    final_episodes = len(genomes) * (len(t_eval) + T_REG_SIZE + len(transfer))
    lifetime_episodes = len(SEED_OFFSET) * len(world.t_mine) * 3
    return {"lifetime_wins": {s: len(w) for s, w in lifetime_wins.items()},
            "t_reg": t_reg, "seeds": seeds, "rounds": rounds,
            "cross_round": {"pairs": len(cross_round),
                            "disagreements": sum(a != b for a, b in cross_round)},
            "cross_seed": {"pairs": len(cross_seed),
                           "disagreements": sum(a != b for a, b in cross_seed)},
            "cells": len(cells), "final": final, "distinct_genomes": len(genomes),
            "episodes": {"lifetime": lifetime_episodes, "gate": gate_episodes,
                         "final": final_episodes,
                         "total": lifetime_episodes + gate_episodes + final_episodes}}


# -- building the script offline ---------------------------------------------------------
PROPOSAL_STYLE = {"when_to_apply": "household tasks with objects to move",
                  "procedure": [{"id": "S1", "text": "look around, then act"}]}


def _proposal(name: str, principle: str, families, generality: float) -> dict:
    return {"name": name, "principle": principle, "when_to_apply": PROPOSAL_STYLE["when_to_apply"],
            "procedure": PROPOSAL_STYLE["procedure"],
            "applicability": {"task_families": sorted(families)},
            "estimated_generality": generality}


LONG_PRINCIPLE = ("Before every action, list each receptacle in the room, recall what was "
                  "seen in it, and say why the next action follows from that list; ") * 30


def played_games(splits: Mapping, bank: Mapping) -> list[tuple[str, str]]:
    """(split, game_id) of every game the scripted run can play: the pools,
    the scope family's 192 bank games, and every family's first 8 (the
    all-family candidate's look 1 and the out-of-scope window)."""
    x_games = [g["game_id"] for g in bank["families"][SCOPE_FAMILY][:LOOKS[-1]]]
    played = {("train", g) for pool in ("t_mine", "interaction", "ablation")
              for g in pool_ids(splits, pool)}
    played |= {("train", g) for g in x_games}
    played |= {("train", g["game_id"]) for games in bank["families"].values() for g in games[:8]}
    played |= {("valid_unseen", g) for g in pool_ids(splits, "t_eval")}
    return sorted(played)


def choose_script(splits: Mapping, pilot: Mapping, bank: Mapping, *,
                  start_text: Callable[[str, str], str],
                  plan: Callable[[str, str], list[str] | None],
                  generator: Mapping | None = None) -> dict:
    """Pick the designated games of the scripted world. ``start_text(split,
    game_id)`` is a game's start text through the run's environment;
    ``plan(split, game_id)`` a verified winning command list or None. A game
    is designated only when its text is unique among every game the
    scripted run plays, so the stand-in can tell it apart from the text."""
    x = SCOPE_FAMILY
    families = sorted(bank["families"])
    if x not in families:
        raise RehearsalScriptError(f"the bank has no family {x!r}")
    others = [f for f in families if f != x]
    t_mine, t_eval = pool_ids(splits, "t_mine"), pool_ids(splits, "t_eval")
    interaction, transfer = pool_ids(splits, "interaction"), pool_ids(splits, "ablation")
    x_games = [g["game_id"] for g in bank["families"][x][:LOOKS[-1]]]
    texts = {key: text_sha256(start_text(*key)) for key in played_games(splits, bank)}
    counts: dict[str, int] = {}
    for sha in texts.values():
        counts[sha] = counts.get(sha, 0) + 1
    plans: dict[str, list[str]] = {}
    taken: set[str] = set()

    def winnable(split: str, game: str) -> bool:
        if counts[texts[(split, game)]] != 1:
            return False
        if game not in plans:
            commands = plan(split, game)
            if not commands:
                return False
            plans[game] = list(commands)
        return True

    def pick(split: str, games: Sequence[str], n: int, what: str) -> list[str]:
        chosen = []
        for game in games:
            if len(chosen) == n:
                break
            if game not in taken and winnable(split, game):
                chosen.append(game)
                taken.add(game)
        if len(chosen) < n:
            raise RehearsalScriptError(f"{what}: only {len(chosen)} of {n} text-unique "
                                       f"winnable games")
        return chosen

    lifetime = pick("train", t_mine, LIFETIME_WINNERS, "the lifetime's winners")
    d11, v11, d23 = lifetime[0], lifetime[1], lifetime[2]
    t_reg = sorted(lifetime, key=t_reg_rank)[:T_REG_SIZE]
    r1, r2 = [g for g in t_reg if g not in (d11, v11, d23)][:2]
    # out of scope for an X-scoped candidate: the omitted families' first games
    out_window = interleave(bank, others, OUT_SCOPE_GAMES)
    o1, o2 = pick("train", out_window, 2, "the out-of-scope cells")
    seg = [x_games[48 * k:48 * (k + 1)] for k in range(4)]
    x_c, x_d = pick("train", seg[0], 2, "the X cells")
    a = pick("train", seg[0], 3, "the look-1 wins")
    b = pick("train", seg[1], 8, "the look-2 wins")
    c = pick("train", seg[2], 6, "the look-3 wins")
    d = pick("train", seg[3], 4, "the look-4 wins")
    i1 = pick("train", interaction, 1, "the harmful pair's game")[0]
    c1_window = [g for g in interleave(bank, families, LOOKS[0]) if g not in out_window]
    c1_wins = pick("train", c1_window, 11, "C1's look-1 wins")
    final_g0 = pick("valid_unseen", t_eval, 6, "G0's final T_eval wins")
    final_c1 = pick("valid_unseen", t_eval, 3, "C1's extra final wins")
    final_transfer = pick("train", transfer, 2, "G0's final transfer wins")
    final_t_reg = [g for g in t_reg if g not in (r1, r2)][:4]
    names = ["rehearsal skill one", "rehearsal skill two", "rehearsal skill three",
             "rehearsal skill four", "rehearsal skill five", "rehearsal skill six"]
    principle = "Check the receptacle named in the task first; {}."
    candidates = {
        "11": [
            {**_proposal(names[0], principle.format("then act on the object"), families, 0.9),
             "wins": sorted(c1_wins + [d11]), "loses": []},
            {**_proposal(names[1], principle.format("then carry both objects"), [x], 0.8),
             "wins": sorted(a + b), "loses": []},
            {**_proposal(names[2], principle.format("then place them together"), [x], 0.7),
             "wins": sorted(a + b[:1] + c), "loses": []},
            {**_proposal(names[3], principle.format("then look twice"), [x], 0.6),
             "wins": sorted(a + b[:1] + c[:1] + d), "loses": sorted([o1, o2, r1, r2])},
            {**_proposal(names[4], LONG_PRINCIPLE, [x], 0.5), "overhead": "fail",
             "wins": sorted(a[:1]), "loses": [x_c]},
        ],
        "23": [{**_proposal(names[5], principle.format("then wait"), [x], 0.5),
                "wins": [], "loses": sorted([x_c, x_d])}],
        "41": [],
    }
    final_lost = final_g0[0]
    script = {
        "schema": SCRIPT_SCHEMA,
        "manifests": {"split_manifest_sha256": splits["manifest_sha256"],
                      "pilot_sha256": pilot["pilot_sha256"], "bank_sha256": bank["bank_sha256"]},
        "texts": {f"{split}/{game}": sha for (split, game), sha in sorted(texts.items())},
        "plans": dict(sorted(plans.items())),
        "lifetime": {"L": lifetime},
        "teacher": {"11": {"pair_task": d11, "overflow_task": v11}, "23": {"pair_task": d23}},
        "candidates": candidates,
        "cells": {"11": sorted([x_c, o1, o2, r1, r2, i1]), "23": sorted([x_c, x_d])},
        "pair_losses": [{"names": sorted(names[:2]), "games": [i1]}],
        "final": {"g0_wins": sorted(final_g0 + final_t_reg + final_transfer),
                  "extra_wins": {names[0]: sorted(final_c1)},
                  "losses": {names[4]: [final_lost]}},
        "generator": dict(generator or {}),
    }
    validate_script(script, splits, pilot, bank)
    return script


def validate_script(script: Mapping, splits: Mapping, pilot: Mapping, bank: Mapping) -> None:
    """Fail closed when the script was built for other manifests, or when a
    designated game has no plan or a shared text."""
    if script.get("schema") != SCRIPT_SCHEMA:
        raise RehearsalScriptError(f"not a {SCRIPT_SCHEMA} script")
    want = {"split_manifest_sha256": splits["manifest_sha256"],
            "pilot_sha256": pilot["pilot_sha256"], "bank_sha256": bank["bank_sha256"]}
    if script["manifests"] != want:
        raise RehearsalScriptError(f"the script was built for {script['manifests']}, not {want}")
    played = {f"{split}/{game}" for split, game in played_games(splits, bank)}
    if set(script["texts"]) != played:
        raise RehearsalScriptError(f"the script's start texts cover {len(script['texts'])} games, "
                                   f"not the {len(played)} the scripted run plays")
    counts: dict[str, int] = {}
    for sha in script["texts"].values():
        counts[sha] = counts.get(sha, 0) + 1
    by_game = {key.split("/", 1)[1]: sha for key, sha in script["texts"].items()}
    designated = set(script["lifetime"]["L"]) | {g for cs in script["candidates"].values()
                                                  for c in cs for g in c["wins"]}
    designated |= {g for games in script["cells"].values() for g in games}
    designated |= set(script["final"]["g0_wins"]) | {
        g for games in script["final"].get("extra_wins", {}).values() for g in games}
    for game in sorted(designated):
        if game not in by_game or counts[by_game[game]] != 1:
            raise RehearsalScriptError(f"{game} is designated but its start text is not "
                                       f"unique among the played games")
        if not script["plans"].get(game):
            raise RehearsalScriptError(f"{game} is designated but has no winning plan")


# -- the census ----------------------------------------------------------------------------
# a census process opens this many games, then a fresh one starts: TextWorld 1.7.0 keeps a
# 32 MB copy of fast-downward's library per opened game until its process ends (deleted,
# still mapped; unloading it crashes the process at exit), as the run's workers do (50)
CENSUS_TASKS_PER_CHILD = 25


def census_games(splits: Mapping, bank: Mapping, script: Mapping) -> list[tuple[str, str]]:
    """Every game the run may play that the scripted run does not: which
    bank games a run plays depends on its model's candidates, so each one
    is opened once before the A100 is paid for."""
    played = {tuple(key.split("/", 1)) for key in script["texts"]}
    every = {("train", g["game_id"]) for games in bank["families"].values() for g in games}
    every |= {("valid_unseen" if pool == "t_eval" else "train", g["game_id"])
              for pool in ("t_mine", "t_eval", "interaction", "ablation")
              for g in splits["pools"][pool]["games"]}
    return sorted(every - played)


def open_game(env_factory, split: str, game_id: str) -> str | None:
    """Open one game through the run's factory and read its start text: None,
    or what went wrong (run in a worker process)."""
    try:
        env = env_factory(split, game_id)
        try:
            env.start()
        finally:
            close = getattr(env, "close", None)
            if callable(close):
                close()
    except Exception as exc:  # noqa: BLE001 - reported, the census fails
        return f"{type(exc).__name__}: {exc}"[:300]
    return None


# -- the stand-in's side ------------------------------------------------------------------
def _section(text: str, start: str, end: str | None = None) -> str:
    if start not in text:
        return ""
    body = text.split(start, 1)[1]
    return body.split(end, 1)[0] if end and end in body else body


def prompt_skills(system: str) -> list[str]:
    """The skill names a ReAct system prompt expresses (runtime/skills.py)."""
    block = _section(system, "Expressed skills (use when applicable):\n")
    return [line[2:].split(": ", 1)[0] for line in block.splitlines() if line.startswith("- ")]


def evidence_of(prompt: str) -> dict | None:
    """The canonical evidence a teacher prompt carries (traits/miner)."""
    at = prompt.find(EVIDENCE_START)
    if at < 0:
        return None
    try:
        document, _ = json.JSONDecoder().raw_decode(prompt, at)
    except ValueError:
        return None
    return document if isinstance(document, dict) else None


class RunPolicy:
    """The stand-in's replies in the scripted world. ``respond(payload)``
    returns (200, text) or (400, why); every request is logged with its
    kind, game, skills and seed. ``preflight_task`` is the one task text
    outside the script it answers (``cli preflight``'s)."""

    def __init__(self, script: Mapping, splits: Mapping, *, preflight_task: str | None = None):
        from evaluation.exp0001 import rehearsal
        self._r = rehearsal
        self.world = RunWorld(script, splits)
        self.script = script
        self._preflight = None if preflight_task is None else text_sha256(preflight_task)
        self._game_of = {}
        counts: dict[str, int] = {}
        for sha in script["texts"].values():
            counts[sha] = counts.get(sha, 0) + 1
        for key, sha in script["texts"].items():
            split, game = key.split("/", 1)
            self._game_of[sha] = None if counts[sha] > 1 else (split, game)
        self._plans = {g: list(p) for g, p in script["plans"].items()}
        self._names = {int(s): {c["name"] for c in cs} for s, cs in script["candidates"].items()}
        self._lock = threading.Lock()
        self.log: list[dict] = []
        self.kinds = {"lifetime": 0, "gate": 0, "final": 0, "teacher": 0, "preflight": 0}
        self.overflows = 0

    def game(self, payload: Mapping) -> str | None:
        """The game a ReAct request plays (None for a text the script does
        not tell apart)."""
        messages = payload.get("messages") or [{}]
        known = self._game_of.get(text_sha256(_section(
            messages[-1].get("content", ""), "Task: ", "\n\nObservations so far:")))
        return None if known is None else known[1]

    # what a request is
    @staticmethod
    def kind(payload: Mapping) -> str:
        if payload.get("response_format"):
            return "teacher"
        if payload.get("temperature") == 0.7:
            return "lifetime"
        return "gate" if payload.get("seed") is not None else "final"

    def _note(self, entry: dict) -> None:
        with self._lock:
            self.log.append(entry)
            if entry["status"] == 200:
                self.kinds[entry["kind"]] += 1

    def respond(self, payload: Mapping) -> tuple[int, str]:
        kind = self.kind(payload)
        if kind == "teacher":
            return self._teacher(payload)
        messages = payload["messages"]
        system, user = messages[0]["content"], messages[-1]["content"]
        task = _section(user, "Task: ", "\n\nObservations so far:")
        sha = text_sha256(task)
        skills = prompt_skills(system)
        seed = payload.get("seed")
        entry = {"kind": kind, "seed": seed, "skills": skills, "text": sha[:16]}
        if sha == self._preflight and kind in ("final", "gate") and not skills:
            entry.update(kind="preflight", status=200)
            self._note(entry)
            return 200, self._r._final("This is a preflight call.")
        if sha not in self._game_of:
            return self._refused(entry, f"a {kind} request for a game the script does not "
                                        f"play (start text {sha[:16]})")
        known = self._game_of[sha]
        entry["game"] = None if known is None else known[1]
        leak = [w for w in ("task_families", "estimated_generality") if w in system + user]
        if leak:
            return self._refused(entry, f"evaluation-only metadata {leak} in a ReAct prompt")
        if known is not None and known[0] == "valid_unseen" and kind != "final":
            return self._refused(entry, "a T_eval game outside the final evaluation")
        win = False
        if kind == "lifetime":
            attempt = self.world.attempts.get(seed)
            if attempt is None or (known is not None and attempt[1] != known[1]):
                return self._refused(entry, f"seed {seed} is no attempt seed of this game")
            win = known is not None and self.world.lifetime_win(*attempt)
        elif kind == "gate":
            if seed not in self._names or not set(skills) <= self._names[seed]:
                return self._refused(entry, f"a gate request of seed {seed} with skills "
                                            f"{skills}")
            win = known is not None and self.world.gate_win(seed, known[1], skills)
        else:
            win = known is not None and self.world.final_win(known[1], skills)
        entry["win"] = win
        text = self._react(user, self._plans.get(entry.get("game")) if win else None)
        entry["status"] = 200
        self._note(entry)
        return 200, text

    def _refused(self, entry: dict, why: str) -> tuple[int, str]:
        entry["status"] = 400
        entry["why"] = why
        self._note(entry)
        return 400, f"the stand-in refuses: {why}"

    def _react(self, user: str, plan: list[str] | None) -> str:
        r = self._r
        history = user.split("\n\nObservations so far:\n", 1)[-1].rsplit("\n\nStep ", 1)[0]
        done = max((int(n) for n in r._OBSERVATION.findall(history)), default=0)
        if plan is not None:
            if done < len(plan):
                return r._react("Following the plan.", plan[done])
            return r._final("The task is done.")
        if done >= SHORT_LOSS_STEPS:
            return r._final("I could not finish the task.")
        goto = sorted(c for c in r._last_admissible(user) if c.startswith("go to ")) or ["look"]
        return r._react("Looking around.", goto[done % len(goto)])

    def _teacher(self, payload: Mapping) -> tuple[int, str]:
        seed = payload.get("seed")
        evidence = evidence_of(payload["messages"][-1]["content"])
        if evidence is None:  # the preflight's teacher call carries no trajectories
            self._note({"kind": "preflight", "seed": seed, "status": 200})
            return 200, json.dumps({"proposals": []})
        known = self._game_of.get(text_sha256(evidence.get("task", "")))
        game = None if known is None else known[1]
        entry = {"kind": "teacher", "seed": seed, "game": game,
                 "form": evidence.get(TEACHER_FORM_KEY, "full")}
        plan = self.script["teacher"].get(str(seed), {})
        if game is not None and game == plan.get("overflow_task") and \
                TEACHER_FORM_KEY not in evidence:
            # the server's context-window refusal (vLLM's words), for this pair's full form:
            # the miner must retry without the admissible commands (ADR-0009 D8)
            with self._lock:
                self.overflows += 1
            entry["status"] = 400
            entry["why"] = OVERFLOW
            self._note(entry)
            return 400, ("This model's maximum context length is 32768 tokens. However, you "
                         "requested 34816 tokens (the rehearsal's scripted overflow).")
        proposals = []
        if game is not None and game == plan.get("pair_task"):
            proposals = [{k: v for k, v in c.items() if k not in SCRIPT_ONLY}
                         for c in self.script["candidates"][str(seed)]]
        entry["proposals"] = len(proposals)
        entry["status"] = 200
        self._note(entry)
        return 200, json.dumps({"proposals": proposals}, sort_keys=True)


def run_protocol_issues(payload: Mapping) -> list[str]:
    """What in a chat request breaks the run's protocol, per kind: the
    lifetime at 0.7 with an attempt seed, the gate at 0 with a mining seed,
    the final evaluation at 0 without a seed (each with the stop sequence and
    at most 512 tokens), the teacher at 0.7 with a mining seed and 2,048
    tokens; one choice and the neutral sampling everywhere."""
    from evaluation.exp0001.protocol import MAX_TOKENS_PER_REQUEST, MINER_MAX_TOKENS, \
        MINING_SEEDS, MINING_TEMPERATURE
    from evaluation.exp0001.rehearsal import xgrammar_unsupported
    from runtime.model_adapters.openai_server import NEUTRAL_SAMPLING
    from runtime.policies.planner_react import STOP_SEQUENCES
    issues = []
    if payload.get("n") != 1:
        issues.append(f"n is {payload.get('n')!r}, not 1")
    sampling = {key: payload.get(key) for key in NEUTRAL_SAMPLING}
    if sampling != dict(NEUTRAL_SAMPLING):
        issues.append(f"sampling {sampling} is not the neutral {dict(NEUTRAL_SAMPLING)}")
    seed, temperature = payload.get("seed"), payload.get("temperature")
    if payload.get("response_format"):
        if temperature != MINING_TEMPERATURE or seed not in MINING_SEEDS or \
                payload.get("max_tokens") != MINER_MAX_TOKENS:
            issues.append(f"a teacher call at temperature {temperature!r}, seed {seed!r}, "
                          f"max_tokens {payload.get('max_tokens')!r}")
        unsupported = xgrammar_unsupported(
            payload["response_format"].get("json_schema", {}).get("schema"))
        if unsupported:
            issues.append(f"the schema has {unsupported}")
        return issues
    allowed = (temperature == MINING_TEMPERATURE and isinstance(seed, int)
               and not isinstance(seed, bool)) or \
        (temperature == 0.0 and (seed is None or seed in MINING_SEEDS))
    if not allowed:
        issues.append(f"a ReAct call at temperature {temperature!r} with seed {seed!r}")
    if payload.get("stop") != list(STOP_SEQUENCES):
        issues.append(f"stop {payload.get('stop')!r} is not {list(STOP_SEQUENCES)!r}")
    if not isinstance(payload.get("max_tokens"), int) or \
            not 0 < payload["max_tokens"] <= MAX_TOKENS_PER_REQUEST:
        issues.append(f"max_tokens {payload.get('max_tokens')!r} is outside 1.."
                      f"{MAX_TOKENS_PER_REQUEST}")
    return issues


def fault_plan(script: Mapping, bank: Mapping, expected: Mapping) -> dict:
    """Where the faults strike in this script: the blip at the first attempt
    of the lifetime's fourth winner in seed 11, the death at the second-
    ranked candidate's look-2 games of seed 11, which only round R2 plays."""
    per = expected["seeds"][DEATH_SEED]
    if len(per["candidates"]) < 2:
        raise RehearsalScriptError(f"seed {DEATH_SEED} needs two candidates for the death")
    name = per["candidates"][1]
    if per["per_candidate"][name]["stage2"]["stop_look"] < DEATH_ROUND:
        raise RehearsalScriptError(f"{name!r} stops before look {DEATH_ROUND}")
    allocation = interleave(bank, per["per_candidate"][name]["families"], LOOKS[-1])
    return {"blip_seed": attempt_seed(DEATH_SEED, script["lifetime"]["L"][3], 0),
            "blip_seconds": BLIP_SECONDS, "death_skill": name,
            "death_games": allocation[LOOKS[DEATH_ROUND - 2]:LOOKS[DEATH_ROUND - 1]]}


class Faults:
    """The rehearsal's server faults, decided from the request: a blip (503
    for ``blip_seconds`` at the first lifetime request of ``blip_seed``,
    which the adapter's retries absorb) and a death (503 from the first gate
    request of ``death_skill`` on a ``death_games`` game until cleared)."""

    def __init__(self, *, blip_seed: int | None, blip_seconds: float, death_skill: str | None,
                 death_games: Iterable[str], game_of: Callable[[Mapping], str | None],
                 clock: Callable[[], float]):
        self.blip_seed, self.blip_seconds = blip_seed, blip_seconds
        self.death_skill, self.death_games = death_skill, set(death_games)
        self._game_of, self._clock = game_of, clock
        self.blip_started: float | None = None
        self.dead = False
        self.died = False
        self._lock = threading.Lock()

    def __call__(self, payload: Mapping) -> bool:
        with self._lock:
            if self.dead:
                return True
            if self.blip_seed is not None and payload.get("seed") == self.blip_seed \
                    and payload.get("temperature") == 0.7 and self.blip_started is None:
                self.blip_started = self._clock()
            if self.blip_started is not None and \
                    self._clock() - self.blip_started < self.blip_seconds:
                return True
            if self.death_skill is not None and not self.died and \
                    payload.get("temperature") == 0.0 and payload.get("seed") is not None:
                messages = payload.get("messages") or [{}]
                if prompt_skills(messages[0].get("content", "")) == [self.death_skill] and \
                        self._game_of(payload) in self.death_games:
                    self.dead = self.died = True
                    return True
            return False

    def clear(self) -> None:
        with self._lock:
            self.dead = False


# -- the checks ----------------------------------------------------------------------------
def _episode_keys(lines: Sequence[Mapping]) -> list[str]:
    return [r["key"] for r in lines if r.get("type") == "episode"]


def first_pass_checks(lines: Sequence[Mapping], faults: Faults, *, aborted: int) -> dict:
    """After the server's death: the blip was absorbed on the way, and the
    pass ended inside seed 11's R2 with the attempts in flight charged."""
    keys = _episode_keys(lines)
    rounds = {int(k.split("/")[2][1:]) for k in keys if k.startswith(f"gate/{DEATH_SEED}/")}
    later = [k for k in keys if k.startswith("final/") or
             (k.startswith("gate/") and not k.startswith(f"gate/{DEATH_SEED}/"))]
    return {"the blip came, and the adapter's retries absorbed it":
                faults.blip_started is not None and faults.died,
            f"the server died in seed {DEATH_SEED}'s R{DEATH_ROUND}":
                rounds == set(range(1, DEATH_ROUND + 1)) and not later,
            "the death's attempts were charged": aborted >= 1}


def second_pass_checks(lines: Sequence[Mapping], expected: Mapping, *, code: int,
                       report_written: bool, restores: int) -> dict:
    """After the pause: the pass resumed from the restored directory, ran
    every gate round and stopped admitting inside the final evaluation."""
    keys = _episode_keys(lines)
    final = [k for k in keys if k.startswith("final/")]
    return {"the second pass started from the mirror's restore": restores == 1,
            "the second pass paused inside the final evaluation":
                code == 0 and not report_written
                and 0 < len(final) < expected["episodes"]["final"],
            "every gate episode was recorded before the final evaluation":
                sum(k.startswith("gate/") for k in keys) == expected["episodes"]["gate"]}


def policy_checks(policy: RunPolicy, expected: Mapping) -> dict:
    """What the stand-in saw: no request refused but the scripted overflow,
    whose pair the miner sent again in the reduced form (ADR-0009 D8), and
    one teacher call per lifetime pair otherwise."""
    log = list(policy.log)
    refused = [e for e in log if e["status"] != 200 and e.get("why") != OVERFLOW]
    teacher = [e for e in log if e["kind"] == "teacher"]
    overflowing = {(int(s), t["overflow_task"]) for s, t in policy.script["teacher"].items()
                   if t.get("overflow_task")}
    retried = [e for e in teacher if (e["seed"], e["game"]) in overflowing]
    pairs = sum(expected["lifetime_wins"].values())  # each L task wins once per seed: a pair
    return {"the stand-in refused no request": not refused,
            "one scripted overflow, sent again in the reduced form (ADR-0009 D8)":
                policy.overflows == len(overflowing) == 1
                and [(e["form"], e["status"]) for e in retried] == [
                    ("full", 400), ("without-admissible-commands", 200)],
            "one teacher call per lifetime pair, and the overflow's retry":
                sum(e["form"] == "full" for e in teacher) == pairs
                and len(teacher) == pairs + policy.overflows,
            "the scripted proposals reached the miner":
                sum(e.get("proposals", 0) for e in teacher) == sum(
                    len(s["candidates"]) for s in expected["seeds"].values())}


def _cell_trajectory(run_id: str, g0_genome_id: str, seed: int, gate_round: int,
                     task: str) -> str:
    """The trajectory id of a G0 cell, re-derived from the two rules
    (episodes.trajectory_id_for, gate_runner.config_fingerprint), not by
    calling them."""
    fingerprint = hashlib.sha256(f"{g0_genome_id}|".encode("utf-8")).hexdigest()[:8]
    key = f"gate/{seed}/R{gate_round}/cell/{task}/{fingerprint}"
    return "T-" + hashlib.sha256(f"{run_id}|{key}".encode("utf-8")).hexdigest()[:12]


def _names(run_dir: Path, seed: int) -> dict[str, str]:
    """gene id -> the proposal name, from the frozen candidate set and the registry."""
    candidates = json.loads((run_dir / f"seed-{seed}" / "candidates.json").read_text())
    root = run_dir / "registry"
    from genome.validation.registry import TraitRegistry
    registry = TraitRegistry(root)
    return {c["candidate"]["gene_id"]: json.loads(registry.resolve(c["candidate"]["artifact"]))[
        "name"] for c in candidates["candidates"]}


def gate_checks(run_dir: Path, expected: Mapping, *, run_id: str, g0_genome_id: str) -> dict:
    """Per seed: the frozen candidate set, every candidate's stage path and
    verdict, the rounds, and that every comparison read its own round's cell."""
    checks: dict[str, bool] = {}
    for s, exp in expected["seeds"].items():
        names = _names(run_dir, s)
        order = [names[c["candidate"]["gene_id"]] for c in json.loads(
            (run_dir / f"seed-{s}" / "candidates.json").read_text())["candidates"]]
        checks[f"seed {s}: the frozen candidates"] = order == exp["candidates"]
        gate = json.loads((run_dir / f"seed-{s}" / "gate.json").read_text())
        verdicts = {names[g]: v for g, v in gate["verdicts"].items()}
        checks[f"seed {s}: the verdicts"] = verdicts == {
            n: e["verdict"] for n, e in exp["per_candidate"].items()}
        eliminated = {names[g]: (names[b] if b else None)
                      for g, b in gate["eliminated_by_interaction"].items()}
        checks[f"seed {s}: the eliminations"] = eliminated == {
            n: e["eliminated_by"] for n, e in exp["per_candidate"].items()}
        stages: dict[str, dict] = {}
        for child in gate["children"].values():
            m = child["measurements"]
            stages.setdefault(names.get(m.get("candidate_gene_id"), "?"), {})[child["stage"]] = m
        ok_paths = ok_r5 = True
        for name, e in exp["per_candidate"].items():
            got = stages.get(name, {})
            ablation = got.get("ablation", {})
            path = e["stage2"]
            want_stop = (path["stop_look"], path["decision"], path["reason"])
            stop = ablation.get("stop") or {}
            looks = [(x.get("n_plus"), x.get("n_minus")) for x in ablation.get("looks", ())]
            ok_paths &= (stop.get("look"), stop.get("decision"), stop.get("reason")) == want_stop
            ok_paths &= looks == [(x["n_plus"], x["n_minus"]) for x in path["looks"]]
            ok_paths &= ablation.get("alpha_met") is path["alpha_met"]
            ok_paths &= (ablation.get("contribution") or {}).get("exact") == \
                f"{path['contribution'].numerator}/{path['contribution'].denominator}"
            ok_paths &= all(o["round"] == o["look"] == o["cell"]["round"] and o["cell"]["seed"] == s
                            for o in ablation.get("paired_outcomes", ()))
            replay = (got.get("replay") or {}).get("episodes") or [{}]
            ok_paths &= replay[0].get("success") is e["stage1_success"]
            out = got.get("generalization_out_of_scope", {})
            stage4 = got.get("interaction_regression", {})
            if e["out_of_scope"].get("not_run"):
                ok_r5 &= out.get("not_run") is True and stage4.get("not_run") is True
            elif e["out_of_scope"].get("vacuous"):
                ok_r5 &= out.get("vacuous") is True and out.get("stage_pass") is True
            else:
                ok_r5 &= out.get("stage_pass") is e["out_of_scope"]["pass"]
                ok_r5 &= (out.get("summary") or {}).get("net_loss") == e["out_of_scope"]["net_loss"]
            if not e["regression"].get("not_run"):
                regression = stage4.get("regression") or {}
                ok_r5 &= regression.get("regression_pass", regression.get("passed")) \
                    is e["regression"]["pass"]
            overhead = stage4.get("expression_overhead") or {}
            ok_r5 &= overhead.get("overhead_pass") is e["overhead_pass"]
        checks[f"seed {s}: every stage-2 path (ADR-0012 D4)"] = ok_paths
        checks[f"seed {s}: R5 and 4B as scripted (D5-D7)"] = ok_r5
        rounds = {r: {k: v for k, v in body.items()} for r, body in gate["rounds"].items()}
        checks[f"seed {s}: the rounds, cells and readers (D2)"] = rounds == exp_rounds(expected, s)
        bound = True
        for child in gate["children"].values():
            m = child["measurements"]
            for o in m.get("paired_outcomes", ()):
                if "cell" in o:
                    bound &= o["cell"]["trajectory_id"] == _cell_trajectory(
                        run_id, g0_genome_id, s, o["cell"]["round"], o["task_id"])
            for ep in ((m.get("regression") or {}).get("paired_g0_run") or {}).get("episodes", ()):
                bound &= ep["round"] == 5 and ep["trajectory_id"] == _cell_trajectory(
                    run_id, g0_genome_id, s, 5, ep["task_id"])
            for pair in (m.get("interaction") or {}).get("pair_evidence", ()):
                for o in pair["canonical_evidence"]["paired_outcomes"]:
                    if o["condition"] == "neither":
                        bound &= o["trajectory_id"] == _cell_trajectory(
                            run_id, g0_genome_id, s, 6, o["task_id"])
        checks[f"seed {s}: every comparison read its own round's cell (D2)"] = bound
    return checks


def exp_rounds(expected: Mapping, seed: int) -> dict:
    return {r: {"with_trait": body["with_trait"], "cells": body["cells"],
                "cell_readers": dict(body["cell_readers"])}
            for r, body in expected["rounds"][seed].items()}


def _sha(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def ledger_checks(lines: Sequence[Mapping], expected: Mapping, *,
                  interrupted: str | None) -> dict:
    """D2's order and D9's reserve, from the ledger alone: rounds in order
    per seed; within a session each round's admissions in ascending sha256
    of their keys; one passed reserve record per non-empty round per
    session, with the round's unrecorded count and the 1,358 reserve."""
    checks: dict[str, bool] = {}
    episodes = [r for r in lines if r.get("type") == "episode"]
    keys = [r["key"] for r in episodes]
    checks["every episode recorded once"] = len(keys) == len(set(keys))
    checks["the episode total"] = len(keys) == expected["episodes"]["total"]
    order_ok = True
    for s in expected["seeds"]:
        rounds = [int(k.split("/")[2][1:]) for k in keys if k.startswith(f"gate/{s}/")]
        order_ok &= rounds == sorted(rounds)
    checks["rounds in order per seed (D2)"] = order_ok
    hashed = True
    current, admitted = None, []
    for r in lines:
        kind = r.get("type")
        if kind in ("session", "reserve"):
            hashed &= admitted == sorted(admitted, key=_sha)
            admitted = []
            current = r.get("label") if kind == "reserve" else None
        elif kind == "attempt" and current and r["key"].startswith(current + "/"):
            admitted.append(r["key"])
    hashed &= admitted == sorted(admitted, key=_sha)
    checks["each round's queue in ascending sha256 of its keys (D2)"] = hashed
    reserves = [r for r in lines if r.get("type") == "reserve"]
    want = [f"gate/{s}/{r}" for s in expected["seeds"] for r in expected["rounds"][s]]
    labels = [r["label"] for r in reserves]
    first = {}
    for r in reserves:
        first.setdefault(r["label"], r)
    checks["one reserve check per round, one more for the interrupted round (D9)"] = \
        sorted(labels) == sorted(want + ([interrupted] if interrupted else []))
    checks["every reserve check passed with the 1,358 reserve"] = all(
        r["passed"] is True and r["final_reserve"] == FINAL_EVALUATION_RESERVE for r in reserves)
    checks["each round's first check counts its whole queue"] = all(
        first[f"gate/{s}/{name}"]["round_episodes"] == body["with_trait"] + body["cells"]
        for s in expected["seeds"] for name, body in expected["rounds"][s].items()
        if f"gate/{s}/{name}" in first)
    return checks


def report_checks(report: Mapping, expected: Mapping, lines: Sequence[Mapping]) -> dict:
    checks: dict[str, bool] = {"the run completed": report.get("status") == "complete"}
    gates = report.get("gates") or {}
    ok = True
    for s, exp in expected["seeds"].items():
        stops: dict[str, int] = {}
        for e in exp["per_candidate"].values():
            p = e["stage2"]
            label = f"{p['decision']}:{p['reason']}@look{p['stop_look']}"
            stops[label] = stops.get(label, 0) + 1
        ok &= gates.get(str(s)) == {
            "candidates": len(exp["candidates"]), "promoted": len(exp["promoted"]),
            "all_family_candidates": sum(1 for e in exp["per_candidate"].values()
                                         if e["out_of_scope"].get("vacuous")),
            "stage2_stops": dict(sorted(stops.items())),
            "reached_round_5": sum(1 for e in exp["per_candidate"].values()
                                   if e["stage2"]["alpha_met"])}
    checks["the report's gate summaries"] = ok
    noise = report.get("cell_disagreement") or {}
    checks["the D8 cell measures"] = (
        (noise.get("cross_round") or {}).get("pairs") == expected["cross_round"]["pairs"]
        and noise["cross_round"]["disagreements"] == expected["cross_round"]["disagreements"]
        and (noise.get("cross_seed") or {}).get("pairs") == expected["cross_seed"]["pairs"]
        and noise["cross_seed"]["disagreements"] == expected["cross_seed"]["disagreements"]
        and noise.get("cells") == expected["cells"])
    final = {}
    for r in lines:
        if r.get("type") == "episode" and r["key"].startswith("final/"):
            _, genome, task_set, _task = r["key"].split("/", 3)
            final.setdefault(genome, {}).setdefault(task_set, 0)
            final[genome][task_set] += bool((r.get("outcome") or {}).get("success"))
    by_arms = {tuple(sorted(g["arms"])): g["genome_id"] for g in report.get("final_genomes", ())}
    ok = set(by_arms) == set(expected["final"])
    for arms, want in expected["final"].items():
        got = final.get(by_arms.get(arms), {})
        ok &= (got.get("t_eval", 0), got.get("t_reg", 0), got.get("transfer", 0)) == (
            want["t_eval_successes"], want["t_reg_successes"], want["transfer_successes"])
    checks["the final cells and their sharing (ADR-0009 D2)"] = ok
    return checks


def passed_record(*, checks: Mapping[str, bool], commit: str, python: str, software: Mapping,
                  model: Mapping, protocol_sha256: str, script: Mapping, workers: int,
                  expected: Mapping, measures: Mapping, finished_at: str) -> dict:
    return {"schema": RUN_REHEARSAL_PASSED_SCHEMA, "kind": "run",
            "passed": all(checks.values()), "checks": dict(checks),
            "code": {"commit": commit}, "python": python, "software": dict(software),
            "model": dict(model), "protocol_sha256": protocol_sha256,
            "script_sha256": script_sha256(script), "workers": workers,
            "episodes": expected["episodes"], "measures": dict(measures),
            "note": "a rehearsal against a stand-in model server: it checks that the run's code "
                    "ran through every gate path at the locked protocol, and measures nothing "
                    "about any model",
            "finished_at": finished_at}


def protocol_sha256(protocol_block: Mapping) -> str:
    return hashlib.sha256(json.dumps(protocol_block, sort_keys=True, separators=(",", ":"))
                          .encode("utf-8")).hexdigest()


def check_run_rehearsal(passed: Mapping, *, commit: str, python: str, software: Mapping,
                        model_id: str, revision: str | None, tokenizer: str | None,
                        protocol: str, workers: int | None = None) -> dict:
    """What differs between a passed run rehearsal and this machine and
    checkout (empty when it covers them)."""
    if passed.get("schema") != RUN_REHEARSAL_PASSED_SCHEMA or passed.get("kind") != "run" \
            or passed.get("passed") is not True:
        return {"record": {"rehearsed": passed.get("schema"), "here": "a passed run rehearsal"}}
    now = {"commit": commit, "model_id": model_id, "python": python, "software": dict(software),
           "protocol_sha256": protocol}
    then = {"commit": passed["code"]["commit"], "model_id": passed["model"]["model_id"],
            "python": passed["python"], "software": passed["software"],
            "protocol_sha256": passed["protocol_sha256"]}
    if revision is not None:
        now["revision"], then["revision"] = revision, passed["model"]["revision"]
    if tokenizer is not None:
        now["tokenizer"], then["tokenizer"] = tokenizer, passed["model"]["tokenizer"]
    if workers is not None:
        now["workers"], then["workers"] = workers, passed["workers"]
    return {key: {"rehearsed": then[key], "here": now[key]} for key in now if then[key] != now[key]}
