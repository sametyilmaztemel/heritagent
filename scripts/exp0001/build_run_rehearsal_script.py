"""Build the run rehearsal's committed script (``cli rehearse-run``; issue
#13 Phase B): ``evaluation/exp0001/run_rehearsal_script.json``.

Offline, once per change of the manifests or of the scripted world, by a
builder with the ALFWorld data and ``pip install -e ".[alfworld]"``; never
at rehearsal time. For every game the scripted run can play
(``run_rehearsal.played_games``) it opens the game through the run's own
environment factory and reads its start text, asks TextWorld's planner for
the game's ``policy_commands`` and replays them through the run's factory:
a plan counts only when the run's environment reports ``won``. Then
``run_rehearsal.choose_script`` picks the designated games (text-unique
and won by their plan) and the script is written with its generator's
provenance. The same data and versions give the same bytes.

Usage: python scripts/exp0001/build_run_rehearsal_script.py --data-root <json_2.1.1>
           [--out evaluation/exp0001/run_rehearsal_script.json] [--workers 4]
"""

from __future__ import annotations

import argparse
import functools
import json
import multiprocessing
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

TASKS_PER_CHILD = 20  # an open leaves files behind in its process (libdownward)


@functools.lru_cache(maxsize=None)
def _manifests() -> tuple[dict, dict, dict]:
    from evaluation.exp0001.provenance import verify_pre_run_artifacts
    manifests = REPO_ROOT / "experiments" / "manifests"
    splits, pilot, bank = (json.loads((manifests / name).read_text(encoding="utf-8")) for name in (
        "EXP-0001-splits.json", "EXP-0001-pilot.json", "EXP-0001-benefit-bank.json"))
    verify_pre_run_artifacts(splits, pilot, bank)
    return splits, pilot, bank


def planned_commands(gamefile: str, max_steps: int) -> list[str]:
    """TextWorld's planner on one game, opened as the run opens it (the
    demangled names) plus ``policy_commands``."""
    import textworld
    import textworld.gym
    from alfworld.agents.environment.alfred_tw_env import AlfredDemangler, AlfredInfos
    from textworld.envs.pddl import textgen

    from runtime.tool_adapters.alfworld import (
        patch_textworld_eval,
        planner_environments,
        release_planner,
    )

    patch_textworld_eval(textgen, textworld.__version__)
    request = textworld.EnvInfos(won=True, admissible_commands=True, policy_commands=True,
                                 extras=["gamefile"])
    env = textworld.gym.make(textworld.gym.register_game(
        gamefile, request, max_episode_steps=max_steps,
        wrappers=[AlfredDemangler(shuffle=False), AlfredInfos]))
    planners = planner_environments(env)
    try:
        _, infos = env.reset()
        return list(infos.get("policy_commands") or [])
    finally:
        env.close()
        release_planner(planners)  # as the run's environment does on close


@functools.lru_cache(maxsize=None)
def _factory(data_root: str):
    from evaluation.exp0001.backends import AlfworldEnvFactory
    from evaluation.exp0001.benefit_bank import bank_content_index
    splits, _, bank = _manifests()
    return AlfworldEnvFactory(data_root, splits, extra_games=bank_content_index(bank))


def survey(data_root: str, split: str, game_id: str) -> dict:
    """One game: its start text through the run's factory, and the planner's
    commands if replaying them through the run's factory wins it."""
    from runtime.seed_g0 import EPISODE_MAX_STEPS
    from runtime.tool_adapters.alfworld import ACT_TOOL

    factory = _factory(data_root)
    row = {"split": split, "game_id": game_id, "text": None, "plan": None, "error": None}
    try:
        plan = planned_commands(f"{data_root}/{split}/{game_id}/game.tw-pddl", EPISODE_MAX_STEPS)
        env = factory(split, game_id)
        try:
            row["text"] = env.start()
            for command in plan[:EPISODE_MAX_STEPS]:
                if env.call(ACT_TOOL.name, {"command": command}).terminal:
                    break
            if plan and env.record().won:
                row["plan"] = plan
        finally:
            env.close()
    except Exception as exc:  # noqa: BLE001 - reported; such a game is never designated
        row["error"] = f"{type(exc).__name__}: {exc}"[:300]
    return row


def main(argv: list[str] | None = None) -> int:
    from importlib.metadata import version

    from evaluation.exp0001 import run_rehearsal as rr

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", type=Path, required=True,
                        help="ALFWorld's json_2.1.1 directory")
    parser.add_argument("--out", type=Path, default=rr.SCRIPT_PATH)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)

    splits, pilot, bank = _manifests()
    games = rr.played_games(splits, bank)
    jobs = ([str(args.data_root)] * len(games), [g[0] for g in games], [g[1] for g in games])
    options = {"max_tasks_per_child": TASKS_PER_CHILD} if sys.version_info >= (3, 11) else {}
    with ProcessPoolExecutor(max_workers=args.workers,
                             mp_context=multiprocessing.get_context("spawn"), **options) as pool:
        rows = {}
        for done, row in enumerate(pool.map(survey, *jobs, chunksize=2), start=1):
            rows[(row["split"], row["game_id"])] = row
            if done % 50 == 0 or done == len(games):
                print(f"{done}/{len(games)} games surveyed", flush=True)
    errors = [r for r in rows.values() if r["error"] or r["text"] is None]
    if errors:
        for r in errors[:10]:
            print(f"ERROR {r['split']}/{r['game_id']}: {r['error']}", file=sys.stderr)
        return 1
    script = rr.choose_script(
        splits, pilot, bank, start_text=lambda split, game: rows[(split, game)]["text"],
        plan=lambda split, game: rows[(split, game)]["plan"],
        generator={"script": "scripts/exp0001/build_run_rehearsal_script.py",
                   "planner": "TextWorld policy_commands, replayed through the run's "
                              "environment factory (won)",
                   "versions": {name: version(name) for name in ("alfworld", "textworld")},
                   "played_games": len(games),
                   "winning_plans": sum(r["plan"] is not None for r in rows.values())})
    args.out.write_text(json.dumps(script, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    expected = rr.expected_outcome(script, splits, bank)
    print(f"{args.out}: {len(games)} played games, {len(script['plans'])} plans; the run will "
          f"play {expected['episodes']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
