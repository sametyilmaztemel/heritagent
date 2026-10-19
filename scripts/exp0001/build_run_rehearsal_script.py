"""Build the run rehearsal's committed script (``cli rehearse-run``; issue
#13 Phase B): ``evaluation/exp0001/run_rehearsal_script.json``.

Offline, once per change of the manifests or of the scripted world, by a
builder with the ALFWorld data and ``pip install -e ".[alfworld]"``; never
at rehearsal time. For every game the scripted run can play
(``run_rehearsal.played_games``) it opens the game through the run's own
environment factory and reads its start text. For each game whose start
text no other played game shares, it asks TextWorld's planner for the
game's ``policy_commands``, in a process of its own under a memory cap and
a time limit, and replays them through the run's factory: a plan counts
only when the run's environment reports ``won``. Then
``run_rehearsal.choose_script`` picks the designated games (text-unique
and won by their plan) and the script is written with its generator's
provenance. The texts and plans are cached (``--cache``, outside the
checkout by default), so an interrupted build resumes; a plan that hit the
time limit or the memory cap is tried again then. The same data, versions
and plans give the same bytes (which plans hit the time limit can depend
on the machine's load).

Usage: python scripts/exp0001/build_run_rehearsal_script.py --data-root <json_2.1.1>
           [--out evaluation/exp0001/run_rehearsal_script.json] [--cache DIR]
           [--workers 3] [--memory-gb 4] [--timeout 600]
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

# TextWorld keeps a 32 MB copy of fast-downward's library per opened game until its
# process ends (unloading it crashes the process at exit): a worker reads this many
# start texts, then a fresh one starts
TASKS_PER_CHILD = 25


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

    from runtime.tool_adapters.alfworld import patch_textworld_eval

    patch_textworld_eval(textgen, textworld.__version__)
    request = textworld.EnvInfos(won=True, admissible_commands=True, policy_commands=True,
                                 extras=["gamefile"])
    env = textworld.gym.make(textworld.gym.register_game(
        gamefile, request, max_episode_steps=max_steps,
        wrappers=[AlfredDemangler(shuffle=False), AlfredInfos]))
    try:
        _, infos = env.reset()
        return list(infos.get("policy_commands") or [])
    finally:
        env.close()


@functools.lru_cache(maxsize=None)
def _factory(data_root: str):
    from evaluation.exp0001.backends import AlfworldEnvFactory
    from evaluation.exp0001.benefit_bank import bank_content_index
    splits, _, bank = _manifests()
    return AlfworldEnvFactory(data_root, splits, extra_games=bank_content_index(bank))


def start_text(data_root: str, split: str, game_id: str) -> dict:
    """One game's start text through the run's factory (no planner)."""
    row = {"split": split, "game_id": game_id, "text": None, "error": None}
    try:
        env = _factory(data_root)(split, game_id)
        try:
            row["text"] = env.start()
        finally:
            env.close()
    # TextWorld's translator ends a malformed game with SystemExit
    except (Exception, SystemExit) as exc:  # noqa: BLE001 - reported; the build fails
        row["error"] = f"{type(exc).__name__}: {exc}"[:300]
    return row


def winning_plan(data_root: str, split: str, game_id: str) -> dict:
    """The planner's commands for one game, if replaying them through the
    run's factory wins it (run in a process of its own: see plan_in_process)."""
    from runtime.seed_g0 import EPISODE_MAX_STEPS
    from runtime.tool_adapters.alfworld import ACT_TOOL

    row = {"split": split, "game_id": game_id, "plan": None, "error": None}
    try:
        plan = planned_commands(f"{data_root}/{split}/{game_id}/game.tw-pddl", EPISODE_MAX_STEPS)
        env = _factory(data_root)(split, game_id)
        try:
            env.start()
            for command in plan[:EPISODE_MAX_STEPS]:
                if env.call(ACT_TOOL.name, {"command": command}).terminal:
                    break
            if plan and env.record().won:
                row["plan"] = plan
            elif plan:
                row["error"] = "the plan does not win through the run's factory"
        finally:
            env.close()
    except (Exception, SystemExit) as exc:  # noqa: BLE001 - reported; never designated
        row["error"] = f"{type(exc).__name__}: {exc}"[:300]
    return row


LOAD_DEPENDENT = ("the planner took more than", "the planner's process ended")


def plan_in_process(data_root: str, split: str, game_id: str, *, memory_gb: float,
                    timeout: float) -> dict:
    """``winning_plan`` in a fresh process under a memory cap (set by the
    process itself) and a time limit: TextWorld's planner exhausts memory on
    some games (one reached 5.3 GB and was killed), and such a game is just
    never designated."""
    import subprocess

    row = {"split": split, "game_id": game_id, "plan": None, "error": None}
    try:
        done = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--data-root",
                               data_root, "--memory-gb", str(memory_gb), "--plan-one", split,
                               game_id], capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        row["error"] = f"the planner took more than {timeout:.0f} s"
        return row
    lines = done.stdout.strip().splitlines()
    try:
        answer = json.loads(lines[-1]) if done.returncode == 0 and lines else None
    except ValueError:
        answer = None
    if not isinstance(answer, dict) or (answer.get("split"), answer.get("game_id")) != \
            (split, game_id):
        row["error"] = f"the planner's process ended with {done.returncode}: " \
                       f"{done.stderr.strip()[-200:]}"
        return row
    return answer


def _cached(path: Path) -> dict:
    rows = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except ValueError:  # the torn last line of an interrupted build
                continue
            rows[(row["split"], row["game_id"])] = row
    return rows


def _bounded_map(pool, function, jobs, workers: int, on_result) -> None:
    """``function(*job)`` for every job, at most ``workers`` in flight, each
    result handed to ``on_result`` as it comes: a pool that replaces its
    workers stops replacing them when every job is submitted up front, on
    Python 3.11, 3.12 and 3.14 (CPython gh-115634)."""
    from concurrent.futures import FIRST_COMPLETED, wait
    queue, pending = iter(jobs), set()

    def submit_next():
        for job in queue:
            pending.add(pool.submit(function, *job))
            return

    for _ in range(workers):
        submit_next()
    while pending:
        done, _ = wait(pending, return_when=FIRST_COMPLETED)
        for future in done:
            pending.discard(future)
            on_result(future.result())
            submit_next()


def _append(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def main(argv: list[str] | None = None) -> int:
    from collections import Counter
    from concurrent.futures import ThreadPoolExecutor
    from importlib.metadata import version

    from evaluation.exp0001 import run_rehearsal as rr

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", type=Path, required=True,
                        help="ALFWorld's json_2.1.1 directory")
    parser.add_argument("--out", type=Path, default=rr.SCRIPT_PATH)
    parser.add_argument("--cache", type=Path, default=None,
                        help="where the surveyed texts and plans are kept between attempts "
                             "(default: heritagent-run-rehearsal-script under the system's "
                             "temporary directory, outside the checkout)")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--memory-gb", type=float, default=4.0,
                        help="the planner's memory cap per game")
    parser.add_argument("--timeout", type=float, default=600.0,
                        help="the planner's time limit per game, in seconds")
    parser.add_argument("--plan-one", nargs=2, metavar=("SPLIT", "GAME_ID"),
                        help=argparse.SUPPRESS)  # the planner's own process
    args = parser.parse_args(argv)
    if args.plan_one:  # this process alone: its memory cap does not reach the builder's
        import resource
        limit = int(args.memory_gb * 2 ** 30)
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
        print(json.dumps(winning_plan(str(args.data_root), *args.plan_one), sort_keys=True))
        return 0

    splits, pilot, bank = _manifests()
    games = rr.played_games(splits, bank)
    import tempfile
    cache = args.cache or Path(tempfile.gettempdir()) / "heritagent-run-rehearsal-script"
    cache.mkdir(parents=True, exist_ok=True)
    texts = _cached(cache / "texts.jsonl")
    todo = [g for g in games if g not in texts or texts[g]["error"]]
    options = {"max_tasks_per_child": TASKS_PER_CHILD} if sys.version_info >= (3, 11) else {}
    done = []

    def text_read(row: dict) -> None:
        texts[(row["split"], row["game_id"])] = row
        _append(cache / "texts.jsonl", row)
        done.append(1)
        if len(done) % 50 == 0 or len(done) == len(todo):
            print(f"{len(done)}/{len(todo)} start texts read", flush=True)

    with ProcessPoolExecutor(max_workers=args.workers,
                             mp_context=multiprocessing.get_context("spawn"), **options) as pool:
        _bounded_map(pool, start_text, [(str(args.data_root), *g) for g in todo], args.workers,
                     text_read)
    errors = [r for r in texts.values() if r["error"] or r["text"] is None]
    if errors:
        for r in errors[:10]:
            print(f"ERROR {r['split']}/{r['game_id']}: {r['error']}", file=sys.stderr)
        return 1
    # plans only for games whose start text no other played game shares: no other is ever
    # designated (run_rehearsal.choose_script)
    counts = Counter(texts[g]["text"] for g in games)
    unique = [g for g in games if counts[texts[g]["text"]] == 1]
    plans = _cached(cache / "plans.jsonl")
    todo = [g for g in unique if g not in plans
            or (plans[g]["error"] or "").startswith(LOAD_DEPENDENT)]
    with ThreadPoolExecutor(max_workers=args.workers) as threads:
        for done, row in enumerate(threads.map(lambda g: plan_in_process(
                str(args.data_root), *g, memory_gb=args.memory_gb, timeout=args.timeout),
                todo), start=1):
            plans[(row["split"], row["game_id"])] = row
            _append(cache / "plans.jsonl", row)
            if done % 25 == 0 or done == len(todo):
                print(f"{done}/{len(todo)} plans tried", flush=True)
    script = rr.choose_script(
        splits, pilot, bank, start_text=lambda split, game: texts[(split, game)]["text"],
        plan=lambda split, game: (plans.get((split, game)) or {}).get("plan"),
        generator={"script": "scripts/exp0001/build_run_rehearsal_script.py",
                   "planner": "TextWorld policy_commands, replayed through the run's "
                              "environment factory (won)",
                   "versions": {name: version(name) for name in ("alfworld", "textworld")},
                   "python": sys.version.split()[0],
                   "played_games": len(games), "text_unique": len(unique),
                   "winning_plans": sum(plans[g]["plan"] is not None for g in unique)})
    args.out.write_text(json.dumps(script, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    expected = rr.expected_outcome(script, splits, bank)
    print(f"{args.out}: {len(games)} played games, {len(unique)} with a start text of their own, "
          f"{len(script['plans'])} plans kept; the run will play {expected['episodes']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
