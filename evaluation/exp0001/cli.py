"""Command line for EXP-0001 on one GPU node (issue #13 Phase B; ADR-0008).
The Colab notebooks call these commands; each one can also be run by hand.

  gpu-check       the A100 80 GB check (fails closed on any other GPU)
  fetch-data      download ALFWorld release 0.2.2's data (sha256-checked), the
                  data the frozen manifests were built from
  verify-data     hash every game of the split and pilot manifests
  mirror-check    write, replace, read and time small files in the mirror
                  directory (on Colab, Google Drive)
  run-commit      the code commit a run started at, from its mirror (empty for
                  a new run)
  prepare-model   download and hash one pinned model revision -> prepared.json
  serve           start ``vllm serve`` on it and wait until it answers
  preflight       two calls outside any ledger that check the serving stack:
                  a ReAct-format reply with the stop sequence, and the
                  miner's structured output under its JSON schema (the
                  teacher first runs only after the 540 lifetime episodes)
  pilot           the calibration pilot -> pilot-report.json
  run             EXP-0001: write (or check) the run-start record, restore the
                  run directory from its mirror if it is missing, run or
                  resume, snapshot to the mirror
  status          a read-only summary of a run (or pilot) directory
  repair-ledger   cut off a torn last ledger line, keeping it on record

``run`` and ``pilot`` take ``--workers`` for parallel episodes against the
one server (ADR-0008). Every run directory lives outside the repository
checkout, which must be clean: the run-start record names its commit."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from evaluation.exp0001.backends import (
    AlfworldEnvFactory,
    BackendError,
    gpu_inventory,
    hf_token_counter,
    require_a100_80gb,
)
from evaluation.exp0001.episodes import EpisodeExecutor, RunPaused, protocol_budgets
from evaluation.exp0001.data import fetch_data, verify_data
from evaluation.exp0001.ledger import LedgerError, RunLedger, repair_torn_tail
from evaluation.exp0001.mirror import (
    MirrorError,
    RunMirror,
    check_mirror_dir,
    latest_snapshot,
    restore,
    snapshot_file,
)
from evaluation.exp0001.pilot_run import (
    PILOT_REPORT_SCHEMA,
    PILOT_RUN_ID,
    project_exp0001,
    run_pilot,
)
from evaluation.exp0001.protocol import CAPS
from evaluation.exp0001.provenance import (
    ProvenanceError,
    build_run_start_record,
    freeze_json,
    verify_pre_run_artifacts,
)
from evaluation.exp0001.serving import (
    DEFAULT_MAX_MODEL_LEN,
    DEFAULT_MODEL,
    DEFAULT_PORT,
    load_prepared,
    prepare_model,
    server_command,
    server_environment,
    start_server,
    wait_until_ready,
)
from runtime.model_adapters.openai_server import VLLMServerAdapter

MANIFESTS = Path("experiments") / "manifests"
SPLITS_FILE, PILOT_FILE = "EXP-0001-splits.json", "EXP-0001-pilot.json"


class CliError(RuntimeError):
    """A precondition of a command does not hold."""


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: Path, document) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def git_state(repo: Path) -> tuple[str, bool]:
    """The checkout's commit and whether it differs from it (untracked files
    count; build outputs are ignored by .gitignore)."""
    def git(*args) -> str:
        return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True,
                              check=True).stdout.strip()
    return git("rev-parse", "HEAD"), bool(git("status", "--porcelain"))


def _manifests(repo: Path) -> tuple[dict, dict]:
    splits, pilot = _json(repo / MANIFESTS / SPLITS_FILE), _json(repo / MANIFESTS / PILOT_FILE)
    verify_pre_run_artifacts(splits, pilot)  # the pre-registered ones, verified
    return splits, pilot


def _server_version(url: str) -> str | None:
    import urllib.request
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/version", timeout=30) as response:
            return json.loads(response.read().decode("utf-8")).get("version")
    except Exception:  # noqa: BLE001 - recorded as unknown
        return None


def _adapter(args, prepared: dict) -> VLLMServerAdapter:
    adapter = VLLMServerAdapter(args.server_url, prepared["model_id"], prepared["revision"],
                                backend_version=_server_version(args.server_url))
    adapter.require_served()
    return adapter


class Progress:
    """The ledger hook of a notebook run: the mirror's snapshot, and a line
    every ``every`` finished episodes so a long cell shows it is alive."""

    def __init__(self, mirror: RunMirror | None, every: int, clock=time.monotonic):
        self._mirror, self._every, self._clock = mirror, every, clock
        self._episodes, self._started = 0, clock()

    def __call__(self, record: dict) -> None:
        if self._mirror is not None:
            self._mirror.on_append(record)
        if record.get("type") == "episode":
            self._episodes += 1
            if self._every and self._episodes % self._every == 0:
                minutes = (self._clock() - self._started) / 60
                print(f"[progress] {self._episodes} episodes in this session "
                      f"({minutes:.1f} min); last {record['key']}", flush=True)
        elif record.get("type") in ("complete", "incomplete"):
            print(f"[progress] the run is {record['type']}: {record.get('reason')}", flush=True)


def _session_info(gpu: dict, adapter: VLLMServerAdapter, workers: int,
                  server_info: Path | None) -> dict:
    metadata = adapter.metadata()
    settings = _json(server_info) if server_info and Path(server_info).exists() else None
    return {"accelerator": gpu, "server": metadata.capabilities.get("server"),
            "vllm": metadata.backend_version, "server_settings": settings, "workers": workers,
            "opened_at": _now()}


# -- commands -----------------------------------------------------------------------
def cmd_gpu_check(args) -> int:
    print(json.dumps(require_a100_80gb(gpu_inventory())))
    return 0


def cmd_fetch_data(args) -> int:
    root = fetch_data(args.target)
    print(f"ALFWorld data at {root}")
    return 0


def cmd_verify_data(args) -> int:
    splits, pilot = _manifests(Path(args.repo))
    print(json.dumps(verify_data(args.data_root, splits, pilot)))
    return 0


def cmd_mirror_check(args) -> int:
    print(json.dumps(check_mirror_dir(args.mirror_dir)))
    return 0


def cmd_run_commit(args) -> int:
    data = snapshot_file(args.mirror_dir, "run-start.json") if Path(args.mirror_dir).exists() \
        else None
    print("" if data is None else json.loads(data.decode("utf-8"))["code"]["commit"])
    return 0


def cmd_prepare_model(args) -> int:
    prepared = prepare_model(args.model, revision=args.revision, cache_dir=args.cache_dir)
    _write_json(args.out, prepared)
    print(f"{prepared['model_id']}@{prepared['revision']} weights sha256 "
          f"{prepared['weights']['sha256']} -> {args.out}")
    return 0


def cmd_serve(args) -> int:
    prepared = load_prepared(args.prepared)
    command = server_command(prepared, port=args.port, max_model_len=args.max_model_len,
                             gpu_memory_utilization=args.gpu_memory_utilization)
    environment = server_environment(batch_invariant=not args.no_batch_invariant)
    process = start_server(command, args.log, environment=environment)
    wait_until_ready(f"http://127.0.0.1:{args.port}", prepared["model_id"], process=process,
                     log_path=args.log, timeout_seconds=args.timeout)
    _write_json(args.info, {"command": command, "environment": environment, "pid": process.pid,
                            "port": args.port, "started_at": _now()})
    print(f"the model server (pid {process.pid}) serves {prepared['model_id']} on port "
          f"{args.port}; log {args.log}, settings {args.info}")
    return 0


def cmd_preflight(args) -> int:
    from runtime.model_adapters.base import GenerationSettings, ModelMessage
    from runtime.policies.planner_react import STOP_SEQUENCES
    from traits.miner.miner import output_schema

    prepared = load_prepared(args.prepared)
    adapter = _adapter(args, prepared)
    reply = adapter.generate(
        [ModelMessage("user", "Reply with one line: Thought: ready")],
        GenerationSettings(temperature=0.0, max_tokens=64, seed=None, stop=STOP_SEQUENCES))
    if reply.prompt_tokens is None or reply.completion_tokens is None:
        raise CliError("the server reports no token usage; the run needs exact counts")
    teacher = adapter.structured_generate(  # validated against the schema by the adapter
        [ModelMessage("user", "Propose at most one short household-task skill as JSON that "
                              "follows the given schema.")],
        GenerationSettings(temperature=0.7, max_tokens=2048, seed=11), output_schema())
    print(json.dumps({"react_reply": reply.text[:200], "react_tokens": [
        reply.prompt_tokens, reply.completion_tokens], "teacher_reply": json.loads(teacher.text),
        "teacher_finish_reason": teacher.finish_reason}, indent=2))
    return 0


def _restore_if_missing(local: Path, mirror_dir: Path | None) -> None:
    if mirror_dir is None or (local.exists() and any(local.iterdir())):
        return
    if latest_snapshot(mirror_dir) is None:
        return  # a fresh run
    doc = restore(mirror_dir, local)
    print(f"restored {local} from mirror snapshot {doc['seq']} ({len(doc['files'])} files)")


def cmd_pilot(args) -> int:
    from runtime.expression import compile_runtime_config
    from runtime.seed_g0 import build_g0_seed_genome
    from genome.validation.registry import TraitRegistry

    gpu = require_a100_80gb(gpu_inventory())
    prepared = load_prepared(args.prepared)
    repo = Path(args.repo)
    commit, dirty = git_state(repo)
    if dirty:
        raise CliError(f"the checkout at {repo} has changes; the pilot report names a clean commit")
    splits, pilot = _manifests(repo)
    verify_data(args.data_root, splits, pilot)  # seconds; before any episode
    out = Path(args.out_dir)
    _restore_if_missing(out, args.mirror_dir)
    report_path = out / "pilot-report.json"
    if report_path.exists():
        print(report_path.read_text(encoding="utf-8"))
        return 0
    adapter = _adapter(args, prepared)
    mirror = RunMirror(out, args.mirror_dir) if args.mirror_dir else None
    ledger = RunLedger(out / "ledger.jsonl", run_id=PILOT_RUN_ID, clock=time.time, caps=CAPS,
                       session_info=_session_info(gpu, adapter, args.workers, args.server_info),
                       on_append=Progress(mirror, args.progress_every))
    registry = TraitRegistry(out / "registry")
    g0 = build_g0_seed_genome(registry)
    pilot_games = {("train", g["game_id"]): g["content_sha256"] for g in pilot["games"]}
    executor = EpisodeExecutor(
        model=adapter, env_factory=AlfworldEnvFactory(args.data_root, splits,
                                                      extra_games=pilot_games),
        ledger=ledger, trajectories_dir=out / "trajectories", run_id=PILOT_RUN_ID,
        code_commit=commit, budgets=protocol_budgets(), workers=args.workers)
    started = _now()
    try:
        measured = run_pilot(executor, pilot, compile_runtime_config(g0, registry))
    finally:
        executor.close()
    projection = project_exp0001(measured, t_mine_size=len(splits["pools"]["t_mine"]["games"]),
                                 t_eval_size=len(splits["pools"]["t_eval"]["games"]),
                                 transfer_size=len(splits["pools"]["ablation"]["games"]))
    metadata = adapter.metadata()
    report = {"schema": PILOT_REPORT_SCHEMA, "experiment": "EXP-0001",
              "pilot_sha256": pilot["pilot_sha256"], "started_at": started,
              "finished_at": _now(), "code": {"commit": commit, "dirty": dirty},
              "model": {"model_id": metadata.model_id, "revision": metadata.revision,
                        "backend": metadata.backend, "backend_version": metadata.backend_version,
                        "weights_sha256": prepared["weights"]["sha256"]},
              "accelerator": gpu, "workers": args.workers,
              "note": "a calibration measurement; it tunes nothing and is no EXP-0001 result",
              **measured, "projection": projection}
    freeze_json(report_path, report)
    if mirror:
        mirror.snapshot()
    print(json.dumps({"measures": report["measures"], "projection": projection}, indent=2))
    return 0


def _run_start(args, local: Path, prepared: dict, adapter: VLLMServerAdapter, splits: dict,
               pilot: dict, commit: str, dirty: bool, tokenizer: str) -> dict:
    path = local / "run-start.json"
    if path.exists():  # a resumed run: the same code, model and weights, or nothing
        record = _json(path)
        if args.run_id and record["run_id"] != args.run_id:
            raise CliError(f"{local} holds run {record['run_id']!r}, not {args.run_id!r}")
        if record["code"]["commit"] != commit or dirty:
            raise CliError(f"the run started at commit {record['code']['commit']}; this checkout "
                           f"is {commit}{' (dirty)' if dirty else ''}. Check out that commit.")
        if record["weights"]["sha256"] != prepared["weights"]["sha256"]:
            raise CliError("the prepared weights differ from the ones the run started with")
        return record
    if not args.run_id:
        raise CliError("a new run needs --run-id")
    from genome.validation.registry import TraitRegistry
    from runtime.seed_g0 import build_g0_seed_genome
    metadata = adapter.metadata()
    model = {"model_id": metadata.model_id, "revision": metadata.revision,
             "backend": metadata.backend, "backend_version": metadata.backend_version}
    return build_run_start_record(
        code_commit=commit, code_dirty=dirty, model={**model, "tokenizer": tokenizer},
        teacher=model, weights=prepared["weights"], split_manifest=splits,
        pilot_manifest=pilot, g0_genome=build_g0_seed_genome(TraitRegistry(local / "registry")),
        run_id=args.run_id, started_at=_now())


def cmd_run(args) -> int:
    from evaluation.exp0001.pipeline import Exp0001Run

    gpu = require_a100_80gb(gpu_inventory())
    prepared = load_prepared(args.prepared)
    repo, local = Path(args.repo), Path(args.local_dir)
    commit, dirty = git_state(repo)
    splits, pilot = _manifests(repo)
    verify_data(args.data_root, splits, pilot)  # seconds; before any episode
    _restore_if_missing(local, args.mirror_dir)
    adapter = _adapter(args, prepared)
    counter = hf_token_counter(prepared["model_id"], prepared["revision"],
                               path=Path(prepared["path"]))
    record = _run_start(args, local, prepared, adapter, splits, pilot, commit, dirty,
                        counter.identity)
    mirror = RunMirror(local, args.mirror_dir) if args.mirror_dir else None
    started = time.monotonic()
    budget = None if args.stop_after_hours is None else args.stop_after_hours * 3600
    run = Exp0001Run(run_dir=local, run_start=record, split_manifest=splits, model=adapter,
                     teacher=adapter, env_factory=AlfworldEnvFactory(args.data_root, splits),
                     token_counter=counter, workers=args.workers,
                     session_info=_session_info(gpu, adapter, args.workers, args.server_info),
                     on_ledger_append=Progress(mirror, args.progress_every),
                     pause_when=None if budget is None
                     else (lambda: time.monotonic() - started > budget))
    try:
        report = run.run()
    except RunPaused as exc:
        print(f"paused after {args.stop_after_hours} h of this session ({exc}); every admitted "
              f"episode is recorded. Run the same command again to resume.")
        return 0
    finally:
        if mirror:
            mirror.snapshot()
            if mirror.failures:
                print(f"[mirror] {mirror.failures} snapshot(s) failed; last: {mirror.last_error}",
                      file=sys.stderr)
    print(json.dumps({k: report.get(k) for k in ("status", "stage", "reason", "episodes",
                                                   "model_calls", "aborted_attempts")}, indent=2))
    return 0


def cmd_status(args) -> int:
    run_dir = Path(args.run_dir)
    record = run_dir / "run-start.json"
    if record.exists():
        start = _json(record)
        run_id, caps = start["run_id"], start["protocol"]["caps"]
    else:  # a pilot directory
        run_id, caps = PILOT_RUN_ID, CAPS
    ledger = RunLedger(run_dir / "ledger.jsonl", run_id=run_id, clock=time.time, caps=caps,
                       readonly=True)
    summary = {"run_id": run_id, **ledger.summary()}
    for name in ("report.json", "pilot-report.json"):
        if (run_dir / name).exists():
            summary[name] = {k: _json(run_dir / name).get(k) for k in ("status", "stage", "reason")}
    if args.mirror_dir:
        latest = latest_snapshot(args.mirror_dir)
        summary["mirror_snapshot"] = None if latest is None else latest["seq"]
    print(json.dumps(summary, indent=2))
    return 0


def cmd_repair_ledger(args) -> int:
    note = repair_torn_tail(Path(args.run_dir) / "ledger.jsonl")
    print("the ledger ends with a complete record; nothing to repair" if note is None
          else json.dumps(note))
    return 0


def parser() -> argparse.ArgumentParser:
    top = argparse.ArgumentParser(prog="python -m evaluation.exp0001.cli",
                                  description=__doc__.splitlines()[0])
    sub = top.add_subparsers(dest="command", required=True)

    sub.add_parser("gpu-check").set_defaults(func=cmd_gpu_check)

    p = sub.add_parser("fetch-data")
    p.add_argument("--target", type=Path, required=True, help="becomes ALFWORLD_DATA")
    p.set_defaults(func=cmd_fetch_data)

    p = sub.add_parser("verify-data")
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--repo", type=Path, default=Path("."))
    p.set_defaults(func=cmd_verify_data)

    p = sub.add_parser("mirror-check")
    p.add_argument("--mirror-dir", type=Path, required=True)
    p.set_defaults(func=cmd_mirror_check)

    p = sub.add_parser("run-commit")
    p.add_argument("--mirror-dir", type=Path, required=True)
    p.set_defaults(func=cmd_run_commit)

    p = sub.add_parser("prepare-model")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--revision", default=None, help="a commit, branch or tag; default the "
                   "hub's current commit (pinned in prepared.json)")
    p.add_argument("--cache-dir", type=Path, default=None)
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(func=cmd_prepare_model)

    p = sub.add_parser("serve")
    p.add_argument("--prepared", type=Path, required=True)
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--max-model-len", type=int, default=DEFAULT_MAX_MODEL_LEN)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    p.add_argument("--log", type=Path, default=Path("vllm-server.log"))
    p.add_argument("--info", type=Path, default=Path("vllm-server.json"),
                   help="where the server's command and settings are written")
    p.add_argument("--no-batch-invariant", action="store_true",
                   help="faster, but a reply may then depend on the batch (ADR-0008 D4)")
    p.add_argument("--timeout", type=float, default=1800.0)
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("preflight")
    p.add_argument("--prepared", type=Path, required=True)
    p.add_argument("--server-url", default=f"http://127.0.0.1:{DEFAULT_PORT}")
    p.set_defaults(func=cmd_preflight)

    for name, func in (("pilot", cmd_pilot), ("run", cmd_run)):
        p = sub.add_parser(name)
        p.add_argument("--prepared", type=Path, required=True)
        p.add_argument("--data-root", type=Path, required=True,
                       help="ALFWorld's json_2.1.1 directory")
        p.add_argument("--repo", type=Path, default=Path("."))
        p.add_argument("--server-url", default=f"http://127.0.0.1:{DEFAULT_PORT}")
        p.add_argument("--server-info", type=Path, default=Path("vllm-server.json"),
                       help="the settings file `serve` wrote; recorded in the session")
        p.add_argument("--workers", type=int, default=16)
        p.add_argument("--progress-every", type=int, default=50)
        p.add_argument("--mirror-dir", type=Path, default=None,
                       help="snapshots of the run directory (on Colab: Google Drive)")
        if name == "pilot":
            p.add_argument("--out-dir", type=Path, required=True)
        else:
            p.add_argument("--local-dir", type=Path, required=True)
            p.add_argument("--run-id", default=None, help="required for a new run")
            p.add_argument("--stop-after-hours", type=float, default=None,
                           help="admit no new episode after this many hours of this session; "
                                "the run stays resumable (compute-unit budgeting)")
        p.set_defaults(func=func)

    p = sub.add_parser("status")
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--mirror-dir", type=Path, default=None)
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("repair-ledger")
    p.add_argument("--run-dir", type=Path, required=True)
    p.set_defaults(func=cmd_repair_ledger)
    return top


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        return args.func(args)
    except (CliError, BackendError, ProvenanceError, LedgerError, MirrorError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
