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
  preflight       two calls outside any ledger that check the serving stack,
                  each in its own message shape: G0's ReAct reply with the
                  stop sequence, and the teacher's structured output under
                  the miner's JSON schema (the teacher first runs only after
                  the 540 lifetime episodes)
  pilot           the calibration pilot -> pilot-report.json
  rehearse        the pilot without a GPU or a model: the same checks, games,
                  workers, ledger, tokenizer, mirror and report against a
                  stand-in server, with an outage and a restore on the way
                  (run it on a CPU runtime before the A100 is paid for)
  rehearsal-check whether a passed rehearsal covers this machine (commit,
                  model, Python, pinned software)
  run             EXP-0001: write (or check) the run-start record, restore the
                  run directory from its mirror if it is missing, run or
                  resume, snapshot to the mirror
  status          a read-only summary of a run (or pilot) directory
  snapshot        take a snapshot of a run (or pilot) directory now, for example
                  after Drive is mounted again (exit code 3 if it fails)
  repair-ledger   cut off a torn last ledger line, keeping it on record

Exit codes: 0 done; 2 an error the message explains (a run directory
another command still holds, or a request the model server refused or
failed); 130 stopped (the stop button, or the
notebook kernel that started the command restarted or died: environment
variable ``HERITAGENT_STOP_WITH_PARENT``), with every recorded episode kept;
3 when a ``run`` or ``pilot`` that finished, paused or was stopped, or a
``snapshot``, failed its last snapshot; an unexpected exception exits through
Python's traceback (1). A failed last snapshot leaves ``UNMIRRORED.json`` in
the run directory, also after an error: the newest records exist only on
this machine, and the notebooks keep the runtime. So does a SIGTERM while
the command holds its directory (the stop button's second press). The one
exception is another machine taking over the mirror: it holds the run now,
and the command stops with exit 2 and no marker. The last snapshot is not
cut short by a stop. ``run``, ``pilot`` and ``snapshot`` hold the run
directory's one-writer lock from before they touch the mirror until after
their last snapshot.

``run`` and ``pilot`` take ``--workers`` for parallel episodes against the
one server (ADR-0008). Every run directory lives outside the repository
checkout, which must be clean: the run-start record names its commit."""

from __future__ import annotations

import argparse
import contextlib
import json
import subprocess
import sys
import time
from collections.abc import Mapping
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
    port_answers,
    prepare_model,
    process_alive,
    server_command,
    server_environment,
    software_stack,
    start_server,
    wait_until_ready,
)
from runtime.model_adapters.errors import AdapterError
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


def _adapter(args, prepared: dict, **options) -> VLLMServerAdapter:
    from evaluation.exp0001.protocol import CHAT_TEMPLATE_KWARGS, CONTEXT_WINDOW
    adapter = VLLMServerAdapter(args.server_url, prepared["model_id"], prepared["revision"],
                                backend_version=_server_version(args.server_url),
                                chat_template_kwargs=CHAT_TEMPLATE_KWARGS, **options)
    adapter.require_served()
    window = adapter.served_context_window()
    if window != CONTEXT_WINDOW:  # pre-registered (ADR-0009 D4)
        raise CliError(f"the server's context window is {window}, the protocol's is "
                       f"{CONTEXT_WINDOW}; restart `serve` without --max-model-len")
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
                _out(f"[progress] {self._episodes} episodes in this session "
                          f"({minutes:.1f} min); last {record['key']}")
        elif record.get("type") in ("complete", "incomplete"):
            _out(f"[progress] the run is {record['type']}: {record.get('reason')}")


def _out(line: str) -> None:
    """A line on stdout that never raises: a command whose notebook kernel
    died writes into a pipe nobody reads (BrokenPipeError)."""
    try:
        print(line, flush=True)
    except OSError:
        pass


def _session_info(gpu: dict, adapter: VLLMServerAdapter, workers: int,
                  server_info: Path | None) -> dict:
    metadata = adapter.metadata()
    settings = _json(server_info) if server_info and Path(server_info).exists() else None
    return {"accelerator": gpu, "server": metadata.capabilities.get("server"),
            "vllm": metadata.backend_version, "software": software_stack(),
            "server_settings": settings, "workers": workers, "opened_at": _now()}


def _environment(verified: Mapping) -> dict:
    """The software stack and the data the run or pilot uses (ADR-0008 D4, D6)."""
    return {"software": software_stack(),
            "data": {"alfworld_release": verified["release"], "archives": verified["archives"],
                     "games_verified": verified["games"]}}


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
    environment = server_environment()
    url = f"http://127.0.0.1:{args.port}"
    earlier = _json(args.info) if Path(args.info).exists() else {}
    if process_alive(earlier.get("pid")):
        if (earlier.get("command"), earlier.get("environment"), earlier.get("port")) == \
                (command, environment, args.port):
            # this command's own server from an earlier attempt (an interrupted serve, say):
            # wait for it if it is still loading, then reuse it
            wait_until_ready(url, prepared["model_id"], process=_Pid(earlier["pid"]),
                             log_path=args.log, timeout_seconds=args.timeout)
            print(f"the model server (pid {earlier['pid']}) from an earlier attempt serves "
                  f"{prepared['model_id']} with these settings on port {args.port}; reusing it")
            return 0
        raise CliError(f"an earlier model server (pid {earlier['pid']}) with other settings still "
                       f"runs, perhaps still loading, and holds the GPU. Kill it (kill "
                       f"{earlier['pid']}), or use Runtime > Disconnect and delete runtime; "
                       f"restarting the kernel does not stop it.")
    if port_answers(args.port):
        raise CliError(f"a server already answers on port {args.port}, and it is not one this "
                       f"command started (find it with: pgrep -f 'vllm serve'). Kill it, or use "
                       f"Runtime > Disconnect and delete runtime; restarting the kernel does not "
                       f"stop it.")
    process = start_server(command, args.log, environment=environment)
    # written before the wait, so a serve interrupted while the model loads finds it
    _write_json(args.info, {"command": command, "environment": environment, "pid": process.pid,
                            "port": args.port, "started_at": _now()})
    wait_until_ready(url, prepared["model_id"], process=process, log_path=args.log,
                     timeout_seconds=args.timeout)
    print(f"the model server (pid {process.pid}) serves {prepared['model_id']} on port "
          f"{args.port}; log {args.log}, settings {args.info}")
    return 0


class _Pid:
    """A running process known only by its pid, for ``wait_until_ready``."""

    def __init__(self, pid: int):
        self.pid, self.returncode = pid, None

    def poll(self):
        if process_alive(self.pid):
            return None
        self.returncode = "unknown (not a child of this process)"
        return self.returncode


def cmd_preflight(args) -> int:
    prepared = load_prepared(args.prepared)
    print(json.dumps(_preflight(_adapter(args, prepared)), indent=2))
    return 0


def _preflight(adapter: VLLMServerAdapter) -> dict:
    """G0's own message shape (system and user, ReActPlanner) with its stop
    sequence, and the teacher's (one user message, ``teacher_messages``)
    under the miner's schema: a chat template that refuses either shape
    fails here, before any episode."""
    from evaluation.benchmarks.alfworld.splits import ALFWORLD_TASK_TYPES
    from runtime.model_adapters.base import GenerationSettings
    from runtime.policies.planner_react import STOP_SEQUENCES, ReActPlanner
    from runtime.tool_adapters.alfworld import ACT_TOOL
    from traits.miner.miner import output_schema, render_teacher_prompt, teacher_messages

    task = ("You are in the middle of a room.\n\nYour task is to: look at the room.\n\n"
            "Admissible commands:\nlook")
    planner = ReActPlanner({"style": "react", "max_plan_steps": 1})
    reply = adapter.generate(
        planner.build_messages(task=task, observations=[task], skills=(), tools=[ACT_TOOL],
                               step=1, max_steps=1),
        GenerationSettings(temperature=0.0, max_tokens=64, seed=None, stop=STOP_SEQUENCES))
    if reply.prompt_tokens is None or reply.completion_tokens is None:
        raise CliError("the server reports no token usage; the run needs exact counts")
    prompt = render_teacher_prompt(
        evidence_text="(a preflight call: no trajectories; propose at most one short skill)",
        family_universe=tuple(sorted(ALFWORLD_TASK_TYPES)), max_proposals=1)
    teacher = adapter.structured_generate(  # validated against the schema by the adapter
        teacher_messages(prompt), GenerationSettings(temperature=0.7, max_tokens=2048, seed=11),
        output_schema())
    return {"react_reply": reply.text[:200], "react_tokens": [
        reply.prompt_tokens, reply.completion_tokens], "teacher_reply": json.loads(teacher.text),
        "teacher_finish_reason": teacher.finish_reason}


EXIT_UNMIRRORED = 3
UNMIRRORED = "UNMIRRORED.json"  # left in a run directory whose last snapshot failed
PILOT_START = "pilot-start.json"  # the code and model a pilot directory belongs to


def _say(message: str) -> None:
    """A line on stderr that never raises: a command whose notebook kernel
    died writes into a pipe nobody reads (BrokenPipeError)."""
    try:
        print(message, file=sys.stderr, flush=True)
    except OSError:
        pass


@contextlib.contextmanager
def _sigint_deferred():
    """The last snapshot is never cut short: a SIGINT (the stop button, or the
    watcher of a dead kernel) that arrives during it is dropped, since the
    command ends right after. A second stop press still ends the command at
    once: the notebook helper sends SIGTERM to its process group."""
    import signal
    import threading
    previous = signal.getsignal(signal.SIGINT)
    if previous is None or threading.current_thread() is not threading.main_thread():
        yield  # no Python handler to restore, or not the main thread: leave it alone
        return
    signal.signal(signal.SIGINT, lambda *_: None)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, previous)


def _final_snapshot(mirror: RunMirror | None, local: Path, *, marker: bool = True) -> bool:
    """The last snapshot of a session; False (with a marker file and a loud
    message) when it fails, so nobody releases the machine that holds the
    only copy of the newest records. It is not interrupted, and anything
    that stops it short of a whole snapshot counts as a failure; only another
    machine taking over the mirror is raised (ADR-0008 D5). A rehearsal's
    records are disposable: ``marker=False`` leaves none."""
    if mirror is None:
        return True
    with _sigint_deferred():  # the snapshot and its marker, as one step
        path, error = local / UNMIRRORED, None
        try:
            seq = mirror.snapshot()
        except MirrorError:
            raise
        except (Exception, KeyboardInterrupt) as exc:  # noqa: BLE001 - nothing was mirrored
            seq, error = None, repr(exc)
        if seq is not None:
            path.unlink(missing_ok=True)
            return True
        if not marker:
            _say(f"the rehearsal's last snapshot failed ({error or mirror.last_error})")
            return False
        _write_json(path, {"failures": mirror.failures,
                           "last_error": error or mirror.last_error, "at": _now()})
    _say(f"UNMIRRORED: the last snapshot failed ({error or mirror.last_error}); the newest "
         f"records of {local} exist only on this machine. Mount Google Drive again and run "
         f"`snapshot` before the runtime is released.")
    return False


def _restore_if_missing(local: Path, mirror_dir: Path | None) -> None:
    if mirror_dir is None or (local.exists() and any(local.iterdir())):
        return
    if latest_snapshot(mirror_dir) is None:
        return  # a fresh run
    doc = restore(mirror_dir, local)
    (local / UNMIRRORED).unlink(missing_ok=True)  # describes another machine, never mirrored
    _out(f"restored {local} from mirror snapshot {doc['seq']} ({len(doc['files'])} files)")


STOPPED = "stopped: every recorded episode is kept; run the same command again to resume"


@contextlib.contextmanager
def _run_dir_lock(local: Path, *, marker: bool = True):
    """The run directory's one-writer lock, held from before the mirror is
    taken over until after the last snapshot. A second command on the same
    directory (one started while a command whose kernel restarted still
    stops) fails here, before it touches the mirror or the ledger. A
    rehearsal (``marker=False``) leaves no unmirrored marker on SIGTERM: its
    records are disposable, and a marker would keep the notebook's runtime."""
    from evaluation.exp0001.pipeline import lock_run_dir
    local.mkdir(parents=True, exist_ok=True)
    lock = lock_run_dir(local)
    restore = _marker_on_sigterm(local) if marker else (lambda: None)
    try:
        yield lock
    finally:
        restore()
        lock.close()


def _marker_on_sigterm(local: Path):
    """While a command holds its run directory, SIGTERM (the stop button's
    second press ends the command's process group at once) first leaves the
    unmirrored marker, then ends the process: the records since the last
    snapshot may exist only here, so the notebooks keep the runtime until
    `snapshot` has run. Returns the function that restores the previous
    handler (a no-op outside the main thread)."""
    import os
    import signal
    import threading
    if threading.current_thread() is not threading.main_thread():
        return lambda: None

    def terminated(signum, frame):
        try:
            _write_json(local / UNMIRRORED, {"failures": None, "at": _now(),
                                             "last_error": "terminated (SIGTERM) before its "
                                                           "last snapshot was confirmed"})
        except Exception as exc:  # noqa: BLE001 - say so; the process ends either way
            _say(f"UNMIRRORED: the marker could not be written ({exc!r}); the newest records "
                 f"of {local} may exist only on this machine. Run `snapshot` before the "
                 f"runtime is released.")
        finally:
            signal.signal(signal.SIGTERM, signal.SIG_DFL)
            os.kill(os.getpid(), signal.SIGTERM)

    previous = signal.signal(signal.SIGTERM, terminated)
    return lambda: signal.signal(signal.SIGTERM, previous if previous is not None
                                 else signal.SIG_DFL)


def _stopped(mirrored: bool) -> int:
    """A command stopped by its stop button or its kernel: 130 once the last
    snapshot is taken, and still 3 when it failed."""
    _say(STOPPED)
    return 130 if mirrored else EXIT_UNMIRRORED


def cmd_pilot(args) -> int:
    gpu = require_a100_80gb(gpu_inventory())
    prepared = load_prepared(args.prepared)

    def counter():  # the miner prompts of the pilot's pairs, with the model's own tokenizer
        return hf_token_counter(prepared["model_id"], prepared["revision"],
                                path=Path(prepared["path"]))

    return _pilot(args, gpu=gpu, prepared=prepared, adapter=lambda: _adapter(args, prepared),
                  counter=counter)


def _pilot(args, *, gpu: dict, prepared: dict, adapter, counter, run_id: str = PILOT_RUN_ID,
           report_name: str = "pilot-report.json", report_fields: Mapping | None = None,
           disposable: bool = False) -> int:
    """The pilot from its checks to its last snapshot; ``rehearse`` runs the
    same against its stand-in server (its own run id and report, and
    ``disposable``: no unmirrored marker)."""
    from runtime.expression import compile_runtime_config
    from runtime.seed_g0 import build_g0_seed_genome
    from genome.validation.registry import TraitRegistry

    repo = Path(args.repo)
    commit, dirty = git_state(repo)
    if dirty:
        raise CliError(f"the checkout at {repo} has changes; the pilot report names a clean commit")
    splits, pilot = _manifests(repo)
    verified = verify_data(args.data_root, splits, pilot)  # seconds; before any episode
    out = Path(args.out_dir)
    _restore_if_missing(out, args.mirror_dir)
    report_path = out / report_name
    if report_path.exists():
        _out(report_path.read_text(encoding="utf-8"))
        return 0
    adapter = adapter()
    counter = counter()
    with _run_dir_lock(out, marker=not disposable):  # one writer, before the mirror is taken over
        # a resumed pilot plays on with the code, model and weights it started with, or not
        # at all
        identity = {"run_id": run_id, "commit": commit, "model_id": prepared["model_id"],
                    "revision": prepared["revision"],
                    "weights_sha256": prepared["weights"]["sha256"]}
        started = out / PILOT_START
        if (out / "ledger.jsonl").exists() and \
                (not started.exists() or _json(started) != identity):
            raise CliError(f"{out} holds a pilot of "
                           f"{_json(started) if started.exists() else 'unknown'} code and "
                           f"model, not {identity}; give this pilot a new directory (and a "
                           f"new mirror)")
        if not started.exists():
            _write_json(started, identity)
        mirror = RunMirror(out, args.mirror_dir) if args.mirror_dir else None
        ledger = RunLedger(out / "ledger.jsonl", run_id=run_id, clock=time.time, caps=CAPS,
                           session_info=_session_info(gpu, adapter, args.workers,
                                                      args.server_info),
                           on_append=Progress(mirror, args.progress_every))
        registry = TraitRegistry(out / "registry")
        g0 = build_g0_seed_genome(registry)
        pilot_games = {("train", g["game_id"]): g["content_sha256"] for g in pilot["games"]}
        executor = EpisodeExecutor(
            model=adapter, env_factory=AlfworldEnvFactory(args.data_root, splits,
                                                          extra_games=pilot_games),
            ledger=ledger, trajectories_dir=out / "trajectories", run_id=run_id,
            code_commit=commit, budgets=protocol_budgets(), workers=args.workers)
        started = _now()
        report, stopped, mirrored = None, False, False
        try:
            try:
                measured = run_pilot(executor, pilot, compile_runtime_config(g0, registry),
                                     count_prompt_tokens=counter.count_teacher_prompt_tokens,
                                     timing_path=out / "pilot-throughput.json")
            finally:
                executor.close()
            report = {**_pilot_report(args, measured, splits, pilot, adapter, prepared, gpu,
                                      verified, commit, dirty, started),
                      **dict(report_fields or {})}
            freeze_json(report_path, report)
        except KeyboardInterrupt:
            stopped = True
        finally:
            mirrored = _final_snapshot(mirror, out, marker=not disposable)
    if stopped:  # a rehearsal's records are disposable: it says itself that it stopped
        return 130 if disposable else _stopped(mirrored)
    _out(json.dumps({"measures": report["measures"], "projection": report["projection"]},
                    indent=2))
    return 0 if mirrored else EXIT_UNMIRRORED


def _pilot_report(args, measured, splits, pilot, adapter, prepared, gpu, verified, commit,
                  dirty, started) -> dict:
    projection = project_exp0001(measured, t_mine_size=len(splits["pools"]["t_mine"]["games"]),
                                 t_eval_size=len(splits["pools"]["t_eval"]["games"]),
                                 transfer_size=len(splits["pools"]["ablation"]["games"]))
    metadata = adapter.metadata()
    report = {"schema": PILOT_REPORT_SCHEMA, "experiment": "EXP-0001",
              "pilot_sha256": pilot["pilot_sha256"], "started_at": started,
              "finished_at": _now(), "code": {"commit": commit, "dirty": dirty},
              "model": {"model_id": metadata.model_id, "revision": metadata.revision,
                        "backend": metadata.backend, "backend_version": metadata.backend_version,
                        "weights_sha256": prepared["weights"]["sha256"],
                        "chat_template_kwargs": metadata.capabilities.get(
                            "chat_template_kwargs")},
              "accelerator": gpu, "workers": args.workers, "environment": _environment(verified),
              "note": "a calibration measurement; it tunes nothing and is no EXP-0001 result",
              **measured, "projection": projection}
    return report


def _empty_dir(path: Path | None, what: str) -> None:
    if path is not None and Path(path).exists() and any(Path(path).iterdir()):
        raise CliError(f"the rehearsal's {what} {path} is not empty; give each rehearsal a new "
                       f"one (a rehearsal never touches a pilot or a run)")


def _ledger_lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


def _rehearsal_stopped() -> int:
    _say("stopped: the rehearsal did not finish; run it again with a new directory and a new "
         "mirror directory")
    return 130


def cmd_rehearse(args) -> int:
    """The pilot without a GPU or a model (``evaluation.exp0001.rehearsal``):
    the same checks, games, workers, ledger, tokenizer, mirror and report,
    against a stand-in server, with one outage and a restore on the way."""
    import multiprocessing
    import shutil
    from concurrent.futures import ProcessPoolExecutor
    from argparse import Namespace

    from evaluation.exp0001 import rehearsal
    from evaluation.exp0001.pilot_run import PILOT_ATTEMPT_SEED, pilot_specs
    from evaluation.exp0001.protocol import attempt_sampling_seed
    from evaluation.exp0001.serving import resolve_revision
    from genome.validation.registry import TraitRegistry
    from runtime.expression import compile_runtime_config
    from runtime.model_adapters.errors import BackendUnavailableError
    from runtime.seed_g0 import build_g0_seed_genome

    out, mirror_dir = Path(args.out_dir), args.mirror_dir
    _empty_dir(out, "directory")
    _empty_dir(mirror_dir, "mirror directory")
    repo = Path(args.repo)
    commit, dirty = git_state(repo)
    if dirty:
        raise CliError(f"the checkout at {repo} has changes; the pilot report names a clean commit")
    splits, pilot = _manifests(repo)
    verify_data(args.data_root, splits, pilot)
    revision = resolve_revision(args.model, args.revision)
    counter = hf_token_counter(args.model, revision)  # the tokenizer alone, no weights
    _out(f"[rehearsal] {args.model}@{revision}: tokenizer loaded; Python {sys.version.split()[0]}")

    games = sorted(g["game_id"] for g in pilot["games"])
    factory = AlfworldEnvFactory(args.data_root, splits, extra_games={
        ("train", g["game_id"]): g["content_sha256"] for g in pilot["games"]})
    max_steps = protocol_budgets().max_steps
    jobs = ([factory] * len(games), [str(args.data_root)] * len(games), games,
            [max_steps] * len(games))
    if args.workers == 1:  # in this process, as the pilot's episodes then are
        plans = list(map(rehearsal.game_plan, *jobs))
    else:
        import os
        hash_seed = os.environ.get("PYTHONHASHSEED")
        os.environ["PYTHONHASHSEED"] = "0"  # the plan workers only: the expert's set orders
        try:
            with ProcessPoolExecutor(max_workers=min(args.workers, len(games)),
                                     mp_context=multiprocessing.get_context("spawn")) as pool:
                plans = list(pool.map(rehearsal.game_plan, *jobs))
        finally:
            if hash_seed is None:
                os.environ.pop("PYTHONHASHSEED", None)
            else:
                os.environ["PYTHONHASHSEED"] = hash_seed
    tasks = [plan["task"] for plan in plans]
    for plan in plans:  # a task text two games share cannot tell them apart: never won
        plan["won"] = plan["won"] and tasks.count(plan["task"]) == 1
    expected = rehearsal.expected_counts(plans)
    _out(f"[rehearsal] {len(plans)} pilot games opened in worker processes; the expert wins "
         f"{sum(p['won'] for p in plans)}; the stand-in will win {expected}")

    policy = rehearsal.StandInPolicy({p["task"]: p["commands"] for p in plans if p["won"]})
    prepared = {"model_id": args.model, "revision": revision, "path": None,
                "weights": {"sha256": None, "note": "a rehearsal loads no weights"}}
    checks: dict[str, bool] = {}
    from evaluation.exp0001.protocol import CHAT_TEMPLATE_KWARGS
    check_schema = rehearsal.xgrammar_check(counter.tokenizer)
    with rehearsal.StandInServer(args.model, policy, count_prompt=counter.count_chat_tokens,
                                 template_kwargs=CHAT_TEMPLATE_KWARGS,
                                 check_schema=check_schema) as server:
        session = Namespace(repo=repo, data_root=args.data_root, out_dir=out,
                            mirror_dir=mirror_dir, workers=args.workers, server_url=server.url,
                            server_info=None, progress_every=args.progress_every)

        def adapter():  # retried briefly: the outage below must end the pass quickly
            return _adapter(session, prepared, transport_retries=1, backoff_seconds=0.05)

        def run():
            return _pilot(session, gpu=dict(rehearsal.REHEARSAL_GPU), prepared=prepared,
                          adapter=adapter, counter=lambda: counter,
                          run_id=rehearsal.REHEARSAL_RUN_ID,
                          report_name=rehearsal.REHEARSAL_REPORT, report_fields={
                              "schema": rehearsal.REHEARSAL_REPORT_SCHEMA,
                              "note": "a rehearsal against a stand-in model server: it checks "
                                      "the code path and measures nothing about any model"},
                          disposable=True)

        preflight = _preflight(adapter())
        checks["preflight"] = preflight["teacher_reply"] == {"proposals": []}
        server.outage_seed = attempt_sampling_seed(PILOT_ATTEMPT_SEED, games[len(games) // 2], 1)
        try:
            if run() == 130:  # stopped (the stop button, or the kernel): no second pass
                return _rehearsal_stopped()
            checks["the outage stopped the first pass"] = False
        except BackendUnavailableError as exc:
            checks["the outage stopped the first pass"] = server.url in str(exc)
        server.outage_seed = None
        first = {r["key"] for r in _ledger_lines(out / "ledger.jsonl")
                 if r.get("type") == "episode"}
        checks["the first pass recorded part of the pilot"] = 0 < len(first)
        checks["the first pass left no unmirrored marker"] = not (out / UNMIRRORED).exists()
        if mirror_dir is not None:
            latest = latest_snapshot(mirror_dir)
            # one snapshot at the first record and one at the end (the 60-second cadence
            # between them needs a longer pass than a fast machine plays)
            checks["the ledger hook snapshotted the first pass"] = \
                latest is not None and latest["seq"] >= 2
            shutil.rmtree(out)  # the rehearsal's own directory: the next pass restores it
        _out(f"[rehearsal] first pass stopped by the outage after {len(first)} episodes; "
             + ("restoring from the mirror" if mirror_dir is not None else "resuming in place"))
        before = (server.requests, server.refused)
        code = run()
        if code == 130:
            return _rehearsal_stopped()
        checks["the second pass finished"] = code == 0
        second_requests = server.requests - before[0] - (server.refused - before[1])
    report = _json(out / rehearsal.REHEARSAL_REPORT)
    lines = _ledger_lines(out / "ledger.jsonl")
    registry = TraitRegistry(out / "registry")
    specs = pilot_specs(pilot, compile_runtime_config(build_g0_seed_genome(registry), registry))
    keys = sorted(spec.key for kind in specs.values() for spec in kind)
    recorded = [r["key"] for r in lines if r.get("type") == "episode"]
    ledger = RunLedger(out / "ledger.jsonl", run_id=rehearsal.REHEARSAL_RUN_ID, clock=time.time,
                       caps=CAPS, readonly=True).summary()
    m = report["measures"]
    checks.update({
        "every pilot episode recorded once": sorted(recorded) == keys,
        # every request of the second pass belongs to an episode the first did not record
        "the second pass played only what the first did not record": second_requests == sum(
            r["model_calls"] for r in lines if r.get("type") == "episode"
            and r["key"] not in first),
        "the outage's attempt was charged": ledger["aborted_attempts"] >= 1,
        "the restore is on record": mirror_dir is None or "restore" in ledger["notes"],
        "the temperature-0 successes": m["g0_success_rate_p0"]["numerator"]
        == expected["t0_successes"],
        "the temperature-0.7 successes": m["temperature_0_7_attempt_success_rate"]["numerator"]
        == expected["t07_successes"],
        "the temperature-0.7 pairs": m["temperature_0_7_pair_yield"]["numerator"]
        == expected["t07_pairs"],
        "solved in any attempt": m["temperature_0_7_solved_in_any_attempt"]["numerator"]
        == expected["t07_solved_in_any_attempt"],
        "the repeats are identical": m["temperature_0_repeat_determinism"]["all_identical"]
        is True,
        "the arm-B smoke test": m["arm_b_rendering_smoke_test"]["passed"] is True,
        "the pairs' teacher prompts counted with the tokenizer":
            m["miner_prompt_tokens_t07_pairs"]["counted"] is True
            and m["miner_prompt_tokens_t07_pairs"]["pairs"] == expected["t07_pairs"],
        # the model's chat template rendered at least one real teacher prompt
        "a pair's teacher prompt was rendered": expected["t07_pairs"] >= 1,
        "no prompt overflowed": m["prompt_tokens_per_call_max"]["context_overflows"] == 0,
    })
    failed = [name for name, ok in checks.items() if not ok]
    summary = {"schema": rehearsal.REHEARSAL_PASSED_SCHEMA, "passed": not failed,
               "code": {"commit": commit}, "python": sys.version.split()[0],
               "software": software_stack(), "model": {"model_id": args.model,
                                                       "revision": revision,
                                                       "tokenizer": counter.identity},
               "workers": args.workers, "expected": expected, "checks": checks,
               "stand_in_requests": server.requests,
               "teacher_schema_compiled_with_xgrammar": server.schemas_compiled > 0,
               "finished_at": _now()}
    _out(json.dumps(summary, indent=2))
    if failed:
        raise CliError(f"the rehearsal failed: {failed}")
    _write_json(out / rehearsal.REHEARSAL_PASSED, summary)
    _out(f"[rehearsal] PASSED; record {out / rehearsal.REHEARSAL_PASSED}")
    return 0


def cmd_rehearsal_check(args) -> int:
    """Whether a passed rehearsal covers this machine: the same clean commit,
    model and revision, Python and pinned software. The notebooks rehearse
    again before the model download when it does not."""
    from evaluation.exp0001 import rehearsal
    passed = _json(args.passed)
    if passed.get("schema") != rehearsal.REHEARSAL_PASSED_SCHEMA or passed.get("passed") is not True:
        raise CliError(f"{args.passed} is not a passed rehearsal")
    commit, dirty = git_state(Path(args.repo))
    if dirty:
        raise CliError(f"the checkout at {args.repo} has changes; a rehearsal covers a clean commit")
    now = {"commit": commit, "model_id": args.model, "python": sys.version.split()[0],
           "software": software_stack()}
    then = {"commit": passed["code"]["commit"], "model_id": passed["model"]["model_id"],
            "python": passed["python"], "software": passed["software"]}
    if args.revision is not None:
        now["revision"], then["revision"] = args.revision, passed["model"]["revision"]
    differ = {key: {"rehearsed": then[key], "here": now[key]} for key in now
              if then[key] != now[key]}
    if differ:
        raise CliError(f"the rehearsal does not cover this machine: {json.dumps(differ)}")
    _out(f"the rehearsal of {commit} with {args.model}@{passed['model']['revision']} passed on "
         f"this Python and software ({passed['finished_at']})")
    return 0


def _run_start(args, local: Path, prepared: dict, adapter: VLLMServerAdapter, splits: dict,
               pilot: dict, commit: str, dirty: bool, tokenizer: str, environment: dict) -> dict:
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
        started_with = (record.get("environment") or {}).get("software")
        if started_with is not None and started_with != environment["software"]:
            changed = {name: (started_with.get(name), version)
                       for name, version in environment["software"].items()
                       if started_with.get(name) != version}
            raise CliError(f"the installed software differs from the run's (recorded, installed): "
                           f"{changed}. Install the notebook's pinned versions (ADR-0008 D4).")
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
        run_id=args.run_id, started_at=_now(), environment=environment)


def cmd_run(args) -> int:
    from evaluation.exp0001.pipeline import Exp0001Run

    gpu = require_a100_80gb(gpu_inventory())
    prepared = load_prepared(args.prepared)
    repo, local = Path(args.repo), Path(args.local_dir)
    commit, dirty = git_state(repo)
    splits, pilot = _manifests(repo)
    verified = verify_data(args.data_root, splits, pilot)  # seconds; before any episode
    _restore_if_missing(local, args.mirror_dir)
    adapter = _adapter(args, prepared)
    counter = hf_token_counter(prepared["model_id"], prepared["revision"],
                               path=Path(prepared["path"]))
    record = _run_start(args, local, prepared, adapter, splits, pilot, commit, dirty,
                        counter.identity, _environment(verified))
    with _run_dir_lock(local) as lock:  # one writer, before the mirror is taken over
        mirror = RunMirror(local, args.mirror_dir) if args.mirror_dir else None
        started = time.monotonic()
        budget = None if args.stop_after_hours is None else args.stop_after_hours * 3600
        run = Exp0001Run(run_dir=local, run_start=record, split_manifest=splits, model=adapter,
                         teacher=adapter, env_factory=AlfworldEnvFactory(args.data_root, splits),
                         token_counter=counter, workers=args.workers,
                         session_info=_session_info(gpu, adapter, args.workers,
                                                    args.server_info),
                         on_ledger_append=Progress(mirror, args.progress_every),
                         pause_when=None if budget is None
                         else (lambda: time.monotonic() - started > budget), lock=lock)
        report, stopped, mirrored = None, False, False
        try:
            report = run.run()
        except RunPaused as exc:
            _out(f"paused after {args.stop_after_hours} h of this session ({exc}); every "
                 f"admitted episode is recorded. Run the same command again to resume.")
        except KeyboardInterrupt:
            stopped = True
        finally:
            mirrored = _final_snapshot(mirror, local)
            if mirror and mirror.failures:
                _say(f"[mirror] {mirror.failures} snapshot(s) failed; last: {mirror.last_error}")
    if stopped:
        return _stopped(mirrored)
    if report is not None:
        _out(json.dumps({k: report.get(k) for k in ("status", "stage", "reason", "episodes",
                                                      "model_calls", "aborted_attempts")},
                        indent=2))
    return 0 if mirrored else EXIT_UNMIRRORED


def cmd_snapshot(args) -> int:
    local = Path(args.run_dir)
    if not (local / "ledger.jsonl").exists():
        raise CliError(f"{local} holds no run or pilot ledger")
    with _run_dir_lock(local):  # a command still running takes its own snapshots
        if not _final_snapshot(RunMirror(local, args.mirror_dir), local):
            return EXIT_UNMIRRORED
    _out(f"snapshot of {local} taken in {args.mirror_dir}")
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

    p = sub.add_parser("rehearse")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--revision", default=None, help="a commit, branch or tag; default the "
                   "hub's current commit")
    p.add_argument("--data-root", type=Path, required=True,
                   help="ALFWorld's json_2.1.1 directory")
    p.add_argument("--repo", type=Path, default=Path("."))
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--progress-every", type=int, default=50)
    p.add_argument("--out-dir", type=Path, required=True, help="a new, empty directory")
    p.add_argument("--mirror-dir", type=Path, default=None,
                   help="a new, empty directory (on Colab, on Google Drive): the rehearsal "
                        "restores the pilot from it once")
    p.set_defaults(func=cmd_rehearse)

    p = sub.add_parser("rehearsal-check")
    p.add_argument("--passed", type=Path, required=True, help="a rehearsal-passed.json")
    p.add_argument("--repo", type=Path, default=Path("."))
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--revision", default=None, help="the revision the run will pin, if known")
    p.set_defaults(func=cmd_rehearsal_check)

    p = sub.add_parser("status")
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--mirror-dir", type=Path, default=None)
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("snapshot")
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--mirror-dir", type=Path, required=True)
    p.set_defaults(func=cmd_snapshot)

    p = sub.add_parser("repair-ledger")
    p.add_argument("--run-dir", type=Path, required=True)
    p.set_defaults(func=cmd_repair_ledger)
    return top


STOP_WITH_PARENT = "HERITAGENT_STOP_WITH_PARENT"  # the starting process's pid (notebook helper)


def _stop_with_parent(parent: int, poll_seconds: float = 2.0) -> None:
    """When ``parent``, the process that started this one, is no longer our
    parent (the notebook's kernel restarted or died, even before this
    process got here), stop as at the stop button: one SIGINT to ourselves,
    so the episodes in flight are recorded and a last snapshot is taken."""
    import os
    import signal
    import threading

    # the stop is a SIGINT: it must raise KeyboardInterrupt even in a process
    # started with SIGINT ignored (a background shell, say)
    signal.signal(signal.SIGINT, signal.default_int_handler)

    def watch():
        while os.getppid() == parent:
            time.sleep(poll_seconds)
        os.kill(os.getpid(), signal.SIGINT)

    threading.Thread(target=watch, name="stop-with-parent", daemon=True).start()


def main(argv: list[str] | None = None) -> int:
    import os
    args = parser().parse_args(argv)
    parent = os.environ.get(STOP_WITH_PARENT, "")
    try:
        if parent.isdigit():  # its SIGINT may come at once: inside the try
            _stop_with_parent(int(parent))
        return args.func(args)
    except (CliError, BackendError, ProvenanceError, LedgerError, MirrorError) as exc:
        _say(f"error: {exc}")
        return 2
    except AdapterError as exc:  # the model server refused or failed a request
        _say(f"error: the model server: {type(exc).__name__}: {exc}")
        return 2
    except KeyboardInterrupt:
        if args.func is cmd_rehearse:  # stopped outside a pass: a rehearsal never resumes
            return _rehearsal_stopped()
        _say(STOPPED)
        return 130
    except RuntimeError as exc:
        from evaluation.exp0001.pipeline import RunError  # a run raised it: already loaded
        if not isinstance(exc, RunError):
            raise
        _say(f"error: {exc}")
        if "in use by another run" in str(exc):
            _say("a previous `run`, `pilot` or `snapshot` on this directory may still be running "
                 "in this machine (one whose notebook kernel restarted stops by itself, saving "
                 "its episodes). Wait for it to end; the notebook's release() keeps the GPU "
                 "until then. Or end its process group.")
        return 2


if __name__ == "__main__":
    sys.exit(main())
