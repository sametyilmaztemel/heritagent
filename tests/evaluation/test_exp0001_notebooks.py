"""The Colab notebooks (issue #13 Phase B; ADR-0008) call the command line
with flags it has, and a failing command stops "Run all": every
``cli(...)`` call of every code cell parses, no cell uses a ``!`` shell
escape (Colab carries on after one fails), the helper raises on a non-zero
exit, and the error handler is installed before the first command."""

import ast
import json
import os
import signal
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from evaluation.exp0001 import cli

NOTEBOOKS = Path(__file__).resolve().parents[2] / "notebooks"
NAMES = ["EXP-0001-pilot.ipynb", "EXP-0001-run.ipynb"]  # the A100 notebooks
REHEARSAL = "EXP-0001-rehearsal.ipynb"  # the CPU notebook that comes before the pilot
RUN_REHEARSAL = "EXP-0001-run-rehearsal.ipynb"  # the CPU notebook that comes before the run
CPU = [REHEARSAL, RUN_REHEARSAL]
ALL = NAMES + CPU
# the optional flags' two branches: every other name reads as "1"
BRANCHES = [{}, {"STOP_AFTER_HOURS": None, "revision": None}]


class Names(dict):
    def __missing__(self, key):
        return "1"


def code_cells(name):
    notebook = json.loads((NOTEBOOKS / name).read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    return ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]


def cli_calls(source):
    return [node for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id == "cli"]


def argv(call, branch):
    captured = []
    names = Names(branch, cli=lambda *args: captured.extend(map(str, args)))
    eval(compile(ast.Expression(call), "<cell>", "eval"), {"__builtins__": {"map": map, "str": str}},
         names)
    return captured


@pytest.mark.parametrize("name", ALL)
def test_every_command_of_the_notebook_parses(name):
    commands = set()
    for source in code_cells(name):
        for call in cli_calls(source):
            for branch in BRANCHES:
                args = cli.parser().parse_args(argv(call, branch))  # an unknown flag exits
                commands.add(args.command)
    data = {"env-check", "mirror-check", "fetch-data", "verify-data"}
    expected = {
        REHEARSAL: data | {"rehearse"}, RUN_REHEARSAL: data | {"rehearse-run"},
        # the pilot rehearses again where a passed rehearsal does not cover the machine
        NAMES[0]: data | {"rehearse", "gpu-check", "prepare-model", "serve", "preflight",
                          "rehearsal-check", "pilot"},
        # the run never does: a run rehearsal takes hours of a CPU runtime
        NAMES[1]: data | {"gpu-check", "prepare-model", "serve", "preflight", "rehearsal-check",
                          "run", "status", "figures"}}[name]
    assert commands == expected


@pytest.mark.parametrize("name", ALL)
def test_no_cell_uses_a_shell_escape_and_the_error_handler_comes_first(name):
    cells = code_cells(name)
    for source in cells:
        assert not [line for line in source.splitlines() if line.lstrip().startswith(("!", "%"))]
    handler = next(i for i, source in enumerate(cells) if "set_custom_exc" in source)
    first_command = next(i for i, source in enumerate(cells) if cli_calls(source))
    gpus = [i for i, source in enumerate(cells) if "nvidia-smi" in source]
    if name in CPU:  # a CPU runtime: no GPU is asked for
        assert not gpus and handler < first_command
        assert "cuda.is_available" not in "".join(cells)
    else:
        assert handler < gpus[0] < first_command
    assert "runtime.unassign()" in cells[handler]


def first(cells, needle):
    return next(i for i, source in enumerate(cells) if needle in source)


@pytest.mark.parametrize("name", NAMES)
def test_the_a100_notebooks_need_a_passed_rehearsal_before_the_model(name):
    """No rehearsal of this commit and model: the notebook stops before the
    install, the model download, the server and any episode. A rehearsal that
    does not cover this machine is repeated here, still before the model
    download."""
    cells = code_cells(name)
    required = first(cells, "if not os.path.exists(PASSED):")
    covers = first(cells, '"rehearsal-check"')
    model = first(cells, '"prepare-model"')
    assert first(cells, "def install():") > required < covers < model < first(cells, 'cli("serve"')
    # before the guard, at most the repository itself is installed (seconds), so
    # the run notebook can read a resumed run's commit
    for source in cells[:required]:
        for line in source.splitlines():
            if '"install"' in line:
                assert line.rstrip().endswith('"-e", REPO], check=True)'), line
    # only the pilot can be resumed by naming its commit
    assert ("heritagent/pilot/" in cells[required]) == ("pilot" in name)
    assert first(cells, '"verify-data"') < covers  # it needs the data
    if "pilot" in name:
        assert '"rehearse"' in cells[covers] and "except RuntimeError:" in cells[covers]
        # a rehearsal repeated here is recorded where the next session's guard looks
        fallback = cells[covers].split("except RuntimeError:", 1)[1]
        assert fallback.index('"rehearse"') < fallback.index('/rehearsal-passed.json", PASSED)')
    else:
        # the run's guard reads the run rehearsal's record and stops where it does not cover
        # this machine: the error handler releases the GPU, nothing is rehearsed on the A100
        assert "rehearsal/run-passed-{COMMIT}" in cells[required]
        assert "EXP-0001-run-rehearsal.ipynb" in cells[required]
        assert [argv(call, {}) for call in cli_calls(cells[covers])] == [[
            "rehearsal-check", "--kind", "run", "--passed", "1", "--repo", "1", "--model", "1",
            "--revision", "1", "--workers", "1"]]
        assert cells[covers].rstrip().endswith("raise")
        assert 'cli("rehearse' not in "".join(cells)  # the run notebook never rehearses
        # the run itself checks the same record, and the preflight sends a gate's request
        run = argv(cli_calls(cells[first(cells, 'cli("run"')])[0], {})
        assert run[run.index("--rehearsal-passed") + 1] == "1"  # PASSED
        assert "--gate-shape" in argv(cli_calls(cells[first(cells, '"preflight"')])[0], {})
        # the guard checks the commit a resumed run continues with
        assert first(cells, '"run-commit"') < required
        assert first(cells, "COMMIT = checkout(run_commit)") < required
        assert "if run_commit:" in cells[required]  # whatever CODE_REF says


def test_the_rehearsal_writes_the_records_the_a100_notebooks_read():
    marker = "passed-{COMMIT}-{MODEL.replace('/', '--')}.json"
    revision = "model-revision-{MODEL.replace('/', '--')}.txt"
    for name in ALL:
        text = "".join(code_cells(name))
        assert marker in text and revision in text, name
    # the run reads only the run rehearsal's record, which only the run rehearsal writes
    for name in (NAMES[1], RUN_REHEARSAL):
        assert f"run-{marker}" in "".join(code_cells(name)), name
    for name in (NAMES[0], REHEARSAL):
        assert f"run-{marker}" not in "".join(code_cells(name)), name
    run_rehearsal = "".join(code_cells(RUN_REHEARSAL))
    assert 'shutil.copy(f"{OUT}/run-rehearsal-passed.json", target)' in run_rehearsal
    # its release waits for a rehearsal that still stops, too
    assert 'b"rehearse-run"' in code_cells(RUN_REHEARSAL)[1]
    for name in NAMES + [REHEARSAL]:  # a rehearsal that still stops is waited for too
        assert 'b"snapshot", b"rehearse")' in code_cells(name)[1], name
    rehearsal = "".join(code_cells(REHEARSAL))
    assert "rehearsal-passed.json" in rehearsal and "shutil.copy(" in rehearsal
    pilot = "".join(code_cells(NAMES[0]))
    # a pilot of another commit or model never resumes this one's directory
    assert 'PILOT_DRIVE = f"{DRIVE_ROOT}/pilot/{COMMIT[:12]}-{MODEL.replace(\'/\', \'--\')}"' \
        in pilot


@pytest.mark.parametrize("name", ALL)
def test_the_notebooks_name_one_model_setting(name):
    from evaluation.exp0001.serving import DEFAULT_MODEL
    settings = code_cells(name)[0]
    assert f'MODEL = "{DEFAULT_MODEL}"' in settings
    calls = "".join(code_cells(name)[1:])
    assert f'"{DEFAULT_MODEL}"' not in calls  # every later cell reads MODEL


def notebook_cli_helper():
    """The ``cli`` helper exactly as the install cell defines it."""
    source = next(s for s in code_cells(NAMES[0]) if "def cli(" in s)
    node = next(n for n in ast.parse(source).body
                if isinstance(n, ast.FunctionDef) and n.name == "cli")
    namespace = {"os": os, "signal": signal, "sys": sys, "subprocess": subprocess,
                 "PY": sys.executable,  # the install cell's own names
                 "command_environment": lambda: dict(os.environ)}
    exec(compile(ast.Module([node], []), "<install cell>", "exec"), namespace)
    return namespace["cli"]


def test_the_helper_raises_on_a_failing_command_and_passes_a_succeeding_one(capsys):
    helper = notebook_cli_helper()
    installs = [next(s for s in code_cells(name) if "def cli(" in s) for name in ALL]
    assert len(set(installs)) == 1  # one install cell in every notebook
    helper("--help")  # exit 0
    assert "usage" in capsys.readouterr().out  # the command's output reaches the cell
    with pytest.raises(RuntimeError, match="no-such-command"):
        helper("no-such-command")  # argparse exits 2


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals")
def test_the_stop_button_reaches_the_command_once_and_lets_it_stop_cleanly(
        tmp_path, monkeypatch, capsys):
    stub = tmp_path / "evaluation" / "exp0001"
    stub.mkdir(parents=True)
    (tmp_path / "evaluation" / "__init__.py").write_text("")
    (stub / "__init__.py").write_text("")
    (stub / "cli.py").write_text(textwrap.dedent("""
        import os, signal, sys, time
        interrupts = []
        signal.signal(signal.SIGINT, lambda *_: interrupts.append(1))
        print(f"group {os.getpgid(0)} pid {os.getpid()} "
              f"stop-with-parent {os.environ.get('HERITAGENT_STOP_WITH_PARENT')}", flush=True)
        print("started", flush=True)
        while not interrupts:
            time.sleep(0.01)
        time.sleep(0.3)  # still draining: a second signal would arrive here
        print(f"stopped cleanly after {len(interrupts)} interrupt(s)", flush=True)
    """))
    monkeypatch.chdir(tmp_path)  # -m finds the stub first
    helper = notebook_cli_helper()
    # the kernel's stop is a SIGINT that raises KeyboardInterrupt, even when pytest
    # itself was started with SIGINT ignored
    previous = signal.signal(signal.SIGINT, signal.default_int_handler)
    timer = threading.Timer(1.0, os.kill, (os.getpid(), signal.SIGINT))
    timer.start()
    try:
        with pytest.raises(KeyboardInterrupt):
            helper("run")
    finally:
        timer.cancel()
        signal.signal(signal.SIGINT, previous)
    out = capsys.readouterr().out
    assert "started" in out and "stopped cleanly after 1 interrupt(s)" in out
    group, pid = (int(x) for x in out.split("group ", 1)[1].split()[0:3:2])
    # its own process group: a signal to the kernel's group never reaches it twice
    assert group == pid != os.getpgrp()
    assert f"stop-with-parent {os.getpid()}" in out and "Durduruluyor" in out
    # Stop is a KeyboardInterrupt, which the error handler does not catch: the runtime
    # stays attached, and the helper and the notebooks name the last cell as the next step
    assert "hâlâ bağlı" in out.split("stopped cleanly", 1)[1] and "release()" in out
    for name in NAMES:
        intro = markdown_cells(name)[0]
        assert "Durdur" in intro and "release()" in intro, name


def error_cell(monkeypatch, *, auto_release, drive_root, flush_fails=False, scan_proc=False):
    """The notebook's error-handler cell, run against fake Colab modules."""
    import types
    calls = []
    colab = types.ModuleType("google.colab")
    drive = types.ModuleType("google.colab.drive")
    runtime = types.ModuleType("google.colab.runtime")

    def flush_and_unmount(timeout_ms):
        assert timeout_ms == 600_000  # a stuck Drive never holds the GPU for long
        calls.append("flush")
        if flush_fails:
            raise OSError("Drive is not mounted")

    drive.flush_and_unmount = flush_and_unmount
    runtime.unassign = lambda: calls.append("unassign")
    colab.drive, colab.runtime = drive, runtime
    google = types.ModuleType("google")
    google.colab = colab
    for name, module in (("google", google), ("google.colab", colab),
                         ("google.colab.drive", drive), ("google.colab.runtime", runtime)):
        monkeypatch.setitem(sys.modules, name, module)
    handlers = []
    shell = types.SimpleNamespace(set_custom_exc=lambda types_, handler: handlers.append(handler),
                                  showtraceback=lambda *a, **k: calls.append("traceback"))
    source = next(s for s in code_cells(NAMES[0]) if "set_custom_exc" in s)
    assert source == next(s for s in code_cells(NAMES[1]) if "set_custom_exc" in s)
    namespace = {"get_ipython": lambda: shell, "AUTO_RELEASE": auto_release,
                 "DRIVE_ROOT": str(drive_root)}
    exec(compile(source, "<error cell>", "exec"), namespace)
    if not scan_proc:  # a real pilot elsewhere on this machine must not change the result
        namespace["running_commands"] = lambda: []
    (handler,) = handlers
    return (lambda error: handler(shell, type(error), error, None)), namespace, calls


def test_the_error_handler_releases_the_gpu_even_when_drive_fails(tmp_path, monkeypatch):
    import os
    import shutil
    real_exists = os.path.exists
    monkeypatch.setattr(os.path, "exists", lambda p: p == "/content/vllm-server.log"
                        or real_exists(p))

    def broken_copy(*args):
        raise OSError("Drive quota exceeded")

    monkeypatch.setattr(shutil, "copy", broken_copy)
    fail, _, calls = error_cell(monkeypatch, auto_release=True, drive_root=tmp_path,
                                flush_fails=True)
    fail(RuntimeError("`cli run` failed"))
    assert calls == ["traceback", "flush", "unassign"]  # the copy and the flush failed


def test_the_error_handler_keeps_an_unmirrored_run_and_respects_auto_release(tmp_path,
                                                                            monkeypatch):
    import glob
    fail, namespace, calls = error_cell(monkeypatch, auto_release=False, drive_root=tmp_path)
    fail(RuntimeError("x"))
    assert calls == ["traceback"]
    fail, namespace, calls = error_cell(monkeypatch, auto_release=True, drive_root=tmp_path)
    real_glob = glob.glob
    monkeypatch.setattr(glob, "glob", lambda pattern: ["/content/exp0001-run/UNMIRRORED.json"]
                        if pattern == "/content/*/UNMIRRORED.json" else real_glob(pattern))
    assert namespace["unmirrored"]() == ["/content/exp0001-run"]
    fail(RuntimeError("`cli run`: UNMIRRORED"))
    assert calls == ["traceback"]  # neither flushed nor released: the run lives only here
    namespace["release"]()  # the final cell's path too
    assert calls == ["traceback"]


def test_release_waits_for_a_command_that_still_stops(tmp_path, monkeypatch, capsys):
    """A `run` whose kernel restarted still records its episodes and takes its
    last snapshot: release() keeps the machine until it has ended."""
    import os
    import subprocess
    fail, namespace, calls = error_cell(monkeypatch, auto_release=True, drive_root=tmp_path,
                                        scan_proc=True)
    sleeper = [sys.executable, "-c", "import time; time.sleep(60)", "-m", "evaluation.exp0001.cli"]
    serving = subprocess.Popen(sleeper + ["serve"])  # not a writer: never holds the release
    stopping = subprocess.Popen(sleeper + ["run", "--local-dir", "/content/exp0001-run"])
    try:
        import time
        deadline = time.monotonic() + 10  # a just-started process shows its argv a moment later
        while stopping.pid not in namespace["running_commands"]() and time.monotonic() < deadline:
            time.sleep(0.05)
        found = namespace["running_commands"]()
        assert stopping.pid in found and serving.pid not in found and os.getpid() not in found
        fail(RuntimeError("`cli run` failed (exit code 2)"))  # the refused second run
        namespace["release"]()  # the last cell
        assert calls == ["traceback"]  # neither flushed nor released
        out = capsys.readouterr().out
        assert "GPU BIRAKILMADI" in out and "os.killpg" in out  # and the way out
        # after a kernel restart the install cell is the one that failed
        assert "hata veren hücreyi ve sonrakileri yeniden" in out
        stopping.kill()
        stopping.wait()
        namespace["running_commands"] = lambda: []  # only this test's processes count here
        namespace["release"]()
        assert calls == ["traceback", "flush", "unassign"]
    finally:
        for process in (stopping, serving):
            process.kill()
            process.wait()


@pytest.mark.parametrize("name", ALL)
def test_the_last_cell_releases_through_release(name):
    """The session's end takes the same path as the error handler: an
    unmirrored run keeps the runtime, and Drive is flushed first."""
    last = code_cells(name)[-1]
    assert [line.strip() for line in last.splitlines() if not line.startswith("#")] == \
        ["if AUTO_RELEASE:", "release()"]


def test_the_notebooks_install_one_serving_build_and_never_alfworld_download():
    for name in ALL:
        text = (NOTEBOOKS / name).read_text(encoding="utf-8")
        assert "alfworld-download\"]" not in text
        assert "fetch-data" in text and "verify-data" in text


def install_cell(name, venv, *, running=(), servers=(), repo=NOTEBOOKS.parent, fail=None):
    """The install cell's own functions against a temporary VENV, its subprocess calls
    recorded (the step named ``fail`` fails as a check=True call does)."""
    source = next(s for s in code_cells(name) if "def install():" in s)
    body = [node for node in ast.parse(source).body
            if not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                    and getattr(node.value.func, "id", None) == "install")
            and not (isinstance(node, ast.FunctionDef) and node.name == "cli")]
    namespace = {"REPO": str(repo), "running_commands": lambda: list(running)}
    exec(compile(ast.Module(body, []), "<install cell>", "exec"), namespace)
    calls = []

    def run(argv, **kwargs):
        argv = [str(arg) for arg in argv]
        calls.append((argv, kwargs.get("env")))
        if fail is not None and steps([(argv, None)]) == [fail]:
            raise subprocess.CalledProcessError(1, argv)
        if "venv" in argv:  # what uv makes
            (venv / "bin").mkdir(parents=True, exist_ok=True)
            (venv / "bin" / "python").write_text("")
            (venv / "pyvenv.cfg").write_text(f"home = /x\nversion_info = {argv[argv.index('--python') + 1]}\n")
        return SimpleNamespace(returncode=0)

    stopped = []

    def stop(pids):
        stopped.extend(pids)
        calls.append((["stop", *map(str, pids)], None))

    namespace.update(subprocess=SimpleNamespace(run=run), VENV=str(venv), PY=f"{venv}/bin/python",
                     vllm_servers=lambda: list(servers), stop_servers=stop)
    namespace["stopped"] = stopped
    return namespace, calls


def steps(calls):
    def kind(argv):
        if argv[0] == "stop":
            return "stop"
        if any(arg.endswith("/uv-requirements.txt") for arg in argv):
            return "uv"
        for word, step in (("venv", "venv"), ("sync", "sync"), ("-e", "editable"),
                           ("check", "check")):
            if word in argv:
                return step
        return " ".join(argv)
    return [kind(argv) for argv, _ in calls]


@pytest.mark.parametrize("name", ALL)
def test_the_install_cell_makes_the_environment_once_and_never_under_a_running_command(
        name, tmp_path):
    venv = tmp_path / "venv"
    cell, calls = install_cell(name, venv)
    cell["install"]()  # a fresh runtime
    assert steps(calls) == ["uv", "venv", "sync", "editable", "check"]
    assert all(env["MPLBACKEND"] == "Agg" and "PYTHONPATH" not in env
               for argv, env in calls[1:])  # the builds see none of the kernel's settings
    stamp = (venv / "heritagent-environment.sha256").read_text()
    calls.clear()
    cell["install"]()  # a kernel restart: the environment stays
    assert steps(calls) == ["uv", "sync", "editable", "check"]
    (venv / "pyvenv.cfg").write_text("version_info = 3.12.0\n")
    calls.clear()
    cell["install"]()  # another Python: made again
    assert steps(calls) == ["uv", "venv", "sync", "editable", "check"]
    assert (venv / "heritagent-environment.sha256").read_text() == stamp
    # a command still running: nothing is touched
    cell, calls = install_cell(name, venv, running=[123])
    with pytest.raises(RuntimeError, match="123"):
        cell["install"]()
    assert calls == []
    # a server from the same environment is reused (a kernel restart in a run)
    cell, calls = install_cell(name, venv, servers=[456])
    cell["install"]()
    assert steps(calls) == ["uv", "sync", "editable", "check"] and cell["stopped"] == []
    # one from an older environment, or from one whose install never finished, is stopped first
    for stamp_text in ("an older environment\n", None):
        stamp_file = venv / "heritagent-environment.sha256"
        if stamp_text is None:
            stamp_file.unlink()
        else:
            stamp_file.write_text(stamp_text)
        cell, calls = install_cell(name, venv, servers=[456])
        cell["install"]()
        assert cell["stopped"] == [456]  # before the environment changes
        assert steps(calls) == ["stop", "uv", "sync", "editable", "check"]
        assert stamp_file.read_text() == stamp


@pytest.mark.parametrize("name", [NAMES[1], RUN_REHEARSAL])
def test_the_install_stamp_covers_every_environment_file_and_follows_a_whole_install(
        name, tmp_path):
    from evaluation.exp0001 import environment
    repo = tmp_path / "repo"
    env = repo / environment.ENVIRONMENT_DIR
    env.mkdir(parents=True)
    files = (environment.PYTHON_VERSION_FILE, environment.UV_REQUIREMENTS_FILE,
             environment.LOCK_FILE, environment.BUILD_CONSTRAINTS_FILE)
    for file in files:
        (env / file).write_text((NOTEBOOKS.parent / environment.ENVIRONMENT_DIR / file)
                                .read_text(encoding="utf-8"), encoding="utf-8")
    stamps = set()
    for changed in (None, *files):
        if changed is not None:
            (env / changed).write_text((env / changed).read_text() + "\n", encoding="utf-8")
        venv = tmp_path / f"venv-{changed}"
        cell, _ = install_cell(name, venv, repo=repo)
        cell["install"]()
        stamps.add((venv / "heritagent-environment.sha256").read_text())
    assert len(stamps) == 1 + len(files)  # each file changes the stamp
    for fail in ("venv", "sync", "editable", "check"):  # a failed step leaves no stamp
        venv = tmp_path / f"failed-{fail}"
        cell, _ = install_cell(name, venv, repo=repo, fail=fail)
        with pytest.raises(subprocess.CalledProcessError):
            cell["install"]()
        assert not (venv / "heritagent-environment.sha256").exists()


def server_functions(tmp_path):
    """The install cell's own server scan and stop, with short waits and a temporary record."""
    source = next(s for s in code_cells(NAMES[1]) if "def install():" in s)
    names = ("vllm_servers", "stop_servers", "SERVER_STOP_SECONDS", "SERVER_INFO")
    body = [node for node in ast.parse(source).body
            if (isinstance(node, ast.FunctionDef) and node.name in names)
            or (isinstance(node, ast.Assign) and node.targets[0].id in names)]
    namespace = {"os": os, "signal": signal, "time": time}
    exec(compile(ast.Module(body, []), "<install cell>", "exec"), namespace)
    namespace.update(SERVER_STOP_SECONDS=(2, 5), SERVER_INFO=str(tmp_path / "vllm-server.json"))
    return namespace


def fake_server(tmp_path, *, ignore_term):
    """A `vllm serve` as the environment's console script runs it, in its own session."""
    script = tmp_path / "bin" / "vllm"
    script.parent.mkdir(exist_ok=True)
    ready = tmp_path / "ready"
    script.write_text(f"#!{sys.executable}\nimport pathlib, signal, time\n"
                      + ("signal.signal(signal.SIGTERM, signal.SIG_IGN)\n" if ignore_term else "")
                      + f"pathlib.Path({str(ready)!r}).write_text('')\ntime.sleep(60)\n")
    script.chmod(0o755)
    server = subprocess.Popen([str(script), "serve", "/content/model"], start_new_session=True)
    return server


@pytest.mark.skipif(not os.path.isdir("/proc"), reason="the scan reads /proc")
@pytest.mark.parametrize("ignore_term", [False, True])
def test_the_install_cell_stops_an_older_environments_server(tmp_path, ignore_term):
    cell = server_functions(tmp_path)
    (tmp_path / "vllm-server.json").write_text("{}")
    server = fake_server(tmp_path, ignore_term=ignore_term)
    try:
        deadline = time.monotonic() + 10
        while not (tmp_path / "ready").exists() and time.monotonic() < deadline:
            time.sleep(0.05)  # its signal handling is in place
        assert server.pid in cell["vllm_servers"]()
        cell["stop_servers"]([server.pid])  # SIGTERM, then SIGKILL to its group
        assert server.pid not in cell["vllm_servers"]()
        # the signal that ended it: SIGTERM, or SIGKILL for one that ignores SIGTERM
        assert server.wait(timeout=5) == -(signal.SIGKILL if ignore_term else signal.SIGTERM)
        assert not (tmp_path / "vllm-server.json").exists()  # `serve` will not wait on it
    finally:
        server.kill()
        server.wait()
    # one that never leaves the scan ends the cell (a pid no process can have)
    with open("/proc/sys/kernel/pid_max", encoding="utf-8") as handle:
        free = int(handle.read()) + 1
    cell = server_functions(tmp_path)
    cell.update(vllm_servers=lambda: [free], SERVER_STOP_SECONDS=(0.1, 0.1))
    with pytest.raises(RuntimeError, match="bırakılacak"):
        cell["stop_servers"]([free])


@pytest.mark.skipif(not os.path.isdir("/proc"), reason="the scan reads /proc")
def test_the_install_cell_finds_a_running_vllm_server(tmp_path):
    source = next(s for s in code_cells(NAMES[1]) if "def install():" in s)
    scan = next(node for node in ast.parse(source).body
                if isinstance(node, ast.FunctionDef) and node.name == "vllm_servers")
    namespace = {"os": os}
    exec(compile(ast.Module([scan], []), "<install cell>", "exec"), namespace)
    script = tmp_path / "bin" / "vllm"  # the environment's console script, as `vllm serve` runs it
    script.parent.mkdir()
    script.write_text(f"#!{sys.executable}\nimport time\ntime.sleep(60)\n")
    script.chmod(0o755)
    server = subprocess.Popen([str(script), "serve", "/content/model"])
    other = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)", "vllm", "serve"])
    try:
        deadline = time.monotonic() + 10
        while server.pid not in namespace["vllm_servers"]() and time.monotonic() < deadline:
            time.sleep(0.05)
        found = namespace["vllm_servers"]()
        assert server.pid in found and other.pid not in found
    finally:
        server.kill()
        other.kill()


@pytest.mark.parametrize("name", ALL)
def test_the_notebooks_install_the_locked_environment_and_check_it_first(name):
    """Owner decision 2026-10-19: Colab's own Python and packages play no
    part. The install cell sets up environments/exp0001-colab/ (the pinned
    uv and Python, the hash-locked packages) in a virtual environment, every
    command runs from it, and the next cell holds it to the lock before the
    data, the model or any command that records."""
    from evaluation.exp0001 import environment
    from evaluation.exp0001.serving import SOFTWARE_PACKAGES
    cells = code_cells(name)
    install = first(cells, "def install():")
    source = cells[install]
    tree = ast.parse(source)
    # the server record the install removes is the one `serve` writes and the run reads
    if name in NAMES:
        info = next(node.value.value for node in tree.body if isinstance(node, ast.Assign)
                    and node.targets[0].id == "SERVER_INFO")
        serve = argv(cli_calls(cells[first(cells, 'cli("serve"')])[0], {})
        runs = argv(cli_calls(cells[first(cells, 'cli("run"' if "run" in name else
                                          'cli("pilot"')])[0], {})
        assert serve[serve.index("--info") + 1] == runs[runs.index("--server-info") + 1] == info
    # the install is the cell's last statement, after the helper: `cli` exists when it stops
    last = tree.body[-1]
    assert isinstance(last, ast.Expr) and getattr(last.value.func, "id", None) == "install"
    assert any(isinstance(node, ast.FunctionDef) and node.name == "cli" for node in tree.body)
    for needed in (f'ENV = f"{{REPO}}/{environment.ENVIRONMENT_DIR.as_posix()}"',
                   f'"--require-hashes",\n                    "--only-binary", ":all:", "-r", '
                   f'f"{{ENV}}/{environment.UV_REQUIREMENTS_FILE}"',
                   'f"{ENV}/python-version"', 'f"{VENV}/pyvenv.cfg"',
                   'made.get("version_info") != python or not os.path.exists(PY)',
                   "busy = running_commands()",
                   '["venv", "--clear", "--managed-python", "--python", python, VENV]',
                   '["pip", "sync", "--python", PY, "--require-hashes", "--index-strategy",\n'
                   '                         "unsafe-best-match", *build, '
                   f'f"{{ENV}}/{environment.LOCK_FILE}"]',
                   f'build = ["--build-constraints", f"{{ENV}}/{environment.BUILD_CONSTRAINTS_FILE}"]',
                   '["pip", "install", "--python", PY, "--no-deps", *build, "-e", REPO]',
                   '["pip", "check", "--python", PY]',
                   'command = [PY, "-u", "-m", "evaluation.exp0001.cli", *map(str, args)]',
                   "env={**command_environment(),"):
        assert needed in source, needed
    assert source.index('"--clear"') > source.index("busy = running_commands()")
    assert "--seed" not in source and "--system-site-packages" not in source
    for package in SOFTWARE_PACKAGES:  # no version outside the environment files
        assert f'"{package}==' not in source and f"{package}[" not in source, package
    # the commands' environment: the venv first on the PATH (vllm serve), none of the
    # kernel's Python or matplotlib settings (MPLBACKEND names a backend the venv lacks)
    namespace = {"os": os, "VENV": "/content/heritagent-venv"}
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == "command_environment")
    exec(compile(ast.Module([function], []), "<install cell>", "exec"), namespace)
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("PYTHONPATH", "/env/python")
        mp.setenv("MPLBACKEND", "module://matplotlib_inline.backend_inline")
        mp.setenv("PATH", "/usr/bin")
        env = namespace["command_environment"]()
    assert env["PATH"] == "/content/heritagent-venv/bin:/usr/bin"
    assert env["VIRTUAL_ENV"] == "/content/heritagent-venv" and env["MPLBACKEND"] == "Agg"
    assert "PYTHONPATH" not in env and env["PYTHONNOUSERSITE"] == "1"
    check = cells[install + 1]
    assert [argv(call, {}) for call in cli_calls(check)] == [["env-check", "--repo", "1"]]
    assert check.index('cli("env-check"') < check.index("subprocess.run([PY,")
    assert "sys.executable" not in check and "try:" not in check and "except" not in check
    probes = check.count("subprocess.run([PY,")
    assert probes and check.count("env=command_environment())") == probes
    assert check.count("matplotlib") >= 2  # what the figures load, under the same environment
    for later in ('"verify-data"', '"prepare-model"', 'cli("rehearse', 'cli("run"', 'cli("pilot"'):
        if later in "".join(cells):
            assert first(cells, later) > install + 1, later


FIGURES = "EXP-0001-figures.ipynb"  # the CPU notebook that redraws a run's figures


def test_the_run_notebook_draws_its_figures_after_the_report_and_before_the_release():
    cells = code_cells(NAMES[1])
    report, figures = first(cells, '"/content/exp0001-run/report.json"'), first(cells, '"figures"')
    assert report < figures == len(cells) - 2  # the release cell comes last
    source = cells[figures]
    call = cli_calls(source)[0]
    args = cli.parser().parse_args(argv(call, {}))
    assert args.run_dir == Path("/content/exp0001-run") and args.mirror_dir is None
    # outside the run directory (a snapshot would sweep it in) and the checkout
    assert 'FIGURES = "/content/exp0001-figures"' in source
    assert 'shutil.copytree(FIGURES, f"{RUN_DRIVE}/figures")' in source


def test_the_figures_notebook_reads_the_mirror_on_a_cpu_runtime():
    cells = code_cells(FIGURES)
    commands = set()
    for source in cells:
        assert not [line for line in source.splitlines() if line.lstrip().startswith(("!", "%"))]
        for call in cli_calls(source):
            args = cli.parser().parse_args(argv(call, {}))
            commands.add(args.command)
            assert args.mirror_dir is not None and args.run_dir is None  # read only, from Drive
    assert commands == {"figures"}
    text = "".join(cells)
    # no GPU asked for and no serving stack installed (the shared handler only names its log)
    assert "nvidia-smi" not in text and "cuda.is_available" not in text
    assert "vllm==" not in text and "torch==" not in text and "[alfworld]" not in text
    # the same error handler, Drive and checkout cells as the CPU rehearsal, handler first
    rehearsal = code_cells(REHEARSAL)
    handler = first(cells, "set_custom_exc")
    assert cells[handler] == next(s for s in rehearsal if "set_custom_exc" in s)
    assert handler < first(cells, "# 2) Google Drive") < first(cells, 'cli("figures"')
    assert cells[first(cells, "def checkout(ref):")] == \
        next(s for s in rehearsal if "def checkout(ref):" in s)
    def helper_source(source):
        node = next(n for n in ast.parse(source).body
                    if isinstance(n, ast.FunctionDef) and n.name == "cli")
        return ast.get_source_segment(source, node)

    mine = cells[first(cells, "def cli(")]
    assert helper_source(mine) == helper_source(next(s for s in rehearsal if "def cli(" in s))
    # the helper's names, defined for Colab's own Python: the kernel's environment as it is
    namespace = {"os": os, "sys": sys}
    tree = ast.parse(mine)
    exec(compile(ast.Module([node for node in tree.body if isinstance(node, ast.Assign) or (
        isinstance(node, ast.FunctionDef) and node.name == "command_environment")], []),
        "<install cell>", "exec"), namespace)
    assert namespace["PY"] == sys.executable
    assert namespace["command_environment"]() == dict(os.environ)
    assert [line.strip() for line in cells[-1].splitlines() if not line.startswith("#")] == \
        ["if AUTO_RELEASE:", "release()"]
    notebook = json.loads((NOTEBOOKS / FIGURES).read_text(encoding="utf-8"))
    assert notebook["metadata"]["accelerator"] == "None"


def markdown_cells(name):
    notebook = json.loads((NOTEBOOKS / name).read_text(encoding="utf-8"))
    return ["".join(cell["source"]) for cell in notebook["cells"]
            if cell["cell_type"] == "markdown"]


@pytest.mark.parametrize("name", ALL + [FIGURES])
def test_a_notebook_is_committed_without_outputs_and_on_its_own_runtime_type(name):
    """A notebook saved from Colab keeps no output (Drive paths, model text) and
    asks for the runtime it is written for: an A100 for the pilot and the run
    (ADR-0008 D1), no GPU for the CPU notebooks."""
    notebook = json.loads((NOTEBOOKS / name).read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["outputs"] == [] and cell["execution_count"] is None, cell["source"][:1]
    metadata = notebook["metadata"]
    if name in NAMES:
        assert metadata["accelerator"] == "GPU" and metadata["colab"]["gpuType"] == "A100"
    else:
        assert metadata["accelerator"] != "GPU" and "gpuType" not in metadata.get("colab", {})


@pytest.mark.parametrize("name", NAMES)
def test_a_finished_run_is_found_where_its_report_is_copied_before_any_spend(name):
    """The 'already finished' check reads the path the report cell writes, and
    it stops the notebook before the install, the model and any episode."""
    cells = code_cells(name)
    done = first(cells, "DONE = ")
    tree = ast.parse(cells[done])
    (value,) = [node.value for node in tree.body if isinstance(node, ast.Assign)
                and getattr(node.targets[0], "id", None) == "DONE"]
    local = "/content/exp0001-run/report.json" if "run" in name else \
        "/content/pilot/pilot-report.json"
    report = first(cells, f'shutil.copy("{local}"')
    targets = [call.args[1] for call in ast.walk(ast.parse(cells[report]))
               if isinstance(call, ast.Call) and ast.unparse(call.func) == "shutil.copy"
               and getattr(call.args[0], "value", None) == local]
    assert any(getattr(target, "id", None) == "DONE" or ast.dump(target) == ast.dump(value)
               for target in targets), [ast.unparse(target) for target in targets]
    bitti = first(cells, "raise Bitti(")
    assert "if os.path.exists(DONE):" in cells[bitti]
    spend = 'cli("run"' if "run" in name else 'cli("pilot"'
    assert bitti < first(cells, "def install():") < first(cells, '"prepare-model"') < \
        first(cells, spend)


def test_the_run_notebook_takes_its_cost_and_its_cap_from_adr_0012():
    """The cost estimate is ADR-0012's (D9 and its experiment-level table), not
    the pilot report's projection, which predates ADR-0012's gate and caps."""
    intro = markdown_cells(NAMES[1])[0]
    (cap,) = [line for line in intro.splitlines() if line.startswith("3. ")]
    assert "ADR-0012 D9" in cap
    cost = intro.split("**Tahmini maliyet", 1)[1]
    assert "pilot projeksiyonuna bakın" not in cost and "5000 bölüm" not in cost
    for needed in ("ADR-0012", "345 bölüm (R)", "134–160 CU", "315 CU", "(M)", "~30–50 dk"):
        assert needed in cost, needed


def test_the_run_notebook_says_where_the_model_revision_comes_from():
    """A resumed run's commit comes from its record on Drive; the revision comes
    from the revision file, which the pilot or a rehearsal writes."""
    intro = markdown_cells(NAMES[1])[0]
    assert "model revizyonuyla devam eder; bunlar Drive'daki kayıttan okunur" not in intro
    assert "model-revision-" in intro
    cells = code_cells(NAMES[1])
    revision = cells[first(cells, "revision_file = ")]
    assert "Önce pilot notebook'unu çalıştırın" not in revision
    assert "pilot ya da prova" in revision
    prepare = cells[first(cells, 'cli("prepare-model"')]
    assert "pilotun sabitlediği" not in prepare and "pilot ya da prova" in prepare


@pytest.mark.parametrize("name", CPU)
def test_the_cpu_rehearsals_stop_when_the_locked_grammar_backend_does_not_import(name):
    """The rehearsals' stand-in compiles the teacher schema with xgrammar; when
    the lock pins it, the environment check imports it before the rehearsal."""
    from evaluation.exp0001 import environment
    lock = (NOTEBOOKS.parent / environment.ENVIRONMENT_DIR / environment.LOCK_FILE).read_text(
        encoding="utf-8")
    if not any(line.startswith("xgrammar==") for line in lock.splitlines()):
        pytest.skip("the lock does not pin xgrammar")
    cells = code_cells(name)
    check = cells[first(cells, 'cli("env-check"')]
    probe = check.split('cli("env-check"', 1)[1].split("assert check.returncode == 0", 1)[0]
    assert "import torch, transformers, tokenizers, matplotlib, xgrammar;" in probe
    assert first(cells, 'cli("rehearse') > first(cells, 'cli("env-check"')


@pytest.mark.parametrize("name", NAMES)
def test_the_gpu_cell_asks_for_an_a100_on_a_runtime_without_nvidia_smi(name, tmp_path,
                                                                       monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path))  # an empty directory: no nvidia-smi
    cells = code_cells(name)
    with pytest.raises(AssertionError, match="A100 GPU seçin"):
        exec(compile(cells[first(cells, "nvidia-smi")], "<gpu cell>", "exec"), {})


@pytest.mark.parametrize("name", ALL)
def test_the_drive_check_asks_for_the_space_the_notebook_writes(name):
    """The run's mirror needs the default bound; the pilot and the CPU
    rehearsals write far less and ask for 1 GB. The cell says that the free
    space is reported and comes from the mount."""
    cells = code_cells(name)
    cell = cells[first(cells, 'cli("mirror-check"')]
    (call,) = cli_calls(cell)
    need = cli.parser().parse_args(argv(call, {})).need_gb
    assert need == (cli.RUN_DRIVE_GB if name == NAMES[1] else 1.0)
    assert "boş alan" in cell and "drive.google.com/settings/storage" in cell


def test_the_run_notebook_stops_on_code_older_than_the_mirror_identity(tmp_path):
    """Code without the mirror identity reads a mirror of the compressed
    layout as empty: its run-commit prints an empty line and its run starts
    a new run there. The run notebook stops on such a CODE_REF before
    run-commit runs."""
    cells = code_cells(NAMES[1])
    cell = cells[first(cells, '"run-commit"')]
    mirror = tmp_path / "evaluation" / "exp0001" / "mirror.py"
    mirror.parent.mkdir(parents=True)
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        return SimpleNamespace(returncode=0, stdout="\n", stderr="")

    def namespace():
        return {"REPO": str(tmp_path), "RUN_DRIVE": str(tmp_path / "drive"), "COMMIT": "c" * 40,
                "subprocess": SimpleNamespace(run=run), "checkout": lambda ref: ref}

    mirror.write_text('MIRROR_SCHEMA = "heritagent/exp0001-mirror/0.1"\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="CODE_REF'e daha yeni bir commit"):
        exec(compile(cell, "<run-commit cell>", "exec"), namespace())
    assert calls == []  # neither the install nor run-commit ran
    mirror.write_text((NOTEBOOKS.parent / "evaluation" / "exp0001" / "mirror.py").read_text(
        encoding="utf-8"), encoding="utf-8")
    exec(compile(cell, "<run-commit cell>", "exec"), namespace())
    assert any("run-commit" in argv for argv in calls)


def test_the_run_notebook_stops_on_code_without_the_mirror_module(tmp_path):
    """A CODE_REF older than the mirror module gets the same Turkish
    message, not a traceback, before anything is installed."""
    cells = code_cells(NAMES[1])
    cell = cells[first(cells, '"run-commit"')]
    calls = []
    namespace = {"REPO": str(tmp_path), "RUN_DRIVE": str(tmp_path / "drive"), "COMMIT": "c" * 40,
                 "subprocess": SimpleNamespace(run=lambda argv, **kwargs: calls.append(argv)),
                 "checkout": lambda ref: ref}
    with pytest.raises(RuntimeError, match="CODE_REF'e daha yeni bir commit"):
        exec(compile(cell, "<run-commit cell>", "exec"), namespace)
    assert calls == []


def test_the_run_rehearsal_notebook_says_which_gate_failures_decide_nothing_there():
    """The scripted world reaches the out-of-scope, 4A and 4B failures only on
    candidates that also fail stage 2 (run_rehearsal's docstring)."""
    intro = "".join(markdown_cells(RUN_REHEARSAL))
    assert "kapsam dışı ve 4A'da kalan, 4B'de kalan aday" in intro
    assert "Kapsam dışı, 4A ve 4B kalışları yalnızca 2. aşamada da kalan adaylarda görülür" \
        in intro and "birim testlerinde" in intro


def run_settings(name, **values):
    """The settings cell's own code with some of its values replaced, as the
    owner would type them."""
    tree = ast.parse(code_cells(name)[0])
    for node in tree.body:
        if isinstance(node, ast.Assign) and node.targets[0].id in values:
            node.value = ast.copy_location(ast.Constant(values[node.targets[0].id]), node.value)
    namespace = {}
    exec(compile(tree, "<settings cell>", "exec"), namespace)
    return namespace


def run_cell_argv(hours):
    cells = code_cells(NAMES[1])
    return argv(cli_calls(cells[first(cells, 'cli("run"')])[0], {"STOP_AFTER_HOURS": hours})


@pytest.mark.parametrize("value", [0, 0.0, -1, -0.5, float("inf"), float("-inf"), float("nan"),
                                   True, "6"])
def test_the_run_notebook_refuses_a_wrong_time_limit_in_its_settings_cell(value):
    """`run --stop-after-hours` refuses these only in the run cell, after the
    install, the data, the model and the server (30-50 minutes on the A100)."""
    with pytest.raises(ValueError, match="STOP_AFTER_HOURS ya None"):
        run_settings(NAMES[1], STOP_AFTER_HOURS=value)
    if not isinstance(value, str):  # what the run cell would have sent
        with pytest.raises(SystemExit):
            cli.parser().parse_args(run_cell_argv(value))


@pytest.mark.parametrize("name", [NAMES[1], RUN_REHEARSAL])
@pytest.mark.parametrize("value", [1, 0, -2, 2.0, 16.0, True, "16", None])
def test_the_run_notebooks_refuse_fewer_than_two_workers_in_their_settings_cell(name, value):
    """`run` and `rehearse-run` refuse fewer than two workers (each opened
    game keeps memory until its worker is replaced) only after the install
    and the data; a value argparse cannot read as a whole number fails there
    too."""
    with pytest.raises(ValueError, match="WORKERS en az 2 olan bir tam sayı"):
        run_settings(name, WORKERS=value)


def test_the_run_notebooks_settings_cells_pass_the_values_run_takes():
    for hours in (None, 6.0, 0.25, 1, 96):
        assert run_settings(NAMES[1], STOP_AFTER_HOURS=hours)["STOP_AFTER_HOURS"] == hours
        args = cli.parser().parse_args(run_cell_argv(hours))
        assert args.stop_after_hours == (None if hours is None else float(hours))
    for name in (NAMES[1], RUN_REHEARSAL):
        for workers in (2, 4, 16):
            assert run_settings(name, WORKERS=workers)["WORKERS"] == workers
        run_settings(name)  # the committed values


def test_the_run_guard_names_older_rehearsed_commits_only_with_the_critics_clearance(tmp_path):
    """An older commit can carry an older protocol (the one before ADR-0011
    note N1 asked the teacher for 10 proposals): the guard keeps listing the
    rehearsed commits but sends the owner to the Critic's clearance on issue
    #13 or to a run rehearsal of the new commit."""
    from evaluation.exp0001.serving import DEFAULT_MODEL
    cells = code_cells(NAMES[1])
    cell = cells[first(cells, "if not os.path.exists(PASSED):")]
    rehearsal = tmp_path / "rehearsal"
    rehearsal.mkdir()
    model = DEFAULT_MODEL.replace("/", "--")

    def guard():
        exec(compile(cell, "<guard cell>", "exec"),
             {"os": os, "DRIVE_ROOT": str(tmp_path), "COMMIT": "a" * 40, "MODEL": DEFAULT_MODEL,
              "run_commit": ""})

    with pytest.raises(RuntimeError) as none:
        guard()
    assert "koşu provası geçmiş kodlar" not in str(none.value)
    for older in ("b" * 40, "c" * 40):
        (rehearsal / f"run-passed-{older}-{model}.json").write_text("{}")
    with pytest.raises(RuntimeError) as older:
        guard()
    message = str(older.value)
    assert f"koşu provası geçmiş kodlar: {'b' * 40}, {'c' * 40}." in message
    assert "issue #13'teki Critic onayı o commit'i adıyla anıyorsa" in message
    assert "yoksa bu commit için koşu provasını çalıştırın" in message
    assert "çalıştırmak için CODE_REF'e birini yazın" not in message
    (rehearsal / f"run-passed-{'a' * 40}-{model}.json").write_text("{}")
    guard()  # this commit's record: the guard passes


def test_the_run_notebook_says_how_mining_and_a_failing_mirror_meet_a_stop():
    """The time limit admits no new episode, and the miner's teacher calls are
    not episodes: the limit waits for that seed's mining, and a Stop there
    repeats it next session (a crash before the frozen set re-mines). A mirror
    that keeps failing pauses the run with its newest records only on the
    runtime."""
    intro = markdown_cells(NAMES[1])[0]
    bullets = {line.split(":", 1)[0]: line for line in intro.splitlines() if line.startswith("- ")}
    assert "süre dolarsa koşu, o tohumun miner adımı bitene dek sürer" in \
        bullets["- `STOP_AFTER_HOURS`"]
    assert "Miner adımında basılan Durdur, o tohumun miner adımını sonraki oturumda baştan " \
        "yineletir" in next(line for key, line in bullets.items() if "**Durdur**" in key)
    # the pause's threshold is the mirror's own constant (job m builds it)
    failures = cli.MIRROR_FAILURES_BEFORE_PAUSE
    assert f"Drive'a kopya üst üste {failures} kez alınamazsa koşu yeni bölüm başlatmaz" in intro
    for needed in ("en yeni kayıtlar o zaman yalnızca bu makinededir",
                   "çalışma zamanını bırakmayın", "force_remount=True",
                   "koşu hücresini (11) yeniden çalıştırın"):
        assert needed in intro, needed
    cells = code_cells(NAMES[1])
    assert cells[first(cells, 'cli("run"')].startswith("# 11)")
