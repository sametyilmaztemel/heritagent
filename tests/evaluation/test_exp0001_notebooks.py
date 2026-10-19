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
    namespace.update(subprocess=SimpleNamespace(run=run), VENV=str(venv), PY=f"{venv}/bin/python",
                     vllm_servers=lambda: list(servers), stop_servers=stopped.extend)
    namespace["stopped"] = stopped
    return namespace, calls


def steps(calls):
    def kind(argv):
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
        assert cell["stopped"] == [456] and steps(calls)[0] == "uv"
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
    assert any(isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
               and getattr(node.value.func, "id", None) == "install" for node in tree.body)
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
