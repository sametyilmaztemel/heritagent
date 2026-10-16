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
from pathlib import Path

import pytest

from evaluation.exp0001 import cli

NOTEBOOKS = Path(__file__).resolve().parents[2] / "notebooks"
NAMES = ["EXP-0001-pilot.ipynb", "EXP-0001-run.ipynb"]
# the optional flags' two branches: every other name reads as "1"
BRANCHES = [{}, {"BATCH_INVARIANT": False, "STOP_AFTER_HOURS": None, "revision": None}]


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


@pytest.mark.parametrize("name", NAMES)
def test_every_command_of_the_notebook_parses(name):
    commands = set()
    for source in code_cells(name):
        for call in cli_calls(source):
            for branch in BRANCHES:
                args = cli.parser().parse_args(argv(call, branch))  # an unknown flag exits
                commands.add(args.command)
    expected = {"gpu-check", "prepare-model", "serve", "preflight", "mirror-check",
                "fetch-data", "verify-data"} | ({"pilot"} if "pilot" in name else {"run", "status"})
    assert commands == expected


@pytest.mark.parametrize("name", NAMES)
def test_no_cell_uses_a_shell_escape_and_the_error_handler_comes_first(name):
    cells = code_cells(name)
    for source in cells:
        assert not [line for line in source.splitlines() if line.lstrip().startswith(("!", "%"))]
    handler = next(i for i, source in enumerate(cells) if "set_custom_exc" in source)
    first_command = next(i for i, source in enumerate(cells) if cli_calls(source))
    gpu = next(i for i, source in enumerate(cells) if "nvidia-smi" in source)
    assert handler < gpu < first_command
    assert "runtime.unassign()" in cells[handler]


def notebook_cli_helper():
    """The ``cli`` helper exactly as the install cell defines it."""
    source = next(s for s in code_cells(NAMES[0]) if "def cli(" in s)
    node = next(n for n in ast.parse(source).body
                if isinstance(n, ast.FunctionDef) and n.name == "cli")
    namespace = {"os": os, "signal": signal, "sys": sys, "subprocess": subprocess}
    exec(compile(ast.Module([node], []), "<install cell>", "exec"), namespace)
    return namespace["cli"]


def test_the_helper_raises_on_a_failing_command_and_passes_a_succeeding_one(capsys):
    helper = notebook_cli_helper()
    installs = [next(s for s in code_cells(name) if "def cli(" in s) for name in NAMES]
    assert installs[0] == installs[1]  # one install cell, the same in both notebooks
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


@pytest.mark.parametrize("name", NAMES)
def test_the_last_cell_releases_through_release(name):
    """The session's end takes the same path as the error handler: an
    unmirrored run keeps the runtime, and Drive is flushed first."""
    last = code_cells(name)[-1]
    assert [line.strip() for line in last.splitlines() if not line.startswith("#")] == \
        ["if AUTO_RELEASE:", "release()"]


def test_the_notebooks_install_one_serving_build_and_never_alfworld_download():
    from evaluation.exp0001.serving import SOFTWARE_PACKAGES
    pyproject = (NOTEBOOKS.parent / "pyproject.toml").read_text(encoding="utf-8")
    for name in NAMES:
        text = (NOTEBOOKS / name).read_text(encoding="utf-8")
        assert "vllm==0.29.0+cu129" in text and "alfworld-download\"]" not in text
        assert "fetch-data" in text and "verify-data" in text
        for package in SOFTWARE_PACKAGES:  # every recorded package has one version
            assert f'{package}==' in text or f'"{package}==' in pyproject, package
