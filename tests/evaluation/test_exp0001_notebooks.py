"""The Colab notebooks (issue #13 Phase B; ADR-0008) call the command line
with flags it has, and a failing command stops "Run all": every
``cli(...)`` call of every code cell parses, no cell uses a ``!`` shell
escape (Colab carries on after one fails), the helper raises on a non-zero
exit, and the error handler is installed before the first command."""

import ast
import json
import subprocess
import sys
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
    namespace = {"sys": sys, "subprocess": subprocess}
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


def test_the_notebooks_install_one_serving_build_and_never_alfworld_download():
    from evaluation.exp0001.serving import SOFTWARE_PACKAGES
    pyproject = (NOTEBOOKS.parent / "pyproject.toml").read_text(encoding="utf-8")
    for name in NAMES:
        text = (NOTEBOOKS / name).read_text(encoding="utf-8")
        assert "vllm==0.29.0+cu129" in text and "alfworld-download\"]" not in text
        assert "fetch-data" in text and "verify-data" in text
        for package in SOFTWARE_PACKAGES:  # every recorded package has one version
            assert f'{package}==' in text or f'"{package}==' in pyproject, package
