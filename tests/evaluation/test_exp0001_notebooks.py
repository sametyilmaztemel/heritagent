"""The Colab notebooks (issue #13 Phase B; ADR-0008) call the command line
with flags it has: every ``evaluation.exp0001.cli`` line of every code cell
parses."""

import json
import re
import shlex
from pathlib import Path

import pytest

from evaluation.exp0001 import cli

NOTEBOOKS = Path(__file__).resolve().parents[2] / "notebooks"
EMPTY = {"flag", "stop", "bi_flag"}  # optional flags the cells may leave empty


def cli_lines(path):
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        for line in "".join(cell["source"]).splitlines():
            if line.startswith("!python -m evaluation.exp0001.cli"):
                yield line


@pytest.mark.parametrize("name", ["EXP-0001-pilot.ipynb", "EXP-0001-run.ipynb"])
def test_every_command_line_of_the_notebook_parses(name):
    lines = list(cli_lines(NOTEBOOKS / name))
    assert lines
    commands = set()
    for line in lines:
        filled = re.sub(r"\{(\w+)\}", lambda m: "" if m.group(1) in EMPTY else "1", line)
        argv = shlex.split(filled)[3:]
        args = cli.parser().parse_args(argv)  # an unknown flag exits
        commands.add(args.command)
    expected = {"gpu-check", "prepare-model", "serve", "preflight", "mirror-check",
                "fetch-data", "verify-data"} | ({"pilot"} if "pilot" in name else {"run", "status"})
    assert commands == expected


def test_the_notebooks_install_one_serving_build_and_never_alfworld_download():
    for name in ("EXP-0001-pilot.ipynb", "EXP-0001-run.ipynb"):
        text = (NOTEBOOKS / name).read_text(encoding="utf-8")
        assert "vllm==0.29.0+cu129" in text and "alfworld-download\"]" not in text
        assert "fetch-data" in text and "verify-data" in text

