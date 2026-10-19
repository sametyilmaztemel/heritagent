"""The EXP-0001 Colab environment (owner decision 2026-10-19): the files the
notebooks install from (``environments/exp0001-colab/``) keep the versions
the calibration pilot recorded and lock every dependency by hash, and
``cli env-check`` holds an installed environment to them."""

import os
import re
import stat
import sys
from pathlib import Path

import pytest

from evaluation.exp0001 import cli, environment
from evaluation.exp0001.serving import SOFTWARE_PACKAGES

ROOT = Path(__file__).resolve().parents[2]
ENV = ROOT / environment.ENVIRONMENT_DIR
LOCK = (ENV / environment.LOCK_FILE).read_text(encoding="utf-8")
LOCKED = environment.pinned_versions(LOCK)
PYTHON = environment.read_pin(ENV, environment.PYTHON_VERSION_FILE)

# what the calibration pilot ran with (its pilot-report.json, 2026-10-16; ADR-0008 D4)
PILOT_SOFTWARE = {"vllm": "0.29.0+cu129", "torch": "2.13.0+cu129", "transformers": "5.19.0",
                  "tokenizers": "0.23.3", "huggingface-hub": "2.2.0", "xgrammar": "0.2.8",
                  "llguidance": "1.7.6", "jinja2": "3.1.6", "textworld": "1.7.0",
                  "alfworld": "0.4.2"}


def test_the_lock_keeps_every_recorded_package_at_the_pilots_version():
    assert set(PILOT_SOFTWARE) == set(SOFTWARE_PACKAGES)
    for name in SOFTWARE_PACKAGES:
        assert LOCKED[environment.canonical(name)] == PILOT_SOFTWARE[name], name


def test_the_lock_resolves_its_input_and_the_packages_own_requirements():
    wanted = (ENV / environment.REQUIREMENTS_FILE).read_text(encoding="utf-8")
    names = [re.match(r"[A-Za-z0-9._-]+", line).group(0) for line in wanted.splitlines()
             if line.strip() and not line.startswith("#")]
    assert names and all(environment.canonical(name) in LOCKED for name in names)
    for name, version in environment.pinned_versions(wanted).items():
        assert LOCKED[name] == version, name
    # the package's dependencies and the extras the notebooks used (alfworld, figures)
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    for name in ("jsonschema", "alfworld", "textworld", "matplotlib"):
        assert f'"{name}' in pyproject and name in LOCKED, name
    assert environment.PROJECT not in LOCKED  # installed on top, from the checkout


def test_every_locked_package_is_held_by_its_hashes_on_the_serving_builds_indexes():
    blocks = re.split(r"\n(?=[A-Za-z0-9])", LOCK.split("\n\n", 1)[1])
    pinned = [block for block in blocks if environment.pinned_versions(block)]
    assert len(pinned) == len(LOCKED) > 100
    assert all("--hash=sha256:" in block for block in pinned)
    indexes = [line for line in LOCK.splitlines() if line.startswith("--")]
    assert indexes == ["--index-url https://pypi.org/simple",
                       "--extra-index-url https://wheels.vllm.ai/0.29.0/cu129",
                       "--extra-index-url https://download.pytorch.org/whl/cu129"]


def test_the_lock_was_resolved_by_the_committed_script_for_the_pinned_python():
    header = LOCK.splitlines()[1].split()
    script = (ROOT / "scripts" / "exp0001" / "lock_colab_environment.sh").read_text(
        encoding="utf-8")
    command = " ".join(script.split("uv pip compile", 1)[1].replace("\\\n", " ").split())
    command = command.replace('"$(cat python-version)"', PYTHON)
    assert header[header.index("compile") + 1:] == command.split()
    for name in (environment.UV_VERSION_FILE, environment.PYTHON_VERSION_FILE):
        assert re.fullmatch(r"\d+\.\d+\.\d+", environment.read_pin(ENV, name)), name
    assert f"--python-version {PYTHON} " in LOCK.splitlines()[1]


def fake_environment(tmp_path, *, python="3.13.15", lock="a==1.0 \\\n    --hash=sha256:00\n"):
    env = tmp_path / "env"
    env.mkdir()
    (env / environment.PYTHON_VERSION_FILE).write_text(python + "\n")
    (env / environment.LOCK_FILE).write_text(lock)
    return env


def server_on(directory):
    directory.mkdir(parents=True, exist_ok=True)
    vllm = directory / "vllm"
    vllm.write_text("#!/bin/sh\n")
    vllm.chmod(vllm.stat().st_mode | stat.S_IEXEC)
    return str(vllm)


def test_the_check_names_each_difference(tmp_path):
    lock = "vllm==0.29.0 \\\n    --hash=sha256:00\nnumpy==2.0 \\\n    --hash=sha256:11\n"
    env = fake_environment(tmp_path, lock=lock)
    own = tmp_path / "venv" / "bin"
    server = server_on(own)
    ok = {"vllm": "0.29.0", "numpy": "2.0", environment.PROJECT: "0.1.0"}

    def differ(**kwargs):
        args = {"python": "3.13.15", "installed": ok, "executable": str(own / "python"),
                "which": lambda name: server, **kwargs}
        return environment.environment_differences(env, **args)

    assert differ() == {}
    assert differ(python="3.13.16") == {"python": {"locked": "3.13.15", "here": "3.13.16"}}
    assert differ(installed={**ok, "numpy": "2.1"}) == {
        "packages": {"numpy": {"locked": "2.0", "here": "2.1"}}}
    assert differ(installed={"vllm": "0.29.0"}) == {
        "packages": {"numpy": {"locked": "2.0", "here": None}}}
    assert differ(installed={**ok, "pip": "26.0"}) == {"not in the lock": {"pip": "26.0"}}
    # `vllm serve` must start this environment's server
    assert list(differ(which=lambda name: None)) == ["vllm on the PATH"]
    other = server_on(tmp_path / "usr" / "bin")
    assert list(differ(which=lambda name: other)) == ["vllm on the PATH"]


def test_env_check_passes_only_in_the_locked_environment(tmp_path, monkeypatch, capsys):
    own = tmp_path / "venv" / "bin"
    server_on(own)
    monkeypatch.setattr(sys, "executable", str(own / "python"))
    monkeypatch.setenv("PATH", f"{own}{os.pathsep}{os.environ.get('PATH', '')}")
    monkeypatch.setattr(sys, "version", f"{PYTHON} (main) [GCC]")
    locked = {**LOCKED, environment.PROJECT: "0.1.0"}
    monkeypatch.setattr(environment, "installed_versions", lambda: dict(locked))
    assert cli.main(["env-check", "--repo", str(ROOT)]) == 0
    assert f"Python {PYTHON}, {len(LOCKED)} packages" in capsys.readouterr().out
    locked["torch"] = "2.13.0"  # a CPU build in place of the CUDA one
    assert cli.main(["env-check", "--repo", str(ROOT)]) == 2
    err = capsys.readouterr().err
    assert "not the locked environment" in err and '"torch"' in err and "2.13.0+cu129" in err
    assert cli.main(["env-check", "--repo", str(tmp_path)]) == 2  # no environment files
    assert "cannot be read" in capsys.readouterr().err


@pytest.mark.parametrize("line, pin", [
    ("vllm==0.29.0+cu129 \\", ("vllm", "0.29.0+cu129")),
    ("Jinja2==3.1.6", ("jinja2", "3.1.6")),
    ("huggingface_hub==2.2.0 ; python_version >= '3.10'", ("huggingface-hub", "2.2.0")),
    ("    --hash=sha256:00", None), ("# via vllm", None), ("matplotlib>=3.8", None),
    ("--extra-index-url https://download.pytorch.org/whl/cu129", None)])
def test_a_pin_is_a_name_and_a_version(line, pin):
    assert environment.pinned_versions(line) == ({} if pin is None else dict([pin]))
