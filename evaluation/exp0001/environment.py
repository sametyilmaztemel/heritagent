"""The EXP-0001 Colab environment (owner decision 2026-10-19): one Python
and one hash-locked set of packages, the same on the CPU runtime that
rehearses the run and on the A100 that runs it.

``environments/exp0001-colab/`` holds the uv version (``uv-version``), the
Python version (``python-version``), the top-level pins
(``requirements.in``) and their hash-locked resolution with every
dependency (``requirements.lock``). The notebooks install them with uv into
a virtual environment of their own, so Colab's own Python and packages play
no part; ``cli env-check`` then holds the installed environment to these
files before anything else runs: the Python version, every locked package
at its locked version, nothing outside the lock but this project, and
``vllm`` on the PATH from this environment (the server runs from it)."""

from __future__ import annotations

import re
import shutil
from collections.abc import Callable, Mapping
from importlib import metadata
from pathlib import Path

ENVIRONMENT_DIR = Path("environments") / "exp0001-colab"
UV_VERSION_FILE = "uv-version"
PYTHON_VERSION_FILE = "python-version"
REQUIREMENTS_FILE = "requirements.in"
LOCK_FILE = "requirements.lock"
PROJECT = "heritagent"  # installed on top of the lock from the checkout, without dependencies

_PIN = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==([^\s\\;]+)")


def canonical(name: str) -> str:
    """A distribution name as PEP 503 compares it."""
    return re.sub(r"[-_.]+", "-", name).lower()


def pinned_versions(text: str) -> dict[str, str]:
    """The ``name==version`` lines of a requirements file (a uv-compiled lock
    or its input): hashes, comments, options and unpinned lines skipped."""
    pins = {}
    for line in text.splitlines():
        match = _PIN.match(line.strip())
        if match:
            pins[canonical(match.group(1))] = match.group(2)
    return pins


def read_pin(env_dir: Path, name: str) -> str:
    return (Path(env_dir) / name).read_text(encoding="utf-8").strip()


def installed_versions() -> dict[str, str]:
    """Every distribution this interpreter sees, by canonical name."""
    return {canonical(dist.metadata["Name"]): dist.version for dist in metadata.distributions()}


def environment_differences(env_dir: Path, *, python: str, installed: Mapping[str, str],
                            executable: str,
                            which: Callable[[str], str | None] = shutil.which) -> dict:
    """What differs between this environment and the locked one (empty when
    it is the locked one)."""
    env_dir = Path(env_dir)
    locked = pinned_versions((env_dir / LOCK_FILE).read_text(encoding="utf-8"))
    differ: dict = {}
    pinned_python = read_pin(env_dir, PYTHON_VERSION_FILE)
    if python != pinned_python:
        differ["python"] = {"locked": pinned_python, "here": python}
    changed = {name: {"locked": version, "here": installed.get(name)}
               for name, version in sorted(locked.items()) if installed.get(name) != version}
    if changed:
        differ["packages"] = changed
    extra = {name: installed[name] for name in sorted(set(installed) - set(locked) - {PROJECT})}
    if extra:
        differ["not in the lock"] = extra
    if "vllm" in locked:  # `vllm serve` runs from the PATH: it must be this environment's
        server = which("vllm")
        own = Path(executable).parent
        if server is None or Path(server).parent.resolve() != own.resolve():
            differ["vllm on the PATH"] = {"locked": str(own / "vllm"), "here": server}
    return differ
