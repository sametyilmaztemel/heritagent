"""scripts/ci/smoke_installed.py's package-data comparison: a wheel that
lacks, or ships a changed copy of, any non-Python file of the packages fails
the CI package job."""

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "ci" / "smoke_installed.py"


@pytest.fixture(scope="module")
def smoke():
    spec = importlib.util.spec_from_file_location("smoke_installed", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tree(root, files):
    for name, content in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    return root


CHECKOUT = {"traits/miner/miner.py": b"code", "traits/miner/prompts/differential-skill-v0.txt": b"prompt",
            "genome/schema/eag-0.1.genome.schema.json": b"{}"}


def test_a_wheel_with_every_package_data_file_passes(smoke, tmp_path):
    repo = _tree(tmp_path / "repo", CHECKOUT)
    site = _tree(tmp_path / "site", {**CHECKOUT, "traits/miner/__pycache__/miner.cpython-313.pyc": b"x"})
    assert smoke.package_data_problems(site, repo) == []


def test_a_wheel_without_the_prompt_template_fails(smoke, tmp_path):
    repo = _tree(tmp_path / "repo", CHECKOUT)
    site = _tree(tmp_path / "site", {name: content for name, content in CHECKOUT.items()
                                     if not name.endswith(".txt")})
    assert smoke.package_data_problems(site, repo) == [
        "package data missing from the wheel: ['traits/miner/prompts/differential-skill-v0.txt']"]


def test_a_wheel_with_a_changed_package_data_file_fails(smoke, tmp_path):
    repo = _tree(tmp_path / "repo", CHECKOUT)
    site = _tree(tmp_path / "site", {**CHECKOUT, "genome/schema/eag-0.1.genome.schema.json": b"{ }"})
    assert smoke.package_data_problems(site, repo) == [
        "package data in the wheel differs from the checkout: ['genome/schema/eag-0.1.genome.schema.json']"]
