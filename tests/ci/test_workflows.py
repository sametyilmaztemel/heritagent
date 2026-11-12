"""The CI workflows' own rules (issue #18): every job has a time limit, a
newer push to a pull request cancels its older CI run, pytest says why it
skipped, the real ALFWorld game runs in its own job, and the CI tools are
pinned."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = ROOT / ".github" / "workflows"


def _load(name):
    import yaml  # no importorskip: the dev extra provides it
    return yaml.safe_load((WORKFLOWS / name).read_text(encoding="utf-8"))


def _runs(job):
    return [step["run"] for step in job["steps"] if "run" in step]


def test_the_dev_extra_brings_yaml_so_the_locked_manifest_tests_run():
    """tests/ci/test_check_artifacts.py reads the locked manifests with yaml;
    without it they skip in every tests job."""
    import yaml
    assert yaml.safe_load("locked_at: '2026-10-14'") == {"locked_at": "2026-10-14"}


def test_every_job_has_a_time_limit():
    for name in ("ci.yml", "link-rot.yml"):
        for job_id, job in _load(name)["jobs"].items():
            assert isinstance(job.get("timeout-minutes"), int), f"{name}: {job_id}"
            assert 0 < job["timeout-minutes"] <= 30, f"{name}: {job_id}"


def test_a_newer_push_cancels_the_older_ci_run_of_a_pull_request_but_not_of_main():
    concurrency = _load("ci.yml")["concurrency"]
    assert "github.ref" in concurrency["group"]
    assert concurrency["cancel-in-progress"] == "${{ github.event_name == 'pull_request' }}"


def test_every_pytest_step_reports_why_it_skipped():
    commands = [run for job in _load("ci.yml")["jobs"].values() for run in _runs(job)
                if "pytest" in run]
    assert commands
    assert all(re.search(r"pytest .*-rs\b", run) for run in commands), commands


def test_the_real_game_job_installs_the_alfworld_extra_and_fails_on_a_skip():
    job = _load("ci.yml")["jobs"]["real-game"]
    assert job["steps"][1]["with"]["python-version"] == "3.13"  # the Colab lock's Python
    runs = _runs(job)
    assert any('".[dev,alfworld]"' in run for run in runs)
    [step] = [step for step in job["steps"] if "test_alfworld_adapter.py" in step.get("run", "")]
    assert step["shell"] == "bash"  # -eo pipefail: a pytest failure is not hidden by tee
    assert 'grep -q "SKIPPED"' in step["run"] and "exit 1" in step["run"]


def test_actions_are_on_majors_that_run_on_node_24():
    minimum = {"actions/checkout": 5, "actions/setup-python": 6}
    for name in ("ci.yml", "link-rot.yml"):
        for job in _load(name)["jobs"].values():
            for step in job["steps"]:
                if "uses" in step:
                    action, major = re.fullmatch(r"([\w./-]+)@v(\d+)", step["uses"]).groups()
                    assert int(major) >= minimum[action], f"{name}: {step['uses']}"


def test_the_ci_tools_are_pinned_to_exact_versions():
    lines = [line.strip() for line in
             (ROOT / "scripts" / "ci" / "requirements-ci.txt").read_text(encoding="utf-8").splitlines()]
    requirements = [line for line in lines if line and not line.startswith("#")]
    assert requirements
    assert all(re.fullmatch(r"[A-Za-z0-9_.-]+==[0-9][0-9A-Za-z.]*", line) for line in requirements), requirements
