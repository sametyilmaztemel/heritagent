"""The builder of the run rehearsal's committed script
(``scripts/exp0001/build_run_rehearsal_script.py``; issue #13 Phase B): it
imports and parses its arguments, and its cache gives back only rows made
with the same ALFWorld and TextWorld versions and manifests.
The synthetic manifests and scripted games; no ALFWorld, no planner."""

import importlib.metadata
import importlib.util
import json
from pathlib import Path

import pytest

from evaluation.exp0001 import data
from evaluation.exp0001 import run_rehearsal as rr
from tests.evaluation.exp0001_workers import GameFactory
from tests.evaluation.test_exp0001_pipeline import PILOTS, bank_of, splits  # noqa: F401 (fixture)

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "exp0001" / \
    "build_run_rehearsal_script.py"


@pytest.fixture
def builder():
    spec = importlib.util.spec_from_file_location("build_run_rehearsal_script", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_builder_imports_and_parses_its_arguments(builder, capsys):
    with pytest.raises(SystemExit) as stopped:
        builder.main(["--help"])
    assert stopped.value.code == 0
    usage = capsys.readouterr().out
    assert all(option in usage for option in ("--data-root", "--out", "--cache", "--workers",
                                              "--memory-gb", "--timeout"))
    with pytest.raises(SystemExit) as stopped:
        builder.main([])  # --data-root is required
    assert stopped.value.code == 2


def start_text(data_root, split, game_id):
    return {"split": split, "game_id": game_id, "text": GameFactory()(split, game_id).start(),
            "error": None}


def plan_in_process(data_root, split, game_id, **limits):
    return {"split": split, "game_id": game_id, "plan": ["win"], "error": None}


def build(builder, monkeypatch, tmp_path, splits, version="0.4.2"):  # noqa: F811
    """``main`` on the synthetic manifests, its texts and plans made in this
    process; returns the games whose text or plan it made."""
    pilot, bank = PILOTS[splits["manifest_sha256"]], bank_of(splits)
    monkeypatch.setattr(builder, "_manifests", lambda: (splits, pilot, bank))
    monkeypatch.setattr(data, "verify_data", lambda *a: {"release": "0.2.2"})
    monkeypatch.setattr(importlib.metadata, "version", lambda name: version)
    made = []

    def in_process(pool, function, jobs, workers, on_result):
        for job in jobs:
            made.append(tuple(job[1:]))
            on_result(function(*job))

    monkeypatch.setattr(builder, "_bounded_map", in_process)
    monkeypatch.setattr(builder, "start_text", start_text)
    monkeypatch.setattr(builder, "plan_in_process", plan_in_process)
    code = builder.main(["--data-root", str(tmp_path / "data"), "--cache",
                         str(tmp_path / "cache"), "--out", str(tmp_path / "script.json"),
                         "--workers", "1"])
    return code, made


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_the_cache_reuses_only_rows_of_the_same_versions_and_manifests(
        builder, monkeypatch, tmp_path, splits):  # noqa: F811
    code, made = build(builder, monkeypatch, tmp_path, splits)
    games = rr.played_games(splits, bank_of(splits))
    assert code == 0 and sorted(set(made)) == sorted(games)
    written = (tmp_path / "script.json").read_bytes()
    # the same versions and manifests: nothing is made again, the same bytes
    code, made = build(builder, monkeypatch, tmp_path, splits)
    assert code == 0 and made == []
    assert (tmp_path / "script.json").read_bytes() == written
    # another TextWorld or ALFWorld version: every text and plan is made again
    code, made = build(builder, monkeypatch, tmp_path, splits, version="9.9.9")
    assert code == 0 and set(games) <= set(made)
    basis = {r["basis"]["versions"]["textworld"] for r in rows(tmp_path / "cache" / "texts.jsonl")}
    assert basis == {"0.4.2", "9.9.9"}


def test_a_cached_row_of_other_manifests_is_made_again(builder, monkeypatch, tmp_path,
                                                       splits):  # noqa: F811
    build(builder, monkeypatch, tmp_path, splits)
    for name in ("texts.jsonl", "plans.jsonl"):
        path = tmp_path / "cache" / name
        changed = rows(path)
        for row in changed:
            row["basis"]["manifests"] = "another"
        path.write_text("".join(json.dumps(r) + "\n" for r in changed), encoding="utf-8")
    code, made = build(builder, monkeypatch, tmp_path, splits)
    assert code == 0 and set(rr.played_games(splits, bank_of(splits))) <= set(made)


def test_a_cached_error_of_a_game_no_longer_played_does_not_fail_the_build(
        builder, monkeypatch, tmp_path, splits):  # noqa: F811
    build(builder, monkeypatch, tmp_path, splits)
    texts = tmp_path / "cache" / "texts.jsonl"
    basis = rows(texts)[0]["basis"]
    with texts.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"split": "train", "game_id": "a game no longer played",
                                 "text": None, "error": "OSError: gone", "basis": basis}) + "\n")
    code, made = build(builder, monkeypatch, tmp_path, splits)
    assert code == 0 and made == []


@pytest.mark.parametrize("matches", [True, False])
def test_the_data_are_checked_against_the_manifests_before_any_row_is_made(
        builder, monkeypatch, tmp_path, splits, matches):  # noqa: F811
    pilot, bank = PILOTS[splits["manifest_sha256"]], bank_of(splits)
    events = []

    def verify(*args):
        events.append(("verify", args))
        if not matches:
            raise RuntimeError("a game file the manifests do not hash")
        return {"release": "0.2.2"}

    def in_process(pool, function, jobs, workers, on_result):
        for job in jobs:
            events.append(("row", job[1:]))
            on_result(function(*job))

    monkeypatch.setattr(builder, "_manifests", lambda: (splits, pilot, bank))
    monkeypatch.setattr(data, "verify_data", verify)
    monkeypatch.setattr(builder, "_bounded_map", in_process)
    monkeypatch.setattr(builder, "start_text", start_text)
    monkeypatch.setattr(builder, "plan_in_process", plan_in_process)
    argv = ["--data-root", str(tmp_path / "data"), "--cache", str(tmp_path / "cache"),
            "--out", str(tmp_path / "script.json"), "--workers", "1"]
    if matches:
        assert builder.main(argv) == 0
    else:
        with pytest.raises(RuntimeError, match="do not hash"):
            builder.main(argv)
    assert events[0] == ("verify", (tmp_path / "data", splits, pilot, bank))
    assert [kind for kind, _ in events].count("verify") == 1
    if not matches:
        assert len(events) == 1 and not (tmp_path / "script.json").exists()
        assert not (tmp_path / "cache").exists()
