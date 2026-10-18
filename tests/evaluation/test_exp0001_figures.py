"""Paper figures (owner request, 2026-10-18): drawn read-only from a run's
records, every number as recorded, an interval only where the analysis
records one, and a run that did not complete marked and given descriptive
figures only (ADR-0006 D5)."""

import csv
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from evaluation.exp0001 import cli, figures
from evaluation.exp0001.figures import OUTPUT_NAMES, FiguresError, make_figures
from evaluation.exp0001.mirror import RunMirror
from evaluation.exp0001.protocol import CAPS, MINING_SEEDS
from evaluation.metrics.final_statistics import (
    CLAIM_STATUS,
    REGRESSION_CLAUSE_SCOPE,
    UNCERTAINTY,
)
from tests.evaluation.test_exp0001_pipeline import (  # noqa: F401 (fixtures)
    PILOTS,
    clean,
    make_run,
    reopened_copy,
    splits,
)

REPO = Path(__file__).resolve().parents[2]
FIGURES = [n for n in OUTPUT_NAMES if n != "run_accounting"]
# SPEC section 34's words, and outcome words the analysis never uses
CLAIM_WORDS = re.compile(r"\b(validated?|first|novel|better|causal|general|model-independent|"
                         r"evolutionary improvement|inheritance|significan\w*|proven|confirm\w*)\b",
                         re.IGNORECASE)


def tree(root):
    """Every file's bytes and mtime: a read-only reader leaves both alone."""
    return {p.relative_to(root).as_posix(): (hashlib.sha256(p.read_bytes()).hexdigest(),
                                             p.stat().st_mtime_ns)
            for p in sorted(root.rglob("*")) if p.is_file()}


def rows(out, name):
    with (out / f"{name}.csv").open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


@pytest.fixture
def texts(monkeypatch):
    """Every string drawn on every saved figure."""
    seen = {}
    save = figures._save

    def capture(fig, out_dir, name):
        from matplotlib.text import Text
        seen[name] = [t.get_text() for t in fig.findobj(Text) if t.get_text()]
        save(fig, out_dir, name)

    monkeypatch.setattr(figures, "_save", capture)
    return seen


def test_a_complete_run_gives_every_figure_and_leaves_the_run_alone(clean, tmp_path):
    run_dir = clean[0]
    before = tree(run_dir)
    out = tmp_path / "figures"
    manifest = make_figures(run_dir=run_dir, out_dir=out)
    assert tree(run_dir) == before
    assert manifest["state"] == "complete" and manifest["source"]["kind"] == "run_dir"
    assert [(f["name"], f["status"]) for f in manifest["figures"]] == \
        [(name, "drawn") for name in OUTPUT_NAMES]
    for name in FIGURES:
        assert (out / f"{name}.pdf").read_bytes().startswith(b"%PDF")
        assert (out / f"{name}.png").read_bytes().startswith(b"\x89PNG")
        assert rows(out, name)
    assert json.loads((out / "figures.json").read_text(encoding="utf-8")) == manifest
    for name, digest in manifest["outputs"].items():
        assert hashlib.sha256((out / name).read_bytes()).hexdigest() == digest
    for rel, digest in manifest["inputs"].items():  # what was read, as it is on disk
        assert hashlib.sha256((run_dir / rel).read_bytes()).hexdigest() == digest
    assert {"run-start.json", "report.json", "ledger.jsonl"} <= set(manifest["inputs"])


def test_the_tables_hold_the_recorded_numbers(clean, tmp_path):
    run_dir, report = clean[0], clean[1]
    out = tmp_path / "figures"
    make_figures(run_dir=run_dir, out_dir=out)
    statistics = report["statistics"]
    contrasts = statistics["t_eval"]["contrasts"]
    for row in rows(out, "fig1_t_eval_contrasts"):
        recorded = contrasts[row["contrast"]]
        assert row["point_exact"] == recorded["point_estimate"]["exact"]
        assert row["lower_exact"] == recorded["lower_bound"]["exact"]
        assert row["upper_exact"] == recorded["upper_bound"]["exact"]
        assert float(row["point_pp"]) == pytest.approx(100 * recorded["point_estimate"]["float"])
    per_seed = statistics["seed_consistency"]["per_seed"]
    for row in rows(out, "fig3_seed_direction"):
        assert row["c_minus_b_exact"] == per_seed[row["seed"]]["c_minus_b"]["exact"]
        assert row["holds"] == str(per_seed[row["seed"]]["holds"])
    success = rows(out, "fig4_t_eval_success")
    # arm A ran once: one row, never three seeds (ADR-0006 D7)
    assert [(r["arm"], r["run"]) for r in success] == [("A", "once")] + [
        (arm, run) for arm in "BC" for run in [*map(str, MINING_SEEDS), "pooled"]]
    for row in success:
        rate = statistics["t_eval"]["success_rate"][row["arm"]]
        recorded = rate["pooled"] if row["run"] in ("once", "pooled") else \
            rate["per_seed"][row["run"]]["rate"]
        assert float(row["rate_pct"]) == pytest.approx(100 * recorded["float"])
    gate = rows(out, "fig5_gate_outcomes")
    for seed in MINING_SEEDS:
        promoted = next(r for r in gate if r["seed"] == str(seed) and r["stage"] == "promoted")
        assert int(promoted["passed"]) == report["gates"][str(seed)]["promoted"]
        assert int(promoted["candidates"]) == report["gates"][str(seed)]["candidates"]
    accounting = {r["item"]: r for r in rows(out, "run_accounting")}
    assert int(accounting["episodes"]["value"]) == report["episodes"]
    assert int(accounting["episodes"]["cap"]) == CAPS["episodes_max"]
    assert accounting["state"]["value"] == "complete"


def test_shared_genomes_are_marked(clean, tmp_path):
    out = tmp_path / "figures"
    make_figures(run_dir=clean[0], out_dir=out)
    shared = {g["genome_id"]: g["arms"] for g in clean[1]["final_genomes"]}
    by_arm = {arm: gid for gid, arms in shared.items() for arm in arms}
    for row in rows(out, "fig3_seed_direction"):
        assert row["c_equals_b_genome"] == str(by_arm[f"C/{row['seed']}"] ==
                                               by_arm[f"B/{row['seed']}"])
        assert row["c_equals_g0_genome"] == str(by_arm[f"C/{row['seed']}"] == by_arm["A/once"])
    accounting = [r for r in rows(out, "run_accounting") if r["item"].startswith("genome ")]
    assert sorted(r["value"] for r in accounting) == sorted(" ".join(a) for a in shared.values())


def test_captions_quote_the_record_and_the_figures_avoid_claim_words(clean, tmp_path, texts):
    out = tmp_path / "figures"
    make_figures(run_dir=clean[0], out_dir=out)
    captions = (out / "captions.md").read_text(encoding="utf-8")
    assert CLAIM_STATUS in captions
    assert UNCERTAINTY.format(seeds=list(MINING_SEEDS)) in captions
    assert REGRESSION_CLAUSE_SCOPE in captions
    assert not CLAIM_WORDS.findall(captions.replace(CLAIM_STATUS, ""))
    assert set(texts) == set(FIGURES)
    for name, strings in texts.items():
        assert not CLAIM_WORDS.findall(" ".join(strings)), name
    # the +5 pp mark is on the C - B row only, and B - A carries no rule
    assert any("+5 pp" in s for s in texts["fig1_t_eval_contrasts"])
    assert "B − A (no rule)" in texts["fig1_t_eval_contrasts"]


def test_a_run_stopped_in_the_final_evaluation_gets_descriptive_figures_only(splits, tmp_path,
                                                                              texts):
    run_dir = tmp_path / "run"
    report = make_run(run_dir, splits, caps={**CAPS, "episodes_max": 540 + 3 * 300 + 50}).run()
    assert report["stage"] == "final"
    out = tmp_path / "figures"
    manifest = make_figures(run_dir=run_dir, out_dir=out)
    status = {f["name"]: f["status"] for f in manifest["figures"]}
    assert manifest["state"] == "incomplete"
    statistics = report["statistics"]
    recorded_t_reg = any(arm["pooled"] for arm in statistics["t_reg"]["regression_rate"].values())
    assert status == {"fig1_t_eval_contrasts": "skipped",
                      "fig2_t_reg": "drawn" if recorded_t_reg else "skipped",
                      "fig3_seed_direction": "skipped", "fig4_t_eval_success": "drawn",
                      "fig5_gate_outcomes": "drawn", "fig6_genome_size": "skipped",
                      "fig7_adaptation_cost": "skipped", "run_accounting": "drawn"}
    assert not (out / "fig1_t_eval_contrasts.pdf").exists()
    success = rows(out, "fig4_t_eval_success")
    assert [r["recorded"] for r in success if r["arm"] == "A"] == \
        [str(statistics["t_eval"]["success_rate"]["A"]["per_seed"]["11"]["recorded"])]
    for name in ("fig4_t_eval_success", "fig5_gate_outcomes"):
        assert any("INCOMPLETE" in s for s in texts[name]), name
    if recorded_t_reg:  # no contrast and no verdict: the T_reg table holds rates only
        assert {r["arm"] for r in rows(out, "fig2_t_reg")} <= {"A", "B", "C"}
    else:
        assert next(f for f in manifest["figures"] if f["name"] == "fig2_t_reg")["reason"] == \
            "no cell of it recorded yet"
    captions = (out / "captions.md").read_text(encoding="utf-8")
    assert "H1 rule" not in captions and "clause met" not in captions
    assert "not comparable" in captions


def test_a_run_stopped_in_the_lifetime_gets_its_accounting(splits, tmp_path):
    run_dir = tmp_path / "run"
    make_run(run_dir, splits, caps={**CAPS, "episodes_max": 100}).run()
    out = tmp_path / "figures"
    manifest = make_figures(run_dir=run_dir, out_dir=out)
    assert manifest["state"] == "incomplete"
    assert [f["name"] for f in manifest["figures"] if f["status"] == "drawn"] == \
        ["run_accounting"]
    accounting = {r["item"]: r["value"] for r in rows(out, "run_accounting")}
    assert accounting["report stage"] == "lifetime" and accounting["episodes"] == "100"


def test_an_unfinished_run_is_interim(clean, tmp_path, texts):
    run_dir = reopened_copy(clean, tmp_path)  # no report yet, ledger still running
    manifest = make_figures(run_dir=run_dir, out_dir=tmp_path / "figures")
    assert manifest["state"] == "unfinished"
    drawn = [f["name"] for f in manifest["figures"] if f["status"] == "drawn"]
    assert drawn == ["fig5_gate_outcomes", "run_accounting"]
    assert any("UNFINISHED" in s for s in texts["fig5_gate_outcomes"])


def test_a_figure_skipped_now_does_not_survive_from_an_earlier_call(clean, splits, tmp_path):
    out = tmp_path / "figures"
    make_figures(run_dir=clean[0], out_dir=out)
    assert (out / "fig1_t_eval_contrasts.pdf").exists()
    stopped = tmp_path / "stopped"
    make_run(stopped, splits, caps={**CAPS, "episodes_max": 100}).run()
    manifest = make_figures(run_dir=stopped, out_dir=out)
    assert not list(out.glob("fig[0-9]_*"))
    assert set(manifest["outputs"]) == {"run_accounting.csv", "captions.md"}


def test_the_mirror_gives_the_same_tables_as_the_run_directory(clean, tmp_path):
    local = tmp_path / "local"
    shutil.copytree(clean[0], local)
    mirror = tmp_path / "mirror"
    assert RunMirror(local, mirror).snapshot() is not None
    before = tree(mirror)
    from_dir, from_mirror = tmp_path / "a", tmp_path / "b"
    one = make_figures(run_dir=local, out_dir=from_dir)
    two = make_figures(mirror_dir=mirror, out_dir=from_mirror)
    assert tree(mirror) == before
    assert two["source"] == {"kind": "mirror", "snapshot_seq": 1}
    assert one["inputs"] == two["inputs"]
    for name in OUTPUT_NAMES:
        assert (from_dir / f"{name}.csv").read_bytes() == \
            (from_mirror / f"{name}.csv").read_bytes(), name


def test_the_output_must_lie_outside_the_run_and_the_inputs_must_exist(clean, tmp_path):
    run_dir = clean[0]
    with pytest.raises(FiguresError, match="outside"):
        make_figures(run_dir=run_dir, out_dir=run_dir / "figures")
    assert not (run_dir / "figures").exists()
    with pytest.raises(FiguresError, match="run-start.json"):
        make_figures(run_dir=tmp_path, out_dir=tmp_path.parent / "out")
    with pytest.raises(FiguresError, match="no whole snapshot"):
        make_figures(mirror_dir=tmp_path, out_dir=tmp_path.parent / "out")
    with pytest.raises(FiguresError, match="either"):
        make_figures(run_dir=run_dir, mirror_dir=tmp_path, out_dir=tmp_path / "out")


def test_the_command(clean, tmp_path, capsys):
    out = tmp_path / "figures"
    assert cli.main(["figures", "--run-dir", str(clean[0]), "--out-dir", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "fig1_t_eval_contrasts: drawn" in printed and f"figures in {out}" in printed
    assert cli.main(["figures", "--run-dir", str(tmp_path / "missing"),
                     "--out-dir", str(out)]) == 2
    with pytest.raises(SystemExit):  # one source only
        cli.parser().parse_args(["figures", "--run-dir", "a", "--mirror-dir", "b",
                                 "--out-dir", "c"])


def test_the_module_imports_without_matplotlib_and_says_what_is_missing(tmp_path):
    """The run stack and CI's wheel test need no matplotlib; drawing names it."""
    code = ("import sys; sys.modules['matplotlib'] = None\n"
            "import evaluation.exp0001.cli, evaluation.exp0001.figures as f\n"
            "try:\n    f._matplotlib()\nexcept f.FiguresError as exc:\n    print(exc)\n")
    result = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True,
                            text=True, check=True)
    assert "pip install -e .[figures]" in result.stdout
