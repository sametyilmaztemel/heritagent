"""Paper figures (owner request, 2026-10-18): drawn read-only from a run's
records, every number as recorded, an interval only where the analysis
records one, and a run that did not complete marked and given descriptive
figures only (ADR-0006 D5)."""

import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from evaluation.exp0001 import cli, figures
from evaluation.exp0001.figures import OUTPUT_NAMES, FiguresError, RunFiles, make_figures
from evaluation.exp0001.mirror import MirrorError, RunMirror, latest_snapshot
from evaluation.exp0001.protocol import CAPS, MINING_SEEDS
from evaluation.metrics.final_statistics import (
    CLAIM_STATUS,
    REGRESSION_CLAUSE_SCOPE,
    UNCERTAINTY,
)
from tests.evaluation.test_exp0001_pipeline import (  # noqa: F401 (fixtures)
    GATE_PER_SEED,
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
    h2 = clean[1]["statistics"]["h2"]
    assert f"H2 rule (all three clauses, as recorded): {'met' if h2['rule_met'] else 'not met'}" \
        in captions
    assert figures.V2_DISCLOSURE in captions  # ADR-0012 D11
    # ADR-0012 D7: the futility-stopped candidate's out-of-scope check, 4A and 4C were not run
    assert "'not run' counts stages a candidate's record says were not run" in captions
    assert "Every candidate is checked at every stage" not in captions
    assert clean[1]["adaptation_cost"]["scope"] in captions
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
    import evaluation.exp0001.pipeline as pipeline
    run_dir = tmp_path / "run"
    with pytest.MonkeyPatch.context() as patch:  # a stop the reserve cannot rule out
        patch.setattr(pipeline, "FINAL_EVALUATION_RESERVE", 0)
        report = make_run(run_dir, splits,
                          caps={**CAPS, "episodes_max": 540 + 3 * GATE_PER_SEED + 50}).run()
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


def copy_run(clean, tmp_path):
    run_dir = tmp_path / "run"
    shutil.copytree(clean[0], run_dir)
    return run_dir


def edit(path, change):
    document = json.loads(path.read_text(encoding="utf-8"))
    change(document)
    path.write_text(json.dumps(document), encoding="utf-8")


def test_vacuous_and_not_run_stages_are_read_from_the_records_children(clean, tmp_path):
    """The aggregate's stages name their children by the list "children", not by id:
    the child is found through ``child_records`` and its ``stage``."""
    run_dir = copy_run(clean, tmp_path)

    def mark(gate):
        first, second = list(gate["aggregates"].values())[:2]
        assert first["stages"]["generalization_out_of_scope"]["measurements_ref"] == "children"
        by_stage = {gate["children"][c]["stage"]: gate["children"][c]
                    for c in first["child_records"]}
        by_stage["generalization_out_of_scope"]["measurements"]["vacuous"] = True
        first["stages"]["generalization_out_of_scope"]["pass"] = True
        other = {gate["children"][c]["stage"]: gate["children"][c]
                 for c in second["child_records"]}
        other["ablation"]["measurements"]["not_run"] = True  # an ADR-0012 D7-style record
        second["stages"]["ablation"]["pass"] = False

    edit(run_dir / "seed-11" / "gate.json", mark)
    out = tmp_path / "figures"
    make_figures(run_dir=run_dir, out_dir=out)
    table = {(r["seed"], r["stage"]): r for r in rows(out, "fig5_gate_outcomes")}
    assert table[("11", "generalization_out_of_scope")]["vacuous"] == "1"
    assert table[("11", "ablation")]["not_run"] == "1"
    assert table[("23", "ablation")]["not_run"] == "0"
    captions = (out / "captions.md").read_text(encoding="utf-8")
    assert "'not run' counts stages" in captions
    assert ", ".join(f"seed {s}: {g['all_family_candidates']}"
                     for s, g in sorted(clean[1]["gates"].items())) in captions


def test_a_complete_report_beside_a_running_ledger_quotes_no_rule_outcome(clean, tmp_path):
    run_dir = copy_run(clean, tmp_path)  # the crash window: report written, ledger not closed
    ledger = run_dir / "ledger.jsonl"
    lines = ledger.read_text(encoding="utf-8").splitlines(keepends=True)
    ledger.write_text("".join(lines[:-1]), encoding="utf-8")
    out = tmp_path / "figures"
    manifest = make_figures(run_dir=run_dir, out_dir=out)
    assert manifest["state"] == "unfinished"
    captions = (out / "captions.md").read_text(encoding="utf-8")
    assert "H2 rule" not in captions and "clause met" not in captions \
        and "clause not met" not in captions and "H1 rule" not in captions
    assert {f["name"] for f in manifest["figures"] if f["status"] == "drawn"} >= \
        {"fig2_t_reg", "fig4_t_eval_success", "fig5_gate_outcomes"}
    assert not (out / "fig1_t_eval_contrasts.pdf").exists()
    assert "(the report is complete; the ledger's terminal record is missing)" in captions
    assert "may rest on different recorded subsets" in captions  # a possibility, not a claim


def test_a_missing_gate_record_is_named(clean, tmp_path):
    run_dir = copy_run(clean, tmp_path)
    (run_dir / "seed-23" / "gate.json").unlink()
    out = tmp_path / "figures"
    manifest = make_figures(run_dir=run_dir, out_dir=out)
    assert manifest["notes"] == ["gate records missing for mining seeds [23]; the report "
                                 "lists them"]
    assert {r["seed"] for r in rows(out, "fig5_gate_outcomes")} == {"11", "41"}
    assert "Gate records missing for mining seeds [23]" in \
        (out / "captions.md").read_text(encoding="utf-8")


def test_a_gate_without_candidates_is_drawn(clean, tmp_path):
    run_dir = copy_run(clean, tmp_path)
    for seed in MINING_SEEDS:
        edit(run_dir / f"seed-{seed}" / "gate.json",
             lambda gate: gate.update(aggregates={}, children={}, verdicts={}))
    manifest = make_figures(run_dir=run_dir, out_dir=tmp_path / "figures")
    assert {f["name"]: f["status"] for f in manifest["figures"]}["fig5_gate_outcomes"] == "drawn"
    assert {r["candidates"] for r in rows(tmp_path / "figures", "fig5_gate_outcomes")} == {"0"}


def test_a_torn_ledger_tail_is_noted_and_the_report_still_counts(clean, tmp_path):
    run_dir = copy_run(clean, tmp_path)
    with (run_dir / "ledger.jsonl").open("a", encoding="utf-8") as handle:
        handle.write('{"type": "heartbeat", "clo')  # a live run cut mid-line
    out = tmp_path / "figures"
    manifest = make_figures(run_dir=run_dir, out_dir=out)
    assert manifest["notes"][0].startswith("ledger: the ledger cannot be read")
    accounting = {r["item"]: r["value"] for r in rows(out, "run_accounting")}
    assert accounting["episodes"] == str(clean[1]["episodes"])
    assert accounting["active hours"] == ""


def test_a_blob_collected_under_the_reader_stops_it(clean, tmp_path):
    local, mirror = copy_run(clean, tmp_path), tmp_path / "mirror"
    RunMirror(local, mirror).snapshot()
    files = RunFiles(mirror_dir=mirror)  # the snapshot is listed whole
    digest = latest_snapshot(mirror)["files"]["report.json"]["sha256"]
    (mirror / "blobs" / digest[:2] / digest).unlink()  # then a live run collects a blob
    with pytest.raises(MirrorError, match="gone from the mirror"):
        files.read("report.json")
    with pytest.raises(FiguresError, match="no whole snapshot"):  # listed after: none whole
        make_figures(mirror_dir=mirror, out_dir=tmp_path / "figures")


def test_a_failure_part_way_leaves_the_earlier_set_whole(clean, tmp_path):
    out = tmp_path / "figures"
    make_figures(run_dir=clean[0], out_dir=out)
    before = tree(out)
    broken = copy_run(clean, tmp_path)
    edit(broken / "report.json",
         lambda report: report["genome_sizes"]["per_arm"]["B"]["runs"].update({"11": {}}))
    with pytest.raises(KeyError):
        make_figures(run_dir=broken, out_dir=out)
    assert tree(out) == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["figures", "run"]  # no staging left
    (out / "notes.txt").write_text("mine", encoding="utf-8")  # not ours: left alone
    make_figures(run_dir=clean[0], out_dir=out)
    assert (out / "notes.txt").read_text(encoding="utf-8") == "mine"


def test_a_seed_where_c_and_b_share_a_genome_is_marked(clean, tmp_path):
    run_dir = copy_run(clean, tmp_path)

    def share(report):
        genomes = report["final_genomes"]
        b = next(g for g in genomes if "B/11" in g["arms"])
        c = next(g for g in genomes if "C/11" in g["arms"])
        c["arms"].remove("C/11")
        b["arms"].append("C/11")

    edit(run_dir / "report.json", share)
    out = tmp_path / "figures"
    make_figures(run_dir=run_dir, out_dir=out)
    seed = {r["seed"]: r for r in rows(out, "fig3_seed_direction")}
    assert seed["11"]["c_equals_b_genome"] == "True" and seed["23"]["c_equals_b_genome"] == "False"
    assert "C = B in mining seeds [11]" in (out / "captions.md").read_text(encoding="utf-8")


def test_a_missing_child_record_and_a_count_the_report_disagrees_with_are_named(clean,
                                                                                 tmp_path):
    run_dir = copy_run(clean, tmp_path)

    def drop_and_mark(gate):
        first = next(iter(gate["aggregates"].values()))
        out_of_scope = next(c for c in first["child_records"]
                            if gate["children"][c]["stage"] == "generalization_out_of_scope")
        del gate["children"][out_of_scope]  # the aggregate's own threshold still marks it
        first["stages"]["generalization_out_of_scope"]["pass"] = True
        first["stages"]["generalization_out_of_scope"]["threshold"] = "vacuous: every family"

    edit(run_dir / "seed-11" / "gate.json", drop_and_mark)
    out = tmp_path / "figures"
    make_figures(run_dir=run_dir, out_dir=out)
    row = next(r for r in rows(out, "fig5_gate_outcomes")
               if r["seed"] == "11" and r["stage"] == "generalization_out_of_scope")
    assert row["missing_child"] == "1" and row["vacuous"] == "1"
    captions = (out / "captions.md").read_text(encoding="utf-8")
    assert "1 stage records lack their child record" in captions
    assert f"Seed 11: the report counts {clean[1]['gates']['11']['all_family_candidates']} " \
           "candidates naming every family, the gate records 1." in captions


def test_an_output_that_cannot_be_written_is_named(clean, tmp_path):
    blocked = tmp_path / "a-file"
    blocked.write_text("", encoding="utf-8")
    with pytest.raises(FiguresError, match="cannot write figures there"):
        make_figures(run_dir=clean[0], out_dir=blocked)


def test_a_not_run_all_family_candidate_counts_as_naming_every_family(clean):
    """The out-of-scope row counts a vacuous pass and a not-run record of an
    all-family candidate alike (ADR-0012 D7)."""
    gate = json.loads((clean[0] / "seed-11" / "gate.json").read_text())
    rows = {r["stage"]: r for r in figures.gate_rows({11: gate})}
    assert rows["generalization_out_of_scope"]["all_family"] == 0
    assert rows["generalization_out_of_scope"]["not_run"] == 1
    for child in gate["children"].values():
        if child["stage"] == "generalization_out_of_scope" and child["measurements"].get("not_run"):
            child["measurements"]["all_family_applicability"] = True
    rows = {r["stage"]: r for r in figures.gate_rows({11: gate})}
    assert rows["generalization_out_of_scope"]["all_family"] == 1


def test_the_v2_disclosure_follows_the_threshold_version(clean, tmp_path):
    """ADR-0012 D11: the disclosure is the v2 gate's; a record of another
    version gets none."""
    # ADR-0012 D11's disclosures, each named
    for clause in ("stage 1 checks only that a successful replay of the source task expresses "
                   "the gene", "the in-scope record is stage 2's games by reference",
                   "in ADR-0012's model (M), rarely binds",
                   "true effects below 0.05 can be promoted",
                   "the seeds' gates read the same benefit-bank games",
                   "H2 can hold only if arm B carries candidates that harm"):
        assert clause in figures.V2_DISCLOSURE, clause
    run_dir = copy_run(clean, tmp_path)

    def to_v1(gate):
        for aggregate in gate["aggregates"].values():
            aggregate["thresholds_used"]["version"] = "EXP-0001-CIG-M0-v1"

    for seed in MINING_SEEDS:
        edit(run_dir / f"seed-{seed}" / "gate.json", to_v1)
    out = tmp_path / "figures"
    make_figures(run_dir=run_dir, out_dir=out)
    captions = (out / "captions.md").read_text(encoding="utf-8")
    assert "EXP-0001-CIG-M0-v1" in captions and figures.V2_DISCLOSURE not in captions


def test_a_rehearsals_stamp_covers_no_label_or_legend():
    """The stamp sits in a strip of its own, under the axes' labels and a
    legend below the axes."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    fig = Figure(figsize=(6.75, 2.0), layout="constrained")
    ax = fig.add_subplot()
    ax.plot([0, 1], [0, 1], label="pooled over mining seeds (A: its one run)")
    ax.plot([0, 1], [1, 0], label="one mining seed")
    ax.set_xlabel("T_eval success difference, percentage points (95 % bootstrap interval)")
    fig.legend(loc="outside lower center", ncols=2)
    figures._stamp(fig)
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    renderer = canvas.get_renderer()
    stamp = next(t for t in fig.texts if t.get_text() == figures.REHEARSAL_STAMP)
    box = stamp.get_window_extent(renderer)
    assert box.y0 >= 0 and box.x1 <= fig.bbox.x1
    for other in (ax.get_tightbbox(renderer), fig.legends[0].get_window_extent(renderer)):
        assert not box.overlaps(other)
    # the layout leaves the raised legend its room
    assert not fig.legends[0].get_window_extent(renderer).overlaps(ax.get_tightbbox(renderer))


def test_matplotlib_loads_whatever_backend_a_notebook_kernel_names():
    """A notebook kernel names its inline backend in MPLBACKEND, which a
    command inherits; in the locked environment that backend does not exist
    and `import matplotlib` would refuse it. The figures load Agg instead."""
    code = ("from evaluation.exp0001.figures import _matplotlib; "
            "matplotlib, _ = _matplotlib(); print(matplotlib.get_backend())")
    env = {**os.environ, "MPLBACKEND": "no-such-backend"}
    plain = subprocess.run([sys.executable, "-c", "import matplotlib"], env=env,
                           capture_output=True, text=True, cwd=REPO)
    assert plain.returncode != 0 and "not a valid value for backend" in plain.stderr
    loaded = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True,
                            text=True, cwd=REPO)
    assert loaded.returncode == 0, loaded.stderr
    assert loaded.stdout.strip().lower() == "agg"
