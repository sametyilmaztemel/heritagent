"""Paper figures for EXP-0001, drawn from what a run recorded (owner request,
2026-10-18). Read-only: a run directory, or the newest whole snapshot of its
mirror, is read and never written.

Every figure shows a quantity the run already recorded in ``report.json``,
``seed-<S>/gate.json`` or its ledger; nothing is re-estimated. The rules
that keep them faithful to the pre-registered analysis (ADR-0006, ADR-0009):

- an interval only on the four pooled contrasts (C-B, C-A, B-A on T_eval;
  reg(C)-reg(B) on T_reg), as recorded; no interval on rates, per-seed
  values or secondary outputs (ADR-0006 D2, D6);
- arm A ran once (D7): it is one row, never three seeds;
- arms that share a genome share their cells (ADR-0009 D2): a per-seed C-B
  of a seed where C and B are one genome is 0 by construction, and is
  marked so;
- the +5 pp mark only on C-B (the H2 success clause); B-A has no rule;
- rule outcomes are quoted from the record (``met``), never re-decided
  from rounded numbers, and carry ``claim_status``;
- an incomplete or unfinished run gets descriptive rates only, marked
  as such (ADR-0006 D5).

Each figure is written as ``<name>.pdf`` (for the paper), ``<name>.png``
and ``<name>.csv`` (its table view). ``captions.md`` holds the captions with
the disclosures the report must carry, and ``figures.json`` names every
input and output with its sha256. matplotlib is imported only when figures
are drawn (``pip install -e .[figures]``)."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import tempfile
import time
from collections.abc import Callable, Mapping
from pathlib import Path

from evaluation.exp0001.ledger import LedgerError, RunLedger
from evaluation.exp0001.mirror import latest_snapshot, read_snapshot_file
from evaluation.metrics.final_statistics import H2_MIN_DELTA, MINING_SEEDS

FIGURES_SCHEMA = "heritagent/exp0001-figures/0.1"
ARMS = ("A", "B", "C")
ARM_LABELS = {"A": "A: G0, no skills added", "B": "B: all frozen candidates",
              "C": "C: gate-promoted candidates"}
# the first three slots of the reference categorical palette, validated together for
# every pair (dataviz validate_palette.js, light mode: CVD dE >= 9.2, normal-vision >= 24);
# A's aqua is below 3:1 against white, so every figure also labels its rows and writes a
# table view. A marker shape per arm keeps them apart in grayscale print.
ARM_COLORS = {"C": "#2a78d6", "B": "#eb6834", "A": "#1baf7a"}
ARM_MARKERS = {"A": "^", "B": "s", "C": "o"}
INK, INK_2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
STAGES = ("replay", "ablation", "generalization_in_scope", "generalization_out_of_scope",
          "interaction_regression")
STAGE_LABELS = {"replay": "1 replay", "ablation": "2 ablation",
                "generalization_in_scope": "3 in scope",
                "generalization_out_of_scope": "3 out of scope",
                "interaction_regression": "4 regression,\ninteraction"}
CONTRASTS = ("C-B", "C-A", "B-A")
OUTPUT_NAMES = ("fig1_t_eval_contrasts", "fig2_t_reg", "fig3_seed_direction",
                "fig4_t_eval_success", "fig5_gate_outcomes", "fig6_genome_size",
                "fig7_adaptation_cost", "run_accounting")
STATES = {"complete": "complete run",
          "incomplete": "INCOMPLETE run: descriptive rates only, no primary outcome",
          "unfinished": "UNFINISHED run (interim): descriptive only, no primary outcome"}


class FiguresError(RuntimeError):
    """The run's records cannot be read or drawn (no run-start record, an
    unreadable mirror, matplotlib missing, an output inside the run)."""


# -- reading ------------------------------------------------------------------

class RunFiles:
    """Read-only access to a run directory, or to the newest whole snapshot
    of its mirror (each file checked against its hash)."""

    def __init__(self, *, run_dir: Path | None = None, mirror_dir: Path | None = None):
        if (run_dir is None) == (mirror_dir is None):
            raise FiguresError("give either a run directory or a mirror directory")
        self.root = Path(run_dir if run_dir is not None else mirror_dir)
        self.kind = "run_dir" if run_dir is not None else "mirror"
        self.snapshot_seq = None
        if not self.root.is_dir():
            raise FiguresError(f"{self.root}: no such directory")
        if self.kind == "mirror":
            self._snapshot = latest_snapshot(self.root)
            if self._snapshot is None:
                raise FiguresError(f"{self.root}: the mirror holds no whole snapshot")
            self.snapshot_seq = self._snapshot["seq"]
        self.digests: dict[str, str] = {}  # every file read, for figures.json

    def read(self, relpath: str) -> bytes | None:
        if self.kind == "mirror":
            data = read_snapshot_file(self.root, self._snapshot, relpath)
        else:
            path = self.root / relpath
            data = path.read_bytes() if path.is_file() else None
        if data is not None:
            self.digests[relpath] = hashlib.sha256(data).hexdigest()
        return data

    def json(self, relpath: str):
        data = self.read(relpath)
        return None if data is None else json.loads(data.decode("utf-8"))


def _rate(value) -> float | None:
    """A recorded fraction ``{"exact": "n/d", "float": x}`` as a float."""
    return None if value is None else float(value["float"])


def _pp(value) -> float | None:
    return None if value is None else 100 * _rate(value)


def _ledger(files: RunFiles, run_start: Mapping) -> tuple[dict | None, str | None]:
    """The ledger's counts, or None and why not (a torn tail of a live run)."""
    data = files.read("ledger.jsonl")
    if data is None:
        return None, "no ledger.jsonl"
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "ledger.jsonl"
        path.write_bytes(data)
        try:
            ledger = RunLedger(path, run_id=run_start["run_id"], clock=time.time,
                               caps=run_start["protocol"]["caps"], readonly=True)
        except LedgerError as exc:
            return None, f"the ledger cannot be read: {exc}"
        return {**ledger.summary(), "restores": sum(n.get("type") == "restore"
                                                    for n in ledger.notes),
                "repairs": sum(n.get("type") == "repair" for n in ledger.notes)}, None


def _gates(files: RunFiles) -> dict[int, dict]:
    gates = {}
    for seed in MINING_SEEDS:
        gate = files.json(f"seed-{seed}/gate.json")
        if gate is not None:
            gates[seed] = gate
    return gates


def _state(report: Mapping | None, ledger: Mapping | None) -> str:
    # a crash can leave a complete report beside a running ledger (pipeline.py): both must say so
    if report is not None and report.get("status") == "complete" \
            and ledger is not None and ledger["state"] == "complete":
        return "complete"
    if report is not None and report.get("status") == "incomplete":
        return "incomplete"
    return "unfinished"


def _arm_genomes(report: Mapping) -> dict[tuple[str, str], str]:
    """(arm, run) -> genome_id, run being "once" for A and the seed for B and C."""
    out = {}
    for entry in report.get("final_genomes") or ():
        for arm_run in entry["arms"]:
            arm, run = arm_run.split("/", 1)
            out[(arm, run)] = entry["genome_id"]
    return out


def collect(files: RunFiles) -> dict:
    """Everything the figures read, gathered once."""
    run_start = files.json("run-start.json")
    if run_start is None:
        raise FiguresError(f"{files.root}: no run-start.json; is this an EXP-0001 run "
                           "directory or its mirror?")
    report = files.json("report.json")
    ledger, ledger_note = _ledger(files, run_start)
    return {"run_start": run_start, "report": report, "ledger": ledger,
            "ledger_note": ledger_note, "gates": _gates(files),
            "state": _state(report, ledger),
            "genomes": _arm_genomes(report) if report else {}}


# -- the tables behind the figures ---------------------------------------------

def _rows_per_run(rates: Mapping, value: Callable) -> list[dict]:
    """One row for A (once), then per seed and pooled for B and C."""
    rows = []
    for arm in ARMS:
        entry = rates.get(arm)
        if entry is None:
            continue
        per_seed = entry.get("per_seed") or {}
        if arm == "A":  # one run, copied under every seed (ADR-0006 D7)
            first = next(iter(per_seed.values()), None)
            rows.append({"arm": arm, "run": "once", **value(first)})
            continue
        for seed in MINING_SEEDS:
            rows.append({"arm": arm, "run": str(seed), **value(per_seed.get(str(seed)))})
        rows.append({"arm": arm, "run": "pooled", **value({"rate": entry.get("pooled")})})
    return rows


def _rate_value(cell) -> dict:
    if cell is None:
        return {"rate_pct": None, "count": None, "recorded": None}
    return {"rate_pct": _pp(cell.get("rate")), "count": cell.get("count"),
            "recorded": cell.get("recorded")}


def t_eval_contrast_rows(statistics: Mapping) -> list[dict]:
    contrasts = statistics["t_eval"]["contrasts"]
    return [{"contrast": name, "point_pp": _pp(c["point_estimate"]),
             "lower_pp": _pp(c["lower_bound"]), "upper_pp": _pp(c["upper_bound"]),
             "point_exact": c["point_estimate"]["exact"],
             "lower_exact": c["lower_bound"]["exact"],
             "upper_exact": c["upper_bound"]["exact"], "n_cells": c.get("n_cells")}
            for name in CONTRASTS for c in [contrasts[name]]]


def seed_rows(statistics: Mapping, genomes: Mapping) -> list[dict]:
    consistency = statistics["seed_consistency"]
    rows = []
    for seed in MINING_SEEDS:
        entry = consistency["per_seed"][str(seed)]
        b, c, a = genomes.get(("B", str(seed))), genomes.get(("C", str(seed))), \
            genomes.get(("A", "once"))
        rows.append({"seed": seed, "c_minus_b_pp": _pp(entry["c_minus_b"]),
                     "c_minus_b_exact": entry["c_minus_b"]["exact"], "holds": entry["holds"],
                     "c_equals_b_genome": c is not None and c == b,
                     "c_equals_g0_genome": c is not None and c == a})
    return rows


def gate_rows(gates: Mapping[int, Mapping]) -> list[dict]:
    """Per seed and stage, how many candidates passed, failed, were not run
    (ADR-0012 D7, if accepted) or passed vacuously (ADR-0007 E6)."""
    rows = []
    for seed, gate in sorted(gates.items()):
        aggregates, children = gate["aggregates"], gate["children"]
        for stage in STAGES:
            counts = {"passed": 0, "failed": 0, "not_run": 0, "vacuous": 0}
            for aggregate in aggregates.values():
                result = aggregate["stages"][stage]
                measurements = (children.get(result.get("measurements_ref")) or {}) \
                    .get("measurements", {})
                if measurements.get("not_run"):
                    counts["not_run"] += 1
                elif result["pass"]:
                    counts["passed"] += 1
                    counts["vacuous"] += bool(measurements.get("vacuous"))
                else:
                    counts["failed"] += 1
            rows.append({"seed": seed, "stage": stage, "candidates": len(aggregates), **counts})
        promoted = sum(v == "promote" for v in gate["verdicts"].values())
        rows.append({"seed": seed, "stage": "promoted", "candidates": len(aggregates),
                     "passed": promoted, "failed": len(aggregates) - promoted,
                     "not_run": 0, "vacuous": 0})
    return rows


def threshold_versions(gates: Mapping[int, Mapping]) -> list[str]:
    return sorted({a["thresholds_used"]["version"] for gate in gates.values()
                   for a in gate["aggregates"].values()})


def genome_size_rows(sizes: Mapping) -> list[dict]:
    rows = []
    for arm in ARMS:
        runs = sizes["per_arm"].get(arm, {}).get("runs", {})
        for run, size in runs.items():
            rows.append({"arm": arm, "run": run, "genes": size["genes"],
                         "skill_genes": size["skill_genes"],
                         "skill_block_tokens": size["skill_block_tokens"]})
    return rows


def adaptation_rows(cost: Mapping) -> list[dict]:
    def median(value):
        return None if value is None else _rate(value["median"])

    rows = []
    for arm in ARMS:
        entry = cost["per_arm"].get(arm)
        if entry is None:
            continue
        for run, c in [*entry["runs"].items(), *([("pooled", entry["pooled"])]
                                                   if arm != "A" else [])]:
            rows.append({"arm": arm, "run": run, "tasks": c["tasks"], "solved": c["solved"],
                         "solved_pct": 100 * c["solved"] / c["tasks"] if c["tasks"] else None,
                         "censored_unsolved": c["censored_unsolved"],
                         "median_tokens_to_success": median(c["total_tokens_to_success"]),
                         "median_steps_to_success": median(c["steps_to_success"])})
    return rows


def accounting_rows(data: Mapping) -> list[dict]:
    caps = data["run_start"]["protocol"]["caps"]
    report, ledger = data["report"] or {}, data["ledger"] or {}
    rows = [("run_id", data["run_start"]["run_id"], None),
            ("state", data["state"], None),
            ("report status", report.get("status"), None),
            ("report stage", report.get("stage"), None),
            ("reason", report.get("reason") or ledger.get("reason"), None),
            ("episodes", ledger.get("episodes", report.get("episodes")), caps["episodes_max"]),
            ("model calls", ledger.get("model_calls", report.get("model_calls")),
             caps["model_calls_max"]),
            ("active hours", ledger.get("active_hours"), caps["wall_clock_hours_max"]),
            ("sessions", ledger.get("sessions"), None),
            ("aborted attempts", ledger.get("aborted_attempts",
                                            report.get("aborted_attempts")), None),
            ("restores", ledger.get("restores"), None),
            ("repairs", ledger.get("repairs"), None),
            ("model", data["run_start"]["model"]["model_id"], None),
            ("code commit", data["run_start"]["code"]["commit"], None)]
    for entry in report.get("final_genomes") or ():
        rows.append((f"genome {entry['genome_id'][:12]}", " ".join(entry["arms"]), None))
    return [{"item": item, "value": value, "cap": cap} for item, value, cap in rows]


# -- drawing --------------------------------------------------------------------

def _matplotlib():
    try:
        import matplotlib
    except ImportError:
        raise FiguresError("matplotlib is not installed: pip install -e .[figures]") from None
    matplotlib.use("Agg", force=True)
    from matplotlib.figure import Figure
    return matplotlib, Figure


STYLE = {"font.size": 8, "font.family": "sans-serif", "axes.edgecolor": AXIS,
         "axes.linewidth": 0.8, "axes.labelcolor": INK_2, "axes.titlesize": 9,
         "axes.titlecolor": INK, "axes.titlelocation": "left", "text.color": INK,
         "xtick.color": AXIS, "ytick.color": AXIS, "xtick.labelcolor": INK_2,
         "ytick.labelcolor": INK_2, "axes.spines.top": False, "axes.spines.right": False,
         "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False,
         "legend.fontsize": 7, "figure.facecolor": "white", "axes.facecolor": "white",
         "savefig.facecolor": "white", "pdf.fonttype": 42, "ps.fonttype": 42,
         "svg.hashsalt": "heritagent", "savefig.dpi": 300}


def _row_axis(ax, labels: list[str], bold: set[int] = frozenset()) -> None:
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    for i, tick in enumerate(ax.get_yticklabels()):
        if i in bold:
            tick.set_fontweight("bold")
    ax.set_ylim(len(labels) - 0.5, -0.5)  # first row on top
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x")
    ax.set_axisbelow(True)


def _run_label(row: Mapping) -> str:
    if row["arm"] == "A":
        return "A (ran once)"
    return f"{row['arm']} pooled" if row["run"] == "pooled" else f"{row['arm']} seed {row['run']}"


def _dot_rows(ax, rows: list[dict], key: str) -> None:
    """Rows by arm and run: filled markers for A and the pooled rows, open
    markers for single seeds."""
    for i, row in enumerate(rows):
        value = row[key]
        if value is None:
            ax.text(0.01, i, "not recorded", transform=ax.get_yaxis_transform(),
                    va="center", color=MUTED, fontsize=7)
            continue
        filled = row["arm"] == "A" or row["run"] == "pooled"
        ax.plot([value], [i], ARM_MARKERS[row["arm"]], markersize=6 if filled else 5,
                color=ARM_COLORS[row["arm"]], markerfacecolor=ARM_COLORS[row["arm"]]
                if filled else "white", markeredgewidth=1.4, zorder=3, clip_on=False)
    _row_axis(ax, [_run_label(r) for r in rows],
              {i for i, r in enumerate(rows) if r["arm"] == "A" or r["run"] == "pooled"})


def _signed(value: float) -> str:
    return f"{value:+.1f}".replace("-", "\u2212")


def _span(row: Mapping) -> str:
    return f"{_signed(row['point_pp'])} [{_signed(row['lower_pp'])}, {_signed(row['upper_pp'])}]"


def _compact(axis) -> None:
    """Large counts read as 20K, not 20000."""
    from matplotlib.ticker import FuncFormatter
    axis.set_major_formatter(FuncFormatter(
        lambda v, _: f"{v / 1000:g}K" if abs(v) >= 1000 else f"{v:g}"))


def _interval(ax, y: float, point, lower, upper, color=INK) -> None:
    ax.hlines(y, lower, upper, color=color, linewidth=1.6, zorder=2)
    ax.plot([point], [y], "o", color=color, markersize=6, markeredgecolor="white",
            markeredgewidth=1.2, zorder=3)


def _zero(ax) -> None:
    ax.axvline(0, color=AXIS, linewidth=1.0, zorder=1)


def _title(fig, title: str, state: str) -> None:
    # a run that did not complete says so on the figure itself, under the title
    fig.suptitle(title if state == "complete" else f"{title}\n{STATES[state]}", x=0.01,
                 ha="left", fontsize=9, color=INK)


def draw_t_eval_contrasts(Figure, rows, state):
    fig = Figure(figsize=(6.75, 1.9), layout="constrained")
    ax = fig.add_subplot()
    for i, row in enumerate(rows):
        _interval(ax, i, row["point_pp"], row["lower_pp"], row["upper_pp"])
        ax.annotate(_span(row), (row["upper_pp"], i), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=7, color=INK_2)
    _zero(ax)
    threshold = float(H2_MIN_DELTA) * 100
    ax.vlines(threshold, -0.32, 0.32, color=INK_2, linewidth=1.0, zorder=1)  # C-B row only
    ax.annotate(f"+{threshold:g} pp (H2)", (threshold, -0.32), xytext=(0, 2),
                textcoords="offset points", ha="center", va="bottom", fontsize=6.5, color=INK_2)
    labels = {"C-B": "C − B", "C-A": "C − A", "B-A": "B − A (no rule)"}
    _row_axis(ax, [labels[r["contrast"]] for r in rows])
    ax.set_xlabel("T_eval success difference, percentage points (95 % bootstrap interval)")
    lo = min(0, *(r["lower_pp"] for r in rows), threshold) - 2
    hi = max(0, *(r["upper_pp"] for r in rows), threshold) + 12
    ax.set_xlim(lo, hi)
    _title(fig, "Arm contrasts on T_eval (pooled over mining seeds)", state)
    return fig


def draw_t_reg(Figure, rate_rows, contrast, state):
    fig = Figure(figsize=(6.75, 2.6), layout="constrained")
    if contrast is not None:
        left, right = fig.subplots(1, 2, width_ratios=[2.2, 1])
    else:
        left, right = fig.add_subplot(), None
    _dot_rows(left, rate_rows, "rate_pct")
    left.set_xlabel("T_reg failure rate, % (higher = more failures)")
    left.set_xlim(-2, 102)
    if right is not None:
        _interval(right, 0, contrast["point_pp"], contrast["lower_pp"], contrast["upper_pp"])
        _zero(right)
        _row_axis(right, ["reg(C) − reg(B)"])
        right.set_xlabel("percentage points\n(95 % bootstrap interval)")
        span = max(5.0, abs(contrast["lower_pp"]), abs(contrast["upper_pp"])) + 3
        right.set_xlim(-span, span)
        right.annotate(_span(contrast), (contrast["point_pp"], 0), xytext=(0, 9),
                       textcoords="offset points", ha="center", fontsize=7, color=INK_2)
    _title(fig, "Regression on T_reg: failure rates per arm and the H2 regression contrast",
           state)
    return fig


def draw_seed_direction(Figure, rows, consistency, state):
    fig = Figure(figsize=(3.35, 2.1), layout="constrained")
    ax = fig.add_subplot()
    for i, row in enumerate(rows):
        color = ARM_COLORS["C"]
        ax.plot([row["c_minus_b_pp"]], [i], "o", markersize=6, color=color,
                markerfacecolor=color if row["holds"] else "white", markeredgewidth=1.4,
                zorder=3)
        note = "holds" if row["holds"] else "does not hold"
        if row["c_equals_b_genome"]:
            note += "; C = B genome, 0 by construction"
        elif row["c_equals_g0_genome"]:
            note += "; C = G0 genome"
        ax.annotate(note, (row["c_minus_b_pp"], i), xytext=(7, 0), textcoords="offset points",
                    va="center", fontsize=6.5, color=INK_2)
    _zero(ax)
    _row_axis(ax, [f"seed {r['seed']}" for r in rows])
    values = [r["c_minus_b_pp"] for r in rows]
    span = max(5.0, *(abs(v) for v in values))
    ax.set_xlim(min(0, *values) - 0.15 * span, max(0, *values) + 1.6 * span)
    ax.set_xlabel("C − B on T_eval, percentage points")
    _title(fig, f"Seed direction: {consistency['holding']} of {len(rows)} hold "
                f"(required {consistency['required']})", state)
    return fig


def draw_t_eval_success(Figure, rows, state):
    fig = Figure(figsize=(3.35, 2.6), layout="constrained")
    ax = fig.add_subplot()
    _dot_rows(ax, rows, "rate_pct")
    ax.set_xlim(-2, 102)
    ax.set_xlabel("T_eval success rate, %")
    _title(fig, "Success on T_eval per arm", state)
    return fig


def draw_gate(Figure, rows, state):
    seeds = sorted({r["seed"] for r in rows})
    order = [*STAGES, "promoted"]
    fig = Figure(figsize=(6.75, 2.3), layout="constrained")
    axes = fig.subplots(1, len(seeds), sharey=True, squeeze=False)[0]
    for ax, seed in zip(axes, seeds):
        mine = {r["stage"]: r for r in rows if r["seed"] == seed}
        k = mine["promoted"]["candidates"]
        for y, stage in enumerate(order):
            row = mine[stage]
            ax.barh(y, row["passed"], height=0.55, color=ARM_COLORS["C"], zorder=2)
            note = str(row["passed"])
            extra = [f"{row[key]} {label}" for key, label in (("vacuous", "vacuous"),
                                                             ("not_run", "not run"))
                     if row[key]]
            if extra:
                note += f" ({', '.join(extra)})"
            ax.annotate(note, (row["passed"], y), xytext=(3, 0), textcoords="offset points",
                        va="center", fontsize=6.5, color=INK_2)
        ax.set_xlim(0, max(k, 1) * 1.7)
        ax.set_title(f"seed {seed}: {k} candidates", fontsize=8, loc="left")
        ax.set_xlabel("candidates passing")
        ax.grid(axis="x")
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(range(len(order)))
    axes[0].set_yticklabels([STAGE_LABELS.get(s, s).replace("\n", " ") for s in order])
    axes[0].set_ylim(len(order) - 0.5, -0.5)
    _title(fig, "Gate outcomes per mining seed: candidates passing each stage, and promoted",
           state)
    return fig


def draw_genome_size(Figure, rows, state):
    fig = Figure(figsize=(6.75, 2.4), layout="constrained")
    left, right = fig.subplots(1, 2, sharey=True)
    for ax, key, label in ((left, "skill_genes", "skill genes"),
                           (right, "skill_block_tokens", "skill-block tokens")):
        _dot_rows(ax, rows, key)
        ax.set_xlabel(label)
        ax.set_xlim(left=0, right=max([r[key] for r in rows] + [1]) * 1.15)
        if key == "skill_block_tokens":
            _compact(ax.xaxis)
    _title(fig, "Genome size per arm (descriptive)", state)
    return fig


def draw_adaptation(Figure, rows, state):
    fig = Figure(figsize=(6.75, 2.6), layout="constrained")
    axes = fig.subplots(1, 3, sharey=True)
    for ax, key, label in ((axes[0], "solved_pct", "solved, % of transfer tasks"),
                           (axes[1], "median_tokens_to_success", "median tokens to success"),
                           (axes[2], "median_steps_to_success", "median steps to success")):
        _dot_rows(ax, rows, key)
        ax.set_xlabel(label)
        top = max([r[key] for r in rows if r[key] is not None] + [1])
        ax.set_xlim(left=0, right=102 if key == "solved_pct" else top * 1.15)
        if key == "median_tokens_to_success":
            _compact(ax.xaxis)
    _title(fig, "Adaptation cost on the transfer subset (H3 preview, descriptive)", state)
    return fig


# -- captions -------------------------------------------------------------------

def _met(clause) -> str:
    return "met" if clause else "not met"


def captions(data: Mapping, drawn: Mapping[str, str]) -> str:
    report = data["report"] or {}
    statistics = report.get("statistics") or {}
    state, run_start = data["state"], data["run_start"]
    claim = statistics.get("claim_status") or ""
    lines = [f"# EXP-0001 figures: {run_start['run_id']}", "",
             f"State: {STATES[state]}. Model: {run_start['model']['model_id']} "
             f"(revision {run_start['model']['revision']}); code commit "
             f"{run_start['code']['commit']}.",
             "Every figure shows quantities the run recorded; nothing is re-estimated. "
             "Serving used no batch-invariant kernels (ADR-0011 D2), so a temperature-0 reply "
             "may depend on the batch.", ""]
    if claim:
        lines += [f"Claim status of every rule outcome below: {claim}.", ""]
    h1, h2 = statistics.get("h1"), statistics.get("h2")
    contrasts = (statistics.get("t_eval") or {}).get("contrasts")
    texts = {
        "fig1_t_eval_contrasts": (
            "Pooled T_eval contrasts in percentage points: point estimate and two-sided 95 % "
            "percentile bootstrap interval as recorded (10,000 resamples, seed 0, task "
            "clusters in task_id order, exact arithmetic; ADR-0006). Uncertainty: "
            f"{(contrasts or {}).get('C-B', {}).get('uncertainty', '')}. The line at 0 applies "
            "to every row; the +5 pp mark applies to C − B only (H2 success clause). B − A has "
            "no decision rule. "
            + (f"H2 success clause: {_met(h2['success_clause']['met'])}; H1 rule: "
               f"{_met(h1['rule_met'])} (rule outcomes as recorded). " if h1 and h2 else "")
            + "Compare with the expectations in design §8 (forecasts, not decision rules)."),
        "fig2_t_reg": (
            "Left: T_reg failure rate per arm (higher = more failures), per mining seed (open "
            "markers) and pooled (filled); arm A ran once and serves every seed (ADR-0006 "
            "D7). No intervals on rates by design (ADR-0006). Right, when the run is "
            "complete: reg(C) − reg(B) with its recorded 95 % bootstrap interval; the H2 "
            "regression clause needs it below 0 with the upper bound below 0"
            + (f" (clause {_met(h2['regression_clause']['met'])}; scope: "
               f"{h2['regression_clause']['scope']})" if h2 else "")
            + ". T_reg holds tasks G0 solved at least once at temperature 0.7, so arm A can "
              "fail them (ADR-0007 E3)."),
        "fig3_seed_direction": (
            "C − B on T_eval per mining seed; filled = the seed holds (C − B > 0; a tie does "
            "not hold), open = it does not (ADR-0006 D3). Where C and B are one genome in a "
            "seed, its cells are shared and the difference is 0 by construction, which "
            "supports nothing (ADR-0009). Per-seed values carry no interval"
            + (f". H2 seed clause: {_met(h2['seed_clause']['met'])}." if h2 else ".")),
        "fig4_t_eval_success": (
            "T_eval success rate per arm, per mining seed (open markers) and pooled (filled); "
            "arm A ran once. Descriptive; no intervals by design (ADR-0006)."
            + (" In an incomplete run the arms' rates rest on different recorded subsets and "
               "are not comparable; the CSV gives each rate's recorded count."
               if state != "complete" else "")),
        "fig5_gate_outcomes": (
            "Candidates passing each gate stage per mining seed (marginal counts: every "
            "candidate is checked at every stage it reaches), and the number promoted (the "
            "gate's verdict). Out-of-scope passes of candidates whose applicability names "
            "every family are vacuous (ADR-0007 E6) and marked; 'not run' marks stages a "
            "candidate never reached. Threshold version: "
            + ", ".join(threshold_versions(data["gates"])) + "."),
        "fig6_genome_size": (
            "Genome size per arm: skill genes and the token count of the rendered skill block "
            f"({(report.get('genome_sizes') or {}).get('token_counter', 'the run tokenizer')}"
            "); arm A ran once. Secondary, descriptive only (ADR-0006 D6)."),
        "fig7_adaptation_cost": (
            "Adaptation cost on the transfer subset (the 20-task ablation pool, ADR-0007 E4): "
            "share solved, and median tokens and steps to success over solved tasks. "
            "Secondary and descriptive (H3 preview); the pool also selected arm C's traits at "
            "the gate's stage 2, so it can favour C."),
        "run_accounting": (
            "Run accounting against the caps frozen in the run-start record: episodes, model "
            "calls and active hours (only active run time counts, ADR-0008 D2), aborted "
            "attempts, restores and repairs (work after a restored snapshot ran again and its "
            "calls and time are not counted), and which arms shared a genome (ADR-0009 D2)."),
    }
    for name, path in drawn.items():
        lines += [f"## {name}", "", f"`{path}`", "", texts[name], ""]
    return "\n".join(lines)


# -- the whole set --------------------------------------------------------------

def _write_csv(path: Path, rows: list[dict]) -> None:
    buffer = io.StringIO()
    fields = list(dict.fromkeys(key for row in rows for key in row))  # in first-seen order
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    path.write_text(buffer.getvalue(), encoding="utf-8")


def _save(fig, out_dir: Path, name: str) -> None:
    fig.savefig(out_dir / f"{name}.pdf", metadata={"CreationDate": None, "Creator": None,
                                                    "Producer": None})
    fig.savefig(out_dir / f"{name}.png", metadata={"Software": None})


def _inside(path: Path, root: Path) -> bool:
    return path.resolve().is_relative_to(root.resolve())


def make_figures(*, run_dir: Path | None = None, mirror_dir: Path | None = None,
                 out_dir: Path, code_commit: str | None = None) -> dict:
    """Draw every figure the run's records allow into ``out_dir``. Returns
    the manifest also written as ``figures.json``; a figure the records do
    not allow yet is listed as skipped, with the reason. ``code_commit``
    names the code that drew them (the run's own commit is in run-start)."""
    files = RunFiles(run_dir=run_dir, mirror_dir=mirror_dir)
    out_dir = Path(out_dir)
    if _inside(out_dir, files.root):
        raise FiguresError("the output directory must be outside the run and mirror "
                           "directories (a snapshot would sweep it in)")
    data = collect(files)
    matplotlib, Figure = _matplotlib()
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in OUTPUT_NAMES:  # a figure skipped now must not survive from an earlier call
        for suffix in (".pdf", ".png", ".csv"):
            (out_dir / f"{name}{suffix}").unlink(missing_ok=True)
    for extra in ("captions.md", "figures.json"):
        (out_dir / extra).unlink(missing_ok=True)
    report, state = data["report"] or {}, data["state"]
    statistics = report.get("statistics") or {}
    complete = state == "complete"

    def contrasts():
        rows = t_eval_contrast_rows(statistics)
        return rows, (lambda: draw_t_eval_contrasts(Figure, rows, state))

    def t_reg():
        rates = (statistics.get("t_reg") or {}).get("regression_rate") or {}
        rows = _rows_per_run(rates, _rate_value)
        contrast = None
        if complete and statistics["t_reg"].get("contrast"):
            c = statistics["t_reg"]["contrast"]["reg(C)-reg(B)"]
            contrast = {"point_pp": _pp(c["point_estimate"]), "lower_pp": _pp(c["lower_bound"]),
                        "upper_pp": _pp(c["upper_bound"])}
            rows = rows + [{"arm": "reg(C)-reg(B)", "run": "pooled contrast",
                            "rate_pct": contrast["point_pp"], "count": None, "recorded": None,
                            "lower_pp": contrast["lower_pp"],
                            "upper_pp": contrast["upper_pp"]}]
        plot_rows = [r for r in rows if r["arm"] in ARMS]
        return rows, (lambda: draw_t_reg(Figure, plot_rows, contrast, state))

    def seeds():
        rows = seed_rows(statistics, data["genomes"])
        return rows, (lambda: draw_seed_direction(Figure, rows, statistics["seed_consistency"],
                                                  state))

    def success():
        rows = _rows_per_run(statistics["t_eval"]["success_rate"], _rate_value)
        return rows, (lambda: draw_t_eval_success(Figure, rows, state))

    def gate():
        rows = gate_rows(data["gates"])
        return rows, (lambda: draw_gate(Figure, rows, state))

    def sizes():
        rows = genome_size_rows(report["genome_sizes"])
        return rows, (lambda: draw_genome_size(Figure, rows, state))

    def adaptation():
        rows = adaptation_rows(report["adaptation_cost"])
        return rows, (lambda: draw_adaptation(Figure, rows, state))

    has_t_eval = bool((statistics.get("t_eval") or {}).get("success_rate"))
    plans = [
        ("fig1_t_eval_contrasts", contrasts if complete and statistics.get("t_eval", {})
         .get("contrasts") else None, "needs a complete run (ADR-0006 D5)"),
        ("fig2_t_reg", t_reg if (statistics.get("t_reg") or {}).get("regression_rate")
         else None, "no final evaluation recorded yet"),
        ("fig3_seed_direction", seeds if complete and statistics.get("seed_consistency")
         else None, "needs a complete run (ADR-0006 D5)"),
        ("fig4_t_eval_success", success if has_t_eval else None,
         "no final evaluation recorded yet"),
        ("fig5_gate_outcomes", gate if data["gates"] and len(threshold_versions(data["gates"]))
         == 1 else None, "no gate recorded yet" if not data["gates"] else
         "the gates used different threshold versions; they are not drawn together"),
        ("fig6_genome_size", sizes if complete and report.get("genome_sizes") else None,
         "needs a complete run"),
        ("fig7_adaptation_cost", adaptation if complete and report.get("adaptation_cost")
         else None, "needs a complete run"),
    ]
    manifest = {"schema": FIGURES_SCHEMA, "run_id": data["run_start"]["run_id"],
                "state": state, "drawn_by": {"code_commit": code_commit},
                "source": {"kind": files.kind,
                                           "snapshot_seq": files.snapshot_seq},
                "figures": [], "notes": []}
    if data["ledger_note"]:
        manifest["notes"].append(f"ledger: {data['ledger_note']}")
    drawn: dict[str, str] = {}
    with matplotlib.rc_context(STYLE):
        for name, plan, why_not in plans:
            if plan is None:
                manifest["figures"].append({"name": name, "status": "skipped",
                                            "reason": why_not})
                continue
            rows, draw = plan()
            _write_csv(out_dir / f"{name}.csv", rows)
            _save(draw(), out_dir, name)
            drawn[name] = f"{name}.pdf"
            manifest["figures"].append({"name": name, "status": "drawn"})
    _write_csv(out_dir / "run_accounting.csv", accounting_rows(data))
    drawn["run_accounting"] = "run_accounting.csv"
    manifest["figures"].append({"name": "run_accounting", "status": "drawn"})
    (out_dir / "captions.md").write_text(captions(data, drawn), encoding="utf-8")
    manifest["inputs"] = dict(sorted(files.digests.items()))
    manifest["outputs"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in sorted(out_dir.iterdir()) if p.is_file()
                           and (p.stem in OUTPUT_NAMES or p.name == "captions.md")}
    (out_dir / "figures.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                                          encoding="utf-8")
    return manifest
