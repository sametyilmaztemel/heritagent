"""EXP-0001 final-evaluation statistics (issue #13 Phase A; ADR-0006).

Pure functions over recorded per-task outcomes. Nothing here runs an
episode, reads T_eval for any decision other than the final analysis, or
retunes a threshold. The locked rules (EXP-0001-design section 5, manifest
``evaluation.statistics``) and the owner's decisions recorded in ADR-0006:

- D1 seeds: for each task the paired arm difference is averaged over the
  mining seeds; the tasks are the bootstrap clusters.
- D8 T_reg is one 40-task set shared by every mining seed; other T_reg
  input fails closed.
- D2 bootstrap: two-sided 95 % percentile, 10,000 resamples, seed 0, the
  clusters in task_id order (the ADR-0005 rule). It is computed with exact
  rational arithmetic on the same index stream as
  ``inheritance.bootstrap.paired_cluster_bootstrap``, so float rounding can
  never decide the strict "CI excludes 0" comparison.
- H2: the pooled C-B on T_eval is at least +5pp and its lower bound is
  above 0; the pooled reg(C)-reg(B) on T_reg is below 0 and its upper bound
  is below 0; and C-B is positive in at least 2 of the 3 mining seeds.
- H1: the pooled C-A on T_eval is above 0 and its lower bound is above 0.
- D3 a seed holds the C-B direction only if its C-B is strictly positive.
- D4 regression rate: the fraction of T_reg tasks an arm fails (every T_reg
  task was solved by G0).
- D5 incomplete data fails closed: no primary verdict unless every
  (arm, seed, task) cell holds exactly one recorded outcome.
- D6 bloat and adaptation cost are descriptive only: no CI, no test.
- D7 arm A (unchanged G0) runs once; its outcomes serve every seed.

The output states rule outcomes only. It never calls a result validated
(SPEC section 32) and makes no causal or inheritance claim (SPEC section 34).
"""

from __future__ import annotations

import hashlib
import random
import statistics
from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from types import MappingProxyType

from evaluation.benchmarks.alfworld.splits import T_EVAL_SIZE
from inheritance.regression import REGRESSION_TASK_COUNT, validate_token_counter

ARMS = ("A", "B", "C")
MINING_SEEDS = (11, 23, 41)  # EXP-0001 manifest, evaluation.mining_seeds
FINAL_BOOTSTRAP_SEED = 0  # ADR-0006 D2
FINAL_BOOTSTRAP_RESAMPLES = 10_000
FINAL_BOOTSTRAP_CONFIDENCE = 0.95
FINAL_CLUSTER_ORDER = "task_id"
H2_MIN_DELTA = Fraction(5, 100)  # "+5pp", compared exactly
MIN_CONSISTENT_SEEDS = 2  # ">= 2/3 mining seeds"

# What every result records, and what the manifest pre-registers (ADR-0006).
FINAL_RULE = MappingProxyType({
    "seed_pooling": "per-task-mean-over-seeds",
    "bootstrap": MappingProxyType({
        "method": "percentile", "confidence": FINAL_BOOTSTRAP_CONFIDENCE,
        "seed": FINAL_BOOTSTRAP_SEED, "resamples": FINAL_BOOTSTRAP_RESAMPLES,
        "cluster_order": FINAL_CLUSTER_ORDER}),
    "seed_direction": "strict-C-gt-B",
    "regression_rate": "t_reg-failure-rate",
    "incomplete_data": "fail-closed-no-primary-verdict",
    "arm_a": "run-once-reused-across-seeds",
    "identical_genomes": "one-run-cells-shared",  # ADR-0009 D2
    "secondary": "descriptive-only",
    "t_reg": "one-set-shared-across-seeds",
})

CLAIM_STATUS = ("pre-registered rule outcome only; not a validated result "
                "(SPEC section 32) and not a causal claim (SPEC section 34)")
BOOTSTRAP_METHOD = "percentile-paired-cluster-bootstrap-exact"
# ADR-0006 D1: the tasks are the clusters and the seeds are pooled, not resampled; what this
# means for the hypotheses is proposed for the owner in
# research/proposals/ADR-0006-notes-proposal.md, N1
UNCERTAINTY = ("task clusters only, conditional on the mining seeds {seeds}; seed-to-seed "
               "variability is not resampled")
# ADR-0006 D4, scoped by ADR-0009 D5 in D5's words; the in-sample reading is proposed for the
# owner in research/proposals/ADR-0006-notes-proposal.md, N4
REGRESSION_CLAUSE_SCOPE = ("measured on T_reg, the suite the gate's stage 4A used to decide "
                           "which traits arm C keeps (ADR-0009 D5)")


class FinalStatisticsError(ValueError):
    """The final-evaluation data do not fit the pre-registered analysis."""


@dataclass(frozen=True)
class ArmRun:
    """One arm's recorded final-evaluation outcomes: success per task id on
    T_eval and on T_reg. Arm A has ``seed=None`` (D7); B and C name their
    mining seed. Outcomes are ALFWorld's ``won`` (ADR-0003 D6): a step
    budget that runs out is a failure, not a missing cell."""

    arm: str
    seed: int | None
    t_eval: Mapping[str, bool]
    t_reg: Mapping[str, bool]

    def __post_init__(self):
        if not isinstance(self.arm, str):
            raise FinalStatisticsError(f"arm must be a str, got {self.arm!r}")
        if self.seed is not None and (isinstance(self.seed, bool)
                                      or not isinstance(self.seed, int)):
            raise FinalStatisticsError(f"seed must be None or an int, got {self.seed!r}")
        for name in ("t_eval", "t_reg"):
            cells = getattr(self, name)
            # a mapping holds one outcome per task; pairs could repeat a task
            if not isinstance(cells, Mapping):
                raise FinalStatisticsError(
                    f"{name} must be a mapping of task id to outcome, got {type(cells).__name__}")
            object.__setattr__(self, name, MappingProxyType(dict(cells)))


@dataclass(frozen=True)
class FinalEvaluationInput:
    """The complete recorded final evaluation of one EXP-0001 run.

    ``t_eval_task_ids`` is the full official T_eval set; ``t_reg_task_ids``
    is the one 40-task T_reg set every mining seed shares (D8). Both are
    lists or tuples, so a repeated id is caught. ``complete`` is False for a
    run that stopped early (a hard cap, D5); it then needs an
    ``incomplete_reason``."""

    t_eval_task_ids: tuple[str, ...]
    t_reg_task_ids: tuple[str, ...]
    runs: tuple[ArmRun, ...]
    complete: bool
    incomplete_reason: str | None = None

    def __post_init__(self):
        for name in ("t_eval_task_ids", "t_reg_task_ids", "runs"):
            value = getattr(self, name)
            if not isinstance(value, (list, tuple)):
                raise FinalStatisticsError(
                    f"{name} must be a list or tuple, got {type(value).__name__}")
            object.__setattr__(self, name, tuple(value))


@dataclass(frozen=True)
class ExactBootstrap:
    point_estimate: Fraction
    lower_bound: Fraction
    upper_bound: Fraction


def exact_cluster_bootstrap(numerators: list[int], denominator: int, *, seed: int,
                            n_resamples: int, confidence: float) -> ExactBootstrap:
    """The percentile paired cluster bootstrap of
    ``inheritance.bootstrap.paired_cluster_bootstrap`` over the deltas
    ``numerator / denominator``, with exact sums.

    It draws the same indices (``random.Random(seed)``, ``randrange(n)``,
    ``n`` draws per resample) and takes the same order statistics, so its
    bounds equal the float version's up to that version's rounding."""
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise FinalStatisticsError(f"bootstrap seed must be an explicit int, got {seed!r}")
    if isinstance(denominator, bool) or not isinstance(denominator, int) or denominator <= 0:
        raise FinalStatisticsError(f"denominator must be a positive int, got {denominator!r}")
    if not numerators:
        raise FinalStatisticsError("bootstrap requires at least one cluster")
    for index, numerator in enumerate(numerators):
        if isinstance(numerator, bool) or not isinstance(numerator, int) \
                or abs(numerator) > denominator:
            raise FinalStatisticsError(
                f"cluster delta {index} must be an int numerator with |delta| <= 1, "
                f"got {numerator!r}/{denominator}")
    if isinstance(n_resamples, bool) or not isinstance(n_resamples, int) or n_resamples <= 0:
        raise FinalStatisticsError("n_resamples must be a positive int")
    if not 0 < confidence < 1:
        raise FinalStatisticsError("confidence must be in (0, 1)")

    n = len(numerators)
    rng = random.Random(seed)
    totals: list[int] = []
    for _ in range(n_resamples):
        total = 0
        for _ in range(n):
            total += numerators[rng.randrange(n)]
        totals.append(total)
    totals.sort()
    # the same index arithmetic as the float version, float alpha included
    alpha = 1.0 - confidence
    lower_index = max(0, int(alpha / 2 * n_resamples))
    upper_index = min(n_resamples - 1, int((1 - alpha / 2) * n_resamples))
    scale = n * denominator
    return ExactBootstrap(point_estimate=Fraction(sum(numerators), scale),
                          lower_bound=Fraction(totals[lower_index], scale),
                          upper_bound=Fraction(totals[upper_index], scale))


def _fraction(value: Fraction) -> dict:
    return {"exact": f"{value.numerator}/{value.denominator}", "float": float(value)}


def _check_ids(ids, label: str, size: int) -> tuple[str, ...]:
    ids = tuple(ids)
    if any(not isinstance(i, str) or not i for i in ids):
        raise FinalStatisticsError(f"{label} task ids must be non-empty strings")
    if len(set(ids)) != len(ids):
        raise FinalStatisticsError(f"{label} task ids must be distinct")
    if len(ids) != size:
        raise FinalStatisticsError(f"{label} must have exactly {size} tasks, got {len(ids)}")
    return ids


def _check_cells(cells: Mapping[str, bool], expected: set[str], label: str,
                 complete: bool) -> None:
    keys = set(cells)
    if keys - expected:
        raise FinalStatisticsError(
            f"{label} has outcomes for tasks outside its set: "
            f"{sorted(map(repr, keys - expected))[:5]}")
    if complete and keys != expected:
        raise FinalStatisticsError(
            f"{label} is missing {len(expected - keys)} task outcomes "
            f"(e.g. {sorted(expected - keys)[:5]}); a complete run needs every cell (D5)")
    for task_id, success in cells.items():
        if not isinstance(success, bool):
            raise FinalStatisticsError(
                f"{label} outcome for {task_id!r} must be a bool, got {success!r}")


def _validate(data: FinalEvaluationInput) -> dict:
    if not isinstance(data, FinalEvaluationInput):
        raise FinalStatisticsError(
            f"expected a FinalEvaluationInput, got {type(data).__name__}")
    if not isinstance(data.complete, bool):
        raise FinalStatisticsError(f"complete must be a bool, got {data.complete!r}")
    if data.complete and data.incomplete_reason is not None:
        raise FinalStatisticsError("a complete run carries no incomplete_reason")
    if not data.complete and (not isinstance(data.incomplete_reason, str)
                              or not data.incomplete_reason.strip()):
        raise FinalStatisticsError("an incomplete run must record its reason (D5)")
    t_eval = _check_ids(data.t_eval_task_ids, "T_eval", T_EVAL_SIZE)
    t_reg = _check_ids(data.t_reg_task_ids, "T_reg", REGRESSION_TASK_COUNT)  # D8: one set
    if set(t_reg) & set(t_eval):
        raise FinalStatisticsError(
            f"T_reg and T_eval must be disjoint (T_reg is drawn from T_mine): "
            f"{sorted(set(t_reg) & set(t_eval))[:5]}")

    expected_keys = {("A", None)} | {(arm, seed) for arm in ("B", "C") for seed in MINING_SEEDS}
    runs: dict = {}
    for run in data.runs:
        if not isinstance(run, ArmRun):
            raise FinalStatisticsError(f"runs must be ArmRun records, got {type(run).__name__}")
        key = (run.arm, run.seed)
        if key not in expected_keys:
            raise FinalStatisticsError(
                f"unexpected run {key!r}: arm A runs once with seed None (D7), arms B and C "
                f"once per mining seed {list(MINING_SEEDS)}")
        if key in runs:
            raise FinalStatisticsError(f"duplicate run {key!r}")
        runs[key] = run
    if set(runs) != expected_keys:
        missing = sorted(expected_keys - set(runs), key=repr)
        raise FinalStatisticsError(f"missing runs {missing}")

    for (arm, seed), run in runs.items():
        label = f"arm {arm}" + ("" if seed is None else f" seed {seed}")
        _check_cells(run.t_eval, set(t_eval), f"{label} T_eval", data.complete)
        _check_cells(run.t_reg, set(t_reg), f"{label} T_reg", data.complete)
    return {"t_eval": t_eval, "t_reg": t_reg, "runs": runs}


def _success(run: ArmRun, task_set: str, task_id: str) -> int:
    return int(getattr(run, task_set)[task_id])


def _pooled_contrast(cells: list[tuple[str, int]]) -> dict:
    """Bootstrap one pooled contrast. ``cells`` are (task_id, delta) with
    delta in {-1, 0, 1}, one per (seed, task): each task's deltas are
    averaged over the seeds (D1), and the tasks, in task_id order, are the
    clusters (D2). Every task has one cell per seed (T_eval; the shared
    T_reg of D8), so a cluster's mean is its delta sum over the seed count."""
    by_task: dict[str, list[int]] = {}
    for task_id, delta in cells:
        by_task.setdefault(task_id, []).append(delta)
    if any(len(deltas) != len(MINING_SEEDS) for deltas in by_task.values()):
        raise FinalStatisticsError("every task needs one cell per mining seed")
    task_ids = sorted(by_task)  # ADR-0006 D2 / ADR-0005: task_id order
    denominator = len(MINING_SEEDS)
    numerators = [sum(by_task[t]) for t in task_ids]
    # the parameters actually used, read once and recorded as used
    seed, n_resamples = FINAL_BOOTSTRAP_SEED, FINAL_BOOTSTRAP_RESAMPLES
    confidence = FINAL_BOOTSTRAP_CONFIDENCE
    result = exact_cluster_bootstrap(numerators, denominator, seed=seed,
                                     n_resamples=n_resamples, confidence=confidence)
    return {
        "point_estimate": _fraction(result.point_estimate),
        "lower_bound": _fraction(result.lower_bound),
        "upper_bound": _fraction(result.upper_bound),
        "bootstrap": {"method": "percentile", "confidence": confidence, "seed": seed,
                      "resamples": n_resamples, "cluster_order": FINAL_CLUSTER_ORDER,
                      "implementation": BOOTSTRAP_METHOD},
        "uncertainty": UNCERTAINTY.format(seeds=list(MINING_SEEDS)),
        "cluster_task_ids": task_ids,
        "cluster_numerators": numerators,
        "cluster_denominator": denominator,
        "n_cells": len(cells),
        "_exact": result,
    }


def _seed_runs(runs: dict, arm: str, seed: int) -> ArmRun:
    return runs[("A", None)] if arm == "A" else runs[(arm, seed)]


def _t_eval_contrast(runs: dict, t_eval: tuple[str, ...], x: str, y: str) -> dict:
    cells = [(task_id, _success(_seed_runs(runs, x, seed), "t_eval", task_id)
              - _success(_seed_runs(runs, y, seed), "t_eval", task_id))
             for seed in MINING_SEEDS for task_id in t_eval]
    return _pooled_contrast(cells)


def _regression_contrast(runs: dict, t_reg: tuple[str, ...], x: str, y: str) -> dict:
    """reg(x) - reg(y): failure indicators, so delta = success(y) - success(x)."""
    cells = [(task_id, _success(_seed_runs(runs, y, seed), "t_reg", task_id)
              - _success(_seed_runs(runs, x, seed), "t_reg", task_id))
             for seed in MINING_SEEDS for task_id in t_reg]
    return _pooled_contrast(cells)


def _rates(runs: dict, task_ids_for, task_set: str, *, failure: bool) -> dict:
    """Per-arm per-seed rates over the recorded cells, plus the pooled rate
    over all of an arm's (seed, task) cells. Descriptive."""
    out = {}
    for arm in ARMS:
        per_seed, hits, total = {}, 0, 0
        for seed in MINING_SEEDS:
            run = _seed_runs(runs, arm, seed)
            cells = getattr(run, task_set)
            recorded = [t for t in task_ids_for(seed) if t in cells]
            count = sum((not cells[t]) if failure else cells[t] for t in recorded)
            per_seed[str(seed)] = {"count": count, "recorded": len(recorded),
                                   "rate": _fraction(Fraction(count, len(recorded)))
                                   if recorded else None}
            hits, total = hits + count, total + len(recorded)
        out[arm] = {"per_seed": per_seed,
                    "pooled": _fraction(Fraction(hits, total)) if total else None}
    return out


def _public(contrast: dict) -> dict:
    return {key: value for key, value in contrast.items() if not key.startswith("_")}


def analyze_final_evaluation(data: FinalEvaluationInput) -> dict:
    """The pre-registered EXP-0001 final analysis (ADR-0006).

    Returns a JSON-serialisable record: the rule, per-arm success and
    regression rates, the pooled T_eval contrasts C-B, C-A and B-A (all
    pairwise arm contrasts, design section 5), the pooled regression
    contrast reg(C)-reg(B), the per-seed C-B directions, and the H1 and H2
    rule outcomes. An incomplete run (D5) gets the descriptive rates only:
    its contrasts and rule outcomes are ``None``."""
    checked = _validate(data)
    t_eval, t_reg, runs = checked["t_eval"], checked["t_reg"], checked["runs"]
    record = {
        "rule": {**FINAL_RULE, "bootstrap": dict(FINAL_RULE["bootstrap"])},
        "claim_status": CLAIM_STATUS,
        "status": "complete" if data.complete else "incomplete",
        "incomplete_reason": data.incomplete_reason,
        "mining_seeds": list(MINING_SEEDS),
        "t_eval": {"task_count": len(t_eval),
                   "success_rate": _rates(runs, lambda seed: t_eval, "t_eval",
                                          failure=False),
                   "contrasts": None},
        "t_reg": {"task_ids": sorted(t_reg),
                  "regression_rate": _rates(runs, lambda seed: t_reg, "t_reg",
                                            failure=True),
                  "contrast": None},
        "seed_consistency": None,
        "h2": None,
        "h1": None,
    }
    if not data.complete:
        return record  # D5: no primary verdict on an incomplete run

    contrasts = {f"{x}-{y}": _t_eval_contrast(runs, t_eval, x, y)
                 for x, y in (("C", "B"), ("C", "A"), ("B", "A"))}
    regression = _regression_contrast(runs, t_reg, "C", "B")
    per_seed = {}
    for seed in MINING_SEEDS:
        delta = Fraction(sum(_success(runs[("C", seed)], "t_eval", t)
                             - _success(runs[("B", seed)], "t_eval", t) for t in t_eval),
                         len(t_eval))
        per_seed[str(seed)] = {"c_minus_b": _fraction(delta), "holds": delta > 0}  # D3
    holding = sum(entry["holds"] for entry in per_seed.values())

    cb, ca, reg = contrasts["C-B"]["_exact"], contrasts["C-A"]["_exact"], regression["_exact"]
    success_clause = cb.point_estimate >= H2_MIN_DELTA and cb.lower_bound > 0
    regression_clause = reg.point_estimate < 0 and reg.upper_bound < 0
    seed_clause = holding >= MIN_CONSISTENT_SEEDS
    record["t_eval"]["contrasts"] = {name: _public(c) for name, c in contrasts.items()}
    record["t_reg"]["contrast"] = {"reg(C)-reg(B)": _public(regression)}
    record["seed_consistency"] = {"per_seed": per_seed, "holding": holding,
                                  "required": MIN_CONSISTENT_SEEDS, "met": seed_clause}
    record["h2"] = {
        "success_clause": {"rule": "C-B >= +5pp and 95% CI lower bound > 0",
                           "met": success_clause},
        "regression_clause": {"rule": "reg(C)-reg(B) < 0 and 95% CI upper bound < 0",
                              "scope": REGRESSION_CLAUSE_SCOPE, "met": regression_clause},
        "seed_clause": {"rule": "C-B > 0 in >= 2 of 3 mining seeds", "met": seed_clause},
        "rule_met": success_clause and regression_clause and seed_clause,
    }
    record["h1"] = {"rule": "C-A > 0 and 95% CI lower bound > 0",
                    "rule_met": ca.point_estimate > 0 and ca.lower_bound > 0}
    return record


# --- D6: descriptive secondary outputs ------------------------------------

def _gene_count(genome: dict) -> tuple[int, int]:
    genes = genome.get("genes") or {}
    slots = sum(len(genes.get(section) or {}) for section in ("cognition", "memory", "execution"))
    skills = len(genes.get("skills") or [])
    return slots + skills, skills


def measure_genome_size(genome: dict, registry, counter) -> dict:
    """genome_size_genes and genome_size_tokens of one arm genome (D6).

    Genes are the genome's gene refs (policy slots plus skills). Tokens are
    the counter's count for the model-visible skill block that renders ALL
    of the genome's skills, whatever their regulation; a genome without
    skills has 0 skill-block tokens. The counter identity and the rendered
    text's SHA-256 are recorded with the count."""
    from runtime.expression import compile_runtime_config
    from runtime.skills import skill_block

    if not isinstance(genome, dict):
        raise FinalStatisticsError(f"genome must be a dict, got {type(genome).__name__}")
    try:
        validate_token_counter(counter)
    except Exception as exc:
        raise FinalStatisticsError(f"invalid token counter: {exc}") from exc
    config = compile_runtime_config(genome, registry)
    text = skill_block(config.skills) if config.skills else ""
    tokens = counter.count_tokens(text) if text else 0
    # the probe in validate_token_counter only counts "": check the real count
    _require_count(tokens, "skill_block_tokens")
    genes, skill_genes = _gene_count(genome)
    return {"genome_id": genome.get("genome_id"), "genes": genes, "skill_genes": skill_genes,
            "skill_block_tokens": tokens,
            "skill_block_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "token_counter": counter.identity}


def _require_count(value, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise FinalStatisticsError(f"{name} must be a non-negative int, got {value!r}")


def _seeded_keys() -> list[tuple[str, int | None]]:
    return [("A", None)] + [(arm, seed) for arm in ("B", "C") for seed in MINING_SEEDS]


def _require_seeded_keys(mapping: Mapping, label: str) -> None:
    if set(mapping) != set(_seeded_keys()):
        raise FinalStatisticsError(
            f"{label} needs exactly the runs {_seeded_keys()}, got {sorted(mapping, key=repr)}")


def summarize_genome_sizes(sizes: Mapping[tuple[str, int | None], dict]) -> dict:
    """Per-arm genome sizes (D6, descriptive). ``sizes`` maps (arm, seed) to
    ``measure_genome_size`` output, with arm A once (seed None)."""
    _require_seeded_keys(sizes, "genome sizes")
    for key, entry in sizes.items():
        if not isinstance(entry, Mapping):
            raise FinalStatisticsError(f"genome size {key!r} must be a mapping")
        for name in ("genes", "skill_genes", "skill_block_tokens"):
            if name not in entry:
                raise FinalStatisticsError(f"genome size {key!r} lacks {name}")
            _require_count(entry[name], f"genome size {key!r} {name}")
        if entry["skill_genes"] > entry["genes"]:
            raise FinalStatisticsError(f"genome size {key!r} has more skill genes than genes")
        counter = entry.get("token_counter")
        if not isinstance(counter, str) or not counter.strip():
            raise FinalStatisticsError(f"genome size {key!r} lacks its token counter identity")
    arm_a = sizes[("A", None)]
    if arm_a["skill_genes"] or arm_a["skill_block_tokens"]:
        raise FinalStatisticsError("arm A is G0 unchanged and carries no skills")
    counters = {entry["token_counter"] for entry in sizes.values()}
    if len(counters) != 1:
        raise FinalStatisticsError(f"every genome size needs the same token counter, got {sorted(counters)}")
    out = {"token_counter": counters.pop(), "per_arm": {}}
    for arm in ARMS:
        keys = [("A", None)] if arm == "A" else [(arm, seed) for seed in MINING_SEEDS]
        rows = {("once" if seed is None else str(seed)): {
            "genes": sizes[(arm, seed)]["genes"],
            "skill_genes": sizes[(arm, seed)]["skill_genes"],
            "skill_block_tokens": sizes[(arm, seed)]["skill_block_tokens"]}
            for arm_, seed in keys}
        out["per_arm"][arm] = {
            "runs": rows,
            "mean_genes": _fraction(Fraction(sum(r["genes"] for r in rows.values()), len(rows))),
            "mean_skill_block_tokens": _fraction(Fraction(
                sum(r["skill_block_tokens"] for r in rows.values()), len(rows)))}
    return out


@dataclass(frozen=True)
class TransferOutcome:
    """One transfer-subset episode (D6): ALFWorld success, and the tokens
    and steps it used."""

    success: bool
    total_tokens: int
    steps: int

    def __post_init__(self):
        if not isinstance(self.success, bool):
            raise FinalStatisticsError(f"success must be a bool, got {self.success!r}")
        for name in ("total_tokens", "steps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise FinalStatisticsError(f"{name} must be a non-negative int, got {value!r}")


def _cost_summary(outcomes: list[TransferOutcome]) -> dict:
    solved = [o for o in outcomes if o.success]
    summary = {"tasks": len(outcomes), "solved": len(solved),
               "censored_unsolved": len(outcomes) - len(solved)}
    for name in ("total_tokens", "steps"):
        values = [getattr(o, name) for o in solved]
        summary[f"{name}_to_success"] = None if not values else {
            "mean": _fraction(Fraction(sum(values), len(values))),
            "median": _fraction(Fraction(statistics.median_low(values) + statistics.median_high(values), 2))}
    return summary


def summarize_adaptation_cost(transfer_task_ids, outcomes: Mapping[tuple[str, int | None],
                                                                   Mapping[str, TransferOutcome]]) -> dict:
    """Descendant adaptation cost on the transfer subset (D6, descriptive):
    per run, the tasks solved and, over the solved tasks, the tokens and
    steps to success; unsolved tasks are counted as censored, never given a
    cost. Every run must cover exactly the same transfer tasks."""
    if not isinstance(transfer_task_ids, (list, tuple)):
        raise FinalStatisticsError("transfer task ids must be a list or tuple")
    ids = tuple(transfer_task_ids)
    if not ids or any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
        raise FinalStatisticsError("transfer task ids must be distinct non-empty strings")
    _require_seeded_keys(outcomes, "adaptation cost")
    out = {"transfer_task_count": len(ids), "per_arm": {}}
    for arm in ARMS:
        keys = [("A", None)] if arm == "A" else [(arm, seed) for seed in MINING_SEEDS]
        rows, pooled = {}, []
        for _, seed in keys:
            run = outcomes[(arm, seed)]
            if not isinstance(run, Mapping):
                raise FinalStatisticsError(f"arm {arm} run {seed} must be a mapping")
            if set(run) != set(ids):
                raise FinalStatisticsError(
                    f"arm {arm} run {seed} must cover exactly the transfer tasks")
            for task_id, outcome in run.items():
                if not isinstance(outcome, TransferOutcome):
                    raise FinalStatisticsError(
                        f"arm {arm} run {seed} task {task_id!r} needs a TransferOutcome")
            row = [run[t] for t in sorted(ids)]
            rows["once" if seed is None else str(seed)] = _cost_summary(row)
            pooled.extend(row)
        out["per_arm"][arm] = {"runs": rows, "pooled": _cost_summary(pooled)}
    return out


__all__ = [
    "ARMS", "MINING_SEEDS", "FINAL_BOOTSTRAP_SEED", "FINAL_BOOTSTRAP_RESAMPLES",
    "FINAL_BOOTSTRAP_CONFIDENCE", "FINAL_CLUSTER_ORDER", "H2_MIN_DELTA",
    "MIN_CONSISTENT_SEEDS", "FINAL_RULE", "CLAIM_STATUS", "FinalStatisticsError",
    "ArmRun", "FinalEvaluationInput", "ExactBootstrap", "exact_cluster_bootstrap",
    "analyze_final_evaluation", "measure_genome_size", "summarize_genome_sizes",
    "TransferOutcome", "summarize_adaptation_cost",
]
