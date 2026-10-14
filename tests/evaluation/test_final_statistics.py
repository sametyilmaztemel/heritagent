"""EXP-0001 final-evaluation statistics (issue #13 Phase A; ADR-0006, owner
decisions D1-D7). Synthetic outcomes only: no episode is run."""

import json
import random
import re
from fractions import Fraction
from pathlib import Path

import pytest

from evaluation.metrics.final_statistics import (
    CLAIM_STATUS,
    FINAL_RULE,
    MINING_SEEDS,
    ArmRun,
    FinalEvaluationInput,
    FinalStatisticsError,
    TransferOutcome,
    analyze_final_evaluation,
    exact_cluster_bootstrap,
    measure_genome_size,
    summarize_adaptation_cost,
    summarize_genome_sizes,
)
from inheritance.bootstrap import paired_cluster_bootstrap
from runtime.seed_g0 import build_g0_seed_genome
from tests.conftest import registry  # noqa: F401 (fixture)
from tests.runtime.conftest import seed_with_skill

T_EVAL = tuple(f"eval-{i:03d}" for i in range(134))
T_REG = tuple(f"reg-{j:02d}" for j in range(40))


def make_data(*, a_eval=lambda i: False, b_eval=lambda s, i: i < 40,
              c_eval=lambda s, i: i < 60, a_reg=lambda t: True,
              b_reg=lambda s, j: j >= 6, c_reg=lambda s, j: j >= 1,
              complete=True, reason=None, drop=()):
    """Arm A once (D7); B and C per seed; one T_reg set for every seed (D8).
    The defaults give C-B = 20/134 in every seed and reg(C)-reg(B) = -5/40."""
    runs = [ArmRun("A", None, {t: a_eval(i) for i, t in enumerate(T_EVAL)},
                   {t: a_reg(t) for t in T_REG})]
    for seed in MINING_SEEDS:
        for arm, ev, rg in (("B", b_eval, b_reg), ("C", c_eval, c_reg)):
            runs.append(ArmRun(arm, seed, {t: ev(seed, i) for i, t in enumerate(T_EVAL)},
                               {t: rg(seed, j) for j, t in enumerate(T_REG)}))
    for arm, seed, task in drop:  # remove recorded cells
        run = next(r for r in runs if (r.arm, r.seed) == (arm, seed))
        runs[runs.index(run)] = ArmRun(arm, seed,
                                       {k: v for k, v in run.t_eval.items() if k != task},
                                       {k: v for k, v in run.t_reg.items() if k != task})
    return FinalEvaluationInput(T_EVAL, T_REG, tuple(runs), complete, reason)


@pytest.fixture(scope="module")
def strong():
    return analyze_final_evaluation(make_data())


@pytest.fixture
def fast(monkeypatch):
    """Rule-logic tests resample 1,000 times; the bound, recompute and
    threshold tests keep the locked 10,000."""
    import evaluation.metrics.final_statistics as module
    monkeypatch.setattr(module, "FINAL_BOOTSTRAP_RESAMPLES", 1_000)


def test_the_exact_bootstrap_is_the_locked_float_bootstrap():
    """D2: same index stream and order statistics as
    inheritance.bootstrap.paired_cluster_bootstrap, with exact sums."""
    rng = random.Random(5)
    for _ in range(40):
        n, den = rng.randint(1, 50), rng.choice([1, 2, 3, 6])
        nums = [rng.randint(-den, den) for _ in range(n)]
        exact = exact_cluster_bootstrap(nums, den, seed=0, n_resamples=400, confidence=0.95)
        floats = paired_cluster_bootstrap([x / den for x in nums], seed=0, n_resamples=400)
        for e, f in ((exact.point_estimate, floats.point_estimate),
                     (exact.lower_bound, floats.lower_bound),
                     (exact.upper_bound, floats.upper_bound)):
            assert abs(float(e) - f) < 1e-12


def test_a_strong_pattern_meets_both_rules(strong):
    assert strong["status"] == "complete"
    cb = strong["t_eval"]["contrasts"]["C-B"]
    assert cb["point_estimate"]["exact"] == "10/67"  # 20/134 in every seed
    assert Fraction(cb["lower_bound"]["exact"]) > 0
    reg = strong["t_reg"]["contrast"]["reg(C)-reg(B)"]
    assert reg["point_estimate"]["exact"] == "-1/8"  # -5/40
    assert Fraction(reg["upper_bound"]["exact"]) < 0
    assert strong["seed_consistency"]["holding"] == 3
    assert strong["h2"]["rule_met"] is True and strong["h1"]["rule_met"] is True
    assert set(strong["t_eval"]["contrasts"]) == {"C-B", "C-A", "B-A"}  # all pairwise
    ba = strong["t_eval"]["contrasts"]["B-A"]  # B wins 40 tasks over A in every seed
    assert (ba["point_estimate"]["exact"], ba["lower_bound"]["exact"],
            ba["upper_bound"]["exact"]) == ("20/67", "15/67", "51/134")
    assert strong["t_eval"]["contrasts"]["C-A"]["point_estimate"]["exact"] == "30/67"
    rates = strong["t_eval"]["success_rate"]
    assert (rates["A"]["pooled"]["exact"], rates["B"]["pooled"]["exact"],
            rates["C"]["pooled"]["exact"]) == ("0/1", "20/67", "30/67")
    assert rates["C"]["per_seed"]["23"] == {"count": 60, "recorded": 134,
                                            "rate": {"exact": "30/67", "float": 30 / 67}}


def test_the_record_states_its_rule_and_claims_nothing(strong):
    assert strong["rule"] == json.loads(json.dumps({**FINAL_RULE,
                                                    "bootstrap": dict(FINAL_RULE["bootstrap"])}))
    assert CLAIM_STATUS.startswith("pre-registered rule outcome only; not a validated result")
    assert "not a causal claim" in CLAIM_STATUS
    assert strong["claim_status"] == CLAIM_STATUS
    # "validated" appears only negated, anywhere in the record
    assert not re.search(r"(?<!not a )validated", json.dumps(strong))
    json.dumps(strong)  # serialisable
    block = strong["t_eval"]["contrasts"]["C-B"]["bootstrap"]
    assert (block["seed"], block["resamples"], block["cluster_order"], block["confidence"]) \
        == (0, 10_000, "task_id", 0.95)


def test_each_bound_recomputes_from_the_record(strong):
    for contrast in list(strong["t_eval"]["contrasts"].values()) \
            + list(strong["t_reg"]["contrast"].values()):
        assert contrast["cluster_task_ids"] == sorted(contrast["cluster_task_ids"])
        again = exact_cluster_bootstrap(contrast["cluster_numerators"],
                                        contrast["cluster_denominator"], seed=0,
                                        n_resamples=10_000, confidence=0.95)
        assert f"{again.lower_bound.numerator}/{again.lower_bound.denominator}" \
            == contrast["lower_bound"]["exact"]
        assert f"{again.upper_bound.numerator}/{again.upper_bound.denominator}" \
            == contrast["upper_bound"]["exact"]


def test_tasks_are_the_clusters_and_seeds_are_averaged(fast):
    """D1: a task C wins in one seed of three contributes 1/3."""
    record = analyze_final_evaluation(make_data(
        b_eval=lambda s, i: False, c_eval=lambda s, i: i < 30 and s == 11))
    cb = record["t_eval"]["contrasts"]["C-B"]
    assert len(cb["cluster_task_ids"]) == 134 and cb["n_cells"] == 402
    assert cb["cluster_denominator"] == 3
    by_task = dict(zip(cb["cluster_task_ids"], cb["cluster_numerators"]))
    assert by_task["eval-000"] == 1 and by_task["eval-100"] == 0
    assert cb["point_estimate"]["exact"] == "5/67"  # 30 wins over 402 cells


@pytest.mark.parametrize("wins,met", [(20, False), (21, True)])
def test_the_five_point_threshold_is_exact(wins, met):
    """20/402 < 0.05 <= 21/402: compared as fractions, never as floats."""
    split = {11: wins // 3 + (wins % 3 > 0), 23: wins // 3 + (wins % 3 > 1)}
    split[41] = wins - split[11] - split[23]
    record = analyze_final_evaluation(make_data(
        b_eval=lambda s, i: False, c_eval=lambda s, i: i < split[s]))
    assert record["t_eval"]["contrasts"]["C-B"]["point_estimate"]["exact"] \
        == str(Fraction(wins, 402))
    assert record["h2"]["success_clause"]["met"] is met


def test_the_seed_clause_needs_two_strictly_positive_seeds(fast):
    """D3: a seed where C ties B does not hold the direction."""
    one = analyze_final_evaluation(make_data(
        b_eval=lambda s, i: False, c_eval=lambda s, i: s == 11 and i < 40))
    assert one["h2"]["success_clause"]["met"] is True  # pooled 40/402
    assert [e["holds"] for e in one["seed_consistency"]["per_seed"].values()] \
        == [True, False, False]
    assert one["h2"]["seed_clause"]["met"] is False and one["h2"]["rule_met"] is False
    two = analyze_final_evaluation(make_data(
        b_eval=lambda s, i: False, c_eval=lambda s, i: s != 41 and i < 30))
    assert two["seed_consistency"]["holding"] == 2 and two["h2"]["seed_clause"]["met"] is True


def test_no_regression_difference_fails_the_regression_clause(fast):
    record = analyze_final_evaluation(make_data(c_reg=lambda s, j: j >= 6))
    reg = record["t_reg"]["contrast"]["reg(C)-reg(B)"]
    assert reg["point_estimate"]["exact"] == "0/1"
    assert record["h2"]["regression_clause"]["met"] is False
    assert record["h2"]["rule_met"] is False


def test_a_noisy_c_minus_b_above_five_points_needs_a_positive_lower_bound(fast):
    """C wins 67 tasks and loses 60: the point is 7/134 >= 0.05, the
    interval reaches below 0, so the success clause fails."""
    record = analyze_final_evaluation(make_data(
        b_eval=lambda s, i: 67 <= i < 127, c_eval=lambda s, i: i < 67))
    cb = record["t_eval"]["contrasts"]["C-B"]
    assert Fraction(cb["point_estimate"]["exact"]) >= Fraction(5, 100)
    assert Fraction(cb["lower_bound"]["exact"]) <= 0
    assert record["h2"]["success_clause"]["met"] is False


def test_a_lower_bound_of_exactly_zero_does_not_exclude_zero():
    """C wins 10 tasks and loses 3 (found by search): the point is 7/134 and,
    under the locked 10,000 resamples, the lower bound is exactly 0, which
    does not exclude 0."""
    record = analyze_final_evaluation(make_data(
        b_eval=lambda s, i: 10 <= i < 13, c_eval=lambda s, i: i < 10))
    cb = record["t_eval"]["contrasts"]["C-B"]
    assert cb["point_estimate"]["exact"] == "7/134" and cb["lower_bound"]["exact"] == "0/1"
    assert record["h2"]["success_clause"]["met"] is False


def test_a_regression_difference_whose_interval_reaches_zero_fails(fast):
    """C fails one T_reg task fewer than B: the point is -1/40, but most
    resamples miss that task, so the upper bound is 0, not below it."""
    record = analyze_final_evaluation(make_data(b_reg=lambda s, j: j >= 2,
                                                c_reg=lambda s, j: j >= 1))
    reg = record["t_reg"]["contrast"]["reg(C)-reg(B)"]
    assert reg["point_estimate"]["exact"] == "-1/40"
    assert reg["upper_bound"]["exact"] == "0/1"
    assert record["h2"]["regression_clause"]["met"] is False


def test_h1_needs_a_lower_bound_above_zero(fast):
    """C beats A on one task: the point is positive, the lower bound is 0."""
    record = analyze_final_evaluation(make_data(a_eval=lambda i: i < 59))
    ca = record["t_eval"]["contrasts"]["C-A"]
    assert Fraction(ca["point_estimate"]["exact"]) > 0
    assert ca["lower_bound"]["exact"] == "0/1"
    assert record["h1"]["rule_met"] is False


def test_regression_rate_is_the_t_reg_failure_rate(strong):
    """D4: B fails 6 of 40 T_reg tasks per seed, C 1, A none."""
    rates = strong["t_reg"]["regression_rate"]
    assert rates["B"]["pooled"]["exact"] == "3/20" and rates["C"]["pooled"]["exact"] == "1/40"
    assert rates["A"]["pooled"]["exact"] == "0/1"
    assert rates["B"]["per_seed"]["41"]["count"] == 6
    assert strong["t_reg"]["task_ids"] == sorted(T_REG)


def test_t_reg_is_one_set_shared_by_every_seed():
    """D8: one 40-task T_reg for every seed; a per-seed mapping fails closed."""
    with pytest.raises(FinalStatisticsError, match="t_reg_task_ids must be a list or tuple"):
        FinalEvaluationInput(T_EVAL, {seed: T_REG for seed in MINING_SEEDS}, (), True)


def test_the_recorded_bootstrap_block_is_what_ran(fast, strong):
    """The block states the parameters the bound was computed with."""
    block = analyze_final_evaluation(make_data())["t_reg"]["contrast"]["reg(C)-reg(B)"]["bootstrap"]
    assert block["resamples"] == 1_000
    full = strong["t_eval"]["contrasts"]["C-B"]["bootstrap"]
    assert {k: full[k] for k in FINAL_RULE["bootstrap"]} == dict(FINAL_RULE["bootstrap"])


def test_h1_needs_c_above_a(fast):
    record = analyze_final_evaluation(make_data(a_eval=lambda i: i < 60))
    assert record["t_eval"]["contrasts"]["C-A"]["point_estimate"]["exact"] == "0/1"
    assert record["h1"]["rule_met"] is False


def test_the_analysis_is_independent_of_input_order(fast):
    data = make_data()
    shuffled = FinalEvaluationInput(
        tuple(reversed(data.t_eval_task_ids)),
        tuple(reversed(data.t_reg_task_ids)),
        tuple(ArmRun(r.arm, r.seed, dict(reversed(list(r.t_eval.items()))),
                     dict(reversed(list(r.t_reg.items())))) for r in reversed(data.runs)),
        True)
    assert json.dumps(analyze_final_evaluation(shuffled), sort_keys=True) \
        == json.dumps(analyze_final_evaluation(data), sort_keys=True)


def test_an_incomplete_run_gets_no_primary_outcome():
    """D5: a run stopped by a cap reports its reason and descriptive rates
    over the recorded cells only."""
    record = analyze_final_evaluation(make_data(
        complete=False, reason="hard cap: episodes_max reached",
        drop=[("C", 41, "eval-000"), ("B", 11, "reg-00")]))
    assert record["status"] == "incomplete"
    assert record["incomplete_reason"] == "hard cap: episodes_max reached"
    for key in ("seed_consistency", "h1", "h2"):
        assert record[key] is None
    assert record["t_eval"]["contrasts"] is None and record["t_reg"]["contrast"] is None
    assert record["t_eval"]["success_rate"]["C"]["per_seed"]["41"]["recorded"] == 133


def _with(data, **changes):
    fields = {"t_eval_task_ids": data.t_eval_task_ids, "t_reg_task_ids": data.t_reg_task_ids,
              "runs": data.runs, "complete": data.complete,
              "incomplete_reason": data.incomplete_reason}
    fields.update(changes)
    return FinalEvaluationInput(**fields)


def _replace_run(data, key, **changes):
    runs = []
    for run in data.runs:
        if (run.arm, run.seed) == key:
            fields = {"arm": run.arm, "seed": run.seed, "t_eval": run.t_eval, "t_reg": run.t_reg}
            fields.update(changes)
            run = ArmRun(**fields)
        runs.append(run)
    return _with(data, runs=tuple(runs))


BASE = make_data()


@pytest.mark.parametrize("data,message", [
    (make_data(drop=[("C", 23, "eval-007")]), "missing 1 task outcomes"),
    (make_data(drop=[("A", None, "reg-03")]), "missing 1 task outcomes"),
    (_replace_run(BASE, ("B", 11), t_eval={**BASE.runs[1].t_eval, "eval-000": 1}), "must be a bool"),
    (_replace_run(BASE, ("B", 11), t_eval={**BASE.runs[1].t_eval, "eval-999": True}), "outside its set"),
    (_replace_run(BASE, ("C", 41), t_reg={**BASE.runs[-1].t_reg, "eval-000": True}), "outside its set"),
    (_with(BASE, runs=BASE.runs + (BASE.runs[1],)), "duplicate run"),
    (_with(BASE, runs=BASE.runs[:-1]), "missing runs"),
    (_with(BASE, runs=(ArmRun("A", 11, BASE.runs[0].t_eval, BASE.runs[0].t_reg),) + BASE.runs[1:]),
     "unexpected run"),
    (_with(BASE, runs=BASE.runs + (ArmRun("D", 11, {}, {}),)), "unexpected run"),
    (_with(BASE, runs=BASE.runs[:-1] + ("not a run",)), "ArmRun records"),
    (_with(BASE, t_eval_task_ids=T_EVAL[:-1]), "exactly 134 tasks"),
    (_with(BASE, t_eval_task_ids=T_EVAL[:-1] + (T_EVAL[0],)), "distinct"),
    (_with(BASE, t_reg_task_ids=T_REG[:39]), "exactly 40 tasks"),
    (_with(BASE, t_reg_task_ids=T_REG[:39] + ("eval-000",)), "disjoint"),
    (_replace_run(BASE, ("B", 11), t_eval={**BASE.runs[1].t_eval, 1: True, "zzz": True}),
     "outside its set"),  # mixed key types still fail closed
    ("not an input", "expected a FinalEvaluationInput"),
    (_with(BASE, complete=False), "must record its reason"),
    (_with(BASE, complete=False, incomplete_reason="  "), "must record its reason"),
    (_with(BASE, incomplete_reason="cap"), "carries no incomplete_reason"),
    (_with(BASE, complete=1), "complete must be a bool"),
])
def test_data_that_do_not_fit_the_analysis_fail_closed(data, message):
    with pytest.raises(FinalStatisticsError, match=message):
        analyze_final_evaluation(data)


def test_the_manifest_records_the_rule():
    """pyyaml is not a test dependency: the final_rule block is read with a
    pattern, comments stripped."""
    root = Path(__file__).resolve().parents[2]
    text = (root / "experiments/manifests/EXP-0001-minimal-cig.yaml").read_text(encoding="utf-8")
    text = re.sub(r"(^|[ \t])#.*", r"\1", text, flags=re.MULTILINE)
    # evaluation -> statistics -> final_rule, each inside its parent's block
    evaluation = re.search(r"^evaluation:[ \t]*\n((?:[ \t]+.*\n|\n)+)", text, re.MULTILINE)
    statistics = re.search(r"^  statistics:[ \t]*\n((?:    .*\n)+)", evaluation.group(1),
                           re.MULTILINE)
    assert statistics, "evaluation.statistics is missing"
    text = statistics.group(1)
    block = re.search(r"^    final_rule:[ \t]*\n((?:      .*\n)+)", text, re.MULTILINE)
    assert block, "evaluation.statistics.final_rule is missing"
    parsed = {}
    for line in block.group(1).splitlines():
        key, value = (part.strip() for part in line.split(":", 1))
        if value.startswith("{"):
            value = {k.strip(): v.strip() for k, v in
                     (item.split(":") for item in value.strip("{}").split(","))}
        parsed[key] = value
    expected = {key: (value if key != "bootstrap" else
                      {k: str(v) for k, v in value.items()})
                for key, value in FINAL_RULE.items()}
    assert parsed == expected
    assert len(re.findall(r"^\s*final_rule:", text, re.MULTILINE)) == 1


class _WordCounter:
    identity = "scripted-word-counter-v1"

    def count_tokens(self, text):
        return len(text.split())


def test_genome_size_counts_genes_and_the_full_skill_block(registry):  # noqa: F811
    g0 = measure_genome_size(build_g0_seed_genome(registry), registry, _WordCounter())
    assert (g0["genes"], g0["skill_genes"], g0["skill_block_tokens"]) == (2, 0, 0)
    gated = seed_with_skill(registry, express_when={
        "all_of": [{"metric": "consecutive_failures", "op": ">=", "value": 2}]})
    one = measure_genome_size(gated, registry, _WordCounter())
    assert (one["genes"], one["skill_genes"]) == (3, 1)
    assert one["skill_block_tokens"] > 0  # rendered whatever its regulation
    assert one["token_counter"] == "scripted-word-counter-v1"


def test_genome_sizes_are_summarised_per_arm(registry):  # noqa: F811
    g0 = measure_genome_size(build_g0_seed_genome(registry), registry, _WordCounter())
    one = measure_genome_size(seed_with_skill(registry), registry, _WordCounter())
    sizes = {("A", None): g0}
    sizes.update({("B", s): one for s in MINING_SEEDS})
    sizes.update({("C", s): (one if s == 11 else g0) for s in MINING_SEEDS})
    summary = summarize_genome_sizes(sizes)
    assert summary["per_arm"]["A"]["runs"] == {"once": {"genes": 2, "skill_genes": 0,
                                                        "skill_block_tokens": 0}}
    assert summary["per_arm"]["C"]["mean_genes"]["exact"] == "7/3"
    with pytest.raises(FinalStatisticsError, match="exactly the runs"):
        summarize_genome_sizes({k: v for k, v in sizes.items() if k != ("A", None)})
    other = {**one, "token_counter": "other-counter"}
    with pytest.raises(FinalStatisticsError, match="same token counter"):
        summarize_genome_sizes({**sizes, ("B", 11): other})


def test_adaptation_cost_counts_unsolved_tasks_as_censored():
    tasks = [f"transfer-{k:02d}" for k in range(4)]
    solved = {t: TransferOutcome(True, 100 * (k + 1), 10 * (k + 1)) for k, t in enumerate(tasks)}
    half = {t: (solved[t] if k < 2 else TransferOutcome(False, 999, 50))
            for k, t in enumerate(tasks)}
    outcomes = {("A", None): half}
    outcomes.update({(arm, s): solved for arm in ("B", "C") for s in MINING_SEEDS})
    summary = summarize_adaptation_cost(tasks, outcomes)
    a = summary["per_arm"]["A"]["runs"]["once"]
    assert (a["solved"], a["censored_unsolved"]) == (2, 2)
    assert a["total_tokens_to_success"]["mean"]["exact"] == "150/1"  # unsolved never costed
    b = summary["per_arm"]["B"]["pooled"]
    assert b["tasks"] == 12 and b["steps_to_success"]["median"]["exact"] == "25/1"
    with pytest.raises(FinalStatisticsError, match="exactly the transfer tasks"):
        summarize_adaptation_cost(tasks[:3], outcomes)
    with pytest.raises(FinalStatisticsError, match="needs a TransferOutcome"):
        summarize_adaptation_cost(tasks, {**outcomes, ("A", None): {t: True for t in tasks}})


@pytest.mark.parametrize("args,message", [
    ((1, 10, 2), "success must be a bool"),
    ((True, -1, 2), "total_tokens must be a non-negative int"),
    ((True, 10, True), "steps must be a non-negative int"),
])
def test_transfer_outcomes_are_validated(args, message):
    with pytest.raises(FinalStatisticsError, match=message):
        TransferOutcome(*args)


@pytest.mark.parametrize("build,message", [
    (lambda: ArmRun("B", 11, list(BASE.runs[1].t_eval.items()), BASE.runs[1].t_reg),
     "t_eval must be a mapping"),  # pairs could repeat a task (D5: one outcome per cell)
    (lambda: ArmRun(1, 11, {}, {}), "arm must be a str"),
    (lambda: ArmRun("B", True, {}, {}), "seed must be None or an int"),
    (lambda: ArmRun("B", "11", {}, {}), "seed must be None or an int"),
    (lambda: FinalEvaluationInput(set(T_EVAL), T_REG, BASE.runs, True), "must be a list or tuple"),
    (lambda: FinalEvaluationInput(T_EVAL, T_REG, iter(BASE.runs), True), "must be a list or tuple"),
])
def test_inputs_of_the_wrong_kind_fail_closed(build, message):
    with pytest.raises(FinalStatisticsError, match=message):
        build()


def test_the_analysis_is_isolated_from_caller_mutation():
    t_eval = dict(BASE.runs[1].t_eval)
    run = ArmRun("B", 11, t_eval, BASE.runs[1].t_reg)
    t_eval["eval-000"] = not t_eval["eval-000"]
    assert run.t_eval["eval-000"] == BASE.runs[1].t_eval["eval-000"]


class _BadCounter:
    identity = "bad-counter"

    def __init__(self, value):
        self.value = value

    def count_tokens(self, text):
        return 0 if not text else self.value


@pytest.mark.parametrize("value", [-7, True, 4.5, "9", None])
def test_an_invalid_token_count_fails_closed(registry, value):  # noqa: F811
    with pytest.raises(FinalStatisticsError, match="skill_block_tokens must be a non-negative int"):
        measure_genome_size(seed_with_skill(registry), registry, _BadCounter(value))


def test_a_counter_without_identity_fails_closed(registry):  # noqa: F811
    class NoIdentity:
        identity = ""

        def count_tokens(self, text):
            return 0
    with pytest.raises(FinalStatisticsError, match="invalid token counter"):
        measure_genome_size(build_g0_seed_genome(registry), registry, NoIdentity())


def test_genome_size_records_its_rendering(registry):  # noqa: F811
    import hashlib

    from runtime.expression import compile_runtime_config
    from runtime.skills import skill_block
    genome = seed_with_skill(registry)
    text = skill_block(compile_runtime_config(genome, registry).skills)
    size = measure_genome_size(genome, registry, _WordCounter())
    assert size["skill_block_tokens"] == len(text.split())
    assert size["skill_block_sha256"] == hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sizes(registry):  # noqa: F811
    g0 = measure_genome_size(build_g0_seed_genome(registry), registry, _WordCounter())
    one = measure_genome_size(seed_with_skill(registry), registry, _WordCounter())
    sizes = {("A", None): g0}
    sizes.update({("B", s): one for s in MINING_SEEDS})
    sizes.update({("C", s): (one if s == 11 else g0) for s in MINING_SEEDS})
    return sizes, one


def test_mean_skill_block_tokens_is_the_run_mean(registry):  # noqa: F811
    sizes, one = _sizes(registry)
    summary = summarize_genome_sizes(sizes)
    def exact(value):
        return f"{value.numerator}/{value.denominator}"
    assert summary["per_arm"]["C"]["mean_skill_block_tokens"]["exact"] \
        == exact(Fraction(one["skill_block_tokens"], 3))
    assert summary["per_arm"]["B"]["mean_skill_block_tokens"]["exact"] \
        == exact(Fraction(one["skill_block_tokens"], 1))


@pytest.mark.parametrize("key,change,message", [
    (("B", 11), {"genes": True}, "genes must be a non-negative int"),
    (("B", 11), {"skill_block_tokens": -50}, "skill_block_tokens must be a non-negative int"),
    (("B", 11), {"skill_genes": 99}, "more skill genes than genes"),
    (("B", 11), {"token_counter": ""}, "lacks its token counter identity"),
    (("A", None), {"skill_genes": 1, "genes": 3}, "arm A is G0 unchanged"),
])
def test_genome_size_entries_are_validated(registry, key, change, message):  # noqa: F811
    sizes, _ = _sizes(registry)
    sizes[key] = {**sizes[key], **change}
    with pytest.raises(FinalStatisticsError, match=message):
        summarize_genome_sizes(sizes)


def test_a_genome_size_entry_missing_a_field_fails_closed(registry):  # noqa: F811
    sizes, _ = _sizes(registry)
    sizes[("C", 23)] = {k: v for k, v in sizes[("C", 23)].items() if k != "genes"}
    with pytest.raises(FinalStatisticsError, match="lacks genes"):
        summarize_genome_sizes(sizes)


def test_the_median_is_the_middle_not_the_mean():
    tasks = [f"transfer-{k:02d}" for k in range(3)]
    skewed = {t: TransferOutcome(True, tokens, steps)
              for t, tokens, steps in zip(tasks, (10, 20, 900), (1, 2, 90))}
    outcomes = {key: skewed for key in [("A", None)] + [(a, s) for a in ("B", "C")
                                                        for s in MINING_SEEDS]}
    a = summarize_adaptation_cost(tasks, outcomes)["per_arm"]["A"]["runs"]["once"]
    assert a["total_tokens_to_success"]["median"]["exact"] == "20/1"
    assert a["total_tokens_to_success"]["mean"]["exact"] == "310/1"
    with pytest.raises(FinalStatisticsError, match="must be a list or tuple"):
        summarize_adaptation_cost(set(tasks), outcomes)
