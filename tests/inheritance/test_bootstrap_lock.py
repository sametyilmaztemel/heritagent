"""The ablation bootstrap is pre-registered (ADR-0005, owner decision
2026-10-14): seed 0, 10,000 resamples, and the paired deltas ordered by
task_id before resampling. Near the tau_c boundary the seed and the delta
order could otherwise decide the stage verdict, so neither is free."""

import json
import re
from pathlib import Path

import pytest

from inheritance.ablation import (
    BOOTSTRAP_CLUSTER_ORDER,
    BOOTSTRAP_N_RESAMPLES,
    BOOTSTRAP_SEED,
    AblationAllocation,
)
from inheritance.bootstrap import paired_cluster_bootstrap
from inheritance.gate import PRIMARY_THRESHOLDS
from inheritance.overlay import EvaluationSetupError
from tests.inheritance.conftest import GENE_ID
from tests.inheritance.test_ablation import FROZEN_STRATA, make_allocation, make_evaluator

# found by search: point estimate 0.25 >= tau_c, and with seed 0 the
# allocation order gives a lower bound of 0.0 (stage fails) while task_id
# order gives 0.05 (stage passes): the order alone decides the verdict
WITH = {2, 5, 7, 9, 11, 14, 15, 19}
WITHOUT = {2, 3, 9}


def outcomes():
    table = {}
    for i in range(20):
        table[(f"abl-{i}", "with_trait")] = (i in WITH, (GENE_ID,))
        table[(f"abl-{i}", "without_trait")] = (i in WITHOUT, ())
    return table


def evaluate(tmp_path, registry, base_config, envelope, allocation):
    evaluator = make_evaluator(tmp_path, registry, base_config, envelope)
    evaluator._runner.outcomes = outcomes()
    return evaluator.evaluate(allocation)


def reordered(allocation, key):
    return AblationAllocation(tasks=tuple(sorted(allocation.tasks, key=key)),
                              frozen_strata=dict(FROZEN_STRATA))


@pytest.mark.parametrize("seed", [1, 7, 12345, -1])
def test_any_other_seed_is_rejected_before_any_episode(tmp_path, registry, base_config,
                                                       envelope, seed):
    with pytest.raises(EvaluationSetupError, match="pre-registered seed 0 .ADR-0005"):
        make_evaluator(tmp_path, registry, base_config, envelope, bootstrap_seed=seed)


def test_the_verdict_does_not_depend_on_the_allocation_order(tmp_path, registry, base_config,
                                                             envelope):
    base = make_allocation(tmp_path, registry)
    results = [evaluate(tmp_path, registry, base_config, envelope, allocation)
               for allocation in (base,
                                  reordered(base, key=lambda t: t.task_id),
                                  reordered(base, key=lambda t: t.task_id[::-1]))]
    assert len({tuple(r.deltas) for r in results}) == 1
    assert len({(r.bootstrap["lower_bound"], r.bootstrap["upper_bound"]) for r in results}) == 1
    assert len({json.dumps(r.child_record["measurements"]["paired_deltas"])
                for r in results}) == 1


def test_the_unordered_deltas_would_flip_the_verdict(tmp_path, registry, base_config,
                                                     envelope):
    """Guards the test above: for this pattern the delta order alone flips
    the stage verdict when the deltas are not sorted first."""
    by_index = [float(i in WITH) - float(i in WITHOUT) for i in range(20)]
    by_task_id = [d for _, d in sorted((f"abl-{i}", d) for i, d in enumerate(by_index))]
    unsorted = paired_cluster_bootstrap(by_index, seed=BOOTSTRAP_SEED)
    locked = paired_cluster_bootstrap(by_task_id, seed=BOOTSTRAP_SEED)
    assert unsorted.point_estimate == locked.point_estimate == 0.25
    assert (unsorted.lower_bound, locked.lower_bound) == (0.0, 0.05)
    result = evaluate(tmp_path, registry, base_config, envelope,
                      make_allocation(tmp_path, registry))
    assert result.bootstrap["lower_bound"] == 0.05 and result.passed is True


def test_the_child_record_recomputes_its_bound(tmp_path, registry, base_config, envelope):
    result = evaluate(tmp_path, registry, base_config, envelope,
                      make_allocation(tmp_path, registry))
    measurements = result.child_record["measurements"]
    recorded = measurements["bootstrap"]
    assert (recorded["seed"], recorded["n_resamples"], recorded["cluster_order"]) == (
        0, 10_000, "task_id")
    task_ids = sorted(measurements["allocation"]["task_ids"])
    assert measurements["paired_delta_task_ids"] == task_ids  # each delta names its task
    assert len(task_ids) == len(measurements["paired_deltas"]) == 20
    recomputed = paired_cluster_bootstrap(measurements["paired_deltas"], seed=recorded["seed"],
                                          n_resamples=recorded["n_resamples"])
    assert (recomputed.lower_bound, recomputed.upper_bound) == (
        recorded["lower_bound"], recorded["upper_bound"])
    assert result.bootstrap["cluster_order"] == "task_id"


def manifest_bootstraps(text: str) -> tuple[list[dict], int]:
    """The ablation stage's pre-registered bootstraps, as key-value maps, and
    the count of bootstrap blocks outside the final-statistics rule. pyyaml
    is not a test dependency: the manifest's flow mapping is read with a
    pattern."""
    # YAML comments are not data: a commented-out line must not count as the
    # pre-registration (review of #63)
    text = re.sub(r"(^|[ \t])#.*", r"\1", text, flags=re.MULTILINE)
    # the ablation stage runs up to the next stage key (review of #63)
    ablation = re.search(r"^    ablation:(.*?)^    generalization:", text, re.MULTILINE | re.DOTALL)
    assert ablation, "EXP-0001 manifest has no ablation stage before generalization"

    def parse(body: str) -> dict:
        return {key.strip(): value.strip().strip('"')
                for key, value in (item.split(":", 1) for item in body.split(", "))}

    blocks = [parse(body) for body in re.findall(r"bootstrap: \{([^}]*)\}", ablation.group(1))]
    # the final-statistics rule (ADR-0006) carries its own bootstrap block
    outside_final = re.sub(r"^evaluation:.*?(?=^\S)", "", text + "\nend:\n",
                           flags=re.MULTILINE | re.DOTALL)
    return blocks, len(re.findall(r"bootstrap: \{", outside_final))


def test_the_manifest_gate_and_code_agree():
    from inheritance.bootstrap import CONFIDENCE
    root = Path(__file__).resolve().parents[2]
    text = (root / "experiments/manifests/EXP-0001-minimal-cig.yaml").read_text(encoding="utf-8")
    matches, anywhere = manifest_bootstraps(text)
    assert anywhere == 1, "no other stage pre-registers one"
    assert len(matches) == 1, "the ablation stage must pre-register its bootstrap exactly once"
    match = matches[0]
    assert (int(match["seed"]), int(match["resamples"]), match["cluster_order"]) == (
        BOOTSTRAP_SEED, BOOTSTRAP_N_RESAMPLES, BOOTSTRAP_CLUSTER_ORDER)
    # ADR-0009 D5: the method, confidence and generator are pre-registered too
    assert (match["method"], float(match["confidence"]), match["generator"],
            match["recomputed_by_gate"]) == (
        "percentile", CONFIDENCE, "random.Random(seed).randrange(n)", "true")
    assert (PRIMARY_THRESHOLDS["bootstrap_seed"], PRIMARY_THRESHOLDS["bootstrap_resamples"],
            PRIMARY_THRESHOLDS["bootstrap_cluster_order"]) == (
        BOOTSTRAP_SEED, BOOTSTRAP_N_RESAMPLES, BOOTSTRAP_CLUSTER_ORDER)


@pytest.mark.parametrize("line", [
    "# bootstrap: {seed: 0, resamples: 10000, cluster_order: task_id}",
    "      # bootstrap: {seed: 0, resamples: 10000, cluster_order: task_id}",
    "      tau_c: 0.05  # bootstrap: {seed: 0, resamples: 10000, cluster_order: task_id}",
])
def test_a_commented_out_pre_registration_does_not_count(line):
    text = f"stages:\n    ablation:\n{line}\n    generalization:\n      x: 1\n"
    assert manifest_bootstraps(text) == ([], 0)


def test_episodes_keep_the_allocation_order(tmp_path, registry, base_config, envelope):
    """Only the bootstrap input is sorted: the episodes run, and are
    recorded, in allocation order (ADR-0005; review of #63)."""
    allocation = reordered(make_allocation(tmp_path, registry), key=lambda t: t.task_id[::-1])
    result = evaluate(tmp_path, registry, base_config, envelope, allocation)
    order = [t.task_id for t in allocation.tasks]
    assert order != sorted(order)
    episodes = result.child_record["measurements"]["episodes"]
    assert [e["task_id"] for e in episodes if e["condition"] == "with_trait"] == order
    deltas = dict(zip(result.child_record["measurements"]["paired_delta_task_ids"],
                      result.child_record["measurements"]["paired_deltas"]))
    for i in range(20):
        assert deltas[f"abl-{i}"] == float(i in WITH) - float(i in WITHOUT)


def test_the_gate_aggregate_records_the_lock(tmp_path):
    """The aggregate's thresholds_used carries the pre-registered seed and
    order (ADR-0005 Recording; review of #63)."""
    from tests.gate.test_gate_orchestrator import build_world
    orchestrator, _, cig_store, _, inputs, _ = build_world(tmp_path, candidates=("a",))
    outcome = orchestrator.run(inputs)
    for aggregate in [outcome.results[0].aggregate] + list(cig_store.aggregates.values()):
        values = aggregate["thresholds_used"]["values"]
        assert (values["bootstrap_seed"], values["bootstrap_resamples"],
                values["bootstrap_cluster_order"]) == (0, 10_000, "task_id")


def test_non_string_task_ids_are_rejected_with_the_allocation(tmp_path, registry):
    """The bootstrap order is string order (ADR-0005): a non-string id fails
    when the allocation is built, not after its 40 episodes (review of #63)."""
    from inheritance.runner import EvaluationTask
    tasks = list(make_allocation(tmp_path, registry).tasks)
    first = tasks[0]
    tasks[0] = EvaluationTask(task_id=0, family=first.family,
                              payload=dict(first.payload), bank_id=first.bank_id)
    with pytest.raises(ValueError, match="must be strings"):
        AblationAllocation(tasks=tuple(tasks), frozen_strata=dict(FROZEN_STRATA))


def _set(block, key, value):
    if value is _DROP:
        block.pop(key)
    else:
        block[key] = value


_DROP = object()


@pytest.mark.parametrize("where,key,value", [
    ("child", "seed", 11),               # the mining seed the old fixture used
    ("child", "seed", False),            # False == 0: the type is checked too
    ("child", "seed", 0.0),              # 0.0 == 0
    ("child", "n_resamples", 9_999),
    ("child", "cluster_order", "allocation"),
    ("child", "cluster_order", _DROP),
    ("child", "bootstrap", _DROP),
    ("evaluation", "seed", 11),
    ("evaluation", "cluster_order", _DROP),
    ("ids", None, ["abl-01", "abl-00"]),  # not in task_id order
    ("ids", None, ["abl-00", "abl-00"]),  # not distinct
    ("ids", None, list(range(20))),       # ascending, but not strings
    ("ids", None, [1] + [f"abl-{i:02d}" for i in range(2, 21)]),  # mixed types
    ("ids", None, _DROP),
])
def test_the_gate_rejects_an_ablation_child_off_the_locked_bootstrap(tmp_path, where, key, value):
    """The aggregate records seed 0 and task_id order as the thresholds
    used, so the gate checks them in the ablation evidence before any
    stage-3/4 episode (review of #63)."""
    import copy
    import dataclasses

    from inheritance.gate import EvidenceCoherenceError
    from tests.gate.test_gate_orchestrator import build_world
    orchestrator, _, _, runner, inputs, _ = build_world(tmp_path, candidates=("a",))
    ablation = inputs[0].ablation_evaluation
    child = copy.deepcopy(ablation.child_record)
    evaluated = ablation.bootstrap
    measurements = child["measurements"]
    if where == "child" and key == "bootstrap":
        measurements.pop("bootstrap")
    elif where == "child":
        _set(measurements["bootstrap"], key, value)
    elif where == "evaluation":
        _set(evaluated, key, value)
    else:
        _set(measurements, "paired_delta_task_ids", value)
    bad = dataclasses.replace(ablation, _child_record=child, _bootstrap=evaluated)
    with pytest.raises(EvidenceCoherenceError, match="ADR-0005|bootstrap is missing"):
        orchestrator.run([dataclasses.replace(inputs[0], ablation_evaluation=bad)])
    assert runner.calls == []
