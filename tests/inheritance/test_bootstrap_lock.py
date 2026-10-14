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
    assert len(task_ids) == len(measurements["paired_deltas"]) == 20
    recomputed = paired_cluster_bootstrap(measurements["paired_deltas"], seed=recorded["seed"],
                                          n_resamples=recorded["n_resamples"])
    assert (recomputed.lower_bound, recomputed.upper_bound) == (
        recorded["lower_bound"], recorded["upper_bound"])
    assert result.bootstrap["cluster_order"] == "task_id"


def test_the_manifest_gate_and_code_agree():
    """pyyaml is not a test dependency: the manifest's flow mapping is read
    with a pattern."""
    root = Path(__file__).resolve().parents[2]
    text = (root / "experiments/manifests/EXP-0001-minimal-cig.yaml").read_text(encoding="utf-8")
    ablation = re.search(r"^    ablation: +\{(.*?)\}\}", text, re.MULTILINE | re.DOTALL)
    assert ablation, "EXP-0001 manifest has no ablation stage"
    matches = re.findall(r"bootstrap: \{seed: (\d+), resamples: (\d+), cluster_order: (\w+)\}",
                         ablation.group(0))
    assert len(matches) == 1, "the ablation stage must pre-register its bootstrap exactly once"
    match = matches[0]
    assert (int(match[0]), int(match[1]), match[2]) == (
        BOOTSTRAP_SEED, BOOTSTRAP_N_RESAMPLES, BOOTSTRAP_CLUSTER_ORDER)
    assert (PRIMARY_THRESHOLDS["bootstrap_seed"], PRIMARY_THRESHOLDS["bootstrap_resamples"],
            PRIMARY_THRESHOLDS["bootstrap_cluster_order"]) == (
        BOOTSTRAP_SEED, BOOTSTRAP_N_RESAMPLES, BOOTSTRAP_CLUSTER_ORDER)
