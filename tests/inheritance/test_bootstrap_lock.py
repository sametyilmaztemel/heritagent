"""The stage-2 bootstrap is pre-registered (ADR-0005, owner decision
2026-10-14): seed 0, 10,000 resamples, and the paired deltas ordered by
task_id before resampling. Since ADR-0012 D4 it decides nothing (the exact
sign test does), but every stage-2 record carries it and the gate recomputes
it (ADR-0009 D5), so neither the seed nor the order is free."""

import copy
import dataclasses
import re
from pathlib import Path

import pytest

from inheritance.ablation import (
    ALLOCATION_SIZE,
    BOOTSTRAP_CLUSTER_ORDER,
    BOOTSTRAP_N_RESAMPLES,
    BOOTSTRAP_SEED,
    AblationAllocation,
)
from inheritance.bootstrap import paired_cluster_bootstrap
from inheritance.gate import PRIMARY_THRESHOLDS
from inheritance.overlay import EvaluationSetupError
from inheritance.runner import EvaluationTask
from tests.inheritance.test_ablation import (
    BANK_SHA,
    FAMILY,
    make_allocation,
    make_evaluator,
    play,
    script,
)

# found by search (issue #63): with seed 0 the index order of these 20
# deltas gives a lower bound of 0.0 and their task_id order gives 0.05
WITH = {2, 5, 7, 9, 11, 14, 15, 19}
WITHOUT = {2, 3, 9}


def reversed_ids_allocation() -> AblationAllocation:
    """An allocation whose play order is the reverse of its task_id order."""
    tasks = tuple(EvaluationTask(task_id=f"bank-{ALLOCATION_SIZE - 1 - j:03d}", family=FAMILY,
                                 payload={"goal": f"game {j}"},
                                 bank_id=f"benefit_bank[{FAMILY}]")
                  for j in range(ALLOCATION_SIZE))
    return AblationAllocation(tasks=tasks, families=(FAMILY,), bank_sha256=BANK_SHA)


def evaluate(tmp_path, registry, base_config, envelope, allocation, pairs):
    evaluator = make_evaluator(tmp_path, registry, base_config, envelope, allocation=allocation)
    script(evaluator.runner, allocation, pairs)
    return play(evaluator)


# look 1: 14 rescues and 3 losses (net 11, p > 1/2000), so the path continues;
# look 2 adds 10 rescues: efficacy, 96 games played
PAIRS = [(True, False)] * 14 + [(False, True)] * 3 + [(False, False)] * 31 + \
    [(True, False)] * 10 + [(False, False)] * 134


@pytest.mark.parametrize("seed", [1, 7, 12345, -1])
def test_any_other_seed_is_rejected_before_any_episode(tmp_path, registry, base_config,
                                                       envelope, seed):
    with pytest.raises(EvaluationSetupError, match="pre-registered seed 0 .ADR-0005"):
        make_evaluator(tmp_path, registry, base_config, envelope, bootstrap_seed=seed)


def test_the_index_order_would_change_the_interval():
    """Why the order is fixed: for this pattern the order of the deltas alone
    moves the interval."""
    by_index = [float(i in WITH) - float(i in WITHOUT) for i in range(20)]
    by_task_id = [d for _, d in sorted((f"abl-{i}", d) for i, d in enumerate(by_index))]
    unsorted = paired_cluster_bootstrap(by_index, seed=BOOTSTRAP_SEED)
    locked = paired_cluster_bootstrap(by_task_id, seed=BOOTSTRAP_SEED)
    assert unsorted.point_estimate == locked.point_estimate == 0.25
    assert (unsorted.lower_bound, locked.lower_bound) == (0.0, 0.05)


def test_the_child_record_recomputes_its_interval_in_task_id_order(tmp_path, registry,
                                                                   base_config, envelope):
    allocation = reversed_ids_allocation()
    result = evaluate(tmp_path, registry, base_config, envelope, allocation, PAIRS)
    measurements = result.child_record["measurements"]
    assert measurements["stop"]["look"] == 2 and len(measurements["paired_outcomes"]) == 96
    recorded = measurements["bootstrap"]
    assert (recorded["seed"], recorded["n_resamples"], recorded["cluster_order"],
            recorded["descriptive"]) == (0, 10_000, "task_id", True)
    played = [t.task_id for t in allocation.tasks[:96]]
    assert measurements["paired_delta_task_ids"] == sorted(played) != played
    by_task = {t.task_id: float(w) - float(c) for t, (w, c) in zip(allocation.tasks, PAIRS)}
    assert measurements["paired_deltas"] == [by_task[t] for t in sorted(played)]
    recomputed = paired_cluster_bootstrap(measurements["paired_deltas"], seed=recorded["seed"],
                                          n_resamples=recorded["n_resamples"])
    assert (recomputed.lower_bound, recomputed.upper_bound) == (
        recorded["lower_bound"], recorded["upper_bound"])
    assert result.bootstrap["cluster_order"] == "task_id"


def test_episodes_keep_the_allocation_order(tmp_path, registry, base_config, envelope):
    """Only the bootstrap input is sorted: the episodes run, and are
    recorded, in allocation order (ADR-0005; review of #63), which also
    decides the looks (ADR-0012 D4)."""
    allocation = reversed_ids_allocation()
    result = evaluate(tmp_path, registry, base_config, envelope, allocation, PAIRS)
    order = [t.task_id for t in allocation.tasks[:96]]
    outcomes = result.child_record["measurements"]["paired_outcomes"]
    assert [o["task_id"] for o in outcomes] == order


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
    assert match["role"] == "descriptive"  # ADR-0012 D4
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


def test_non_string_task_ids_are_rejected_with_the_allocation():
    """The bootstrap order is string order (ADR-0005): a non-string id fails
    when the allocation is built, not after its episodes (review of #63)."""
    tasks = list(make_allocation().tasks)
    first = tasks[0]
    tasks[0] = EvaluationTask(task_id=0, family=first.family,
                              payload=dict(first.payload), bank_id=first.bank_id)
    with pytest.raises(ValueError, match="must be strings"):
        AblationAllocation(tasks=tuple(tasks), families=(FAMILY,), bank_sha256=BANK_SHA)


_DROP = object()


@pytest.mark.parametrize("key,value", [
    ("seed", 11),               # the mining seed the old fixture used
    ("seed", False),            # False == 0: the type is checked too
    ("seed", 0.0),              # 0.0 == 0
    ("n_resamples", 9_999),
    ("cluster_order", "allocation"),
    ("cluster_order", _DROP),
    ("bootstrap", _DROP),
    ("ids", "unsorted"),        # not in task_id order
    ("ids", "duplicated"),      # not distinct
    ("ids", "ints"),            # not strings
    ("ids", _DROP),
])
def test_the_gate_rejects_an_ablation_child_off_the_locked_bootstrap(tmp_path, key, value):
    """The aggregate records seed 0 and task_id order as the thresholds
    used, so the gate checks them in the stage-2 evidence before any R5
    episode (review of #63; ADR-0009 D5)."""
    from inheritance.gate import EvidenceCoherenceError
    from tests.gate.test_gate_orchestrator import build_world
    orchestrator, _, _, runner, inputs, _ = build_world(tmp_path, candidates=("a",))
    ablation = inputs[0].ablation_evaluation
    child = copy.deepcopy(ablation.child_record)
    measurements = child["measurements"]
    ids = measurements["paired_delta_task_ids"]
    if key == "bootstrap":
        measurements.pop("bootstrap")
    elif key == "ids" and value is _DROP:
        measurements.pop("paired_delta_task_ids")
    elif key == "ids":
        measurements["paired_delta_task_ids"] = {
            "unsorted": [ids[1], ids[0]] + ids[2:],
            "duplicated": [ids[0], ids[0]] + ids[2:],
            "ints": list(range(len(ids)))}[value]
    elif value is _DROP:
        measurements["bootstrap"].pop(key)
    else:
        measurements["bootstrap"][key] = value
    bad = dataclasses.replace(ablation, _child_record=child)
    with pytest.raises(EvidenceCoherenceError,
                       match="ADR-0005|bootstrap is missing|in task_id order"):
        orchestrator.run([dataclasses.replace(inputs[0], ablation_evaluation=bad)])
    assert runner.calls == []
