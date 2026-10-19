"""The descriptive backend-noise measures from the gate's G0 cells (ADR-0012
D8): cross-round and cross-seed disagreement, each with an exact 95 %
interval. They decide nothing."""

import pytest

from evaluation.exp0001.noise import cell_disagreement, exact_interval, gate_cells


@pytest.mark.parametrize("k,n,interval", [
    (0, 10, (0.0, 0.308497)),        # 1 - 0.025 ** (1 / 10)
    (10, 10, (0.691503, 1.0)),       # 0.025 ** (1 / 10)
    (5, 10, (0.187086, 0.812914)),
    (1, 20, (0.001265, 0.248733)),
    (0, 1, (0.0, 0.975)),
])
def test_the_clopper_pearson_interval(k, n, interval):
    lower, upper = exact_interval(k, n)
    assert lower == pytest.approx(interval[0], abs=1e-6)
    assert upper == pytest.approx(interval[1], abs=1e-6)


def test_the_interval_is_symmetric_and_contains_the_rate():
    for n in (1, 7, 48):
        for k in range(n + 1):
            lower, upper = exact_interval(k, n)
            assert 0.0 <= lower <= k / n <= upper <= 1.0
            mirror = exact_interval(n - k, n)
            assert lower == pytest.approx(1 - mirror[1], abs=1e-9)


@pytest.mark.parametrize("k,n", [(3, 2), (-1, 4), (0, 0), (True, 4), (1, False)])
def test_an_impossible_count_is_rejected(k, n):
    with pytest.raises(ValueError, match="0 <= k <= n"):
        exact_interval(k, n)


def child(stage, **measurements):
    return {"stage": stage, "measurements": measurements}


def outcome(task_id, gate_round, success):
    return {"task_id": task_id, "round": gate_round,
            "cell": {"round": gate_round, "success": success}}


SEED_11 = {"children": {
    "a/S11-ablation": child("ablation", paired_outcomes=[outcome("g1", 1, True),
                                                         outcome("g2", 1, False)]),
    "a/S11-generalization_out_of_scope": child("generalization_out_of_scope",
                                               paired_outcomes=[outcome("g1", 5, False)]),
    "a/S11-interaction_regression": child(
        "interaction_regression",
        regression={"paired_g0_run": {"episodes": [{"round": 5, "task_id": "t1",
                                                    "success": True}]}},
        interaction={"pair_evidence": [{"canonical_evidence": {"paired_outcomes": [
            {"condition": "neither", "task_id": "ix1", "success": True},
            {"condition": "both", "task_id": "ix1", "success": False}]}}]}),
    "b/S11-generalization_out_of_scope": child("generalization_out_of_scope", not_run=True),
}}
SEED_23 = {"children": {
    "c/S23-ablation": child("ablation", paired_outcomes=[outcome("g1", 1, True),
                                                         outcome("g2", 1, True)])}}


def test_every_reader_of_a_cell_is_collected():
    assert gate_cells(11, SEED_11) == {(11, 1, "g1"): True, (11, 1, "g2"): False,
                                       (11, 5, "g1"): False, (11, 5, "t1"): True,
                                       (11, 6, "ix1"): True}


def test_two_readers_of_one_cell_must_agree():
    gate = {"children": {**SEED_11["children"], "b/S11-ablation": child(
        "ablation", paired_outcomes=[outcome("g1", 1, False)])}}
    with pytest.raises(ValueError, match="R1 cell of 'g1' .seed 11. is recorded with two"):
        gate_cells(11, gate)


def test_the_cross_round_and_cross_seed_measures():
    measures = cell_disagreement({"11": SEED_11, "23": SEED_23})
    assert measures["descriptive_only"] is True and measures["cells"] == 7
    # seed 11's g1 cells of R1 and R5 disagree; no other game has two rounds
    assert measures["cross_round"]["pairs"] == 1
    assert measures["cross_round"]["disagreements"] == 1
    assert measures["cross_round"]["interval_95"] == exact_interval(1, 1)
    # each seed's earliest cell of a game: g1 agrees across the seeds, g2 does not
    assert measures["cross_seed"]["pairs"] == 2 and measures["cross_seed"]["disagreements"] == 1
    assert measures["cross_seed"]["rate"] == {"exact": "1/2", "float": 0.5}
    assert "lower references" in measures["basis"]
    assert measures["interval_note"].startswith(
        "exact (Clopper-Pearson) 95 % intervals over the pairs as if independent")


def test_no_pair_gives_no_rate():
    measures = cell_disagreement({"23": SEED_23})
    assert measures["cross_round"] == {"pairs": 0, "disagreements": 0, "rate": None,
                                       "interval_95": None}
