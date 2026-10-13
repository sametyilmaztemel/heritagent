"""Sensitivity recomputation + commit-atomicity tests (issue #12 criterion
15, handoff §13/§15) — pure recomputation over existing stage evidence."""

import pytest

from genome.validation.errors import RecordConsistencyError
from tests.gate.test_gate_orchestrator import build_world

SENSITIVITY_TAU_C_GRID = (0.03, 0.05, 0.08)
SENSITIVITY_TAU_R_GRID = (0.02, 0.03, 0.05)


def test_sensitivity_pure_zero_calls_primary_untouched(tmp_path):
    world = build_world(tmp_path, candidates=("a", "b"))
    orch, somatic, cig_store, runner, inputs, registry = world
    outcome = orch.run(inputs)
    calls_after_primary = len(runner.calls)
    evidence_before = [(r.gene_id, r.verdict, r.aggregate, r.stage4_child)
                        for r in outcome.results]
    states_before = {r.gene_id: somatic.state(r.gene_id, 1)
                      for r in outcome.results}

    grids = []
    for r in outcome.results:
        grids.append(orch.sensitivity_recomputation(r.gene_id))

    # ZERO additional runner/model/environment calls
    assert len(runner.calls) == calls_after_primary
    # primary children/aggregate/verdicts unchanged
    evidence_after = [(r.gene_id, r.verdict, r.aggregate, r.stage4_child)
                       for r in outcome.results]
    assert evidence_after == evidence_before
    # somatic states unchanged
    assert states_before == {r.gene_id: somatic.state(r.gene_id, 1)
                              for r in outcome.results}
    # grid shape matches ADR-0002
    for grid in grids:
        assert grid["tau_c_grid"] == list(SENSITIVITY_TAU_C_GRID)
        assert grid["tau_r_grid"] == list(SENSITIVITY_TAU_R_GRID)
        assert grid["analysis_only"] is True
        assert grid["runner_calls"] == 0
        assert grid["child_records"] is None
        assert grid["aggregate_verdicts"] is None
        assert grid["somatic_states"] is None


def test_sensitivity_hypothetical_pass_flags_flip_with_thresholds(tmp_path):
    """Alternate tau values flip the HYPOTHETICAL pass flags in the grid —
    proof the recomputation is real — while primary evidence stays frozen."""
    world = build_world(tmp_path, candidates=("a",), treg_success_count=39,
                         interaction={"both": (True, ("a",)),
                                       "neither": (True, ())})
    orch, somatic, cig_store, runner, inputs, registry = world
    outcome = orch.run(inputs)
    gene = outcome.results[0].gene_id
    grid = orch.sensitivity_recomputation(gene)

    # primary: drop = 1/40 = 0.025 -> passes tau_r=0.03 (primary), and also
    # the looser 0.05, but FAILS the stricter 0.02
    regression = grid["regression"]
    assert regression["tau_r=0.02"] is False
    assert regression["tau_r=0.03"] is True
    assert regression["tau_r=0.05"] is True
    # primary evidence unchanged
    assert outcome.results[0].verdict == 'promote'


def test_sensitivity_tau_r_cell_is_exact_at_two_failures(tmp_path):
    """38/40 is a drop of exactly 0.05: the tau_r=0.05 cell must pass it
    (binary floats give 1 - 38/40 = 0.050000000000000044 and would merge the
    0.05 cell into the 0.03 cell), while the primary tau_r=0.03 rejects."""
    world = build_world(tmp_path, candidates=("a",), treg_success_count=38,
                         interaction={"both": (True, ("a",)),
                                      "neither": (True, ())})
    orch, somatic, cig_store, runner, inputs, registry = world
    outcome = orch.run(inputs)
    gene = outcome.results[0].gene_id
    grid = orch.sensitivity_recomputation(gene)
    assert grid["regression"] == {"tau_r=0.02": False, "tau_r=0.03": False,
                                  "tau_r=0.05": True}
    assert outcome.results[0].verdict == "reject"
    assert outcome.results[0].stage4_child["measurements"]["regression"][
        "task_count"] == 40


def test_no_t_eval_dependency(tmp_path):
    """The gate/sensitivity path never reads T_eval: no task id in the run
    references T_eval and sensitivity works from stored evidence alone."""
    world = build_world(tmp_path, candidates=("a",))
    orch, somatic, cig_store, runner, inputs, registry = world
    outcome = orch.run(inputs)
    task_ids = [tid for tid, _ in runner.calls]
    assert all("t_eval" not in tid.lower() for tid in task_ids)
    # sensitivity works without touching the runner again
    grid = orch.sensitivity_recomputation(gene_id=outcome.results[0].gene_id)
    assert grid["runner_calls"] == 0


def test_sensitivity_per_cell_composition_identifiable(tmp_path):
    """Every (tau_c, tau_r) cell recomposes the hypothetical eligible set;
    cells whose eligible set adds no NEW candidates are identifiable and
    report the recomputed elimination + promote composition from the
    existing pair evidence."""
    world = build_world(tmp_path, candidates=("a", "b"),
                         ablation_deltas={"a": 0.9, "b": 0.1},
                         interaction={"both": (False, ()),
                                       "neither": (True, ())})
    orch, somatic, cig_store, runner, inputs, registry = world
    outcome = orch.run(inputs)
    calls_after_primary = len(runner.calls)
    grid = orch.sensitivity_recomputation("a")
    assert len(runner.calls) == calls_after_primary
    # primary: a promotes, b eliminated by the harmful pair
    assert outcome.verdicts == {"a": "promote", "b": "reject"}
    assert grid["primary_eligible"] == ["a", "b"]
    assert len(grid["cells"]) == 9  # 3 tau_c x 3 tau_r
    # primary cell (tau_c=0.05, tau_r=0.03): identical composition to the
    # primary run — b eliminated, a promotes
    cell = grid["cells"]["tau_c=0.05|tau_r=0.03"]
    assert cell["identifiable"] is True
    assert cell["not_identifiable"] is False
    assert cell["hypothetical_eligible"] == ["a", "b"]
    assert cell["eliminated_by_interaction"] == ["b"]
    assert cell["hypothetical_promote"] == ["a"]
    # stricter tau_r=0.02 (drop = 0 <= 0.02 still passes for both) — no new
    # candidates -> still identifiable
    cell_strict = grid["cells"]["tau_c=0.05|tau_r=0.02"]
    assert cell_strict["identifiable"] is True
    # stricter tau_c=0.03: b (delta 0.1 >= 0.03, lb > 0) stays eligible —
    # same set -> identifiable, same resolution
    cell_loose_c = grid["cells"]["tau_c=0.03|tau_r=0.03"]
    assert cell_loose_c["identifiable"] is True
    assert cell_loose_c["eliminated_by_interaction"] == ["b"]


def test_sensitivity_not_identifiable_cell_marks_missing_pairs(tmp_path):
    """b fails PRIMARY ablation (delta 0.04 < tau_c=0.05) so no pair
    evidence ever includes b; the looser tau_c=0.03 grid cells would make b
    newly preliminary-eligible, so the (a, b) pair evidence REQUIRED by
    those cells was never collected -> not_identifiable with the EXACT
    missing pair list, while the tau_c in {0.05, 0.08} cells stay
    identifiable."""
    world = build_world(tmp_path, candidates=("a", "b"),
                         ablation_deltas={"a": 0.9, "b": 0.04},
                         interaction={"both": (True, ("a",)),
                                       "neither": (True, ())})
    orch, somatic, cig_store, runner, inputs, registry = world
    outcome = orch.run(inputs)
    # primary: only a is preliminary-eligible; interaction is vacuous
    assert outcome.verdicts == {"a": "promote", "b": "reject"}
    assert outcome.results[0].stage4_child["measurements"]["interaction"][
        "vacuous"] is True
    calls_after_primary = len(runner.calls)
    evidence_before = [(r.gene_id, r.verdict, r.aggregate, r.stage4_child)
                        for r in outcome.results]
    states_before = {r.gene_id: somatic.state(r.gene_id, 1)
                      for r in outcome.results}
    grid = orch.sensitivity_recomputation("a")
    # zero additional runner calls; primary evidence + stores unchanged
    assert len(runner.calls) == calls_after_primary
    assert [(r.gene_id, r.verdict, r.aggregate, r.stage4_child)
             for r in outcome.results] == evidence_before
    assert {r.gene_id: somatic.state(r.gene_id, 1)
             for r in outcome.results} == states_before
    assert grid["primary_eligible"] == ["a"]
    # tau_c=0.05 / 0.08 cells: b never re-enters -> identifiable
    for tau_c in (0.05, 0.08):
        for tau_r in (0.02, 0.03, 0.05):
            cell = grid["cells"][f"tau_c={tau_c}|tau_r={tau_r}"]
            assert cell["identifiable"] is True
            assert cell["hypothetical_eligible"] == ["a"]
            assert cell["hypothetical_promote"] == ["a"]
    # tau_c=0.03 cells: b (0.04 >= 0.03, lb > 0) is newly eligible -> the
    # required (a, b) pair evidence is missing -> not_identifiable with the
    # exact missing pair list
    for tau_r in (0.02, 0.03, 0.05):
        cell = grid["cells"][f"tau_c=0.03|tau_r={tau_r}"]
        assert cell["identifiable"] is False
        assert cell["not_identifiable"] is True
        assert cell["missing_interaction_pairs"] == [["a", "b"]]
        assert "a" in cell["reason"] and "b" in cell["reason"]
        assert cell["hypothetical_eligible"] == ["a", "b"]
        assert "hypothetical_promote" not in cell


def test_sensitivity_single_newly_eligible_candidate_identifiable(tmp_path):
    """EDGE: a candidate that becomes newly eligible under a looser cell but
    is the ONLY eligible member requires NO interaction pairs -> the cell
    is identifiable and reports the recomposed promote set."""
    # a fails replay (never eligible, never paired); b fails primary
    # ablation (0.04 < 0.05) but passes the looser tau_c=0.03
    world = build_world(tmp_path, candidates=("a", "b"),
                         replay_pass_for={"a": False},
                         ablation_deltas={"a": 0.9, "b": 0.04})
    orch, somatic, cig_store, runner, inputs, registry = world
    outcome = orch.run(inputs)
    # primary: NEITHER candidate eligible -> zero interaction episodes
    assert outcome.verdicts == {"a": "reject", "b": "reject"}
    assert [c for c in runner.calls if c[0].startswith("ix-")] == []
    calls_after_primary = len(runner.calls)
    grid = orch.sensitivity_recomputation("b")
    assert len(runner.calls) == calls_after_primary
    cell = grid["cells"]["tau_c=0.03|tau_r=0.03"]
    # b alone: no unordered pairs required -> identifiable despite being
    # newly eligible
    assert cell["identifiable"] is True
    assert cell["not_identifiable"] is False
    assert cell["hypothetical_eligible"] == ["b"]
    assert cell["eliminated_by_interaction"] == []
    assert cell["hypothetical_promote"] == ["b"]
    # primary evidence/stores untouched
    assert outcome.verdicts == {"a": "reject", "b": "reject"}
    assert {r.gene_id: somatic.state(r.gene_id, 1)
             for r in outcome.results} == {"a": "rejected", "b": "rejected"}
