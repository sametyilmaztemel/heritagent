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
    # the per-cell composition (design section 5) uses the same exact rule
    cells = grid["cells"]
    assert cells["tau_c=0.05|tau_r=0.05"]["hypothetical_eligible"] == [gene]
    assert cells["tau_c=0.05|tau_r=0.05"]["hypothetical_promote"] == [gene]
    for tau_r in ("0.02", "0.03"):
        assert cells[f"tau_c=0.05|tau_r={tau_r}"]["hypothetical_eligible"] == []


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
    """b's stage 2 meets alpha at look 4 with contribution 9/192 < tau_c 0.05
    (ADR-0012 D4), so b ran stage 3 and 4A (D7) but was never eligible and
    no pair evidence includes b. Under the looser tau_c 0.03 b passes on the
    same path, so the (a, b) pair evidence those cells need was never
    collected: not identifiable, with the exact missing pair list; the tau_c
    0.05 and 0.08 cells stay identifiable."""
    world = build_world(tmp_path, candidates=("a", "b"),
                         ablation_deltas={"a": 0.9}, stage2={"b": "alpha_only"},
                         interaction={"both": (True, ("a",)),
                                       "neither": (True, ())})
    orch, somatic, cig_store, runner, inputs, registry = world
    outcome = orch.run(inputs)
    # primary: only a is preliminary-eligible; interaction is vacuous
    assert outcome.verdicts == {"a": "promote", "b": "reject"}
    by_gene = {r.gene_id: r for r in outcome.results}
    assert by_gene["a"].stage4_child["measurements"]["interaction"]["vacuous"] is True
    assert by_gene["b"].stage_results["regression"] is True  # 4A ran for b (alpha met)
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
    b_grid = orch.sensitivity_recomputation("b")["ablation"]
    assert [b_grid[f"tau_c={tau}"]["pass"] for tau in (0.03, 0.05, 0.08)] == \
        [True, False, False]
    assert {b_grid[f"tau_c={tau}"]["stop_look"] for tau in (0.03, 0.05, 0.08)} == {4}
    # tau_c=0.05 / 0.08 cells: b never re-enters -> identifiable
    for tau_c in (0.05, 0.08):
        for tau_r in (0.02, 0.03, 0.05):
            cell = grid["cells"][f"tau_c={tau_c}|tau_r={tau_r}"]
            assert cell["identifiable"] is True
            assert cell["hypothetical_eligible"] == ["a"]
            assert cell["hypothetical_promote"] == ["a"]
    # tau_c=0.03 cells: b is newly eligible -> the required (a, b) pair
    # evidence is missing -> not_identifiable with the exact missing pair list
    for tau_r in (0.02, 0.03, 0.05):
        cell = grid["cells"][f"tau_c=0.03|tau_r={tau_r}"]
        assert cell["identifiable"] is False
        assert cell["not_identifiable"] is True
        assert cell["missing_interaction_pairs"] == [["a", "b"]]
        assert "a" in cell["reason"] and "b" in cell["reason"]
        assert cell["hypothetical_eligible"] == ["a", "b"]
        assert "hypothetical_promote" not in cell


def test_a_futility_stop_never_becomes_eligible_in_any_cell(tmp_path):
    """ADR-0012 D7: a candidate stopped for futility never met alpha, so its
    stage-3 and 4A evidence was not collected; no tau in either grid makes
    it eligible, and the cells stay identifiable."""
    world = build_world(tmp_path, candidates=("a", "b"),
                         ablation_deltas={"a": 0.9, "b": 0.04})
    orch, somatic, cig_store, runner, inputs, registry = world
    outcome = orch.run(inputs)
    assert outcome.verdicts == {"a": "promote", "b": "reject"}
    grid = orch.sensitivity_recomputation("b")
    assert all(cell["identifiable"] is True and cell["hypothetical_eligible"] == ["a"]
               for cell in grid["cells"].values())
    assert all(value is None for value in grid["regression"].values())
    assert all(value["pass"] is False and value["stop_look"] == 1
               for value in grid["ablation"].values())


def test_sensitivity_single_newly_eligible_candidate_identifiable(tmp_path):
    """EDGE: a candidate that becomes newly eligible under a looser cell but
    is the ONLY eligible member requires NO interaction pairs -> the cell
    is identifiable and reports the recomposed promote set."""
    # a fails replay (never eligible, never paired); b meets alpha but fails
    # the point rule at tau_c 0.05 (9/192) and passes at the looser 0.03
    world = build_world(tmp_path, candidates=("a", "b"),
                         replay_pass_for={"a": False},
                         ablation_deltas={"a": 0.9}, stage2={"b": "alpha_only"})
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


def test_a_stricter_tau_c_that_would_have_continued_is_not_identifiable(tmp_path):
    """ADR-0012 D7: a passes stage 2 at look 3 with (10, 0), point 10/144;
    tau_c 0.08 would have continued past that look, so every 0.08 cell needs
    a's unobserved outcome and is not identifiable. The 0.05 and 0.03 cells
    are."""
    world = build_world(tmp_path, candidates=("a", "b"), stage2={"a": "pass_look3"},
                        ablation_deltas={"b": 0.9})
    orch, somatic, cig_store, runner, inputs, registry = world
    outcome = orch.run(inputs)
    assert outcome.verdicts == {"a": "promote", "b": "promote"}
    assert inputs[0].ablation_evaluation.stop["look"] == 3
    grid = orch.sensitivity_recomputation("a")
    assert grid["ablation"]["tau_c=0.08"]["identifiable"] is False
    assert grid["ablation"]["tau_c=0.03"]["identifiable"] is True
    for tau_r in (0.02, 0.03, 0.05):
        strict = grid["cells"][f"tau_c=0.08|tau_r={tau_r}"]
        assert strict["identifiable"] is False and strict["not_identifiable"] is True
        assert strict["unidentified_stage2"] == ["a"] and "hypothetical_promote" not in strict
        for tau_c in (0.03, 0.05):
            cell = grid["cells"][f"tau_c={tau_c}|tau_r={tau_r}"]
            assert cell["identifiable"] is True
            assert cell["hypothetical_promote"] == ["a", "b"]


def test_an_ineligible_candidate_does_not_make_a_cell_unidentifiable(tmp_path):
    """The same a, failing stage 1: whatever stage 2 would have done under
    tau_c 0.08, a is not eligible, so the 0.08 cells stay identifiable."""
    world = build_world(tmp_path, candidates=("a", "b"), stage2={"a": "pass_look3"},
                        ablation_deltas={"b": 0.9}, replay_pass_for={"a": False})
    orch, somatic, cig_store, runner, inputs, registry = world
    orch.run(inputs)
    grid = orch.sensitivity_recomputation("b")
    assert grid["ablation"]["tau_c=0.08"]["identifiable"] is True  # b's own path
    for tau_r in (0.02, 0.03, 0.05):
        cell = grid["cells"][f"tau_c=0.08|tau_r={tau_r}"]
        assert cell["identifiable"] is True and cell["hypothetical_promote"] == ["b"]
