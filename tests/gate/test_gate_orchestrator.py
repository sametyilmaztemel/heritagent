"""CIG gate orchestrator tests (issue #12 criteria 14-16) — fully scripted,
no GPU/live model/ALFWorld."""

import json

import pytest

from genome.validation.errors import RecordConsistencyError
from genome.validation.loader import load_cig
from genome.validation.registry import TraitRegistry
from genome.validation.records import CigRecordStore
from inheritance.gate import (
    CigGateOrchestrator,
    GateCandidateInput,
    GateInputError,
)
from runtime.loop import Budgets
from somatic.store import SomaticStore
from tests.gate.conftest import interaction_tasks
from tests.gate.helpers import (
    BUDGETS,
    MINING_SEED,
    PARENT_CIG,
    ScriptedGateRunner,
    coherent_stage11_children,
    mint_envelope,
)


def build_world(tmp_path, *, candidates=("a", "b"), ranks=None,
                 replay_pass=True, ablation_deltas=None, gen_in_with=True,
                 gen_out_with=True, treg_success_count=40,
                 interaction=None, replay_pass_for=None,
                 commit_stores=True):
    """Build a full scripted world: runner, inputs, orchestrator, stores.

    Every candidate carries its OWN aggregate CIG id (CIG-0007, CIG-0008,
    ...); multi-candidate inputs can never share one parent.
    Generalization defaults: with-trait all succeed, without-trait all
    succeed -> delta 0 (pass both scopes). Interaction defaults: both pass,
    neither pass -> delta 0 (not harmful).
    ablation_deltas: optional per-name ablation contribution.
    commit_stores=False wires the orchestrator WITHOUT stores so a run
    produces results without committing (candidates stay 'candidate')."""
    from inheritance.generalization import GeneralizationAllocation
    from inheritance.regression import RegressionAllocation, TRegTask
    from inheritance.interaction import InteractionAllocation
    from inheritance.runner import EvaluationTask

    registry = TraitRegistry(tmp_path / "registry")
    runner = ScriptedGateRunner()
    inputs = []
    ranks = ranks or {name: i for i, name in enumerate(candidates)}
    ablation_deltas = ablation_deltas or {}
    store = SomaticStore.create(tmp_path / "somatic.jsonl", registry)
    cig_store = CigRecordStore()

    for index, name in enumerate(candidates):
        parent = "CIG-" + format(7 + index, "04d")
        envelope = mint_envelope(registry, name)
        store.add_candidate(envelope)  # real flow: candidate persisted by #9/#10
        gene_id = envelope["candidate"]["gene_id"]
        delta = ablation_deltas.get(name, 0.5)
        rp = replay_pass
        if replay_pass_for is not None and name in replay_pass_for:
            rp = replay_pass_for[name]
        replay, ablation = coherent_stage11_children(
            gene_id, parent=parent, replay_pass=rp, ablation_pass=delta >= 0.05,
            ablation_delta=delta, envelope=envelope,
            base_genome_id=g0_config(registry).genome_id)
        # generalization: with/without per scope
        for scope, w_ok in (("in", gen_in_with), ("out", gen_out_with)):
            for i in range(10):
                tid = f"{scope}-{name}-{i}"
                runner.set(tid, "with_trait", w_ok, (gene_id,))
                runner.set(tid, "without_trait", True, ())
        # regression: first treg_success_count succeed; G0 without the trait
        # solves every T_reg task in the paired run (ADR-0009 D1), so the
        # paired drop is the old absolute one
        for i in range(40):
            runner.set(f"treg-{name}-{i}", "with_trait",
                        i < treg_success_count, (gene_id,))
            runner.set(f"treg-{name}-{i}", "without_trait", True, ())
        # interaction: both/neither outcomes shared per pair; explicit
        # dict form allows without-arm contamination simulation
        interaction = interaction or {"both": (True, (gene_id,)),
                                       "neither": (True, ())}
        both = interaction["both"]
        neither = interaction["neither"]
        for i in range(10):
            tid = f"ix-{i}"
            runner.set(tid, "both", both[0], tuple(both[1]))
            runner.set(tid, "neither", neither[0], tuple(neither[1]))
        in_scope = tuple(EvaluationTask(
            task_id=f"in-{name}-{i}", family="heat_and_place",
            payload={"goal": "g"},
            bank_id="generalization_bank[heat_and_place]") for i in range(10))
        out_scope = tuple(EvaluationTask(
            task_id=f"out-{name}-{i}", family="clean_surface",
            payload={"goal": "g"},
            bank_id="generalization_bank[clean_surface]") for i in range(10))
        gen_alloc = GeneralizationAllocation(
            in_scope_tasks=in_scope, out_scope_tasks=out_scope,
            frozen_in_scope_plan={"heat_and_place": 10},
            frozen_out_scope_plan={"clean_surface": 10},
            declared_frozen_applicability=("heat_and_place",))
        treg = RegressionAllocation(tasks=tuple(
            TRegTask(task_id=f"treg-{name}-{i}", family="heat_and_place",
                      baseline_trajectory_id="T-" + format(i, "04d") + "aaaa")
            for i in range(40)))
        inputs.append(GateCandidateInput(
            candidate_envelope=envelope, frozen_rank=ranks[name],
            replay_evaluation=replay, ablation_evaluation=ablation,
            generalization_allocation=gen_alloc,
            regression_allocation=treg,
            parent_cig_id=parent))

    interaction_alloc = InteractionAllocation(tasks=interaction_tasks(10))
    counter = _WordCounter()
    orch = CigGateOrchestrator(
        registry=registry, base_config=g0_config(registry), runner=runner,
        budgets=BUDGETS, parent_cig_id=PARENT_CIG, mining_seed=MINING_SEED,
        token_counter=counter, interaction_allocation=interaction_alloc,
        somatic_store=store if commit_stores else None,
        cig_record_store=cig_store if commit_stores else None)
    return orch, store, cig_store, runner, inputs, registry


def g0_config(registry):
    from runtime.seed_g0 import g0_runtime_config
    return g0_runtime_config(registry)


class _WordCounter:
    identity = "scripted-word-counter-v1"

    def count_tokens(self, text: str) -> int:
        return len(text.split())


def assert_live_bundle_exposed(cig_store, result):
    """The live CigRecordStore holds the aggregate + all five children of
    this candidate, and the whole store is consistent."""
    agg_id = result.aggregate_cig_id
    assert agg_id in cig_store.aggregates
    stored_agg = cig_store.aggregates[agg_id]
    assert stored_agg["candidate_gene_id"] == result.gene_id
    assert sorted(stored_agg["child_records"]) == \
        sorted(result.aggregate["child_records"])
    assert len(stored_agg["child_records"]) == 5
    for child_id in stored_agg["child_records"]:
        assert child_id in cig_store.children
    cig_store.require_consistent()


def test_single_candidate_promotes_when_all_stages_pass(tmp_path):
    world = build_world(tmp_path, candidates=("a",))
    orch, somatic, cig_store, runner, inputs, registry = world
    outcome = orch.run(inputs)

    gene = inputs[0].candidate_envelope["candidate"]["gene_id"]
    result = outcome.results[0]
    assert result.verdict == "promote"
    # aggregate has exactly the five coherent stage children
    aggregate = result.aggregate
    assert len(aggregate["child_records"]) == 5
    assert set(aggregate["stages"]) == {
        "replay", "ablation", "generalization_in_scope",
        "generalization_out_of_scope", "interaction_regression"}
    # interaction vacuously passed (fewer than 2 eligible candidates) with
    # an explicit reason
    stage4 = result.stage4_child["measurements"]
    assert stage4["interaction"]["pair_count"] == 0
    assert stage4["interaction"]["vacuous"] is True
    assert stage4["interaction"]["reason"]
    # atomic commit: somatic terminal decision = validated
    assert somatic.state(gene, 1) == "validated"
    # gate_reports references the AGGREGATE id only
    stored = somatic.get(gene, 1)
    assert stored["validation"]["gate_reports"] == [PARENT_CIG]
    # origin/artifact/provenance unchanged; no assimilation
    assert stored["candidate"]["origin"] == "acquired"
    assert "cig_record" not in stored["candidate"]["provenance"]
    # live CIG store: aggregate + five children exposed and consistent
    assert_live_bundle_exposed(cig_store, result)
    # the stage-4 child persists the exact regression generation settings
    reg = stage4["regression"]
    assert reg["eval_generation_settings"] == {
        "temperature": 0.0, "max_tokens": 64,
        "seed": MINING_SEED, "stop": ["\nObservation:"]}  # ADR-0003 A2


def test_interaction_two_eligible_twenty_calls_shared_evidence(tmp_path):
    world = build_world(tmp_path, candidates=("a", "b"),
                         interaction={"both": (True, ("a", "b")),
                                       "neither": (True, ())})
    orch, somatic, cig_store, runner, inputs, registry = world
    outcome = orch.run(inputs)
    # 2 eligible -> exactly one unordered pair: 10 tasks x 2 conditions = 20 calls
    ix_calls = [c for c in runner.calls if c[0].startswith("ix-")]
    assert len(ix_calls) == 20
    assert len(set(ix_calls)) == 20
    # both candidates promote (delta 0, not harmful)
    assert set(outcome.verdicts.values()) == {"promote"}
    # both stage-4 children reference the SAME pair evidence hash AND embed
    # the IDENTICAL canonical compact evidence record
    children = [r.stage4_child for r in outcome.results]
    hashes = [c["measurements"]["interaction"]["pair_evidence"][0]
               ["evidence_sha256"] for c in children]
    assert hashes[0] == hashes[1]
    canonical = [c["measurements"]["interaction"]["pair_evidence"][0]
                  ["canonical_evidence"] for c in children]
    assert canonical[0] == canonical[1]
    # canonical gene-id order (not caller order)
    assert canonical[0]["gene_ids"] == ["a", "b"]


def test_harmful_pair_eliminates_lower_contribution(tmp_path):
    """a: strong ablation (0.9); b: weak (0.1). Harmful interaction -> b
    eliminated, a retained -> a promotes, b rejects."""
    world = build_world(tmp_path, candidates=("a", "b"),
                         ablation_deltas={"a": 0.9, "b": 0.1},
                         interaction={"both": (False, ()),
                                       "neither": (True, ())})
    orch, somatic, cig_store, runner, inputs, registry = world
    outcome = orch.run(inputs)
    verdicts = outcome.verdicts
    # a retained (higher contribution) and promotes; b eliminated ->
    # interaction stage fails for b -> reject
    assert verdicts["a"] == "promote"
    assert verdicts["b"] == "reject"
    b_result = [r for r in outcome.results if r.gene_id == "b"][0]
    assert b_result.eliminated_by == "a"
    assert b_result.stage4_child["measurements"]["component_pass_flags"][
        "not_eliminated"] is False
    # both live bundles exposed (b's reject is still a terminal CIG record)
    for result in outcome.results:
        assert_live_bundle_exposed(cig_store, result)


def test_equal_contribution_tie_uses_earlier_frozen_rank(tmp_path):
    """HARMFUL pair with EQUAL ablation contribution -> earlier frozen rank
    (b, rank 0) retained; a eliminated -> a rejects."""
    world = build_world(tmp_path, candidates=("b", "a"),  # b first in list
                         ranks={"b": 0, "a": 5},
                         ablation_deltas={"b": 0.5, "a": 0.5},
                         interaction={"both": (False, ()),
                                       "neither": (True, ())})
    orch, somatic, cig_store, runner, inputs, registry = world
    outcome = orch.run(inputs)
    a_result = [r for r in outcome.results if r.gene_id == "a"][0]
    b_result = [r for r in outcome.results if r.gene_id == "b"][0]
    # equal contribution: earlier frozen rank retained, later eliminated
    assert a_result.eliminated_by == "b"
    assert a_result.verdict == "reject"
    assert b_result.eliminated_by is None
    assert b_result.verdict == "promote"


def test_candidate_ordering_does_not_change_verdicts(tmp_path):
    world1 = build_world(tmp_path / "w1", candidates=("a", "b"))
    world2 = build_world(tmp_path / "w2", candidates=("b", "a"))
    o1 = world1[0].run(world1[4])
    o2 = world2[0].run(world2[4])
    assert o1.verdicts == o2.verdicts


def test_fewer_than_two_eligible_zero_pair_calls(tmp_path):
    """Both candidates fail ablation -> both reject, zero interaction calls,
    explicit vacuous interaction reason."""
    world = build_world(tmp_path, candidates=("a", "b"),
                         ablation_deltas={"a": 0.0, "b": 0.0})
    orch, somatic, cig_store, runner, inputs, registry = world
    outcome = orch.run(inputs)
    ix_calls = [c for c in runner.calls if c[0].startswith("ix-")]
    assert ix_calls == []
    assert set(outcome.verdicts.values()) == {"reject"}
    for r in outcome.results:
        inter = r.stage4_child["measurements"]["interaction"]
        assert inter["pair_count"] == 0
        assert inter["vacuous"] is True
        assert inter["reason"]
    # live bundles still exposed for the rejected candidates
    for result in outcome.results:
        assert_live_bundle_exposed(cig_store, result)


def test_duplicate_parent_cig_ids_rejected_before_any_episode(tmp_path):
    """Two candidates resolving to the SAME aggregate CIG id collide on the
    per-candidate aggregate identity: rejected before any stage-3/4 runner
    call."""
    world = build_world(tmp_path, candidates=("a", "b"))
    orch, somatic, cig_store, runner, inputs, registry = world
    colliding = [
        GateCandidateInput(
            candidate_envelope=inp.candidate_envelope, frozen_rank=inp.frozen_rank,
            replay_evaluation=inp.replay_evaluation,
            ablation_evaluation=inp.ablation_evaluation,
            generalization_allocation=inp.generalization_allocation,
            regression_allocation=inp.regression_allocation,
            parent_cig_id=PARENT_CIG)  # BOTH forced to the same parent
        for inp in inputs]
    calls_before = len(runner.calls)
    with pytest.raises(GateInputError, match="[Uu]NIQUE aggregate CIG ids"):
        orch.run(colliding)
    assert len(runner.calls) == calls_before


def _shared_t_reg_inputs(inputs, seed_b, base_genome_id):
    """Both candidates on candidate a's T_reg; candidate b at ``seed_b``."""
    a, b = inputs
    gid_b = b.candidate_envelope["candidate"]["gene_id"]
    replay, ablation = coherent_stage11_children(
        gid_b, parent=b.parent_cig_id, replay_pass=True, ablation_pass=True,
        seed=seed_b, envelope=b.candidate_envelope, base_genome_id=base_genome_id)
    return [a, GateCandidateInput(
        candidate_envelope=b.candidate_envelope, frozen_rank=b.frozen_rank,
        replay_evaluation=replay, ablation_evaluation=ablation,
        generalization_allocation=b.generalization_allocation,
        regression_allocation=a.regression_allocation,
        parent_cig_id=b.parent_cig_id, mining_seed=seed_b)]


@pytest.mark.parametrize("seed_b,runs", [(MINING_SEED, 1), (MINING_SEED + 1, 2)])
def test_the_shared_g0_run_is_per_mining_seed(tmp_path, seed_b, runs):
    """ADR-0009 D1: G0 plays T_reg once per mining seed at that seed, and
    every candidate of the seed is paired against that run."""
    orch, _, _, runner, inputs, registry = build_world(tmp_path, candidates=("a", "b"))
    seen = []
    play = runner.run_episode

    def recording(task, config, budgets, condition, settings=None):
        seen.append((task.task_id, condition, settings.seed))
        return play(task, config, budgets, condition, settings)

    runner.run_episode = recording
    outcome = orch.run(_shared_t_reg_inputs(inputs, seed_b, g0_config(registry).genome_id))
    g0_runs = [(tid, seed) for tid, condition, seed in seen
               if tid.startswith("treg-") and condition == "without_trait"]
    assert len(g0_runs) == 40 * runs
    assert {seed for _, seed in g0_runs} == {MINING_SEED, seed_b}
    for result, seed in zip(outcome.results, (MINING_SEED, seed_b)):
        regression = result.stage4_child["measurements"]["regression"]
        assert regression["paired_g0_run"]["eval_generation_settings"]["seed"] == seed
        assert regression["eval_generation_settings"]["seed"] == seed
        assert regression["successes_without"] == 40


def test_control_expression_of_target_aborts_no_child(tmp_path):
    """Neither-arm expression of either target = contaminated intervention:
    typed integrity error, stage abort, no completed child record."""
    world = build_world(tmp_path, candidates=("a", "b"),
                         interaction={"both": (True, ("a",)),
                                       "neither": (True, ("b",))})
    orch, somatic, cig_store, runner, inputs, registry = world
    from inheritance.runner import EvaluationIntegrityError
    with pytest.raises(EvaluationIntegrityError, match="contaminated"):
        orch.run(inputs)


def test_reject_verdicts_when_replay_fails(tmp_path):
    world = build_world(tmp_path, candidates=("a", "b"), replay_pass=False,
                         ablation_deltas={"a": 0.0, "b": 0.0})
    orch, somatic, cig_store, runner, inputs, registry = world
    outcome = orch.run(inputs)
    # later primary stages still ran (no short-circuit) and verdicts reject
    for r in outcome.results:
        assert r.verdict == "reject"
        assert r.stage_results["replay"] is False
        # stage 3 still produced evidence: both stage-3 children exist
        stages = r.aggregate["stages"]
        assert isinstance(stages["generalization_in_scope"]["pass"], bool)
        assert isinstance(stages["generalization_out_of_scope"]["pass"], bool)
        assert any(ref.endswith("-generalization_in_scope")
                   for ref in r.aggregate["child_records"])
        assert any(ref.endswith("-generalization_out_of_scope")
                   for ref in r.aggregate["child_records"])
    # somatic decisions: rejected
    for r in outcome.results:
        assert somatic.state(r.gene_id, 1) == "rejected"


def test_invalid_token_counter_fails_closed(tmp_path):
    """validate_token_counter runs at orchestrator construction: a counter
    with a blank identity or a non-integer count contract is rejected
    before any episode."""
    from inheritance.runner import RunnerContractError

    class _BlankIdentity:
        identity = "   "

        def count_tokens(self, text: str) -> int:
            return len(text.split())

    class _FloatCounter:
        identity = "float-counter"

        def count_tokens(self, text: str):
            return 1.5

    registry = TraitRegistry(tmp_path / "registry")
    common = dict(registry=registry, base_config=g0_config(registry),
                  runner=ScriptedGateRunner(), budgets=BUDGETS,
                  parent_cig_id=PARENT_CIG, mining_seed=MINING_SEED)
    with pytest.raises(RunnerContractError, match="identity"):
        CigGateOrchestrator(**common, token_counter=_BlankIdentity())
    with pytest.raises(RunnerContractError, match="non-negative integer"):
        CigGateOrchestrator(**common, token_counter=_FloatCounter())


def _replay_with_child(replay, child):
    """Rebuild the frozen ReplayEvaluation around a (possibly corrupted)
    child record."""
    return type(replay)(
        _passed=replay.passed, _successes_with=replay.successes_with,
        _successes_without=replay.successes_without,
        _delta_count=replay.delta_count, _delta_success=replay.delta_success,
        _expression_ok=replay.expression_ok, _episodes=(),
        _child_record=child, _allocation=replay.allocation)


OTHER_ARTIFACT = "registry://skills/a@1/sha256:" + "0" * 64


@pytest.mark.parametrize("stage", ["replay", "ablation"])
@pytest.mark.parametrize("key,value", [
    ("candidate_version", 2), ("candidate_version", True), ("candidate_version", 1.0),
    ("candidate_version", "1"), ("candidate_version", None),
    ("candidate_artifact", OTHER_ARTIFACT), ("candidate_artifact", None),
    ("base_genome_id", "G-" + "0" * 16), ("base_genome_id", None),
    ("candidate_version", "delete"), ("candidate_artifact", "delete"),
    ("base_genome_id", "delete"),
])
def test_stage12_children_for_another_artifact_or_genome_fail_before_any_episode(
        tmp_path, stage, key, value):
    """ADR-0010 B3: a stage 1-2 child that names another version, artifact or
    base genome, by type or by value, or lacks the field, refuses the batch
    before any episode."""
    import copy
    import dataclasses
    from inheritance.gate import EvidenceCoherenceError
    orch, _, _, runner, inputs, _ = build_world(tmp_path, candidates=("a",))
    evaluation = getattr(inputs[0], f"{stage}_evaluation")
    child = copy.deepcopy(evaluation.child_record)
    if value == "delete":
        del child["measurements"][key]
    else:
        child["measurements"][key] = value
    changed = dataclasses.replace(
        inputs[0], **{f"{stage}_evaluation": dataclasses.replace(evaluation,
                                                                 _child_record=child)})
    with pytest.raises(EvidenceCoherenceError, match=f"{stage} child {key} .*ADR-0010"):
        orch.run([changed])
    assert runner.calls == []


def test_a_gate_on_another_base_genome_refuses_the_evidence_before_any_episode(tmp_path):
    """ADR-0010 B3: the gate's own base genome is the one the children must name."""
    import dataclasses
    from inheritance.gate import EvidenceCoherenceError
    orch, _, _, runner, inputs, _ = build_world(tmp_path, candidates=("a",))
    orch._base_config = dataclasses.replace(orch._base_config, genome_id="G-" + "0" * 16)
    with pytest.raises(EvidenceCoherenceError, match="base_genome_id .*ADR-0010"):
        orch.run(inputs)
    assert runner.calls == []


def test_invalid_stage11_child_rejected_before_any_episode(tmp_path):
    """A structurally invalid #11 child (load_cig failure) and a wrong
    exact child id each fail closed BEFORE any stage-3/4 runner call."""
    world = build_world(tmp_path, candidates=("a",))
    orch, somatic, cig_store, runner, inputs, registry = world
    calls_before = len(runner.calls)

    # corrupt the replay child structurally: stage outside the CIG schema
    # enum -> load_cig must reject it before any episode
    replay = inputs[0].replay_evaluation
    broken_child = dict(replay.child_record)
    broken_child["stage"] = "bogus_stage"
    broken_replay = _replay_with_child(replay, broken_child)
    bad_input = GateCandidateInput(
        candidate_envelope=inputs[0].candidate_envelope, frozen_rank=0,
        replay_evaluation=broken_replay,
        ablation_evaluation=inputs[0].ablation_evaluation,
        generalization_allocation=inputs[0].generalization_allocation,
        regression_allocation=inputs[0].regression_allocation)
    from inheritance.gate import EvidenceCoherenceError
    with pytest.raises(EvidenceCoherenceError, match="structurally invalid"):
        orch.run([bad_input])
    # wrong exact child id
    wrong_id = dict(inputs[0].replay_evaluation.child_record)
    wrong_id["cig_id"] = PARENT_CIG + "/S11-replay-typo"
    wrong_replay = _replay_with_child(replay, wrong_id)
    bad_input2 = GateCandidateInput(
        candidate_envelope=inputs[0].candidate_envelope, frozen_rank=0,
        replay_evaluation=wrong_replay,
        ablation_evaluation=inputs[0].ablation_evaluation,
        generalization_allocation=inputs[0].generalization_allocation,
        regression_allocation=inputs[0].regression_allocation)
    with pytest.raises(EvidenceCoherenceError, match="exact"):
        orch.run([bad_input2])
    assert len(runner.calls) == calls_before


@pytest.mark.parametrize("field,bad_value", [
    ("temperature", 0.7),
    ("temperature", False),      # bool is NOT a valid 0.0 (loose equality trap)
    ("max_tokens", 32),
    ("max_tokens", 64.0),        # float is NOT a strict int
    ("seed", 99),
    ("seed", 11.0),              # float is NOT a strict int
    ("stop", ["x"]),
    ("top_p", 0.9),              # extra key: exactly the four locked keys
])
def test_stage11_settings_must_match_locked_contract(tmp_path, field,
                                                      bad_value):
    """#11 evidence settings are validated as an EXACT TYPED contract —
    exactly the four keys (temperature 0.0 finite non-bool, strict-int
    max_tokens at the gate budget, strict-int seed at the resolved mining
    seed, empty-list stop). Any key/type/value mismatch fails closed with
    zero new stage-3/4 runner calls."""
    from inheritance.gate import EvidenceCoherenceError
    world = build_world(tmp_path, candidates=("a",))
    orch, somatic, cig_store, runner, inputs, registry = world
    calls_before = len(runner.calls)
    replay = inputs[0].replay_evaluation
    child = dict(replay.child_record)
    settings = dict(child["measurements"]["eval_generation_settings"])
    settings[field] = bad_value
    child["measurements"] = dict(
        child["measurements"], eval_generation_settings=settings)
    bad_input = GateCandidateInput(
        candidate_envelope=inputs[0].candidate_envelope, frozen_rank=0,
        replay_evaluation=_replay_with_child(replay, child),
        ablation_evaluation=inputs[0].ablation_evaluation,
        generalization_allocation=inputs[0].generalization_allocation,
        regression_allocation=inputs[0].regression_allocation)
    with pytest.raises(EvidenceCoherenceError, match=field):
        orch.run([bad_input])
    assert len(runner.calls) == calls_before
