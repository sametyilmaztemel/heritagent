"""Shared helpers for CIG gate orchestrator tests (issue #12; recalibrated by
ADR-0012) — fully scripted, no GPU/live model/ALFWorld."""

from runtime.loop import Budgets

from tests.inheritance.scripted_runner import ScriptedRunner

BUDGETS = Budgets(max_steps=4, max_total_retries=10, max_tokens_per_request=64)
PARENT_CIG = "CIG-0007"
MINING_SEED = 11
FAMILY = "heat_and_place"
UNIVERSE = ("clean_surface", "heat_and_place", "navigate")  # the tests' family universe
BANK_SHA = "b" * 64


def proposal(name: str, family: str = FAMILY, families=None) -> dict:
    return {
        "name": name,
        "principle": f"principle of {name}",
        "when_to_apply": "heating tasks",
        "applicability": {"task_families": list(families or [family])},
        "estimated_generality": 0.5,
    }


def mint_envelope(registry, name: str, family: str = FAMILY, families=None) -> dict:
    """Registry-bound acquired candidate envelope whose stored artifact
    carries the frozen applicability."""
    payload = proposal(name, family, families)
    uri = registry.put("skill", name, 1, payload)
    return {
        "candidate": {
            "gene_id": name, "version": 1, "type": "skill",
            "origin": "acquired", "artifact": uri,
            "provenance": {"born_generation": 1,
                           "source_trajectory": "T-mine000aaaa"},
        },
        "validation": {"state": "candidate"},
    }


def stage2_allocation(name: str, families=(FAMILY,)):
    from inheritance.ablation import ALLOCATION_SIZE, AblationAllocation
    from inheritance.runner import EvaluationTask
    m = len(families)
    tasks = tuple(EvaluationTask(task_id=f"s2-{name}-{j:03d}", family=families[j % m],
                                 payload={"goal": "g"},
                                 bank_id=f"benefit_bank[{families[j % m]}]")
                  for j in range(ALLOCATION_SIZE))
    return AblationAllocation(tasks=tasks, families=tuple(families), bank_sha256=BANK_SHA)


def out_of_scope_allocation(name: str, applicability=(FAMILY,), universe=UNIVERSE):
    from inheritance.generalization import OUT_SCOPE_COUNT, OutOfScopeAllocation
    from inheritance.runner import EvaluationTask
    omitted = [f for f in sorted(universe) if f not in applicability]
    tasks = tuple(EvaluationTask(task_id=f"out-{name}-{j:02d}", family=omitted[j % len(omitted)],
                                 payload={"goal": "g"},
                                 bank_id=f"benefit_bank[{omitted[j % len(omitted)]}]")
                  for j in range(OUT_SCOPE_COUNT)) if omitted else ()
    return OutOfScopeAllocation(tasks=tasks, applicability=tuple(applicability),
                                family_universe=tuple(universe), bank_sha256=BANK_SHA)


def stage2_pairs(shape: str, net: int = 24) -> list:
    """(with-trait success, cell success) per allocation game for a stage-2
    path: ``pass`` (efficacy at look 1 with ``net`` rescues, contribution
    net/48), ``futility`` (net 0 at look 1, alpha never met) or
    ``alpha_only`` ((9, 0) at look 4: alpha met, the point rule fails)."""
    pairs = [(False, False)] * 192
    if shape == "pass":
        assert net >= 11, "efficacy at look 1 needs a net of at least 11"
        for index in range(net):
            pairs[index] = (True, False)
    elif shape == "alpha_only":
        for index in (0, 1, 2, 50, 60, 100, 110, 150, 160):
            pairs[index] = (True, False)
    elif shape != "futility":
        raise ValueError(shape)
    return pairs


def stage12_evidence(registry, base_config, envelope, *, parent: str = PARENT_CIG,
                     seed: int = MINING_SEED, replay_pass: bool = True,
                     stage2: str = "pass", net: int = 24, families=(FAMILY,)):
    """Real stage-1 and stage-2 evaluations (ADR-0012 D3, D4) of one
    candidate, driven by scripted outcomes: the evidence rounds R1-R4 hand
    to the orchestrator."""
    from inheritance.ablation import AblationEvaluator
    from inheritance.replay import ReplayAllocation, ReplayEvaluator
    from inheritance.runner import EvaluationTask
    gene_id = envelope["candidate"]["gene_id"]
    runner = ScriptedGateRunner()
    source = EvaluationTask(task_id=f"src-{gene_id}", family=families[0], payload={"goal": "g"},
                            bank_id="t_mine")
    # a failing stage 1 is a successful replay that never expressed the gene
    runner.set(source.task_id, "with_trait", True, (gene_id,) if replay_pass else ())
    common = dict(candidate_envelope=envelope, registry=registry, base_config=base_config,
                  budgets=BUDGETS, parent_cig_id=parent, mining_seed=seed)
    replay = ReplayEvaluator(runner=runner, **common).evaluate(ReplayAllocation(source))
    allocation = stage2_allocation(gene_id, families)
    for task, (with_success, cell_success) in zip(allocation.tasks, stage2_pairs(stage2, net)):
        runner.set(task.task_id, "with_trait", with_success, (gene_id,) if with_success else ())
        runner.set(task.task_id, "without_trait", cell_success, ())
    evaluator = AblationEvaluator(allocation=allocation, **common)
    while not evaluator.stopped:
        evaluator.play_look(evaluator.next_look, runner)
    return replay, evaluator.evaluation()


class ScriptedGateRunner(ScriptedRunner):
    """GateEpisodeRunner covering ALL CIG stages; outcomes keyed by
    (task_id, condition). set() stores success/expressed/trajectory."""

    def set(self, task_id: str, condition: str, success: bool,
            expressed=(), trajectory_id: str | None = None):
        self.outcomes[(task_id, condition)] = (
            success, tuple(expressed),
            trajectory_id or f"T-{task_id.replace('_', '-')}-{condition[:2]}")


def interaction_tasks(count: int = 10):
    from inheritance.runner import EvaluationTask
    return tuple(
        EvaluationTask(task_id=f"ix-{i}", family=FAMILY,
                       payload={"goal": "ix"}, bank_id="interaction_bank")
        for i in range(count))
