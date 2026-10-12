"""Shared fixtures/helpers for CIG gate orchestrator tests (issue #12) —
fully scripted, no GPU/live model/ALFWorld."""

from runtime.loop import Budgets

from tests.inheritance.scripted_runner import ScriptedRunner

BUDGETS = Budgets(max_steps=4, max_total_retries=10, max_tokens_per_request=64)
PARENT_CIG = "CIG-0007"
MINING_SEED = 11


class ScriptedGateRunner(ScriptedRunner):
    """GateEpisodeRunner covering ALL CIG stages; outcomes keyed by
    (task_id, condition). set() stores success/expressed/trajectory."""

    def set(self, task_id: str, condition: str, success: bool,
            expressed=(), trajectory_id=None):
        self.outcomes[(task_id, condition)] = (
            success, tuple(expressed),
            trajectory_id or "T-" + task_id[:8] + condition[:2])


def proposal(name, family="heat_and_place"):
    return {
        "name": name,
        "principle": "principle of " + name,
        "when_to_apply": "heating tasks",
        "applicability": {"task_families": [family]},
        "estimated_generality": 0.5,
    }


def mint_envelope(registry, name, family="heat_and_place"):
    """Registry-bound acquired candidate envelope whose stored artifact
    carries the frozen applicability."""
    payload = proposal(name, family)
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


def coherent_stage11_children(gene_id, *, replay_pass, ablation_pass,
                               ablation_delta=0.5, ablation_lb=0.2):
    """Coherent #11 replay/ablation result+child pairs (measurements agree
    with result.passed)."""
    from inheritance.ablation import AblationEvaluation
    from inheritance.replay import ReplayEvaluation
    replay_child = {
        "cig_id": PARENT_CIG + "/S" + str(MINING_SEED) + "-replay",
        "parent_cig_id": PARENT_CIG, "seed": MINING_SEED, "stage": "replay",
        "measurements": {
            "candidate_gene_id": gene_id, "stage_pass": replay_pass,
            "successes_with": 5 if replay_pass else 2,
            "delta_count": 5 if replay_pass else 1,
            "eval_generation_settings": {"temperature": 0.0, "max_tokens": 64,
                                          "seed": MINING_SEED, "stop": []},
        },
    }
    ablation_child = {
        "cig_id": PARENT_CIG + "/S" + str(MINING_SEED) + "-ablation",
        "parent_cig_id": PARENT_CIG, "seed": MINING_SEED, "stage": "ablation",
        "measurements": {
            "candidate_gene_id": gene_id, "stage_pass": ablation_pass,
            "point_estimate_delta_success": ablation_delta,
            "bootstrap": {"lower_bound": ablation_lb, "n_resamples": 10_000,
                           "seed": MINING_SEED,
                           "method": "percentile-paired-cluster-bootstrap"},
            "eval_generation_settings": {"temperature": 0.0, "max_tokens": 64,
                                          "seed": MINING_SEED, "stop": []},
        },
    }
    replay = ReplayEvaluation(
        _passed=replay_pass, _successes_with=5 if replay_pass else 2,
        _successes_without=0, _delta_count=5 if replay_pass else 1,
        _delta_success=(5 if replay_pass else 1) / 5, _expression_ok=True,
        _episodes=(), _child_record=replay_child,
        _allocation={"source_family": "heat_and_place"})
    ablation = AblationEvaluation(
        _passed=ablation_pass, _deltas=(ablation_delta,) * 20,
        _delta_success=ablation_delta,
        _bootstrap={"method": "percentile-paired-cluster-bootstrap",
                     "seed": MINING_SEED, "n_resamples": 10_000,
                     "lower_bound": ablation_lb,
                     "upper_bound": ablation_delta + 0.1},
        _episodes=(), _child_record=ablation_child,
        _allocation={"source_family": "heat_and_place"})
    return replay, ablation


def interaction_tasks(count=10):
    from inheritance.runner import EvaluationTask
    return tuple(
        EvaluationTask(task_id="ix-" + str(i), family="heat_and_place",
                        payload={"goal": "ix"}, bank_id="interaction_bank")
        for i in range(count))
