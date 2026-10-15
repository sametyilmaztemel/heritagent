"""Shared helpers for CIG gate orchestrator tests (issue #12) — fully
scripted, no GPU/live model/ALFWorld."""

from runtime.loop import Budgets

from tests.inheritance.scripted_runner import ScriptedRunner

BUDGETS = Budgets(max_steps=4, max_total_retries=10, max_tokens_per_request=64)
PARENT_CIG = "CIG-0007"
MINING_SEED = 11


def proposal(name: str, family: str = "heat_and_place") -> dict:
    return {
        "name": name,
        "principle": f"principle of {name}",
        "when_to_apply": "heating tasks",
        "applicability": {"task_families": [family]},
        "estimated_generality": 0.5,
    }


def mint_envelope(registry, name: str, family: str = "heat_and_place") -> dict:
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


def coherent_stage11_children(gene_id: str, *, replay_pass: bool,
                               ablation_pass: bool, ablation_delta: float = 0.5,
                               parent: str = PARENT_CIG,
                               seed: int = MINING_SEED) -> tuple:
    """Coherent #11 replay/ablation result+child pairs (measurements agree
    with result.passed) under the candidate's OWN CIG run identity. The 20
    ablation deltas all equal ``ablation_delta``; the bound is the locked
    bootstrap's on them (ADR-0005; the gate recomputes it, ADR-0009 D5)."""
    from inheritance.ablation import AblationEvaluation
    from inheritance.bootstrap import paired_cluster_bootstrap, stage_passes
    from inheritance.replay import ReplayEvaluation
    task_ids = [f"abl-{i:02d}" for i in range(20)]
    deltas = [ablation_delta] * 20
    bootstrap = paired_cluster_bootstrap(deltas, seed=0, n_resamples=10_000)
    assert stage_passes(bootstrap.point_estimate, bootstrap.lower_bound) is ablation_pass
    replay_child = {
        "cig_id": f"{parent}/S{seed}-replay",
        "parent_cig_id": parent, "seed": seed, "stage": "replay",
        "measurements": {
            "candidate_gene_id": gene_id, "stage_pass": replay_pass,
            "successes_with": 5 if replay_pass else 2,
            "delta_count": 5 if replay_pass else 1,
            "eval_generation_settings": {"temperature": 0.0, "max_tokens": 64,
                                          "seed": seed, "stop": ["\nObservation:"]},
        },
    }
    ablation_child = {
        "cig_id": f"{parent}/S{seed}-ablation",
        "parent_cig_id": parent, "seed": seed, "stage": "ablation",
        "measurements": {
            "candidate_gene_id": gene_id, "stage_pass": ablation_pass,
            "point_estimate_delta_success": bootstrap.point_estimate,
            "allocation": {"task_ids": list(reversed(task_ids))},  # any allocation order
            "paired_deltas": deltas,
            # ADR-0005: the pre-registered bootstrap, not the mining seed
            "paired_delta_task_ids": task_ids,
            "bootstrap": {"lower_bound": bootstrap.lower_bound,
                          "upper_bound": bootstrap.upper_bound, "n_resamples": 10_000,
                          "seed": 0, "cluster_order": "task_id", "confidence": 0.95,
                          "method": "percentile-paired-cluster-bootstrap"},
            "eval_generation_settings": {"temperature": 0.0, "max_tokens": 64,
                                          "seed": seed, "stop": ["\nObservation:"]},
        },
    }
    replay = ReplayEvaluation(
        _passed=replay_pass, _successes_with=5 if replay_pass else 2,
        _successes_without=0, _delta_count=5 if replay_pass else 1,
        _delta_success=(5 if replay_pass else 1) / 5, _expression_ok=True,
        _episodes=(), _child_record=replay_child,
        _allocation={"source_family": "heat_and_place"})
    ablation = AblationEvaluation(
        _passed=ablation_pass, _deltas=tuple(deltas),
        _delta_success=bootstrap.point_estimate,
        _bootstrap={"method": "percentile-paired-cluster-bootstrap",
                     "seed": 0, "n_resamples": 10_000, "cluster_order": "task_id",
                     "lower_bound": bootstrap.lower_bound,
                     "upper_bound": bootstrap.upper_bound, "confidence": 0.95,
                     "point_estimate": bootstrap.point_estimate},
        _episodes=(), _child_record=ablation_child,
        _allocation={"source_family": "heat_and_place"})
    return replay, ablation


class ScriptedGateRunner(ScriptedRunner):
    """GateEpisodeRunner covering ALL CIG stages; outcomes keyed by
    (task_id, condition). set() stores success/expressed/trajectory."""

    def set(self, task_id: str, condition: str, success: bool,
            expressed=(), trajectory_id: str | None = None):
        self.outcomes[(task_id, condition)] = (
            success, tuple(expressed),
            trajectory_id or f"T-{task_id[:8]}{condition[:2]}")


def interaction_tasks(count: int = 10):
    from inheritance.runner import EvaluationTask
    return tuple(
        EvaluationTask(task_id=f"ix-{i}", family="heat_and_place",
                        payload={"goal": "ix"}, bank_id="interaction_bank")
        for i in range(count))
