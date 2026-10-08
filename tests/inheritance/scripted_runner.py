"""Scripted GateEpisodeRunner + evaluation harness shared by replay/ablation
tests — no live model/GPU/ALFWorld."""

from dataclasses import dataclass, field

from inheritance.runner import EpisodeOutcome


@dataclass
class ScriptedRunner:
    """Deterministic scripted runner: per (task_id, condition) outcomes and
    expressed genes; records every call for exact call-count assertions."""

    outcomes: dict = field(default_factory=dict)  # (task_id, condition) -> (success, expressed)
    default_success: bool = False
    default_expressed: tuple = ()
    calls: list = field(default_factory=list)      # (task_id, condition)

    def run_episode(self, task, config, budgets, condition: str) -> EpisodeOutcome:
        self.calls.append((task.task_id, condition))
        success, expressed = self.outcomes.get((task.task_id, condition),
                                                (self.default_success,
                                                 self.default_expressed))
        return EpisodeOutcome(
            task_id=task.task_id, condition=condition, success=success,
            expressed_gene_ids=tuple(expressed),
            status="finished" if success else "budget_exhausted")
