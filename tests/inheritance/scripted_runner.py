"""Scripted GateEpisodeRunner shared by replay/ablation tests — no live
model/GPU/ALFWorld. Acknowledges the locked settings by default;
acknowledge_settings=False + reported_temperature simulate contract
violations (e.g. temperature 0.7)."""

from dataclasses import dataclass, field

from runtime.model_adapters import GenerationSettings

from inheritance.runner import EpisodeOutcome


@dataclass
class ScriptedRunner:
    """Deterministic scripted runner: per (task_id, condition) outcomes,
    expressed genes, and trajectory refs; records every call (including the
    passed settings) for exact contract/ack assertions."""

    outcomes: dict = field(default_factory=dict)
    # (task_id, condition) -> (success, expressed[, trajectory_id])
    default_success: bool = False
    default_expressed: tuple = ()
    default_trajectory_id: str = "T-scripted000"
    acknowledge_settings: bool = True
    reported_temperature: float = 0.0
    calls: list = field(default_factory=list)

    def run_episode(self, task, config, budgets, condition: str,
                    settings=None) -> EpisodeOutcome:
        self.calls.append((task.task_id, condition))
        settings = settings or GenerationSettings(temperature=0.0, max_tokens=64)
        entry = self.outcomes.get((task.task_id, condition))
        if entry is None:
            success, expressed = self.default_success, self.default_expressed
            trajectory_id = self.default_trajectory_id
        else:
            success = entry[0]
            expressed = entry[1]
            trajectory_id = entry[2] if len(entry) > 2 else self.default_trajectory_id
        effective = GenerationSettings(
            temperature=(settings.temperature if self.acknowledge_settings
                          else self.reported_temperature),
            max_tokens=settings.max_tokens,
            seed=settings.seed)
        return EpisodeOutcome(
            task_id=task.task_id, condition=condition, success=success,
            expressed_gene_ids=tuple(expressed),
            status="finished" if success else "budget_exhausted",
            trajectory_id=trajectory_id, effective_settings=effective)
