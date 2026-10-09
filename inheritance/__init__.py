"""CIG stage evaluators (issue #11): replay (stage 1) + ablation (stage 2).

Paired intervention evidence for one somatic candidate: with-trait vs
without-trait, one deterministic execution per (task instance, condition),
eval temperature 0. Emits schema-valid child CIG records for #12 to
aggregate — never a final verdict, never a somatic terminal decision, never
germline assimilation. Adapted-prior-art infrastructure, not a novelty claim.
"""

from inheritance.ablation import (
    AblationAllocation,
    AblationEvaluation,
    AblationEvaluator,
    TAU_C,
)
from inheritance.bootstrap import BootstrapResult, paired_cluster_bootstrap, stage_passes
from inheritance.overlay import (
    EvaluationConfigs,
    EvaluationSetupError,
    build_evaluation_configs,
    evaluation_settings,
)
from inheritance.replay import (
    ReplayAllocation,
    ReplayEvaluation,
    ReplayEvaluator,
    REPLAY_INSTANCE_COUNT,
    REPLAY_MIN_DELTA_COUNT,
    REPLAY_MIN_SUCCESSES_WITH,
)
from inheritance.runner import (
    CONDITIONS,
    EpisodeOutcome,
    EvaluationTask,
    GateEpisodeRunner,
    RunnerContractError,
    validate_outcome,
)

__all__ = [
    "CONDITIONS",
    "AblationAllocation",
    "AblationEvaluation",
    "AblationEvaluator",
    "BootstrapResult",
    "EpisodeOutcome",
    "EvaluationConfigs",
    "EvaluationSetupError",
    "EvaluationTask",
    "GateEpisodeRunner",
    "ReplayAllocation",
    "ReplayEvaluation",
    "ReplayEvaluator",
    "REPLAY_INSTANCE_COUNT",
    "REPLAY_MIN_DELTA_COUNT",
    "REPLAY_MIN_SUCCESSES_WITH",
    "RunnerContractError",
    "TAU_C",
    "build_evaluation_configs",
    "evaluation_settings",
    "paired_cluster_bootstrap",
    "stage_passes",
    "validate_outcome",
]
