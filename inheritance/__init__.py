"""CIG stage evaluators (issue #11; recalibrated by ADR-0012): replay
(stage 1, an expression check on the source task) + ablation (stage 2, a
group-sequential exact paired sign test over the benefit bank).

Paired intervention evidence for one somatic candidate: with-trait against
G0, one deterministic execution per (seed, round, task instance,
condition), eval temperature 0; the G0 side of a round is that round's
cells (ADR-0012 D2). Emits schema-valid child CIG records for the gate to
aggregate — never a final verdict, never a somatic terminal decision, never
germline assimilation. Adapted-prior-art infrastructure, not a novelty claim.
"""

from inheritance.ablation import (
    ALPHAS,
    LOOKS,
    TAU_C,
    AblationAllocation,
    AblationEvaluation,
    AblationEvaluator,
)
from inheritance.bootstrap import BootstrapResult, paired_cluster_bootstrap, sign_test_p
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
    "ALPHAS",
    "CONDITIONS",
    "LOOKS",
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
    "RunnerContractError",
    "TAU_C",
    "build_evaluation_configs",
    "evaluation_settings",
    "paired_cluster_bootstrap",
    "sign_test_p",
    "validate_outcome",
]
