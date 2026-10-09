"""Ablation evaluator (CIG stage 2; issue #11 criteria 7-11).

Exactly 20 distinct task instances from the frozen stratified ablation
allocation; one with-trait + one without-trait deterministic execution per
task (40 episodes total); paired per-task contribution
``d_i = success_with_i - success_without_i``; point estimate =
mean(d_i); deterministic 10k paired cluster bootstrap (percentile CI).

The with-trait arm preserves the trait's regulation/express_when behavior;
the without-trait arm removes the target skill and its regulation only.
Tasks are NOT filtered afterward by whether the trait fired — a non-firing
task legitimately contributes paired Δ=0 (expression is recorded as
observation only).

Pass iff: point estimate >= tau_c (0.05) AND bootstrap 95% lower bound > 0.
Emits a schema-valid ``cig/0.1`` child record
(``<parent>/S<mining_seed>-ablation``). No aggregate/verdict, no somatic
terminal decision, no germline mutation — #12 owns those.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

from genome.validation.loader import load_cig
from runtime.expression import RuntimeConfig
from runtime.loop import Budgets

from inheritance.bootstrap import paired_cluster_bootstrap, stage_passes
from inheritance.overlay import build_evaluation_configs
from inheritance.runner import (
    EvaluationIntegrityError,
    GateEpisodeRunner,
    RunnerContractError,
    validate_gate_setup,
    validate_outcome,
)

ABLATION_INSTANCE_COUNT = 20
TAU_C = 0.05
BOOTSTRAP_SEED = 0            # explicit deterministic default (recorded)
BOOTSTRAP_N_RESAMPLES = 10_000  # ADR-0002 locked
ABLATION_THRESHOLD_TEXT = ("point estimate delta_success >= 0.05 and paired "
                            "cluster bootstrap 95% lower bound > 0")


@dataclass(frozen=True)
class AblationAllocation:
    """The frozen stratified ablation allocation, validated fail-closed:
    exactly 20 distinct task instances whose observed family counts equal
    the frozen stratum plan."""

    tasks: tuple[EvaluationTask, ...]
    frozen_strata: dict[str, int] = field(default_factory=dict)

    def __post_init__(self):
        import copy as _copy
        object.__setattr__(self, "tasks", _copy.deepcopy(self.tasks))
        object.__setattr__(self, "frozen_strata", _copy.deepcopy(self.frozen_strata))
        ids = [t.task_id for t in self.tasks]
        if len(ids) != ABLATION_INSTANCE_COUNT:
            raise ValueError(
                f"ablation allocation requires exactly {ABLATION_INSTANCE_COUNT} "
                f"distinct task instances, got {len(ids)}")
        if len(set(ids)) != len(ids):
            raise ValueError("ablation task ids must be distinct (no replacement)")
        if not self.frozen_strata:
            raise ValueError(
                "frozen_strata plan must be non-empty: the evaluator verifies "
                "the observed family counts against the frozen stratification")
        if sum(self.frozen_strata.values()) != len(self.tasks):
            raise ValueError(
                f"frozen strata counts {self.frozen_strata} sum to "
                f"{sum(self.frozen_strata.values())}, expected "
                f"{len(self.tasks)} task instances")
        observed: dict[str, int] = {}
        for task in self.tasks:
            observed[task.family] = observed.get(task.family, 0) + 1
        if observed != self.frozen_strata:
            raise ValueError(
                f"observed family counts {observed} != frozen stratification "
                f"plan {self.frozen_strata}")

    def allocation_metadata(self) -> dict:
        observed: dict[str, int] = {}
        for task in self.tasks:
            observed[task.family] = observed.get(task.family, 0) + 1
        return {
            "task_ids": [t.task_id for t in self.tasks],
            "task_count": len(self.tasks),
            "observed_family_counts": observed,
            "frozen_strata": dict(self.frozen_strata),
        }


@dataclass(frozen=True)
class AblationEvaluation:
    """Immutable stage-2 evidence (deep-copy accessors) plus the emitted
    schema-valid child CIG record."""

    _passed: bool
    _deltas: tuple[float, ...]
    _delta_success: float
    _bootstrap: dict
    _episodes: tuple[dict, ...]
    _child_record: dict
    _allocation: dict

    @property
    def passed(self) -> bool:
        return self._passed

    @property
    def deltas(self) -> list[float]:
        return copy.deepcopy(list(self._deltas))

    @property
    def delta_success(self) -> float:
        return self._delta_success

    @property
    def bootstrap(self) -> dict:
        return copy.deepcopy(self._bootstrap)

    @property
    def episodes(self) -> tuple[dict, ...]:
        return copy.deepcopy(self._episodes)

    @property
    def child_record(self) -> dict:
        return copy.deepcopy(self._child_record)

    @property
    def allocation(self) -> dict:
        return copy.deepcopy(self._allocation)


class AblationEvaluator:
    """Paired ablation evaluation of one somatic candidate over 20 tasks."""

    def __init__(self, *, candidate_envelope: dict, registry,
                 base_config: RuntimeConfig, runner: GateEpisodeRunner,
                 budgets: Budgets, parent_cig_id: str, mining_seed: int,
                 target_regulation=None,
                 bootstrap_seed: int = BOOTSTRAP_SEED):
        # central preflight BEFORE any episode (zero runner calls on invalid
        # setup): parent grammar, seed representation, child coherence
        validate_gate_setup(parent_cig_id, mining_seed)
        self._registry = registry
        self._base_config = base_config
        self._runner = runner
        self._budgets = budgets
        self._parent_cig_id = parent_cig_id
        self._mining_seed = mining_seed
        self._target_regulation = target_regulation
        # primary protocol is LOCKED (ADR-0002): tau_c/n_resamples are not
        # constructor parameters — sensitivity analysis must use the
        # explicit analysis-only API (evaluate_analysis), which cannot emit
        # primary CIG evidence
        self._tau_c = TAU_C
        self._bootstrap_seed = bootstrap_seed
        self._n_resamples = BOOTSTRAP_N_RESAMPLES
        self._configs = build_evaluation_configs(
            base_config, candidate_envelope, registry,
            target_regulation=target_regulation)
        self._candidate_gene_id = self._configs.target_gene_id

    def evaluate_analysis(self, allocation: AblationAllocation, *, tau_c: float,
                           n_resamples: int, bootstrap_seed: int) -> dict:
        """ANALYSIS-ONLY sensitivity evaluation (ADR-0002: threshold
        sensitivity is secondary). Runs the same paired episodes but CANNOT
        emit primary CIG evidence — returns analysis deltas/bootstrap only,
        with no child record and no stage-pass verdict. Arbitrary tau_c/
        n_resamples are permitted HERE because the analysis result never
        enters the primary gate or the somatic lifecycle."""
        from inheritance.overlay import evaluation_settings
        settings = evaluation_settings(
            max_tokens=self._budgets.max_tokens_per_request,
            seed=self._mining_seed)
        analysis_deltas: list[float] = []
        for task in allocation.tasks:
            outcomes: dict[str, bool] = {}
            for condition, config in (("with_trait", self._configs.with_trait),
                                       ("without_trait", self._configs.without_trait)):
                outcome = self._runner.run_episode(task, config, self._budgets,
                                                    condition, settings)
                validate_outcome(outcome, task.task_id, condition, settings=settings)
                outcomes[condition] = outcome.success
            analysis_deltas.append(float(outcomes["with_trait"])
                                    - float(outcomes["without_trait"]))
        bootstrap = paired_cluster_bootstrap(
            analysis_deltas, seed=bootstrap_seed, n_resamples=n_resamples)
        return {
            "analysis_only": True,
            "deltas": analysis_deltas,
            "bootstrap": {
                "method": bootstrap.method, "seed": bootstrap.seed,
                "n_resamples": bootstrap.n_resamples,
                "lower_bound": bootstrap.lower_bound,
                "upper_bound": bootstrap.upper_bound,
            },
            "tau_c": tau_c,
            "child_record": None,   # analysis-only: never primary evidence
            "stage_pass": None,     # analysis-only: never a stage verdict
        }

    def evaluate(self, allocation: AblationAllocation) -> AblationEvaluation:
        """Run the paired 20-task ablation; returns immutable stage evidence
        + the emitted child CIG record."""
        gene = self._candidate_gene_id
        episodes: list[dict] = []
        deltas: list[float] = []
        expression_observations: list[dict] = []
        seen_pairs: set[tuple[str, str]] = set()
        # locked evaluation settings derived from the fixed budget, shared
        # by both conditions
        from inheritance.overlay import evaluation_settings
        locked_settings = evaluation_settings(
            max_tokens=self._budgets.max_tokens_per_request,
            seed=self._mining_seed)

        for task in allocation.tasks:
            outcomes: dict[str, tuple[bool, tuple[str, ...]]] = {}
            for condition, config in (("with_trait", self._configs.with_trait),
                                       ("without_trait", self._configs.without_trait)):
                pair_key = (task.task_id, condition)
                if pair_key in seen_pairs:
                    raise RunnerContractError(
                        f"duplicate execution for {pair_key}; exactly one run "
                        f"per (task instance, condition) is allowed")
                seen_pairs.add(pair_key)
                outcome = self._runner.run_episode(task, config, self._budgets,
                                                    condition, locked_settings)
                validate_outcome(outcome, task.task_id, condition,
                                  settings=locked_settings)
                outcomes[condition] = (outcome.success,
                                        tuple(outcome.expressed_gene_ids))
                episodes.append({
                    "task_id": task.task_id,
                    "family": task.family,
                    "condition": condition,
                    "success": outcome.success,
                    "expressed_gene_ids": list(outcome.expressed_gene_ids),
                    "trajectory_id": outcome.trajectory_id,
                    "status": outcome.status,
                })
            fired_with = gene in outcomes["with_trait"][1]
            fired_without = gene in outcomes["without_trait"][1]
            if fired_without:
                # contaminated intervention: INVALID evidence — abort the
                # stage with no completed child record (both stages)
                raise EvaluationIntegrityError(
                    f"without-trait episode on {task.task_id!r} expressed the "
                    f"target gene {gene!r}: the paired intervention is "
                    f"contaminated and produced no child CIG record")
            expression_observations.append({
                "task_id": task.task_id,
                "fired_with": fired_with,
                "fired_without": fired_without,
            })
            d_i = float(outcomes["with_trait"][0]) - float(outcomes["without_trait"][0])
            deltas.append(d_i)

        bootstrap = paired_cluster_bootstrap(
            deltas, seed=self._bootstrap_seed, n_resamples=self._n_resamples)
        passed = stage_passes(bootstrap.point_estimate, bootstrap.lower_bound,
                               tau_c=self._tau_c)
        child = self._build_child_record(
            allocation=allocation, episodes=episodes, deltas=deltas,
            bootstrap=bootstrap, expression_observations=expression_observations,
            passed=passed, settings=locked_settings)
        return AblationEvaluation(
            _passed=passed, _deltas=tuple(deltas),
            _delta_success=bootstrap.point_estimate, _bootstrap={
                "method": bootstrap.method,
                "seed": bootstrap.seed,
                "n_resamples": bootstrap.n_resamples,
                "point_estimate": bootstrap.point_estimate,
                "lower_bound": bootstrap.lower_bound,
                "upper_bound": bootstrap.upper_bound,
                "confidence": 0.95,
            },
            _episodes=tuple(copy.deepcopy(episodes)),
            _child_record=copy.deepcopy(child),
            _allocation=copy.deepcopy(allocation.allocation_metadata()))

    def _build_child_record(self, *, allocation: AblationAllocation,
                             episodes: list[dict], deltas: list[float],
                             bootstrap, expression_observations: list[dict],
                             passed: bool, settings) -> dict:
        child_id = f"{self._parent_cig_id}/S{self._mining_seed}-ablation"
        child = {
            "cig_id": child_id,
            "parent_cig_id": self._parent_cig_id,
            "seed": self._mining_seed,
            "stage": "ablation",
            "measurements": {
                "candidate_gene_id": self._candidate_gene_id,
                "allocation": allocation.allocation_metadata(),
                "episodes": episodes,
                "paired_deltas": deltas,
                "point_estimate_delta_success": bootstrap.point_estimate,
                "bootstrap": {
                    "method": bootstrap.method,
                    "seed": bootstrap.seed,
                    "n_resamples": bootstrap.n_resamples,
                    "lower_bound": bootstrap.lower_bound,
                    "upper_bound": bootstrap.upper_bound,
                },
                "tau_c": self._tau_c,
                "thresholds": {"rule": ABLATION_THRESHOLD_TEXT},
                "expression_observations": expression_observations,
                "deterministic_execution_policy": {
                    "runs_per_task_condition": 1,
                    "eval_temperature": 0.0,
                    "paired_conditions": ["with_trait", "without_trait"],
                },
                "eval_generation_settings": {
                                        "temperature": settings.temperature,
                    "max_tokens": settings.max_tokens,
                    "seed": settings.seed,
                    "stop": list(settings.stop),
                },
                "stage_pass": passed,
            },
        }
        load_cig(child)  # schema-valid or fail closed
        return child
