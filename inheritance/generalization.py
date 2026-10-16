"""Stage 3 — Generalization evaluator (issue #12; merged ADR-0002).

In-scope: 10 unseen instances sampled from the candidate's FROZEN
applicability (read from the content-addressed skill artifact, never from
caller labels); pass iff paired delta >= 0.
Out-of-scope: 10 instances outside the declared applicability; pass iff
paired delta >= -0.05. When the applicability names every family of the
family universe the allocation carries, no out-of-scope family exists:
the stage is vacuous (no episode, a recorded vacuous pass, no threshold
applied) and the in-scope tasks must cover every family (ADR-0007 E6,
owner decision). The pooled in-scope delta is then the only harm check.
Frozen family-count plans per scope; observed counts must match exactly;
every task carries canonical `generalization_bank[<family>]` provenance;
no replacement within the trait; cross-trait reuse allowed (logged in
allocation metadata).

Paired intervention semantics from #11: one deterministic execution per
(task instance, condition), temp 0, benchmark success from the runner
outcome, trajectory provenance required, without-trait target expression
is contaminated evidence (integrity error, stage abort, no child).
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass

from genome.validation.errors import RecordConsistencyError
from genome.validation.loader import load_cig, load_somatic
from runtime.expression import RuntimeConfig
from runtime.loop import Budgets

from inheritance.overlay import (
    EvaluationConfigs,
    EvaluationSetupError,
    build_evaluation_configs,
)
from inheritance.runner import (
    EvaluationIntegrityError,
    EvaluationTask,
    GateEpisodeRunner,
    RunnerContractError,
    validate_outcome,
)

IN_SCOPE_COUNT = 10
OUT_SCOPE_COUNT = 10
IN_SCOPE_MIN_DELTA = 0.0
OUT_SCOPE_MIN_DELTA = -0.05
IN_SCOPE_THRESHOLD_TEXT = "paired delta_in >= 0.0"
OUT_SCOPE_THRESHOLD_TEXT = "paired delta_out >= -0.05"
VACUOUS_OUT_OF_SCOPE_REASON = (
    "vacuous: the frozen applicability names every family of the family "
    "universe, so no out-of-scope family exists and no out-of-scope "
    "threshold is applied; the only harm check is the pooled in-scope delta "
    "(>= 0) over 10 games spanning every family, not a per-family check "
    "(ADR-0007 E6)")
VACUOUS_OUT_OF_SCOPE_THRESHOLD_TEXT = "not applied: vacuous out-of-scope stage (ADR-0007 E6)"


@dataclass(frozen=True)
class GeneralizationAllocation:
    """Frozen stage-3 allocation: exactly 10 in-scope + 10 out-of-scope
    distinct tasks (0 out-of-scope for an all-family applicability, ADR-0007
    E6) with per-scope frozen family-count plans. Applicability
    is resolved from the stored artifact (not caller labels) and every
    in-scope family must be inside it, every out-of-scope family outside.
    Tasks are deep-frozen at construction."""

    in_scope_tasks: tuple[EvaluationTask, ...]
    out_scope_tasks: tuple[EvaluationTask, ...]
    frozen_in_scope_plan: dict
    frozen_out_scope_plan: dict
    declared_frozen_applicability: tuple[str, ...]
    # the frozen family universe; required only for a vacuous out-of-scope
    # stage (no out-of-scope tasks), which ADR-0007 E6 allows only when the
    # applicability names every family of the universe
    family_universe: tuple[str, ...] = ()

    def __post_init__(self):
        import copy as _copy
        import types
        object.__setattr__(self, "in_scope_tasks",
                            _copy.deepcopy(self.in_scope_tasks))
        object.__setattr__(self, "out_scope_tasks",
                            _copy.deepcopy(self.out_scope_tasks))
        object.__setattr__(self, "frozen_in_scope_plan",
                            types.MappingProxyType(_copy.deepcopy(dict(self.frozen_in_scope_plan))))
        object.__setattr__(self, "frozen_out_scope_plan",
                            types.MappingProxyType(_copy.deepcopy(dict(self.frozen_out_scope_plan))))
        object.__setattr__(self, "declared_frozen_applicability",
                            tuple(sorted(self.declared_frozen_applicability)))
        object.__setattr__(self, "family_universe", tuple(sorted(self.family_universe)))

        ids = [t.task_id for t in self.in_scope_tasks] + \
            [t.task_id for t in self.out_scope_tasks]
        if len(self.in_scope_tasks) != IN_SCOPE_COUNT:
            raise ValueError(f"in-scope allocation requires exactly "
                              f"{IN_SCOPE_COUNT} tasks, got {len(self.in_scope_tasks)}")
        if self.out_scope_vacuous:
            universe = set(self.family_universe)
            if not universe:
                raise ValueError(
                    "an empty out-of-scope allocation needs the frozen family "
                    "universe (vacuous out-of-scope stage, ADR-0007 E6)")
            if set(self.declared_frozen_applicability) != universe:
                raise ValueError(
                    f"the out-of-scope stage is vacuous only when the "
                    f"applicability {list(self.declared_frozen_applicability)} "
                    f"names every family of {sorted(universe)}")
            if dict(self.frozen_out_scope_plan):
                raise ValueError("a vacuous out-of-scope stage has an empty plan")
            if {t.family for t in self.in_scope_tasks} != universe:
                raise ValueError("with a vacuous out-of-scope stage the in-scope "
                                  "tasks must cover every family of the universe")
        elif len(self.out_scope_tasks) != OUT_SCOPE_COUNT:
            raise ValueError(f"out-of-scope allocation requires exactly "
                              f"{OUT_SCOPE_COUNT} tasks, got {len(self.out_scope_tasks)}")
        if len(set(ids)) != len(ids):
            raise ValueError("generalization task ids must be distinct across "
                              "both scopes (no replacement within the trait)")
        for scope, tasks, plan in (("in_scope", self.in_scope_tasks,
                                     self.frozen_in_scope_plan),
                                    ("out_scope", self.out_scope_tasks,
                                      self.frozen_out_scope_plan)):
            if not plan and not (scope == "out_scope" and self.out_scope_vacuous):
                raise ValueError(f"{scope} frozen family-count plan must be "
                                  f"non-empty")
            observed: dict[str, int] = {}
            for task in tasks:
                observed[task.family] = observed.get(task.family, 0) + 1
            if observed != dict(plan):
                raise ValueError(f"{scope} observed family counts {observed} != "
                                  f"frozen plan {dict(plan)}")
            for task in tasks:
                expected_bank = f"generalization_bank[{task.family}]"
                if task.bank_id != expected_bank:
                    raise ValueError(
                        f"{scope} task {task.task_id!r} bank provenance "
                        f"{task.bank_id!r} != canonical {expected_bank!r}")

    @property
    def out_scope_vacuous(self) -> bool:
        return not self.out_scope_tasks

    def validate_against_applicability(self, frozen_families: tuple[str, ...]) -> None:
        if self.out_scope_vacuous and set(frozen_families) != set(self.family_universe):
            raise ValueError(
                f"the out-of-scope stage is vacuous only when the stored "
                f"applicability {sorted(frozen_families)} names every family of "
                f"{list(self.family_universe)}")
        in_scope_families = {t.family for t in self.in_scope_tasks}
        out_scope_families = {t.family for t in self.out_scope_tasks}
        outside = in_scope_families - set(frozen_families)
        if outside:
            raise ValueError(
                f"in-scope families {sorted(outside)} are not in the frozen "
                f"applicability {sorted(frozen_families)}")
        overlap = out_scope_families & set(frozen_families)
        if overlap:
            raise ValueError(
                f"out-of-scope families {sorted(overlap)} overlap the frozen "
                f"applicability {sorted(frozen_families)}")

    def allocation_metadata(self, resolved_applicability=None) -> dict:
        """resolved_applicability: the evaluator's artifact-resolved frozen
        families; when supplied, overrides the (ignored) declared label so
        audit metadata always reports the actual enforced scope."""
        applicability = list(resolved_applicability or
                              self.declared_frozen_applicability)
        metadata = {
            "in_scope_task_ids": [t.task_id for t in self.in_scope_tasks],
            "out_scope_task_ids": [t.task_id for t in self.out_scope_tasks],
            "in_scope_family_plan": dict(self.frozen_in_scope_plan),
            "out_scope_family_plan": dict(self.frozen_out_scope_plan),
            "frozen_applicability": applicability,
            "bank_provenance": sorted({t.bank_id for t in
                                        self.in_scope_tasks + self.out_scope_tasks
                                        if t.bank_id}),
        }
        if self.out_scope_vacuous:
            metadata["out_scope_vacuous"] = True
            metadata["family_universe"] = list(self.family_universe)
        return metadata


@dataclass(frozen=True)
class GeneralizationEvaluation:
    """Immutable stage-3 evidence: both scoped results plus the two emitted
    child CIG records."""

    _in_scope_passed: bool
    _out_scope_passed: bool
    _delta_in: float
    _delta_out: float | None  # None for a vacuous out-of-scope stage
    _episodes: tuple[dict, ...]
    _in_scope_child: dict
    _out_scope_child: dict
    _allocation: dict

    @property
    def in_scope_passed(self) -> bool:
        return self._in_scope_passed

    @property
    def out_scope_passed(self) -> bool:
        return self._out_scope_passed

    @property
    def passed(self) -> bool:
        return self._in_scope_passed and self._out_scope_passed

    @property
    def delta_in(self) -> float:
        return self._delta_in

    @property
    def delta_out(self) -> float | None:
        return self._delta_out

    @property
    def episodes(self) -> tuple[dict, ...]:
        return copy.deepcopy(self._episodes)

    @property
    def in_scope_child(self) -> dict:
        return copy.deepcopy(self._in_scope_child)

    @property
    def out_scope_child(self) -> dict:
        return copy.deepcopy(self._out_scope_child)

    @property
    def allocation(self) -> dict:
        return copy.deepcopy(self._allocation)

    @property
    def child_records(self) -> tuple[dict, ...]:
        return (self.in_scope_child, self.out_scope_child)


def resolve_frozen_applicability(envelope: dict, registry) -> tuple[str, ...]:
    """Read the frozen evaluation-only applicability from the
    content-addressed stored skill artifact (never from caller labels)."""
    validated = load_somatic(copy.deepcopy(envelope), registry)
    artifact = json.loads(registry.resolve(validated["candidate"]["artifact"]))
    families = sorted(set(
        artifact["applicability"]["task_families"]))
    if not families:
        raise ValueError("stored artifact declares an empty applicability")
    return tuple(families)


class GeneralizationEvaluator:
    """Paired stage-3 evaluation over 10 in-scope + 10 out-of-scope tasks
    (in-scope only when the out-of-scope stage is vacuous, ADR-0007 E6)."""

    def __init__(self, *, candidate_envelope: dict, registry,
                 base_config: RuntimeConfig, runner: GateEpisodeRunner,
                 budgets: Budgets, parent_cig_id: str, mining_seed: int,
                 target_regulation=None):
        from inheritance.overlay import evaluation_settings
        self._registry = registry
        self._base_config = base_config
        self._runner = runner
        self._budgets = budgets
        self._parent_cig_id = parent_cig_id
        self._mining_seed = mining_seed
        # bind the candidate + frozen applicability ONCE from the validated
        # artifact (never from caller labels)
        self._configs: EvaluationConfigs = build_evaluation_configs(
            base_config, candidate_envelope, registry,
            target_regulation=target_regulation)
        self._candidate_gene_id = self._configs.target_gene_id
        self._frozen_applicability = resolve_frozen_applicability(
            candidate_envelope, registry)
        self._settings = evaluation_settings(
            max_tokens=budgets.max_tokens_per_request, seed=mining_seed)

    def evaluate(self, allocation: GeneralizationAllocation) -> GeneralizationEvaluation:
        """Validate the allocation against the artifact-declared frozen
        applicability, run the 40 paired episodes (20 when the out-of-scope
        stage is vacuous, ADR-0007 E6), emit both child records."""
        from inheritance.runner import validate_gate_setup
        validate_gate_setup(self._parent_cig_id, self._mining_seed)
        frozen_families = self._frozen_applicability
        allocation.validate_against_applicability(frozen_families)

        gene = self._candidate_gene_id
        episodes: list[dict] = []
        seen_pairs: set[tuple[str, str]] = set()

        def run_scope(scope: str, tasks) -> tuple[float, list[dict]]:
            nonlocal episodes
            scope_episodes: list[dict] = []
            successes_with = 0
            successes_without = 0
            for task in tasks:
                for condition, config in (
                        ("with_trait", self._configs.with_trait),
                        ("without_trait", self._configs.without_trait)):
                    pair_key = (task.task_id, condition)
                    if pair_key in seen_pairs:
                        raise RunnerContractError(
                            f"duplicate execution for {pair_key}; exactly one "
                            f"run per (task instance, condition) is allowed")
                    seen_pairs.add(pair_key)
                    outcome = self._runner.run_episode(task, config,
                                                        self._budgets, condition,
                                                        self._settings)
                    validate_outcome(outcome, task.task_id, condition,
                                      settings=self._settings)
                    if condition == "without_trait" and \
                            gene in outcome.expressed_gene_ids:
                        raise EvaluationIntegrityError(
                            f"without-trait {scope} episode on "
                            f"{task.task_id!r} expressed the target gene "
                            f"{gene!r}: contaminated intervention, no child "
                            f"record")
                    if condition == "with_trait" and outcome.success:
                        successes_with += 1
                    if condition == "without_trait" and outcome.success:
                        successes_without += 1
                    episodes.append({
                        "scope": scope, "task_id": task.task_id,
                        "family": task.family, "condition": condition,
                        "success": outcome.success,
                        "expressed_gene_ids": list(outcome.expressed_gene_ids),
                        "trajectory_id": outcome.trajectory_id,
                        "status": outcome.status,
                    })
            delta = (successes_with - successes_without) / len(tasks)
            return delta, [{"successes_with": successes_with,
                             "successes_without": successes_without,
                             "task_count": len(tasks)}]

        delta_in, in_scope_summary = run_scope("in_scope",
                                                allocation.in_scope_tasks)
        in_scope_passed = delta_in >= IN_SCOPE_MIN_DELTA
        if allocation.out_scope_vacuous:  # ADR-0007 E6: no episode, a vacuous pass
            delta_out = None
            out_scope_summary = [{"successes_with": 0, "successes_without": 0,
                                   "task_count": 0}]
            out_scope_passed = True
        else:
            delta_out, out_scope_summary = run_scope("out_scope",
                                                      allocation.out_scope_tasks)
            out_scope_passed = delta_out >= OUT_SCOPE_MIN_DELTA

        in_child = self._build_child(
            child_id=f"{self._parent_cig_id}/S{self._mining_seed}-generalization_in_scope",
            scope="in_scope", allocation=allocation,
            frozen_families=frozen_families, summaries=in_scope_summary,
            delta=delta_in, threshold=IN_SCOPE_MIN_DELTA,
            threshold_text=IN_SCOPE_THRESHOLD_TEXT, passed=in_scope_passed,
            scope_episodes=[e for e in episodes if e["scope"] == "in_scope"])
        out_child = self._build_child(
            child_id=f"{self._parent_cig_id}/S{self._mining_seed}-generalization_out_of_scope",
            scope="out_of_scope", allocation=allocation,
            frozen_families=frozen_families, summaries=out_scope_summary,
            delta=delta_out, threshold=OUT_SCOPE_MIN_DELTA,
            threshold_text=OUT_SCOPE_THRESHOLD_TEXT, passed=out_scope_passed,
            scope_episodes=[e for e in episodes if e["scope"] == "out_scope"])
        return GeneralizationEvaluation(
            _in_scope_passed=in_scope_passed, _out_scope_passed=out_scope_passed,
            _delta_in=delta_in, _delta_out=delta_out,
            _episodes=tuple(copy.deepcopy(episodes)),
            _in_scope_child=copy.deepcopy(in_child),
            _out_scope_child=copy.deepcopy(out_child),
            _allocation=copy.deepcopy(allocation.allocation_metadata(
                resolved_applicability=self._frozen_applicability)))

    def _build_child(self, *, child_id: str, scope: str,
                      allocation: GeneralizationAllocation,
                      frozen_families: tuple[str, ...], summaries: list[dict],
                      delta: float, threshold: float, threshold_text: str,
                      passed: bool, scope_episodes: list[dict]) -> dict:
        child = {
            "cig_id": child_id,
            "parent_cig_id": self._parent_cig_id,
            "seed": self._mining_seed,
            "stage": (f"generalization_in_scope" if scope == "in_scope"
                       else "generalization_out_of_scope"),
            "measurements": {
                "candidate_gene_id": self._candidate_gene_id,
                **self._configs.binding(),  # ADR-0010 (the vacuous child too)
                "scope": scope,
                "frozen_applicability": list(frozen_families),
                "allocation": allocation.allocation_metadata(
                    resolved_applicability=self._frozen_applicability),
                "summary": summaries,
                "scope_episodes": scope_episodes,
                "delta": delta,
                "locked_threshold": {"min_delta": threshold,
                                      "rule": threshold_text},
                "deterministic_execution_policy": {
                    "runs_per_task_condition": 1,
                    "eval_temperature": 0.0,
                    "paired_conditions": ["with_trait", "without_trait"],
                },
                "eval_generation_settings": {
                    "temperature": self._settings.temperature,
                    "max_tokens": self._settings.max_tokens,
                    "seed": self._mining_seed,
                    "stop": list(self._settings.stop),
                },
                "stage_pass": passed,
            },
        }
        if scope == "out_of_scope" and allocation.out_scope_vacuous:
            child["measurements"]["vacuous"] = True
            child["measurements"]["vacuous_reason"] = VACUOUS_OUT_OF_SCOPE_REASON
            child["measurements"]["locked_threshold"] = {
                "min_delta": None, "rule": VACUOUS_OUT_OF_SCOPE_THRESHOLD_TEXT}
        load_cig(child)
        return child