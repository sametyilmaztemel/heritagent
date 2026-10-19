"""Stage 3 — Generalization evaluator (issue #12; ADR-0002; recalibrated by
ADR-0012 D5 and D7, owner-accepted 2026-10-19).

In-scope: the ``generalization_in_scope`` child records stage 2's paired
outcomes by reference and passes iff Δ ≥ 0 over the games stage 2 played
(the locked in-scope rule). It runs no episode and adds no test: whenever
stage 3 runs, stage 2's α was met, which implies net > 0 (ADR-0012 D5). It
is kept because the record store requires the five M0 stage keys and
because stage 2's games are the held-out in-scope evidence SPEC §9 asks
stage 3 to provide (the reading the owner accepted, Q6).

Out-of-scope, when the frozen applicability omits at least one family:
24 benefit-bank games over the omitted families, round-robin in name
order, each family's first games in bank order (24, 12, 8, 6, or 5/5/5/5/4
per omitted family), played with the trait in round R5 against R5's G0
cells. Pass iff Δ_out ≥ −0.05 in exact arithmetic, i.e. a net loss (cell
successes minus with-trait successes) of at most 1 of 24. An all-family
applicability gets a vacuous pass (ADR-0007 E6). It runs only for a
candidate whose stage-2 p_k met α_k at some look (ADR-0012 D7); for any
other its child carries ``stage_pass: false``, ``measurements.not_run:
true`` and the reason.

Applicability is read from the content-addressed skill artifact, never
from caller labels. One deterministic execution per (seed, round, task,
condition), temperature 0; the G0 side is the round's cell; target
expression in a cell is contaminated evidence (integrity error, no child).
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from fractions import Fraction

from genome.validation.loader import load_cig, load_somatic
from runtime.expression import RuntimeConfig
from runtime.loop import Budgets
from trajectory.storage.canonical import canonical_json, sha256_hex

from inheritance.overlay import EvaluationConfigs, build_evaluation_configs
from inheritance.runner import (
    EvaluationIntegrityError,
    EvaluationTask,
    GateEpisodeRunner,
    RunnerContractError,
    validate_gate_setup,
    validate_outcome,
)

OUT_SCOPE_COUNT = 24
IN_SCOPE_MIN_DELTA = 0.0
OUT_SCOPE_MIN_DELTA = -0.05
OUT_SCOPE_MIN_DELTA_EXACT = Fraction(-1, 20)
OUT_SCOPE_ROUND = 5  # ADR-0012 D2
BENEFIT_BANK = "benefit_bank"
IN_SCOPE_THRESHOLD_TEXT = "delta_in >= 0.0 over stage 2's games (by reference, ADR-0012 D5)"
OUT_SCOPE_THRESHOLD_TEXT = ("delta_out >= -0.05 over 24 benefit-bank games, exact: net loss <= 1 "
                            "(ADR-0012 D5)")
OUT_SCOPE_RULE = ("24 benefit_bank games over the omitted families in name order, round-robin: "
                  "game j is the floor(j/m)-th game, in bank order, of the (j mod m)-th omitted "
                  "family")
VACUOUS_OUT_OF_SCOPE_REASON = (
    "vacuous: the frozen applicability names every family of the family "
    "universe, so no out-of-scope family exists and no out-of-scope "
    "threshold is applied (ADR-0007 E6)")
VACUOUS_OUT_OF_SCOPE_THRESHOLD_TEXT = "not applied: vacuous out-of-scope stage (ADR-0007 E6)"
NOT_RUN_REASON = ("not run: no stage-2 look met its alpha_k, so the out-of-scope check, 4A and "
                  "4C do not run and no threshold in the sensitivity grids could change the "
                  "verdict (ADR-0012 D7)")
NOT_RUN_THRESHOLD_TEXT = "not run: stage 2 never met alpha_k (ADR-0012 D7)"


def _exact(value: Fraction) -> dict:
    return {"exact": f"{value.numerator}/{value.denominator}", "float": float(value)}


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


@dataclass(frozen=True)
class OutOfScopeAllocation:
    """The frozen out-of-scope allocation (ADR-0012 D5), validated
    fail-closed: 24 distinct benefit-bank games interleaved over the
    families the applicability omits, in name order, or none when the
    applicability names every family of the universe (vacuous, ADR-0007
    E6). ``bank_sha256`` names the bank manifest whose order the builder
    read."""

    tasks: tuple[EvaluationTask, ...]
    applicability: tuple[str, ...]
    family_universe: tuple[str, ...]
    bank_sha256: str

    def __post_init__(self):
        object.__setattr__(self, "tasks", copy.deepcopy(tuple(self.tasks)))
        object.__setattr__(self, "applicability", tuple(sorted(self.applicability)))
        object.__setattr__(self, "family_universe", tuple(sorted(self.family_universe)))
        universe = set(self.family_universe)
        if not universe or len(universe) != len(self.family_universe):
            raise ValueError("the out-of-scope allocation needs the frozen family universe")
        if not self.applicability or not set(self.applicability) <= universe:
            raise ValueError(f"the applicability {list(self.applicability)} must be a non-empty "
                             f"subset of {sorted(universe)}")
        if not isinstance(self.bank_sha256, str) or len(self.bank_sha256) != 64:
            raise ValueError("the out-of-scope allocation must name the benefit bank's sha256")
        omitted = self.omitted_families
        if not omitted:
            if self.tasks:
                raise ValueError("a vacuous out-of-scope stage has no games (ADR-0007 E6)")
            return
        ids = [t.task_id for t in self.tasks]
        if len(ids) != OUT_SCOPE_COUNT or len(set(ids)) != len(ids):
            raise ValueError(f"the out-of-scope allocation requires exactly {OUT_SCOPE_COUNT} "
                             f"distinct games, got {len(ids)} ({len(set(ids))} distinct)")
        for j, task in enumerate(self.tasks):
            family = omitted[j % len(omitted)]
            if task.family != family:
                raise ValueError(f"out-of-scope game {j} ({task.task_id!r}) is of family "
                                 f"{task.family!r}, its slot belongs to {family!r}")
            if task.bank_id != f"{BENEFIT_BANK}[{family}]":
                raise ValueError(f"out-of-scope game {task.task_id!r} has bank provenance "
                                 f"{task.bank_id!r}, not {BENEFIT_BANK}[{family}]")

    @property
    def omitted_families(self) -> tuple[str, ...]:
        return tuple(f for f in self.family_universe if f not in self.applicability)

    @property
    def vacuous(self) -> bool:
        return not self.omitted_families

    def allocation_metadata(self) -> dict:
        counts: dict[str, int] = {}
        for task in self.tasks:
            counts[task.family] = counts.get(task.family, 0) + 1
        metadata = {"task_ids": [t.task_id for t in self.tasks], "task_count": len(self.tasks),
                    "omitted_families": list(self.omitted_families),
                    "family_counts": counts, "frozen_applicability": list(self.applicability),
                    "rule": OUT_SCOPE_RULE, "bank": BENEFIT_BANK,
                    "bank_sha256": self.bank_sha256}
        if self.vacuous:
            metadata["out_scope_vacuous"] = True
            metadata["family_universe"] = list(self.family_universe)
        return metadata


@dataclass(frozen=True)
class GeneralizationEvaluation:
    """Immutable stage-3 evidence: both scoped results plus the two emitted
    child CIG records."""

    _in_scope_passed: bool
    _out_scope_passed: bool
    _delta_in: Fraction
    _delta_out: Fraction | None  # None when vacuous or not run
    _in_scope_child: dict
    _out_scope_child: dict
    _out_scope_state: str        # "run", "vacuous" or "not_run"

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
    def delta_in(self) -> Fraction:
        return self._delta_in

    @property
    def delta_out(self) -> Fraction | None:
        return self._delta_out

    @property
    def out_scope_state(self) -> str:
        return self._out_scope_state

    @property
    def in_scope_child(self) -> dict:
        return copy.deepcopy(self._in_scope_child)

    @property
    def out_scope_child(self) -> dict:
        return copy.deepcopy(self._out_scope_child)

    @property
    def child_records(self) -> tuple[dict, ...]:
        return (self.in_scope_child, self.out_scope_child)


def paired_outcomes_sha256(paired_outcomes) -> str:
    """The digest an in-scope child names stage 2's outcomes by."""
    return sha256_hex(canonical_json(list(paired_outcomes)).encode("utf-8"))


class GeneralizationEvaluator:
    """Stage 3 of one candidate (ADR-0012 D5, D7): the in-scope child from
    stage 2's record, and the out-of-scope check in round R5."""

    def __init__(self, *, candidate_envelope: dict, registry,
                 base_config: RuntimeConfig, budgets: Budgets, parent_cig_id: str,
                 mining_seed: int, target_regulation=None):
        from inheritance.overlay import evaluation_settings
        validate_gate_setup(parent_cig_id, mining_seed)
        self._budgets = budgets
        self._parent_cig_id = parent_cig_id
        self._mining_seed = mining_seed
        # bind the candidate + frozen applicability ONCE from the validated
        # artifact (never from caller labels)
        self._configs: EvaluationConfigs = build_evaluation_configs(
            base_config, candidate_envelope, registry, target_regulation=target_regulation)
        self._gene = self._configs.target_gene_id
        self._frozen_applicability = resolve_frozen_applicability(candidate_envelope, registry)
        self._settings = evaluation_settings(max_tokens=budgets.max_tokens_per_request,
                                             seed=mining_seed)

    def evaluate(self, ablation, allocation: OutOfScopeAllocation,
                 runner: GateEpisodeRunner | None) -> GeneralizationEvaluation:
        """Both children. The out-of-scope check plays in R5 only when
        stage 2 met α at some look; ``runner`` is round R5's (unused
        otherwise)."""
        if not isinstance(allocation, OutOfScopeAllocation):
            raise RunnerContractError("stage 3 needs a frozen OutOfScopeAllocation")
        if allocation.applicability != self._frozen_applicability:
            raise ValueError(f"the out-of-scope allocation is for applicability "
                             f"{list(allocation.applicability)}, the candidate's frozen "
                             f"applicability is {list(self._frozen_applicability)}")
        in_child, delta_in = self._in_scope_child(ablation)
        if not ablation.alpha_met:
            out_child, delta_out, state = self._not_run_child(allocation), None, "not_run"
        elif allocation.vacuous:
            out_child, delta_out, state = self._vacuous_child(allocation), None, "vacuous"
        else:
            if runner is None:
                raise RunnerContractError("the out-of-scope check needs round R5's runner")
            out_child, delta_out = self._run_out_of_scope(allocation, runner)
            state = "run"
        return GeneralizationEvaluation(
            _in_scope_passed=in_child["measurements"]["stage_pass"],
            _out_scope_passed=out_child["measurements"]["stage_pass"],
            _delta_in=delta_in, _delta_out=delta_out,
            _in_scope_child=in_child, _out_scope_child=out_child, _out_scope_state=state)

    # -- children -------------------------------------------------------------
    def _base(self, stage: str) -> dict:
        return {"cig_id": f"{self._parent_cig_id}/S{self._mining_seed}-{stage}",
                "parent_cig_id": self._parent_cig_id, "seed": self._mining_seed,
                "stage": stage}

    def _common(self, scope: str) -> dict:
        return {"candidate_gene_id": self._gene,
                **self._configs.binding(),  # ADR-0010 (every child, by reference and not run too)
                "scope": scope, "frozen_applicability": list(self._frozen_applicability),
                "eval_generation_settings": {
                    "temperature": self._settings.temperature,
                    "max_tokens": self._settings.max_tokens,
                    "seed": self._mining_seed, "stop": list(self._settings.stop)}}

    def _in_scope_child(self, ablation) -> tuple[dict, Fraction]:
        record = ablation.child_record
        if record["measurements"].get("candidate_gene_id") != self._gene:
            raise RunnerContractError(f"the stage-2 record is {record['measurements'].get('candidate_gene_id')!r}'s, "
                                      f"not {self._gene!r}'s")
        outcomes = ablation.paired_outcomes
        n_plus = sum(1 for o in outcomes if o["with_trait"]["success"] and not o["cell"]["success"])
        n_minus = sum(1 for o in outcomes if o["cell"]["success"] and not o["with_trait"]["success"])
        games = len(outcomes)
        delta = Fraction(n_plus - n_minus, games)
        passed = delta >= 0
        child = self._base("generalization_in_scope")
        child["measurements"] = {
            **self._common("in_scope"),
            "by_reference": {"stage": "ablation", "child_id": record["cig_id"],
                             "paired_outcomes_sha256": paired_outcomes_sha256(outcomes),
                             "games": games, "n_plus": n_plus, "n_minus": n_minus,
                             "net": n_plus - n_minus},
            "episodes_run": 0,
            "delta": _exact(delta),
            "locked_threshold": {"min_delta": IN_SCOPE_MIN_DELTA, "rule": IN_SCOPE_THRESHOLD_TEXT},
            "stage_pass": passed,
        }
        load_cig(child)
        return child, delta

    def _vacuous_child(self, allocation: OutOfScopeAllocation) -> dict:
        child = self._base("generalization_out_of_scope")
        child["measurements"] = {
            **self._common("out_of_scope"),
            "allocation": allocation.allocation_metadata(),
            "vacuous": True, "vacuous_reason": VACUOUS_OUT_OF_SCOPE_REASON,
            "locked_threshold": {"min_delta": None, "rule": VACUOUS_OUT_OF_SCOPE_THRESHOLD_TEXT},
            "episodes_run": 0, "stage_pass": True,
        }
        load_cig(child)
        return child

    def _not_run_child(self, allocation: OutOfScopeAllocation) -> dict:
        child = self._base("generalization_out_of_scope")
        child["measurements"] = {
            **self._common("out_of_scope"),
            "allocation": allocation.allocation_metadata(),
            "not_run": True, "not_run_reason": NOT_RUN_REASON,
            "all_family_applicability": allocation.vacuous,
            "locked_threshold": {"min_delta": None, "rule": NOT_RUN_THRESHOLD_TEXT},
            "episodes_run": 0, "stage_pass": False,
        }
        load_cig(child)
        return child

    def _run_out_of_scope(self, allocation: OutOfScopeAllocation,
                          runner: GateEpisodeRunner) -> tuple[dict, Fraction]:
        outcomes = []
        seen: set[str] = set()
        for task in allocation.tasks:
            if task.task_id in seen:  # pragma: no cover - the allocation guarantees it
                raise RunnerContractError(f"duplicate out-of-scope game {task.task_id!r}")
            seen.add(task.task_id)
            with_trait = runner.run_episode(task, self._configs.with_trait, self._budgets,
                                            "with_trait", self._settings)
            validate_outcome(with_trait, task.task_id, "with_trait", settings=self._settings)
            cell = runner.run_episode(task, self._configs.without_trait, self._budgets,
                                      "without_trait", self._settings)
            validate_outcome(cell, task.task_id, "without_trait", settings=self._settings)
            if self._gene in cell.expressed_gene_ids:
                raise EvaluationIntegrityError(
                    f"the G0 cell of out-of-scope game {task.task_id!r} expressed the target "
                    f"gene {self._gene!r}: contaminated intervention, no child record")
            outcomes.append({
                "task_id": task.task_id, "family": task.family, "round": OUT_SCOPE_ROUND,
                "with_trait": _episode(with_trait),
                "cell": {"seed": self._mining_seed, "round": OUT_SCOPE_ROUND, **_episode(cell)}})
        successes_with = sum(o["with_trait"]["success"] for o in outcomes)
        cell_successes = sum(o["cell"]["success"] for o in outcomes)
        delta = Fraction(successes_with - cell_successes, len(outcomes))
        passed = delta >= OUT_SCOPE_MIN_DELTA_EXACT
        child = self._base("generalization_out_of_scope")
        child["measurements"] = {
            **self._common("out_of_scope"),
            "allocation": allocation.allocation_metadata(),
            "round": OUT_SCOPE_ROUND,
            "paired_outcomes": outcomes,
            "summary": {"successes_with": successes_with, "cell_successes": cell_successes,
                        "task_count": len(outcomes),
                        "net_loss": cell_successes - successes_with},
            "delta": _exact(delta),
            "locked_threshold": {"min_delta": OUT_SCOPE_MIN_DELTA,
                                 "min_delta_exact": _exact(OUT_SCOPE_MIN_DELTA_EXACT),
                                 "rule": OUT_SCOPE_THRESHOLD_TEXT},
            "deterministic_execution_policy": {
                "runs_per_task_condition": 1, "eval_temperature": 0.0,
                "paired_conditions": ["with_trait", "g0_cell"],
                "cells": "the seed's R5 cells (ADR-0012 D2)"},
            "episodes_run": len(outcomes),
            "stage_pass": passed,
        }
        load_cig(child)
        return child, delta


def _episode(outcome) -> dict:
    return {"success": outcome.success, "expressed_gene_ids": list(outcome.expressed_gene_ids),
            "trajectory_id": outcome.trajectory_id, "status": outcome.status}
