"""Ablation evaluator (CIG stage 2; ADR-0012 D4, owner-accepted 2026-10-19).

Stage 2 is a pooled, group-sequential exact paired sign test over the
benefit bank (ADR-0012 D1):

- **Allocation.** 192 bank games over the candidate's frozen applicability
  families (m families, in name order): allocation game j (0-based) is the
  ⌊j/m⌋-th game, in bank order, of the (j mod m)-th family. Every candidate
  that needs n games of a family gets the same first n.
- **Looks** after n₁…n₄ = 48, 96, 144 and 192 games. At look k the
  candidate plays games [n₍k−1₎, n_k) once each with the trait, in round
  R_k, against that round's G0 cells (ADR-0012 D2): the runner serves the
  without-trait side from the round's cell of the game.
- **Counts over the first n_k games.** n⁺ = games the with-trait execution
  solved and the cell failed; n⁻ = the reverse; net = n⁺ − n⁻;
  point = net / n_k; p_k = P(Bin(n⁺ + n⁻, ½) ≥ n⁺), exact; p_k = 1 when no
  game is discordant.
- **Efficacy:** pass at look k iff p_k ≤ α_k and point ≥ τ_c = 0.05, with
  α = (0.0005, 0.0005, 0.001, 0.003). **Futility:** at looks 1–3 stop and
  fail iff net ≤ ⌊n_k/40⌋. Without efficacy at look 4 the stage fails.

Every decision is exact (integers and ``Fraction``). Games are never
filtered by expression; expression is recorded. The child keeps every
paired outcome in allocation order with its task id and cell, the
per-look counts, each p_k, the stopping look and reason, and the ADR-0005
percentile bootstrap interval over the games played (seed 0, task_id
order, 10,000 resamples), descriptive only. The candidate's stage-2
contribution is its point at the stopping look. ``stage2_evidence_issues``
recomputes the decision from a record, as the gate does before any round-5
episode and whenever a run reads its gate artifact back.

Emits a schema-valid ``cig/0.1`` child record
(``<parent>/S<mining_seed>-ablation``). No aggregate/verdict, no somatic
terminal decision, no germline mutation — the gate owns those.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction

from genome.validation.loader import load_cig
from runtime.expression import RuntimeConfig
from runtime.loop import Budgets

from inheritance.bootstrap import CONFIDENCE, paired_cluster_bootstrap, sign_test_p
from inheritance.overlay import EvaluationSetupError, build_evaluation_configs
from inheritance.runner import (
    EvaluationIntegrityError,
    EvaluationTask,
    GateEpisodeRunner,
    RunnerContractError,
    validate_gate_setup,
    validate_outcome,
)

LOOKS = (48, 96, 144, 192)
ALLOCATION_SIZE = LOOKS[-1]
ALPHAS = (Fraction(1, 2000), Fraction(1, 2000), Fraction(1, 1000), Fraction(3, 1000))
TAU_C = Fraction(1, 20)
FUTILITY_DIVISOR = 40  # stop and fail at looks 1-3 iff net <= floor(n_k / 40)
BENEFIT_BANK = "benefit_bank"
BOOTSTRAP_SEED = 0            # pre-registered (ADR-0005); descriptive since ADR-0012 D4
BOOTSTRAP_N_RESAMPLES = 10_000
# ADR-0005: the paired deltas enter the bootstrap ordered by task_id
BOOTSTRAP_CLUSTER_ORDER = "task_id"
BOOTSTRAP_METHOD = "percentile-paired-cluster-bootstrap"
ALLOCATION_RULE = ("192 benefit_bank games over the frozen applicability families in name "
                   "order: game j is the floor(j/m)-th game, in bank order, of the "
                   "(j mod m)-th family")
ABLATION_THRESHOLD_TEXT = (
    f"exact one-sided paired sign test at looks {'/'.join(map(str, LOOKS))}: pass at look k "
    f"iff p_k <= alpha_k ({', '.join(map(str, ALPHAS))}) and net/n_k >= {float(TAU_C)}; at "
    f"looks 1-{len(LOOKS) - 1} stop and fail iff net <= floor(n_k/{FUTILITY_DIVISOR}); no "
    f"efficacy at look {len(LOOKS)} fails (ADR-0012 D4)")
CELL_CONDITION = "without_trait"


def exact(value: Fraction) -> dict:
    """A fraction as the report and the records carry it: exact, then float."""
    return {"exact": f"{value.numerator}/{value.denominator}", "float": float(value)}


def parse_exact(record) -> Fraction | None:
    """The fraction an ``exact`` record holds, or None if it is malformed."""
    if not isinstance(record, Mapping) or not isinstance(record.get("exact"), str):
        return None
    try:
        value = Fraction(record["exact"])
    except (ValueError, ZeroDivisionError):
        return None
    return value if record.get("float") == float(value) else None


def look_decision(look: int, n_plus: int, n_minus: int, *,
                  tau_c: Fraction = TAU_C) -> dict:
    """ADR-0012 D4's decision at look ``look`` (1-4) from the counts over its
    first ``LOOKS[look-1]`` games: ``pass`` (efficacy), ``fail`` (futility at
    looks 1-3, or no efficacy at look 4) or ``continue``. ``tau_c`` other
    than 0.05 is for the analysis-only sensitivity grid (D7)."""
    if isinstance(look, bool) or look not in range(1, len(LOOKS) + 1):
        raise ValueError(f"look must be 1..{len(LOOKS)}, got {look!r}")
    n_k = LOOKS[look - 1]
    if n_plus + n_minus > n_k:
        raise ValueError(f"{n_plus} + {n_minus} discordant games exceed the {n_k} of look {look}")
    p = sign_test_p(n_plus, n_minus)
    net = n_plus - n_minus
    point = Fraction(net, n_k)
    alpha_met = p <= ALPHAS[look - 1]
    if alpha_met and point >= tau_c:
        decision, reason = "pass", "efficacy"
    elif look < len(LOOKS) and net <= n_k // FUTILITY_DIVISOR:
        decision, reason = "fail", "futility"
    elif look == len(LOOKS):
        decision, reason = "fail", "no efficacy at the last look"
    else:
        decision, reason = "continue", None
    return {"look": look, "games": n_k, "n_plus": n_plus, "n_minus": n_minus, "net": net,
            "point": point, "p": p, "alpha": ALPHAS[look - 1], "alpha_met": alpha_met,
            "decision": decision, "reason": reason}


def sequential_path(pairs: Sequence[tuple[bool, bool]], *,
                    tau_c: Fraction = TAU_C) -> tuple[list[dict], dict | None]:
    """The looks ADR-0012 D4 takes over ``pairs`` ((with-trait success, cell
    success) in allocation order), up to its stopping look. The second value
    is the stopping look's decision, or None when the rule would continue
    past the games given (not identifiable from them)."""
    looks: list[dict] = []
    for look, n_k in enumerate(LOOKS, start=1):
        if len(pairs) < n_k:
            return looks, None
        n_plus = sum(1 for with_success, cell in pairs[:n_k] if with_success and not cell)
        n_minus = sum(1 for with_success, cell in pairs[:n_k] if cell and not with_success)
        decision = look_decision(look, n_plus, n_minus, tau_c=tau_c)
        looks.append(decision)
        if decision["decision"] != "continue":
            return looks, decision
    raise AssertionError("the last look always decides")  # pragma: no cover


def look_record(decision: Mapping, gate_round: int) -> dict:
    """A look's decision as the child record keeps it (JSON, exact)."""
    return {"look": decision["look"], "round": gate_round, "games": decision["games"],
            "n_plus": decision["n_plus"], "n_minus": decision["n_minus"],
            "net": decision["net"], "point": exact(decision["point"]),
            "p": exact(decision["p"]), "alpha": exact(decision["alpha"]),
            "alpha_met": decision["alpha_met"], "decision": decision["decision"],
            "reason": decision["reason"]}


def look_round(look: int) -> int:
    """Look k plays in round R_k (ADR-0012 D2)."""
    return look


@dataclass(frozen=True)
class AblationAllocation:
    """The frozen stage-2 allocation (ADR-0012 D4), validated fail-closed:
    192 distinct bank games interleaved over the frozen applicability
    families in name order, each game of its slot's family and carrying
    ``benefit_bank[<family>]`` provenance. ``bank_sha256`` names the bank
    manifest the games were taken from (in bank order: the builder of the
    allocation reads the manifest's order, which this class cannot see)."""

    tasks: tuple[EvaluationTask, ...]
    families: tuple[str, ...]
    bank_sha256: str

    def __post_init__(self):
        object.__setattr__(self, "tasks", copy.deepcopy(tuple(self.tasks)))
        object.__setattr__(self, "families", tuple(self.families))
        ids = [t.task_id for t in self.tasks]
        if len(ids) != ALLOCATION_SIZE:
            raise ValueError(f"the stage-2 allocation requires exactly {ALLOCATION_SIZE} "
                             f"games, got {len(ids)}")
        if not all(isinstance(i, str) for i in ids):
            raise ValueError("stage-2 task ids must be strings (ADR-0005 orders them as strings)")
        if len(set(ids)) != len(ids):
            raise ValueError("stage-2 task ids must be distinct (no replacement)")
        if not self.families or list(self.families) != sorted(set(self.families)):
            raise ValueError(f"the allocation families {list(self.families)} must be the frozen "
                             f"applicability, distinct and in name order")
        if not isinstance(self.bank_sha256, str) or len(self.bank_sha256) != 64:
            raise ValueError("the allocation must name the benefit bank's sha256")
        m = len(self.families)
        for j, task in enumerate(self.tasks):
            family = self.families[j % m]
            if task.family != family:
                raise ValueError(f"allocation game {j} ({task.task_id!r}) is of family "
                                 f"{task.family!r}, its slot belongs to {family!r}")
            if task.bank_id != f"{BENEFIT_BANK}[{family}]":
                raise ValueError(f"allocation game {task.task_id!r} has bank provenance "
                                 f"{task.bank_id!r}, not {BENEFIT_BANK}[{family}]")

    def look_tasks(self, look: int) -> tuple[EvaluationTask, ...]:
        start = LOOKS[look - 2] if look > 1 else 0
        return self.tasks[start:LOOKS[look - 1]]

    def allocation_metadata(self) -> dict:
        counts: dict[str, int] = {}
        for task in self.tasks:
            counts[task.family] = counts.get(task.family, 0) + 1
        return {"task_ids": [t.task_id for t in self.tasks], "task_count": len(self.tasks),
                "families": list(self.families), "family_counts": counts,
                "rule": ALLOCATION_RULE, "bank": BENEFIT_BANK, "bank_sha256": self.bank_sha256,
                "looks": list(LOOKS)}


@dataclass(frozen=True)
class AblationEvaluation:
    """Immutable stage-2 evidence (deep-copy accessors) plus the emitted
    schema-valid child CIG record."""

    _passed: bool
    _alpha_met: bool
    _stop: dict
    _looks: tuple[dict, ...]
    _contribution: Fraction
    _paired_outcomes: tuple[dict, ...]
    _bootstrap: dict
    _child_record: dict
    _allocation: dict

    @property
    def passed(self) -> bool:
        return self._passed

    @property
    def alpha_met(self) -> bool:
        """Some look's p_k met its α_k (ADR-0012 D7: stage 3's out-of-scope
        check and 4A run exactly for these candidates)."""
        return self._alpha_met

    @property
    def stop(self) -> dict:
        return copy.deepcopy(self._stop)

    @property
    def looks(self) -> list[dict]:
        return copy.deepcopy(list(self._looks))

    @property
    def contribution(self) -> Fraction:
        """The point at the stopping look (ADR-0012 D4): what a harmful pair
        compares (D6)."""
        return self._contribution

    @property
    def delta_success(self) -> float:
        return float(self._contribution)

    @property
    def pairs(self) -> list[tuple[bool, bool]]:
        """(with-trait success, cell success) per game played, in allocation order."""
        return [(o["with_trait"]["success"], o["cell"]["success"]) for o in self._paired_outcomes]

    @property
    def paired_outcomes(self) -> list[dict]:
        return copy.deepcopy(list(self._paired_outcomes))

    @property
    def bootstrap(self) -> dict:
        return copy.deepcopy(self._bootstrap)

    @property
    def child_record(self) -> dict:
        return copy.deepcopy(self._child_record)

    @property
    def allocation(self) -> dict:
        return copy.deepcopy(self._allocation)


_BOOTSTRAP_MEMO: dict[tuple[float, ...], object] = {}


def descriptive_bootstrap(deltas: Sequence[float]):
    """ADR-0005's interval over the paired deltas (in task_id order),
    memoised: a gate pass recomputes the same interval several times."""
    key = tuple(deltas)
    if key not in _BOOTSTRAP_MEMO:
        _BOOTSTRAP_MEMO[key] = paired_cluster_bootstrap(
            list(key), seed=BOOTSTRAP_SEED, n_resamples=BOOTSTRAP_N_RESAMPLES)
    return _BOOTSTRAP_MEMO[key]


class AblationEvaluator:
    """Stage 2 of one candidate, look by look (ADR-0012 D4). ``play_look(k,
    runner)`` plays look k's games through ``runner`` (round R_k's runner,
    which serves the without-trait side from the round's cells); once a look
    decides, ``evaluation()`` builds the evidence and the child record."""

    def __init__(self, *, candidate_envelope: dict, registry,
                 base_config: RuntimeConfig, budgets: Budgets, parent_cig_id: str,
                 mining_seed: int, allocation: AblationAllocation,
                 target_regulation=None, bootstrap_seed: int = BOOTSTRAP_SEED):
        # central preflight BEFORE any episode (zero runner calls on invalid setup)
        validate_gate_setup(parent_cig_id, mining_seed)
        if isinstance(bootstrap_seed, bool) or not isinstance(bootstrap_seed, int):
            raise EvaluationSetupError(
                f"bootstrap seed must be an explicit int, got {bootstrap_seed!r}")
        if bootstrap_seed != BOOTSTRAP_SEED:
            raise EvaluationSetupError(
                f"bootstrap seed {bootstrap_seed} differs from the pre-registered "
                f"seed {BOOTSTRAP_SEED} (ADR-0005; EXP-0001 manifest)")
        if not isinstance(allocation, AblationAllocation):
            raise EvaluationSetupError("stage 2 needs a frozen AblationAllocation")
        from inheritance.generalization import resolve_frozen_applicability
        from inheritance.overlay import evaluation_settings
        applicability = resolve_frozen_applicability(candidate_envelope, registry)
        if tuple(applicability) != allocation.families:
            raise EvaluationSetupError(
                f"the stage-2 allocation is over {list(allocation.families)}, the candidate's "
                f"frozen applicability is {list(applicability)} (ADR-0012 D4)")
        self._budgets = budgets
        self._parent_cig_id = parent_cig_id
        self._mining_seed = mining_seed
        self._allocation = allocation
        self._configs = build_evaluation_configs(
            base_config, candidate_envelope, registry, target_regulation=target_regulation)
        self._gene = self._configs.target_gene_id
        self._settings = evaluation_settings(max_tokens=budgets.max_tokens_per_request,
                                             seed=mining_seed)
        self._outcomes: list[dict] = []
        self._looks: list[dict] = []
        self._stop: dict | None = None

    @property
    def gene_id(self) -> str:
        return self._gene

    @property
    def stopped(self) -> bool:
        return self._stop is not None

    @property
    def next_look(self) -> int | None:
        return None if self.stopped else len(self._looks) + 1

    def play_look(self, look: int, runner: GateEpisodeRunner) -> dict:
        """Play look ``look``'s games in round R_look and decide; returns the
        look's decision (``continue``, ``pass`` or ``fail``)."""
        if look != self.next_look:
            raise RunnerContractError(f"{self._gene}: look {look} requested, the next look is "
                                      f"{self.next_look} (stage 2 plays its looks in order and "
                                      f"never after its stop)")
        gate_round = look_round(look)
        played: list[dict] = []  # kept only once the whole look has played
        for task in self._allocation.look_tasks(look):
            with_trait = runner.run_episode(task, self._configs.with_trait, self._budgets,
                                            "with_trait", self._settings)
            validate_outcome(with_trait, task.task_id, "with_trait", settings=self._settings)
            cell = runner.run_episode(task, self._configs.without_trait, self._budgets,
                                      CELL_CONDITION, self._settings)
            validate_outcome(cell, task.task_id, CELL_CONDITION, settings=self._settings)
            if self._gene in cell.expressed_gene_ids:
                # contaminated intervention: INVALID evidence, no child record
                raise EvaluationIntegrityError(
                    f"the G0 cell of {task.task_id!r} (round {gate_round}) expressed the target "
                    f"gene {self._gene!r}: the paired intervention is contaminated and produced "
                    f"no child CIG record")
            played.append({
                "task_id": task.task_id, "family": task.family, "look": look,
                "round": gate_round,
                "with_trait": _episode(with_trait),
                "cell": {"seed": self._mining_seed, "round": gate_round, **_episode(cell)},
                "fired": self._gene in with_trait.expressed_gene_ids})
        self._outcomes.extend(played)
        pairs = [(o["with_trait"]["success"], o["cell"]["success"]) for o in self._outcomes]
        n_plus = sum(1 for w, c in pairs if w and not c)
        n_minus = sum(1 for w, c in pairs if c and not w)
        decision = look_decision(look, n_plus, n_minus)
        self._looks.append(decision)
        if decision["decision"] != "continue":
            self._stop = decision
        return dict(decision)

    def evaluation(self) -> AblationEvaluation:
        """The stage-2 evidence once a look has decided."""
        if self._stop is None:
            raise RunnerContractError(f"{self._gene}: stage 2 has not decided yet (looks "
                                      f"played: {len(self._looks)})")
        ordered = sorted((o["task_id"], float(o["with_trait"]["success"])
                          - float(o["cell"]["success"])) for o in self._outcomes)
        deltas = [d for _, d in ordered]
        bootstrap = descriptive_bootstrap(deltas)
        block = {"method": bootstrap.method, "seed": bootstrap.seed,
                 "n_resamples": bootstrap.n_resamples, "cluster_order": BOOTSTRAP_CLUSTER_ORDER,
                 "confidence": CONFIDENCE, "point_estimate": bootstrap.point_estimate,
                 "lower_bound": bootstrap.lower_bound, "upper_bound": bootstrap.upper_bound,
                 "descriptive": True}
        passed = self._stop["decision"] == "pass"
        alpha_met = any(look["alpha_met"] for look in self._looks)
        contribution = self._stop["point"]
        looks = [look_record(look, look_round(look["look"])) for look in self._looks]
        stop = {"look": self._stop["look"], "round": look_round(self._stop["look"]),
                "games": self._stop["games"], "decision": self._stop["decision"],
                "reason": self._stop["reason"]}
        child = {
            "cig_id": f"{self._parent_cig_id}/S{self._mining_seed}-ablation",
            "parent_cig_id": self._parent_cig_id,
            "seed": self._mining_seed,
            "stage": "ablation",
            "measurements": {
                "candidate_gene_id": self._gene,
                **self._configs.binding(),  # ADR-0010
                "allocation": self._allocation.allocation_metadata(),
                "paired_outcomes": copy.deepcopy(self._outcomes),
                "looks": looks,
                "stop": stop,
                "alpha_met": alpha_met,
                "contribution": exact(contribution),
                "point_estimate_delta_success": float(contribution),
                # descriptive (ADR-0012 D4): the deltas in task_id order
                "paired_deltas": deltas,
                "paired_delta_task_ids": [task_id for task_id, _ in ordered],
                "bootstrap": {k: v for k, v in block.items() if k != "point_estimate"},
                "tau_c": exact(TAU_C),
                "thresholds": thresholds_record(),
                "deterministic_execution_policy": {
                    "runs_per_task_condition": 1, "eval_temperature": 0.0,
                    "paired_conditions": ["with_trait", "g0_cell"],
                    "cells": "one G0 execution per (seed, round, game), read only in its "
                             "round (ADR-0012 D2)",
                },
                "eval_generation_settings": {
                    "temperature": self._settings.temperature,
                    "max_tokens": self._settings.max_tokens,
                    "seed": self._settings.seed,
                    "stop": list(self._settings.stop),
                },
                "stage_pass": passed,
            },
        }
        load_cig(child)  # schema-valid or fail closed
        return AblationEvaluation(
            _passed=passed, _alpha_met=alpha_met, _stop=stop,
            _looks=tuple(copy.deepcopy(looks)), _contribution=contribution,
            _paired_outcomes=tuple(copy.deepcopy(self._outcomes)), _bootstrap=block,
            _child_record=child, _allocation=self._allocation.allocation_metadata())


def _episode(outcome) -> dict:
    return {"success": outcome.success, "expressed_gene_ids": list(outcome.expressed_gene_ids),
            "trajectory_id": outcome.trajectory_id, "status": outcome.status}


def thresholds_record() -> dict:
    """The stage-2 rule as every child records it (ADR-0012 D4, D10)."""
    return {"rule": ABLATION_THRESHOLD_TEXT, "looks": list(LOOKS),
            "alphas": [exact(a) for a in ALPHAS], "tau_c": exact(TAU_C),
            "futility": f"net <= floor(n_k / {FUTILITY_DIVISOR}) at looks 1-{len(LOOKS) - 1}"}


def _is_int(value, expected: int) -> bool:
    """Strict: True == 1 and 1.0 == 1 must not pass."""
    return type(value) is int and value == expected


def stage2_evidence_issues(measurements: Mapping, *, mining_seed: int | None = None,
                           families: Sequence[str] | None = None) -> list[str]:
    """ADR-0012 D4: recompute the stage-2 decision from the record alone. The
    allocation must be 192 distinct games interleaved over its families in
    name order (``families``, when given, the candidate's frozen
    applicability); the paired outcomes must be its first games in order,
    each read against the cell of its own look's round (D2), and no cell may
    have expressed the target gene; every look's counts, p_k, decision, the
    stop, ``alpha_met``, the contribution, ``stage_pass`` and the recorded
    rule must be the rule's; the descriptive bootstrap must be on the locked
    ADR-0005 values and recompute. Returns the issues (empty if none)."""
    issues: list[str] = []
    allocation = measurements.get("allocation")
    outcomes = measurements.get("paired_outcomes")
    if not isinstance(allocation, Mapping) or not isinstance(allocation.get("task_ids"), list) \
            or len(allocation["task_ids"]) != ALLOCATION_SIZE:
        return [f"stage 2: the allocation does not list {ALLOCATION_SIZE} games"]
    task_ids = allocation["task_ids"]
    if not all(isinstance(t, str) for t in task_ids) or len(set(task_ids)) != len(task_ids):
        return ["stage 2: the allocation's games are not distinct string ids (no replacement "
                "within a candidate, ADR-0012 D4)"]
    slots = allocation.get("families")
    if not isinstance(slots, list) or not slots or not all(isinstance(f, str) for f in slots) \
            or slots != sorted(set(slots)):
        return ["stage 2: the allocation's families are not distinct names in name order"]
    if families is not None and slots != sorted(families):
        issues.append(f"stage 2: the allocation is over {slots}, the candidate's frozen "
                      f"applicability is {sorted(families)} (ADR-0012 D4)")
    if not isinstance(outcomes, list) or not outcomes:
        return ["stage 2: the record holds no paired outcomes"]
    if len(outcomes) > ALLOCATION_SIZE:
        return [f"stage 2: the record holds {len(outcomes)} paired outcomes, more than the "
                f"{ALLOCATION_SIZE} games of the allocation"]
    gene = measurements.get("candidate_gene_id")
    pairs: list[tuple[bool, bool]] = []
    for index, outcome in enumerate(outcomes):
        if not isinstance(outcome, Mapping):
            return [f"stage 2: paired outcome {index} is malformed"]
        with_trait, cell = outcome.get("with_trait"), outcome.get("cell")
        if not isinstance(with_trait, Mapping) or not isinstance(cell, Mapping) \
                or type(with_trait.get("success")) is not bool \
                or type(cell.get("success")) is not bool \
                or not isinstance(cell.get("expressed_gene_ids"), list):
            return [f"stage 2: paired outcome {index} lacks boolean with-trait and cell outcomes "
                    f"and the cell's expressed genes"]
        if outcome.get("task_id") != task_ids[index]:
            issues.append(f"stage 2: game {index} is {outcome.get('task_id')!r}, the allocation's "
                          f"game {index} is {task_ids[index]!r}")
        if outcome.get("family") != slots[index % len(slots)]:
            issues.append(f"stage 2: game {index} is of family {outcome.get('family')!r}, its "
                          f"slot belongs to {slots[index % len(slots)]!r}")
        look = next(k for k, n_k in enumerate(LOOKS, start=1) if index < n_k)
        expected_round = look_round(look)
        if not (_is_int(outcome.get("look"), look) and _is_int(outcome.get("round"), expected_round)
                and _is_int(cell.get("round"), expected_round)):
            issues.append(f"stage 2: game {index} belongs to look {look} and round "
                          f"{expected_round}; the record says look {outcome.get('look')!r}, "
                          f"round {outcome.get('round')!r}, cell round {cell.get('round')!r} "
                          f"(ADR-0012 D2: a cell is read only in its round)")
        if mining_seed is not None and not _is_int(cell.get("seed"), mining_seed):
            issues.append(f"stage 2: game {index} read a cell of seed {cell.get('seed')!r}, not "
                          f"{mining_seed}")
        if gene is not None and gene in cell["expressed_gene_ids"]:
            issues.append(f"stage 2: the cell of game {index} expressed the target gene "
                          f"{gene!r} (a contaminated intervention)")
        pairs.append((with_trait["success"], cell["success"]))
    if measurements.get("tau_c") != exact(TAU_C) \
            or measurements.get("thresholds") != thresholds_record():
        issues.append("stage 2: the recorded rule is not ADR-0012 D4's")
    looks, stop = sequential_path(pairs)
    if stop is None or stop["games"] != len(pairs):
        issues.append(f"stage 2: the rule does not stop after the {len(pairs)} games recorded "
                      f"(it stops after {stop['games'] if stop else 'more than ' + str(len(pairs))})")
        return issues
    recorded = measurements.get("looks")
    expected = [look_record(look, look_round(look["look"])) for look in looks]
    if recorded != expected:
        issues.append("stage 2: the recorded looks are not the rule's recomputed looks")
    expected_stop = {"look": stop["look"], "round": look_round(stop["look"]),
                     "games": stop["games"], "decision": stop["decision"],
                     "reason": stop["reason"]}
    if measurements.get("stop") != expected_stop:
        issues.append(f"stage 2: the recorded stop {measurements.get('stop')!r} is not the "
                      f"rule's {expected_stop!r}")
    alpha_met = any(look["alpha_met"] for look in looks)
    if measurements.get("alpha_met") is not alpha_met:
        issues.append(f"stage 2: alpha_met {measurements.get('alpha_met')!r} != recomputed "
                      f"{alpha_met}")
    if parse_exact(measurements.get("contribution")) != stop["point"] \
            or measurements.get("point_estimate_delta_success") != float(stop["point"]):
        issues.append("stage 2: the contribution is not the point at the stopping look")
    if measurements.get("stage_pass") is not (stop["decision"] == "pass"):
        issues.append(f"stage 2: stage_pass {measurements.get('stage_pass')!r} != the rule's "
                      f"{stop['decision'] == 'pass'}")
    issues.extend(_bootstrap_issues(measurements, outcomes))
    return issues


def _bootstrap_issues(measurements: Mapping, outcomes: list) -> list[str]:
    """The descriptive interval: on the locked ADR-0005 values, over the
    games played in task_id order, recomputing exactly."""
    issues: list[str] = []
    block = measurements.get("bootstrap")
    if not isinstance(block, Mapping):
        return ["stage 2: the descriptive bootstrap is missing"]
    locked = {"seed": BOOTSTRAP_SEED, "n_resamples": BOOTSTRAP_N_RESAMPLES,
              "cluster_order": BOOTSTRAP_CLUSTER_ORDER, "method": BOOTSTRAP_METHOD,
              "confidence": CONFIDENCE, "descriptive": True}
    for key, value in locked.items():
        got = block.get(key)
        if type(got) is not type(value) or got != value:  # strict: True == 1 must not pass
            issues.append(f"stage 2 bootstrap.{key} {got!r} != pre-registered {value!r} "
                          f"(ADR-0005)")
    if not all(isinstance(o.get("task_id"), str) for o in outcomes):
        return issues + ["stage 2: a paired outcome's task id is not a string (ADR-0005 orders "
                         "the deltas by task id)"]
    ordered = sorted((o["task_id"], float(o["with_trait"]["success"])
                      - float(o["cell"]["success"])) for o in outcomes)
    if measurements.get("paired_delta_task_ids") != [t for t, _ in ordered] \
            or measurements.get("paired_deltas") != [d for _, d in ordered]:
        issues.append("stage 2: the paired deltas are not the games played, in task_id order")
        return issues
    bootstrap = descriptive_bootstrap([d for _, d in ordered])
    for key in ("lower_bound", "upper_bound"):
        if block.get(key) != getattr(bootstrap, key):
            issues.append(f"stage 2 bootstrap.{key} {block.get(key)!r} != recomputed "
                          f"{getattr(bootstrap, key)!r}")
    return issues


def tau_c_sensitivity(pairs: Sequence[tuple[bool, bool]], tau_c: Fraction) -> dict:
    """ADR-0012 D7: the stage-2 decision under another τ_c, recomputed on the
    recorded path. Not identifiable when that τ_c would have continued past
    the observed stopping look."""
    _, stop = sequential_path(pairs, tau_c=tau_c)
    if stop is None:  # an earlier stop under this tau_c needs no more games
        return {"identifiable": False, "pass": None, "stop_look": None, "contribution": None,
                "reason": "this tau_c would have continued past the observed stopping look"}
    return {"identifiable": True, "pass": stop["decision"] == "pass", "stop_look": stop["look"],
            "contribution": exact(stop["point"]), "reason": stop["reason"]}
