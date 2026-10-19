"""CIG gate orchestrator (issue #12; recalibrated by ADR-0012, owner-accepted
2026-10-19): stage 3/4 + aggregate verdict + somatic terminal decision.

A seed's gate runs in six rounds (ADR-0012 D2). Rounds R1-R4 (stage 1 and
stage 2's looks) are the caller's; this orchestrator takes their evidence
and runs R5 and R6 through the runner of each round, which serves every
without-trait and neither side from that round's G0 cells.

Orchestration order for a frozen candidate set:
1. validate inputs — all before any episode: unique frozen ranks, unique
   gene ids, UNIQUE resolved aggregate CIG ids, a supplied TokenCounter,
   every candidate's evaluation config buildable, and, when both stores
   are supplied (the run will commit), no aggregate id already live and
   every candidate still the stored somatic ``candidate`` with exactly the
   evaluated reference;
2. consume the stage-1 and stage-2 children (never rerun): structural
   ``load_cig`` validation, exact ``<parent>/S<seed>-<stage>`` ids, the
   locked settings, ADR-0010 B3's binding, stage 1's expression rule (D3)
   and stage 2's sequential decision recomputed from its record (D4) —
   failing closed before any R5 episode;
3. stage 3 for ALL candidates: the in-scope child by reference to stage 2
   (D5); the out-of-scope check in R5 for every candidate whose stage-2
   p_k met α_k at some look, a vacuous pass for an all-family one, and a
   not-run child for the others (D7);
4. 4A in R5 for the same candidates, paired against the seed's R5 T_reg
   cells (played only if some candidate reaches R5); a not-run 4A for the
   others; 4B (overhead) for every candidate;
5. preliminary eligible set (every stage passes);
6. R6: every unordered pair of that set evaluated once, both against the
   seed's R6 cells, gene ids in canonical order;
7. harmful pairs resolved simultaneously from COMPLETE evidence: the
   member with the lower stage-2 contribution (the point at its stopping
   look) is eliminated; exact tie -> earlier frozen rank; ambiguous rank
   -> fail closed;
8. stage-4 child for every candidate (load_cig-validated);
9. final aggregate + verdict per candidate;
10. prevalidated batch commit when stores are supplied: every bundle and
    every somatic decision is prevalidated against a staged clone of the
    CURRENT live CIG store; only after all prevalidations pass are the
    live stores mutated (somatic decision first, then the CIG bundle). The
    mutation phase is not transactional (see ``_commit_results``).

A scientifically valid stage FAIL still produces immutable child evidence.
Only integrity/setup/runtime failures abort. Sensitivity is analysis-only
pure recomputation with explicit per-cell identifiability (D7). Boundaries:
no germline assimilation, no T_eval access, no threshold retuning, and no
"v1 shadow" verdict (D10).
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations

from genome.validation.errors import RecordConsistencyError
from genome.validation.loader import load_cig
from genome.validation.records import CigRecordStore
from runtime.expression import RuntimeConfig
from runtime.loop import Budgets
from runtime.policies.planner_react import STOP_SEQUENCES

from inheritance.ablation import (
    ALPHAS,
    BOOTSTRAP_CLUSTER_ORDER,
    BOOTSTRAP_N_RESAMPLES,
    BOOTSTRAP_SEED,
    FUTILITY_DIVISOR,
    LOOKS,
    TAU_C,
    AblationEvaluation,
    exact,
    stage2_evidence_issues,
    tau_c_sensitivity,
)
from inheritance.bootstrap import CONFIDENCE
from inheritance.generalization import (
    IN_SCOPE_MIN_DELTA,
    NOT_RUN_REASON,
    NOT_RUN_THRESHOLD_TEXT,
    OUT_SCOPE_COUNT,
    OUT_SCOPE_MIN_DELTA,
    OUT_SCOPE_MIN_DELTA_EXACT,
    GeneralizationEvaluation,
    GeneralizationEvaluator,
    resolve_frozen_applicability,
)
from inheritance.interaction import HARMFUL_PAIR_DELTA, InteractionEvaluator
from inheritance.overlay import (
    build_evaluation_configs,
    evaluation_settings,
)
from inheritance.regression import (
    REGRESSION_THRESHOLD_TEXT,
    TAU_K_TOKENS,
    TAU_R,
    RegressionEvaluation,
    RegressionEvaluator,
    TokenCounter,
    measure_expression_overhead,
    paired_drop_within,
    run_regression_baseline,
    validate_token_counter,
)
from inheritance.replay import REPLAY_THRESHOLD_TEXT, ReplayEvaluation
from inheritance.runner import validate_gate_setup

THRESHOLD_VERSION = "EXP-0001-CIG-M0-v2"  # ADR-0012 D10
PRIMARY_THRESHOLDS = {
    "replay_rule": "source task once with the trait: gene expressed or episode failed (D3)",
    "ablation_looks": list(LOOKS),
    "ablation_alphas": [f"{a.numerator}/{a.denominator}" for a in ALPHAS],
    "tau_c": float(TAU_C),
    "ablation_futility": f"net <= floor(n_k/{FUTILITY_DIVISOR}) at looks 1-{len(LOOKS) - 1}",
    "in_scope_min_delta": IN_SCOPE_MIN_DELTA,
    "in_scope_rule": "by reference to stage 2's games (D5)",
    "out_scope_games": OUT_SCOPE_COUNT,
    "out_scope_min_delta": OUT_SCOPE_MIN_DELTA,
    "tau_r": TAU_R,
    "harmful_pair_delta": HARMFUL_PAIR_DELTA,
    "tau_k_tokens": TAU_K_TOKENS,
    "rounds": "R1 stage 1 and look 1; R2-R4 looks 2-4; R5 out-of-scope and 4A; R6 4C; one G0 "
              "cell per (seed, round, game), read only in its round (D2)",
    "bootstrap_confidence": CONFIDENCE,    # descriptive since ADR-0012 D4
    "bootstrap_resamples": BOOTSTRAP_N_RESAMPLES,
    "bootstrap_seed": BOOTSTRAP_SEED,                  # ADR-0005, pre-registered
    "bootstrap_cluster_order": BOOTSTRAP_CLUSTER_ORDER,  # ADR-0005
}
SENSITIVITY_TAU_C_GRID = (0.03, 0.05, 0.08)
SENSITIVITY_TAU_R_GRID = (0.02, 0.03, 0.05)
REPLAY_STAGE_TEXT = REPLAY_THRESHOLD_TEXT
ABLATION_STAGE_TEXT = (f"exact paired sign test at looks {'/'.join(map(str, LOOKS))}: p_k <= "
                       f"alpha_k ({', '.join(map(str, ALPHAS))}) and point >= {float(TAU_C)}; "
                       f"futility net <= floor(n/{FUTILITY_DIVISOR}) (ADR-0012 D4)")
IN_SCOPE_STAGE_TEXT = (f"delta_in >= {IN_SCOPE_MIN_DELTA} over stage 2's games (by reference, "
                       f"ADR-0012 D5)")
OUT_SCOPE_STAGE_TEXT = (f"delta_out >= {OUT_SCOPE_MIN_DELTA} over {OUT_SCOPE_COUNT} games, "
                        f"exact: net loss <= {-int(OUT_SCOPE_MIN_DELTA_EXACT * OUT_SCOPE_COUNT)} "
                        f"(ADR-0012 D5)")
VACUOUS_OUT_SCOPE_STAGE_TEXT = "vacuous: no out-of-scope family (ADR-0007 E6)"
STAGE4_TEXT = (f"paired regression drop <= {TAU_R} against the R5 cells (ADR-0009 D1; "
               f"ADR-0012 D2), overhead <= {TAU_K_TOKENS} tokens, not eliminated")
STAGE4_NOT_RUN_TEXT = (f"4A and 4C not run: stage 2 never met alpha_k (ADR-0012 D7); overhead "
                       f"<= {TAU_K_TOKENS} tokens")
VACUOUS_INTERACTION_REASON = (
    "vacuous: fewer than two preliminary-eligible candidates, so no "
    "both-vs-neither interaction contrast is defined for this candidate")
INELIGIBLE_INTERACTION_REASON = (
    "not evaluated: the candidate is not preliminary-eligible, so it enters no "
    "both-vs-neither pair")


class GateInputError(ValueError):
    """Invalid orchestrator input (fail closed before any episode)."""


class EvidenceCoherenceError(ValueError):
    """#11 evidence does not cohere with this candidate/aggregate (fail
    closed; replay/ablation are never rerun)."""


def ablation_evidence_issues(measurements: dict, evaluated: AblationEvaluation | None = None,
                             *, mining_seed: int | None = None,
                             families=None) -> list[str]:
    """ADR-0012 D4 (as ADR-0009 D5 did for the bootstrap): a stage-2 verdict
    is evidence only if it recomputes from its record. The sequential
    decision (per-look counts, exact p_k, efficacy and futility, the stop,
    ``alpha_met``, the contribution and ``stage_pass``) is recomputed from
    the paired outcomes, each of which must read the cell of its own look's
    round; the allocation must be over ``families`` (the candidate's frozen
    applicability, when given); the descriptive bootstrap must be on the
    locked ADR-0005 values and recompute. ``evaluated`` is the evaluation the
    gate consumes, whose pass, ``alpha_met``, contribution, looks, stop and
    paired outcomes (which the in-scope child and the sensitivity grid read)
    must be the record's; a resumed run passes None."""
    issues = stage2_evidence_issues(measurements, mining_seed=mining_seed, families=families)
    if evaluated is not None and not issues:
        if evaluated.passed is not measurements.get("stage_pass") \
                or evaluated.alpha_met is not measurements.get("alpha_met") \
                or exact(evaluated.contribution) != measurements.get("contribution") \
                or evaluated.looks != measurements.get("looks") \
                or evaluated.stop != measurements.get("stop") \
                or evaluated.paired_outcomes != measurements.get("paired_outcomes"):
            issues.append("the stage-2 evaluation's pass, alpha_met, contribution, looks, stop "
                          "or paired outcomes are not its record's (ADR-0012 D4)")
    return issues


@dataclass(frozen=True)
class GateCandidateInput:
    """One frozen candidate: its rank, its completed stage-1 and stage-2
    evidence (rounds R1-R4), its own CIG run identity (parent id + mining
    seed) and the frozen out-of-scope and T_reg allocations. Ranks must be
    unique non-negative ints."""

    candidate_envelope: dict
    frozen_rank: int
    replay_evaluation: ReplayEvaluation
    ablation_evaluation: AblationEvaluation
    generalization_allocation: object   # OutOfScopeAllocation (ADR-0012 D5)
    regression_allocation: object
    parent_cig_id: str | None = None    # falls back to batch default
    mining_seed: int | None = None      # falls back to batch default

    def __post_init__(self):
        object.__setattr__(self, "candidate_envelope",
                            copy.deepcopy(self.candidate_envelope))
        if isinstance(self.frozen_rank, bool) or not isinstance(self.frozen_rank, int) \
                or self.frozen_rank < 0:
            raise GateInputError(
                f"frozen rank {self.frozen_rank!r} must be a non-negative integer")


@dataclass(frozen=True)
class CandidateGateResult:
    """Immutable per-candidate gate outcome (deep-copy accessors)."""

    _gene_id: str
    _candidate_version: int
    _candidate: dict
    _verdict: str
    _aggregate: dict
    _stage4_child: dict
    _stage_results: dict
    _eliminated_by: str | None
    _parent_cig_id: str

    @property
    def gene_id(self) -> str:
        return self._gene_id

    @property
    def candidate_version(self) -> int:
        """Somatic candidate version, derived from the evaluated envelope."""
        return self._candidate_version

    @property
    def candidate(self) -> dict:
        """The evaluated candidate gene reference (gene_id, version, type,
        content-addressed artifact, origin, provenance) — the commit binds the
        somatic decision to exactly this reference."""
        return copy.deepcopy(self._candidate)

    @property
    def verdict(self) -> str:
        return self._verdict

    @property
    def aggregate(self) -> dict:
        return copy.deepcopy(self._aggregate)

    @property
    def stage4_child(self) -> dict:
        return copy.deepcopy(self._stage4_child)

    @property
    def stage_results(self) -> dict:
        return copy.deepcopy(self._stage_results)

    @property
    def eliminated_by(self) -> str | None:
        return self._eliminated_by

    @property
    def parent_cig_id(self) -> str:
        return self._parent_cig_id

    @property
    def aggregate_cig_id(self) -> str:
        return self._aggregate["cig_id"]


@dataclass(frozen=True)
class GateOutcome:
    """Immutable full-set orchestration outcome. ``committed`` is True when
    stores were supplied and the prevalidated batch commit completed."""

    _results: tuple[CandidateGateResult, ...]
    _parent_cig_id: str
    _mining_seed: int
    _committed: bool = False

    @property
    def results(self) -> tuple[CandidateGateResult, ...]:
        return tuple(self._results)

    @property
    def parent_cig_id(self) -> str:
        return self._parent_cig_id

    @property
    def mining_seed(self) -> int:
        return self._mining_seed

    @property
    def committed(self) -> bool:
        return self._committed

    @property
    def verdicts(self) -> dict[str, str]:
        return {r.gene_id: r.verdict for r in self._results}


class CigGateOrchestrator:
    """Evaluates stage 3/4 for every candidate (rounds R5 and R6), resolves
    interactions, builds aggregates/verdicts, and (when stores are
    supplied) commits after batch prevalidation: somatic decisions first,
    CIG bundles exposed on the live store only after.

    ``runner`` is either one runner used for every round (unit tests) or a
    callable ``round -> runner`` returning round R5's and R6's runners, each
    of which serves the without-trait and neither sides from its round's G0
    cells (ADR-0012 D2)."""

    def __init__(self, *, registry, base_config: RuntimeConfig,
                 runner, budgets: Budgets,
                 parent_cig_id: str = "CIG-0007", mining_seed: int = 0,
                 token_counter: TokenCounter | None = None,
                 interaction_allocation=None,
                 somatic_store=None, cig_record_store: CigRecordStore | None = None,
                 target_regulation_by_gene: dict | None = None):
        validate_gate_setup(parent_cig_id, mining_seed)
        self._registry = registry
        self._base_config = base_config
        self._runner = runner
        self._budgets = budgets
        self._parent_cig_id = parent_cig_id  # batch default
        self._mining_seed = mining_seed      # batch default
        if token_counter is not None:
            validate_token_counter(token_counter)
        self._token_counter = token_counter
        self._interaction_allocation = interaction_allocation
        self._somatic_store = somatic_store
        self._cig_record_store = cig_record_store
        self._target_regulation_by_gene = dict(target_regulation_by_gene or {})
        self._last: _LastRun | None = None

    def _commits(self) -> bool:
        """A run commits only when BOTH stores are supplied (step 10)."""
        return self._somatic_store is not None and self._cig_record_store is not None

    def _round_runner(self, gate_round: int):
        if hasattr(self._runner, "run_episode"):
            return self._runner
        return self._runner(gate_round)

    # -- public entry point ---------------------------------------------------
    def run(self, inputs: list[GateCandidateInput]) -> GateOutcome:
        self._last = None  # sensitivity never answers from an earlier or aborted run
        inputs = tuple(inputs)  # snapshot: later caller mutation cannot alias the run
        ranks = [i.frozen_rank for i in inputs]
        if len(set(ranks)) != len(ranks) or any(r < 0 for r in ranks):
            raise GateInputError(
                f"frozen ranks must be unique non-negative integers, got {ranks}")
        validated = [self._validated_of(i.candidate_envelope) for i in inputs]
        gene_ids = [v["candidate"]["gene_id"] for v in validated]
        if len(set(gene_ids)) != len(gene_ids):
            dupes = sorted({g for g in gene_ids if gene_ids.count(g) > 1})
            raise GateInputError(f"duplicate candidate gene ids {dupes}")
        identities = [self._parent_seed_for(inp) for inp in inputs]
        parents = [p for p, _ in identities]
        if len(set(parents)) != len(parents):
            dupes = sorted({p for p in parents if parents.count(p) > 1})
            raise GateInputError(
                f"candidates must resolve to UNIQUE aggregate CIG ids (one "
                f"aggregate per candidate); duplicates: {dupes}")
        # setup that needs no episode fails BEFORE any episode
        if inputs and self._token_counter is None:
            raise GateInputError(
                "stage-4B expression overhead requires a TokenCounter; "
                "no counter was supplied")
        for inp in inputs:  # overlay validity (e.g. target already in base)
            self._configs_by_gene(inp)
        if self._commits():  # the same condition under which step 10 commits
            existing = sorted(set(parents) & set(self._cig_record_store.aggregates))
            if existing:
                raise GateInputError(
                    f"aggregate CIG ids already exist in the live store: {existing}")
            for candidate in (v["candidate"] for v in validated):
                issue = _somatic_binding_issue(self._somatic_store, candidate)
                if issue:
                    raise RecordConsistencyError(issue)

        # step 2: consume the stage-1/2 evidence, fail closed on incoherence —
        # BEFORE any R5 episode (ADR-0010 B3; ADR-0012 D4)
        for inp in inputs:
            self._consume_stage12_evidence(inp)
        # last setup check: a closed or stale journal handle would refuse the
        # step-10 appends (an empty batch appends nothing)
        if inputs and self._commits():
            self._somatic_store.check_appendable()

        # step 3, R5: stage 3 for ALL candidates (the out-of-scope check plays
        # only for those whose stage-2 alpha was met, ADR-0012 D7)
        reaching = [inp.ablation_evaluation.alpha_met for inp in inputs]
        runner5 = self._round_runner(5) if any(reaching) else None
        gen_results: dict[str, GeneralizationEvaluation] = {}
        for inp, (parent, seed) in zip(inputs, identities):
            gid = inp.candidate_envelope["candidate"]["gene_id"]
            evaluator = GeneralizationEvaluator(
                candidate_envelope=inp.candidate_envelope, registry=self._registry,
                base_config=self._base_config, budgets=self._budgets,
                parent_cig_id=parent, mining_seed=seed,
                target_regulation=self._target_regulation_by_gene.get(gid))
            gen_results[gid] = evaluator.evaluate(
                inp.ablation_evaluation, inp.generalization_allocation,
                runner5 if inp.ablation_evaluation.alpha_met else None)

        # step 4, R5: the seed's T_reg cells (only if some candidate reaches
        # R5), then 4A for those candidates; 4B for every candidate
        reg_results: dict[str, RegressionEvaluation | None] = {}
        overhead_results: dict[str, dict] = {}
        baselines: dict[tuple, object] = {}

        def baseline_key(inp, seed):
            return seed, tuple(t.task_id for t in inp.regression_allocation.tasks)

        for inp, (_, seed) in zip(inputs, identities):
            if not inp.ablation_evaluation.alpha_met:
                continue
            key = baseline_key(inp, seed)
            if key not in baselines:
                baselines[key] = run_regression_baseline(
                    base_config=self._base_config, runner=runner5,
                    budgets=self._budgets, mining_seed=seed,
                    allocation=inp.regression_allocation)
        for inp, (parent, seed) in zip(inputs, identities):
            gid = inp.candidate_envelope["candidate"]["gene_id"]
            if inp.ablation_evaluation.alpha_met:
                reg_evaluator = RegressionEvaluator(
                    candidate_envelope=inp.candidate_envelope, registry=self._registry,
                    base_config=self._base_config, runner=runner5,
                    budgets=self._budgets, parent_cig_id=parent,
                    mining_seed=seed,
                    target_regulation=self._target_regulation_by_gene.get(gid))
                reg_results[gid] = reg_evaluator.evaluate(
                    inp.regression_allocation, baselines[baseline_key(inp, seed)])
            else:
                reg_results[gid] = None  # not run (ADR-0012 D7)
            target = [s for s in self._configs_by_gene(inp).with_trait.skills
                      if s.gene_id == gid][0]
            oh = measure_expression_overhead(target, self._token_counter)
            overhead_results[gid] = {
                "tokens": oh.tokens, "threshold": oh.threshold,
                "passed": oh.passed, "counter_identity": oh.counter_identity,
                "rendered_sha256": oh.rendered_sha256,
            }

        # step 5: preliminary eligible set — every stage passes
        eligible: list[str] = []
        for inp, gid in zip(inputs, gene_ids):
            regression = reg_results[gid]
            if (inp.replay_evaluation.passed
                    and inp.ablation_evaluation.passed
                    and gen_results[gid].passed
                    and regression is not None and regression.passed
                    and overhead_results[gid]["passed"]):
                eligible.append(gid)

        # step 6, R6: interaction pairs, computed once, resolved together
        pair_evidence: dict[tuple[str, str], object] = {}
        if len(eligible) >= 2:
            if self._interaction_allocation is None:
                raise GateInputError(
                    ">=2 preliminary-eligible candidates require an interaction "
                    "allocation; none was supplied")
            ordered = sorted(
                (inp for inp, gid in zip(inputs, gene_ids) if gid in eligible),
                key=lambda i: i.frozen_rank)
            seeds = {self._parent_seed_for(inp)[1] for inp in ordered}
            if len(seeds) != 1:
                raise GateInputError(f"the eligible candidates come from mining seeds "
                                     f"{sorted(seeds)}; a seed's pairs read that seed's R6 "
                                     f"cells (ADR-0012 D2)")
            interaction = InteractionEvaluator(
                registry=self._registry, base_config=self._base_config,
                runner=self._round_runner(6), budgets=self._budgets,
                target_regulation_by_gene=self._target_regulation_by_gene)
            locked = evaluation_settings(
                max_tokens=self._budgets.max_tokens_per_request, seed=seeds.pop())
            for a, b in combinations(ordered, 2):
                evidence = interaction.evaluate_pair(
                    a.candidate_envelope, b.candidate_envelope,
                    self._interaction_allocation, locked)
                key = tuple(sorted(evidence.gene_ids))
                pair_evidence[key] = evidence

        # resolve ALL harmful pairs simultaneously from complete evidence; the
        # contribution is stage 2's point at its stopping look (ADR-0012 D6)
        contribution = {gid: inp.ablation_evaluation.contribution
                        for inp, gid in zip(inputs, gene_ids)}
        rank_by_gene = {gid: inp.frozen_rank
                        for inp, gid in zip(inputs, gene_ids)}
        eliminated = self._resolve_harmful_pairs(
            set(eligible), contribution, rank_by_gene, pair_evidence)

        # steps 8-9: stage-4 children + aggregates + verdicts
        results: list[CandidateGateResult] = []
        for inp, gid, (parent, seed) in zip(inputs, gene_ids, identities):
            evaluated = validated[gene_ids.index(gid)]["candidate"]
            version = evaluated["version"]
            stage4 = self._build_stage4_child(
                gid, parent, seed, reg_results[gid], overhead_results[gid],
                pair_evidence, eliminated, eligible=gid in eligible,
                eligible_count=len(eligible), evaluated=evaluated)
            aggregate = self._build_aggregate(
                inp, gid, parent, gen_results[gid], stage4,
                evaluated=evaluated)
            regression = reg_results[gid]
            results.append(CandidateGateResult(
                _gene_id=gid, _candidate_version=version,
                _candidate=copy.deepcopy(evaluated),
                _verdict=aggregate["verdict"],
                _aggregate=aggregate, _stage4_child=stage4,
                _stage_results={
                    "replay": inp.replay_evaluation.passed,
                    "ablation": inp.ablation_evaluation.passed,
                    "generalization_in_scope": gen_results[gid].in_scope_passed,
                    "generalization_out_of_scope": gen_results[gid].out_scope_passed,
                    "regression": regression is not None and regression.passed,
                    "overhead": overhead_results[gid]["passed"],
                    "eliminated": gid in eliminated,
                },
                _eliminated_by=eliminated.get(gid), _parent_cig_id=parent))

        # step 10: prevalidated batch commit when stores are supplied
        last = _LastRun(inputs=inputs, gene_ids=tuple(gene_ids),
                        gen_results=gen_results,
                        reg_results=reg_results,
                        overhead_results=overhead_results,
                        pair_evidence=pair_evidence,
                        eligible=eligible,
                        eliminated=eliminated, outcome=None)
        committed = False
        self._last = last  # _bundle_children reads this run's evidence
        if self._commits():
            try:
                self._commit_results(results,
                                     somatic_store=self._somatic_store,
                                     cig_record_store=self._cig_record_store)
            except BaseException:
                self._last = None  # no sensitivity answers from a failed commit
                raise
            committed = True

        outcome = GateOutcome(_results=tuple(results),
                              _parent_cig_id=self._parent_cig_id,
                              _mining_seed=self._mining_seed,
                              _committed=committed)
        self._last.outcome = outcome
        return outcome

    # -- sensitivity (pure recomputation, zero calls) ---------------------------
    def sensitivity_recomputation(self, gene_id: str) -> dict:
        """ADR-0002 §15 as amended by ADR-0012 D7: analysis-only threshold
        recomputation over the immutable primary evidence (tau_c/tau_r
        grids). ZERO runner/model/environment calls; no mutation of
        children/aggregate/verdicts/SomaticStore; no T_eval dependency.

        tau_c applies only to stage 2, recomputed on the recorded sequential
        path: a candidate is not identifiable under a tau_c that would have
        continued past its observed stopping look (D7), and a cell that needs
        such a candidate's outcome is not identifiable. tau_r applies only to
        4A, where it ran; a candidate whose stage 2 never met alpha fails
        every cell (its stage-3/4A evidence was not collected, and no tau in
        either grid changes that, D7). Replay, the out-of-scope check and the
        overhead stay fixed primary evidence. A cell is also not identifiable
        when the interaction pairs its hypothetical eligible set needs were
        never collected; identifiable cells report the recomposed
        eliminations and promotions, with each candidate's contribution
        taken at its hypothetical stopping look."""
        run = self._last
        if run is None:
            raise GateInputError("no primary run has been executed yet")
        if gene_id not in run.gene_ids:
            raise GateInputError(f"unknown candidate gene id {gene_id!r}")
        replay_pass, gen_out, overhead_ok, reg_counts, pairs = {}, {}, {}, {}, {}
        for inp, gid in zip(run.inputs, run.gene_ids):
            replay_pass[gid] = inp.replay_evaluation.passed
            gen_out[gid] = run.gen_results[gid].out_scope_passed
            overhead_ok[gid] = run.overhead_results[gid]["passed"]
            regression = run.reg_results[gid]
            reg_counts[gid] = None if regression is None else (
                regression.successes_with, regression.successes_without, regression.task_count)
            pairs[gid] = inp.ablation_evaluation.pairs
        stage2 = {gid: {tau: tau_c_sensitivity(pairs[gid], Fraction(repr(tau)))
                        for tau in SENSITIVITY_TAU_C_GRID} for gid in run.gene_ids}
        grid = {"gene_id": gene_id, "tau_c_grid": list(SENSITIVITY_TAU_C_GRID),
                "tau_r_grid": list(SENSITIVITY_TAU_R_GRID),
                "analysis_only": True,
                "runner_calls": 0,
                "child_records": None, "aggregate_verdicts": None,
                "somatic_states": None}
        grid["replay"] = {"fixed_at": REPLAY_THRESHOLD_TEXT,
                          "hypothetical_pass": replay_pass[gene_id]}
        grid["ablation"] = {f"tau_c={tau}": stage2[gene_id][tau]
                            for tau in SENSITIVITY_TAU_C_GRID}
        grid["regression"] = {
            f"tau_r={tau}": (None if reg_counts[gene_id] is None
                             else paired_drop_within(*reg_counts[gene_id], tau))
            for tau in SENSITIVITY_TAU_R_GRID}
        if reg_counts[gene_id] is None:
            grid["regression_note"] = NOT_RUN_REASON
        grid["interaction_note"] = (
            "interaction pair evidence exists only for pairs collected in "
            "the primary preliminary-eligible set; a cell is identifiable "
            "iff every unordered pair within its hypothetical eligible set "
            "was collected, otherwise it is marked not_identifiable with "
            "the exact missing pairs")
        rank_by_gene = {inp.candidate_envelope["candidate"]["gene_id"]:
                        inp.frozen_rank for inp in run.inputs}
        cells: dict[str, dict] = {}
        for tau_c in SENSITIVITY_TAU_C_GRID:
            for tau_r in SENSITIVITY_TAU_R_GRID:
                cell: dict = {"tau_c": tau_c, "tau_r": tau_r}
                # only a candidate whose other stages pass under this cell can be
                # eligible, so only its stage-2 outcome is needed here
                unidentified = sorted(
                    gid for gid in run.gene_ids
                    if not stage2[gid][tau_c]["identifiable"]
                    and replay_pass[gid] and gen_out[gid] and overhead_ok[gid]
                    and reg_counts[gid] is not None
                    and paired_drop_within(*reg_counts[gid], tau_r))
                if unidentified:
                    cell.update({
                        "identifiable": False, "not_identifiable": True,
                        "unidentified_stage2": unidentified,
                        "reason": ("under this tau_c the stage-2 rule of "
                                   f"{unidentified} would have continued past the observed "
                                   "stopping look (ADR-0012 D7)")})
                    cells[f"tau_c={tau_c}|tau_r={tau_r}"] = cell
                    continue
                hyp_eligible = sorted(
                    gid for gid in run.gene_ids
                    if (replay_pass[gid] and stage2[gid][tau_c]["pass"]
                        and gen_out[gid] and overhead_ok[gid]
                        and reg_counts[gid] is not None
                        and paired_drop_within(*reg_counts[gid], tau_r)))
                cell["hypothetical_eligible"] = hyp_eligible
                required_pairs = sorted(
                    tuple(sorted(pair))
                    for pair in combinations(hyp_eligible, 2))
                missing_pairs = [list(pair) for pair in required_pairs
                                 if pair not in run.pair_evidence]
                if missing_pairs:
                    cell["identifiable"] = False
                    cell["not_identifiable"] = True
                    cell["missing_interaction_pairs"] = missing_pairs
                    cell["reason"] = (
                        "identifying the interaction composition requires "
                        f"pair evidence for {missing_pairs}, which was "
                        "never collected in the primary run; the "
                        "composition cannot be identified from existing "
                        "T_gate evidence")
                else:
                    contribution = {
                        gid: Fraction(stage2[gid][tau_c]["contribution"]["exact"])
                        for gid in hyp_eligible}
                    elim = self._resolve_harmful_pairs(
                        set(hyp_eligible), contribution, rank_by_gene,
                        run.pair_evidence)
                    cell["identifiable"] = True
                    cell["not_identifiable"] = False
                    cell["eliminated_by_interaction"] = sorted(elim)
                    cell["hypothetical_promote"] = sorted(
                        g for g in hyp_eligible if g not in elim)
                cells[f"tau_c={tau_c}|tau_r={tau_r}"] = cell
        grid["primary_eligible"] = sorted(run.eligible)
        grid["cells"] = cells
        grid["runner_calls"] = 0
        return grid

    # -- internals ------------------------------------------------------------------
    def _binding(self, evaluated: dict) -> dict:
        """The child fields of ADR-0010 B1 for an evaluated candidate."""
        return {"candidate_version": evaluated["version"],
                "candidate_artifact": evaluated["artifact"],
                "base_genome_id": self._base_config.genome_id}

    def _validated_of(self, envelope: dict) -> dict:
        from inheritance.overlay import _validate_candidate
        return _validate_candidate(envelope, self._registry)

    def _parent_seed_for(self, inp: GateCandidateInput) -> tuple[str, int]:
        """Per-candidate CIG run identity: the input's own parent/seed when
        supplied, otherwise the batch defaults."""
        parent = inp.parent_cig_id or self._parent_cig_id
        seed = inp.mining_seed if inp.mining_seed is not None else self._mining_seed
        validate_gate_setup(parent, seed)
        return parent, seed

    def _configs_by_gene(self, inp: GateCandidateInput):
        return build_evaluation_configs(
            self._base_config, inp.candidate_envelope, self._registry)

    @staticmethod
    def _resolve_harmful_pairs(eligible_ids: set[str], contribution: dict,
                               rank_by_gene: dict, pair_evidence: dict
                               ) -> dict[str, str]:
        """Resolve every harmful pair among ``eligible_ids`` simultaneously
        from complete evidence: the member with the lower stage-2
        contribution is eliminated; exact contribution tie -> earlier frozen
        rank retained; ambiguous rank -> fail closed."""
        eliminated: dict[str, str] = {}
        for pair_key in sorted(pair_evidence):
            gid_a, gid_b = pair_key
            if gid_a not in eligible_ids or gid_b not in eligible_ids:
                continue
            if not pair_evidence[pair_key].harmful:
                continue
            contrib_a, contrib_b = contribution[gid_a], contribution[gid_b]
            if contrib_a > contrib_b:
                eliminated[gid_b] = gid_a
            elif contrib_b > contrib_a:
                eliminated[gid_a] = gid_b
            else:
                ra, rb = rank_by_gene[gid_a], rank_by_gene[gid_b]
                if ra == rb:
                    raise GateInputError(
                        f"exact contribution tie between {gid_a!r}/{gid_b!r} "
                        f"with ambiguous frozen ranks; failing closed")
                loser = gid_b if ra < rb else gid_a
                eliminated[loser] = gid_a if loser == gid_b else gid_b
        return eliminated

    def _consume_stage12_evidence(self, inp: GateCandidateInput) -> None:
        """Fail closed unless the stage-1 and stage-2 evidence coheres with
        this candidate's own CIG run identity — BEFORE any R5 episode:
        structural ``load_cig`` validity, candidate match, exact
        ``<parent>/S<seed>-<stage>`` child ids, parent match, seed match,
        stage names, pass agreement, EXACT locked eval settings
        (temperature 0.0, gate max_tokens, resolved mining seed, the ReAct
        protocol stop of ADR-0003 A2), the evaluated version, artifact and
        base genome (ADR-0010 B3), unique child ids, stage 1's expression
        rule (ADR-0012 D3) and stage 2's sequential decision recomputed from
        its record (D4)."""
        evaluated = self._validated_of(inp.candidate_envelope)["candidate"]
        gene_id = evaluated["gene_id"]
        binding = self._binding(evaluated)
        parent, seed = self._parent_seed_for(inp)
        issues: list[str] = []
        seen_ids: set[str] = set()
        for stage, evaluation in (("replay", inp.replay_evaluation),
                                  ("ablation", inp.ablation_evaluation)):
            child = evaluation.child_record
            try:
                load_cig(copy.deepcopy(child))
            except Exception as exc:
                issues.append(f"{stage} child record is structurally invalid: "
                              f"{exc}")
                continue
            m = child["measurements"]
            expected_id = f"{parent}/S{seed}-{stage}"
            if child["cig_id"] != expected_id:
                issues.append(f"{stage} child id {child['cig_id']!r} != exact "
                              f"{expected_id!r}")
            if m.get("candidate_gene_id") != gene_id:
                issues.append(f"{stage} child candidate_gene_id "
                              f"{m.get('candidate_gene_id')!r} != {gene_id!r}")
            # ADR-0010 B3: the artifact and base genome the child measured
            for key, expected in binding.items():
                got = m.get(key)
                if type(got) is not type(expected) or got != expected:
                    issues.append(f"{stage} child {key} {got!r} != the evaluated "
                                  f"{expected!r} (ADR-0010)")
            if child["parent_cig_id"] != parent:
                issues.append(f"{stage} child parent_cig_id "
                              f"{child['parent_cig_id']!r} != {parent!r}")
            child_seed = child["seed"]
            if isinstance(child_seed, bool) or not isinstance(child_seed, int) \
                    or child_seed != seed:
                issues.append(f"{stage} child seed {child_seed!r} != "
                              f"mining seed {seed!r} (strict int, non-bool)")
            if child["stage"] != stage:
                issues.append(f"{stage} child has stage {child['stage']!r}")
            if child["cig_id"] in seen_ids:
                issues.append(f"duplicate child id {child['cig_id']!r}")
            seen_ids.add(child["cig_id"])
            stage_pass = m.get("stage_pass")
            if not isinstance(stage_pass, bool) \
                    or not isinstance(evaluation.passed, bool) \
                    or evaluation.passed is not stage_pass:
                issues.append(f"{stage} result.passed {evaluation.passed!r} != "
                              f"child measurements.stage_pass "
                              f"{stage_pass!r} (strict bool on both sides)")
            if stage == "replay":
                issues.extend(replay_evidence_issues(m, gene_id))
            else:
                issues.extend(ablation_evidence_issues(
                    m, evaluation, mining_seed=seed,
                    families=resolve_frozen_applicability(inp.candidate_envelope,
                                                          self._registry)))
            issues.extend(self._settings_issues(stage, m.get("eval_generation_settings"), seed))
        if issues:
            raise EvidenceCoherenceError(sorted(issues))

    def _settings_issues(self, stage: str, settings, seed: int) -> list[str]:
        """EXACT deterministic-settings provenance: exactly the four locked
        keys, with strict types (loose Python equality is not trusted: 0.0
        == False and 64 == 64.0 must NOT validate)."""
        if not isinstance(settings, dict):
            return [f"{stage} child lacks eval_generation_settings provenance"]
        issues: list[str] = []
        expected_keys = {"temperature", "max_tokens", "seed", "stop"}
        actual_keys = set(settings)
        if actual_keys != expected_keys:
            issues.append(
                f"{stage} eval_generation_settings keys "
                f"{sorted(actual_keys)} != exactly "
                f"{sorted(expected_keys)} "
                f"(missing: {sorted(expected_keys - actual_keys)}, "
                f"unexpected: {sorted(actual_keys - expected_keys)})")
        temperature = settings.get("temperature")
        if isinstance(temperature, bool) \
                or not isinstance(temperature, (int, float)) \
                or not math.isfinite(temperature) \
                or temperature != 0.0:
            issues.append(
                f"{stage} eval_generation_settings.temperature "
                f"{temperature!r} != locked 0.0 (finite number, "
                f"non-bool)")
        max_tokens = settings.get("max_tokens")
        if isinstance(max_tokens, bool) \
                or not isinstance(max_tokens, int) \
                or max_tokens != self._budgets.max_tokens_per_request:
            issues.append(
                f"{stage} eval_generation_settings.max_tokens "
                f"{max_tokens!r} != locked "
                f"{self._budgets.max_tokens_per_request!r} (strict int, "
                f"non-bool)")
        eval_seed = settings.get("seed")
        if isinstance(eval_seed, bool) or not isinstance(eval_seed, int) \
                or eval_seed != seed:
            issues.append(
                f"{stage} eval_generation_settings.seed {eval_seed!r} "
                f"!= locked mining seed {seed!r} (strict int, non-bool)")
        stop = settings.get("stop")
        if not isinstance(stop, list) or stop != list(STOP_SEQUENCES):
            issues.append(
                f"{stage} eval_generation_settings.stop {stop!r} != "
                f"locked {list(STOP_SEQUENCES)!r} (ADR-0003 A2)")
        return issues

    def _build_stage4_child(self, gene_id: str, parent: str, seed: int,
                            reg_eval: RegressionEvaluation | None, overhead: dict,
                            pair_evidence: dict, eliminated: dict[str, str], *,
                            eligible: bool, eligible_count: int, evaluated: dict) -> dict:
        overhead_pass = overhead["passed"]
        overhead_block = {
            "tokens": overhead["tokens"],
            "tau_k_tokens": overhead["threshold"],
            "overhead_pass": overhead_pass,
            "counter_identity": overhead["counter_identity"],
            "rendered_sha256": overhead["rendered_sha256"],
        }
        if reg_eval is None:  # ADR-0012 D7: 4A and 4C not run, 4B measured
            measurements = {
                "candidate_gene_id": gene_id,
                **self._binding(evaluated),  # ADR-0010
                "not_run": True,
                "not_run_parts": ["regression", "interaction"],
                "not_run_reason": NOT_RUN_REASON,
                "regression": {"not_run": True, "reason": NOT_RUN_REASON,
                               "rule": REGRESSION_THRESHOLD_TEXT, "tau_r": TAU_R},
                "expression_overhead": overhead_block,
                "interaction": {"not_run": True, "reason": NOT_RUN_REASON,
                                "pair_evidence": [], "pair_count": 0,
                                "eliminated_by_interaction": False},
                "component_pass_flags": {"regression": False, "overhead": overhead_pass,
                                         "not_eliminated": True},
                "stage_pass": False,
            }
        else:
            pairs_for_gene = []
            for key in sorted(pair_evidence):
                evidence = pair_evidence[key]
                if gene_id not in evidence.gene_ids:
                    continue
                other = evidence.gene_ids[0] if \
                    evidence.gene_ids[1] == gene_id else evidence.gene_ids[1]
                # the canonical compact evidence record is embedded identically
                # in BOTH candidates' children (durable, reconstructable)
                pairs_for_gene.append({
                    "evidence_sha256": evidence.evidence_sha256,
                    "canonical_evidence": evidence.canonical_evidence,
                    "other_gene_id": other,
                    "delta_pair": evidence.delta_pair,
                    "harmful": evidence.harmful,
                    "retained": gene_id not in eliminated,
                    # the neither side is the seed's R6 cells (ADR-0012 D2, D6)
                    "neither_cells": {"seed": seed, "round": 6},
                })
            eliminated_by_interaction = gene_id in eliminated
            regression_pass = reg_eval.passed
            if pairs_for_gene:
                reason = None
            elif not eligible:
                reason = INELIGIBLE_INTERACTION_REASON
            else:
                reason = VACUOUS_INTERACTION_REASON if eligible_count < 2 else None
            measurements = {
                "candidate_gene_id": gene_id,
                **self._binding(evaluated),  # ADR-0010
                "regression": {
                    "allocation": reg_eval.allocation,
                    "episodes": reg_eval.episodes,
                    "successes_with": reg_eval.successes_with,
                    "successes_without": reg_eval.successes_without,
                    "task_count": reg_eval.task_count,
                    "regression_drop": reg_eval.regression_drop,
                    "rule": REGRESSION_THRESHOLD_TEXT,
                    "paired_g0_run": reg_eval.baseline,
                    "tau_r": TAU_R,
                    "regression_pass": regression_pass,
                    # exact generation settings every T_reg episode ran with
                    "eval_generation_settings": reg_eval.locked_settings,
                },
                "expression_overhead": overhead_block,
                "interaction": {
                    "pair_evidence": pairs_for_gene,
                    "pair_count": len(pairs_for_gene),
                    "vacuous": not pairs_for_gene,
                    "reason": reason,
                    "eliminated_by_interaction": eliminated_by_interaction,
                },
                "component_pass_flags": {
                    "regression": regression_pass,
                    "overhead": overhead_pass,
                    "not_eliminated": not eliminated_by_interaction,
                },
                "stage_pass": regression_pass and overhead_pass and not eliminated_by_interaction,
            }
        child = {
            "cig_id": f"{parent}/S{seed}-interaction_regression",
            "parent_cig_id": parent,
            "seed": seed,
            "stage": "interaction_regression",
            "measurements": measurements,
        }
        # the completed child is structurally valid evidence even when no
        # stores are supplied
        load_cig(child)
        return child

    def _build_aggregate(self, inp: GateCandidateInput, gene_id: str,
                         parent: str, gen: GeneralizationEvaluation, stage4: dict, *,
                         evaluated: dict) -> dict:
        gen_in = gen.in_scope_child
        gen_out = gen.out_scope_child
        out_threshold = {"run": OUT_SCOPE_STAGE_TEXT, "vacuous": VACUOUS_OUT_SCOPE_STAGE_TEXT,
                         "not_run": NOT_RUN_THRESHOLD_TEXT}[gen.out_scope_state]
        stages = {
            "replay": {"measurements_ref": "children",
                       "pass": inp.replay_evaluation.passed,
                       "threshold": REPLAY_STAGE_TEXT},
            "ablation": {"measurements_ref": "children",
                         "pass": inp.ablation_evaluation.passed,
                         "threshold": ABLATION_STAGE_TEXT},
            "generalization_in_scope": {
                "measurements_ref": "children",
                "pass": gen.in_scope_passed,
                "threshold": IN_SCOPE_STAGE_TEXT},
            "generalization_out_of_scope": {
                "measurements_ref": "children",
                "pass": gen.out_scope_passed,
                "threshold": out_threshold},
            "interaction_regression": {
                "measurements_ref": "children",
                "pass": stage4["measurements"]["stage_pass"],
                "threshold": (STAGE4_NOT_RUN_TEXT if stage4["measurements"].get("not_run")
                              else STAGE4_TEXT)},
        }
        child_ids = [inp.replay_evaluation.child_record["cig_id"],
                     inp.ablation_evaluation.child_record["cig_id"],
                     gen_in["cig_id"], gen_out["cig_id"],
                     stage4["cig_id"]]
        all_pass = all(s["pass"] for s in stages.values())
        verdict = "promote" if all_pass else "reject"
        aggregate = {
            "cig_id": parent,
            "candidate_gene_id": gene_id,
            "genome_id": self._base_config.genome_id,
            "child_records": child_ids,
            "stages": stages,
            "verdict": verdict,
            "thresholds_used": {
                "version": THRESHOLD_VERSION,
                "values": copy.deepcopy(PRIMARY_THRESHOLDS),
            },
            # which artifact this verdict is about (ADR-0001 reason (c): which
            # CIG record promoted which artifact)
            "artifacts": {"candidate": {
                "gene_id": evaluated["gene_id"],
                "version": evaluated["version"],
                "artifact": evaluated["artifact"],
            }, "base_genome": {"genome_id": self._base_config.genome_id}},  # ADR-0010 B1
        }
        load_cig(aggregate)
        return aggregate

    # -- prevalidated batch commit (step 10) ---------------------------------------------
    def _commit_results(self, results: list[CandidateGateResult], *,
                         somatic_store, cig_record_store: CigRecordStore) -> None:
        """Prevalidated BATCH commit: every bundle is prevalidated against a
        staged clone of the CURRENT live CigRecordStore, the staged store is
        verified as a whole (ADR-0010 B4), and every somatic candidate (at
        the version derived from the evaluated envelope) is checked BEFORE
        any live mutation, so a prevalidation failure leaves
        both live stores untouched. Only then are the live stores mutated,
        per candidate: somatic terminal decision first, then the CIG bundle
        is exposed on the live store.

        The mutation phase itself is not transactional: if a somatic
        ``decide()`` or a bundle exposure fails part-way (e.g. a journal
        write error), the candidates before it stay committed. Terminal
        decisions are never rewritten, so such a batch must be resolved by
        hand (issue #12 handoff section 13 scopes atomicity to the
        in-process prevalidation)."""
        staged_cig = cig_record_store.snapshot()
        for result in results:
            staged_cig.add_bundle(result.aggregate,
                                   self._bundle_children(result))
        # the store-wide check, with every bundle of the batch staged: the two
        # copies of each pair evidence span two bundles (ADR-0010 B4)
        staged_cig.require_consistent()
        planned: list[tuple[CandidateGateResult, int]] = []
        for result in results:
            version = result.candidate_version
            candidate = result.candidate
            if (candidate.get("gene_id"), candidate.get("version")) != (result.gene_id, version):
                raise RecordConsistencyError(
                    f"result for {result.gene_id!r} v{version} carries candidate "
                    f"{candidate.get('gene_id')!r} v{candidate.get('version')!r}; "
                    f"the checked and the decided reference must be the same")
            issue = _somatic_binding_issue(somatic_store, candidate)
            if issue:
                raise RecordConsistencyError(issue)
            planned.append((result, version))
        # commit: somatic decision first, then expose the bundle.
        # verdict -> terminal state: promote -> validated,
        # reject -> rejected (neither is germline assimilation)
        for result, version in planned:
            somatic_store.decide(result.gene_id, version,
                                  "validated" if result.verdict == "promote"
                                  else "rejected",
                                  [result.aggregate_cig_id])
            cig_record_store.add_bundle(result.aggregate,
                                         self._bundle_children(result))

    def _bundle_children(self, result: CandidateGateResult) -> list[dict]:
        """Children ordered per the aggregate declaration, sourced from this
        run's #11 evidence + locally produced stage-3/4 children."""
        run = self._last
        children: list[dict] = [result.stage4_child]
        for inp, gid in zip(run.inputs, run.gene_ids):
            if gid != result.gene_id:
                continue
            children.append(inp.replay_evaluation.child_record)
            children.append(inp.ablation_evaluation.child_record)
            gen = run.gen_results[gid]
            children.append(gen.in_scope_child)
            children.append(gen.out_scope_child)
        by_id = {c["cig_id"]: c for c in children}
        return [by_id[cid] for cid in result.aggregate["child_records"]]


def replay_evidence_issues(measurements: dict, gene_id: str) -> list[str]:
    """ADR-0012 D3 recomputed from the stage-1 record: exactly one
    with-trait episode, on the source task, in round R1, and ``stage_pass``
    iff the target gene was expressed in it or it failed."""
    episodes = measurements.get("episodes")
    allocation = measurements.get("allocation") or {}
    if not isinstance(episodes, list) or len(episodes) != 1 \
            or not isinstance(episodes[0], dict):
        return [f"replay child holds {episodes!r} episodes, not exactly one (ADR-0012 D3)"]
    episode = episodes[0]
    issues = []
    source = allocation.get("source_task_id")
    if episode.get("condition") != "with_trait" or type(episode.get("round")) is not int \
            or episode.get("round") != 1 or not isinstance(source, str) \
            or episode.get("task_id") != source:
        issues.append("replay child's episode is not the source task with the trait in round "
                      "R1 (ADR-0012 D3)")
    success, expressed = episode.get("success"), episode.get("expressed_gene_ids")
    if type(success) is not bool or not isinstance(expressed, list):
        return issues + ["replay child's episode lacks a boolean success and expressed genes"]
    expected = gene_id in expressed or not success
    if measurements.get("stage_pass") is not expected:
        issues.append(f"replay stage_pass {measurements.get('stage_pass')!r} != the expression "
                      f"rule's {expected} (ADR-0012 D3)")
    return issues


def _somatic_binding_issue(somatic_store, evaluated: dict) -> str | None:
    """None iff the store holds exactly the evaluated candidate reference,
    still in state ``candidate`` (issue #12 handoff section 13)."""
    gene_id, version = evaluated["gene_id"], evaluated["version"]
    if somatic_store.state(gene_id, version) != "candidate":
        return f"{gene_id!r} v{version} is no longer a somatic candidate"
    stored = somatic_store.candidate_ref(gene_id, version)
    if stored != evaluated:
        differing = sorted(k for k in set(stored) | set(evaluated)
                           if stored.get(k) != evaluated.get(k))
        return (f"{gene_id!r} v{version}: the stored somatic candidate is not "
                f"the evaluated one (differs in {', '.join(differing)}); "
                f"refusing to bind the verdict to a different reference")
    return None


@dataclass
class _LastRun:
    inputs: list
    gene_ids: list
    gen_results: dict
    reg_results: dict
    overhead_results: dict
    pair_evidence: dict
    eligible: list
    eliminated: dict
    outcome: GateOutcome
