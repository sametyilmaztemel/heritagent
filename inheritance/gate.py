"""CIG gate orchestrator (issue #12): stage 3/4 + aggregate verdict +
somatic terminal decision.

Orchestration order (handoff §14) for a frozen candidate set:
1. validate inputs — all before any episode: unique frozen ranks, unique
   gene ids, UNIQUE resolved aggregate CIG ids (per-candidate parent/seed
   identity), a supplied TokenCounter, every candidate's evaluation
   config buildable, and, when both stores are supplied (i.e. the run
   will commit), no aggregate id already live and every candidate still
   the stored somatic ``candidate`` with exactly the evaluated reference;
2. consume #11 replay+ablation children (never rerun): structural
   ``load_cig`` validation + exact ``<parent>/S<seed>-<stage>`` child ids
   + full coherence, failing closed before any stage-3/4 runner call;
3. stage-3 generalization for ALL candidates;
4. one shared G0 run on T_reg per mining seed at the gate's locked settings,
   then
   regression (paired against it, ADR-0009 D1) + overhead for ALL
   candidates (the token counter contract is validated fail-closed before
   first use);
5. preliminary eligible set;
6. every unordered interaction pair in that set evaluated exactly once,
   gene ids in canonical order, candidate target regulations preserved;
7. harmful pairs resolved simultaneously from COMPLETE evidence (higher
   ablation contribution retained; exact tie -> earlier frozen rank;
   ambiguous rank -> fail closed);
8. stage-4 child for every candidate (exact regression generation
   settings persisted, canonical pair evidence embedded durably,
   ``load_cig``-validated before any result is returned);
9. final aggregate + verdict per candidate;
10. prevalidated batch commit when stores are supplied: every bundle and
    every somatic decision is prevalidated against a staged clone of the
    CURRENT live CIG store; only after all prevalidations pass are the
    live stores mutated (somatic decision first — with the candidate
    version derived from the evaluated envelope — then the CIG bundle
    exposed via the live store). The mutation phase is not transactional
    (see ``_commit_results``).

A scientifically valid stage FAIL still produces immutable child evidence
and later primary stages continue (ADR-0002 sensitivity recomputation
needs all stage evidence). Only integrity/setup/runtime failures abort.

Each candidate carries its OWN CIG run identity (parent CIG id + mining
seed); two candidates may never share one aggregate id. Boundaries: no
germline assimilation, no T_eval access, no EXP-0001 execution, no
threshold retuning; sensitivity is analysis-only pure recomputation with
explicit per-cell identifiability.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from itertools import combinations

from genome.validation.errors import RecordConsistencyError
from genome.validation.loader import load_cig
from genome.validation.records import CigRecordStore
from runtime.expression import RuntimeConfig
from runtime.loop import Budgets
from runtime.policies.planner_react import STOP_SEQUENCES

from inheritance.ablation import (
    BOOTSTRAP_CLUSTER_ORDER,
    BOOTSTRAP_N_RESAMPLES,
    BOOTSTRAP_SEED,
    TAU_C,
    AblationEvaluation,
)
from inheritance.bootstrap import CONFIDENCE, paired_cluster_bootstrap, stage_passes
from inheritance.generalization import (
    GeneralizationEvaluation,
    GeneralizationEvaluator,
)
from inheritance.interaction import (
    InteractionEvaluator,
)
from inheritance.overlay import (
    build_evaluation_configs,
    evaluation_settings,
)
from inheritance.regression import (
    REGRESSION_THRESHOLD_TEXT,
    RegressionEvaluation,
    RegressionEvaluator,
    TokenCounter,
    measure_expression_overhead,
    paired_drop_within,
    run_regression_baseline,
    validate_token_counter,
)
from inheritance.replay import ReplayEvaluation
from inheritance.runner import validate_gate_setup

THRESHOLD_VERSION = "EXP-0001-CIG-M0-v1"
PRIMARY_THRESHOLDS = {
    "replay_min_success": "3/5",
    "replay_min_delta": "+2/5",
    "tau_c": 0.05,
    "in_scope_min_delta": 0.0,
    "out_scope_min_delta": -0.05,
    "tau_r": 0.03,
    "harmful_pair_delta": -0.05,
    "tau_k_tokens": 512,
    "bootstrap_confidence": 0.95,
    "bootstrap_resamples": 10_000,
    "bootstrap_seed": 0,                  # ADR-0005, pre-registered
    "bootstrap_cluster_order": "task_id",  # ADR-0005
}
SENSITIVITY_TAU_C_GRID = (0.03, 0.05, 0.08)
SENSITIVITY_TAU_R_GRID = (0.02, 0.03, 0.05)
VACUOUS_INTERACTION_REASON = (
    "vacuous: fewer than two preliminary-eligible candidates, so no "
    "both-vs-neither interaction contrast is defined for this candidate")


class GateInputError(ValueError):
    """Invalid orchestrator input (fail closed before any episode)."""


class EvidenceCoherenceError(ValueError):
    """#11 evidence does not cohere with this candidate/aggregate (fail
    closed; replay/ablation are never rerun)."""


BOOTSTRAP_METHOD = "percentile-paired-cluster-bootstrap"


def ablation_evidence_issues(measurements: dict, evaluated: dict | None = None) -> list[str]:
    """ADR-0005, hardened by ADR-0009 D5: an ablation verdict is evidence
    only under the pre-registered bootstrap (the percentile paired cluster
    bootstrap at confidence 0.95; seed 0 for ``random.Random``, one
    ``randrange(n)`` per resampled cluster; 10,000 resamples; the deltas in
    task_id order). The declared values are checked in the child record and
    in the evaluation, and the bound is recomputed from the record's deltas:
    its lower and upper bound, its point estimate and the stage verdict must
    be exactly the record's. ``evaluated`` is the evaluation's own block (the
    one the sensitivity recomputation reads); a resumed run passes None."""
    issues: list[str] = []
    locked = {"seed": BOOTSTRAP_SEED, "n_resamples": BOOTSTRAP_N_RESAMPLES,
              "cluster_order": BOOTSTRAP_CLUSTER_ORDER, "method": BOOTSTRAP_METHOD}
    blocks = [("child bootstrap", measurements.get("bootstrap"))]
    if evaluated is not None:
        blocks.append(("evaluation bootstrap", evaluated))
    for where, block in blocks:
        if not isinstance(block, dict):
            issues.append(f"ablation {where} is missing")
            continue
        for key, value in locked.items():
            got = block.get(key)
            # strict type: True == 1 and 0.0 == 0 must not validate
            if type(got) is not type(value) or got != value:
                issues.append(f"ablation {where}.{key} {got!r} != pre-registered "
                              f"{value!r} (ADR-0005)")
        confidence = block.get("confidence")
        if type(confidence) is not float or confidence != CONFIDENCE:
            issues.append(f"ablation {where}.confidence {confidence!r} != pre-registered "
                          f"{CONFIDENCE!r} (ADR-0009 D5)")
    ids = measurements.get("paired_delta_task_ids")
    if not isinstance(ids, list) or not all(isinstance(i, str) for i in ids) \
            or any(a >= b for a, b in zip(ids, ids[1:])):
        issues.append(f"ablation child paired_delta_task_ids {ids!r} are not "
                      f"distinct strings in task_id order (ADR-0005)")
        return issues
    allocated = (measurements.get("allocation") or {}).get("task_ids")
    if not isinstance(allocated, list) or sorted(allocated) != ids:
        issues.append("ablation child paired_delta_task_ids are not the allocation's tasks "
                      "(ADR-0009 D5)")
    deltas = measurements.get("paired_deltas")
    if not isinstance(deltas, list) or len(deltas) != len(ids):
        issues.append(f"ablation child has {len(deltas) if isinstance(deltas, list) else deltas!r}"
                      f" paired deltas for {len(ids)} tasks (ADR-0009 D5)")
        return issues
    try:
        bootstrap = paired_cluster_bootstrap(deltas, seed=BOOTSTRAP_SEED,
                                             n_resamples=BOOTSTRAP_N_RESAMPLES,
                                             confidence=CONFIDENCE)
    except ValueError as exc:
        return issues + [f"ablation child paired deltas do not recompute: {exc}"]
    block = measurements.get("bootstrap") if isinstance(measurements.get("bootstrap"), dict) \
        else {}
    recomputed = {"lower_bound": bootstrap.lower_bound, "upper_bound": bootstrap.upper_bound}
    for where, source in [("child bootstrap", block)] + (
            [("evaluation bootstrap", evaluated)] if isinstance(evaluated, dict) else []):
        for key, value in recomputed.items():
            if source.get(key) != value:
                issues.append(f"ablation {where}.{key} {source.get(key)!r} != recomputed "
                              f"{value!r} (ADR-0009 D5)")
    if measurements.get("point_estimate_delta_success") != bootstrap.point_estimate:
        issues.append(f"ablation point estimate "
                      f"{measurements.get('point_estimate_delta_success')!r} != recomputed "
                      f"{bootstrap.point_estimate!r} (ADR-0009 D5)")
    expected = stage_passes(bootstrap.point_estimate, bootstrap.lower_bound, tau_c=TAU_C)
    if measurements.get("stage_pass") is not expected:
        issues.append(f"ablation stage_pass {measurements.get('stage_pass')!r} != the locked "
                      f"rule on the recomputed bound ({expected}) (ADR-0009 D5)")
    return issues


@dataclass(frozen=True)
class GateCandidateInput:
    """One frozen candidate: its rank, the already-completed #11 evidence,
    its own CIG run identity (parent id + mining seed) and the frozen
    stage-3/4 allocations. Ranks must be unique non-negative ints."""

    candidate_envelope: dict
    frozen_rank: int
    replay_evaluation: ReplayEvaluation
    ablation_evaluation: AblationEvaluation
    generalization_allocation: object
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
    """Evaluates stage 3/4 for every candidate, resolves interactions,
    builds aggregates/verdicts, and (when stores are supplied) commits after
    batch prevalidation: somatic decisions first, CIG bundles exposed on the
    live store only after."""

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

        # step 2: consume #11 evidence, fail closed on incoherence —
        # BEFORE any stage-3/4 episode
        for inp in inputs:
            self._consume_stage12_evidence(inp)
        # last setup check: a closed or stale journal handle would refuse the
        # step-10 appends (an empty batch appends nothing)
        if inputs and self._commits():
            self._somatic_store.check_appendable()

        # step 3: stage-3 generalization for ALL candidates
        gen_results: dict[str, GeneralizationEvaluation] = {}
        for inp, (parent, seed) in zip(inputs, identities):
            gid = inp.candidate_envelope["candidate"]["gene_id"]
            evaluator = GeneralizationEvaluator(
                candidate_envelope=inp.candidate_envelope, registry=self._registry,
                base_config=self._base_config, runner=self._runner,
                budgets=self._budgets, parent_cig_id=parent, mining_seed=seed,
                target_regulation=self._target_regulation_by_gene.get(gid))
            gen_results[gid] = evaluator.evaluate(inp.generalization_allocation)

        # step 4: one shared G0 run on T_reg (ADR-0009 D1), then regression
        # paired against it + overhead for ALL candidates
        reg_results: dict[str, RegressionEvaluation] = {}
        overhead_results: dict[str, dict] = {}
        # one G0 run per distinct (mining seed, T_reg), before any candidate,
        # at the seed each candidate is measured at (EXP-0001 has one seed
        # and one T_reg per gate run, ADR-0006 D8, so one run of 40 episodes)
        baselines: dict[tuple, object] = {}

        def baseline_key(inp, seed):
            return seed, tuple(t.task_id for t in inp.regression_allocation.tasks)

        for inp, (_, seed) in zip(inputs, identities):
            key = baseline_key(inp, seed)
            if key not in baselines:
                baselines[key] = run_regression_baseline(
                    base_config=self._base_config, runner=self._runner,
                    budgets=self._budgets, mining_seed=seed,
                    allocation=inp.regression_allocation)
        for inp, (parent, seed) in zip(inputs, identities):
            gid = inp.candidate_envelope["candidate"]["gene_id"]
            reg_evaluator = RegressionEvaluator(
                candidate_envelope=inp.candidate_envelope, registry=self._registry,
                base_config=self._base_config, runner=self._runner,
                budgets=self._budgets, parent_cig_id=parent,
                mining_seed=seed,
                target_regulation=self._target_regulation_by_gene.get(gid))
            reg_results[gid] = reg_evaluator.evaluate(
                inp.regression_allocation, baselines[baseline_key(inp, seed)])
            target = [s for s in self._configs_by_gene(inp).with_trait.skills
                       if s.gene_id == gid][0]
            oh = measure_expression_overhead(target, self._token_counter)
            overhead_results[gid] = {
                "tokens": oh.tokens, "threshold": oh.threshold,
                "passed": oh.passed, "counter_identity": oh.counter_identity,
                "rendered_sha256": oh.rendered_sha256,
            }

        # step 5: preliminary eligible set — replay AND ablation AND both
        # generalization scopes AND regression AND overhead must all pass
        eligible: list[str] = []
        for inp, gid in zip(inputs, gene_ids):
            if (inp.replay_evaluation.passed
                    and inp.ablation_evaluation.passed
                    and gen_results[gid].passed and gen_results[gid].out_scope_passed
                    and reg_results[gid].passed
                    and overhead_results[gid]["passed"]):
                eligible.append(gid)

        # steps 6-7: interaction pairs, computed once, resolved together
        pair_evidence: dict[tuple[str, str], object] = {}
        if len(eligible) >= 2:
            if self._interaction_allocation is None:
                raise GateInputError(
                    ">=2 preliminary-eligible candidates require an interaction "
                    "allocation; none was supplied")
            interaction = InteractionEvaluator(
                registry=self._registry, base_config=self._base_config,
                runner=self._runner, budgets=self._budgets,
                target_regulation_by_gene=self._target_regulation_by_gene)
            locked = evaluation_settings(
                max_tokens=self._budgets.max_tokens_per_request,
                seed=self._mining_seed)
            ordered = sorted(
                (inp for inp, gid in zip(inputs, gene_ids) if gid in eligible),
                key=lambda i: i.frozen_rank)
            for a, b in combinations(ordered, 2):
                evidence = interaction.evaluate_pair(
                    a.candidate_envelope, b.candidate_envelope,
                    self._interaction_allocation, locked)
                key = tuple(sorted(evidence.gene_ids))
                pair_evidence[key] = evidence

        # resolve ALL harmful pairs simultaneously from complete evidence
        contribution = {gid: inp.ablation_evaluation.delta_success
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
                pair_evidence, eliminated)
            aggregate = self._build_aggregate(
                inp, gid, parent, gen_results.get(gid), stage4,
                evaluated=evaluated)
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
                    "regression": reg_results[gid].passed,
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
        """ADR-0002 §15: analysis-only threshold recomputation over the
        immutable primary stage measurements (tau_c/tau_r grids). ZERO
        runner/model/environment calls; no mutation of children/aggregate/
        verdicts/SomaticStore; no T_eval dependency.

        Per (tau_c, tau_r) cell the hypothetical eligible set is recomposed
        from the frozen stage evidence (replay/generalization/overhead
        fixed; ablation under tau_c, regression under tau_r). A cell is
        identifiable iff the interaction pair evidence REQUIRED by its
        hypothetical eligible set — every unordered pair within that set —
        was actually collected in the primary run; otherwise the cell is
        marked ``not_identifiable`` with the exact missing pairs. A single
        newly eligible candidate requires no pairs and stays
        identifiable; identifiable cells report the recomposed elimination
        + promote sets recomputed from the existing pair evidence."""
        run = self._last
        if run is None:
            raise GateInputError("no primary run has been executed yet")
        if gene_id not in run.gene_ids:
            raise GateInputError(f"unknown candidate gene id {gene_id!r}")
        replay_pass, ablation_delta, ablation_lb = {}, {}, {}
        gen_in, gen_out, overhead_ok, reg_counts = {}, {}, {}, {}
        for inp, gid in zip(run.inputs, run.gene_ids):
            replay_pass[gid] = inp.replay_evaluation.passed
            ablation_delta[gid] = inp.ablation_evaluation.delta_success
            ablation_lb[gid] = inp.ablation_evaluation.bootstrap["lower_bound"]
            gen_in[gid] = run.gen_results[gid].in_scope_passed
            gen_out[gid] = run.gen_results[gid].out_scope_passed
            overhead_ok[gid] = run.overhead_results[gid]["passed"]
            reg_counts[gid] = (run.reg_results[gid].successes_with,
                               run.reg_results[gid].successes_without,
                               run.reg_results[gid].task_count)
        grid = {"gene_id": gene_id, "tau_c_grid": list(SENSITIVITY_TAU_C_GRID),
                 "tau_r_grid": list(SENSITIVITY_TAU_R_GRID),
                 "analysis_only": True,
                 "runner_calls": 0,
                 "child_records": None, "aggregate_verdicts": None,
                 "somatic_states": None}
        # ADR-0002: tau_c grid applies ONLY to ablation; tau_r grid ONLY to
        # regression. Replay/generalization/overhead remain fixed primary
        # evidence.
        grid["replay"] = {"fixed_at": "successes_with >= 3/5 and delta >= +2/5",
                           "hypothetical_pass": replay_pass[gene_id]}
        grid["ablation"] = {f"tau_c={tau}":
                             (ablation_delta[gene_id] >= tau
                              and ablation_lb[gene_id] > 0)
                            for tau in SENSITIVITY_TAU_C_GRID}
        grid["regression"] = {f"tau_r={tau}":
                               paired_drop_within(*reg_counts[gene_id], tau)
                               for tau in SENSITIVITY_TAU_R_GRID}
        grid["interaction_note"] = (
            "interaction pair evidence exists only for pairs collected in "
            "the primary preliminary-eligible set; a cell is identifiable "
            "iff every unordered pair within its hypothetical eligible set "
            "was collected, otherwise it is marked not_identifiable with "
            "the exact missing pairs")
        contribution = {gid: ablation_delta[gid] for gid in run.gene_ids}
        rank_by_gene = {inp.candidate_envelope["candidate"]["gene_id"]:
                         inp.frozen_rank for inp in run.inputs}
        cells: dict[str, dict] = {}
        for tau_c in SENSITIVITY_TAU_C_GRID:
            for tau_r in SENSITIVITY_TAU_R_GRID:
                hyp_eligible = sorted(
                    gid for gid in run.gene_ids
                    if (replay_pass[gid] and gen_in[gid] and gen_out[gid]
                        and overhead_ok[gid]
                        and ablation_delta[gid] >= tau_c
                        and ablation_lb[gid] > 0
                        and paired_drop_within(*reg_counts[gid], tau_r)))
                # identifiability = REQUIRED pair evidence exists: every
                # unordered pair WITHIN the hypothetical eligible set must
                # have been collected in the primary run. A single (or
                # empty) eligible set requires no pairs and stays
                # identifiable even when its members were newly eligible.
                required_pairs = sorted(
                    tuple(sorted(pair))
                    for pair in combinations(hyp_eligible, 2))
                missing_pairs = [list(pair) for pair in required_pairs
                                  if pair not in run.pair_evidence]
                cell: dict = {"tau_c": tau_c, "tau_r": tau_r,
                               "hypothetical_eligible": hyp_eligible}
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
        from complete evidence: the lower-ablation-contribution member is
        eliminated; exact contribution tie -> earlier frozen rank retained;
        ambiguous rank -> fail closed."""
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
        """Fail closed unless the #11 evidence coheres with this candidate's
        own CIG run identity — BEFORE any stage-3/4 episode: structural
        ``load_cig`` validity, candidate match, exact
        ``<parent>/S<seed>-<stage>`` child ids, parent match, seed match,
        stage names, pass agreement, EXACT locked eval settings
        (temperature 0.0, gate max_tokens, resolved mining seed, the ReAct
        protocol stop of ADR-0003 A2), the pre-registered ablation
        bootstrap (ADR-0005), and unique child ids."""
        gene_id = self._validated_of(inp.candidate_envelope)["candidate"]["gene_id"]
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
            if stage == "ablation":
                issues.extend(ablation_evidence_issues(m, evaluation.bootstrap))
            settings = m.get("eval_generation_settings")
            if not isinstance(settings, dict):
                issues.append(f"{stage} child lacks eval_generation_settings "
                               f"provenance")
                continue
            # EXACT deterministic-settings provenance: exactly the four
            # locked keys, with strict types (loose Python equality is not
            # trusted: 0.0 == False and 64 == 64.0 must NOT validate)
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
        if issues:
            raise EvidenceCoherenceError(sorted(issues))

    def _build_stage4_child(self, gene_id: str, parent: str, seed: int,
                             reg_eval: RegressionEvaluation, overhead: dict,
                             pair_evidence: dict,
                             eliminated: dict[str, str]) -> dict:
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
            })
        vacuous = not pairs_for_gene
        eliminated_by_interaction = gene_id in eliminated
        regression_pass = reg_eval.passed
        overhead_pass = overhead["passed"]
        stage_pass = regression_pass and overhead_pass and \
            not eliminated_by_interaction
        child = {
            "cig_id": f"{parent}/S{seed}-interaction_regression",
            "parent_cig_id": parent,
            "seed": seed,
            "stage": "interaction_regression",
            "measurements": {
                "candidate_gene_id": gene_id,
                "regression": {
                    "allocation": reg_eval.allocation,
                    "episodes": reg_eval.episodes,
                    "successes_with": reg_eval.successes_with,
                    "successes_without": reg_eval.successes_without,
                    "task_count": reg_eval.task_count,
                    "regression_drop": reg_eval.regression_drop,
                    "rule": REGRESSION_THRESHOLD_TEXT,
                    "paired_g0_run": reg_eval.baseline,
                    "tau_r": 0.03,
                    "regression_pass": regression_pass,
                    # exact generation settings every T_reg episode ran with
                    "eval_generation_settings": reg_eval.locked_settings,
                },
                "expression_overhead": {
                    "tokens": overhead["tokens"],
                    "tau_k_tokens": overhead["threshold"],
                    "overhead_pass": overhead_pass,
                    "counter_identity": overhead["counter_identity"],
                    "rendered_sha256": overhead["rendered_sha256"],
                },
                "interaction": {
                    "pair_evidence": pairs_for_gene,
                    "pair_count": len(pairs_for_gene),
                    "vacuous": vacuous,
                    "reason": VACUOUS_INTERACTION_REASON if vacuous else None,
                    "eliminated_by_interaction": eliminated_by_interaction,
                },
                "component_pass_flags": {
                    "regression": regression_pass,
                    "overhead": overhead_pass,
                    "not_eliminated": not eliminated_by_interaction,
                },
                "stage_pass": stage_pass,
            },
        }
        # the completed child is structurally valid evidence even when no
        # stores are supplied
        load_cig(child)
        return child

    def _build_aggregate(self, inp: GateCandidateInput, gene_id: str,
                          parent: str, gen, stage4: dict, *,
                          evaluated: dict) -> dict:
        gen_in = gen.in_scope_child
        gen_out = gen.out_scope_child
        stages = {
            "replay": {"measurements_ref": "children",
                        "pass": inp.replay_evaluation.passed,
                        "threshold": "successes_with >= 3/5 and delta >= +2/5 "
                                      "and expression check"},
            "ablation": {"measurements_ref": "children",
                          "pass": inp.ablation_evaluation.passed,
                          "threshold": "bootstrap 95% LB > 0 and delta >= 0.05"},
            "generalization_in_scope": {
                "measurements_ref": "children",
                "pass": gen.in_scope_passed,
                "threshold": "delta_in >= 0.0"},
            "generalization_out_of_scope": {
                "measurements_ref": "children",
                "pass": gen.out_scope_passed,
                "threshold": ("vacuous: no out-of-scope family (ADR-0007 E6)"
                              if gen_out["measurements"].get("vacuous")
                              else "delta_out >= -0.05")},
            "interaction_regression": {
                "measurements_ref": "children",
                "pass": stage4["measurements"]["stage_pass"],
                "threshold": "paired regression drop <= 0.03 (ADR-0009 D1), overhead <= "
                              "512 tokens, not eliminated"},
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
                "values": dict(PRIMARY_THRESHOLDS),
            },
            # which artifact this verdict is about (ADR-0001 reason (c): which
            # CIG record promoted which artifact)
            "artifacts": {"candidate": {
                "gene_id": evaluated["gene_id"],
                "version": evaluated["version"],
                "artifact": evaluated["artifact"],
            }},
        }
        load_cig(aggregate)
        return aggregate

    # -- prevalidated batch commit (step 10) ---------------------------------------------
    def _commit_results(self, results: list[CandidateGateResult], *,
                         somatic_store, cig_record_store: CigRecordStore) -> None:
        """Prevalidated BATCH commit: every bundle is prevalidated against a
        staged clone of the CURRENT live CigRecordStore and every somatic
        candidate (at the version derived from the evaluated envelope) is
        checked BEFORE any live mutation, so a prevalidation failure leaves
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
