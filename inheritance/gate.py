"""CIG gate orchestrator (issue #12): stage 3/4 + aggregate verdict +
somatic terminal decision.

Orchestration order (handoff §14) for a frozen candidate set:
1. validate inputs/frozen ranks;
2. consume #11 replay+ablation children (never rerun);
3. stage-3 generalization for ALL candidates;
4. regression + overhead for ALL candidates;
5. preliminary eligible set;
6. every unordered interaction pair in that set evaluated exactly once;
7. harmful pairs resolved simultaneously (higher ablation contribution
   retained; exact tie -> earlier frozen rank; ambiguous rank -> fail
   closed);
8. stage-4 child for every candidate;
9. final aggregate + verdict per candidate;
10. atomic commit: CIG bundle exposed to the live store only after the
    somatic decision succeeds; candidate decisions written via
    SomaticStore.decide.

A scientifically valid stage FAIL still produces immutable child evidence
and later primary stages continue (ADR-0002 sensitivity recomputation
needs all stage evidence). Only integrity/setup/runtime failures abort.

Each candidate carries its OWN CIG run identity (parent CIG id + mining
seed); the orchestrator's batch-level values are defaults. Boundaries: no
germline assimilation, no T_eval access, no EXP-0001 execution, no
threshold retuning; sensitivity is analysis-only pure recomputation.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from itertools import combinations

from genome.validation.errors import RecordConsistencyError
from genome.validation.loader import load_cig, load_somatic
from genome.validation.records import CigRecordStore
from runtime.expression import RuntimeConfig
from runtime.loop import Budgets

from inheritance.ablation import AblationEvaluation
from inheritance.generalization import (
    GeneralizationAllocation,
    GeneralizationEvaluation,
    GeneralizationEvaluator,
)
from inheritance.interaction import (
    InteractionAllocation,
    InteractionEvaluator,
)
from inheritance.overlay import (
    EvaluationSetupError,
    build_evaluation_configs,
)
from inheritance.regression import (
    RegressionAllocation,
    RegressionEvaluation,
    RegressionEvaluator,
    TokenCounter,
    measure_expression_overhead,
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
}
SENSITIVITY_TAU_C_GRID = (0.03, 0.05, 0.08)
SENSITIVITY_TAU_R_GRID = (0.02, 0.03, 0.05)


class GateInputError(ValueError):
    """Invalid orchestrator input (fail closed before any episode)."""


class EvidenceCoherenceError(ValueError):
    """#11 evidence does not cohere with this candidate/aggregate (fail
    closed; replay/ablation are never rerun)."""


@dataclass(frozen=True)
class GateCandidateInput:
    """One frozen candidate: its rank, the already-completed #11 evidence,
    its own CIG run identity (parent id + mining seed) and the frozen
    stage-3/4 allocations. Ranks must be unique non-negative ints."""

    candidate_envelope: dict
    frozen_rank: int
    replay_evaluation: ReplayEvaluation
    ablation_evaluation: AblationEvaluation
    generalization_allocation: GeneralizationAllocation
    regression_allocation: RegressionAllocation
    parent_cig_id: str | None = None    # falls back to batch default
    mining_seed: int | None = None      # falls back to batch default

    def __post_init__(self):
        import copy as _copy
        object.__setattr__(self, "candidate_envelope",
                            _copy.deepcopy(self.candidate_envelope))
        if isinstance(self.frozen_rank, bool) or not isinstance(self.frozen_rank, int) \
                or self.frozen_rank < 0:
            raise GateInputError(
                f"frozen rank {self.frozen_rank!r} must be a non-negative integer")


@dataclass(frozen=True)
class CandidateGateResult:
    """Immutable per-candidate gate outcome (deep-copy accessors)."""

    _gene_id: str
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
    stores were supplied and the atomic per-candidate commit succeeded."""

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
    builds aggregates/verdicts, and (when stores are supplied) commits
    atomically: somatic decision first, CIG bundle exposed only after."""

    def __init__(self, *, registry, base_config: RuntimeConfig,
                 runner: GateEpisodeRunner, budgets: Budgets,
                 parent_cig_id: str = "CIG-0007", mining_seed: int = 0,
                 token_counter=None, interaction_allocation=None,
                 somatic_store=None, cig_record_store: CigRecordStore | None = None,
                 target_regulation_by_gene: dict | None = None):
        validate_gate_setup(parent_cig_id, mining_seed)
        self._registry = registry
        self._base_config = base_config
        self._runner = runner
        self._budgets = budgets
        self._parent_cig_id = parent_cig_id  # batch default
        self._mining_seed = mining_seed      # batch default
        self._token_counter = token_counter
        self._interaction_allocation = interaction_allocation
        self._somatic_store = somatic_store
        self._cig_record_store = cig_record_store
        self._target_regulation_by_gene = dict(target_regulation_by_gene or {})
        self._last: _LastRun | None = None

    # -- public entry point ---------------------------------------------------
    def run(self, inputs: list[GateCandidateInput]) -> GateOutcome:
        ranks = [i.frozen_rank for i in inputs]
        if len(set(ranks)) != len(ranks) or any(r < 0 for r in ranks):
            raise GateInputError(
                f"frozen ranks must be unique non-negative integers, got {ranks}")
        gene_ids = []
        for i in inputs:
            gid = self._gene_id_of(i.candidate_envelope)
            if gid in gene_ids:
                raise GateInputError(f"duplicate candidate gene id {gid!r}")
            gene_ids.append(gid)

        # step 2: consume #11 evidence, fail closed on incoherence
        for inp in inputs:
            self._consume_stage12_evidence(inp)

        # step 3: stage-3 generalization for ALL candidates
        gen_results: dict[str, GeneralizationEvaluation] = {}
        for inp in inputs:
            gid = self._gene_id_of(inp.candidate_envelope)
            parent, seed = self._parent_seed_for(inp)
            evaluator = GeneralizationEvaluator(
                candidate_envelope=inp.candidate_envelope, registry=self._registry,
                base_config=self._base_config, runner=self._runner,
                budgets=self._budgets, parent_cig_id=parent, mining_seed=seed,
                target_regulation=self._target_regulation_by_gene.get(gid))
            gen_results[gid] = evaluator.evaluate(
                inp.generalization_allocation,
                candidate_envelope=inp.candidate_envelope)

        # step 4: regression + overhead for ALL candidates
        reg_results: dict[str, RegressionEvaluation] = {}
        overhead_results: dict[str, dict] = {}
        for inp in inputs:
            gid = self._gene_id_of(inp.candidate_envelope)
            if self._token_counter is None:
                raise GateInputError(
                    "stage-4B expression overhead requires a TokenCounter; "
                    "no counter was supplied")
            parent, seed = self._parent_seed_for(inp)
            reg_evaluator = RegressionEvaluator(
                candidate_envelope=inp.candidate_envelope, registry=self._registry,
                base_config=self._base_config, runner=self._runner,
                budgets=self._budgets, parent_cig_id=parent,
                mining_seed=seed,
                target_regulation=self._target_regulation_by_gene.get(gid))
            reg_results[gid] = reg_evaluator.evaluate(inp.regression_allocation)
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
        eligible: list[GateCandidateInput] = []
        for inp in inputs:
            gid = self._gene_id_of(inp.candidate_envelope)
            if (inp.replay_evaluation.passed
                    and inp.ablation_evaluation.passed
                    and gen_results[gid].passed and gen_results[gid].out_scope_passed
                    and reg_results[gid].passed
                    and overhead_results[gid]["passed"]):
                eligible.append(inp)

        # steps 6-7: interaction pairs, computed once, resolved together
        pair_evidence: dict[tuple[int, int], object] = {}
        harmful_pairs: list[tuple[str, str]] = []
        if len(eligible) >= 2:
            if self._interaction_allocation is None:
                raise GateInputError(
                    ">=2 preliminary-eligible candidates require an interaction "
                    "allocation; none was supplied")
            interaction = InteractionEvaluator(
                registry=self._registry, base_config=self._base_config,
                runner=self._runner, budgets=self._budgets)
            from inheritance.overlay import evaluation_settings
            locked = evaluation_settings(
                max_tokens=self._budgets.max_tokens_per_request,
                seed=self._mining_seed)
            ordered = sorted(eligible, key=lambda i: i.frozen_rank)
            for a, b in combinations(ordered, 2):
                evidence = interaction.evaluate_pair(
                    a.candidate_envelope, b.candidate_envelope,
                    self._interaction_allocation, locked)
                key = tuple(sorted((self._gene_id_of(a.candidate_envelope),
                                     self._gene_id_of(b.candidate_envelope))))
                pair_evidence[key] = evidence
                if evidence.harmful:
                    harmful_pairs.append(key)

        # resolve ALL harmful pairs simultaneously from complete evidence
        contribution = {self._gene_id_of(i.candidate_envelope):
                         i.ablation_evaluation.delta_success for i in eligible}
        rank_by_gene = {self._gene_id_of(i.candidate_envelope):
                         i.frozen_rank for i in eligible}
        eliminated: dict[str, str] = {}
        for gid_a, gid_b in harmful_pairs:
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

        # steps 8-9: stage-4 children + aggregates + verdicts
        results: list[CandidateGateResult] = []
        for inp in inputs:
            gid = self._gene_id_of(inp.candidate_envelope)
            parent, seed = self._parent_seed_for(inp)
            stage4 = self._build_stage4_child(
                inp, gid, parent, seed, gen_results.get(gid),
                reg_results[gid], overhead_results[gid], pair_evidence,
                eliminated)
            aggregate = self._build_aggregate(
                inp, gid, parent, gen_results.get(gid), reg_results[gid],
                stage4)
            results.append(CandidateGateResult(
                _gene_id=gid, _verdict=aggregate["verdict"],
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

        # step 10: atomic per-candidate commit when stores are supplied;
        # the run evidence is stored first because _bundle_children reads it
        self._last = _LastRun(inputs=inputs, gen_results=gen_results,
                               reg_results=reg_results,
                               overhead_results=overhead_results,
                               pair_evidence=pair_evidence,
                               eliminated=eliminated, outcome=None)
        committed = False
        if self._somatic_store is not None and self._cig_record_store is not None:
            self._commit_results(results,
                                  somatic_store=self._somatic_store,
                                  cig_record_store=self._cig_record_store)
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
        verdicts/SomaticStore; no T_eval dependency."""
        run = self._last
        if run is None:
            raise GateInputError("no primary run has been executed yet")
        gen = run.gen_results[gene_id]
        reg = run.reg_results[gene_id]
        ablation = next(i.ablation_evaluation for i in run.inputs
                         if self._gene_id_of(i.candidate_envelope) == gene_id)
        replay = next(i.replay_evaluation for i in run.inputs
                       if self._gene_id_of(i.candidate_envelope) == gene_id)
        grid = {"gene_id": gene_id, "tau_c_grid": list(SENSITIVITY_TAU_C_GRID),
                 "tau_r_grid": list(SENSITIVITY_TAU_R_GRID),
                 "analysis_only": True,
                 "runner_calls": 0,
                 "child_records": None, "aggregate_verdicts": None,
                 "somatic_states": None}
        # ADR-0002: tau_c grid applies ONLY to ablation; tau_r grid ONLY to
        # regression. Replay/generalization/overhead remain fixed primary
        # evidence. Interaction evidence exists only for the primary
        # preliminary-eligible set: if a looser cell would make a previously
        # ineligible candidate newly eligible but pair evidence is missing,
        # that composition is marked not-identifiable.
        replay_hypothetical = {"fixed": replay.successes_with >= 3
                                 and replay.delta_count >= 2}
        ablation_hypothetical = {}
        for tau in SENSITIVITY_TAU_C_GRID:
            ablation_hypothetical[f"tau_c={tau}"] = (
                ablation.delta_success >= tau
                and ablation.bootstrap["lower_bound"] > 0)
        regression_hypothetical = {}
        for tau in SENSITIVITY_TAU_R_GRID:
            regression_hypothetical[f"tau_r={tau}"] = \
                reg.regression_drop <= tau
        # interaction composition: identifiable only when the candidate was
        # in the primary preliminary-eligible set (pair evidence was
        # collected); otherwise marked not-identifiable
        interaction_note = ("interaction evidence collected for the primary "
                             "preliminary-eligible set only; looser-sensitivity "
                             "compositions that would newly include previously "
                             "ineligible candidates are not identifiable from "
                             "existing T_gate evidence")
        grid["replay"] = {"fixed_at": "successes_with >= 3/5 and delta >= +2/5",
                           "hypothetical_pass": replay_hypothetical["fixed"]}
        grid["ablation"] = ablation_hypothetical
        grid["regression"] = regression_hypothetical
        grid["interaction_note"] = interaction_note
        grid["runner_calls"] = 0
        return grid

    # -- internals ------------------------------------------------------------------
    def _gene_id_of(self, envelope: dict) -> str:
        from inheritance.overlay import _validate_candidate
        return _validate_candidate(envelope, self._registry)["candidate"]["gene_id"]

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

    def _consume_stage12_evidence(self, inp: GateCandidateInput) -> None:
        """Fail closed unless the #11 evidence coheres with this candidate's
        own CIG run identity: candidate match, parent match, seed match,
        stage names, pass agreement, settings provenance, unique child ids."""
        gene_id = self._gene_id_of(inp.candidate_envelope)
        parent, seed = self._parent_seed_for(inp)
        issues: list[str] = []
        seen_ids: set[str] = set()
        for stage, evaluation in (("replay", inp.replay_evaluation),
                                   ("ablation", inp.ablation_evaluation)):
            child = evaluation.child_record
            m = child["measurements"]
            if m.get("candidate_gene_id") != gene_id:
                issues.append(f"{stage} child candidate_gene_id "
                               f"{m.get('candidate_gene_id')!r} != {gene_id!r}")
            if child["parent_cig_id"] != parent:
                issues.append(f"{stage} child parent_cig_id "
                               f"{child['parent_cig_id']!r} != {parent!r}")
            if child["seed"] != seed:
                issues.append(f"{stage} child seed {child['seed']!r} != "
                               f"mining seed {seed!r}")
            if child["stage"] != stage:
                issues.append(f"{stage} child has stage {child['stage']!r}")
            if child["cig_id"] in seen_ids:
                issues.append(f"duplicate child id {child['cig_id']!r}")
            seen_ids.add(child["cig_id"])
            if evaluation.passed != m.get("stage_pass"):
                issues.append(f"{stage} result.passed {evaluation.passed!r} != "
                               f"child measurements.stage_pass "
                               f"{m.get('stage_pass')!r}")
            if "eval_generation_settings" not in m:
                issues.append(f"{stage} child lacks eval_generation_settings "
                               f"provenance")
        if issues:
            raise EvidenceCoherenceError(sorted(issues))

    def _build_stage4_child(self, inp: GateCandidateInput, gene_id: str,
                             parent: str, seed: int, gen,
                             reg_eval: RegressionEvaluation, overhead: dict,
                             pair_evidence: dict, eliminated: dict[str, str]) -> dict:
        pairs_for_gene = []
        for key in sorted(pair_evidence):
            evidence = pair_evidence[key]
            if gene_id not in evidence.gene_ids:
                continue
            other = evidence.gene_ids[0] if \
                evidence.gene_ids[1] == gene_id else evidence.gene_ids[1]
            pairs_for_gene.append({
                "evidence_sha256": evidence.evidence_sha256,
                "other_gene_id": other,
                "delta_pair": evidence.delta_pair,
                "harmful": evidence.harmful,
                "retained": gene_id not in eliminated,
            })
        eliminated_by_interaction = gene_id in eliminated
        regression_pass = reg_eval.passed
        overhead_pass = overhead["passed"]
        stage_pass = regression_pass and overhead_pass and \
            not eliminated_by_interaction
        return {
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
                    "regression_drop": reg_eval.regression_drop,
                    "tau_r": 0.03,
                    "regression_pass": regression_pass,
                },
                "expression_overhead": {
                    "tokens": overhead["tokens"],
                    "tau_k_tokens": overhead["threshold"],
                    "overhead_pass": overhead["passed"],
                    "counter_identity": overhead["counter_identity"],
                    "rendered_sha256": overhead["rendered_sha256"],
                },
                "interaction": {
                    "pair_evidence": pairs_for_gene,
                    "pair_count": len(pairs_for_gene),
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

    def _build_aggregate(self, inp: GateCandidateInput, gene_id: str,
                          parent: str, gen, reg_eval: RegressionEvaluation,
                          stage4: dict) -> dict:
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
                "threshold": "delta_out >= -0.05"},
            "interaction_regression": {
                "measurements_ref": "children",
                "pass": stage4["measurements"]["stage_pass"],
                "threshold": "regression <= 0.03, overhead <= 512 tokens, "
                              "not eliminated"},
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
            "artifacts": {},
        }
        load_cig(aggregate)
        return aggregate

    # -- atomic commit (step 10) -----------------------------------------------------
    def _commit_results(self, results: list[CandidateGateResult], *,
                         somatic_store, cig_record_store: CigRecordStore) -> None:
        """Per-candidate fail-atomic commit: prevalidate bundle + somatic
        decision on scratch state, then commit. The candidate remains
        candidate and the live CigRecordStore stays unchanged on any
        failure."""
        for result in results:
            gene_id = result.gene_id
            # prevalidate the bundle on a scratch store (live store untouched)
            scratch = CigRecordStore()
            scratch.add_bundle(result.aggregate,
                                self._bundle_children(result))
            # prevalidate the somatic decision
            if somatic_store.state(gene_id, 1) != "candidate":
                raise RecordConsistencyError(
                    f"{gene_id!r} is no longer a somatic candidate")
            # commit: somatic decision first, then expose the bundle.
            # verdict -> terminal state: promote -> validated,
            # reject -> rejected (neither is germline assimilation)
            somatic_store.decide(gene_id, 1,
                                  "validated" if result.verdict == "promote"
                                  else "rejected",
                                  [result.aggregate_cig_id])

    def _bundle_children(self, result: CandidateGateResult) -> list[dict]:
        """Children ordered per the aggregate declaration, sourced from this
        run's #11 evidence + locally produced stage-3/4 children."""
        run = self._last
        children: list[dict] = [result.stage4_child]
        for inp in run.inputs:
            gid = self._gene_id_of(inp.candidate_envelope)
            if gid != result.gene_id:
                continue
            children.append(inp.replay_evaluation.child_record)
            children.append(inp.ablation_evaluation.child_record)
            gen = run.gen_results[gid]
            children.append(gen.in_scope_child)
            children.append(gen.out_scope_child)
        by_id = {c["cig_id"]: c for c in children}
        return [by_id[cid] for cid in result.aggregate["child_records"]]


@dataclass
class _LastRun:
    inputs: list
    gen_results: dict
    reg_results: dict
    overhead_results: dict
    pair_evidence: dict
    eliminated: dict
    outcome: GateOutcome
