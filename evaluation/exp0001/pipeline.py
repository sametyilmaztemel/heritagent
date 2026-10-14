"""The EXP-0001 run, end to end (issue #13 Phase A; design sections 2-8;
ADR-0006, ADR-0007).

Order: every seed's G0 lifetime, then the one shared T_reg, then per
mining seed the miner (frozen candidate set), the CIG (replay and ablation
per candidate, then stages 3-4, interactions and the somatic decisions),
and the arm genomes; then the final evaluation (arm A once, arms B and C
per seed, on T_eval, T_reg and the transfer subset) and the final
statistics. T_eval is touched only by the final evaluation.

Every stage result is a write-once artifact in the run directory and
every episode a ledger record, so a stopped run resumes without running
any episode or teacher call twice. A hard cap, or a protocol rule that
fails closed, stops the run as incomplete with its reason; an incomplete
run gets no primary verdict (ADR-0006 D5). Nothing here labels a result
as validated (SPEC section 32)."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping
from pathlib import Path

from evaluation.exp0001.allocations import (
    AllocationError,
    ablation_allocation,
    evaluation_task,
    generalization_allocation,
    interaction_allocation,
    pool_tasks,
    replay_allocation,
)
from evaluation.exp0001.arms import arm_genomes
from evaluation.exp0001.episodes import EpisodeExecutor, EpisodeSpec, protocol_budgets
from evaluation.exp0001.gate_runner import LedgerGateRunner
from evaluation.exp0001.ledger import CapExceeded, RunLedger
from evaluation.exp0001.lifetime import (
    LifetimeError,
    derive_t_reg,
    eligible_count,
    mining_batch,
    run_lifetime,
    t_mine_tasks,
    t_reg_record,
)
from evaluation.exp0001.protocol import (
    CANDIDATE_CAP,
    CAPS,
    MAX_TOKENS_PER_REQUEST,
    MINING_SEEDS,
    MINING_TEMPERATURE,
)
from evaluation.exp0001.provenance import ProvenanceError, freeze_json, verify_run_start_record
from evaluation.metrics.final_statistics import (
    ArmRun,
    FinalEvaluationInput,
    TransferOutcome,
    analyze_final_evaluation,
    measure_genome_size,
    summarize_adaptation_cost,
    summarize_genome_sizes,
)
from genome.validation.records import CigRecordStore
from genome.validation.registry import TraitRegistry
from inheritance.ablation import AblationEvaluator
from inheritance.gate import CigGateOrchestrator, GateCandidateInput
from inheritance.overlay import evaluation_settings
from inheritance.regression import RegressionAllocation, TRegTask
from inheritance.replay import ReplayEvaluator
from runtime.expression import compile_runtime_config
from runtime.seed_g0 import build_g0_seed_genome
from somatic.store import SomaticStore
from trajectory.storage.canonical import canonical_json, sha256_hex
from traits.miner.miner import MiningSettings, TraitMiner

MINER_MAX_TOKENS = 2048  # traits.miner.MiningSettings default
CANDIDATES_SCHEMA = "heritagent/exp0001-candidates/0.1"
GATE_SCHEMA = "heritagent/exp0001-gate/0.1"
ARMS_SCHEMA = "heritagent/exp0001-arms/0.1"
REPORT_SCHEMA = "heritagent/exp0001-report/0.1"
SHARED = "once"  # arm A's run label (ADR-0006 D7)


class RunError(RuntimeError):
    """The run directory is inconsistent with the protocol (for example a
    stage committed without its frozen artifact); it needs a human."""


def cig_id(mining_seed: int, rank: int | None) -> str:
    """CIG-<seed><rank+1>, two digits each; rank None is the seed's batch id."""
    return f"CIG-{mining_seed:02d}{0 if rank is None else rank + 1:02d}"


def _load(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _t_reg_allocation(doc: Mapping) -> RegressionAllocation:
    return RegressionAllocation(tasks=tuple(
        TRegTask(task_id=t["task_id"], family=t["family"],
                 baseline_trajectory_id=t["baseline_trajectory_id"],
                 payload={k: v for k, v in t.items()
                          if k not in ("task_id", "family", "baseline_trajectory_id")})
        for t in doc["tasks"]))


class Exp0001Run:
    def __init__(self, *, run_dir: Path, run_start: dict, split_manifest: dict, model, teacher,
                 env_factory: Callable[[str, str], object], token_counter,
                 ledger_clock: Callable[[], float] = time.time,
                 episode_clock: Callable[[], float] | None = None,
                 sleep: Callable[[float], None] = time.sleep, caps: Mapping = CAPS):
        self.dir = Path(run_dir)
        verify_run_start_record(run_start)
        if run_start["splits"]["manifest_sha256"] != split_manifest["manifest_sha256"]:
            raise ProvenanceError("the run-start record names another split manifest")
        self.registry = TraitRegistry(self.dir / "registry")
        self.g0 = build_g0_seed_genome(self.registry)
        if run_start["g0"]["genome_id"] != self.g0["genome_id"]:
            raise ProvenanceError("the run-start record names another G0")
        freeze_json(self.dir / "run-start.json", run_start)
        self.run_start = run_start
        self.splits = split_manifest
        self.teacher = teacher
        self.token_counter = token_counter
        self.g0_config = compile_runtime_config(self.g0, self.registry)
        self.budgets = protocol_budgets()
        self.ledger = RunLedger(self.dir / "ledger.jsonl", run_id=run_start["run_id"],
                                clock=ledger_clock, caps=caps)
        self.executor = EpisodeExecutor(
            model=model, env_factory=env_factory, ledger=self.ledger,
            trajectories_dir=self.dir / "trajectories", run_id=run_start["run_id"],
            code_commit=run_start["code"]["commit"], budgets=self.budgets,
            clock=episode_clock, sleep=sleep)
        self.t_mine = t_mine_tasks(split_manifest)
        self._final_cells: dict | None = None

    # -- stages -------------------------------------------------------------
    def _mine(self, seed: int, results) -> list[dict]:
        path = self.dir / f"seed-{seed}" / "candidates.json"
        frozen = _load(path)
        key = f"mining/{seed}"
        if frozen is None:
            batch = mining_batch(results, self.t_mine, seed)
            # one teacher call per pair (traits.miner.TraitMiner.mine)
            if not self.ledger.admit_calls(key, len(batch.pairs)):
                raise RunError(f"{key} was charged but its frozen candidate set is missing")
            miner = TraitMiner(self.teacher, self.registry,
                               settings=MiningSettings(temperature=MINING_TEMPERATURE, seed=seed,
                                                       max_tokens=MINER_MAX_TOKENS,
                                                       max_proposals=CANDIDATE_CAP),
                               born_generation=0)
            mined = miner.mine(batch)
            self.ledger.record_calls(key, model_calls=len(batch.pairs))
            frozen = freeze_json(path, {
                "schema": CANDIDATES_SCHEMA, "mining_seed": seed,
                "candidates": list(mined.candidates),
                "candidate_set_sha256": mined.candidate_set_sha256,
                "mining_records": list(mined.mining_records), "rejected": list(mined.rejected)})
        digest = sha256_hex(canonical_json(frozen["candidates"]).encode("utf-8"))
        if digest != frozen["candidate_set_sha256"]:  # the miner's own digest rule
            raise RunError(f"{path}: the candidate set does not match its digest")
        return frozen["candidates"]

    def _somatic(self, seed: int, candidates) -> SomaticStore:
        path = self.dir / f"seed-{seed}" / "somatic.jsonl"
        store = (SomaticStore.open(path, self.registry) if path.exists()
                 else SomaticStore.create(path, self.registry))
        for envelope in candidates:
            gene, version = envelope["candidate"]["gene_id"], envelope["candidate"]["version"]
            state = store.state(gene, version)
            if state is None:
                store.add_candidate(envelope)
            elif state != "candidate":
                raise RunError(f"seed {seed}: {gene!r} is already {state} but the gate artifact "
                               f"is missing; the decision cannot be re-run")
        return store

    def _gate(self, seed: int, candidates, t_reg: RegressionAllocation, results) -> dict:
        path = self.dir / f"seed-{seed}" / "gate.json"
        done = _load(path)
        if done is not None:
            return done
        content = {t.task_id: t.content_sha256 for t in self.t_mine}
        by_trajectory = {r.trajectory_id: r for r in results}
        # every allocation is built, and fails closed, before any gate episode
        plans = []
        for rank, envelope in enumerate(candidates):
            source = by_trajectory.get(envelope["candidate"]["provenance"]["source_trajectory"])
            if source is None:
                raise RunError(f"{envelope['candidate']['gene_id']!r} cites a trajectory outside "
                               f"seed {seed}'s lifetime")
            source_task = evaluation_task({"game_id": source.task_id, "family": source.family,
                                           "content_sha256": content[source.task_id]}, "t_mine")
            plans.append((rank, envelope, replay_allocation(self.splits, source_task),
                          generalization_allocation(self.splits, envelope, self.registry)))
        ablation, interaction = ablation_allocation(self.splits), interaction_allocation(self.splits)
        somatic = self._somatic(seed, candidates)

        inputs = []
        for rank, envelope, replay_plan, generalization_plan in plans:
            gene = envelope["candidate"]["gene_id"]
            common = dict(candidate_envelope=envelope, registry=self.registry,
                          base_config=self.g0_config, budgets=self.budgets,
                          parent_cig_id=cig_id(seed, rank), mining_seed=seed)
            replay = ReplayEvaluator(
                runner=LedgerGateRunner(self.executor, f"gate/{seed}/{gene}/replay", self.budgets),
                **common).evaluate(replay_plan)
            ablated = AblationEvaluator(
                runner=LedgerGateRunner(self.executor, f"gate/{seed}/{gene}/ablation",
                                        self.budgets),
                **common).evaluate(ablation)
            inputs.append(GateCandidateInput(
                candidate_envelope=envelope, frozen_rank=rank, replay_evaluation=replay,
                ablation_evaluation=ablated, generalization_allocation=generalization_plan,
                regression_allocation=t_reg, parent_cig_id=cig_id(seed, rank), mining_seed=seed))
        records = CigRecordStore()
        outcome = CigGateOrchestrator(
            registry=self.registry, base_config=self.g0_config,
            runner=LedgerGateRunner(self.executor, f"gate/{seed}/stage3-4", self.budgets),
            budgets=self.budgets, parent_cig_id=cig_id(seed, None), mining_seed=seed,
            token_counter=self.token_counter, interaction_allocation=interaction,
            somatic_store=somatic, cig_record_store=records).run(inputs)
        if not outcome.committed:
            raise RunError(f"seed {seed}: the gate did not commit its decisions")
        records.require_consistent()
        issues = somatic.verify()
        if issues:
            raise RunError(f"seed {seed}: the somatic journal does not verify: {issues}")
        return freeze_json(path, {
            "schema": GATE_SCHEMA, "mining_seed": seed,
            "verdicts": outcome.verdicts,
            "aggregate_ids": {r.gene_id: r.aggregate_cig_id for r in outcome.results},
            "eliminated_by_interaction": {r.gene_id: r.eliminated_by for r in outcome.results},
            "aggregates": records.aggregates, "children": records.children})

    def _arms(self, seed: int, candidates, gate: Mapping) -> dict:
        path = self.dir / f"seed-{seed}" / "arms.json"
        done = _load(path)
        if done is None:
            genomes = arm_genomes(self.g0, candidates, gate["verdicts"], gate["aggregate_ids"],
                                  self.registry, seed)
            done = freeze_json(path, {"schema": ARMS_SCHEMA, "mining_seed": seed, **genomes})
        return {"B": done["B"], "C": done["C"]}

    def _evaluate(self, arm: str, label: str, genome: dict, tasks, task_set: str,
                  cells: dict) -> None:
        config = compile_runtime_config(genome, self.registry)
        settings = evaluation_settings(max_tokens=MAX_TOKENS_PER_REQUEST, seed=None)
        for task in tasks:
            result = self.executor.run(EpisodeSpec(
                key=f"final/{arm}/{label}/{task_set}/{task.task_id}", task_id=task.task_id,
                family=task.family, split=task.payload["split"], config=config, settings=settings,
                tags={"stage": "final", "arm": arm, "run": label, "task_set": task_set}))
            cells[task.task_id] = result

    def _final(self, genomes: Mapping[tuple[str, int | None], dict], t_reg) -> dict:
        t_eval = pool_tasks(self.splits, "t_eval")
        transfer = pool_tasks(self.splits, "ablation")  # ADR-0007 E4
        t_reg_tasks = [evaluation_task({"game_id": t.task_id, "family": t.family,
                                        "content_sha256": t.payload["content_sha256"]}, "t_mine")
                       for t in t_reg.tasks]
        self._final_cells = {key: {"t_eval": {}, "t_reg": {}, "transfer": {}} for key in genomes}
        for (arm, seed), genome in genomes.items():
            label = SHARED if seed is None else str(seed)
            cells = self._final_cells[(arm, seed)]
            self._evaluate(arm, label, genome, t_eval, "t_eval", cells["t_eval"])
            self._evaluate(arm, label, genome, t_reg_tasks, "t_reg", cells["t_reg"])
            self._evaluate(arm, label, genome, transfer, "transfer", cells["transfer"])
        return {
            "statistics": analyze_final_evaluation(self._final_input(t_reg, complete=True)),
            "genome_sizes": summarize_genome_sizes({
                key: measure_genome_size(genome, self.registry, self.token_counter)
                for key, genome in genomes.items()}),
            "adaptation_cost": summarize_adaptation_cost(
                [t.task_id for t in transfer],
                {key: {task_id: TransferOutcome(success=r.success,
                                                total_tokens=r.prompt_tokens + r.completion_tokens,
                                                steps=r.steps)
                       for task_id, r in cells["transfer"].items()}
                 for key, cells in self._final_cells.items()}),
        }

    def _final_input(self, t_reg, *, complete: bool, reason: str | None = None):
        runs = tuple(ArmRun(arm=arm, seed=seed,
                            t_eval={t: r.success for t, r in cells["t_eval"].items()},
                            t_reg={t: r.success for t, r in cells["t_reg"].items()})
                     for (arm, seed), cells in self._final_cells.items())
        return FinalEvaluationInput(
            t_eval_task_ids=tuple(t.task_id for t in pool_tasks(self.splits, "t_eval")),
            t_reg_task_ids=tuple(t.task_id for t in t_reg.tasks), runs=runs,
            complete=complete, incomplete_reason=reason)

    # -- the run ----------------------------------------------------------------
    def run(self) -> dict:
        if self.ledger.state != "running":  # a finished run is never run again
            report = _load(self.dir / "report.json")
            if report is None:
                raise RunError(f"the run is {self.ledger.state} but has no report")
            return report
        stage, t_reg = "lifetime", None
        try:
            lifetimes = {seed: run_lifetime(self.executor, self.t_mine, self.g0_config, seed)
                         for seed in MINING_SEEDS}
            stage = "t_reg"
            t_reg_doc = _load(self.dir / "t_reg.json")
            if t_reg_doc is None:
                t_reg_doc = freeze_json(self.dir / "t_reg.json", t_reg_record(
                    derive_t_reg(lifetimes, self.t_mine), eligible_count(lifetimes)))
            t_reg = _t_reg_allocation(t_reg_doc)
            genomes: dict = {("A", None): self.g0}
            for seed in MINING_SEEDS:
                stage = f"mining/{seed}"
                candidates = self._mine(seed, lifetimes[seed])
                stage = f"gate/{seed}"
                gate = self._gate(seed, candidates, t_reg, lifetimes[seed])
                arms = self._arms(seed, candidates, gate)
                genomes[("B", seed)], genomes[("C", seed)] = arms["B"], arms["C"]
            stage = "final"
            final = self._final({key: genomes[key] for key in sorted(
                genomes, key=lambda k: (k[0], -1 if k[1] is None else k[1]))}, t_reg)
        except (CapExceeded, LifetimeError, AllocationError) as exc:
            reason = str(exc)
            if not isinstance(exc, CapExceeded):
                self.ledger.abandon(f"{stage}: {reason}")
            report = {"schema": REPORT_SCHEMA, "status": "incomplete", "stage": stage,
                      "reason": self.ledger.reason, "episodes": self.ledger.episodes,
                      "model_calls": self.ledger.model_calls, "statistics": None}
            if stage == "final" and self._final_cells is not None:
                report["statistics"] = analyze_final_evaluation(
                    self._final_input(t_reg, complete=False, reason=self.ledger.reason))
            return freeze_json(self.dir / "report.json", report)
        self.ledger.complete()
        return freeze_json(self.dir / "report.json", {
            "schema": REPORT_SCHEMA, "status": "complete", "stage": "done", "reason": None,
            "episodes": self.ledger.episodes, "model_calls": self.ledger.model_calls, **final})
