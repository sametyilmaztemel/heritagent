"""The EXP-0001 run, end to end (issue #13 Phase A; design sections 2-8;
ADR-0006, ADR-0007).

Order: every seed's G0 lifetime, then the one shared T_reg, then per
mining seed the miner (frozen candidate set), the CIG (replay and ablation
per candidate, then stages 3-4, interactions and the somatic decisions),
and the arm genomes; then the final evaluation (arm A once, arms B and C
per seed, on T_eval, T_reg and the transfer subset) and the final
statistics. T_eval is touched only by the final evaluation.

Every stage result is a write-once artifact in the run directory and
every episode a ledger record, so a stopped run resumes without running a
recorded episode, or a frozen seed's teacher calls, again. A crash inside
an episode or a mining batch re-runs that unit, and its reservation stays
charged. The gate's decisions reach their artifact before the somatic
journal; on completion and on a protocol stop the report is frozen before
the ledger's terminal record. A hard cap writes its terminal record first:
a crash before its report leaves a report rebuilt from the ledger, without
the stage or partial statistics. A hard cap, or a protocol rule that fails
closed, stops the run as incomplete with its reason; an incomplete run gets
no primary verdict (ADR-0006 D5). Nothing here labels a result as
validated (SPEC section 32).

With ``workers`` > 1 independent episodes run in parallel (ADR-0008): the
lifetimes of all seeds at once, the final evaluation at once, and the
gate in three stretches whose episodes do not depend on each other's
outcomes (replay and ablation of every candidate; generalization and
regression of every candidate; the interaction pairs of the eligible
set). A dry pass of the real gate code collects each stretch's episodes,
they run in parallel, and the real pass then finds them recorded; a
recorded gate episode the real pass does not request fails the run closed,
in this session and in every later one.
The episodes and decisions are the sequential run's; the order in which
the episodes run is not."""

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
from evaluation.exp0001.gate_runner import LedgerGateRunner, PrefetchGateRunner, PrefetchStop
from evaluation.exp0001.ledger import CapExceeded, RunLedger
from evaluation.exp0001.lifetime import (
    LifetimeError,
    derive_t_reg,
    eligible_count,
    mining_batch,
    run_lifetimes,
    t_mine_tasks,
    t_reg_record,
)
from evaluation.exp0001.protocol import (
    CANDIDATE_CAP,
    LOCKED_SPLIT_MANIFEST_SHA256,
    MAX_TOKENS_PER_REQUEST,
    MINER_CALLS_PER_PAIR_MAX,
    MINER_MAX_TOKENS,
    MINING_SEEDS,
    MINING_TEMPERATURE,
)
from evaluation.exp0001.provenance import (
    ProvenanceError,
    freeze_json,
    verify_pre_run_artifacts,
    verify_run_start_record,
)
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
from runtime.model_adapters.base import ModelAdapter
from runtime.seed_g0 import build_g0_seed_genome
from somatic.store import SomaticStore
from trajectory.storage.canonical import canonical_json, sha256_hex
from traits.miner.evidence import EVIDENCE_FORMS
from traits.miner.miner import MiningSettings, TraitMiner

CANDIDATES_SCHEMA = "heritagent/exp0001-candidates/0.1"


def teacher_calls(mining_records) -> int:
    """The teacher calls behind the frozen mining records: one per evidence
    form a pair was sent in (ADR-0009 D8). A record names the last form; the
    forms before it overflowed the context window."""
    return sum(1 + EVIDENCE_FORMS.index(record.get("evidence_form", EVIDENCE_FORMS[0]))
               for record in mining_records)
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
    """One EXP-0001 run in one run directory, single-use: the directory is
    locked from construction until run() returns (one writer); the caps, the
    model, the teacher and the token counter must be the ones the frozen
    run-start record names."""

    def __init__(self, *, run_dir: Path, run_start: dict, split_manifest: dict, model, teacher,
                 env_factory: Callable[[str, str], object], token_counter,
                 ledger_clock: Callable[[], float] = time.time,
                 episode_clock: Callable[[], float] | None = None,
                 sleep: Callable[[float], None] = time.sleep,
                 expected_split_sha256: str = LOCKED_SPLIT_MANIFEST_SHA256,
                 workers: int = 1, sleep_between_retries: bool = True,
                 session_info: Mapping | None = None,
                 on_ledger_append: Callable[[dict], None] | None = None,
                 pause_when: Callable[[], bool] | None = None,
                 lock=None):
        """``workers`` > 1 runs episodes in that many worker processes
        (the model and the environment factory must pickle; ADR-0008).
        ``session_info`` goes into the ledger's session record;
        ``on_ledger_append`` is the run mirror's hook; when ``pause_when``
        holds before a new admission the run stops with RunPaused, still
        running, after recording every episode in flight. ``lock`` is the
        caller's ``lock_run_dir`` handle on ``run_dir``, which the caller
        holds longer than the run (the cli: through the last snapshot);
        without it the run takes and releases its own."""
        self.dir = Path(run_dir)
        self._lock, self._owns_lock = lock, lock is None
        self.executor = None
        verify_run_start_record(run_start)
        # the split manifest must verify, be the pre-registered one, and be
        # the one the record names
        verify_pre_run_artifacts(split_manifest, None, expected_split_sha256=expected_split_sha256)
        if run_start["splits"]["manifest_sha256"] != split_manifest["manifest_sha256"]:
            raise ProvenanceError("the run-start record names another split manifest")
        for name, adapter in (("model", model), ("teacher", teacher)):
            metadata = adapter.metadata()
            # the backend version too: one serving build for the whole run (ADR-0008 D4)
            actual = {"model_id": metadata.model_id, "revision": metadata.revision,
                      "backend": metadata.backend, "backend_version": metadata.backend_version}
            recorded = {k: run_start[name][k] for k in actual}
            if actual != recorded:
                raise ProvenanceError(f"the {name} adapter is {actual}, the run-start record "
                                      f"names {recorded}")
        if token_counter.identity != run_start["model"]["tokenizer"]:
            raise ProvenanceError(f"the token counter is {token_counter.identity!r}, the run-start "
                                  f"record names {run_start['model']['tokenizer']!r}")
        self.dir.mkdir(parents=True, exist_ok=True)
        if self._lock is None:
            self._lock = lock_run_dir(self.dir)
        try:
            self._open(run_start, split_manifest, model, teacher, env_factory, token_counter,
                       ledger_clock, episode_clock, sleep,
                       dict(workers=workers, sleep_between_retries=sleep_between_retries,
                            pause_when=pause_when),
                       session_info, on_ledger_append)
        except BaseException:
            self.close()  # a half-built run never keeps the directory locked
            raise

    def _open(self, run_start, split_manifest, model, teacher, env_factory, token_counter,
              ledger_clock, episode_clock, sleep, parallel, session_info,
              on_ledger_append) -> None:
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
        # the caps in force are the recorded ones: never enlarged silently
        self.ledger = RunLedger(self.dir / "ledger.jsonl", run_id=run_start["run_id"],
                                clock=ledger_clock, caps=run_start["protocol"]["caps"],
                                session_info={**dict(session_info or {}),
                                              "workers": parallel["workers"]},
                                on_append=on_ledger_append)
        self.executor = EpisodeExecutor(
            model=model, env_factory=env_factory, ledger=self.ledger,
            trajectories_dir=self.dir / "trajectories", run_id=run_start["run_id"],
            code_commit=run_start["code"]["commit"], budgets=self.budgets,
            clock=episode_clock, sleep=sleep, **parallel)
        self.t_mine = t_mine_tasks(split_manifest)
        self._final_cells: dict | None = None

    def close(self) -> None:
        """Stop the worker processes and release the run directory."""
        if self.executor is not None:
            self.executor.close()
        if self._lock is not None:
            if self._owns_lock:
                self._lock.close()
            self._lock = None

    # -- stages -------------------------------------------------------------
    def _mine(self, seed: int, results) -> list[dict]:
        path = self.dir / f"seed-{seed}" / "candidates.json"
        frozen = _load(path)
        key = f"mining/{seed}"
        if frozen is None:
            batch = mining_batch(results, self.t_mine, seed)
            # at most two teacher calls per pair (traits.miner.TraitMiner.mine,
            # ADR-0009 D8); a crash before the frozen set re-mines, and the
            # aborted attempt stays charged
            if not self.ledger.admit_calls(key, MINER_CALLS_PER_PAIR_MAX * len(batch.pairs)):
                # charged only after the set was frozen: the set was removed by hand
                raise RunError(f"{key} was charged but its frozen candidate set is missing")
            miner = TraitMiner(_Heartbeat(self.teacher, self.ledger.heartbeat), self.registry,
                               settings=MiningSettings(temperature=MINING_TEMPERATURE, seed=seed,
                                                       max_tokens=MINER_MAX_TOKENS,
                                                       max_proposals=CANDIDATE_CAP),
                               born_generation=0)
            mined = miner.mine(batch)
            frozen = freeze_json(path, {
                "schema": CANDIDATES_SCHEMA, "mining_seed": seed,
                "candidates": list(mined.candidates),
                "candidate_set_sha256": mined.candidate_set_sha256,
                "mining_records": list(mined.mining_records), "rejected": list(mined.rejected)})
        if not self.ledger.is_charged(key):
            # frozen before the calls were recorded (or a crash in between)
            self.ledger.record_calls(key, model_calls=teacher_calls(frozen["mining_records"]))
        digest = sha256_hex(canonical_json(frozen["candidates"]).encode("utf-8"))
        if digest != frozen["candidate_set_sha256"]:  # the miner's own digest rule
            raise RunError(f"{path}: the candidate set does not match its digest")
        if frozen["mining_seed"] != seed or len(frozen["candidates"]) > CANDIDATE_CAP:
            raise RunError(f"{path} is not seed {seed}'s frozen candidate set of at most "
                           f"{CANDIDATE_CAP}")
        return frozen["candidates"]

    def _somatic(self, seed: int, candidates) -> SomaticStore:
        path = self.dir / f"seed-{seed}" / "somatic.jsonl"
        store = (SomaticStore.open(path, self.registry) if path.exists()
                 else SomaticStore.create(path, self.registry))
        for envelope in candidates:
            gene, version = envelope["candidate"]["gene_id"], envelope["candidate"]["version"]
            if store.state(gene, version) is None:
                store.add_candidate(envelope)
        return store

    def _check_gate(self, seed: int, candidates, gate: Mapping) -> None:
        """A gate artifact must decide exactly the frozen candidates, and its
        verdicts must be the ones its aggregate records carry."""
        genes = {c["candidate"]["gene_id"]: c["candidate"] for c in candidates}
        if set(gate["verdicts"]) != set(genes) or set(gate["aggregate_ids"]) != set(genes):
            raise RunError(f"seed {seed}: gate.json does not decide exactly the frozen candidates")
        for gene, verdict in gate["verdicts"].items():
            aggregate = gate["aggregates"].get(gate["aggregate_ids"][gene])
            if aggregate is None or aggregate.get("verdict") != verdict \
                    or aggregate.get("candidate_gene_id") != gene:
                raise RunError(f"seed {seed}: gate.json's verdict for {gene!r} is not its "
                               f"aggregate record's")
            bound = (aggregate.get("artifacts") or {}).get("candidate") or {}
            if (bound.get("gene_id"), bound.get("version"), bound.get("artifact")) != \
                    (gene, genes[gene]["version"], genes[gene]["artifact"]):
                raise RunError(f"seed {seed}: {gene!r} was not gated on its frozen artifact")

    def _apply_decisions(self, seed: int, candidates, gate: Mapping) -> None:
        """The gate artifact is written first; the somatic journal then takes
        its decisions, so a crash in between is finished on resume."""
        somatic = self._somatic(seed, candidates)
        for envelope in candidates:
            gene, version = envelope["candidate"]["gene_id"], envelope["candidate"]["version"]
            state = "validated" if gate["verdicts"][gene] == "promote" else "rejected"
            current = somatic.state(gene, version)
            if current == "candidate":
                somatic.decide(gene, version, state, [gate["aggregate_ids"][gene]])
            elif current != state:
                raise RunError(f"seed {seed}: {gene!r} is {current} in the somatic journal, the "
                               f"gate artifact says {state}")
        issues = somatic.verify()
        somatic.close()
        if issues:
            raise RunError(f"seed {seed}: the somatic journal does not verify: {issues}")

    def _gate(self, seed: int, candidates, t_reg: RegressionAllocation, results) -> dict:
        path = self.dir / f"seed-{seed}" / "gate.json"
        done = _load(path)
        if done is None:
            done = self._run_gate(seed, candidates, t_reg, results, path)
        self._check_gate(seed, candidates, done)
        self._apply_decisions(seed, candidates, done)
        return done

    def _run_gate(self, seed: int, candidates, t_reg, results, path: Path) -> dict:
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
        self._somatic(seed, candidates).close()  # the persistent journal holds the candidates

        requested: list[str] = []  # every key the real pass asks for

        def runner(scope: str, dry=None):
            if dry is None:
                return LedgerGateRunner(self.executor, scope, self.budgets, requested=requested)
            return PrefetchGateRunner(self.executor, scope, self.budgets, collected=dry[0],
                                      collect=dry[1])

        def stages12(dry=None) -> list[GateCandidateInput]:
            inputs = []
            for rank, envelope, replay_plan, generalization_plan in plans:
                gene = envelope["candidate"]["gene_id"]
                common = dict(candidate_envelope=envelope, registry=self.registry,
                              base_config=self.g0_config, budgets=self.budgets,
                              parent_cig_id=cig_id(seed, rank), mining_seed=seed)
                replay = ReplayEvaluator(runner=runner(f"gate/{seed}/{gene}/replay", dry),
                                         **common).evaluate(replay_plan)
                ablated = AblationEvaluator(runner=runner(f"gate/{seed}/{gene}/ablation", dry),
                                            **common).evaluate(ablation)
                inputs.append(GateCandidateInput(
                    candidate_envelope=envelope, frozen_rank=rank, replay_evaluation=replay,
                    ablation_evaluation=ablated, generalization_allocation=generalization_plan,
                    regression_allocation=t_reg, parent_cig_id=cig_id(seed, rank),
                    mining_seed=seed))
            return inputs

        def orchestrate(inputs, dry=None):
            # the gate commits into an in-memory mirror of the journal; its
            # artifact is frozen before the persistent journal takes the decisions
            mirror = SomaticStore.create(None, self.registry)
            for envelope in candidates:
                mirror.add_candidate(envelope)
            records = CigRecordStore()
            outcome = CigGateOrchestrator(
                registry=self.registry, base_config=self.g0_config,
                runner=runner(f"gate/{seed}/stage3-4", dry),
                budgets=self.budgets, parent_cig_id=cig_id(seed, None), mining_seed=seed,
                token_counter=self.token_counter, interaction_allocation=interaction,
                somatic_store=mirror, cig_record_store=records).run(inputs)
            return outcome, records

        parallel = self.executor.workers > 1
        if parallel:  # stretch 1: replay and ablation of every candidate
            self._prefetch(stages12, lambda condition: True)
        inputs = stages12()
        if parallel:
            # stretch 2: generalization and regression of every candidate (the
            # pass ends at the first unrecorded interaction episode)
            self._prefetch(lambda dry: orchestrate(inputs, dry),
                           lambda condition: condition in ("with_trait", "without_trait"))
            # stretch 3: the interaction pairs of the now-recorded eligible set
            self._prefetch(lambda dry: orchestrate(inputs, dry),
                           lambda condition: condition in ("both", "neither"))
        outcome, records = orchestrate(inputs)
        # every recorded gate episode of this seed must be one the gate asked
        # for; checked against the ledger, so a resumed run fails closed too
        unrequested = sorted(set(self.ledger.done_keys(f"gate/{seed}/")) - set(requested))
        if unrequested:
            raise RunError(f"seed {seed}: {len(unrequested)} recorded gate episodes were not "
                           f"requested by the gate (first {unrequested[0]!r}); the run left the "
                           f"protocol and needs a human")
        if not outcome.committed:
            raise RunError(f"seed {seed}: the gate did not commit its decisions")
        records.require_consistent()
        return freeze_json(path, {
            "schema": GATE_SCHEMA, "mining_seed": seed,
            "verdicts": outcome.verdicts,
            "aggregate_ids": {r.gene_id: r.aggregate_cig_id for r in outcome.results},
            "eliminated_by_interaction": {r.gene_id: r.eliminated_by for r in outcome.results},
            "aggregates": records.aggregates, "children": records.children})

    def _prefetch(self, dry_pass, collect) -> None:
        """One parallel stretch of the gate: the dry pass runs no episode,
        and ends at PrefetchStop; the episodes it collected run in parallel."""
        collected: list[EpisodeSpec] = []
        try:
            dry_pass((collected, collect))
        except PrefetchStop:
            pass  # the end of the stretch
        except Exception:  # noqa: BLE001
            # a placeholder led the dry pass into an error after it had
            # collected; the real pass meets any real error on its own
            pass
        if collected:
            self.executor.run_many(collected)

    def _arms(self, seed: int, candidates, gate: Mapping) -> dict:
        genomes = arm_genomes(self.g0, candidates, gate["verdicts"], gate["aggregate_ids"],
                              self.registry, seed)
        # rebuilt every time: a frozen arms.json with other content fails closed
        done = freeze_json(self.dir / f"seed-{seed}" / "arms.json",
                           {"schema": ARMS_SCHEMA, "mining_seed": seed, **genomes})
        return {"B": done["B"], "C": done["C"]}

    def _final(self, genomes: Mapping[tuple[str, int | None], dict], t_reg) -> dict:
        t_eval = pool_tasks(self.splits, "t_eval")
        transfer = pool_tasks(self.splits, "ablation")  # ADR-0007 E4
        t_reg_tasks = [evaluation_task({"game_id": t.task_id, "family": t.family,
                                        "content_sha256": t.payload["content_sha256"]}, "t_mine")
                       for t in t_reg.tasks]
        self._final_cells = {key: {"t_eval": {}, "t_reg": {}, "transfer": {}} for key in genomes}
        settings = evaluation_settings(max_tokens=MAX_TOKENS_PER_REQUEST, seed=None)
        planned = []  # (cells, task_id, spec) in the sequential run's order
        for (arm, seed), genome in genomes.items():
            label = SHARED if seed is None else str(seed)
            config = compile_runtime_config(genome, self.registry)
            for task_set, tasks in (("t_eval", t_eval), ("t_reg", t_reg_tasks),
                                    ("transfer", transfer)):
                for task in tasks:
                    planned.append((self._final_cells[(arm, seed)][task_set], task.task_id,
                                    EpisodeSpec(
                        key=f"final/{arm}/{label}/{task_set}/{task.task_id}",
                        task_id=task.task_id, family=task.family, split=task.payload["split"],
                        config=config, settings=settings,
                        tags={"stage": "final", "arm": arm, "run": label,
                              "task_set": task_set})))
        try:
            results = self.executor.run_many([spec for _, _, spec in planned])
        except BaseException:
            # a stop inside the final evaluation: the recorded episodes still
            # feed the incomplete report's partial statistics
            for cells, task_id, spec in planned:
                recorded = self.executor.recorded(spec)
                if recorded is not None:
                    cells[task_id] = recorded
            raise
        for (cells, task_id, _), result in zip(planned, results):
            cells[task_id] = result
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

    def _report(self, status: str, stage: str, reason: str | None, **content) -> dict:
        return {"schema": REPORT_SCHEMA, "status": status, "stage": stage, "reason": reason,
                "episodes": self.ledger.episodes, "model_calls": self.ledger.model_calls,
                "aborted_attempts": self.ledger.aborted_attempts, **content}

    # -- the run ----------------------------------------------------------------
    def run(self) -> dict:
        """Run (or resume) once. The object is single-use: its ledger view is
        as of construction, so a later resume needs a new Exp0001Run."""
        if self._lock is None:
            raise RunError("this run object is closed; construct a new Exp0001Run to resume")
        try:
            return self._run()
        finally:
            self.close()

    def _run(self) -> dict:
        report_path = self.dir / "report.json"
        if self.ledger.state != "running":  # a finished run is never run again
            report = _load(report_path)
            if report is None:
                if self.ledger.state != "incomplete":
                    raise RunError(f"the run is {self.ledger.state} but has no report")
                # stopped by a cap, then a crash before its report was written
                report = freeze_json(report_path, self._report(
                    "incomplete", "unknown (report rebuilt from the ledger)", self.ledger.reason,
                    statistics=None))
            return report
        stage, t_reg = "lifetime", None
        try:
            lifetimes = run_lifetimes(self.executor, self.t_mine, self.g0_config, MINING_SEEDS)
            stage = "t_reg"
            # re-derived every time from the ledger; a frozen t_reg.json with
            # other content fails closed
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
            reason = self.ledger.reason if isinstance(exc, CapExceeded) else f"{stage}: {exc}"
            statistics = None
            if stage == "final" and self._final_cells is not None:
                statistics = analyze_final_evaluation(
                    self._final_input(t_reg, complete=False, reason=reason))
            # the report is frozen before a protocol stop reaches the ledger
            report = freeze_json(report_path, self._report("incomplete", stage, reason,
                                                           statistics=statistics))
            if self.ledger.state == "running":
                self.ledger.abandon(reason)
            return report
        report = freeze_json(report_path, self._report("complete", "done", None, **final))
        self.ledger.complete()  # after the report, so a crash in between re-runs to it
        return report


class _Heartbeat(ModelAdapter):
    """The miner's teacher, stamping the ledger's clock after every call, so a
    long mining stage keeps the session's active time and the mirror current."""

    def __init__(self, adapter: ModelAdapter, beat: Callable[[], None]):
        self._adapter, self._beat = adapter, beat

    def _then_beat(self, call, *args):
        try:
            return call(*args)
        finally:
            self._beat()

    def generate(self, messages, settings):
        return self._then_beat(self._adapter.generate, messages, settings)

    def structured_generate(self, messages, settings, schema):
        return self._then_beat(self._adapter.structured_generate, messages, settings, schema)

    def tool_call(self, messages, settings, tools):
        return self._then_beat(self._adapter.tool_call, messages, settings, tools)

    def estimate_tokens(self, text):
        return self._adapter.estimate_tokens(text)

    def metadata(self):
        return self._adapter.metadata()


def lock_run_dir(run_dir: Path):
    """An exclusive advisory lock on the run directory for one writer; a
    second process (or object) on the same directory fails closed. Close the
    returned handle to release it."""
    import fcntl
    handle = open(run_dir / ".lock", "a+")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        handle.close()
        raise RunError(f"{run_dir} is in use by another run") from exc
    return handle
