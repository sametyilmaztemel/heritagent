"""The EXP-0001 run, end to end (issue #13 Phase A; design sections 2-8;
ADR-0006, ADR-0007).

Order: every seed's G0 lifetime, then the one shared T_reg, then per
mining seed the miner (frozen candidate set), the CIG in six rounds
(ADR-0012 D2: R1 stage 1 and stage-2 look 1, R2-R4 looks 2-4, R5 the
out-of-scope check and 4A, R6 the interaction pairs; then the somatic
decisions), and the arm genomes; then the
final evaluation (each distinct genome once, on T_eval, T_reg and the
transfer subset: arm A, and arms B and C per seed, read the same cells
wherever their genomes coincide; ADR-0009 D1, D2) and the final
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
lifetimes of all seeds at once and the final evaluation at once. The gate
runs, at every worker count, as one queue per round (ADR-0012 D2): a dry
pass of the real gate code collects the round's with-trait episodes and
the G0 cells they are compared with, a reserve check makes sure the caps
leave room for them and the final evaluation (D9), and they run in
ascending sha256 of their keys; a second dry pass must find the round
complete. A read pass then runs the gate code against recorded episodes
only: an unrecorded request, or a recorded gate episode it does not
request, fails the run closed, in this session and in every later one."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping
from pathlib import Path

from evaluation.exp0001.allocations import (
    AllocationError,
    evaluation_task,
    interaction_allocation,
    out_of_scope_allocation,
    pool_tasks,
    replay_allocation,
    stage2_allocation,
)
from evaluation.exp0001.arms import arm_genomes
from evaluation.exp0001.episodes import (
    EpisodeExecutor,
    EpisodeSpec,
    protocol_budgets,
    trajectory_id_for,
)
from evaluation.exp0001.gate_runner import (
    GatePass,
    GateQueueError,
    PrefetchStop,
    config_fingerprint,
    key_round,
    queue_order,
)
from evaluation.exp0001.ledger import CapExceeded, RunLedger
from evaluation.exp0001.noise import cell_disagreement
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
    FINAL_EVALUATION_RESERVE,
    GATE_ROUNDS,
    LOCKED_BENEFIT_BANK_SHA256,
    LOCKED_PILOT_SHA256,
    LOCKED_SPLIT_MANIFEST_SHA256,
    MAX_TOKENS_PER_REQUEST,
    MINER_CALLS_PER_PAIR_MAX,
    MINER_MAX_TOKENS,
    MINING_SEEDS,
    MINING_TEMPERATURE,
)
from evaluation.exp0001.provenance import (
    RUN_START_SCHEMA,
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
from genome.validation.errors import GenomeValidationError
from genome.validation.records import CigRecordStore
from genome.validation.registry import TraitRegistry
from inheritance.ablation import LOOKS, AblationEvaluator
from inheritance.gate import (
    SENSITIVITY_TAU_C_GRID,
    SENSITIVITY_TAU_R_GRID,
    THRESHOLD_VERSION,
    CigGateOrchestrator,
    GateCandidateInput,
    ablation_evidence_issues,
    replay_evidence_issues,
)
from inheritance.generalization import paired_outcomes_sha256, resolve_frozen_applicability
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


def gate_summary(gate: Mapping) -> dict:
    """What a seed's gate decided, for the report (ADR-0009 D5): its
    candidates, how many it promoted, how many named every family (whose
    out-of-scope stage is vacuous, ADR-0007 E6, or not run), how stage 2
    stopped (ADR-0012 D4) and how many candidates reached round R5 (D7)."""
    stops: dict[str, int] = {}
    reached = 0
    for child in gate["children"].values():
        if child.get("stage") == "ablation":
            stop = child["measurements"]["stop"]
            label = f"{stop['decision']}:{stop['reason']}@look{stop['look']}"
            stops[label] = stops.get(label, 0) + 1
            reached += child["measurements"]["alpha_met"] is True
    return {"candidates": len(gate["verdicts"]),
            "promoted": sum(v == "promote" for v in gate["verdicts"].values()),
            "all_family_candidates": sum(
                1 for child in gate["children"].values()
                if child.get("stage") == "generalization_out_of_scope"
                and (child["measurements"].get("vacuous") is True
                     or child["measurements"].get("all_family_applicability") is True)),
            "stage2_stops": dict(sorted(stops.items())),
            "reached_round_5": reached}


ADAPTATION_COST_SCOPE = (
    "measured on the transfer subset, the 20-task ablation pool (ADR-0007 E4); since ADR-0012 "
    "the gate plays the benefit bank, so this pool selected no trait; descriptive only "
    "(ADR-0006 D6; ADR-0009 D5)")


def _record_store_issues(aggregates: Mapping, children: Mapping) -> list[str]:
    """Rebuild a ``CigRecordStore`` from frozen records, one bundle per
    aggregate, and return what does not verify (empty when consistent)."""
    store = CigRecordStore()
    declared: set = set()
    try:
        for cig_id, aggregate in sorted(aggregates.items()):
            if aggregate.get("cig_id") != cig_id:
                return [f"aggregate {cig_id!r} is stored as {aggregate.get('cig_id')!r}"]
            refs = aggregate["child_records"]
            declared.update(refs)
            store.add_bundle(aggregate, [children[ref] for ref in refs])
    except GenomeValidationError as exc:
        return list(exc.errors)
    except (KeyError, TypeError) as exc:
        return [f"a declared record is missing or malformed ({exc!r})"]
    undeclared = sorted(set(children) - declared)
    if undeclared:
        return [f"undeclared child records {undeclared}"]
    return store.verify()


def teacher_calls(mining_records) -> int:
    """The teacher calls behind the frozen mining records: one per evidence
    form a pair was sent in (ADR-0009 D8). A record names the last form; the
    forms before it overflowed the context window."""
    return sum(1 + EVIDENCE_FORMS.index(record.get("evidence_form", EVIDENCE_FORMS[0]))
               for record in mining_records)
GATE_SCHEMA = "heritagent/exp0001-gate/0.2"  # 0.2: rounds and cells (ADR-0012 D2)
ARMS_SCHEMA = "heritagent/exp0001-arms/0.1"
REPORT_SCHEMA = "heritagent/exp0001-report/0.1"
SHARED = "once"  # arm A's run label (ADR-0006 D7)


def _round_summary(reader: GatePass) -> dict:
    """Per round of a seed's gate: the with-trait episodes and the cells the
    read pass requested, and how many readers each cell had (ADR-0012 D2:
    reuse within a round is logged)."""
    rounds: dict[str, dict] = {}
    for key in reader.requested:
        gate_round = key_round(key)
        entry = rounds.setdefault(f"R{gate_round}", {"with_trait": 0, "cells": 0,
                                                     "cell_readers": {}})
        entry["cells" if key.split("/")[3] == "cell" else "with_trait"] += 1
    for (gate_round, task_id), readers in sorted(reader.cell_reads.items()):
        rounds[f"R{gate_round}"]["cell_readers"][task_id] = readers
    return dict(sorted(rounds.items()))


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

    def __init__(self, *, run_dir: Path, run_start: dict, split_manifest: dict,
                 pilot_manifest: dict, benefit_bank: dict, model, teacher,
                 env_factory: Callable[[str, str], object], token_counter,
                 ledger_clock: Callable[[], float] = time.time,
                 episode_clock: Callable[[], float] | None = None,
                 sleep: Callable[[float], None] = time.sleep,
                 expected_split_sha256: str = LOCKED_SPLIT_MANIFEST_SHA256,
                 expected_pilot_sha256: str = LOCKED_PILOT_SHA256,
                 expected_bank_sha256: str = LOCKED_BENEFIT_BANK_SHA256,
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
        self._benefit_bank = benefit_bank
        self._lock, self._owns_lock = lock, lock is None
        self.executor = None
        verify_run_start_record(run_start)
        # a record of this code's protocol: another schema or threshold version is another run
        if run_start.get("schema") != RUN_START_SCHEMA \
                or (run_start.get("gate") or {}).get("threshold_version") != THRESHOLD_VERSION \
                or not isinstance(run_start.get("benefit_bank"), Mapping):
            raise ProvenanceError(f"the run-start record is not a {RUN_START_SCHEMA} record of "
                                  f"thresholds {THRESHOLD_VERSION} with a benefit bank "
                                  f"(ADR-0012 D1, D10)")
        # the split, pilot and bank manifests must verify, be the pre-registered
        # ones, and be the ones the record names (ADR-0012 D1)
        verify_pre_run_artifacts(split_manifest, pilot_manifest, benefit_bank,
                                 expected_split_sha256=expected_split_sha256,
                                 expected_pilot_sha256=expected_pilot_sha256,
                                 expected_bank_sha256=expected_bank_sha256)
        if run_start["splits"]["manifest_sha256"] != split_manifest["manifest_sha256"]:
            raise ProvenanceError("the run-start record names another split manifest")
        if run_start["pilot"]["pilot_sha256"] != pilot_manifest["pilot_sha256"]:
            raise ProvenanceError("the run-start record names another pilot manifest")
        if run_start["benefit_bank"]["bank_sha256"] != benefit_bank["bank_sha256"]:
            raise ProvenanceError("the run-start record names another benefit bank")
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
        self.bank = self._benefit_bank
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
        self._gate_summaries: dict[str, dict] = {}  # per seed, as each gate is decided
        self._gates: dict[str, dict] = {}  # per seed, the frozen gate artifact
        self._sensitivity: dict[str, dict] = {}  # per seed, from its frozen gate artifact

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
        # the stored records verify as a record store: every bundle, its
        # artifact and base genome, and both copies of each pair (ADR-0010 B4)
        issues = _record_store_issues(gate["aggregates"], gate["children"])
        if issues:
            raise RunError(f"seed {seed}: gate.json's CIG records do not verify: {issues}")
        # stage 1's expression rule (ADR-0012 D3) and stage 2's decision (D4)
        # recompute from their records, every comparison read its own round's
        # cell (D2), the in-scope child is stage 2's record by reference (D5),
        # and a candidate's out-of-scope, 4A and 4C parts ran iff its stage-2
        # alpha was met (D7)
        try:
            families = {c["candidate"]["gene_id"]: resolve_frozen_applicability(c, self.registry)
                        for c in candidates}
            issues = self._gate_record_issues(seed, gate, families)
        except (KeyError, IndexError, TypeError, AttributeError, ValueError) as exc:
            issues = [f"a record is malformed ({exc!r})"]
        if issues:
            raise RunError(f"seed {seed}: gate.json does not recompute: {issues}")
        # the frozen tau_c/tau_r grid is the pre-planned analysis-only one, per gated
        # candidate. The check is structural: the cells' contents are not re-derived, since
        # no verdict, arm or statistic reads them (the report carries them, secondary)
        sensitivity = gate.get("sensitivity")
        if not isinstance(sensitivity, Mapping) or set(sensitivity) != set(genes):
            raise RunError(f"seed {seed}: gate.json's sensitivity grids are not the frozen "
                           f"candidates'")
        cells = {f"tau_c={tau_c}|tau_r={tau_r}" for tau_c in SENSITIVITY_TAU_C_GRID
                 for tau_r in SENSITIVITY_TAU_R_GRID}
        for gene, grid in sensitivity.items():
            if not isinstance(grid, Mapping) or grid.get("gene_id") != gene \
                    or grid.get("analysis_only") is not True \
                    or type(grid.get("runner_calls")) is not int or grid["runner_calls"] != 0 \
                    or grid.get("tau_c_grid") != list(SENSITIVITY_TAU_C_GRID) \
                    or grid.get("tau_r_grid") != list(SENSITIVITY_TAU_R_GRID) \
                    or not isinstance(grid.get("cells"), Mapping) or set(grid["cells"]) != cells:
                raise RunError(f"seed {seed}: gate.json's sensitivity grid for {gene!r} is not "
                               f"the pre-planned analysis-only grid")

    def _cell_trajectory(self, seed: int, gate_round: int, task_id: str) -> str:
        """The trajectory id of the (seed, round) G0 cell of a game."""
        key = f"gate/{seed}/R{gate_round}/cell/{task_id}/{config_fingerprint(self.g0_config)}"
        return trajectory_id_for(self.run_start["run_id"], key)

    def _gate_record_issues(self, seed: int, gate: Mapping, families: Mapping) -> list[str]:
        """What in a frozen gate artifact does not recompute; ``families`` is
        each candidate's frozen applicability, from its stored artifact."""
        issues: list[str] = []
        by_aggregate: dict[str, dict] = {}
        for child_id, child in gate["children"].items():
            by_aggregate.setdefault(child.get("parent_cig_id"), {})[child.get("stage")] = child
        genes = {aggregate_id: gene for gene, aggregate_id in gate["aggregate_ids"].items()}
        for aggregate_id, stages in sorted(by_aggregate.items()):
            replay = stages.get("replay")
            found = (["has no stage-1 record"] if replay is None else
                     replay_evidence_issues(replay["measurements"], genes.get(aggregate_id)))
            issues.extend(f"{aggregate_id}: {issue}" for issue in found)
            ablation = stages.get("ablation")
            if ablation is None:
                issues.append(f"{aggregate_id} has no stage-2 record")
                continue
            m = ablation["measurements"]
            found = ablation_evidence_issues(m, mining_seed=seed,
                                             families=families.get(genes.get(aggregate_id), ()))
            if m["allocation"].get("bank_sha256") != self.bank["bank_sha256"]:
                found.append("stage 2 was not drawn from this run's benefit bank (ADR-0012 D1)")
            if found:
                issues.extend(f"{aggregate_id}: {issue}" for issue in found)
                continue
            for outcome in m["paired_outcomes"]:
                if outcome["cell"]["trajectory_id"] != \
                        self._cell_trajectory(seed, outcome["round"], outcome["task_id"]):
                    issues.append(f"{aggregate_id}: stage 2's game {outcome['task_id']!r} did not "
                                  f"read its round-R{outcome['round']} cell (ADR-0012 D2)")
                    break
            in_scope = stages.get("generalization_in_scope", {}).get("measurements", {})
            reference = in_scope.get("by_reference") or {}
            if reference.get("child_id") != ablation["cig_id"] or \
                    reference.get("paired_outcomes_sha256") != \
                    paired_outcomes_sha256(m["paired_outcomes"]):
                issues.append(f"{aggregate_id}: the in-scope child is not stage 2's record by "
                              f"reference (ADR-0012 D5)")
            out = stages.get("generalization_out_of_scope", {}).get("measurements", {})
            stage4 = stages.get("interaction_regression", {}).get("measurements", {})
            ran = m["alpha_met"] is True
            if (out.get("not_run") is True) == ran or (stage4.get("not_run") is True) == ran:
                issues.append(f"{aggregate_id}: its out-of-scope check and 4A must run iff stage "
                              f"2 met alpha ({ran}) (ADR-0012 D7)")
            if (out.get("allocation") or {}).get("bank_sha256") != self.bank["bank_sha256"]:
                issues.append(f"{aggregate_id}: the out-of-scope games were not drawn from this "
                              f"run's benefit bank (ADR-0012 D1)")
            for outcome in out.get("paired_outcomes", ()):
                if outcome["cell"]["trajectory_id"] != \
                        self._cell_trajectory(seed, 5, outcome["task_id"]):
                    issues.append(f"{aggregate_id}: out-of-scope game {outcome['task_id']!r} did "
                                  f"not read its R5 cell (ADR-0012 D2)")
                    break
            baseline = (stage4.get("regression") or {}).get("paired_g0_run") or {}
            for episode in baseline.get("episodes", ()):
                if episode["trajectory_id"] != self._cell_trajectory(seed, 5, episode["task_id"]):
                    issues.append(f"{aggregate_id}: T_reg task {episode['task_id']!r} was not "
                                  f"paired with its R5 cell (ADR-0012 D2)")
                    break
            for pair in (stage4.get("interaction") or {}).get("pair_evidence", ()):
                for outcome in pair["canonical_evidence"]["paired_outcomes"]:
                    if outcome["condition"] == "neither" and outcome["trajectory_id"] != \
                            self._cell_trajectory(seed, 6, outcome["task_id"]):
                        issues.append(f"{aggregate_id}: a pair's neither side on "
                                      f"{outcome['task_id']!r} is not its R6 cell (ADR-0012 D2)")
                        break
        return issues

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
            plans.append((rank, envelope, replay_allocation(source_task),
                          stage2_allocation(self.bank, envelope, self.registry),
                          out_of_scope_allocation(self.bank, envelope, self.registry)))
        interaction = interaction_allocation(self.splits)
        self._somatic(seed, candidates).close()  # the persistent journal holds the candidates

        def gate_code(gate_pass: GatePass):
            """The whole gate of the seed, round by round (ADR-0012 D2)."""
            replays, stage1, stage2 = {}, {}, {}
            # every evaluator is built, and fails closed, before the first request
            for rank, envelope, replay_plan, stage2_plan, _ in plans:
                gene = envelope["candidate"]["gene_id"]
                common = dict(candidate_envelope=envelope, registry=self.registry,
                              base_config=self.g0_config, budgets=self.budgets,
                              parent_cig_id=cig_id(seed, rank), mining_seed=seed)
                replays[gene] = (ReplayEvaluator(runner=gate_pass.runner(1, f"{gene}/replay"),
                                                 **common), replay_plan)
                stage2[gene] = AblationEvaluator(allocation=stage2_plan, **common)
            # R1: stage 1 (ADR-0012 D3)
            for gene, (evaluator, replay_plan) in replays.items():
                stage1[gene] = evaluator.evaluate(replay_plan)
            # R1-R4: stage 2's looks for every candidate still running (D4)
            for look in range(1, len(LOOKS) + 1):
                for gene, evaluator in stage2.items():
                    if evaluator.next_look == look:
                        evaluator.play_look(look, gate_pass.runner(look, f"{gene}/ablation"))
            inputs = []
            for rank, envelope, _, _, out_plan in plans:
                gene = envelope["candidate"]["gene_id"]
                inputs.append(GateCandidateInput(
                    candidate_envelope=envelope, frozen_rank=rank,
                    replay_evaluation=stage1[gene], ablation_evaluation=stage2[gene].evaluation(),
                    generalization_allocation=out_plan, regression_allocation=t_reg,
                    parent_cig_id=cig_id(seed, rank), mining_seed=seed))
            # R5-R6: the gate commits into an in-memory mirror of the journal; its
            # artifact is frozen before the persistent journal takes the decisions
            mirror = SomaticStore.create(None, self.registry)
            for envelope in candidates:
                mirror.add_candidate(envelope)
            records = CigRecordStore()
            gate = CigGateOrchestrator(
                registry=self.registry, base_config=self.g0_config,
                runner=lambda gate_round: gate_pass.runner(
                    gate_round, "stage3-4" if gate_round == 5 else "pairs"),
                budgets=self.budgets, parent_cig_id=cig_id(seed, None), mining_seed=seed,
                token_counter=self.token_counter, interaction_allocation=interaction,
                somatic_store=mirror, cig_record_store=records)
            return gate.run(inputs), records, gate

        def gate_pass(**options) -> GatePass:
            return GatePass(self.executor, seed=seed, g0_config=self.g0_config,
                            budgets=self.budgets, **options)

        def dry(gate_round: int) -> list[EpisodeSpec]:
            collector = gate_pass(mode="dry", collect_round=gate_round)
            try:
                gate_code(collector)
            except PrefetchStop:
                pass  # the round's last request has been reached
            except Exception:  # noqa: BLE001
                if not collector.collected:
                    raise  # no placeholder was answered, so the error is real: fail here
                # a placeholder can lead the dry pass astray only after the
                # round's requests (rounds are sequential); the completeness
                # pass below, which answers none, meets any real error
            return collector.collected

        for gate_round in range(1, GATE_ROUNDS + 1):
            queue = dry(gate_round)
            if not queue:
                continue
            # ADR-0012 D9: room for the round and the final evaluation, or stop
            self.ledger.check_reserve(f"gate/{seed}/R{gate_round}", round_episodes=len(queue),
                                      final_reserve=FINAL_EVALUATION_RESERVE)
            self.executor.run_many(sorted(queue, key=queue_order))
            left = dry(gate_round)
            if left:
                raise RunError(f"seed {seed}: round R{gate_round} needed {len(left)} more "
                               f"episodes after its queue ran (first {left[0].key!r}); a round "
                               f"must be one queue (ADR-0012 D2)")
        reader = gate_pass(mode="read")
        try:
            outcome, records, gate = gate_code(reader)
        except GateQueueError as exc:
            raise RunError(f"seed {seed}: {exc}") from exc
        # every recorded gate episode of this seed must be one the gate asked
        # for; checked against the ledger, so a resumed run fails closed too
        unrequested = sorted(set(self.ledger.done_keys(f"gate/{seed}/")) - set(reader.requested))
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
            "aggregates": records.aggregates, "children": records.children,
            # what each round ran and which cells its readers shared (ADR-0012 D2); the
            # reserve checks (D9) stay in the ledger, which also holds a resumed session's
            "rounds": _round_summary(reader),
            # the pre-planned tau_c/tau_r grid (manifest sensitivity_secondary): a pure
            # recomputation over the stage evidence above, frozen with it (ADR-0012 D7)
            "sensitivity": {r.gene_id: gate.sensitivity_recomputation(r.gene_id)
                            for r in outcome.results}})

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
        # ADR-0009 D2: each distinct genome plays each task once; every (arm,
        # seed) whose genome it is reads the same cells
        sharing: dict[str, list[str]] = {}
        cells_by_genome: dict[str, dict] = {}
        distinct = []
        for (arm, seed), genome in genomes.items():
            genome_id = genome["genome_id"]
            if genome_id not in cells_by_genome:
                cells_by_genome[genome_id] = {"t_eval": {}, "t_reg": {}, "transfer": {}}
                distinct.append(genome)
            sharing.setdefault(genome_id, []).append(
                f"{arm}/{SHARED if seed is None else seed}")
        self._final_cells = {key: cells_by_genome[genome["genome_id"]]
                             for key, genome in genomes.items()}
        settings = evaluation_settings(max_tokens=MAX_TOKENS_PER_REQUEST, seed=None)
        planned = []  # (cells, task_id, spec) in the sequential run's order
        for genome in distinct:
            genome_id = genome["genome_id"]
            config = compile_runtime_config(genome, self.registry)
            for task_set, tasks in (("t_eval", t_eval), ("t_reg", t_reg_tasks),
                                    ("transfer", transfer)):
                for task in tasks:
                    planned.append((cells_by_genome[genome_id][task_set], task.task_id,
                                    EpisodeSpec(
                        key=f"final/{genome_id}/{task_set}/{task.task_id}",
                        task_id=task.task_id, family=task.family, split=task.payload["split"],
                        config=config, settings=settings,
                        tags={"stage": "final", "genome_id": genome_id,
                              "arms": " ".join(sharing[genome_id]), "task_set": task_set})))
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
            # secondary and analysis-only: no verdict, arm or statistic uses it
            "sensitivity": dict(self._sensitivity),
            # which arms read which genome's cells (ADR-0009 D2)
            "final_genomes": [{"genome_id": genome["genome_id"],
                               "arms": sharing[genome["genome_id"]]} for genome in distinct],
            "genome_sizes": summarize_genome_sizes({
                key: measure_genome_size(genome, self.registry, self.token_counter)
                for key, genome in genomes.items()}),
            "adaptation_cost": {"scope": ADAPTATION_COST_SCOPE, **summarize_adaptation_cost(
                [t.task_id for t in transfer],
                {key: {task_id: TransferOutcome(success=r.success,
                                                total_tokens=r.prompt_tokens + r.completion_tokens,
                                                steps=r.steps)
                       for task_id, r in cells["transfer"].items()}
                 for key, cells in self._final_cells.items()})},
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
                "aborted_attempts": self.ledger.aborted_attempts,
                "gates": dict(self._gate_summaries),
                # descriptive backend-noise references from the G0 cells (ADR-0012 D8)
                "cell_disagreement": cell_disagreement(self._gates) if self._gates else None,
                # restores and repairs: work after a restored snapshot ran twice, and
                # its calls and time are not counted (ADR-0008 D5)
                "ledger_notes": [dict(note) for note in self.ledger.notes], **content}

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
                self._gate_summaries[str(seed)] = gate_summary(gate)
                self._gates[str(seed)] = gate
                self._sensitivity[str(seed)] = gate["sensitivity"]
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
