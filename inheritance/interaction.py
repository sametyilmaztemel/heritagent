"""Stage 4C — pairwise interaction evaluator (issue #12; ADR-0002
Amendment A1).

Locked M0 contrast: **both candidate traits vs neither trait** on the same
10 frozen interaction tasks. One deterministic execution per task/condition
(20 calls per unordered pair); no single-trait reruns.

delta_pair = mean(success_both - success_neither)
harmful iff delta_pair < -0.05.

Interaction evidence is SHARED between the two candidates of a pair: a
deterministic evidence id/hash over the CANONICAL (gene-id sorted) pair
order, the frozen 10-task allocation, paired outcomes + trajectories,
locked settings, delta_pair and harmful flag. Each candidate's stage-4
child embeds the same canonical compact evidence record (byte-identical on
both sides) plus the evidence hash, so the pair evidence is durably
reconstructable from either child.

The both-trait overlay preserves the exact base config and adds each
candidate's target skill WITH its frozen target regulation (candidates
never become constitutive by dropping an ``express_when`` condition). The
neither-trait arm is the exact base config. Pair members are validated
through the #11 overlay boundary; gene ids are always handled in canonical
sorted order so caller order cannot change evidence or verdicts.

Pair resolution happens ONLY after all pair evidence is computed:
eliminate the lower-ablation-contribution candidate of each harmful pair;
exact contribution tie -> earlier frozen candidate rank; ambiguous rank ->
fail closed. Evaluation order cannot change verdicts.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

from runtime.expression import RuntimeConfig
from runtime.loop import Budgets
from runtime.model_adapters import GenerationSettings

from inheritance.overlay import (
    EvaluationSetupError,
    build_evaluation_configs,
)
from inheritance.runner import (
    EvaluationIntegrityError,
    EvaluationTask,
    GateEpisodeRunner,
    validate_outcome,
)

INTERACTION_TASK_COUNT = 10
HARMFUL_PAIR_DELTA = -0.05


@dataclass(frozen=True)
class InteractionAllocation:
    """The frozen 10-task interaction allocation shared by every evaluated
    pair. Deep-frozen at construction."""

    tasks: tuple[EvaluationTask, ...]

    def __post_init__(self):
        import copy as _copy
        object.__setattr__(self, "tasks", _copy.deepcopy(self.tasks))
        if len(self.tasks) != INTERACTION_TASK_COUNT:
            raise ValueError(
                f"interaction allocation requires exactly "
                f"{INTERACTION_TASK_COUNT} tasks, got {len(self.tasks)}")
        ids = [t.task_id for t in self.tasks]
        if len(set(ids)) != len(ids):
            raise ValueError("interaction task ids must be distinct")

    def allocation_metadata(self) -> dict:
        return {"task_count": len(self.tasks),
                 "task_ids": [t.task_id for t in self.tasks]}


@dataclass(frozen=True)
class InteractionPairEvidence:
    """Immutable shared evidence for one unordered candidate pair."""

    _gene_ids: tuple[str, ...]          # canonical sorted order
    _delta_pair: float
    _harmful: bool
    _episodes: tuple[dict, ...]
    _allocation: dict
    _evidence_sha256: str
    _locked_settings: dict
    _canonical_evidence: dict           # durable compact record, hash domain

    @property
    def gene_ids(self) -> tuple[str, ...]:
        return self._gene_ids

    @property
    def delta_pair(self) -> float:
        return self._delta_pair

    @property
    def harmful(self) -> bool:
        return self._harmful

    @property
    def episodes(self) -> tuple[dict, ...]:
        return copy.deepcopy(self._episodes)

    @property
    def allocation(self) -> dict:
        return copy.deepcopy(self._allocation)

    @property
    def evidence_sha256(self) -> str:
        return self._evidence_sha256

    @property
    def locked_settings(self) -> dict:
        return copy.deepcopy(self._locked_settings)

    @property
    def canonical_evidence(self) -> dict:
        """The durable compact evidence record (the exact hash domain):
        embedded identically in both candidates' stage-4 children."""
        return copy.deepcopy(self._canonical_evidence)


class InteractionEvaluator:
    """Evaluates both-traits-vs-neither for one unordered candidate pair,
    exactly once per pair."""

    def __init__(self, *, registry, base_config: RuntimeConfig,
                 runner: GateEpisodeRunner, budgets: Budgets,
                 target_regulation_by_gene: dict | None = None):
        self._registry = registry
        self._base_config = base_config
        self._runner = runner
        self._budgets = budgets
        # frozen per-candidate target regulations; a regulated candidate
        # keeps its express_when condition in the both-trait overlay
        self._target_regulation_by_gene = dict(target_regulation_by_gene or {})

    def evaluate_pair(self, envelope_a: dict, envelope_b: dict,
                       allocation: InteractionAllocation,
                       locked_settings: GenerationSettings
                       ) -> InteractionPairEvidence:
        """Validate both candidates, build the both-trait overlay, run the
        20 paired episodes (both/neither), compute delta_pair and the
        harmful flag, and return the shared immutable evidence."""
        both_config, neither_config, gene_ids = self._build_pair_configs(
            envelope_a, envelope_b)

        episodes: list[dict] = []
        successes_both = 0
        successes_neither = 0
        for task in allocation.tasks:
            for condition, config in (("both", both_config),
                                       ("neither", neither_config)):
                outcome = self._runner.run_episode(task, config, self._budgets,
                                                    condition, locked_settings)
                validate_outcome(outcome, task.task_id, condition,
                                  settings=locked_settings)
                # contamination: either target gene expressed in the
                # NEITHER arm invalidates the intervention
                leaked = [g for g in gene_ids
                           if g in outcome.expressed_gene_ids]
                if condition == "neither" and leaked:
                    raise EvaluationIntegrityError(
                        f"neither-trait episode on {task.task_id!r} expressed "
                        f"target gene(s) {leaked}: contaminated intervention, "
                        f"no pair evidence")
                if condition == "both":
                    successes_both += int(outcome.success)
                else:
                    successes_neither += int(outcome.success)
                episodes.append({
                    "task_id": task.task_id, "condition": condition,
                    "success": outcome.success,
                    "expressed_gene_ids": list(outcome.expressed_gene_ids),
                    "trajectory_id": outcome.trajectory_id,
                })
        delta_pair = (successes_both - successes_neither) / len(allocation.tasks)
        harmful = delta_pair < HARMFUL_PAIR_DELTA
        allocation_meta = allocation.allocation_metadata()
        from trajectory.storage.canonical import canonical_json, sha256_hex
        locked = {"temperature": locked_settings.temperature,
                   "max_tokens": locked_settings.max_tokens,
                   "seed": locked_settings.seed,
                   "stop": list(locked_settings.stop)}
        # canonical compact evidence record = the exact hash domain; it is
        # embedded (identically) in both candidates' stage-4 children
        canonical_evidence = {
            "kind": "heritagent-interaction-pair",
            "gene_ids": list(gene_ids),
            "allocation": allocation_meta,
            "paired_outcomes": [{"task_id": e["task_id"], "condition": e["condition"],
                                  "success": e["success"],
                                  "trajectory_id": e["trajectory_id"]}
                                 for e in episodes],
            "delta_pair": delta_pair,
            "harmful": harmful,
            "locked_settings": locked,
        }
        evidence_sha256 = sha256_hex(
            canonical_json(canonical_evidence).encode("utf-8"))
        return InteractionPairEvidence(
            _gene_ids=gene_ids, _delta_pair=delta_pair, _harmful=harmful,
            _episodes=tuple(copy.deepcopy(episodes)),
            _allocation=allocation_meta, _evidence_sha256=evidence_sha256,
            _locked_settings=dict(locked),
            _canonical_evidence=copy.deepcopy(canonical_evidence))

    def _build_pair_configs(self, envelope_a: dict, envelope_b: dict
                             ) -> tuple[RuntimeConfig, RuntimeConfig, tuple[str, ...]]:
        """Both-trait overlay = exact base + skill A + skill B (each
        independently validated through the #11 overlay boundary, each with
        its frozen target regulation); control = exact base config with
        neither. Duplicate gene ids across pair members are rejected before
        episodes. Returned gene ids are in CANONICAL (sorted) order and the
        both-arm skills are inserted in that same canonical order."""
        gene_a = envelope_a["candidate"]["gene_id"]
        gene_b = envelope_b["candidate"]["gene_id"]
        if gene_a == gene_b:
            raise EvaluationSetupError(
                f"interaction pair members share gene_id {gene_a!r}")
        configs_a = build_evaluation_configs(
            self._base_config, envelope_a, self._registry,
            target_regulation=self._target_regulation_by_gene.get(gene_a))
        configs_b = build_evaluation_configs(
            self._base_config, envelope_b, self._registry,
            target_regulation=self._target_regulation_by_gene.get(gene_b))
        existing = {s.gene_id for s in self._base_config.skills}
        clash = {gene_a, gene_b} & existing
        if clash:
            raise EvaluationSetupError(
                f"pair genes {sorted(clash)} already exist in the base config")
        gene_ids = tuple(sorted((gene_a, gene_b)))
        configs_by_gene = {gene_a: configs_a, gene_b: configs_b}
        combined_skills = list(self._base_config.skills)
        regulation = dict(self._base_config.regulation)
        for gene in gene_ids:
            target = [s for s in
                       configs_by_gene[gene].with_trait.skills
                       if s.gene_id == gene][0]
            combined_skills.append(target)
            regulation.update(configs_by_gene[gene].with_trait.regulation)
        both = RuntimeConfig(
            genome_id=self._base_config.genome_id,
            policies=self._base_config.policies,
            skills=tuple(combined_skills), regulation=regulation)
        neither = RuntimeConfig(
            genome_id=self._base_config.genome_id,
            policies=self._base_config.policies,
            skills=self._base_config.skills,
            regulation=dict(self._base_config.regulation))
        return both, neither, gene_ids
