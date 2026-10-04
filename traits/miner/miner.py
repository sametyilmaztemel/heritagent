"""Trait Miner v0 (issue #9): SkillRL-adapted differential distillation.

Pipeline (deterministic except the single teacher call per pair):

    verified normalized trajectories
        -> MiningBatch (typed success/failure pairs, frozen family universe)
        -> canonical differential evidence (+ evidence_sha256)
        -> structured_generate (temperature 0.7, explicit seed)
        -> schema-validated proposals (zero proposals is valid)
        -> deterministic identity / batch-wide exact dedupe
        -> batch-wide ranking (estimated_generality desc, discovery-order tie-break)
        -> K cap
        -> frozen candidate set (somatic envelopes, origin="acquired")

Research positioning: this is ADAPTED PRIOR ART (SkillRL, arXiv:2602.08234),
not a novelty claim. Mined skills are UNVALIDATED somatic hypotheses — only
CIG evidence (#11/#12) can later justify germline assimilation. The miner
never alters the frozen set after returning it. No semantic near-duplicate
merging happens in M0 (that would add another uncontrolled model decision).
"""

from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass
from pathlib import Path

from genome.validation.registry import TraitRegistry
from trajectory.storage.canonical import canonical_json, sha256_hex
from runtime.model_adapters import (
    GenerationSettings,
    ModelAdapter,
    ModelMessage,
    ModelResponseError,
)

from traits.miner.evidence import render_pair_evidence
from traits.miner.inputs import MiningBatch
from traits.miner.records import build_mining_record

_PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "differential-skill-v0.txt"
_PROPOSAL_SCHEMA_PATH = Path(__file__).resolve().parents[1] / "schemas" / \
    "mined-skill-proposal-0.1.schema.json"


def prompt_template_text() -> str:
    """Exact committed template (UTF-8 text)."""
    return _PROMPT_PATH.read_text(encoding="utf-8")


def prompt_template_sha256() -> str:
    """SHA-256 over the exact UTF-8 template file bytes (committed artifact)."""
    return sha256_hex(_PROMPT_PATH.read_bytes())


def output_schema() -> dict:
    return json.loads(_PROPOSAL_SCHEMA_PATH.read_text(encoding="utf-8"))


def output_schema_sha256() -> str:
    return sha256_hex(_PROPOSAL_SCHEMA_PATH.read_bytes())


@dataclass(frozen=True)
class MiningSettings:
    """EXP-0001 mining defaults: temperature 0.7, explicit seed, explicit
    max tokens; K = 10 candidate cap."""

    temperature: float = 0.7
    seed: int = 0
    max_tokens: int = 2048
    max_proposals: int = 10  # K (EXP-0001 frozen cap)


@dataclass(frozen=True)
class FrozenCandidateSet:
    """Immutable mining output. `candidates` are validated somatic envelopes
    (origin="acquired", state=candidate) in ranked order; the set is frozen
    (deep-copied on construction) and identified by `candidate_set_sha256`
    over the selected ordered set. `rejected` lists every non-selected
    proposal with a machine-readable reason."""

    candidates: tuple[dict, ...]
    candidate_set_sha256: str
    mining_records: tuple[dict, ...]
    rejected: tuple[dict, ...]

    def __post_init__(self):
        object.__setattr__(self, "candidates", copy.deepcopy(self.candidates))
        object.__setattr__(self, "mining_records", copy.deepcopy(self.mining_records))
        object.__setattr__(self, "rejected", copy.deepcopy(self.rejected))


class TraitMiner:
    """Differential skill distillation over normalized trajectory pairs."""

    def __init__(self, model: ModelAdapter, registry: TraitRegistry, *,
                 settings: MiningSettings, born_generation: int):
        self._model = model
        self._registry = registry
        self._settings = settings
        self._born_generation = born_generation

    # -- teacher interaction ---------------------------------------------------
    def _ask_teacher(self, *, evidence_text: str,
                     family_universe: tuple[str, ...]) -> dict:
        template = prompt_template_text()
        system = (template
                   .replace("<<MAX_PROPOSALS>>", str(self._settings.max_proposals))
                   .replace("<<FAMILIES>>", ", ".join(family_universe))
                   .replace("<<EVIDENCE>>", evidence_text))
        settings = GenerationSettings(temperature=self._settings.temperature,
                                       max_tokens=self._settings.max_tokens,
                                       seed=self._settings.seed)
        result = self._model.structured_generate(
            [ModelMessage(role="system", content=system)],
            settings, output_schema())
        return json.loads(result.text)

    # -- mining pipeline ---------------------------------------------------------
    def mine(self, batch: MiningBatch) -> FrozenCandidateSet:
        """Mine the batch and return the frozen candidate set.

        Phases: per-pair evidence -> teacher call -> schema-validated
        proposals -> applicability validation -> deterministic identity ->
        batch-wide exact dedupe (before the cap) -> batch-wide ranking
        (estimated_generality desc, discovery-order tie-break) -> K cap ->
        freeze. Every non-selected proposal is recorded with a
        machine-readable reason."""
        candidates: list[dict] = []       # discovered entries (pre-cap), in discovery order
        seen_payloads: dict[str, str] = {}  # canonical payload -> gene_id
        rejected: list[dict] = []
        records: list[dict] = []
        global_discovery_order = 0

        for pair in batch.pairs:
            evidence_text, evidence_sha256 = render_pair_evidence(pair)
            record_id = f"MR-{evidence_sha256[:12]}"
            raw_response: dict | None = None
            pair_rejected: list[dict] = []
            pair_accepted: list[dict] = []
            pair_discovery_order: list[str] = []

            try:
                raw_response = self._ask_teacher(
                    evidence_text=evidence_text, family_universe=batch.family_universe)
                proposals = list(raw_response.get("proposals") or [])
            except ModelResponseError as exc:
                # fail closed: the malformed teacher output yields no
                # proposals but remains fully auditable
                pair_rejected.append({"reason": "malformed_output",
                                       "proposal_name": None, "detail": str(exc)})
                proposals = []

            for order, proposal in enumerate(proposals):
                name = proposal.get("name")
                payload = {
                    "name": proposal["name"],
                    "principle": proposal["principle"],
                    "when_to_apply": proposal["when_to_apply"],
                    "procedure": proposal.get("procedure") or [],
                    "applicability": proposal["applicability"],
                }
                payload_json = canonical_json(payload)

                invalid = [f for f in payload["applicability"]["task_families"]
                           if f not in batch.family_universe]
                if invalid:
                    rejected.append({"reason": "invalid_applicability",
                                      "proposal_name": name,
                                      "detail": f"families outside frozen universe: "
                                                 f"{sorted(invalid)}"})
                    pair_rejected.append(rejected[-1])
                    continue

                if payload_json in seen_payloads:
                    rejected.append({"reason": "exact_duplicate",
                                      "proposal_name": name, "detail": None})
                    pair_rejected.append(rejected[-1])
                    continue

                gene_id = self._gene_identity(payload)
                seen_payloads[payload_json] = gene_id
                artifact_payload = dict(payload)
                artifact_uri = self._registry.put("skill", gene_id, 1, artifact_payload)
                envelope = {
                    "candidate": {
                        "gene_id": gene_id,
                        "version": 1,
                        "type": "skill",
                        "origin": "acquired",
                        "artifact": artifact_uri,
                        "provenance": {
                            "born_generation": self._born_generation,
                            "source_trajectory": pair.success.trajectory_id,
                            "notes": record_id,
                        },
                    },
                    "validation": {"state": "candidate"},
                }
                entry = {"envelope": envelope,
                          "estimated_generality": proposal["estimated_generality"],
                          "discovery_order": global_discovery_order,
                          "source_trajectory_id": pair.success.trajectory_id}
                global_discovery_order += 1
                candidates.append(entry)
                pair_accepted.append(entry)
                pair_discovery_order.append(gene_id)

            records.append(build_mining_record(
                mining_record_id=record_id,
                success_trajectory_ids=[pair.success.trajectory_id],
                failure_trajectory_ids=[pair.failure.trajectory_id],
                task_families=[pair.task_family],
                evidence_sha256=evidence_sha256,
                prompt_template_sha256=prompt_template_sha256(),
                output_schema_sha256=output_schema_sha256(),
                mining_seed=self._settings.seed,
                generation_settings={"temperature": self._settings.temperature,
                                      "max_tokens": self._settings.max_tokens,
                                      "seed": self._settings.seed},
                model_metadata=vars(self._model.metadata()),
                raw_structured_response=raw_response,
                accepted_proposals=[{"gene_id": e["envelope"]["candidate"]["gene_id"],
                                      "version": 1,
                                      "discovery_order": e["discovery_order"],
                                      "estimated_generality": e["estimated_generality"],
                                      "source_trajectory_id": e["source_trajectory_id"]}
                                     for e in pair_accepted],
                rejected=pair_rejected,
                discovery_order=pair_discovery_order,
            ))

        # batch-wide ranking (locked rule) then K cap
        ranked = sorted(candidates, key=lambda entry:
                         (-entry["estimated_generality"], entry["discovery_order"]))
        selected: list[dict] = []
        for entry in ranked:
            if len(selected) < self._settings.max_proposals:
                selected.append(entry)
            else:
                rejected.append({"reason": "outside_cap",
                                  "proposal_name": entry["envelope"]["candidate"]["gene_id"],
                                  "detail": None})
        return self._freeze([entry["envelope"] for entry in selected], records, rejected)

    # -- helpers ------------------------------------------------------------------
    @staticmethod
    def _gene_identity(payload: dict) -> str:
        """Deterministic skill gene identity for identical canonical payloads
        (runtime fields + frozen applicability): slugified name + content-
        derived suffix; version is 1 for a newly mined identity."""
        canonical = canonical_json(payload)
        suffix = sha256_hex(canonical.encode("utf-8"))[:8]
        slug = re.sub(r"[^a-z0-9_]+", "_", payload["name"].lower()).strip("_") or "skill"
        return f"{slug}_{suffix}"

    @staticmethod
    def _freeze(envelopes: list[dict], records: list[dict],
                rejected: list[dict]) -> FrozenCandidateSet:
        selected = copy.deepcopy(envelopes)
        digest = sha256_hex(canonical_json(selected).encode("utf-8"))
        return FrozenCandidateSet(candidates=tuple(selected),
                                   candidate_set_sha256=digest,
                                   mining_records=tuple(records),
                                   rejected=tuple(rejected))
