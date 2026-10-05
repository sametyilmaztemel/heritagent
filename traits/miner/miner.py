"""Trait Miner v0 (issue #9): SkillRL-adapted differential distillation.

Two-phase pipeline (deterministic except the teacher calls):

    Phase 1 (per pair): evidence -> structured teacher call -> validated
    proposal DESCRIPTORS (schema-checked, applicability-checked, exact-
    deduped) — nothing is bound to the registry yet.

    Phase 2 (batch-wide): ranking (estimated_generality desc,
    discovery-order tie-break) -> K cap -> FAIL-ATOMIC persistence:
    all selected envelopes are prevalidated through the existing somatic
    boundary BEFORE any registry mutation, then registered (prospective
    URI == returned URI verified) and post-write re-verified. Mining
    records are reconciled so every proposal is exactly one of accepted
    or rejected (with `outside_cap` recorded in its SOURCE teacher call's
    record).

Research positioning: this is ADAPTED PRIOR ART (SkillRL, arXiv:2602.08234),
not a novelty claim. Mined skills are UNVALIDATED somatic hypotheses — only
CIG evidence (#11/#12) can later justify germline assimilation. The miner
never alters the frozen set after returning it. No semantic near-duplicate
merging happens in M0 (that would add another uncontrolled model decision).

Canonical candidate semantics (locked): `applicability.task_families[]` is
sorted before payload hashing, so family order never affects gene identity
or dedupe. **First valid discovery owns `estimated_generality` and
`discovery_order`; later exact duplicates are rejected and cannot modify
its score.**
"""

from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass
from pathlib import Path

from genome.validation.loader import load_somatic
from trajectory.recorder.errors import RecorderError
from genome.validation.registry import TraitRegistry
from runtime.model_adapters import (
    GenerationSettings,
    ModelAdapter,
    ModelMessage,
    ModelResponseError,
)

from trajectory.storage.canonical import canonical_json, sha256_hex
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
    """Immutable mining output. Internal storage is deep-copied; the public
    accessors return deep copies, so callers can never mutate the frozen
    candidate set, the mining records, the rejection list, or the digest
    relationship. `candidates` are validated somatic envelopes
    (origin="acquired", state=candidate) in ranked order."""

    _candidates: tuple[dict, ...]
    _candidate_set_sha256: str
    _mining_records: tuple[dict, ...]
    _rejected: tuple[dict, ...]

    def __post_init__(self):
        for name in ("_candidates", "_mining_records", "_rejected"):
            object.__setattr__(self, name, copy.deepcopy(getattr(self, name)))

    @property
    def candidates(self) -> tuple[dict, ...]:
        return copy.deepcopy(self._candidates)

    @property
    def candidate_set_sha256(self) -> str:
        return self._candidate_set_sha256

    @property
    def mining_records(self) -> tuple[dict, ...]:
        return copy.deepcopy(self._mining_records)

    @property
    def rejected(self) -> tuple[dict, ...]:
        return copy.deepcopy(self._rejected)


class TraitMiner:
    """Differential skill distillation over normalized trajectory pairs."""

    def __init__(self, model: ModelAdapter, registry: TraitRegistry, *,
                 settings: MiningSettings, born_generation: int):
        if born_generation < 0:
            raise ValueError("born_generation must be >= 0 (early defense: the "
                             "somatic schema rejects negative generations)")
        self._model = model
        self._registry = registry
        self._settings = settings
        self._born_generation = born_generation

    # -- teacher interaction ---------------------------------------------------
    def _record_identity(self, evidence_sha256: str, rendered_prompt_sha256: str,
                         source_trajectory_ids: tuple[str, str], model_metadata) -> str:
        """Audit identity for ONE teacher call: derived from evidence, the
        exact rendered prompt, prompt/output-schema hashes, mining
        seed/settings, model identity AND the source trajectory ids — NOT
        from the teacher output (malformed calls are audited too)."""
        identity_input = {
            "evidence_sha256": evidence_sha256,
            "rendered_prompt_sha256": rendered_prompt_sha256,
            "prompt_template_sha256": prompt_template_sha256(),
            "output_schema_sha256": output_schema_sha256(),
            "mining_seed": self._settings.seed,
            "generation_settings": {"temperature": self._settings.temperature,
                                     "max_tokens": self._settings.max_tokens},
            "model": {"model_id": model_metadata.model_id,
                       "revision": model_metadata.revision,
                       "backend": model_metadata.backend},
            "source_trajectory_ids": {"success": source_trajectory_ids[0],
                                       "failure": source_trajectory_ids[1]},
        }
        digest = sha256_hex(canonical_json(identity_input).encode("utf-8"))
        return f"MR-{digest[:12]}"

    def _render_prompt(self, *, evidence_text: str,
                       family_universe: tuple[str, ...]) -> tuple[str, str]:
        """Render the final teacher message; returns (rendered_prompt,
        rendered_prompt_sha256) — available even before the model call so
        malformed calls are audited with the exact prompt identity."""
        template = prompt_template_text()
        rendered_prompt = (template
                            .replace("<<MAX_PROPOSALS>>", str(self._settings.max_proposals))
                            .replace("<<FAMILIES>>", ", ".join(family_universe))
                            .replace("<<EVIDENCE>>", evidence_text))
        return rendered_prompt, sha256_hex(rendered_prompt.encode("utf-8"))

    def _ask_teacher(self, rendered_prompt: str) -> tuple[dict, str]:
        """Call the teacher on the rendered prompt; returns
        (parsed_response, raw_text). Raises ModelResponseError (with
        raw_text) on malformed output."""
        settings = GenerationSettings(temperature=self._settings.temperature,
                                       max_tokens=self._settings.max_tokens,
                                       seed=self._settings.seed)
        result = self._model.structured_generate(
            [ModelMessage(role="system", content=rendered_prompt)],
            settings, output_schema())
        return json.loads(result.text), result.text

    # -- mining pipeline ---------------------------------------------------------
    def mine(self, batch: MiningBatch) -> FrozenCandidateSet:
        """Mine the batch and return the frozen candidate set.

        Phases: per-pair evidence -> teacher call -> schema-validated
        proposals -> applicability validation -> deterministic identity ->
        batch-wide exact dedupe (before the cap) -> batch-wide ranking
        (estimated_generality desc, discovery-order tie-break) -> K cap ->
        FAIL-ATOMIC persistence (prevalidate all selected envelopes, then
        registry binding + post-write verification) -> mining-record
        reconciliation -> freeze."""
        descriptors: list[dict] = []       # validated proposals, pre-cap
        seen_payloads: dict[str, str] = {}  # canonical payload -> gene_id
        rejected: list[dict] = []
        record_drafts: list[dict] = []
        global_discovery_order = 0
        model_metadata = self._model.metadata()

        # -- phase 1: per-pair teacher calls -> validated descriptors ---------
        for pair in batch.pairs:
            evidence_text, evidence_sha256 = render_pair_evidence(pair)
            source_ids = (pair.success.trajectory_id, pair.failure.trajectory_id)
            raw_response: dict | None = None
            raw_teacher_text: str | None = None
            rendered_prompt, rendered_prompt_sha256 = self._render_prompt(
                evidence_text=evidence_text, family_universe=batch.family_universe)
            pair_rejected: list[dict] = []
            pair_descriptors: list[dict] = []
            pair_discovery_order: list[str] = []
            # call identity is fully determined before the call: malformed
            # calls are audited under the same identity
            record_id = self._record_identity(evidence_sha256,
                                               rendered_prompt_sha256,
                                               source_ids, model_metadata)

            try:
                raw_response, raw_teacher_text = self._ask_teacher(rendered_prompt)
            except ModelResponseError as exc:
                # fail closed: the malformed teacher output yields no
                # proposals but the exact raw output stays auditable
                raw_teacher_text = exc.raw_text
                pair_rejected.append({"reason": "malformed_output",
                                       "proposal_name": None, "proposal_index": None,
                                       "gene_id": None, "discovery_order": None,
                                       "detail": str(exc)})
                proposals = []
            else:
                proposals = list(raw_response.get("proposals") or [])

            for order, proposal in enumerate(proposals):
                name = proposal.get("name")
                # canonical candidate semantics: applicability families are a
                # SET — sorted before hashing so family order never affects
                # identity or dedupe
                payload = {
                    "name": proposal["name"],
                    "principle": proposal["principle"],
                    "when_to_apply": proposal["when_to_apply"],
                    "procedure": proposal.get("procedure") or [],
                    "applicability": {"task_families":
                                       sorted(proposal["applicability"]["task_families"])},
                }
                payload_json = canonical_json(payload)

                invalid = [f for f in payload["applicability"]["task_families"]
                           if f not in batch.family_universe]
                if invalid:
                    entry = {"reason": "invalid_applicability", "proposal_name": name,
                              "proposal_index": order,
                              "gene_id": self._gene_identity(payload),
                              "discovery_order": None,
                              "detail": f"families outside frozen universe: {sorted(invalid)}"}
                    rejected.append(entry)
                    pair_rejected.append(entry)
                    continue

                if payload_json in seen_payloads:
                    # duplicate policy (locked): first valid discovery owns
                    # estimated_generality and discovery_order; later exact
                    # duplicates are rejected and cannot modify its score
                    entry = {"reason": "exact_duplicate", "proposal_name": name,
                              "proposal_index": order,
                              "gene_id": seen_payloads[payload_json],
                              "discovery_order": None, "detail": None}
                    rejected.append(entry)
                    pair_rejected.append(entry)
                    continue

                gene_id = self._gene_identity(payload)
                seen_payloads[payload_json] = gene_id
                descriptor = {"payload": payload, "payload_json": payload_json,
                               "gene_id": gene_id,
                               "estimated_generality": proposal["estimated_generality"],
                               "discovery_order": global_discovery_order,
                               "proposal_index": order,
                               "record_index": len(record_drafts),
                               "record_id": record_id,
                               "source_trajectory_id": pair.success.trajectory_id,
                               "proposal_name": name}
                global_discovery_order += 1
                descriptors.append(descriptor)
                pair_descriptors.append(descriptor)
                pair_discovery_order.append(gene_id)

            record_drafts.append({
                "record_id": record_id, "pair": pair,
                "evidence_sha256": evidence_sha256,
                "rendered_prompt_sha256": rendered_prompt_sha256,
                "raw_response": raw_response, "raw_teacher_text": raw_teacher_text,
                "pair_rejected": pair_rejected,
                "pair_descriptors": pair_descriptors,
                "pair_discovery_order": pair_discovery_order,
                "outside_cap": [],
            })

        # -- phase 2: batch-wide ranking, cap, FAIL-ATOMIC persistence --------
        ranked = sorted(descriptors, key=lambda d:
                         (-d["estimated_generality"], d["discovery_order"]))
        selected: list[dict] = []
        for descriptor in ranked:
            if len(selected) < self._settings.max_proposals:
                selected.append(descriptor)
            else:
                entry = {"reason": "outside_cap",
                          "proposal_name": descriptor["proposal_name"],
                          "proposal_index": descriptor["proposal_index"],
                          "gene_id": descriptor["gene_id"],
                          "discovery_order": descriptor["discovery_order"],
                          "detail": None}
                rejected.append(entry)
                record_drafts[descriptor["record_index"]]["outside_cap"].append(entry)

        # FAIL-ATOMIC persistence: prevalidate ALL selected envelopes before
        # mutating the registry
        prospective: list[tuple[dict, dict, str, str]] = []  # (envelope, payload, uri, gene_id)
        prevalidation_issues: list[str] = []
        for descriptor in selected:
            payload_bytes = canonical_json(descriptor["payload"]).encode("utf-8")
            digest = sha256_hex(payload_bytes)
            artifact_uri = f"registry://skills/{descriptor['gene_id']}@1/sha256:{digest}"
            envelope = {
                "candidate": {
                    "gene_id": descriptor["gene_id"],
                    "version": 1,
                    "type": "skill",
                    "origin": "acquired",
                    "artifact": artifact_uri,
                    "provenance": {
                        "born_generation": self._born_generation,
                        "source_trajectory": descriptor["source_trajectory_id"],
                        "notes": descriptor["record_id"],
                    },
                },
                "validation": {"state": "candidate"},
            }
            try:
                # structural + semantic prevalidation without a registry
                load_somatic(envelope, registry=None)
            except Exception as exc:  # any prevalidation failure aborts atomically
                prevalidation_issues.append(f"{descriptor['gene_id']}: {exc}")
            prospective.append((envelope, descriptor["payload"], artifact_uri,
                                  descriptor["gene_id"]))
        if prevalidation_issues:
            # ZERO registry bindings/artifacts: nothing was written yet
            raise RecorderError("selected-candidate prevalidation failed; "
                                 "registry left untouched: "
                                 f"{sorted(prevalidation_issues)}")

        # bind + post-write integrity verification; the returned URI must
        # equal the prospective URI (canonical hashing is deterministic)
        candidate_envelopes: list[dict] = []
        for envelope, payload, artifact_uri, gene_id in prospective:
            returned_uri = self._registry.put("skill", gene_id, 1, payload)
            if returned_uri != artifact_uri:
                raise RecorderError(
                    f"registry returned URI {returned_uri!r} != prospective "
                    f"{artifact_uri!r} for {gene_id!r}")
            load_somatic(envelope, self._registry)  # post-write integrity check
            candidate_envelopes.append(envelope)

        # -- reconcile mining records with the FINAL selection -----------------
        records: list[dict] = []
        for index, draft in enumerate(record_drafts):
            accepted = [{"gene_id": d["gene_id"], "version": 1,
                          "discovery_order": d["discovery_order"],
                          "estimated_generality": d["estimated_generality"],
                          "source_trajectory_id": d["source_trajectory_id"]}
                         for d in selected if d["record_index"] == index]
            rec_rejected = list(draft["pair_rejected"]) + list(draft["outside_cap"])
            records.append(build_mining_record(
                mining_record_id=draft["record_id"],
                success_trajectory_ids=[draft["pair"].success.trajectory_id],
                failure_trajectory_ids=[draft["pair"].failure.trajectory_id],
                task_families=[draft["pair"].task_family],
                evidence_sha256=draft["evidence_sha256"],
                rendered_prompt_sha256=draft["rendered_prompt_sha256"],
                prompt_template_sha256=prompt_template_sha256(),
                output_schema_sha256=output_schema_sha256(),
                mining_seed=self._settings.seed,
                generation_settings={"temperature": self._settings.temperature,
                                      "max_tokens": self._settings.max_tokens,
                                      "seed": self._settings.seed},
                model_metadata={"model_id": model_metadata.model_id,
                                 "revision": model_metadata.revision,
                                 "backend": model_metadata.backend,
                                 "backend_version": model_metadata.backend_version,
                                 "capabilities": copy.deepcopy(
                                     model_metadata.capabilities)},
                family_universe=list(batch.family_universe),
                raw_structured_response=draft["raw_response"],
                raw_teacher_text=draft["raw_teacher_text"],
                accepted_proposals=accepted,
                rejected=rec_rejected,
                discovery_order=draft["pair_discovery_order"],
            ))

        return self._freeze(candidate_envelopes, records, rejected)

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
        return FrozenCandidateSet(_candidates=tuple(selected),
                                   _candidate_set_sha256=digest,
                                   _mining_records=tuple(copy.deepcopy(records)),
                                   _rejected=tuple(copy.deepcopy(rejected)))
