"""Runtime-input projection (ADR-0002, final round R3; ADR-0001 D1).

Evaluation-only metadata — ``applicability.task_families[]`` above all — is
stripped from every runtime model input in every arm (A, B, C). Runtime
consumers only ever see the projected payloads produced here; the assembly
layer takes these outputs as its input.

- ``project_skill``: runtime skill payload — allowlist
  {name, principle, when_to_apply, procedure}.
- ``project_skillrl``: strict SkillRL baseline payload — additionally strips
  ``procedure[]`` (ADR-0001 D1 projection adapter).
- ``project_for_runtime``: whole-genome projection for the expression layer;
  performs a defense-in-depth scan asserting no evaluation-only key survives.
"""

from __future__ import annotations

import json

from genome.validation.errors import GenomeValidationError
from genome.validation.registry import TraitRegistry

RUNTIME_SKILL_FIELDS = ("name", "principle", "when_to_apply", "procedure")
SKILLRL_SKILL_FIELDS = ("name", "principle", "when_to_apply")
REQUIRED_SKILL_FIELDS = ("name", "principle", "when_to_apply")

# Fields that exist for gate sampling / audit / lineage bookkeeping only and
# must never reach a runtime model input.
EVALUATION_ONLY_KEYS = (
    "applicability",
    "task_families",
    "evaluation",
    "evaluation_metadata",
    "fitness",
    "gate_reports",
    "cig_record",
    "provenance",
)


def _required_fields(payload: dict, fields, label: str) -> None:
    missing = [f for f in fields if f not in payload]
    if missing:
        raise GenomeValidationError(f"{label}: missing required field(s) {missing}")


def project_skill(payload: dict) -> dict:
    """Runtime skill payload: allowlist of runtime-approved fields."""
    if not isinstance(payload, dict):
        raise GenomeValidationError("skill payload must be a JSON object")
    _required_fields(payload, REQUIRED_SKILL_FIELDS, "skill payload")
    return {k: payload[k] for k in RUNTIME_SKILL_FIELDS if k in payload}


def project_skillrl(payload: dict) -> dict:
    """Strict SkillRL baseline payload: additionally strips procedure[]."""
    if not isinstance(payload, dict):
        raise GenomeValidationError("skill payload must be a JSON object")
    _required_fields(payload, REQUIRED_SKILL_FIELDS, "skill payload")
    return {k: payload[k] for k in SKILLRL_SKILL_FIELDS}


def _project_policy(payload: dict) -> dict:
    """Policy/regulator configs pass through minus evaluation-only keys."""
    if not isinstance(payload, dict):
        raise GenomeValidationError("policy payload must be a JSON object")
    return {k: v for k, v in payload.items() if k not in EVALUATION_ONLY_KEYS}


def _assert_no_evaluation_only(projection: dict) -> None:
    text = json.dumps(projection)
    leaked = [key for key in EVALUATION_ONLY_KEYS if f'"{key}"' in text]
    if leaked:
        raise GenomeValidationError(f"runtime projection leaked evaluation-only keys: {leaked}")


def project_for_runtime(genome: dict, registry: TraitRegistry) -> dict:
    """Project a genome into runtime-consumable payloads.

    Validates its own input at the boundary — ``load_genome`` runs here
    (structural invariants, typed slots, URI grammar, name/version/kind
    semantics, and with the registry: digest + binding integrity) — so the
    projection never depends on the caller having pre-validated the genome.
    Reads every gene artifact from the registry and strips evaluation-only
    metadata. The output contains genome identity + regulation (runtime
    logic) and projected payloads only.
    """
    from genome.validation.loader import load_genome

    load_genome(genome, registry)
    projection: dict = {
        "genome_id": genome["genome_id"],
        "schema_version": genome["schema_version"],
        "genes": {},
        "regulation": genome.get("regulation", {}),
    }
    genes = genome.get("genes", {})
    for section in ("cognition", "memory", "execution"):
        section_out = {}
        for slot, ref in (genes.get(section) or {}).items():
            payload = json.loads(registry.resolve(ref["artifact"]))
            section_out[slot] = _project_policy(payload)
        if section_out:
            projection["genes"][section] = section_out
    skills_out = []
    for ref in genes.get("skills") or []:
        payload = json.loads(registry.resolve(ref["artifact"]))
        skills_out.append(project_skill(payload))
    if skills_out:
        projection["genes"]["skills"] = skills_out
    _assert_no_evaluation_only(projection)
    return projection
