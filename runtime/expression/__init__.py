"""EAG → runtime configuration compilation (issue #7 criterion 4, 9).

The compilation path REUSES the merged #4 validation/projection layer
(`genome.validation.load_genome` + `project_for_runtime`) — it does not
reimplement a second EAG parser. Evaluation-only metadata therefore cannot
reach runtime config by construction (the projection's leak-guard runs
inside `project_for_runtime`). On top of the projection, this module
validates the payload shapes the G0 runtime actually consumes (fail closed)
and builds the typed RuntimeConfig.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from genome.validation.errors import GenomeValidationError
from genome.validation.registry import TraitRegistry
from genome.validation.projection import project_for_runtime
from jsonschema import Draft202012Validator

from runtime.regulation import Condition

_SCHEMA_DIR = Path(__file__).resolve().parent / "payload_schemas"

_SKILL_SCHEMA_CACHE: dict | None = None


def _load_schema(name: str) -> dict:
    return json.loads((_SCHEMA_DIR / name).read_text(encoding="utf-8"))


def _skill_schema() -> dict:
    global _SKILL_SCHEMA_CACHE
    if _SKILL_SCHEMA_CACHE is None:
        _SKILL_SCHEMA_CACHE = _load_schema("runtime_skill.schema.json")
    return _SKILL_SCHEMA_CACHE


def validate_runtime_skill_payload(payload: dict) -> dict:
    """Public runtime-skill validation boundary shared by
    compile_runtime_config and the CIG evaluation overlay: JSON Schema +
    procedure shape + unique stable step IDs. Returns the payload unchanged
    or raises GenomeValidationError (fail closed)."""
    from jsonschema import Draft202012Validator

    errors = [e.message for e in Draft202012Validator(_skill_schema()).iter_errors(payload)]
    if errors:
        raise GenomeValidationError(
            [f"runtime skill payload: {m}" for m in sorted(errors)])
    ids = [step["id"] for step in payload.get("procedure") or []]
    if len(ids) != len(set(ids)):
        raise GenomeValidationError(
            "runtime skill payload: procedure step ids must be unique")
    return payload


_PLANNER_SCHEMA = _load_schema("planner_policy.schema.json")
_RETRY_SCHEMA = _load_schema("retry_policy.schema.json")
_SKILL_SCHEMA = _load_schema("runtime_skill.schema.json")

# G0-required policy slots with strict payload contracts (criterion 9).
_TYPED_POLICY_SCHEMAS: dict[str, dict] = {
    "cognition.planner": _PLANNER_SCHEMA,
    "execution.retry_policy": _RETRY_SCHEMA,
}


@dataclass(frozen=True)
class RuntimePolicy:
    """A compiled policy gene consumed by a named runtime component."""

    slot: str            # e.g. "cognition.planner"
    gene_id: str
    config: dict         # validated, evaluation-free payload


@dataclass(frozen=True)
class RuntimeSkill:
    """A runtime-projected skill with stable ids for future attribution."""

    gene_id: str
    name: str
    principle: str
    when_to_apply: str
    procedure: tuple[dict, ...] = ()   # [{id, text}] — stable step ids
    condition: Condition | None = None  # None = constitutively expressed


@dataclass(frozen=True)
class RuntimeConfig:
    """Immutable runtime configuration compiled from a validated EAG."""

    genome_id: str
    policies: dict[str, RuntimePolicy]
    skills: tuple[RuntimeSkill, ...]
    regulation: dict[str, Condition] = field(default_factory=dict)

    def is_active(self, gene_id: str, counters: dict) -> bool:
        """Constitutive genes are always active; regulated genes follow
        their condition deterministically for a given counter state."""
        condition = self.regulation.get(gene_id)
        return True if condition is None else condition.evaluate(counters)


def _validate_payload(payload: dict, schema: dict, label: str) -> dict:
    errors = [e.message for e in Draft202012Validator(schema).iter_errors(payload)]
    if errors:
        raise GenomeValidationError([f"{label}: {m}" for m in sorted(errors)])
    return payload


def _validate_procedure_ids(skill_payload: dict, gene_id: str) -> None:
    steps = skill_payload.get("procedure") or []
    ids = [step["id"] for step in steps]
    if len(ids) != len(set(ids)):
        raise GenomeValidationError(f"skill {gene_id!r}: procedure step ids must be unique")
    for step in steps:
        if not str(step["id"]).strip():
            raise GenomeValidationError(f"skill {gene_id!r}: procedure step id must be non-empty")


def compile_runtime_config(genome: dict, registry: TraitRegistry) -> RuntimeConfig:
    """Compile a validated EAG genome into a RuntimeConfig (fail closed)."""
    projection = project_for_runtime(genome, registry)  # #4 path: validation + leak-guard
    genes = genome.get("genes", {})

    policies: dict[str, RuntimePolicy] = {}
    for section in ("cognition", "memory", "execution"):
        for slot, ref in (genes.get(section) or {}).items():
            path = f"{section}.{slot}"
            payload = projection["genes"].get(section, {}).get(slot)
            if payload is None:
                raise GenomeValidationError(f"{path}: projection produced no payload")
            if path in _TYPED_POLICY_SCHEMAS:
                _validate_payload(payload, _TYPED_POLICY_SCHEMAS[path], path)
            policies[path] = RuntimePolicy(slot=path, gene_id=ref["gene_id"], config=payload)

    regulation = {gene_id: Condition.from_genome(rule)
                  for gene_id, rule in (genome.get("regulation") or {}).items()}

    skills: list[RuntimeSkill] = []
    skill_refs = list(genes.get("skills") or [])
    projected_skills = list(projection["genes"].get("skills") or [])
    if len(skill_refs) != len(projected_skills):
        raise GenomeValidationError("projection returned a different number of skills than the genome holds")
    for ref, payload in zip(skill_refs, projected_skills):
        validate_runtime_skill_payload(payload)  # shared boundary (overlay parity)
        skills.append(RuntimeSkill(
            gene_id=ref["gene_id"],
            name=payload["name"],
            principle=payload["principle"],
            when_to_apply=payload["when_to_apply"],
            procedure=tuple(payload.get("procedure") or ()),
            condition=regulation.get(ref["gene_id"]),
        ))
    if len({s.gene_id for s in skills}) != len(skills):
        raise GenomeValidationError("runtime skills contain duplicate gene ids")

    for gene_id in regulation:
        if gene_id not in {p.gene_id for p in policies.values()} | {s.gene_id for s in skills}:
            raise GenomeValidationError(f"regulation references gene {gene_id!r} absent from runtime config")

    return RuntimeConfig(genome_id=genome["genome_id"], policies=policies,
                         skills=tuple(skills), regulation=regulation)
