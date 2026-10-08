"""Evaluation-only runtime overlay for somatic candidates (issue #11 §2).

A somatic candidate has ``origin="acquired"``; the germline genome schema
intentionally rejects that origin. Evaluating a candidate therefore never
rewrites provenance and never injects a fake gene ref into a genome. Instead
this module builds an evaluation-only overlay:

    with_trait_config    = base RuntimeConfig + target RuntimeSkill (+ its
                           explicitly supplied frozen regulation, if any)
    without_trait_config = the exact base RuntimeConfig

Hard invariants (tested):
- the base RuntimeConfig is never mutated (frozen dataclasses; new instances
  are constructed);
- registry artifacts are never rewritten (read-only resolution);
- the target skill payload passes the existing ``project_skill`` allowlist /
  leak guard, so evaluation-only metadata cannot enter runtime messages;
- if the target gene already exists in the base config the evaluation setup
  is REJECTED (no double injection);
- the without-trait config contains neither the target skill nor its
  regulation.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass

from genome.validation.errors import GenomeValidationError
from genome.validation.loader import load_somatic
from genome.validation.registry import TraitRegistry
from runtime.expression import RuntimeConfig, RuntimeSkill
from runtime.model_adapters import GenerationSettings
from runtime.regulation import Condition

EVAL_TEMPERATURE = 0.0  # locked: one deterministic execution per (task, condition)


class EvaluationSetupError(ValueError):
    """The evaluation overlay cannot be constructed (fail closed)."""


@dataclass(frozen=True)
class EvaluationConfigs:
    """Paired evaluation configs. Only the validation envelope differs
    between the two arms — here, only the presence of the target skill."""

    with_trait: RuntimeConfig
    without_trait: RuntimeConfig
    target_gene_id: str


def _validate_candidate(envelope: dict, registry: TraitRegistry) -> dict:
    """Fail closed unless this is a fresh acquired candidate with valid
    registry binding/digest (existing boundary; no parallel validator)."""
    try:
        validated = load_somatic(copy.deepcopy(envelope), registry)
    except GenomeValidationError as exc:
        raise EvaluationSetupError(
            f"candidate fails somatic validation: {exc}") from None
    if validated["validation"]["state"] != "candidate":
        raise EvaluationSetupError(
            f"candidate state must be 'candidate', got "
            f"{validated['validation']['state']!r}")
    if validated["candidate"]["origin"] != "acquired":
        raise EvaluationSetupError(
            f"candidate origin must be 'acquired', got "
            f"{validated['candidate']['origin']!r}")
    return validated


def build_evaluation_configs(base_config: RuntimeConfig, envelope: dict,
                              registry: TraitRegistry, *,
                              target_regulation: Condition | None = None) -> EvaluationConfigs:
    """Build the paired with/without-trait evaluation configs.

    ``target_regulation`` preserves an explicitly supplied frozen
    ``express_when`` condition (future candidates); current #9 candidates
    have none → constitutive expression."""
    validated = _validate_candidate(envelope, registry)
    candidate = validated["candidate"]
    gene_id = candidate["gene_id"]

    if any(skill.gene_id == gene_id for skill in base_config.skills):
        raise EvaluationSetupError(
            f"target gene {gene_id!r} already exists in the base runtime "
            f"config; refusing to double-inject")

    # resolve the artifact (read-only) and run the existing projection
    # allowlist/leak guard — evaluation-only metadata cannot survive
    payload = json.loads(registry.resolve(candidate["artifact"]))
    from genome.validation.projection import project_skill  # merged #4 path
    from runtime.expression import validate_runtime_skill_payload  # same boundary
    projected = validate_runtime_skill_payload(project_skill(payload))

    target_skill = RuntimeSkill(
        gene_id=gene_id,
        name=projected["name"],
        principle=projected["principle"],
        when_to_apply=projected["when_to_apply"],
        procedure=tuple(projected.get("procedure") or ()),
        condition=target_regulation,
    )

    with_trait = RuntimeConfig(
        genome_id=base_config.genome_id,
        policies=base_config.policies,
        skills=base_config.skills + (target_skill,),
        regulation=dict(base_config.regulation),
    )
    if target_regulation is not None:
        with_trait.regulation[gene_id] = target_regulation

    without_trait = RuntimeConfig(
        genome_id=base_config.genome_id,
        policies=base_config.policies,
        skills=base_config.skills,
        regulation=dict(base_config.regulation),
    )

    # invariant: base config untouched
    if base_config.skills != without_trait.skills or \
            len(with_trait.skills) != len(base_config.skills) + 1:
        raise EvaluationSetupError("overlay corrupted the base configuration")
    return EvaluationConfigs(with_trait=with_trait, without_trait=without_trait,
                              target_gene_id=gene_id)


def evaluation_settings(max_tokens: int = 512, seed: int | None = None) -> GenerationSettings:
    """Locked evaluation settings: temperature 0, one deterministic run per
    (task instance, condition). Nonzero temperatures are rejected."""
    return GenerationSettings(temperature=0.0, max_tokens=max_tokens, seed=seed)
