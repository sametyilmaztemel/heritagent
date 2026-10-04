"""HeritAgent Trait Miner v0 (issue #9).

SkillRL-adapted differential distillation: verified normalized trajectory
pairs -> canonical differential evidence -> structured teacher proposals ->
deterministic identity/dedupe/ranking/cap -> frozen somatic candidate set.
Mined skills are UNVALIDATED somatic hypotheses (origin="acquired"); only
CIG evidence (#11/#12) can later justify germline assimilation. Adapted
prior art, not a novelty claim."""

from traits.miner.inputs import MiningBatch, MiningPair, MinerInputError
from traits.miner.evidence import render_pair_evidence
from traits.miner.records import build_mining_record
from traits.miner.miner import (
    FrozenCandidateSet,
    MiningSettings,
    TraitMiner,
    output_schema,
    output_schema_sha256,
    prompt_template_sha256,
    prompt_template_text,
)

__all__ = [
    "FrozenCandidateSet",
    "MiningBatch",
    "MiningPair",
    "MiningSettings",
    "MinerInputError",
    "TraitMiner",
    "build_mining_record",
    "output_schema",
    "output_schema_sha256",
    "prompt_template_sha256",
    "prompt_template_text",
    "render_pair_evidence",
]
