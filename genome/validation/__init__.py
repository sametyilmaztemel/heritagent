"""Loader/validator, registry, projection, record stores, and diff for EAG v0.1."""

from genome.validation.errors import GenomeValidationError
from genome.validation.registry import TraitRegistry
from genome.validation.loader import load_genome, load_somatic, load_cig
from genome.validation.projection import (
    project_for_runtime,
    project_skill,
    project_skillrl,
)
from genome.validation.records import CigRecordStore, SomaticStore
from genome.validation.diff import diff_genomes

__all__ = [
    "GenomeValidationError",
    "TraitRegistry",
    "load_genome",
    "load_somatic",
    "load_cig",
    "project_for_runtime",
    "project_skill",
    "project_skillrl",
    "CigRecordStore",
    "SomaticStore",
    "diff_genomes",
]
