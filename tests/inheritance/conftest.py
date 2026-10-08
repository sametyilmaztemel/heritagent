"""Shared fixtures for inheritance (CIG stage) tests — scripted, no GPU."""

import pytest

from genome.validation.registry import TraitRegistry
from runtime.loop import Budgets
from runtime.model_adapters import ScriptedAdapter
from runtime.seed_g0 import g0_runtime_config
from tests.somatic.conftest import acquired_envelope

GENE_ID = "look_before_heat_v1"
PARENT_CIG = "CIG-0007"
MINING_SEED = 11


@pytest.fixture
def registry(tmp_path):
    return TraitRegistry(tmp_path / "registry")


@pytest.fixture
def envelope(registry):
    return acquired_envelope(registry)


@pytest.fixture
def budgets():
    return Budgets(max_steps=4, max_total_retries=10, max_tokens_per_request=64)


@pytest.fixture
def base_config(registry):
    return g0_runtime_config(registry)
