"""Shared fixtures for runtime tests — no GPU, no live model, no ALFWorld."""

import pytest

from genome.validation.registry import TraitRegistry
from runtime.expression import compile_runtime_config
from runtime.loop import ToolObservation
from runtime.model_adapters import ToolSpec
from runtime.seed_g0 import build_g0_seed_genome, g0_runtime_config

SKILL_GENE_ID = "look_before_heat_v1"


@pytest.fixture
def registry(tmp_path):
    return TraitRegistry(tmp_path / "registry")


@pytest.fixture
def g0_config(registry):
    return g0_runtime_config(registry)


class ScriptedEnv:
    """Scripted ToolEnvironment: per-tool queues of ToolObservations."""

    def __init__(self, results: dict[str, list[ToolObservation]], tools: list[ToolSpec] | None = None):
        self._results = {name: list(queue) for name, queue in results.items()}
        self.tools = tools or [ToolSpec(name=name, description="", parameters_schema={})
                               for name in self._results]
        self.calls: list[tuple[str, dict]] = []

    def list_tools(self) -> list[ToolSpec]:
        return self.tools

    def call(self, name: str, arguments: dict) -> ToolObservation:
        self.calls.append((name, arguments))
        queue = self._results.get(name)
        if not queue:
            return ToolObservation(ok=False, content=f"no scripted result for {name!r}")
        return queue.pop(0)


def seed_with_skill(registry: TraitRegistry, *, express_when: dict | None = None,
                    procedure: bool = True) -> dict:
    """G0 seed genome plus one skill gene carrying evaluation-only metadata
    (applicability) that must never reach runtime/model inputs."""
    genome = build_g0_seed_genome(registry)
    payload = {
        "name": SKILL_GENE_ID,
        "principle": "Inspect object affordances before heating.",
        "when_to_apply": "Heat tasks requiring pre-inspection.",
        "applicability": {"task_families": ["heat_and_place"]},  # evaluation-only
    }
    if procedure:
        payload["procedure"] = [{"id": "S1", "text": "find the object"},
                                {"id": "S2", "text": "examine affordances"}]
    uri = registry.put("skill", SKILL_GENE_ID, 1, payload)
    genome["genes"]["skills"].append({
        "gene_id": SKILL_GENE_ID, "version": 1, "type": "skill",
        "origin": "seed", "artifact": uri,
        "provenance": {"born_generation": 0},
    })
    if express_when is not None:
        genome["regulation"] = {SKILL_GENE_ID: {"express_when": express_when}}
    return genome


def config_with_skill(registry: TraitRegistry, **kwargs):
    return compile_runtime_config(seed_with_skill(registry, **kwargs), registry)


def react_action(tool: str, arguments: dict, thought: str = "Proceeding.") -> str:
    import json
    return (f"Thought: {thought}\nAction: {tool}\n"
            f"Action Input: {json.dumps(arguments, sort_keys=True)}")


def react_final(answer: str, thought: str = "Done.") -> str:
    return f"Thought: {thought}\nFinal Answer: {answer}"
