"""G0 seed runtime configuration (Milestone 0; issue #7 criterion 6).

The seed genome matches the EXP-0001 manifest (`G0-seed-react`: ReAct-style
planner + exponential retry policy, no skills). These are runtime policy
implementations selected by genome slots — not hard-coded special cases in
the agent loop.
"""

from __future__ import annotations

from genome.validation.registry import TraitRegistry

from runtime.expression import RuntimeConfig, compile_runtime_config

PLANNER_GENE_ID = "planner_react_v1"
PLANNER_VERSION = 1
PLANNER_PAYLOAD = {"style": "react", "max_plan_steps": 8}

RETRY_GENE_ID = "retry_backoff_v1"
RETRY_VERSION = 1
RETRY_PAYLOAD = {"max_retries": 3, "backoff": "exponential"}

G0_GENOME_ID = "G-0f1e2d3c"
G0_LINEAGE_ID = "L-alpha"


def build_g0_seed_genome(registry: TraitRegistry) -> dict:
    """Register the seed policy artifacts and return the G0 seed genome."""
    planner_uri = registry.put("policy", PLANNER_GENE_ID, PLANNER_VERSION, PLANNER_PAYLOAD)
    retry_uri = registry.put("policy", RETRY_GENE_ID, RETRY_VERSION, RETRY_PAYLOAD)
    return {
        "genome_id": G0_GENOME_ID,
        "schema_version": "0.1",
        "generation": 0,
        "lineage_id": G0_LINEAGE_ID,
        "parent": None,
        "genes": {
            "cognition": {
                "planner": {
                    "gene_id": PLANNER_GENE_ID, "version": PLANNER_VERSION, "type": "policy",
                    "origin": "seed", "artifact": planner_uri,
                    "provenance": {"born_generation": 0},
                }
            },
            "execution": {
                "retry_policy": {
                    "gene_id": RETRY_GENE_ID, "version": RETRY_VERSION, "type": "policy",
                    "origin": "seed", "artifact": retry_uri,
                    "provenance": {"born_generation": 0},
                }
            },
            "skills": [],
        },
    }


def g0_runtime_config(registry: TraitRegistry) -> RuntimeConfig:
    return compile_runtime_config(build_g0_seed_genome(registry), registry)
