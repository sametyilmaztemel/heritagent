"""G0 seed runtime configuration (Milestone 0; issue #7 criterion 6).

The seed genome matches the EXP-0001 manifest (`G0-seed-react`: ReAct-style
planner + exponential retry policy, no skills). These are runtime policy
implementations selected by genome slots — not hard-coded special cases in
the agent loop.

The G0 ``genome_id`` is the canonical content hash of the seed's heritable
functional content (ADR-0001 Amendment A2, issue #22) — computed by
:func:`genome.validation.identity.compute_genome_id`, never hard-coded.
"""

from __future__ import annotations

from genome.validation.identity import compute_genome_id
from genome.validation.registry import TraitRegistry

from runtime.expression import RuntimeConfig, compile_runtime_config

# ADR-0003 D4: planner_react_v2 plans up to 50 steps (v1 capped G0 at 8)
PLANNER_GENE_ID = "planner_react_v2"
PLANNER_VERSION = 1
PLANNER_PAYLOAD = {"style": "react", "max_plan_steps": 50}

RETRY_GENE_ID = "retry_backoff_v1"
RETRY_VERSION = 1
RETRY_PAYLOAD = {"max_retries": 3, "backoff": "exponential"}

G0_LINEAGE_ID = "L-alpha"

# ADR-0003 D4: the EXP-0001 episode budget, Budgets.max_steps, identical in
# every arm and recorded in the manifest (environment.episode_max_steps)
EPISODE_MAX_STEPS = 50


def build_g0_seed_genome(registry: TraitRegistry) -> dict:
    """Register the seed policy artifacts and return the G0 seed genome
    with its canonical content-hash ``genome_id`` set (A2: functional
    content first, id computed from it, validator accepts it)."""
    planner_uri = registry.put("policy", PLANNER_GENE_ID, PLANNER_VERSION, PLANNER_PAYLOAD)
    retry_uri = registry.put("policy", RETRY_GENE_ID, RETRY_VERSION, RETRY_PAYLOAD)
    genome = {
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
    genome["genome_id"] = compute_genome_id(genome)
    return genome


def g0_runtime_config(registry: TraitRegistry) -> RuntimeConfig:
    return compile_runtime_config(build_g0_seed_genome(registry), registry)
