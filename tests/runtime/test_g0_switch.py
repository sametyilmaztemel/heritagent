"""ADR-0003 D4, the G0 switch (after PR #29): G0 plans with planner_react_v2
up to 50 steps, the EXP-0001 episode budget is 50 steps in every arm, and
the manifest records both. G0's canonical genome_id changes with the
planner gene, as A2 of ADR-0001 requires for any heritable change."""

import re
from pathlib import Path

from genome.validation.identity import compute_genome_id
from runtime.loop import Budgets, ToolObservation, run_agent
from runtime.model_adapters import ScriptedAdapter
from runtime.seed_g0 import (
    EPISODE_MAX_STEPS,
    PLANNER_GENE_ID,
    PLANNER_PAYLOAD,
    build_g0_seed_genome,
    g0_runtime_config,
)
from tests.runtime.conftest import ScriptedEnv, react_action

MANIFEST = (Path(__file__).resolve().parents[2]
            / "experiments/manifests/EXP-0001-minimal-cig.yaml").read_text(encoding="utf-8")


def test_g0_plans_with_planner_react_v2_up_to_50_steps(registry):
    genome = build_g0_seed_genome(registry)
    planner = genome["genes"]["cognition"]["planner"]
    assert (planner["gene_id"], planner["version"]) == ("planner_react_v2", 1)
    assert PLANNER_PAYLOAD == {"style": "react", "max_plan_steps": 50}
    assert EPISODE_MAX_STEPS == 50
    assert genome["genome_id"] == compute_genome_id(genome)


def test_a_g0_episode_runs_to_the_50_step_budget(registry):
    """Neither the planner cap nor the budget stops G0 before step 50."""
    config = g0_runtime_config(registry)
    adapter = ScriptedAdapter([react_action("act", {"command": "look"})] * EPISODE_MAX_STEPS)
    env = ScriptedEnv({"act": [ToolObservation(ok=True, content="You see a fridge.")]
                              * EPISODE_MAX_STEPS})
    result = run_agent(config=config, model=adapter, env=env, task="t",
                       budgets=Budgets(max_steps=EPISODE_MAX_STEPS, max_total_retries=0,
                                       max_tokens_per_request=64))
    started = result.events.of_kind("run_started")[0].data
    assert (started["max_steps"], started["effective_max_steps"],
            started["planner_max_plan_steps"]) == (50, 50, 50)
    assert result.status == "budget_exhausted" and result.steps == 50
    assert len(env.calls) == 50


def section(name: str) -> str:
    """One top-level manifest section (pyyaml is not a test dependency)."""
    match = re.search(rf"^{name}:\n((?:[ #].*\n|\n)*)", MANIFEST, re.MULTILINE)
    assert match, f"EXP-0001 manifest has no {name} section"
    return match.group(1)


def test_the_manifest_records_the_switch():
    """Scoped to the sections ADR-0003 names (review of the G0 switch)."""
    initial = re.findall(r"^  initial: G0-seed-react +# (.*)$", section("genome"), re.MULTILINE)
    assert len(initial) == 1 and initial[0].startswith(f"{PLANNER_GENE_ID} + retry_backoff_v1")
    steps = re.findall(r"^  episode_max_steps: (\d+)", section("environment"), re.MULTILINE)
    assert steps == [str(EPISODE_MAX_STEPS)]
    assert len(re.findall(r"episode_max_steps:", MANIFEST)) == 1
