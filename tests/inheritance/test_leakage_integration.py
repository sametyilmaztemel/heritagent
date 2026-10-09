"""Runtime-leakage integration through the REAL run_agent/events path
(issue #11 criterion 12) — with-trait model messages contain only projected
runtime skill fields; without-trait messages contain no target skill
content; expression evidence comes from skills_expressed events."""

import json

from inheritance.overlay import build_evaluation_configs
from runtime.loop import Budgets, run_agent
from runtime.model_adapters import ScriptedAdapter
from tests.miner.conftest import make_batch
from tests.miner.test_miner import GOLDEN_PROPOSALS  # noqa: F401 — reuse proposal
from tests.runtime.conftest import ScriptedEnv, ToolObservation

FORBIDDEN = ("applicability", "task_families", "estimated_generality",
              "evidence_rationale", "success", "reward", "source_trajectory",
              "provenance", "cig_record", "gate_reports")


def test_with_trait_messages_contain_only_projected_fields(tmp_path, registry, base_config,
                                                             envelope):
    from traits.miner.miner import MiningSettings, TraitMiner
    batch, _ = make_batch(tmp_path, registry)
    miner = TraitMiner(model=ScriptedAdapter([{"proposals": [
        {"name": "look_before_heat_v1",
         "principle": "Inspect the heater before use.",
         "when_to_apply": "Any task that requires heating an object.",
         "applicability": {"task_families": ["heat_and_place"]},
         "estimated_generality": 0.5,
         "evidence_rationale": "failure shows the heater was never probed"},
    ]}]), registry=registry, settings=MiningSettings(seed=11), born_generation=1)
    frozen = miner.mine(batch)
    envelope = frozen.candidates[0]

    configs = build_evaluation_configs(base_config, envelope, registry)
    adapter = ScriptedAdapter([react_action := __import__(
        "tests.runtime.conftest", fromlist=["react_action"]).react_action(
        "heat_object", {"object": "plate"}),
        __import__("tests.runtime.conftest", fromlist=["react_final"]).react_final(
        "The plate is heated.")])
    env = ScriptedEnv({"heat_object": [ToolObservation(ok=True, content="heated")]})
    events = []
    result = run_agent(config=configs.with_trait, model=adapter, env=env,
                        task="Heat the plate.",
                        budgets=Budgets(max_steps=4, max_total_retries=10,
                                         max_tokens_per_request=64),
                        hooks=(events.append,))
    assert result.status == "finished"

    rendered = json.dumps([
        [{"role": m.role, "content": m.content} for m in request.messages]
        for request in adapter.requests])
    for forbidden in FORBIDDEN:
        assert forbidden not in rendered, f"{forbidden} leaked into model messages"
    # the projected runtime skill content IS present
    assert "look_before_heat_v1" in rendered, f"RENDERED: {rendered[:800]}"
    assert "Inspect the heater before use." in rendered
    # expression evidence comes from the event path
    expressed_events = [e for e in events if e.kind == "skills_expressed"]
    # the gene id carries its deterministic content suffix: compare exactly
    assert any(configs.target_gene_id in e.data["skills"]
                for e in expressed_events)


def react_action(tool, arguments, thought="Proceeding."):
    import json as _json
    return (f"Thought: {thought}\nAction: {tool}\n"
            f"Action Input: {_json.dumps(arguments, sort_keys=True)}")


def react_final(answer, thought="Done."):
    return f"Thought: {thought}\nFinal Answer: {answer}"


def test_without_trait_messages_contain_no_target_skill(tmp_path, registry, base_config,
                                                          envelope):
    configs = build_evaluation_configs(base_config, envelope, registry)
    adapter = ScriptedAdapter([react_final("The plate is not heated.")])
    env = ScriptedEnv({})
    run_agent(config=configs.without_trait, model=adapter, env=env,
              task="Heat the plate.",
              budgets=Budgets(max_steps=1, max_total_retries=1,
                               max_tokens_per_request=64))
    rendered = json.dumps([
        [{"role": m.role, "content": m.content} for m in request.messages]
        for request in adapter.requests])
    assert "look_before_heat_v1" not in rendered
    assert "Inspect the heater before use." not in rendered
