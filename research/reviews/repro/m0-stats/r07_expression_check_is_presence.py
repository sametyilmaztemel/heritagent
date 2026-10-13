"""R07 - For constitutive candidates (target_regulation=None, the M0 default:
overlay.py:78-80 "current #9 candidates have none -> constitutive
expression"), the replay "expression check" reduces to a PRESENCE check.

runtime/loop.py:165-166 emits skills_expressed with every skill whose
regulation is active; runtime/skills.py:16-19 says constitutive skills ALWAYS
express. The runner contract (EpisodeOutcome docstring, inheritance/runner.py:149-158)
defines expressed_gene_ids as gene ids observed in
skills_expressed events. Hence a constitutive target is "expressed" in every
with-trait episode from step 1, whether or not the model's behaviour used it.

Prereg (EXP-0001-design.md:47): "trait must fire in every successful
with-trait episode (expression check, not just presence)"; issue #11: "trait
must actually fire".

This script builds a minimal GateEpisodeRunner over the REAL run_agent path
(the shape a future benchmark runner is contracted to have), with a model that
emits byte-identical responses in both conditions (it never acts on the
skill), and shows that the ReplayEvaluator's expression check passes.
A regulated target (consecutive_failures >= 2) is shown for contrast.

Run from /home/user/heritagent:
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. <venv314>/bin/python <this file>
"""
import tempfile
from pathlib import Path

from genome.validation.registry import TraitRegistry
from inheritance.overlay import build_evaluation_configs
from inheritance.replay import ReplayAllocation, ReplayEvaluator
from inheritance.runner import EpisodeOutcome, EvaluationTask
from runtime.loop import Budgets, run_agent
from runtime.model_adapters import ScriptedAdapter
from runtime.regulation import Condition
from runtime.seed_g0 import g0_runtime_config
from tests.runtime.conftest import ScriptedEnv
from tests.somatic.conftest import acquired_envelope

FINAL = "Thought: Done.\nFinal Answer: placed."

tmp = Path(tempfile.mkdtemp(prefix="r07-", dir=str(Path(__file__).parent)))
registry = TraitRegistry(tmp / "registry")
envelope = acquired_envelope(registry)
base = g0_runtime_config(registry)


class RunAgentRunner:
    """Minimal contract-conforming runner over run_agent. Benchmark success is
    supplied by a scripted oracle per (task, condition) - independent of the
    model's behaviour - exactly as the contract allows."""

    def __init__(self, oracle):
        self.oracle = oracle
        self.raw = {}

    def run_episode(self, task, config, budgets, condition, settings):
        adapter = ScriptedAdapter([FINAL])
        events = []
        run_agent(config=config, model=adapter, env=ScriptedEnv({}), task="Put the plate away.",
                  budgets=budgets, hooks=(events.append,))
        expressed = []
        for e in events:
            if e.kind == "skills_expressed":
                for g in e.data["skills"]:
                    if g not in expressed:
                        expressed.append(g)
        self.raw[(task.task_id, condition)] = "\n".join(
            m.content for r in adapter.requests for m in r.messages)
        return EpisodeOutcome(task_id=task.task_id, condition=condition,
                              success=self.oracle[(task.task_id, condition)],
                              expressed_gene_ids=tuple(expressed), status="finished",
                              trajectory_id=f"T-{task.task_id}-{condition}",
                              effective_settings=settings)


BANK = "replay_bank[heat_and_place]"
alloc = ReplayAllocation(
    source_task=EvaluationTask(task_id="src-1", family="heat_and_place", payload={}),
    bank_variants=tuple(EvaluationTask(task_id=f"bank-{i}", family="heat_and_place",
                                       payload={}, bank_id=BANK) for i in range(1, 5)),
    source_family="heat_and_place", replay_bank_id=BANK)
ids = [t.task_id for t in alloc.tasks]
oracle = {}
for k, tid in enumerate(ids):
    oracle[(tid, "with_trait")] = k < 3        # 3/5 with
    oracle[(tid, "without_trait")] = k == 0    # 1/5 without -> net +2

for label, regulation in (("constitutive (M0 default)", None),
                          ("regulated: consecutive_failures >= 2",
                           Condition.from_genome({"express_when": {"all_of": [
                               {"metric": "consecutive_failures", "op": ">=", "value": 2}]}}))):
    runner = RunAgentRunner(oracle)
    ev = ReplayEvaluator(candidate_envelope=envelope, registry=registry, base_config=base,
                         runner=runner, budgets=Budgets(max_steps=2, max_total_retries=1,
                                                        max_tokens_per_request=64),
                         parent_cig_id="CIG-0007", mining_seed=11, target_regulation=regulation)
    res = ev.evaluate(alloc)
    gene = build_evaluation_configs(base, envelope, registry).target_gene_id
    with_eps = [e for e in res.episodes if e.get("condition") == "with_trait"]
    print(f"[{label}]")
    print(f"  target gene expressed in with-trait episodes: "
          f"{[gene in e['expressed_gene_ids'] for e in with_eps]}")
    print(f"  successes_with={res.successes_with} delta_count={res.delta_count} "
          f"expression_ok={res.expression_ok} passed={res.passed}")
    in_with = all("look_before_heat_v1" in runner.raw[(t, "with_trait")] for t in ids)
    in_without = any("look_before_heat_v1" in runner.raw[(t, "without_trait")] for t in ids)
    print(f"  skill text in with-trait prompts: {in_with}; in without-trait prompts: {in_without}")
    print(f"  model output in BOTH arms is the same scripted reply {FINAL!r} "
          f"(the skill never changed behaviour)")
