import sys, json, tempfile
sys.path.insert(0, "/home/user/heritagent")
sys.dont_write_bytecode = True
from pathlib import Path
from genome.validation.registry import TraitRegistry
from runtime.seed_g0 import g0_runtime_config, build_g0_seed_genome
from runtime.loop import run_agent, Budgets, ToolObservation
from runtime.model_adapters import ToolSpec
from runtime.model_adapters.fake import ScriptedAdapter
from runtime.expression import compile_runtime_config

def act(i): return f"Thought: t\nAction: act\nAction Input: {json.dumps({'i': i})}"
class Env:
    def list_tools(self): return [ToolSpec(name="act", description="", parameters_schema={})]
    def call(self, n, a): return ToolObservation(ok=True, content="ok")

d = tempfile.mkdtemp()
reg = TraitRegistry(Path(d)/"r")
cfg = g0_runtime_config(reg)
m = ScriptedAdapter([act(i) for i in range(60)])
res = run_agent(config=cfg, model=m, env=Env(), task="t", budgets=Budgets(max_steps=50, max_total_retries=10))
print("status", res.status, "steps", res.steps if hasattr(res,'steps') else res)
print("model calls", len(m.requests))
print([e.data for e in res.events.of_kind("budget_exhausted")])

# can planner gene be raised to 50?
g = build_g0_seed_genome(reg)
try:
    reg.put("policy", "planner_react_v2", 2, {"style":"react","max_plan_steps":50})
    g["genes"]["cognition"]["planner"].update(gene_id="planner_react_v2", version=2,
        artifact=reg.put("policy","planner_react_v2",2,{"style":"react","max_plan_steps":50}))
    compile_runtime_config(g, reg); print("50 accepted")
except Exception as e:
    print("50 rejected:", type(e).__name__, str(e)[:200])

# final answer and rejected tool consume steps
m2 = ScriptedAdapter(["Thought: t\nAction: nope\nAction Input: {}"]*8)
r2 = run_agent(config=cfg, model=m2, env=Env(), task="t", budgets=Budgets(max_steps=50, max_total_retries=10))
print("rejected-tool episode: env calls 0, steps consumed:", len(m2.requests), r2.status)
