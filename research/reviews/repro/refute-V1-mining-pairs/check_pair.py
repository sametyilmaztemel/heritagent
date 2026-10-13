import sys, tempfile, pathlib
sys.path.insert(0, "/home/user/heritagent")
sys.dont_write_bytecode = True
from genome.validation.registry import TraitRegistry
from runtime.loop import Budgets, ToolObservation, run_agent
from runtime.model_adapters import ScriptedAdapter
from runtime.seed_g0 import g0_runtime_config
from trajectory.normalization import load_normalized
from trajectory.recorder.context import OutcomeAnnotation, TrajectoryContext
from trajectory.recorder.recorder import TrajectoryRecorder
from tests.runtime.conftest import ScriptedEnv
from tests.miner.conftest import react, final
from traits.miner.inputs import MiningPair

TASK = "Your task is to: put a clean apple in fridge."
tmp = pathlib.Path(tempfile.mkdtemp(dir="."))
reg = TraitRegistry(tmp / "registry")

def ep(name, game, success):
    cfg = g0_runtime_config(reg)
    ad = ScriptedAdapter([])
    ad.enqueue_text(react("go", {"target": "fridge 1"}))
    if success:
        ad.enqueue_text(final("done"))
    env = ScriptedEnv({"go": [ToolObservation(ok=success, content=f"{game} obs")]})
    ctx = TrajectoryContext(genome_id="G-0f1e2d3c", run_id=f"R-{name}", environment="alfworld",
                            benchmark="ALFWorld", split="train", seed=11,
                            metadata={"game_file": game})
    p = tmp / f"{name}.jsonl"
    rec = TrajectoryRecorder(ctx, p)
    run_agent(config=cfg, model=ad, env=env, task=TASK,
              budgets=Budgets(max_steps=4 if success else 1, max_total_retries=10 if success else 0,
                              max_tokens_per_request=64), hooks=(rec,))
    rec.attach_outcome(OutcomeAnnotation(success=success, reward=float(success),
                                         evaluator_id="e", evaluator_version="1"))
    rec.finalize()
    return load_normalized(p)

s = ep("s", "pick_clean_then_place-Apple-None-Fridge-1/trial_A", True)
f = ep("f", "pick_clean_then_place-Apple-None-Fridge-17/trial_B", False)
print("provenance keys:", sorted(s.provenance))
print("game metadata visible to pair check:", "game_file" in s.provenance)
pair = MiningPair(success=s, failure=f, task_family="pick_clean_then_place")
print("ACCEPTED cross-instance same-text pair:", pair.success.trajectory_id != pair.failure.trajectory_id)
