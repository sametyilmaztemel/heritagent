"""Chain: malformed #11 child (stage_pass=1, temperature=False, extra key) passes gate preflight AND is committed to the live CIG store with verify() clean."""
import copy, dataclasses, pathlib, tempfile
from inheritance.gate import GateCandidateInput
from tests.gate.test_gate_orchestrator import build_world
with tempfile.TemporaryDirectory() as d:
    orch, somatic, cig_store, runner, inputs, registry = build_world(pathlib.Path(d), candidates=("a",))
    inp = inputs[0]
    child = copy.deepcopy(inp.replay_evaluation.child_record)
    child["measurements"]["stage_pass"] = 1
    child["measurements"]["eval_generation_settings"] = {"temperature": False, "max_tokens": 64.0, "seed": 11.0, "stop": [], "top_p": 0.9}
    ev = dataclasses.replace(inp.replay_evaluation, _child_record=child)
    res = orch.run([dataclasses.replace(inp, replay_evaluation=ev)])
    print("run result type:", type(res).__name__, getattr(res, "committed", None))
    kids = cig_store.children if not callable(getattr(cig_store, "children", None)) else cig_store.children()
    rec = kids[f"{inp.parent_cig_id}/S11-replay"] if isinstance(kids, dict) else [k for k in kids if k["cig_id"].endswith("replay")][0]
    print("live stored replay stage_pass:", repr(rec["measurements"]["stage_pass"]))
    print("live stored settings:", rec["measurements"]["eval_generation_settings"])
    print("verify():", cig_store.verify())
