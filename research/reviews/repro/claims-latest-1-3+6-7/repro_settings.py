"""L2 + L7 repro: #11 eval_generation_settings preflight at c6ce4c1."""
import copy, dataclasses, pathlib, tempfile
from inheritance.gate import EvidenceCoherenceError, GateCandidateInput
from tests.gate.test_gate_orchestrator import build_world
from tests.gate.helpers import BUDGETS, MINING_SEED

print("MINING_SEED", MINING_SEED, "max_tokens", BUDGETS.max_tokens_per_request)

def run_case(label, stage, mutate_settings=None, mutate_measurements=None):
    with tempfile.TemporaryDirectory() as d:
        orch, somatic, cig_store, runner, inputs, registry = build_world(pathlib.Path(d), candidates=("a",))
        inp = inputs[0]
        ev = inp.replay_evaluation if stage == "replay" else inp.ablation_evaluation
        child = copy.deepcopy(ev.child_record)
        m = child["measurements"]
        if mutate_settings:
            s = dict(m["eval_generation_settings"]); mutate_settings(s); m["eval_generation_settings"] = s
        if mutate_measurements:
            mutate_measurements(m)
        new_ev = dataclasses.replace(ev, _child_record=child)
        kw = dict(candidate_envelope=inp.candidate_envelope, frozen_rank=0,
                  replay_evaluation=inp.replay_evaluation, ablation_evaluation=inp.ablation_evaluation,
                  generalization_allocation=inp.generalization_allocation,
                  regression_allocation=inp.regression_allocation, parent_cig_id=inp.parent_cig_id)
        kw["replay_evaluation" if stage == "replay" else "ablation_evaluation"] = new_ev
        before = len(runner.calls)
        try:
            orch.run([GateCandidateInput(**kw)])
            print(f"[ACCEPTED] {stage:8s} {label}: run completed, new runner calls={len(runner.calls)-before}")
        except EvidenceCoherenceError as e:
            print(f"[rejected] {stage:8s} {label}: new calls={len(runner.calls)-before} :: {str(e)[:130]}")

with tempfile.TemporaryDirectory() as d:
    w = build_world(pathlib.Path(d), candidates=("a",))
    print("baseline settings replay:", w[4][0].replay_evaluation.child_record["measurements"]["eval_generation_settings"])
    print("baseline settings ablation:", w[4][0].ablation_evaluation.child_record["measurements"]["eval_generation_settings"])

for stage in ("replay", "ablation"):
    run_case("baseline (unchanged)", stage)
    print("-- L2 value mismatches --")
    run_case("temperature=0.7", stage, lambda s: s.__setitem__("temperature", 0.7))
    run_case("max_tokens=32", stage, lambda s: s.__setitem__("max_tokens", 32))
    run_case("seed=99", stage, lambda s: s.__setitem__("seed", 99))
    run_case("stop=['x']", stage, lambda s: s.__setitem__("stop", ["x"]))
    run_case("settings missing", stage, mutate_measurements=lambda m: m.pop("eval_generation_settings"))
    print("-- L7 type/shape --")
    run_case("temperature=False", stage, lambda s: s.__setitem__("temperature", False))
    run_case("temperature=0 (int)", stage, lambda s: s.__setitem__("temperature", 0))
    run_case("temperature=-0.0", stage, lambda s: s.__setitem__("temperature", -0.0))
    run_case("max_tokens=64.0", stage, lambda s: s.__setitem__("max_tokens", float(BUDGETS.max_tokens_per_request)))
    run_case("seed=float", stage, lambda s: s.__setitem__("seed", float(MINING_SEED)))
    run_case("stop=() tuple", stage, lambda s: s.__setitem__("stop", ()))
    run_case("extra key top_p=0.9", stage, lambda s: s.__setitem__("top_p", 0.9))
    run_case("missing stop key", stage, lambda s: s.pop("stop"))
    print("-- adjacent: child-level seed / stage_pass typing in same preflight --")
    run_case("child seed float", stage, mutate_measurements=None) if False else None
    run_case("stage_pass=1 (int for True)", stage, mutate_measurements=lambda m: m.__setitem__("stage_pass", 1 if m["stage_pass"] is True else 0))
