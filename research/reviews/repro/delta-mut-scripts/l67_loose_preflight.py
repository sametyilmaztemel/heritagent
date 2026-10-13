"""L67 at 6014bc7: loose != comparisons in _consume_stage12_evidence
(gate.py:574 seed, gate.py:582 stage_pass)."""
import copy, dataclasses, tempfile
from pathlib import Path
from tests.gate.test_gate_orchestrator import build_world
from tests.gate.helpers import MINING_SEED
print("MINING_SEED", MINING_SEED)

def attempt(label, stage, field, value, commit):
    orch, somatic, cig, runner, inputs, registry = build_world(
        Path(tempfile.mkdtemp()), candidates=("a",), commit_stores=commit)
    inp = inputs[0]
    ev = getattr(inp, f"{stage}_evaluation")
    child = copy.deepcopy(ev.child_record)
    if field == "stage_pass": child["measurements"]["stage_pass"] = value
    else: child[field] = value
    ev2 = dataclasses.replace(ev, _child_record=child)
    inp2 = dataclasses.replace(inp, **{f"{stage}_evaluation": ev2})
    try:
        out = orch.run([inp2])
        r = out.results[0]
        stored = r.aggregate["child_records"]
        print(f"{label:48s} commit={commit!s:5s} ACCEPTED: verdict={r.verdict} committed={out.committed} runner_calls={len(runner.calls)} child_ids={[c for c in stored if stage in c]}")
    except Exception as e:
        print(f"{label:48s} commit={commit!s:5s} rejected: {type(e).__name__} runner_calls={len(runner.calls)}: {str(e)[:150]}")

for commit in (False, True):
    attempt("replay stage_pass=1 (passed True)", "replay", "stage_pass", 1, commit)
    attempt("replay stage_pass=1.0 (passed True)", "replay", "stage_pass", 1.0, commit)
    attempt("replay stage_pass='true'", "replay", "stage_pass", "true", commit)
    attempt("replay stage_pass='false'", "replay", "stage_pass", "false", commit)
    attempt("replay seed=float(MINING_SEED)", "replay", "seed", float(MINING_SEED), commit)
    attempt("replay seed=str(MINING_SEED)", "replay", "seed", str(MINING_SEED), commit)
    attempt("ablation stage_pass=1", "ablation", "stage_pass", 1, commit)
