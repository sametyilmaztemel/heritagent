"""R06 - Exhaustive check of the stage-1 replay rule against an independent
reading of the locked prereg (EXP-0001-design.md:47; yaml:78-80; issue #11):

  PASS iff successes_with >= 3 of 5
       AND (successes_with - successes_without) >= 2      (net +2/5)
       AND target gene expressed in EVERY successful with-trait episode.

All 2^5 x 2^5 = 1024 with/without outcome patterns are run through the merged
ReplayEvaluator (scripted runner), each under two expression regimes:
  E1: every successful with-trait episode expresses the gene;
  E2: the first successful with-trait episode does NOT express it.
Also reports the side effect at replay.py:251-252 (a non-episode dict is
appended to the 'episodes' list when expression fails).

Run from /home/user/heritagent:
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. <venv314>/bin/python <this file>
"""
import itertools
import tempfile
from pathlib import Path

from genome.validation.registry import TraitRegistry
from inheritance.replay import ReplayAllocation, ReplayEvaluator
from inheritance.runner import EvaluationTask
from runtime.loop import Budgets
from runtime.seed_g0 import g0_runtime_config
from tests.inheritance.scripted_runner import ScriptedRunner
from tests.somatic.conftest import acquired_envelope

GENE = "look_before_heat_v1"
tmp = Path(tempfile.mkdtemp(prefix="r06-", dir=str(Path(__file__).parent)))
registry = TraitRegistry(tmp / "registry")
envelope = acquired_envelope(registry)
base = g0_runtime_config(registry)
BANK = "replay_bank[heat_and_place]"
alloc = ReplayAllocation(
    source_task=EvaluationTask(task_id="src-1", family="heat_and_place", payload={}),
    bank_variants=tuple(EvaluationTask(task_id=f"bank-{i}", family="heat_and_place",
                                       payload={}, bank_id=BANK) for i in range(1, 5)),
    source_family="heat_and_place", replay_bank_id=BANK)
ids = [t.task_id for t in alloc.tasks]


def reference(w, wo, expressed_ok):
    sw, so = sum(w), sum(wo)
    return sw >= 3 and (sw - so) >= 2 and expressed_ok


mismatches = 0
n = 0
episodes_len = {}
pass_count = {"E1": 0, "E2": 0}
for w in itertools.product([False, True], repeat=5):
    for wo in itertools.product([False, True], repeat=5):
        for regime in ("E1", "E2"):
            outcomes = {}
            first_success_seen = False
            for tid, ws, wos in zip(ids, w, wo):
                express = ws
                if regime == "E2" and ws and not first_success_seen:
                    express = False
                if ws:
                    first_success_seen = True
                outcomes[(tid, "with_trait")] = (ws, (GENE,) if express else ())
                outcomes[(tid, "without_trait")] = (wos, ())
            expressed_ok = not (regime == "E2" and any(w))
            ev = ReplayEvaluator(candidate_envelope=envelope, registry=registry, base_config=base,
                                 runner=ScriptedRunner(outcomes=outcomes),
                                 budgets=Budgets(max_steps=4, max_total_retries=10,
                                                 max_tokens_per_request=64),
                                 parent_cig_id="CIG-0007", mining_seed=11)
            res = ev.evaluate(alloc)
            n += 1
            ref = reference(w, wo, expressed_ok)
            pass_count[regime] += res.passed
            if res.passed != ref:
                mismatches += 1
                print("MISMATCH", w, wo, regime, res.passed, ref)
            if abs(res.delta_success - (sum(w) - sum(wo)) / 5) > 0:
                print("delta_success mismatch", w, wo)
            episodes_len.setdefault(len(res.episodes), 0)
            episodes_len[len(res.episodes)] += 1

print(f"evaluations: {n}; mismatches vs prereg reference: {mismatches}")
print(f"passing patterns: E1={pass_count['E1']}/1024, E2={pass_count['E2']}/1024")
print(f"len(result.episodes) distribution: {episodes_len}  (10 = 5 tasks x 2 conditions)")

# the 11th 'episode'
outcomes = {(t, "with_trait"): (True, ()) for t in ids} | {(t, "without_trait"): (False, ()) for t in ids}
ev = ReplayEvaluator(candidate_envelope=envelope, registry=registry, base_config=base,
                     runner=ScriptedRunner(outcomes=outcomes),
                     budgets=Budgets(max_steps=4, max_total_retries=10, max_tokens_per_request=64),
                     parent_cig_id="CIG-0007", mining_seed=11)
res = ev.evaluate(alloc)
last = res.child_record["measurements"]["episodes"][-1]
print(f"\nexpression failure: len(episodes)={len(res.episodes)}; last 'episode' keys={sorted(last)}")
print(f"  episodes with a 'task_id' key: {sum(1 for e in res.episodes if 'task_id' in e)}")
