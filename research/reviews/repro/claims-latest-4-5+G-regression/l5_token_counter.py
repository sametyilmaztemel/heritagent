"""L5 repro: actual-count validation through the full gate path."""
import tempfile
from pathlib import Path
from tests.gate.test_gate_orchestrator import build_world
from inheritance.runner import RunnerContractError

class Sneaky:
    identity = "sneaky-v1"
    def __init__(self, v): self.v = v
    def count_tokens(self, text):
        return 0 if text == "" else self.v

class IntSub(int):
    pass

for bad in (1.5, 3.0, -1, True, False, None, "12", IntSub(7)):
    tmp = Path(tempfile.mkdtemp())
    orch, somatic, cig_store, runner, inputs, registry = build_world(tmp, candidates=("a",))
    orch._token_counter = Sneaky(bad)  # construction probe would pass ('' -> 0)
    try:
        out = orch.run(inputs)
        print(f"{bad!r:>8}: NOT REJECTED tokens={out.results[0].stage4_child['measurements']['expression_overhead']['tokens']!r} verdict={out.verdicts}")
    except RunnerContractError as e:
        print(f"{bad!r:>8}: rejected; somatic={somatic.state('a',1)} cig_aggs={len(cig_store.aggregates)} children={len(cig_store.children)} :: {str(e)[:70]}")

# construction path: Sneaky passes validate_token_counter
from inheritance.regression import validate_token_counter
validate_token_counter(Sneaky(1.5)); print("probe passes Sneaky(1.5): yes")
