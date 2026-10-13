"""R2: harmful-pair chain a>b>c. Harmful pairs (a,b) and (b,c); (a,c) is
NOT harmful. The simultaneous per-pair rule eliminates b (by a) AND c (by
b, which is itself eliminated), so only a promotes although {a, c} contains
no harmful pair. The per-pair resolution is also not recorded: b's child
marks the (b,c) pair 'retained': False although b won that pair.
Run from the PR checkout with PYTHONPATH=."""
import tempfile, pathlib
from tests.gate.test_gate_orchestrator import build_world

tmp = pathlib.Path(tempfile.mkdtemp())
orch, somatic, cig, runner, inputs, reg = build_world(
    tmp, candidates=("a", "b", "c"),
    ablation_deltas={"a": 0.9, "b": 0.5, "c": 0.1})

HARMFUL = [{"a", "b"}, {"b", "c"}]
real = runner.run_episode
def run_episode(task, config, budgets, condition, settings=None):
    out = real(task, config, budgets, condition, settings)
    if condition == "both":
        genes = {s.gene_id for s in config.skills} & {"a", "b", "c"}
        if genes in HARMFUL:
            from dataclasses import replace
            out = replace(out, success=False, expressed_gene_ids=tuple(sorted(genes)),
                          status="budget_exhausted")
        else:
            from dataclasses import replace
            out = replace(out, expressed_gene_ids=tuple(sorted(genes)))
    return out
runner.run_episode = run_episode

outcome = orch.run(inputs)
print("verdicts:", outcome.verdicts)
for r in outcome.results:
    inter = r.stage4_child["measurements"]["interaction"]
    print(r.gene_id, "eliminated_by(in-memory only):", r.eliminated_by,
          "| child pairs:",
          [(p["other_gene_id"], p["harmful"], p["retained"]) for p in inter["pair_evidence"]],
          "| child keys:", sorted(inter))
assert outcome.verdicts == {"a": "promote", "b": "reject", "c": "reject"}
print("REPRODUCED: c eliminated by already-eliminated b; {a,c} non-harmful "
      "but only a promotes. eliminated_by is not persisted in the child.")
