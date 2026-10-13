"""Second batch of mutation-style edits on COPIES under out/repro/mut/."""
from pathlib import Path

MUT = Path(__file__).resolve().parents[1] / "mut"


def sub(path: Path, old: str, new: str) -> None:
    text = path.read_text()
    assert text.count(old) == 1, (path, old[:70], text.count(old))
    path.write_text(text.replace(old, new))


# M4: tau_k expression overhead no longer gates anything (stage-4 pass and
# preliminary eligibility ignore it).
g = MUT / "overhead_not_gating" / "inheritance" / "gate.py"
sub(g, "        stage_pass = regression_pass and overhead_pass and \\\n",
       "        stage_pass = regression_pass and True and \\\n")
sub(g, "                    and reg_results[gid].passed\n"
       "                    and overhead_results[gid][\"passed\"]):",
       "                    and reg_results[gid].passed):  # MUTATION")

# M5: generalization scopes dropped from the preliminary eligible set (a
# generalization-failing candidate is paired and can eliminate others).
g = MUT / "gen_not_in_eligibility" / "inheritance" / "gate.py"
sub(g, "                    and gen_results[gid].passed and gen_results[gid].out_scope_passed\n",
       "                    # MUTATION: generalization ignored for eligibility\n")

# M6: the orchestrator silently drops every candidate's frozen target
# regulation (stage 3, stage 4A and the stage-4C pair overlay) -> regulated
# candidates are evaluated constitutively.
g = MUT / "regulation_dropped_in_gate" / "inheritance" / "gate.py"
text = g.read_text()
n = text.count("target_regulation=self._target_regulation_by_gene.get(gid))")
assert n == 2, n
text = text.replace("target_regulation=self._target_regulation_by_gene.get(gid))",
                    "target_regulation=None)  # MUTATION")
assert text.count("target_regulation_by_gene=self._target_regulation_by_gene)") == 1
text = text.replace("target_regulation_by_gene=self._target_regulation_by_gene)",
                    "target_regulation_by_gene={})  # MUTATION")
g.write_text(text)
print("mutations batch 2 applied")
