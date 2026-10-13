"""Apply ONE mutation per COPY of merged main (6014bc7) under delta-mut/<name>.
Never touches /home/user/heritagent."""
import shutil, subprocess, sys
from pathlib import Path
SRC = Path("/home/user/heritagent")
MUT = Path("<scratch>/out/repro/delta-mut")

def fresh(name):
    d = MUT / name
    if d.exists(): shutil.rmtree(d)
    shutil.copytree(SRC, d, ignore=shutil.ignore_patterns(".git", "__pycache__"))
    return d

def sub(path, old, new, count=1):
    t = path.read_text()
    assert t.count(old) == count, (path, old[:70], t.count(old))
    path.write_text(t.replace(old, new))

G = "inheritance/gate.py"
muts = {}
def m(f): muts[f.__name__] = f; return f

@m
def M1_rank_only_harmful(d):
    sub(d/G, "            contrib_a, contrib_b = contribution[gid_a], contribution[gid_b]\n",
             "            contrib_a, contrib_b = 0.0, 0.0  # MUTATION\n")
@m
def M4a_overhead_not_in_stage4_pass(d):
    sub(d/G, "        stage_pass = regression_pass and overhead_pass and \\\n",
             "        stage_pass = regression_pass and True and \\\n")
@m
def M4b_overhead_not_gating_anywhere(d):
    M4a_overhead_not_in_stage4_pass(d)
    sub(d/G, "                    and reg_results[gid].passed\n                    and overhead_results[gid][\"passed\"]):",
             "                    and reg_results[gid].passed):  # MUTATION")
@m
def M4c_overhead_not_in_eligibility(d):
    sub(d/G, "                    and reg_results[gid].passed\n                    and overhead_results[gid][\"passed\"]):",
             "                    and reg_results[gid].passed):  # MUTATION")
@m
def M5_gen_not_in_eligibility(d):
    sub(d/G, "                    and gen_results[gid].passed and gen_results[gid].out_scope_passed\n",
             "                    # MUTATION\n")
@m
def T1_no_ablation_preflight(d):
    sub(d/G, "        for stage, evaluation in ((\"replay\", inp.replay_evaluation),\n                                   (\"ablation\", inp.ablation_evaluation)):\n",
             "        for stage, evaluation in ((\"replay\", inp.replay_evaluation),):  # MUTATION\n")
@m
def T1b_no_preflight_at_all(d):
    sub(d/G, "        for inp in inputs:\n            self._consume_stage12_evidence(inp)\n",
             "        pass  # MUTATION\n")
@m
def T2_regulation_dropped_all(d):
    sub(d/G, "target_regulation=self._target_regulation_by_gene.get(gid))", "target_regulation=None)  # MUTATION", 2)
    sub(d/G, "target_regulation_by_gene=self._target_regulation_by_gene)", "target_regulation_by_gene={})  # MUTATION")
@m
def T2a_regulation_dropped_gen(d):
    t = (d/G).read_text()
    old = "budgets=self._budgets, parent_cig_id=parent, mining_seed=seed,\n                target_regulation=self._target_regulation_by_gene.get(gid))"
    sub(d/G, old, "budgets=self._budgets, parent_cig_id=parent, mining_seed=seed,\n                target_regulation=None)  # MUTATION")
@m
def T2b_regulation_dropped_reg(d):
    sub(d/G, "                mining_seed=seed,\n                target_regulation=self._target_regulation_by_gene.get(gid))",
             "                mining_seed=seed,\n                target_regulation=None)  # MUTATION")
@m
def T2c_regulation_dropped_interaction(d):
    sub(d/G, "target_regulation_by_gene=self._target_regulation_by_gene)", "target_regulation_by_gene={})  # MUTATION")
@m
def T3a_add_bundle_nonatomic(d):
    sub(d/"genome/validation/records.py", "        # per-child checks (staged; store untouched)\n",
        "        self._aggregates[staged_agg['cig_id']] = staged_agg  # MUTATION\n        # per-child checks (staged; store untouched)\n")
@m
def T3b_add_bundle_children_first_nonatomic(d):
    # insert each child into the live store as it is checked
    sub(d/"genome/validation/records.py", "            stage = child[\"stage\"]\n            if stage not in staged_agg[\"stages\"]:\n",
        "            self._children[child['cig_id']] = child  # MUTATION\n            stage = child[\"stage\"]\n            if stage not in staged_agg[\"stages\"]:\n")
@m
def T3c_stage3_hidden_from_runner_on_replay_fail(d):
    # replay-failing candidates run stage 3 against a throwaway deep copy of
    # the runner: the shared runner records ZERO stage-3 episodes for them
    # (observationally a short-circuit from the test's point of view)
    sub(d/G, "            gen_results[gid] = evaluator.evaluate(inp.generalization_allocation)\n",
             "            if not inp.replay_evaluation.passed:  # MUTATION\n"
             "                evaluator._runner = copy.deepcopy(self._runner)\n"
             "            gen_results[gid] = evaluator.evaluate(inp.generalization_allocation)\n")

if __name__ == "__main__":
    names = sys.argv[1:] or list(muts)
    for n in names:
        d = fresh(n); muts[n](d)
        diff = subprocess.run(["diff", "-ru", str(SRC/"inheritance"), str(d/"inheritance")], capture_output=True, text=True).stdout
        diff += subprocess.run(["diff", "-ru", str(SRC/"genome"), str(d/"genome")], capture_output=True, text=True).stdout
        (MUT/f"{n}.diff").write_text(diff)
        print("applied", n)
