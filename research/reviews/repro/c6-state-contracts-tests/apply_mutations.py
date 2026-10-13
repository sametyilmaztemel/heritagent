"""Apply mutation-style edits to the COPIES under out/repro/mut/<name>/
(never the pinned pr28c checkout). Each mutation deletes/weakens a guarded
behavior; if the copy's test suite still passes, the tests do not guard it.
"""
from pathlib import Path

MUT = Path(__file__).resolve().parents[1] / "mut"


def sub(path: Path, old: str, new: str) -> None:
    text = path.read_text()
    assert text.count(old) == 1, (path, old[:60], text.count(old))
    path.write_text(text.replace(old, new))


# M1: harmful-pair resolution ignores ablation contribution entirely and
# always retains the earlier frozen rank (the documented primary rule —
# "lower-ablation-contribution member is eliminated" — deleted).
gate = MUT / "rank_only_harmful" / "inheritance" / "gate.py"
sub(gate,
    "            contrib_a, contrib_b = contribution[gid_a], contribution[gid_b]\n"
    "            if contrib_a > contrib_b:",
    "            contrib_a, contrib_b = 0.0, 0.0  # MUTATION: contribution ignored\n"
    "            if contrib_a > contrib_b:")

# M2: CigRecordStore.add_bundle stops being all-or-nothing: the aggregate is
# inserted into the live dict BEFORE the per-child checks run.
rec = MUT / "add_bundle_nonatomic" / "genome" / "validation" / "records.py"
sub(rec,
    "        # per-child checks (staged; store untouched)\n",
    "        self._aggregates[staged_agg['cig_id']] = staged_agg  # MUTATION\n"
    "        # per-child checks (staged; store untouched)\n")

# M3: the #11 preflight only consumes the REPLAY child; the ablation child
# is never structurally/identity/settings/stage_pass checked.
gate3 = MUT / "no_ablation_preflight" / "inheritance" / "gate.py"
sub(gate3,
    "        for stage, evaluation in ((\"replay\", inp.replay_evaluation),\n"
    "                                   (\"ablation\", inp.ablation_evaluation)):\n"
    "            child = evaluation.child_record\n"
    "            try:\n"
    "                load_cig(copy.deepcopy(child))",
    "        for stage, evaluation in ((\"replay\", inp.replay_evaluation),):  # MUTATION\n"
    "            child = evaluation.child_record\n"
    "            try:\n"
    "                load_cig(copy.deepcopy(child))")
print("mutations applied")
