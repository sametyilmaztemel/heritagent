"""R8: measure_expression_overhead() fails closed on English SUBSTRINGS
('applicability', 'provenance', ...) in the projected, runtime-approved
skill text. A legitimate candidate whose principle mentions 'provenance'
aborts the WHOLE batch with RunnerContractError after its stage-3 and
T_reg episodes were already spent; no evidence/verdict is produced for any
candidate. Run from the PR checkout with PYTHONPATH=."""
import tempfile, pathlib
from tests.gate.test_gate_orchestrator import build_world
from inheritance.regression import measure_expression_overhead
from inheritance.overlay import build_evaluation_configs
from runtime.expression import RuntimeSkill

tmp = pathlib.Path(tempfile.mkdtemp())
orch, somatic, cig, runner, inputs, reg = build_world(tmp, candidates=("a",))
skill = RuntimeSkill(gene_id="x", name="check_label",
                     principle="Check the provenance label of an object before placing it.",
                     when_to_apply="sorting tasks", procedure=(), condition=None)
try:
    measure_expression_overhead(skill, orch._token_counter)
except Exception as e:
    print(type(e).__name__, ":", e)

# end-to-end: projection accepts the text, gate aborts after episodes
from genome.validation.projection import project_skill
from runtime.expression import validate_runtime_skill_payload
payload = {"name": "label_check", "principle": "Verify the provenance label first.",
           "when_to_apply": "sorting tasks",
           "applicability": {"task_families": ["heat_and_place"]},
           "estimated_generality": 0.5}
print("projection accepts:", validate_runtime_skill_payload(project_skill(payload)))

# end-to-end through the orchestrator: candidate b's principle mentions
# 'applicability' (plain English); the run aborts after spending episodes
import tests.gate.helpers as H
orig = H.proposal
def proposal(name, family="heat_and_place"):
    p = orig(name, family)
    if name == "b":
        p["principle"] = "Consider the applicability of each receptacle."
    return p
H.proposal = proposal
tmp2 = pathlib.Path(tempfile.mkdtemp())
orch, somatic, cig, runner, inputs, reg = build_world(tmp2, candidates=("a", "b"))
try:
    orch.run(inputs)
except Exception as e:
    print("e2e:", type(e).__name__, "after", len(runner.calls), "runner calls;",
          "somatic states:", {g: somatic.state(g, 1) for g in ("a", "b")},
          "| live CIG aggregates:", list(cig.aggregates))
