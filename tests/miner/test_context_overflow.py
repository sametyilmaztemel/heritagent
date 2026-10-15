"""ADR-0009 D8: a teacher prompt that does not fit the context window is
sent once more without the admissible-command lists, then the pair is
rejected as context_overflow. Nothing is truncated."""

import json

from runtime.loop import Budgets, ToolObservation, run_agent
from runtime.model_adapters import ScriptedAdapter
from runtime.model_adapters.errors import ContextWindowExceeded
from runtime.seed_g0 import g0_runtime_config
from runtime.tool_adapters.alfworld import render_observation
from traits.miner.evidence import (
    EVIDENCE_FORMS,
    WITHOUT_ADMISSIBLE_COMMANDS,
    render_pair_evidence,
    without_admissible_commands,
)
from traits.miner.inputs import MiningBatch, MiningPair
from traits.miner.miner import MiningSettings, TraitMiner, render_teacher_prompt
from traits.miner.records import mining_record_validator
from trajectory.normalization import load_normalized
from trajectory.recorder.context import OutcomeAnnotation, TrajectoryContext
from trajectory.recorder.recorder import TrajectoryRecorder
from tests.miner.conftest import FAMILIES, TASK, final, make_pair, react
from tests.miner.test_miner import GOLDEN_PROPOSALS
from tests.runtime.conftest import ScriptedEnv

COMMANDS = [f"go to cabinet {i}" for i in range(1, 120)]


def episode(tmp_path, registry, *, name, success):
    """A scripted episode whose tool results carry ADR-0003 D2's list."""
    adapter = ScriptedAdapter([react("heat_object", {"object": "plate"})])
    if success:
        adapter.enqueue_text(final("The plate is heated."))
    content = render_observation("You heat the plate." if success else "Nothing happens.",
                                 COMMANDS)
    env = ScriptedEnv({"heat_object": [ToolObservation(ok=success, content=content)]})
    path = tmp_path / f"{name}.jsonl"
    recorder = TrajectoryRecorder(TrajectoryContext(genome_id="G-0f1e2d3c", run_id=f"R-{name}"),
                                  path)
    run_agent(config=g0_runtime_config(registry), model=adapter, env=env, task=TASK,
              budgets=Budgets(max_steps=4 if success else 1,
                              max_total_retries=10 if success else 0,
                              max_tokens_per_request=64), hooks=(recorder,))
    recorder.attach_outcome(OutcomeAnnotation(success=success, reward=float(success),
                                              evaluator_id="scripted-evaluator",
                                              evaluator_version="1.0"))
    recorder.finalize()
    return load_normalized(path)


def long_batch(tmp_path, registry):
    pair = MiningPair(success=episode(tmp_path, registry, name="s", success=True),
                      failure=episode(tmp_path, registry, name="f", success=False),
                      task_family="heat_and_place")
    return MiningBatch(pairs=(pair,), family_universe=FAMILIES), pair


class Window(ScriptedAdapter):
    """A teacher whose server refuses prompts longer than ``limit`` characters."""

    def __init__(self, limit, responses=()):
        super().__init__(list(responses))
        self.limit = limit
        self.prompts = []

    def structured_generate(self, messages, settings, schema):
        self.prompts.append(messages[0].content)
        if len(messages[0].content) > self.limit:
            raise ContextWindowExceeded("This model's maximum context length is 32768 tokens.")
        return super().structured_generate(messages, settings, schema)


def between_the_forms(pair):
    """A limit the short form's prompt fits and the full form's does not."""
    full, short = (len(render_teacher_prompt(evidence_text=render_pair_evidence(pair, form)[0],
                                             family_universe=FAMILIES, max_proposals=10))
                   for form in EVIDENCE_FORMS)
    assert short < full
    return (short + full) // 2


def miner_with(registry, teacher):
    return TraitMiner(model=teacher, registry=registry, settings=MiningSettings(seed=11),
                      born_generation=1)


def test_the_second_form_drops_only_the_admissible_lists(tmp_path, registry):
    batch, pair = long_batch(tmp_path, registry)
    full, full_sha = render_pair_evidence(pair)
    short, short_sha = render_pair_evidence(pair, WITHOUT_ADMISSIBLE_COMMANDS)
    assert EVIDENCE_FORMS == ("full", WITHOUT_ADMISSIBLE_COMMANDS)
    assert "go to cabinet 7" in full and "go to cabinet 7" not in short
    assert full_sha != short_sha and len(short) < len(full) / 3
    document, compact = json.loads(full), json.loads(short)
    assert compact.pop("form") == WITHOUT_ADMISSIBLE_COMMANDS and "form" not in document
    step = compact["failed_trajectory"]["steps"][0]["tool_calls"][0]
    assert step["result_content"] == "Nothing happens."
    # everything else is the full form's
    for member in ("successful_trajectory", "failed_trajectory"):
        for a, b in zip(document[member]["steps"], compact[member]["steps"]):
            for call in a["tool_calls"]:
                call["result_content"] = without_admissible_commands(call["result_content"])
            assert a == b
    assert {k: v for k, v in document.items() if "trajectory" not in k} == \
        {k: v for k, v in compact.items() if "trajectory" not in k}


def test_the_full_forms_bytes_are_unchanged(tmp_path, registry):
    pair = make_pair(tmp_path, registry)  # no admissible lists at all
    text, _ = render_pair_evidence(pair)
    assert "form" not in json.loads(text)
    assert render_pair_evidence(pair, "full") == render_pair_evidence(pair)
    assert without_admissible_commands("x") == "x" and without_admissible_commands(None) is None


def test_a_prompt_that_fits_only_without_the_lists_is_mined_in_that_form(tmp_path, registry):
    batch, pair = long_batch(tmp_path, registry)
    short, short_sha = render_pair_evidence(pair, WITHOUT_ADMISSIBLE_COMMANDS)
    teacher = Window(limit=between_the_forms(pair),
                     responses=[{"proposals": [GOLDEN_PROPOSALS[0]]}])
    frozen = miner_with(registry, teacher).mine(batch)
    assert len(teacher.prompts) == 2 and short in teacher.prompts[1]
    record = frozen.mining_records[0]
    assert record["evidence_form"] == WITHOUT_ADMISSIBLE_COMMANDS
    assert record["evidence_sha256"] == short_sha
    assert record["rejected"] == [] and len(frozen.candidates) == 1
    assert not list(mining_record_validator().iter_errors(record))


def test_a_prompt_that_fits_in_neither_form_rejects_the_pair(tmp_path, registry):
    batch, pair = long_batch(tmp_path, registry)
    teacher = Window(limit=100)
    frozen = miner_with(registry, teacher).mine(batch)
    assert len(teacher.prompts) == 2  # one call per form, none sent twice
    assert frozen.candidates == ()
    record = frozen.mining_records[0]
    assert [r["reason"] for r in record["rejected"]] == ["context_overflow"]
    assert "maximum context length" in record["rejected"][0]["detail"]
    assert record["raw_teacher_text"] is None and record["raw_structured_response"] is None
    assert [r["reason"] for r in frozen.rejected] == ["context_overflow"]
    assert not list(mining_record_validator().iter_errors(record))


def test_a_prompt_that_fits_is_sent_once_in_full(tmp_path, registry):
    batch, pair = long_batch(tmp_path, registry)
    full, full_sha = render_pair_evidence(pair)
    teacher = Window(limit=10**9, responses=[{"proposals": []}])
    frozen = miner_with(registry, teacher).mine(batch)
    assert len(teacher.prompts) == 1 and full in teacher.prompts[0]
    assert frozen.mining_records[0]["evidence_form"] == "full"
    assert frozen.mining_records[0]["evidence_sha256"] == full_sha


def test_a_malformed_reply_in_the_second_form_is_malformed_output(tmp_path, registry):
    batch, pair = long_batch(tmp_path, registry)
    teacher = Window(limit=between_the_forms(pair), responses=[{"proposals": [], "extra": 1}])
    frozen = miner_with(registry, teacher).mine(batch)
    record = frozen.mining_records[0]
    assert record["evidence_form"] == WITHOUT_ADMISSIBLE_COMMANDS
    assert [r["reason"] for r in record["rejected"]] == ["malformed_output"]
