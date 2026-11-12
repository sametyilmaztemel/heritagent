"""The teacher's per-call request and the batch cap are two numbers (ADR-0011
note N1, owner decision of 2026-11-12, "22 de b olsun"): each teacher call's
prompt asks for at most 5 proposals, K = 10 stays the cap of the whole batch,
and the request is not enforced. The prompt template, its hash, the output
schema and the 2,048-token budget do not change. ScriptedAdapter only; no
model."""

import hashlib
from pathlib import Path

import pytest

from runtime.model_adapters import ScriptedAdapter
from traits.miner.evidence import FULL, render_pair_evidence
from traits.miner.inputs import MiningBatch
from traits.miner.miner import (
    MiningSettings,
    TraitMiner,
    prompt_template_sha256,
    render_teacher_prompt,
)
from tests.miner.conftest import FAMILIES, make_batch, make_pair

REPO = Path(__file__).resolve().parents[2]
TEMPLATE = REPO / "traits/miner/prompts/differential-skill-v0.txt"
# the committed template's bytes; ADR-0011 note N1 leaves the template and its hash as they are
TEMPLATE_SHA256 = "36b2c57a40256cc057b8fe28100c600ab3a11eb270a8d688a283d6012bab8905"
ASKS_FIVE = "Return at most 5 proposals. Return ZERO proposals when the"
ASKS_TEN = "Return at most 10 proposals. Return ZERO proposals when the"


def proposal(index: int, family: str = "heat_and_place", generality: float = 0.5) -> dict:
    return {"name": f"skill_{index:02d}", "principle": f"principle {index}",
            "when_to_apply": "when applicable", "applicability": {"task_families": [family]},
            "estimated_generality": generality}


def scripted(registry, replies, **settings) -> TraitMiner:
    model = ScriptedAdapter([])
    for reply in replies:
        model.enqueue_structured(reply)
    return TraitMiner(model=model, registry=registry,
                      settings=MiningSettings(seed=11, **settings), born_generation=1)


def sent_prompts(miner: TraitMiner) -> list[str]:
    return [request.messages[-1].content for request in miner._model.requests]


def request_lines(prompt: str) -> list[str]:
    return [line for line in prompt.splitlines() if line.startswith("Return at most ")]


def test_the_defaults_ask_each_call_for_five_and_cap_the_batch_at_ten():
    settings = MiningSettings()
    assert settings.proposals_per_call == 5 and settings.max_proposals == 10


def test_the_template_bytes_are_pinned():
    """A template edit fails here: note N1 changes the rendered number only."""
    assert hashlib.sha256(TEMPLATE.read_bytes()).hexdigest() == TEMPLATE_SHA256
    assert prompt_template_sha256() == TEMPLATE_SHA256


def test_the_sent_prompt_differs_from_the_k_rendering_only_in_the_request_line(tmp_path,
                                                                                registry):
    batch, _ = make_batch(tmp_path, registry)
    miner = scripted(registry, [{"proposals": []}])
    miner.mine(batch)
    [sent] = sent_prompts(miner)
    evidence_text, _ = render_pair_evidence(batch.pairs[0], FULL)
    assert evidence_text in sent
    with_k = render_teacher_prompt(evidence_text=evidence_text,
                                   family_universe=batch.family_universe, max_proposals=10)
    assert with_k.count("\n") == sent.count("\n")
    differing = [(a, b) for a, b in zip(with_k.splitlines(), sent.splitlines()) if a != b]
    assert differing == [(ASKS_TEN, ASKS_FIVE)]
    assert sent == render_teacher_prompt(evidence_text=evidence_text,
                                         family_universe=batch.family_universe, max_proposals=5)


def test_k_is_still_the_batch_cap_over_three_calls_of_five(tmp_path, registry):
    """Three pairs, five valid distinct proposals each: every call asked
    for five, fifteen are valid, ten are selected and five are outside_cap."""
    pairs = tuple(make_pair(tmp_path / family, registry, family=family) for family in FAMILIES)
    replies = [{"proposals": [proposal(10 * n + i, family) for i in range(5)]}
               for n, family in enumerate(FAMILIES)]
    miner = scripted(registry, replies)
    frozen = miner.mine(MiningBatch(pairs=pairs, family_universe=FAMILIES))
    assert [request_lines(p) for p in sent_prompts(miner)] == [[ASKS_FIVE]] * 3
    assert len(frozen.candidates) == 10
    reasons = [entry["reason"] for entry in frozen.rejected]
    assert reasons == ["outside_cap"] * 5
    # equal generality: discovery order fills K, so the third call's five are outside it
    assert [len(r["accepted_proposals"]) for r in frozen.mining_records] == [5, 5, 0]


def test_a_reply_with_more_proposals_than_asked_is_processed_as_before(tmp_path, registry):
    """The request is not enforced (ADR-0011 note N1): seven valid distinct
    proposals in one reply give seven candidates under K = 10, and none is
    rejected for the count."""
    batch, _ = make_batch(tmp_path, registry)
    miner = scripted(registry, [{"proposals": [proposal(i) for i in range(7)]}])
    frozen = miner.mine(batch)
    assert [request_lines(p) for p in sent_prompts(miner)] == [[ASKS_FIVE]]
    assert len(frozen.candidates) == 7
    assert frozen.rejected == ()
    [record] = frozen.mining_records
    assert len(record["accepted_proposals"]) == 7 and record["rejected"] == []


@pytest.mark.parametrize("field", ["proposals_per_call", "max_proposals"])
@pytest.mark.parametrize("value", [0, -1, True, "5"])
def test_the_miner_refuses_a_count_that_is_not_a_positive_int(registry, field, value):
    with pytest.raises(ValueError, match=f"MiningSettings.{field} must be an int >= 1"):
        TraitMiner(model=ScriptedAdapter([]), registry=registry,
                   settings=MiningSettings(seed=11, **{field: value}), born_generation=1)
