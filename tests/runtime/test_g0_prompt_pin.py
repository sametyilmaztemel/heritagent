"""Pins the exact messages G0's model receives in EXP-0001 (system message,
user message per step, format retry), through the real code path: G0's
compiled config, the execution loop, the ALFWorld adapter over a scripted
TextWorld session and a scripted model.

Why these are pinned:
- ADR-0003 D1: one tool, ``act``. D2 and A1: every observation, the
  initial one included, ends with the sorted admissible commands. D7: the
  task text is the initial observation without the banner. (A2's stop
  sequence is a generation setting, not part of the messages.)
- ADR-0003 note N1, owner decision of 2026-11-12 ("21 a kalsın"): of the
  episode so far, the G0 prompt shows the task and the observations. The
  agent's own earlier Thoughts, Actions and Action Inputs are not shown,
  with two exceptions: a format retry inside one step (the rejected reply
  and the correction, up to three times in a step), and the error text of
  a rejected tool call, which stays in the history and can quote an
  argument's name or its value. After an inadmissible command the agent
  sees "Nothing happens." without the command it sent. This is part of
  EXP-0001's frozen G0 prompt.
- ADR-0009 D5: the task text is repeated as observation [0].
- The main session checked on 2026-11-12 that the same scripted episode
  gives identical messages at the calibration pilot's commit 492d06b and at
  main 1a16a40 (V). This file passes on an export of either commit (V,
  2026-11-12). Since the pilot's commit, an Action Input holding NaN,
  Infinity, an overflowing number, a lone surrogate escape or nesting
  deeper than 32 levels is a format failure retried inside the step (PR
  #83); at 492d06b it was a rejected tool call that used a step, or, for
  the lone surrogate, a command sent to the game. So the sentence before
  holds only for what this file scripts, which has no such reply. The
  pilot's measures, on which ADR-0012's calibration rests, were taken with
  this prompt.

A change to any literal here is a change to what every arm's model sees.
It needs an owner decision, and a new experiment id once the EXP-0001 run
has started. Do not update a literal to make a test pass.

No alfworld or textworld package is needed: the adapter runs over a
scripted session, as in tests/runtime/test_alfworld_adapter.py.
"""

import hashlib
import json

import pytest

from evaluation.exp0001.arms import g1_genome, germline_ref
from runtime.expression import compile_runtime_config
from runtime.loop import Budgets, run_agent
from runtime.model_adapters import ScriptedAdapter
from runtime.seed_g0 import EPISODE_MAX_STEPS, build_g0_seed_genome
from runtime.tool_adapters.alfworld import AlfworldEnvironment
from tests.runtime.conftest import react_action, react_final

DECISION = ("ADR-0003 note N1, owner decision 2026-11-12 ('21 a kalsın'): "
            "of the episode so far G0's frozen prompt shows the task and the "
            "observations, not the agent's own earlier replies, except a format "
            "retry inside one step and the error text of a rejected tool call")

GAME_ID = "pin/trial_T0001"
ADMISSIBLE = ["take mug 1 from desk 1", "look", "go to desk 1"]  # the adapter sorts them
INITIAL = ("-= Welcome to TextWorld, ALFRED! =-\n\nYou are in the middle of a room. "
           "Looking quickly around you, you see a desk 1.\n\nYour task is to: put a mug in desk.")
ARRIVE = "You arrive at desk 1. On the desk 1, you see a mug 1."
INADMISSIBLE_COMMAND = "take mug 2 from desk 1"


class ScriptedSession:
    """TextWorld's single-game gym API: reset() and step(command)."""

    def __init__(self):
        self.commands = []

    def reset(self):
        return INITIAL, {"admissible_commands": list(ADMISSIBLE), "won": False}

    def step(self, command):
        self.commands.append(command)
        infos = {"admissible_commands": list(ADMISSIBLE), "won": False}
        if command == "go to desk 1":
            return ARRIVE, 0.0, False, infos
        return "Nothing happens.", 0.0, False, infos


# The agent's replies. Each Thought is a sentence no observation contains,
# so its absence from later messages can be checked.
THOUGHTS = ["Walking over to the writing desk is my opening move.",
            "Grabbing the second mug should finish it.",
            "Hovering above the furniture might reveal more.",
            "Peering about using a misnamed argument key."]
NOT_REACT = "Let me just glance around the room quietly for now."
REPLIES = [
    react_action("act", {"command": "go to desk 1"}, thought=THOUGHTS[0]),        # step 1: admissible
    react_action("act", {"command": INADMISSIBLE_COMMAND}, thought=THOUGHTS[1]),  # step 2: inadmissible
    NOT_REACT,                                                                    # step 3: format retry
    react_action("fly", {"altitude_meters": 3}, thought=THOUGHTS[2]),             # step 3: unknown tool
    react_action("act", {"cmd": "look"}, thought=THOUGHTS[3]),                    # step 4: invalid arguments
    react_final("the mug sits on the desk", thought="Stopping the scripted episode now."),  # step 5
]
# (step of the call that produced the reply, the texts of that reply)
REPLY_TEXTS = [
    (1, [THOUGHTS[0], "Action: act", 'Action Input: {"command": "go to desk 1"}',
         '{"command": "go to desk 1"}']),
    (2, [THOUGHTS[1], "Action: act", f'{{"command": "{INADMISSIBLE_COMMAND}"}}', INADMISSIBLE_COMMAND]),
    (3, [NOT_REACT]),
    (3, [THOUGHTS[2], "Action: fly", '{"altitude_meters": 3}', "altitude_meters"]),
    (4, [THOUGHTS[3], "Action: act", '{"cmd": "look"}']),
]

SYSTEM_PROMPT_HEAD = """\
You are a ReAct agent. Solve the task step by step using the available tools.
Respond with exactly ONE step in one of these formats:

Thought: <one sentence of reasoning>
Action: <tool name>
Action Input: <a single JSON object with the tool's arguments>

or, when the task is complete:

Thought: <one sentence of reasoning>
Final Answer: <the final answer>

Available tools:
- act: Send one text command to the household game, for example 'go to fridge 1' or 'take apple 1 from countertop 1'.
  Action Input JSON Schema: {"additionalProperties": false, "properties": {"command": {"minLength": 1, "type": "string"}}, "required": ["command"], "type": "object"}

Expressed skills (use when applicable):
"""
SYSTEM_G0 = SYSTEM_PROMPT_HEAD + "No skills are currently expressed."

USER_STEP_1 = """\
Task: You are in the middle of a room. Looking quickly around you, you see a desk 1.

Your task is to: put a mug in desk.

Admissible commands:
go to desk 1
look
take mug 1 from desk 1

Observations so far:
[0] You are in the middle of a room. Looking quickly around you, you see a desk 1.

Your task is to: put a mug in desk.

Admissible commands:
go to desk 1
look
take mug 1 from desk 1

Step 1 of at most 50. Provide the next single step."""

# The message the owner's decision names: step 2, after one admissible command.
USER_STEP_2 = """\
Task: You are in the middle of a room. Looking quickly around you, you see a desk 1.

Your task is to: put a mug in desk.

Admissible commands:
go to desk 1
look
take mug 1 from desk 1

Observations so far:
[0] You are in the middle of a room. Looking quickly around you, you see a desk 1.

Your task is to: put a mug in desk.

Admissible commands:
go to desk 1
look
take mug 1 from desk 1
[1] You arrive at desk 1. On the desk 1, you see a mug 1.

Admissible commands:
go to desk 1
look
take mug 1 from desk 1

Step 2 of at most 50. Provide the next single step."""

# After the inadmissible command: "Nothing happens.", without the command.
USER_STEP_3 = """\
Task: You are in the middle of a room. Looking quickly around you, you see a desk 1.

Your task is to: put a mug in desk.

Admissible commands:
go to desk 1
look
take mug 1 from desk 1

Observations so far:
[0] You are in the middle of a room. Looking quickly around you, you see a desk 1.

Your task is to: put a mug in desk.

Admissible commands:
go to desk 1
look
take mug 1 from desk 1
[1] You arrive at desk 1. On the desk 1, you see a mug 1.

Admissible commands:
go to desk 1
look
take mug 1 from desk 1
[2] Nothing happens.

Admissible commands:
go to desk 1
look
take mug 1 from desk 1

Step 3 of at most 50. Provide the next single step."""

FORMAT_CORRECTION = ("Your previous reply was not valid ReAct format. "
                     "Reply again using exactly one required format.")
ENTRY_UNKNOWN_TOOL = "rejected tool call 'fly': unknown tool 'fly': not in the offered tool catalog"
ENTRY_INVALID_ARGUMENTS = ("rejected tool call 'act': 'command' is a required property; "
                           "Additional properties are not allowed ('cmd' was unexpected)")

# The arm-B/C form: one assimilated skill gene, constitutive (no regulation).
SKILL_GENE_ID = "pin_search_first_v1"
SKILL_PAYLOAD = {"name": "search likely places first",
                 "principle": "Look in the receptacles where the object is most likely to be.",
                 "when_to_apply": "household tasks that need an object you cannot see yet",
                 "procedure": [{"id": "S1", "text": "list the receptacles you can see"},
                               {"id": "S2", "text": "open the closed ones before moving on"}]}
SKILL_BLOCK = """\
- search likely places first: Look in the receptacles where the object is most likely to be. (apply when: household tasks that need an object you cannot see yet)
    S1: list the receptacles you can see
    S2: open the closed ones before moving on"""

G0_GENOME_ID = "G-9c4a9846a9f50260"
# sha256 of json.dumps(<the six calls' (step, messages)>, sort_keys, ensure_ascii=False)
EPISODE_MESSAGES_SHA256 = "63aa134310f48c9f8818755c894174a3b07c0b35083a62d956540fe73cdf53d5"


def run_episode(config, session=None, replies=REPLIES):
    """Plays a scripted episode (by default the six-call one); returns the
    session and each model call as (step, [(role, content), ...])."""
    session = ScriptedSession() if session is None else session
    env = AlfworldEnvironment(session, game_id=GAME_ID)
    task = env.start()
    result = run_agent(config=config, model=ScriptedAdapter(list(replies)), env=env, task=task,
                       budgets=Budgets(max_steps=EPISODE_MAX_STEPS, max_total_retries=10,
                                       max_tokens_per_request=512),
                       sleep=lambda seconds: None)
    assert result.status == "finished", result.status
    calls = [(event.step, [(m["role"], m["content"]) for m in event.data["messages"]])
             for event in result.events.of_kind("model_called")]
    return session, calls


@pytest.fixture
def g0_episode(g0_config):
    return run_episode(g0_config)


def test_the_scripted_episode_takes_the_planned_path(g0_episode):
    session, calls = g0_episode
    path = ("the scripted episode left its planned path; the other tests index "
            "its calls by that path, so fix the script, not the literals")
    assert session.commands == ["go to desk 1", INADMISSIBLE_COMMAND], path
    assert [step for step, _ in calls] == [1, 2, 3, 3, 4, 5], path
    assert [len(messages) for _, messages in calls] == [2, 2, 2, 4, 2, 2], path


SYSTEM_MESSAGE = ("ADR-0003 D1 (one tool, 'act', and its schema) and D2 (the "
                  "observation rendering): G0's system message is part of the "
                  "frozen G0 prompt; a failure here need not be about note N1")


def test_g0_system_message_has_one_tool_and_no_skill(g0_episode):
    _, calls = g0_episode
    for _, messages in calls:
        assert messages[0] == ("system", SYSTEM_G0), SYSTEM_MESSAGE


def test_step_1_user_message_is_the_task_and_observation_0(g0_episode):
    _, calls = g0_episode
    assert calls[0][1][1] == ("user", USER_STEP_1), DECISION


def test_step_2_user_message_after_an_admissible_command_shows_only_observations(g0_episode):
    _, calls = g0_episode
    assert calls[1][1][1] == ("user", USER_STEP_2), DECISION


def test_step_3_user_message_after_an_inadmissible_command_shows_nothing_happens(g0_episode):
    _, calls = g0_episode
    assert calls[2][1][1] == ("user", USER_STEP_3), DECISION


def test_the_inadmissible_command_sent_appears_in_no_later_call(g0_episode):
    _, calls = g0_episode
    assert INADMISSIBLE_COMMAND not in ADMISSIBLE
    for step, messages in calls[2:]:
        for role, content in messages:
            assert INADMISSIBLE_COMMAND not in content, (DECISION, step, role)


def test_the_agents_own_replies_appear_in_no_later_steps_messages(g0_episode):
    _, calls = g0_episode
    for reply_step, texts in REPLY_TEXTS:
        for step, messages in calls:
            if step <= reply_step:
                continue
            for role, content in messages:
                for text in texts:
                    assert text not in content, (DECISION, text, step, role)


def test_a_format_retry_shows_the_rejected_reply_only_inside_its_own_step(g0_episode):
    """Exception (i): the retried call carries the rejected reply and the
    correction; the next step's call carries neither."""
    _, calls = g0_episode
    assert calls[3] == (3, [("system", SYSTEM_G0),
                            ("user", USER_STEP_3),
                            ("assistant", NOT_REACT),
                            ("user", FORMAT_CORRECTION)]), DECISION
    for _, messages in calls[4:]:
        assert [role for role, _ in messages] == ["system", "user"]
        for _, content in messages:
            assert NOT_REACT not in content and FORMAT_CORRECTION not in content, DECISION


def test_rejected_tool_calls_show_the_tool_name_and_schema_errors_only(g0_episode):
    """Exception (ii): the loop's rejection text names the tool and, for
    invalid arguments, the schema errors."""
    _, calls = g0_episode
    tail_3 = "\n\nStep 3 of at most 50. Provide the next single step."
    assert USER_STEP_3.endswith(tail_3)
    head = USER_STEP_3[:-len(tail_3)]
    user_step_4 = (head + f"\n[3] {ENTRY_UNKNOWN_TOOL}"
                   + "\n\nStep 4 of at most 50. Provide the next single step.")
    user_step_5 = (head + f"\n[3] {ENTRY_UNKNOWN_TOOL}\n[4] {ENTRY_INVALID_ARGUMENTS}"
                   + "\n\nStep 5 of at most 50. Provide the next single step.")
    assert calls[4] == (4, [("system", SYSTEM_G0), ("user", user_step_4)]), DECISION
    assert calls[5] == (5, [("system", SYSTEM_G0), ("user", user_step_5)]), DECISION


def arm_config_with_one_skill(registry):
    """G0 plus one assimilated, constitutive skill gene, built as arms B and
    C build theirs (evaluation/exp0001/arms.py)."""
    g0 = build_g0_seed_genome(registry)
    artifact = registry.put("skill", SKILL_GENE_ID, 1, dict(SKILL_PAYLOAD))
    envelope = {"candidate": {"gene_id": SKILL_GENE_ID, "version": 1, "artifact": artifact,
                              "provenance": {"born_generation": 1,
                                             "source_trajectory": "trajectory/pin-0001"}}}
    ref = germline_ref(envelope, cig_record="cig/pin-aggregate", arm="B", mining_seed=11)
    return compile_runtime_config(g1_genome(g0, [ref], registry), registry)


def test_one_expressed_skill_changes_only_the_skill_block_of_the_system_message(
        registry, g0_episode):
    config = arm_config_with_one_skill(registry)
    assert config.genome_id != G0_GENOME_ID and len(config.skills) == 1
    _, calls = run_episode(config)
    _, g0_calls = g0_episode
    assert [step for step, _ in calls] == [step for step, _ in g0_calls]
    for (_, messages), (_, g0_messages) in zip(calls, g0_calls):
        assert messages[0] == ("system", SYSTEM_PROMPT_HEAD + SKILL_BLOCK), DECISION
        assert messages[1:] == g0_messages[1:], DECISION


def test_g0_genome_id_is_the_pilots(g0_config):
    """No other test pins this id (searched 2026-11-12)."""
    assert g0_config.genome_id == G0_GENOME_ID, (
        "G0's genome id differs from the calibration pilot's G0, on which "
        "ADR-0012's calibration rests; any change of G0's genome content "
        "changes this id, not only a change of the prompt")


def test_the_whole_episodes_messages_hash_is_unchanged(g0_episode):
    _, calls = g0_episode
    blob = json.dumps([[step, [list(m) for m in messages]] for step, messages in calls],
                      sort_keys=True, ensure_ascii=False)
    assert hashlib.sha256(blob.encode("utf-8")).hexdigest() == EPISODE_MESSAGES_SHA256, DECISION


def test_a_rejected_tool_calls_error_text_can_quote_an_arguments_value(g0_config):
    """Exception (ii), the value case: a command that is not a string is
    quoted in the entry (jsonschema 4.26.0 words it so)."""
    replies = [react_action("act", {"command": 3}, thought="Sending a number as the command."),
               react_final("done", thought="Stopping the scripted episode now.")]
    session, calls = run_episode(g0_config, replies=replies)
    assert session.commands == []
    head = USER_STEP_1[:-len("\n\nStep 1 of at most 50. Provide the next single step.")]
    user_step_2 = (head + "\n[1] rejected tool call 'act': 3 is not of type 'string'"
                   + "\n\nStep 2 of at most 50. Provide the next single step.")
    assert calls[1] == (2, [("system", SYSTEM_G0), ("user", user_step_2)]), DECISION


# A second scripted run with a long history: 13 admissible steps, each with
# its own command, observation and admissible commands, and one observation
# longer than 6,000 characters. The six-call episode above has at most five
# short entries, so a change that bites only on a long history (a window of
# the last entries, a cut entry, a padded index) passes it.
LONG_TASK_TEXT = ("You are in the middle of a room. Looking quickly around you, you see a desk 1, "
                  "a drawer 1, a drawer 2 and a shelf 1.\n\n"
                  "Your task is to: examine the pencil with the desklamp.")
LONG_INITIAL = "-= Welcome to TextWorld, ALFRED! =-\n\n" + LONG_TASK_TEXT
LONG_ADMISSIBLE_0 = ["go to shelf 1", "go to drawer 1", "look", "go to drawer 2", "inventory",
                     "go to desk 1"]
LONG_OBSERVATION = ("You open the drawer 1. The drawer 1 is open. In it, you see "
                    + ", ".join(f"a paper {n}" for n in range(1, 601)) + ".")
# (command, observation, admissible commands after it), in the game's order
LONG_STEPS = [
    ("go to drawer 1", "You arrive at drawer 1. The drawer 1 is closed.",
     ["open drawer 1", "go to shelf 1", "look", "go to drawer 2", "inventory"]),
    ("open drawer 1", LONG_OBSERVATION,
     ["close drawer 1", "take paper 1 from drawer 1", "look", "go to drawer 2"]),
    ("close drawer 1", "You close the drawer 1.",
     ["open drawer 1", "go to drawer 2", "look", "inventory"]),
    ("go to drawer 2", "You arrive at drawer 2. The drawer 2 is closed.",
     ["open drawer 2", "go to drawer 1", "go to shelf 1", "look"]),
    ("open drawer 2", "You open the drawer 2. The drawer 2 is open. In it, you see a pencil 1 and a pen 1.",
     ["take pencil 1 from drawer 2", "take pen 1 from drawer 2", "close drawer 2", "look"]),
    ("take pencil 1 from drawer 2", "You pick up the pencil 1 from the drawer 2.",
     ["put pencil 1 in/on drawer 2", "examine pencil 1", "go to desk 1", "inventory"]),
    ("inventory", "You are carrying: a pencil 1.",
     ["go to shelf 1", "go to desk 1", "examine pencil 1", "look"]),
    ("go to shelf 1", "You arrive at shelf 1. On the shelf 1, you see a book 1 and a vase 1.",
     ["take book 1 from shelf 1", "take vase 1 from shelf 1", "examine shelf 1", "go to desk 1"]),
    ("examine shelf 1", "You look closely at the shelf 1. A book 1 leans on a vase 1.",
     ["go to desk 1", "go to drawer 1", "look", "inventory"]),
    ("go to desk 1", "You arrive at desk 1. On the desk 1, you see a desklamp 1 and a keychain 1.",
     ["use desklamp 1", "take keychain 1 from desk 1", "examine desk 1", "look"]),
    ("look", "You are facing the desk 1. Next to it, you see nothing.",
     ["use desklamp 1", "examine pencil 1", "go to shelf 1", "inventory"]),
    ("examine pencil 1", "There's nothing special about pencil 1.",
     ["use desklamp 1", "put pencil 1 in/on desk 1", "look", "inventory"]),
    ("use desklamp 1", "You turn on the desklamp 1.",
     ["examine pencil 1", "put pencil 1 in/on desk 1", "look", "inventory"]),
]
LONG_REPLIES = ([react_action("act", {"command": command}) for command, _, _ in LONG_STEPS]
                + [react_final("the pencil is examined under the lit desklamp")])
# sha256 of the last call's user message (UTF-8)
LONG_LAST_USER_MESSAGE_SHA256 = "99349f3c840a9643cce94921ac1ce5e588e7e3e158104c53f45c592f4ef9fd36"


class LongScriptedSession:
    """The long run's game: each command has its own observation and
    admissible commands; records what it returned."""

    def __init__(self):
        self.commands = []
        self.returned = []  # (observation text, admissible commands), the reset's first

    def reset(self):
        self.returned.append((LONG_INITIAL, list(LONG_ADMISSIBLE_0)))
        return LONG_INITIAL, {"admissible_commands": list(LONG_ADMISSIBLE_0), "won": False}

    def step(self, command):
        command_, observation, admissible = LONG_STEPS[len(self.commands)]
        assert command == command_, (command, command_)
        self.commands.append(command)
        self.returned.append((observation, list(admissible)))
        return observation, 0.0, False, {"admissible_commands": list(admissible), "won": False}


def rendered(text, admissible):
    """ADR-0003 D2 by hand: the text, a blank line, the header, the sorted
    commands one per line."""
    return text + "\n\nAdmissible commands:\n" + "\n".join(sorted(admissible))


@pytest.fixture
def long_episode(g0_config):
    return run_episode(g0_config, session=LongScriptedSession(), replies=LONG_REPLIES)


def long_expected_user_message(session, step):
    """The user message of ``step``, built from the task text and the
    observations the scripted session returned, not from the code under test."""
    assert session.returned[0][0] == LONG_INITIAL
    task = rendered(LONG_TASK_TEXT, session.returned[0][1])
    entries = [task] + [rendered(text, admissible) for text, admissible in session.returned[1:step]]
    return ("Task: " + task + "\n\nObservations so far:\n"
            + "\n".join(f"[{i}] {entry}" for i, entry in enumerate(entries))
            + f"\n\nStep {step} of at most 50. Provide the next single step.")


def test_the_long_run_takes_its_planned_path(long_episode):
    session, calls = long_episode
    path = "the long scripted run left its planned path; fix the script, not the literals"
    assert session.commands == [command for command, _, _ in LONG_STEPS], path
    assert len(session.commands) >= 12 and len(set(session.commands)) == len(session.commands), path
    assert len({text for text, _ in session.returned}) == len(session.returned), path
    assert max(len(text) for text, _ in session.returned) > 6000, path
    assert [step for step, _ in calls] == list(range(1, len(LONG_STEPS) + 2)), path
    assert [len(messages) for _, messages in calls] == [2] * (len(LONG_STEPS) + 1), path


def test_the_long_runs_last_user_message_shows_every_observation_in_full(long_episode):
    session, calls = long_episode
    step, messages = calls[-1]
    assert step == len(LONG_STEPS) + 1
    expected = long_expected_user_message(session, step)
    assert messages == [("system", SYSTEM_G0), ("user", expected)], DECISION
    digest = hashlib.sha256(messages[1][1].encode("utf-8")).hexdigest()
    assert digest == LONG_LAST_USER_MESSAGE_SHA256, DECISION


def test_every_step_of_the_long_run_shows_the_observations_so_far(long_episode):
    session, calls = long_episode
    for step, messages in calls:
        assert messages[1] == ("user", long_expected_user_message(session, step)), (DECISION, step)
