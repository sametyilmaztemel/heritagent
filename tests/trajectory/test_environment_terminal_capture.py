"""An environment-ended episode (ADR-0003 D5) records and finalizes as a
complete, schema-valid trajectory with status ``environment_terminal``."""

import json
from pathlib import Path

from jsonschema import Draft202012Validator

from runtime.loop import ToolObservation
from tests.runtime.conftest import react_action
from tests.trajectory.conftest import run_captured


def test_environment_terminal_trajectory_finalizes_schema_valid(tmp_path, registry, context):
    result, recorder, path, env, adapter = run_captured(
        tmp_path, registry, context,
        script=[react_action("act", {"command": "put egg in fridge"})],
        env_results={"act": [ToolObservation(ok=True, content="done", terminal=True)]})
    finalized = recorder.finalize()
    assert result.status == finalized.status == "environment_terminal"
    assert finalized.document["complete"] is True
    schema = json.loads(Path("trajectory/schema/trajectory-0.1.schema.json").read_text())
    Draft202012Validator(schema).validate(finalized.document)
    assert finalized.document["events"][-1]["kind"] == "environment_terminal"
