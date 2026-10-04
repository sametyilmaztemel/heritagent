"""Mining audit record (issue #9 criterion 6).

One immutable, schema-validated record per teacher call. This is MINING
evidence — it makes the extraction reproducible and auditable. It is NOT CIG
evidence: no causal claims and no cig_record exist at this stage. The
successful trajectory id doubles as the primary somatic
`provenance.source_trajectory`; the full differential source set lives here.
"""

from __future__ import annotations

import json
from pathlib import Path

from jsonschema import Draft202012Validator

from trajectory.recorder.errors import RecorderError

_SCHEMA_PATH = Path(__file__).resolve().parents[1] / "schemas" / "mining-record-0.1.schema.json"
_VALIDATOR: Draft202012Validator | None = None


def mining_record_validator() -> Draft202012Validator:
    global _VALIDATOR
    if _VALIDATOR is None:
        schema = json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))
        Draft202012Validator.check_schema(schema)
        _VALIDATOR = Draft202012Validator(schema)
    return _VALIDATOR


def build_mining_record(*, mining_record_id: str, success_trajectory_ids: list[str],
                         failure_trajectory_ids: list[str], task_families: list[str],
                         evidence_sha256: str, prompt_template_sha256: str,
                         output_schema_sha256: str, mining_seed: int,
                         generation_settings: dict, model_metadata: dict,
                         raw_structured_response: dict | None,
                         accepted_proposals: list[dict], rejected: list[dict],
                         discovery_order: list[str]) -> dict:
    record = {
        "record": "mining",
        "schema_version": "0.1",
        "mining_record_id": mining_record_id,
        "success_trajectory_ids": list(success_trajectory_ids),
        "failure_trajectory_ids": list(failure_trajectory_ids),
        "task_families": list(task_families),
        "evidence_sha256": evidence_sha256,
        "prompt_template_sha256": prompt_template_sha256,
        "output_schema_sha256": output_schema_sha256,
        "mining_seed": mining_seed,
        "generation_settings": dict(generation_settings),
        "model_metadata": dict(model_metadata),
        "raw_structured_response": raw_structured_response,
        "accepted_proposals": list(accepted_proposals),
        "rejected": list(rejected),
        "discovery_order": list(discovery_order),
    }
    errors = [e.message for e in mining_record_validator().iter_errors(record)]
    if errors:
        raise RecorderError(f"mining record violates its schema: {sorted(errors)}")
    return record


__all__ = ["build_mining_record", "mining_record_validator"]
