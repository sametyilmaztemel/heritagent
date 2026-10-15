"""The EXP-0001 run-start record (issue #13 Phase A; SPEC section 24;
manifest ``required_at_run_time``).

Frozen before the first real episode and written once: a second write with
different content fails closed. It names everything a result must be
reported with (SPEC: seed, model revision, genome revision, dataset split,
code commit): the code commit, the model identity and weights hash, the
split, pilot and pool hashes, the miner prompt and schema hashes, G0's
canonical genome id and digest, the candidate cap and ranking rule, the CIG
threshold version and values, both bootstrap rules, the generation settings
and stop, the episode budget, the hard caps and the mining seeds, and the
miner's teacher model. The split and pilot manifests must verify from
their own hashes and equal the pre-registered ones."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

from evaluation.exp0001.protocol import (
    LOCKED_PILOT_SHA256,
    LOCKED_SPLIT_MANIFEST_SHA256,
    PROTOCOL,
)
from evaluation.metrics.final_statistics import FINAL_RULE

RUN_START_SCHEMA = "heritagent/exp0001-run-start/0.1"
REQUIRED_AT_RUN_TIME = ("model-weights-hash", "code-commit", "split-hashes", "miner-prompt-hash",
                        "frozen-candidate-set-artifact", "gate-thresholds", "run-log")


class ProvenanceError(ValueError):
    """The run-start record is incomplete, inconsistent or already frozen
    with other content."""


def plain(obj):
    """JSON-ready copy: mappings become dicts, tuples become lists."""
    if isinstance(obj, Mapping):
        return {str(k): plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [plain(v) for v in obj]
    return obj


def canonical_sha256(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode("utf-8")).hexdigest()


def sha256_files(directory: Path, patterns=("*.safetensors", "*.bin", "*.json", "*.model",
                                            "*.txt")) -> dict:
    """Weights hash of a model directory: SHA-256 of every matching file,
    and one digest over the sorted (relative path, file digest) list."""
    directory = Path(directory)
    if not directory.is_dir():
        raise ProvenanceError(f"model directory not found: {directory}")
    files = sorted({p for pattern in patterns for p in directory.rglob(pattern) if p.is_file()})
    if not files:
        raise ProvenanceError(f"no model files under {directory}")
    digests = {}
    for path in files:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
        digests[path.relative_to(directory).as_posix()] = digest.hexdigest()
    return {"files": digests, "sha256": canonical_sha256(sorted(digests.items()))}


def _require_str(value, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProvenanceError(f"{name} must be a non-empty string, got {value!r}")
    return value


def _is_hex(value, length: int) -> bool:
    return isinstance(value, str) and len(value) == length and \
        all(c in "0123456789abcdef" for c in value)


def verify_pre_run_artifacts(split_manifest: dict, pilot_manifest: dict | None, *,
                             expected_split_sha256: str = LOCKED_SPLIT_MANIFEST_SHA256,
                             expected_pilot_sha256: str = LOCKED_PILOT_SHA256) -> None:
    """The split (and pilot) manifest must verify from its own hashes and be
    the committed one (manifest run_protocol); anything else is another
    experiment."""
    from evaluation.benchmarks.alfworld.splits import SplitError, verify_manifest
    from evaluation.exp0001.pilot import verify_pilot_manifest
    try:
        verify_manifest(split_manifest)
        if pilot_manifest is not None:
            verify_pilot_manifest(pilot_manifest, split_manifest)
    except SplitError as exc:
        raise ProvenanceError(f"a pre-run artifact does not verify: {exc}") from exc
    if split_manifest["manifest_sha256"] != expected_split_sha256:
        raise ProvenanceError(f"split manifest {split_manifest['manifest_sha256']} is not the "
                              f"pre-registered one ({expected_split_sha256})")
    if pilot_manifest is not None and pilot_manifest["pilot_sha256"] != expected_pilot_sha256:
        raise ProvenanceError(f"pilot manifest {pilot_manifest['pilot_sha256']} is not the "
                              f"pre-registered one ({expected_pilot_sha256})")


def _model_block(model: Mapping, name: str, *, tokenizer: bool) -> dict:
    keys = ("model_id", "revision", "backend") + (("tokenizer",) if tokenizer else ())
    for key in keys:  # the manifest pins the revision at run time
        _require_str(model.get(key), f"{name}.{key}")
    return {k: model.get(k) for k in ("model_id", "revision", "backend", "backend_version")
            + (("tokenizer",) if tokenizer else ())}


def build_run_start_record(*, code_commit: str, code_dirty: bool, model: Mapping,
                           teacher: Mapping, weights: Mapping, split_manifest: dict,
                           pilot_manifest: dict, g0_genome: dict, run_id: str,
                           started_at: str, environment: Mapping | None = None,
                           expected_split_sha256: str = LOCKED_SPLIT_MANIFEST_SHA256,
                           expected_pilot_sha256: str = LOCKED_PILOT_SHA256) -> dict:
    """Assemble the record. ``model`` is the agent adapter's metadata
    (model_id, revision, backend, backend_version) plus the identity of the
    token counter's tokenizer; ``teacher`` is the miner's adapter metadata;
    ``weights`` is ``sha256_files`` output for the served weights;
    ``environment`` names the installed software stack and the verified data
    (ADR-0008 D4, D6). The split and pilot manifests must verify and be the
    pre-registered ones."""
    from genome.validation.identity import compute_genome_id
    from inheritance.ablation import BOOTSTRAP_CLUSTER_ORDER, BOOTSTRAP_N_RESAMPLES, BOOTSTRAP_SEED
    from inheritance.gate import PRIMARY_THRESHOLDS, THRESHOLD_VERSION
    from inheritance.overlay import evaluation_settings
    from runtime.policies.planner_react import STOP_SEQUENCES
    from runtime.seed_g0 import EPISODE_MAX_STEPS
    from traits.miner.miner import output_schema_sha256, prompt_template_sha256

    if len(code_commit or "") != 40 or any(c not in "0123456789abcdef" for c in code_commit):
        raise ProvenanceError(f"code_commit must be a full 40-hex commit, got {code_commit!r}")
    if not isinstance(code_dirty, bool):
        raise ProvenanceError("code_dirty must be a bool")
    if code_dirty:
        raise ProvenanceError("the run must start from a clean checkout (code_dirty is True)")
    model_block = _model_block(model, "model", tokenizer=True)
    teacher_block = _model_block(teacher, "teacher", tokenizer=False)
    if not isinstance(weights, Mapping) or not _is_hex(weights.get("sha256"), 64):
        raise ProvenanceError("weights must carry the sha256 (64 hex) of the served weights")
    verify_pre_run_artifacts(split_manifest, pilot_manifest,
                             expected_split_sha256=expected_split_sha256,
                             expected_pilot_sha256=expected_pilot_sha256)
    _require_str(run_id, "run_id")
    _require_str(started_at, "started_at")
    if g0_genome.get("genome_id") != compute_genome_id(g0_genome):
        raise ProvenanceError("G0's genome_id is not its canonical content id")

    record = {
        "schema": RUN_START_SCHEMA,
        "experiment": "EXP-0001",
        "run_id": run_id,
        "started_at": started_at,
        "code": {"commit": code_commit, "dirty": False},
        "model": model_block,
        "teacher": teacher_block,
        "weights": {"sha256": weights["sha256"], "files": dict(weights.get("files", {}))},
        "splits": {"manifest_sha256": split_manifest["manifest_sha256"],
                   "plan": split_manifest["plan"],
                   "pools": {name: pool["sha256"] for name, pool in split_manifest["pools"].items()}},
        "pilot": {"pilot_sha256": pilot_manifest["pilot_sha256"]},
        "miner": {"prompt_sha256": prompt_template_sha256(),
                  "output_schema_sha256": output_schema_sha256()},
        "g0": {"genome_id": g0_genome["genome_id"], "genome_sha256": canonical_sha256(g0_genome)},
        "candidate_set": {"cap": PROTOCOL["candidate_cap"],
                          "ranking": list(PROTOCOL["candidate_ranking"])},
        "gate": {"threshold_version": THRESHOLD_VERSION, "thresholds": dict(PRIMARY_THRESHOLDS),
                 "ablation_bootstrap": {"seed": BOOTSTRAP_SEED, "resamples": BOOTSTRAP_N_RESAMPLES,
                                        "cluster_order": BOOTSTRAP_CLUSTER_ORDER}},
        "final_statistics": plain(FINAL_RULE),
        "generation": {"eval": _settings(evaluation_settings(
                           max_tokens=PROTOCOL["budgets"]["max_tokens_per_request"], seed=None)),
                       "mining_temperature": PROTOCOL["mining_temperature"],
                       "stop": list(STOP_SEQUENCES),
                       "episode_max_steps": EPISODE_MAX_STEPS},
        "protocol": plain(PROTOCOL),
        "required_at_run_time": list(REQUIRED_AT_RUN_TIME),
        "environment": None if environment is None else plain(environment),
    }
    record["record_sha256"] = canonical_sha256(record)
    return record


def _settings(settings) -> dict:
    return {"temperature": settings.temperature, "max_tokens": settings.max_tokens,
            "stop": list(settings.stop)}


def verify_run_start_record(record: dict) -> None:
    body = {k: v for k, v in record.items() if k != "record_sha256"}
    if canonical_sha256(body) != record.get("record_sha256"):
        raise ProvenanceError("record_sha256 does not match the run-start record")


def freeze_json(path: Path, document) -> dict:
    """Write a run artifact once. Writing the identical content again is a
    no-op (a resumed run); any other content fails closed."""
    document = plain(document)
    path = Path(path)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != document:
            raise ProvenanceError(f"{path} is frozen with other content; a frozen run artifact "
                                  f"is never replaced")
        return existing
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n",
                   encoding="utf-8")
    tmp.replace(path)
    return document


def freeze_run_start_record(path: Path, record: dict) -> dict:
    """Write the record once. Writing the identical record again is a no-op
    (a resumed run); any other content fails closed."""
    verify_run_start_record(record)
    path = Path(path)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != record:
            raise ProvenanceError(f"{path} already holds another run-start record "
                                  f"({existing.get('record_sha256')})")
        return existing
    return freeze_json(path, record)
