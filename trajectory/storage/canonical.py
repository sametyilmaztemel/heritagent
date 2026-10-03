"""Deterministic canonical serialization + integrity digest primitives.

Digest domain (v0.1, locked):
    content_sha256 = sha256(canonical_json({
        "kind": "heritagent-trajectory",
        "schema_version": "0.1",
        "trajectory_id": <stable opaque id, allocated at recorder creation>,
        "context": <header context dict>,
        "events": [<ordered event record dicts>],
        "finalization": {"status": ..., "outcome": ..., "incomplete_reason": ...},
    }))
The trajectory id is INTEGRITY-BOUND (inside the domain) but is NOT derived
from the digest: it is allocated once at recorder creation, written to the
header, and survives crash recovery and post-run outcome changes. Any
change to the id, the context, any event payload/order, or the finalization
metadata changes the digest. The digest itself is stored in the
finalization record and therefore is not part of its own input.

Canonical JSON is strict: `allow_nan=False` — non-finite floats
(NaN/Infinity) are rejected at serialization time.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

SCHEMA_VERSION = "0.1"
DIGEST_DOMAIN_KIND = "heritagent-trajectory"


def canonical_json(obj: Any) -> str:
    """Deterministic canonical JSON: UTF-8, sorted keys, compact stable
    separators, non-ASCII preserved, non-finite floats rejected."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


def canonical_bytes(obj: Any) -> bytes:
    return canonical_json(obj).encode("utf-8")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def content_digest(trajectory_id: str, context: dict, events: list[dict],
                   finalization: dict) -> str:
    """Deterministic integrity digest over id + header + ordered events +
    finalization metadata (see module docstring for the locked domain)."""
    return sha256_hex(canonical_bytes({
        "kind": DIGEST_DOMAIN_KIND,
        "schema_version": SCHEMA_VERSION,
        "trajectory_id": trajectory_id,
        "context": context,
        "events": events,
        "finalization": finalization,
    }))
