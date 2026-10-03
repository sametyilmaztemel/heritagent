"""Deterministic canonical serialization + integrity digest primitives.

Digest domain (v0.1, locked):
    content_sha256 = sha256(canonical_json({
        "kind": "heritagent-trajectory",
        "schema_version": "0.1",
        "context": <header context dict>,
        "events": [<ordered event record dicts>],
        "finalization": {"status": ..., "outcome": ..., "incomplete_reason": ...},
    }))
Any change to the context, any event payload/order, or the finalization
metadata changes the digest. The digest itself is stored in the
finalization record and therefore is not part of its own input.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

SCHEMA_VERSION = "0.1"
DIGEST_DOMAIN_KIND = "heritagent-trajectory"


def canonical_json(obj: Any) -> str:
    """Deterministic canonical JSON: UTF-8, sorted keys, compact stable
    separators, non-ASCII preserved (encoded to UTF-8 bytes downstream)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def canonical_bytes(obj: Any) -> bytes:
    return canonical_json(obj).encode("utf-8")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def content_digest(context: dict, events: list[dict], finalization: dict) -> str:
    """Deterministic integrity digest over header + ordered events +
    finalization metadata (see module docstring for the locked domain)."""
    return sha256_hex(canonical_bytes({
        "kind": DIGEST_DOMAIN_KIND,
        "schema_version": SCHEMA_VERSION,
        "context": context,
        "events": events,
        "finalization": finalization,
    }))


def trajectory_id_from_digest(digest: str) -> str:
    """Stable opaque trajectory id: `T-<first 12 digest hex chars>`. v0.1
    deliberately does NOT define full canonical trajectory identity beyond
    this (see handoff criterion 6)."""
    return f"T-{digest[:12]}"
