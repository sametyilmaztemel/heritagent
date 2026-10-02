"""Content-addressed trait registry (ADR-0001 D6, D8).

Artifacts are stored once, keyed by the SHA-256 of their canonical bytes.
The canonical form of a dict payload is compact JSON with sorted keys and
UTF-8 encoding; byte payloads are stored verbatim. Artifact URIs follow the
canonical grammar::

    registry://<kind>/<gene_id>@<version>/sha256:<64-hex>

with kind in {policies, skills, regulators}. Resolution re-verifies that the
stored bytes hash to the digest in the URI (immutable-by-digest). A binding
index additionally pins each ``(kind, gene_id, version)`` identity to the
digest it was first stored with: rebinding an identity to different content
is rejected — artifact changes require a version increment (ADR-0001 D6).
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from genome.validation.errors import GenomeValidationError, RegistryIntegrityError

KIND_BY_TYPE = {"policy": "policies", "skill": "skills", "regulatory": "regulators"}

URI_PATTERN = re.compile(
    r"^registry://(?P<kind>policies|skills|regulators)/"
    r"(?P<name>[a-z0-9_]+(?:_v[0-9]+)?)@(?P<version>[1-9][0-9]*)/"
    r"sha256:(?P<digest>[0-9a-f]{64})$"
)

INDEX_FILE = "index.json"


def canonical_bytes(payload) -> bytes:
    """Canonical serialization used for content addressing."""
    if isinstance(payload, bytes):
        return payload
    if isinstance(payload, str):
        return payload.encode("utf-8")
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def parse_uri(uri: str) -> dict:
    """Parse a registry URI; raises GenomeValidationError on grammar violations."""
    match = URI_PATTERN.match(uri or "")
    if not match:
        raise GenomeValidationError(f"artifact URI does not match canonical grammar "
                                    f"registry://<kind>/<name>@<version>/sha256:<64-hex>: {uri!r}")
    return match.groupdict()


class TraitRegistry:
    """Append-only, content-addressed store for gene payload artifacts."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._bindings: dict[str, str] | None = None  # lazy (kind, gene_id, version) -> digest

    def _index(self) -> dict[str, str]:
        if self._bindings is None:
            index_path = self.root / INDEX_FILE
            self._bindings = json.loads(index_path.read_text()) if index_path.exists() else {}
        return self._bindings

    def _save_index(self) -> None:
        index_path = self.root / INDEX_FILE
        tmp = index_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._index(), sort_keys=True, indent=1))
        tmp.replace(index_path)

    def put(self, gene_type: str, gene_id: str, version: int, payload) -> str:
        """Store a payload and return its canonical artifact URI.

        Rejects rebinding an existing ``(kind, gene_id, version)`` identity
        to different content (ADR-0001 D6: changes require a version bump).
        """
        try:
            kind = KIND_BY_TYPE[gene_type]
        except KeyError:
            raise GenomeValidationError(f"unknown gene type {gene_type!r}; "
                                        f"expected one of {sorted(KIND_BY_TYPE)}") from None
        data = canonical_bytes(payload)
        digest = hashlib.sha256(data).hexdigest()
        binding = f"{kind}/{gene_id}@{version}"
        bound = self._index().get(binding)
        if bound is not None and bound != digest:
            raise RegistryIntegrityError(
                f"{binding} is already bound to sha256:{bound}; refusing to rebind to "
                f"sha256:{digest} — identity changes require a version increment (ADR-0001 D6)")
        target = self.root / kind / digest
        if target.exists():
            if target.read_bytes() != data:
                raise RegistryIntegrityError(
                    f"digest collision for {digest}: existing artifact differs from new payload")
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_suffix(".tmp")
            tmp.write_bytes(data)
            tmp.replace(target)
        if bound is None:
            self._index()[binding] = digest
            self._save_index()
        return f"registry://{kind}/{gene_id}@{version}/sha256:{digest}"

    def resolve(self, uri: str) -> bytes:
        """Return stored bytes, verifying the stored content matches the URI digest."""
        parts = parse_uri(uri)
        path = self.root / parts["kind"] / parts["digest"]
        if not path.exists():
            raise GenomeValidationError(f"artifact not found in registry: {uri!r}")
        data = path.read_bytes()
        actual = hashlib.sha256(data).hexdigest()
        if actual != parts["digest"]:
            raise RegistryIntegrityError(
                f"stored artifact for {parts['digest']} hashes to {actual}; registry content was tampered with")
        return data

    def verify_binding(self, uri: str) -> None:
        """Verify the URI digest agrees with the identity binding (defense in depth
        against hand-crafted URIs that point a known identity at other content)."""
        parts = parse_uri(uri)
        binding = f"{parts['kind']}/{parts['name']}@{parts['version']}"
        bound = self._index().get(binding)
        if bound is not None and bound != parts["digest"]:
            raise RegistryIntegrityError(
                f"{binding} is bound to sha256:{bound}; URI claims sha256:{parts['digest']} "
                f"(ADR-0001 D6: artifact changes require a version increment)")

    def check_ref(self, gene_ref: dict) -> dict:
        """Semantic checks for a geneRef against this registry.

        Verifies URI grammar (defense in depth beyond the schema pattern),
        name == gene_id, version == geneRef.version, kind == geneRef.type,
        that the stored bytes hash to the URI digest, and that the identity
        binding agrees with the URI digest.
        """
        uri = gene_ref.get("artifact", "")
        parts = parse_uri(uri)
        errors = []
        if parts["name"] != gene_ref.get("gene_id"):
            errors.append(f"artifact URI name {parts['name']!r} != gene_id {gene_ref.get('gene_id')!r}")
        if int(parts["version"]) != gene_ref.get("version"):
            errors.append(f"artifact URI version {parts['version']} != gene version {gene_ref.get('version')}")
        expected_kind = KIND_BY_TYPE.get(gene_ref.get("type"))
        if parts["kind"] != expected_kind:
            errors.append(f"artifact URI kind {parts['kind']!r} does not match gene type "
                          f"{gene_ref.get('type')!r} (expected {expected_kind!r})")
        if errors:
            raise GenomeValidationError(errors)
        self.resolve(uri)  # raises on tampering / digest mismatch / missing artifact
        self.verify_binding(uri)  # raises on identity rebinding
        return parts
