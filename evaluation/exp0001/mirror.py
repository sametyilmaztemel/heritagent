"""A crash-consistent mirror of an EXP-0001 run directory (ADR-0008).

On Colab the run works in a local directory, which is lost with the VM;
snapshots of it go to a mirror directory on Google Drive. A snapshot is
taken in the run's own process right after a ledger record is appended,
when no run file is half-written. So it holds a state the run could also
have crashed in, which a resume already handles (the pipeline's
write-ahead order). Files in flight in worker processes (the trajectory of
an unrecorded episode) may be caught half-written; a resume replaces them.

Every file is stored once under its content hash
(``blobs/<2 hex>/<sha256>``). A snapshot is a numbered manifest
(``snapshot-<seq>.json``) of path -> hash, written after its blobs, so a
crash during a snapshot leaves the previous one whole. A restore takes the
newest snapshot whose blobs all exist, checks every file against its hash,
and records itself as a ``restore`` note in the ledger.

Work done after the last snapshot is lost with the VM: its episodes run
again after a restore, and the model calls and active time they took are
not counted. The restore notes keep every such gap on record (ADR-0008)."""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import sys
import time
from collections.abc import Callable, Mapping
from pathlib import Path, PurePosixPath

from evaluation.exp0001.ledger import TERMINAL, append_note

MIRROR_SCHEMA = "heritagent/exp0001-mirror/0.1"
SKIPPED_NAMES = (".lock",)
KEEP_SNAPSHOTS = 2
GC_EVERY = 20  # snapshots between blob collections (listing blobs is the slow part)
OWNER_FILE = "owner.json"
_SNAPSHOT = re.compile(r"^snapshot-(\d{8})\.json$")


class MirrorError(RuntimeError):
    """The mirror cannot be used safely (no whole snapshot, a hash
    mismatch, a non-empty restore target, another writer)."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _blob(mirror: Path, digest: str) -> Path:
    return mirror / "blobs" / digest[:2] / digest


def _write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with tmp.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _local_files(local: Path) -> list[Path]:
    return sorted(p for p in local.rglob("*")
                  if p.is_file() and p.name not in SKIPPED_NAMES
                  and not p.name.endswith(".tmp") and ".tmp-" not in p.name)


def _snapshot_paths(mirror: Path) -> list[tuple[int, Path]]:
    """Snapshot manifests, newest first."""
    if not mirror.is_dir():
        return []
    found = []
    for path in mirror.iterdir():
        match = _SNAPSHOT.match(path.name)
        if match:
            found.append((int(match.group(1)), path))
    return sorted(found, reverse=True)


def _read_snapshot(seq: int, path: Path) -> dict | None:
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(doc, dict) or doc.get("schema") != MIRROR_SCHEMA or doc.get("seq") != seq \
            or not isinstance(doc.get("files"), dict):
        return None
    return doc


def _blob_names(mirror: Path) -> set[str]:
    """Every blob in the mirror, from one listing per prefix directory (a
    Drive round trip per directory, not per file)."""
    root = mirror / "blobs"
    if not root.is_dir():
        return set()
    return {blob.name for prefix in root.iterdir() if prefix.is_dir()
            for blob in prefix.iterdir() if not blob.name.startswith(".")}


def latest_snapshot(mirror_dir: Path) -> dict | None:
    """The newest snapshot whose blobs all exist, or None."""
    mirror = Path(mirror_dir)
    snapshots = _snapshot_paths(mirror)
    if not snapshots:
        return None
    blobs = _blob_names(mirror)
    for seq, path in snapshots:
        doc = _read_snapshot(seq, path)
        if doc is not None and all(entry["sha256"] in blobs for entry in doc["files"].values()):
            return doc
    return None


def snapshot_file(mirror_dir: Path, relpath: str) -> bytes | None:
    """One file of the newest whole snapshot (for example the run's
    run-start.json), checked against its hash; None if absent."""
    mirror = Path(mirror_dir)
    doc = latest_snapshot(mirror)
    entry = None if doc is None else doc["files"].get(relpath)
    if entry is None:
        return None
    data = _blob(mirror, entry["sha256"]).read_bytes()
    if _sha256(data) != entry["sha256"]:
        raise MirrorError(f"{relpath}: the mirrored content does not match its hash")
    return data


def check_mirror_dir(mirror_dir: Path, *, files: int = 20,
                     clock: Callable[[], float] = time.monotonic) -> dict:
    """What the mirror needs from its filesystem (on Colab, Drive): a write
    with fsync, a replace over an existing file, a read back, a delete; and
    the time per small file. Run before the model download and any episode."""
    probe = Path(mirror_dir) / ".mirror-check"
    probe.mkdir(parents=True, exist_ok=True)
    target = probe / "replace-me.json"
    _write_atomic(target, b"1")
    _write_atomic(target, b"22")  # a replace over an existing file
    if target.read_bytes() != b"22":
        raise MirrorError(f"{probe}: a replaced file does not read back")
    started = clock()
    for index in range(files):
        _write_atomic(probe / f"small-{index}", b"x" * 1024)
    seconds = (clock() - started) / files
    for path in probe.iterdir():
        path.unlink()
    probe.rmdir()
    return {"mirror_dir": str(mirror_dir), "seconds_per_small_file": round(seconds, 4)}


def _relative(rel: str) -> PurePosixPath:
    path = PurePosixPath(rel)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise MirrorError(f"snapshot path {rel!r} is not inside the run directory")
    return path


def restore(mirror_dir: Path, local_dir: Path) -> dict:
    """Rebuild the run directory from the newest whole snapshot. The target
    must be empty or absent: a restore never overwrites a run."""
    mirror, local = Path(mirror_dir), Path(local_dir)
    if local.exists() and any(local.iterdir()):
        raise MirrorError(f"{local} is not empty; a restore never overwrites a run directory")
    doc = latest_snapshot(mirror)
    if doc is None:
        raise MirrorError(f"{mirror} holds no whole snapshot")
    for rel, entry in sorted(doc["files"].items()):
        data = _blob(mirror, entry["sha256"]).read_bytes()
        if _sha256(data) != entry["sha256"]:
            raise MirrorError(f"{rel}: the mirrored content does not match its hash")
        _write_atomic(local / Path(*_relative(rel).parts), data)
    ledger = local / "ledger.jsonl"
    if ledger.exists():
        digest = _sha256(json.dumps(doc, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        append_note(ledger, {"type": "restore", "snapshot_seq": doc["seq"],
                             "snapshot_sha256": digest})
    return doc


class RunMirror:
    """Snapshots of ``local_dir`` into ``mirror_dir``. Pass ``on_append`` to
    the run ledger: it snapshots at most every ``min_interval_seconds``, and
    always after a terminal record. A failed snapshot (Drive unavailable,
    say) is counted and reported, and the run goes on; the next one
    retries. Another process taking over the mirror stops this one."""

    def __init__(self, local_dir: Path, mirror_dir: Path, *, min_interval_seconds: float = 60.0,
                 clock: Callable[[], float] = time.monotonic,
                 warn: Callable[[str], None] | None = None):
        self.local, self.mirror = Path(local_dir), Path(mirror_dir)
        self._interval = float(min_interval_seconds)
        self._clock = clock
        self._warn = warn or (lambda message: print(message, file=sys.stderr, flush=True))
        self._last: float | None = None
        self._cache: dict[str, tuple[tuple[int, int], str, int]] = {}  # rel -> (stat, sha, size)
        self._known: set[str] = set()  # blobs known to exist in the mirror
        self.failures = 0
        self.last_error: str | None = None
        try:
            self.mirror.mkdir(parents=True, exist_ok=True)
            snapshots = _snapshot_paths(self.mirror)
            self._seq = snapshots[0][0] if snapshots else 0
            self._known = _blob_names(self.mirror)
            self._token = secrets.token_hex(16)
            _write_atomic(self.mirror / OWNER_FILE,
                          json.dumps({"token": self._token, "pid": os.getpid()}).encode("utf-8"))
        except OSError as exc:
            raise MirrorError(f"the mirror directory {self.mirror} cannot be used ({exc}); on "
                              f"Colab, mount Google Drive first") from exc

    @property
    def seq(self) -> int:
        return self._seq

    def on_append(self, record: Mapping) -> None:
        if record.get("type") in TERMINAL or self._last is None \
                or self._clock() - self._last >= self._interval:
            self.snapshot()

    def snapshot(self) -> int | None:
        """Take a snapshot now; returns its number, or None if it failed."""
        self._check_owner()
        try:
            seq = self._snapshot()
        except OSError as exc:
            self.failures += 1
            self.last_error = str(exc)
            self._warn(f"[mirror] snapshot failed ({self.failures} so far): {exc}")
            return None
        self._last = self._clock()
        return seq

    def _check_owner(self) -> None:
        try:
            owner = json.loads((self.mirror / OWNER_FILE).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return  # unreadable for now (Drive): the snapshot itself will report it
        if owner.get("token") != self._token:
            raise MirrorError(f"another process took over the mirror {self.mirror}; this run "
                              f"stops so that one run directory never has two writers")

    def _snapshot(self) -> int:
        files = {}
        for path in _local_files(self.local):
            rel = path.relative_to(self.local).as_posix()
            stat = path.stat()
            key = (stat.st_size, stat.st_mtime_ns)  # taken before the read
            cached = self._cache.get(rel)
            if cached is None or cached[0] != key or cached[1] not in self._known:
                data = path.read_bytes()
                digest = _sha256(data)
                if digest not in self._known and not _blob(self.mirror, digest).exists():
                    _write_atomic(_blob(self.mirror, digest), data)
                self._known.add(digest)
                cached = self._cache[rel] = (key, digest, len(data))
            files[rel] = {"sha256": cached[1], "size": cached[2]}
        seq = self._seq + 1
        doc = {"schema": MIRROR_SCHEMA, "seq": seq, "files": files}
        _write_atomic(self.mirror / f"snapshot-{seq:08d}.json",
                      json.dumps(doc, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        self._seq = seq
        self._collect(seq)
        return seq

    def _collect(self, seq: int) -> None:
        snapshots = _snapshot_paths(self.mirror)
        for _, path in snapshots[KEEP_SNAPSHOTS:]:
            path.unlink(missing_ok=True)
        if seq % GC_EVERY:
            return
        referenced: set[str] = set()
        for number, path in snapshots[:KEEP_SNAPSHOTS]:
            doc = _read_snapshot(number, path)
            if doc is None:
                return  # never collect against a manifest that cannot be read
            referenced |= {entry["sha256"] for entry in doc["files"].values()}
        for blob in (self.mirror / "blobs").glob("*/*"):
            if blob.name not in referenced and not blob.name.startswith("."):
                blob.unlink(missing_ok=True)
                self._known.discard(blob.name)
