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
newest snapshot whose blobs all exist and match their hashes, and records
itself as a ``restore`` note in the ledger.

A mirror belongs to one run directory: its id (``mirror-id.json``) is
written once, when the mirror is made, and the run directory keeps it
(``.mirror-id.json``, never snapshotted). A snapshot goes only into the
mirror its run directory belongs to, and only while that mirror is still
there: an unmounted Drive leaves an empty mount point, which never becomes
a mirror.

Work done after the last snapshot is lost with the VM: its episodes run
again after a restore, and the model calls and active time they took are
not counted. The restore notes keep every such gap on record (ADR-0008)."""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import shutil
import sys
import time
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path, PurePosixPath

from evaluation.exp0001.ledger import TERMINAL, append_note

MIRROR_SCHEMA = "heritagent/exp0001-mirror/0.1"
MIRROR_ID_SCHEMA = "heritagent/exp0001-mirror-id/0.1"
MIRROR_ID_FILE = "mirror-id.json"  # in the mirror: written once, when the mirror is made
BINDING_FILE = ".mirror-id.json"  # in the run directory: the id of the mirror it belongs to
SKIPPED_NAMES = (".lock", "UNMIRRORED.json", BINDING_FILE)  # these describe this machine only
START_RECORDS = ("run-start.json", "pilot-start.json")
KEEP_SNAPSHOTS = 2
GC_EVERY = 20  # snapshots between blob collections (listing blobs is the slow part)
OWNER_FILE = "owner.json"
RESTORING = ".restoring"  # a restore's staging directory, beside its target
_SNAPSHOT = re.compile(r"^snapshot-(\d{8})\.json$")


class MirrorError(RuntimeError):
    """The mirror cannot be used safely (no whole snapshot, a hash
    mismatch, a non-empty restore target, another writer, another run's
    mirror)."""


class _NotWhole(MirrorError):
    """One snapshot cannot be restored (a file fails its hash or is gone)."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _blob(mirror: Path, digest: str) -> Path:
    return mirror / "blobs" / digest[:2] / digest


def _write_atomic(path: Path, data: bytes) -> None:
    """Write through a temporary file and a replace, into a directory that
    exists: nothing here creates a parent, so a missing mirror fails the
    write. A failed write leaves no temporary file behind."""
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with tmp.open("wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass  # the collection removes it later
        raise


def _local_files(local: Path) -> list[Path]:
    return sorted(p for p in local.rglob("*")
                  if p.is_file() and p.name not in SKIPPED_NAMES
                  and not p.name.endswith(".tmp") and ".tmp-" not in p.name)


def _snapshot_paths(mirror: Path) -> list[tuple[int, Path]]:
    """Snapshot manifests, newest first; none for a mirror that does not
    exist. A listing that fails otherwise raises OSError."""
    try:
        entries = list(mirror.iterdir())
    except FileNotFoundError:
        return []
    found = []
    for path in entries:
        match = _SNAPSHOT.match(path.name)
        if match:
            found.append((int(match.group(1)), path))
    return sorted(found, reverse=True)


def _read_snapshot(seq: int, path: Path) -> dict | None:
    """A manifest, or None when it is not one of this schema or is gone. A
    read that fails (OSError) is raised: an unreadable manifest is never
    taken for an absent one."""
    try:
        doc = json.loads(path.read_bytes().decode("utf-8"))
    except FileNotFoundError:
        return None  # collected since the listing
    except ValueError:
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


def _unreadable(mirror: Path, exc: BaseException) -> MirrorError:
    return MirrorError(f"the mirror {mirror} cannot be read now ({exc}); Google Drive may be "
                       f"unmounted or still syncing. Mount it, wait a few minutes and run the "
                       f"command again")


def _whole_snapshots(mirror: Path) -> Iterator[dict]:
    """The snapshots whose blobs all exist, newest first. A listing or
    manifest that cannot be read raises MirrorError."""
    try:
        snapshots = _snapshot_paths(mirror)
        blobs = _blob_names(mirror) if snapshots else set()
        for seq, path in snapshots:
            doc = _read_snapshot(seq, path)
            if doc is not None and all(entry["sha256"] in blobs
                                       for entry in doc["files"].values()):
                yield doc
    except OSError as exc:
        raise _unreadable(mirror, exc) from None


def latest_snapshot(mirror_dir: Path) -> dict | None:
    """The newest snapshot whose blobs all exist, or None. A mirror that
    cannot be read now raises MirrorError, never None."""
    return next(_whole_snapshots(Path(mirror_dir)), None)


def mirror_state(mirror_dir: Path) -> str:
    """``"empty"`` (no snapshot manifest: a new run may start here; blobs
    without any manifest hold nothing a restore could use), ``"whole"`` (a
    whole snapshot), or ``"broken"`` (manifests, but none whole: Drive may
    still be syncing, or uploads were lost; never a new run). A mirror that
    cannot be read now raises MirrorError."""
    mirror = Path(mirror_dir)
    if latest_snapshot(mirror) is not None:
        return "whole"
    try:
        return "broken" if _snapshot_paths(mirror) else "empty"
    except OSError as exc:
        raise _unreadable(mirror, exc) from None


def snapshot_file(mirror_dir: Path, relpath: str) -> bytes | None:
    """One file of the newest whole snapshot (for example the run's
    run-start.json), checked against its hash; None if absent."""
    doc = latest_snapshot(mirror_dir)
    return None if doc is None else read_snapshot_file(mirror_dir, doc, relpath)


def read_snapshot_file(mirror_dir: Path, snapshot: Mapping, relpath: str) -> bytes | None:
    """One file of ``snapshot`` (a document from ``latest_snapshot``), checked
    against its hash; None if the snapshot lacks it. Several files of one
    snapshot read this way list the blobs once, not once per file."""
    entry = snapshot["files"].get(relpath)
    if entry is None:
        return None
    try:
        data = _blob(Path(mirror_dir), entry["sha256"]).read_bytes()
    except FileNotFoundError:  # collected since the listing: a run is writing the mirror
        raise MirrorError(f"{relpath}: its blob is gone from the mirror; a run may be "
                          "writing it. Read again when no run is active") from None
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


def _read_id(path: Path) -> str | None:
    """The mirror id a file holds, None if there is no such file."""
    try:
        doc = json.loads(path.read_bytes().decode("utf-8"))
    except FileNotFoundError:
        return None
    except ValueError as exc:
        raise OSError(f"{path} is not a mirror id ({exc})") from None
    if not isinstance(doc, dict) or doc.get("schema") != MIRROR_ID_SCHEMA \
            or not isinstance(doc.get("mirror_id"), str):
        raise OSError(f"{path} is not a mirror id")
    return doc["mirror_id"]


def _id_document(mirror_id: str) -> bytes:
    return json.dumps({"schema": MIRROR_ID_SCHEMA, "mirror_id": mirror_id}).encode("utf-8")


def _start_identity(files: Mapping[str, Mapping]) -> str | None:
    """The run a set of snapshot entries belongs to: its start record."""
    for name in START_RECORDS:
        if name in files:
            return f"{name}:{files[name]['sha256']}"
    return None


def _restore_into(mirror: Path, doc: Mapping, staging: Path,
                  progress: Callable[[int, int], None] | None) -> None:
    total = len(doc["files"])
    if progress is not None:
        progress(0, total)
    for done, (rel, entry) in enumerate(sorted(doc["files"].items()), 1):
        try:
            data = _blob(mirror, entry["sha256"]).read_bytes()
        except FileNotFoundError:
            raise _NotWhole(f"{rel}: its blob is gone from the mirror (a run may be writing "
                            f"it)") from None
        except OSError as exc:
            raise _unreadable(mirror, exc) from None
        if _sha256(data) != entry["sha256"]:
            raise _NotWhole(f"{rel}: the mirrored content does not match its hash")
        target = staging / Path(*_relative(rel).parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        if progress is not None:
            progress(done, total)


def restore(mirror_dir: Path, local_dir: Path,
            progress: Callable[[int, int], None] | None = None) -> dict:
    """Rebuild the run directory from the newest whole snapshot whose every
    file matches its hash (ADR-0008 D5), all or nothing: the files, the
    mirror id and the restore note go into a staging directory beside the
    target, which is renamed into place at the end. An error, a stop or a
    kill part-way leaves the target as it was; the next restore removes the
    staging directory. The target must be empty or absent: a restore never
    overwrites a run. ``progress(done, total)`` follows the files."""
    import fcntl
    mirror, local = Path(mirror_dir), Path(local_dir)
    local.parent.mkdir(parents=True, exist_ok=True)
    staging = local.with_name(local.name + RESTORING)
    with open(local.with_name(f".{local.name}.restore-lock"), "a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise MirrorError(f"another restore of {local} is running") from None
        if local.exists() and any(local.iterdir()):
            raise MirrorError(f"{local} is not empty; a restore never overwrites a run "
                              f"directory")
        shutil.rmtree(staging, ignore_errors=True)  # a killed restore's: no record in it
        try:
            mirror_id = _read_id(mirror / MIRROR_ID_FILE)
        except OSError as exc:
            raise _unreadable(mirror, exc) from None
        failed = []
        try:
            for doc in _whole_snapshots(mirror):
                try:
                    _restore_into(mirror, doc, staging, progress)
                except _NotWhole as exc:
                    failed.append(f"snapshot {doc['seq']}: {exc}")
                    _print_warning(f"[restore] snapshot {doc['seq']} is not whole ({exc}); "
                                   f"trying the next older one")
                    shutil.rmtree(staging, ignore_errors=True)
                    continue
                break
            else:
                raise MirrorError(f"{mirror} holds no whole snapshot"
                                  + (f" ({'; '.join(failed)})" if failed else ""))
            staging.mkdir(exist_ok=True)  # a snapshot of no files
            if mirror_id is not None:  # a mirror written before ids existed has none
                _write_atomic(staging / BINDING_FILE, _id_document(mirror_id))
            ledger = staging / "ledger.jsonl"
            if ledger.exists():
                digest = _sha256(json.dumps(doc, sort_keys=True,
                                            separators=(",", ":")).encode("utf-8"))
                append_note(ledger, {"type": "restore", "snapshot_seq": doc["seq"],
                                     "snapshot_sha256": digest})
            if local.exists():
                local.rmdir()  # empty, checked above
            os.rename(staging, local)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
    return doc


def _print_warning(message: str) -> None:
    try:
        print(message, file=sys.stderr, flush=True)
    except OSError:
        pass  # nobody reads this process's output any more (its notebook kernel died)


class RunMirror:
    """Snapshots of ``local_dir`` into ``mirror_dir``. Pass ``on_append`` to
    the run ledger: it snapshots at most every ``min_interval_seconds``, and
    always after a terminal record. A failed snapshot (Drive unavailable,
    say) is counted and reported, and the run goes on; the next one retries
    after the interval, or at a terminal record. Another process taking
    over the mirror stops this one.

    The mirror must be the run directory's own (``_attach``); every
    snapshot first checks that it still is, so an unmounted Drive fails
    the snapshot instead of taking it on this machine's disk."""

    def __init__(self, local_dir: Path, mirror_dir: Path, *, min_interval_seconds: float = 60.0,
                 clock: Callable[[], float] = time.monotonic,
                 warn: Callable[[str], None] | None = None):
        self.local, self.mirror = Path(local_dir), Path(mirror_dir)
        self._interval = float(min_interval_seconds)
        self._clock = clock
        self._warn = warn or _print_warning
        self._last: float | None = None
        self._cache: dict[str, tuple[tuple[int, int], str, int]] = {}  # rel -> (stat, sha, size)
        self._known: set[str] = set()  # blobs known to exist in the mirror
        self._stale = False  # a snapshot failed: _known may name blobs the mirror lacks
        self.failures = 0
        self.last_error: str | None = None
        try:
            self._attach()
            snapshots = _snapshot_paths(self.mirror)
            self._seq = snapshots[0][0] if snapshots else 0
            self._known = _blob_names(self.mirror)
            self._token = secrets.token_hex(16)
            _write_atomic(self.mirror / OWNER_FILE,
                          json.dumps({"token": self._token, "pid": os.getpid()}).encode("utf-8"))
        except OSError as exc:
            raise MirrorError(f"the mirror directory {self.mirror} cannot be used ({exc}); on "
                              f"Colab, mount Google Drive first") from exc

    def _attach(self) -> None:
        """The mirror checked against the run directory (ADR-0008 D5: one
        run directory, one mirror), and the two bound. A run directory that was
        mirrored before takes only its own mirror, which must exist. Any
        other takes a mirror without snapshots (and makes it, without
        parents), or one whose snapshots hold its own start record (a mirror
        written before ids existed). A run directory behind its mirror's
        newest whole snapshot is refused: resuming it would overwrite newer
        records."""
        bound = _read_id(self.local / BINDING_FILE)
        if bound is not None and not self.mirror.is_dir():
            raise MirrorError(f"the mirror directory {self.mirror} cannot be used: it does not "
                              f"exist now, and {self.local} was mirrored before (is Google "
                              f"Drive unmounted?). Mount Drive again (force_remount=True) and "
                              f"run the command again; check that --mirror-dir names this "
                              f"run's own mirror")
        if bound is None:
            self.mirror.mkdir(exist_ok=True)  # no parents: an unmounted Drive fails here
        mine = _read_id(self.mirror / MIRROR_ID_FILE)
        if bound is not None and mine != bound:
            raise MirrorError(f"{self.mirror} is not the mirror {self.local} was mirrored to "
                              f"(its id {mine}, the run directory's {bound}): give the run's own "
                              f"--mirror-dir")
        if bound is None:
            snapshots = _snapshot_paths(self.mirror)
            if snapshots:
                newest = next((doc for doc in (_read_snapshot(*s) for s in snapshots)
                               if doc is not None), None)
                theirs = None if newest is None else _start_identity(newest["files"])
                if theirs is None or theirs != self._local_start():
                    raise MirrorError(
                        f"{self.mirror} holds the snapshots of another run directory, not of "
                        f"{self.local}: give this run directory its own --mirror-dir (a new "
                        f"run needs a new mirror, a new RUN_ID)")
            if mine is None:
                mine = secrets.token_hex(16)
                _write_atomic(self.mirror / MIRROR_ID_FILE, _id_document(mine))
            self.local.mkdir(parents=True, exist_ok=True)
            _write_atomic(self.local / BINDING_FILE, _id_document(mine))
        self._require_not_behind()

    def _local_start(self) -> str | None:
        for name in START_RECORDS:
            path = self.local / name
            if path.exists():
                return f"{name}:{_sha256(path.read_bytes())}"
        return None

    def _require_not_behind(self) -> None:
        """The complete records of the ledger in the newest whole snapshot
        are the start of the local ledger: the ledger only grows (a repair
        cuts only a torn last line, a restore appends a note), so anything
        else is a stale or diverged copy."""
        ledger = self.local / "ledger.jsonl"
        if not ledger.exists():
            return
        for doc in _whole_snapshots(self.mirror):
            try:
                theirs = read_snapshot_file(self.mirror, doc, "ledger.jsonl")
            except MirrorError:
                continue  # a file that fails its hash: that snapshot is not whole
            if theirs is None:
                return
            if not ledger.read_bytes().startswith(theirs[:theirs.rfind(b"\n") + 1]):
                raise MirrorError(
                    f"the ledger in the mirror's snapshot {doc['seq']} ({self.mirror}) holds "
                    f"records that {ledger} lacks: this run directory is behind its mirror, "
                    f"and resuming it would overwrite newer records. Move {self.local} away "
                    f"and run the command again to restore the run from the mirror")
            return

    @property
    def seq(self) -> int:
        return self._seq

    def on_append(self, record: Mapping) -> None:
        if record.get("type") in TERMINAL or self._last is None \
                or self._clock() - self._last >= self._interval:
            self.snapshot()

    def snapshot(self) -> int | None:
        """Take a snapshot now; returns its number, or None if it failed."""
        try:
            self._check_mirror()
            seq = self._snapshot()
        except OSError as exc:
            self.failures += 1
            self.last_error = str(exc)
            self._stale = True
            self._last = self._clock()  # a failing Drive is not retried at every record
            self._warn(f"[mirror] snapshot failed ({self.failures} so far): {exc}")
            return None
        self._last = self._clock()
        return seq

    def _check_mirror(self) -> None:
        """The mirror is still there and still held by this process: its
        owner file names this process. One that cannot be read (an
        unmounted Drive leaves an empty mount point) fails the snapshot, an
        OSError, and never passes it."""
        try:
            owner = json.loads((self.mirror / OWNER_FILE).read_bytes().decode("utf-8"))
        except (OSError, ValueError) as exc:
            raise OSError(f"the mirror {self.mirror} cannot be read ({exc}); is Google Drive "
                          f"still mounted?") from None
        if not isinstance(owner, dict) or owner.get("token") != self._token:
            raise MirrorError(f"another process took over the mirror {self.mirror}; this run "
                              f"stops so that one run directory never has two writers")

    def _snapshot(self) -> int:
        if self._stale:  # after a failure (a remount, say) the mirror is listed again
            self._known = _blob_names(self.mirror)
            self._stale = False
        files = {}
        for path in _local_files(self.local):
            rel = path.relative_to(self.local).as_posix()
            stat = path.stat()
            key = (stat.st_size, stat.st_mtime_ns)  # taken before the read
            cached = self._cache.get(rel)
            if cached is None or cached[0] != key or cached[1] not in self._known:
                data = path.read_bytes()
                digest = _sha256(data)
                blob = _blob(self.mirror, digest)
                if digest not in self._known and not blob.exists():
                    for directory in (self.mirror / "blobs", blob.parent):
                        directory.mkdir(exist_ok=True)  # never a parent above the mirror
                    _write_atomic(blob, data)
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
            try:
                doc = _read_snapshot(number, path)
            except OSError:
                doc = None
            if doc is None:
                return  # never collect against a manifest that cannot be read
            referenced |= {entry["sha256"] for entry in doc["files"].values()}
        for leftover in self.mirror.glob(".*.tmp-*"):  # of writes that failed
            leftover.unlink(missing_ok=True)
        for blob in (self.mirror / "blobs").glob("*/*"):
            if ".tmp-" in blob.name or (blob.name not in referenced
                                        and not blob.name.startswith(".")):
                blob.unlink(missing_ok=True)
                self._known.discard(blob.name)
