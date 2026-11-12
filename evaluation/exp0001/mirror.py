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
itself as a ``restore`` note in the ledger. It also leaves a restore record
(``.restore.json``, never snapshotted) in the run directory until the
directory's first snapshot: the snapshot it restored, the snapshots it had
to skip and why, and a restore note still to be appended when the restored
ledger ends with a torn line (``repair-ledger`` appends it after its cut).

A blob holds its file gzip-compressed (layout 0.2), still named by the hash
of the uncompressed file, which every reader checks after decompressing:
the ledger, rewritten into a new blob at every snapshot, and the
trajectories, which repeat the prompt at every step, take a small part of
their size on Drive, where deleted files stay in the Trash and count
against storage for 30 days. The manifest names its layout; the readers
also read the uncompressed layout 0.1 (the calibration pilot's mirror). A
manifest of a layout this code does not know fails every reader, and a
writer never continues a mirror of another layout.

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

import gzip
import hashlib
import json
import os
import re
import secrets
import shutil
import sys
import time
import zlib
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path, PurePosixPath

from evaluation.exp0001.ledger import TERMINAL, append_note

MIRROR_SCHEMA = "heritagent/exp0001-mirror/0.2"  # the layout this code writes: gzip blobs
UNCOMPRESSED_SCHEMA = "heritagent/exp0001-mirror/0.1"  # read only: raw blobs
GZIP_LEVEL = 6  # with mtime 0: the same file always makes the same blob
MIRROR_ID_SCHEMA = "heritagent/exp0001-mirror-id/0.1"
MIRROR_ID_FILE = "mirror-id.json"  # in the mirror: written once, when the mirror is made
BINDING_FILE = ".mirror-id.json"  # in the run directory: the id of the mirror it belongs to
RESTORE_RECORD = ".restore.json"  # in a restored run directory until its first snapshot
RESTORE_SCHEMA = "heritagent/exp0001-restore/0.1"
SKIPPED_NAMES = (".lock", "UNMIRRORED.json", BINDING_FILE, RESTORE_RECORD)  # this machine only
START_RECORDS = ("run-start.json", "pilot-start.json")
# Drive uploads the mount's writes in the background, perhaps not in the
# order they were made (U), and each kept manifest names its own ledger
# blob: a margin of kept snapshots leaves a whole one on Drive even when
# the newest few never arrived. Compressed, a kept snapshot costs a few MB
# at the end of a run.
KEEP_SNAPSHOTS = 5
GC_EVERY = 20  # snapshots between blob collections (listing blobs is the slow part)
OWNER_FILE = "owner.json"
RESTORING = ".restoring"  # a restore's staging directory, beside its target
_SNAPSHOT = re.compile(r"^snapshot-(\d{8})\.json$")


class MirrorError(RuntimeError):
    """The mirror cannot be used safely (no whole snapshot, a hash
    mismatch, a non-empty restore target, another writer, another run's
    mirror)."""


class NothingUnmirrored(MirrorError):
    """A refusal that leaves no record only on this machine: the mirror
    holds all the run directory holds (a directory behind its mirror)."""


class RestoreNotePending(MirrorError):
    """A restored run directory whose restore note waits for
    ``repair-ledger``. The refusal leaves the unmirrored marker, so the
    notebook keeps the runtime in which ``repair-ledger`` is run (ADR-0008
    D5)."""


class _NotWhole(MirrorError):
    """One snapshot cannot be restored (a file fails its hash or is gone)."""

    def __init__(self, rel: str, reason: str):
        super().__init__(f"{rel}: {reason}")
        self.rel = rel


class _NotAnId(OSError):
    """A mirror id file that is there but holds no whole mirror id."""


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
    """A manifest, or None when it is not a manifest or is gone. A read
    that fails (OSError) is raised: an unreadable manifest is never taken
    for an absent one. Nor is one of a layout this code does not know
    (MirrorError)."""
    try:
        doc = json.loads(path.read_bytes().decode("utf-8"))
    except FileNotFoundError:
        return None  # collected since the listing
    except ValueError:
        return None
    if not isinstance(doc, dict) or doc.get("seq") != seq or not isinstance(doc.get("files"), dict):
        return None
    if doc.get("schema") not in (MIRROR_SCHEMA, UNCOMPRESSED_SCHEMA):
        raise MirrorError(f"{path} is a snapshot of the layout {doc.get('schema')!r}, which this "
                          f"code cannot read: the mirror was written by a newer version of the "
                          f"code. Read it with the code of the commit the run started with")
    return doc


def _unpack(snapshot: Mapping, rel: str, data: bytes) -> bytes:
    """The file a blob of ``snapshot`` holds: gzip-compressed in layout 0.2,
    as it is in 0.1. A blob that does not decompress is not whole."""
    if snapshot["schema"] == UNCOMPRESSED_SCHEMA:
        return data
    try:
        return gzip.decompress(data)
    except (OSError, EOFError, zlib.error) as exc:
        raise _NotWhole(rel, f"the mirrored content does not match its hash (its compressed "
                             f"blob cannot be read: {exc})") from None


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


def _holds_blobs(mirror: Path) -> bool:
    """Whether any blob is in the mirror; stops at the first one."""
    root = mirror / "blobs"
    if not root.is_dir():
        return False
    return any(not blob.name.startswith(".") for prefix in root.iterdir() if prefix.is_dir()
               for blob in prefix.iterdir())


def mirror_state(mirror_dir: Path) -> str:
    """``"empty"`` (no snapshot manifest and no blob: a new run may start
    here), ``"whole"`` (a whole snapshot), or ``"broken"`` (manifests or
    blobs, but no whole snapshot: Drive may still be syncing, or uploads
    were lost, and the blobs are what a recovery from the Drive Trash
    needs; never a new run). A mirror that cannot be read now raises
    MirrorError."""
    mirror = Path(mirror_dir)
    if latest_snapshot(mirror) is not None:
        return "whole"
    try:
        return "broken" if _snapshot_paths(mirror) or _holds_blobs(mirror) else "empty"
    except OSError as exc:
        raise _unreadable(mirror, exc) from None


_RECOVERY = ("Google Drive may still be syncing from the last session. Wait a few minutes and "
             "run again; never start a new run on this mirror. If a machine still holds the run "
             "directory, `snapshot` from that directory makes the mirror whole again. If none "
             "does and the mirror stays like this, the snapshot files deleted before are in the "
             "Drive Trash for 30 days (restore them to the mirror there), or start the run again "
             "under a new RUN_ID")


def broken_mirror_text(mirror_dir: Path) -> str:
    """What to do with a mirror that holds snapshot files but no whole
    snapshot."""
    return f"the mirror {mirror_dir} holds snapshot files but no whole snapshot: {_RECOVERY}"


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
    data = _unpack(snapshot, relpath, data)
    if _sha256(data) != entry["sha256"]:
        raise MirrorError(f"{relpath}: the mirrored content does not match its hash")
    return data


def _space(path: Path) -> tuple[int | None, int | None]:
    """The total and free bytes of the filesystem holding ``path``; None for
    both when it does not say (what Drive's mount reports is unverified)."""
    try:
        usage = shutil.disk_usage(path)
    except OSError:
        return None, None
    return (None, None) if usage.total <= 0 else (usage.total, usage.free)


def check_mirror_dir(mirror_dir: Path, *, files: int = 20, need_bytes: int = 0,
                     clock: Callable[[], float] = time.monotonic) -> dict:
    """What the mirror needs from its filesystem (on Colab, Drive): a write
    with fsync, a replace over an existing file, a read back, a delete; and
    the time per small file (ADR-0008 D5). Run before the model download and
    any episode. Its free space, when the filesystem reports it, must reach
    ``need_bytes``; an unknown one is reported as unknown (None)."""
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
    total, free = _space(Path(mirror_dir))
    if free is not None and free < need_bytes:
        raise MirrorError(
            f"the mirror's filesystem ({mirror_dir}) has {free / 1e9:.1f} GB free of "
            f"{total / 1e9:.1f} GB, below the {need_bytes / 1e9:.1f} GB asked for (mirror-check "
            f"--need-gb; a rehearsal or the pilot needs far less than a run). These figures are "
            f"what the Drive mount reports and may be wrong: compare them with "
            f"drive.google.com/settings/storage. Free space on Google Drive: delete files you do "
            f"not need, then empty the Drive Trash (deleted snapshots stay there for 30 days and "
            f"count against storage), allow "
            f"time for the storage figure to update (up to a few days after many deletions), "
            f"mount Drive again (force_remount=True) and run mirror-check again")
    return {"mirror_dir": str(mirror_dir), "seconds_per_small_file": round(seconds, 4),
            "total_bytes": total, "free_bytes": free, "need_bytes": need_bytes}


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
        raise _NotAnId(f"{path} is not a mirror id ({exc})") from None
    if not isinstance(doc, dict) or doc.get("schema") != MIRROR_ID_SCHEMA \
            or not isinstance(doc.get("mirror_id"), str):
        raise _NotAnId(f"{path} is not a mirror id")
    return doc["mirror_id"]


def _torn_id(path: Path, exc: BaseException) -> MirrorError:
    return MirrorError(
        f"{exc}: the file says which run directory the mirror belongs to, and it is torn or "
        f"empty. If Google Drive was mounted just now, wait a few minutes and run the command "
        f"again. Otherwise run `snapshot` from the machine that holds the run directory (it "
        f"writes the file again); if no machine holds it any more, delete {path}, and the next "
        f"command restores the run from its snapshots and writes a new one")


def _id_document(mirror_id: str) -> bytes:
    return json.dumps({"schema": MIRROR_ID_SCHEMA, "mirror_id": mirror_id}).encode("utf-8")


def _start_identity(files: Mapping[str, Mapping]) -> str | None:
    """The run a set of snapshot entries belongs to: its start record."""
    for name in START_RECORDS:
        if name in files:
            return f"{name}:{files[name]['sha256']}"
    return None


def _restore_into(mirror: Path, doc: Mapping, staging: Path,
                  progress: Callable[[int, int], None] | None) -> list[_NotWhole]:
    """The files of ``doc`` into ``staging``; the files that are not whole,
    each one (none for a whole snapshot): a restore records every blob that
    failed, so that later snapshots never trust one by name."""
    total, failed = len(doc["files"]), []
    if progress is not None:
        progress(0, total)
    for done, (rel, entry) in enumerate(sorted(doc["files"].items()), 1):
        try:
            data = _blob(mirror, entry["sha256"]).read_bytes()
        except FileNotFoundError:
            failed.append(_NotWhole(rel, "its blob is gone from the mirror (a run may be "
                                         "writing it)"))
            continue
        except OSError as exc:
            raise _unreadable(mirror, exc) from None
        try:
            data = _unpack(doc, rel, data)
        except _NotWhole as exc:
            failed.append(exc)
            continue
        if _sha256(data) != entry["sha256"]:
            failed.append(_NotWhole(rel, "the mirrored content does not match its hash"))
            continue
        if failed:
            continue  # checked, not written: this snapshot is not restored
        target = staging / Path(*_relative(rel).parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        if progress is not None:
            progress(done, total)
    return failed


def restore(mirror_dir: Path, local_dir: Path,
            progress: Callable[[int, int], None] | None = None) -> dict:
    """Rebuild the run directory from the newest whole snapshot whose every
    file matches its hash (ADR-0008 D5), all or nothing: the files, the
    mirror id and the restore note go into a staging directory beside the
    target, which is renamed into place at the end. An error, a stop or a
    kill part-way leaves the target as it was; the next restore removes the
    staging directory. The target must be empty or absent: a restore never
    overwrites a run. ``progress(done, total)`` follows the files.

    A ledger that ends with a torn line is restored as it is, never cut
    here (ADR-0008 D5): its restore note waits in the restore record until
    ``repair-ledger`` has cut the line on purpose."""
    import fcntl
    mirror, local = Path(mirror_dir), Path(local_dir)
    local.parent.mkdir(parents=True, exist_ok=True)
    staging = local.with_name(local.name + RESTORING)
    with open(local.with_name(f".{local.name}.restore-lock"), "a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise MirrorError(f"another restore of {local} is running: wait until it ends "
                              f"and run the command again") from None
        if local.exists() and any(local.iterdir()):
            raise MirrorError(f"{local} is not empty; a restore never overwrites a run "
                              f"directory")
        shutil.rmtree(staging, ignore_errors=True)  # a killed restore's: no record in it
        try:
            mirror_id = _read_id(mirror / MIRROR_ID_FILE)
        except _NotAnId as exc:
            raise _torn_id(mirror / MIRROR_ID_FILE, exc) from None
        except OSError as exc:
            raise _unreadable(mirror, exc) from None
        failed, skipped = [], []
        try:
            for doc in _whole_snapshots(mirror):
                bad = _restore_into(mirror, doc, staging, progress)
                if not bad:
                    break
                failed.append(f"snapshot {doc['seq']}: {'; '.join(map(str, bad))}")
                skipped += [{"seq": doc["seq"], "file": exc.rel,
                             "sha256": doc["files"][exc.rel]["sha256"], "reason": str(exc)}
                            for exc in bad]
                _print_warning(f"[restore] snapshot {doc['seq']} is not whole "
                               f"({'; '.join(map(str, bad))}); trying the next older one")
                shutil.rmtree(staging, ignore_errors=True)
            else:
                raise MirrorError(f"{mirror} holds no whole snapshot"
                                  + (f" ({'; '.join(failed)})" if failed else "")
                                  + f". {_RECOVERY}")
            staging.mkdir(exist_ok=True)  # a snapshot of no files
            if mirror_id is not None:  # a mirror written before ids existed has none
                _write_atomic(staging / BINDING_FILE, _id_document(mirror_id))
            digest = _sha256(json.dumps(doc, sort_keys=True,
                                        separators=(",", ":")).encode("utf-8"))
            note = {"type": "restore", "snapshot_seq": doc["seq"], "snapshot_sha256": digest}
            ledger = staging / "ledger.jsonl"
            torn = ledger.exists() and not ledger.read_bytes().endswith(b"\n")
            _write_atomic(staging / RESTORE_RECORD, json.dumps(
                {"schema": RESTORE_SCHEMA, "snapshot_seq": doc["seq"], "snapshot_sha256": digest,
                 "skipped": skipped, "pending_note": note if torn else None},
                sort_keys=True).encode("utf-8"))
            if ledger.exists() and not torn:
                append_note(ledger, note)
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


def restore_record(local_dir: Path) -> dict | None:
    """The restore record of a run directory restored and not snapshotted
    since, or None."""
    path = Path(local_dir) / RESTORE_RECORD
    try:
        doc = json.loads(path.read_bytes().decode("utf-8"))
    except FileNotFoundError:
        return None
    except ValueError:
        doc = None
    if not isinstance(doc, dict) or doc.get("schema") != RESTORE_SCHEMA:
        raise MirrorError(f"{path} is not a restore record (a restore writes it whole, and "
                          f"nothing else writes it). Keep {local_dir} and its records as they "
                          f"are, and report the file before going on")
    return doc


def require_no_pending_note(local_dir: Path) -> None:
    """A restored run directory whose restore note waits for
    ``repair-ledger`` is not resumed, snapshotted or taken over."""
    record = restore_record(local_dir)
    if record is not None and record.get("pending_note") is not None:
        raise RestoreNotePending(_pending_note_text(Path(local_dir)))


def _pending_note_text(local: Path) -> str:
    return (f"{local} was restored from its mirror, and its ledger ends with a torn line (a "
            f"write a crash interrupted), so its restore note is still to be appended. Run "
            f"`python -m evaluation.exp0001.cli repair-ledger --run-dir {local}` (in a "
            f"notebook: a new cell `cli(\"repair-ledger\", \"--run-dir\", \"{local}\")`; it cuts "
            f"the line on purpose, keeps the cut bytes and appends both notes), then the same "
            f"command again")


def append_pending_restore_note(local_dir: Path) -> dict | None:
    """The restore note a restore of a ledger with a torn last line left
    pending, appended once ``repair-ledger`` has cut the line, and cleared
    from the restore record (ADR-0008 D5). None when no note is pending."""
    local = Path(local_dir)
    record = restore_record(local)
    if record is None or record.get("pending_note") is None:
        return None
    note, ledger = record["pending_note"], local / "ledger.jsonl"
    lines = ledger.read_bytes().splitlines()
    try:
        appended = bool(lines) and json.loads(lines[-1]) == note
    except ValueError:
        appended = False  # a torn last line: append_note refuses it
    if not appended:  # else a repair stopped before it cleared the record
        append_note(ledger, note)
    _write_atomic(local / RESTORE_RECORD,
                  json.dumps({**record, "pending_note": None}, sort_keys=True).encode("utf-8"))
    return note


class RunMirror:
    """Snapshots of ``local_dir`` into ``mirror_dir``. Pass ``on_append`` to
    the run ledger: it snapshots at most every ``min_interval_seconds``, and
    always after a terminal record. A failed snapshot (Drive unavailable,
    say) is counted and reported, and the run goes on; the next one retries
    after the interval, or at a terminal record. Another process taking
    over the mirror stops this one.

    The mirror must be the run directory's own (``_attach``); every
    snapshot first checks that it still is, so an unmounted Drive fails
    the snapshot instead of taking it on this machine's disk. A run
    directory restored and not snapshotted since takes the mirror over
    (ADR-0008 D5), and writes again the blobs its restore found failing
    their hash when a file of it has that content."""

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
        require_no_pending_note(self.local)
        self._restored = restore_record(self.local)
        # blobs a restore found failing their hash: never trusted by name
        self._suspect = {skip["sha256"] for skip in (self._restored or {}).get("skipped", [])}
        try:
            snapshots, docs = self._attach()
            self._seq = snapshots[0][0] if snapshots else 0
            self._known = _blob_names(self.mirror)
            if self._restored is None:  # a restored directory takes the mirror over
                self._require_not_behind(docs)
            self._token = secrets.token_hex(16)
            _write_atomic(self.mirror / OWNER_FILE,
                          json.dumps({"token": self._token, "pid": os.getpid()}).encode("utf-8"))
        except OSError as exc:
            where = "" if self.mirror.parent.is_dir() else (
                f"; if Drive is mounted, its folder {self.mirror.parent} does not exist: check "
                f"--mirror-dir, or make the folder (the notebooks' mirror-check cell does)")
            raise MirrorError(f"the mirror directory {self.mirror} cannot be used ({exc}); on "
                              f"Colab, mount Google Drive first{where}") from exc

    def _attach(self) -> tuple[list[tuple[int, Path]], list[dict]]:
        """The mirror checked against the run directory (ADR-0008 D5: one
        run directory, one mirror), and the two bound. A run directory that was
        mirrored before takes only its own mirror, which must exist. Any
        other takes a mirror without snapshots or blobs (and makes it,
        without parents), or one whose snapshots hold its own start record (a
        mirror written before ids existed). Returns the snapshot listing and
        the manifests that could be read, newest first, each read once."""
        bound = _read_id(self.local / BINDING_FILE)
        if bound is not None and not self.mirror.is_dir():
            raise MirrorError(f"the mirror directory {self.mirror} cannot be used: it does not "
                              f"exist now, and {self.local} was mirrored before (is Google "
                              f"Drive unmounted?). Mount Drive again (force_remount=True) and "
                              f"run the command again; check that --mirror-dir names this "
                              f"run's own mirror")
        if bound is None:
            self.mirror.mkdir(exist_ok=True)  # no parents: an unmounted Drive fails here
        snapshots = _snapshot_paths(self.mirror)
        docs = [doc for doc in (_read_snapshot(*s) for s in snapshots) if doc is not None]
        self._require_own_layout(docs)
        try:
            mine = _read_id(self.mirror / MIRROR_ID_FILE)
        except _NotAnId as exc:
            mine = self._id_again(bound, docs, exc)
        if bound is not None and mine != bound:
            raise MirrorError(f"{self.mirror} is not the mirror {self.local} was mirrored to "
                              f"(its id {mine}, the run directory's {bound}): give the run's own "
                              f"--mirror-dir")
        if bound is None:
            if snapshots:
                theirs = None if not docs else _start_identity(docs[0]["files"])
                if theirs is None or theirs != self._local_start():
                    raise MirrorError(
                        f"{self.mirror} holds the snapshots of another run directory, not of "
                        f"{self.local}: give this run directory its own --mirror-dir (a new "
                        f"run needs a new mirror, a new RUN_ID)")
            elif _holds_blobs(self.mirror):
                raise MirrorError(broken_mirror_text(self.mirror))
            if mine is None:
                mine = secrets.token_hex(16)
                _write_atomic(self.mirror / MIRROR_ID_FILE, _id_document(mine))
            self.local.mkdir(parents=True, exist_ok=True)
            _write_atomic(self.local / BINDING_FILE, _id_document(mine))
        return snapshots, docs

    def _id_again(self, bound: str | None, docs: list[dict], exc: OSError) -> str:
        """A torn or empty mirror id written again from the binding of the
        run directory whose start record the mirror's newest snapshot holds."""
        own = self._local_start()
        if bound is None or not docs or own is None or _start_identity(docs[0]["files"]) != own:
            raise _torn_id(self.mirror / MIRROR_ID_FILE, exc)
        _write_atomic(self.mirror / MIRROR_ID_FILE, _id_document(bound))
        self._warn(f"[mirror] {exc}; written again from {self.local}'s binding, whose start "
                   f"record the mirror's newest snapshot holds")
        return bound

    def _require_own_layout(self, docs: list[dict]) -> None:
        """A run continues at the commit it started with, so its mirror
        holds one layout: a mirror with snapshots of another layout is never
        written into, before anything is written."""
        for doc in docs:
            if doc["schema"] != MIRROR_SCHEMA:
                raise MirrorError(
                    f"{self.mirror} holds snapshots of the layout {doc['schema']!r}, written by "
                    f"older code; this code writes {MIRROR_SCHEMA!r} and never mixes the two. "
                    f"Continue this run (or pilot) with the code of the commit it started with "
                    f"(CODE_REF), or start a new run under a new RUN_ID")

    def _local_start(self) -> str | None:
        for name in START_RECORDS:
            path = self.local / name
            if path.exists():
                return f"{name}:{_sha256(path.read_bytes())}"
        return None

    def _require_not_behind(self, docs: list[dict]) -> None:
        """The complete records of the ledger in the newest whole snapshot
        are the start of the local ledger: the ledger only grows (a repair
        cuts only a torn last line, a restore appends a note), so anything
        else is a stale or diverged copy. A directory just restored is not
        checked: it takes the mirror over."""
        ledger = self.local / "ledger.jsonl"
        if not ledger.exists():
            return
        for doc in docs:
            if not all(entry["sha256"] in self._known for entry in doc["files"].values()):
                continue
            try:
                theirs = read_snapshot_file(self.mirror, doc, "ledger.jsonl")
            except MirrorError:
                continue  # a file that fails its hash: that snapshot is not whole
            if theirs is None:
                return
            if not ledger.read_bytes().startswith(theirs[:theirs.rfind(b"\n") + 1]):
                raise NothingUnmirrored(
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
                if digest in self._suspect or (digest not in self._known and not blob.exists()):
                    for directory in (self.mirror / "blobs", blob.parent):
                        directory.mkdir(exist_ok=True)  # never a parent above the mirror
                    _write_atomic(blob, gzip.compress(data, compresslevel=GZIP_LEVEL, mtime=0))
                    self._suspect.discard(digest)
                self._known.add(digest)
                cached = self._cache[rel] = (key, digest, len(data))
            files[rel] = {"sha256": cached[1], "size": cached[2]}
        seq = self._seq + 1
        doc = {"schema": MIRROR_SCHEMA, "seq": seq, "files": files}
        _write_atomic(self.mirror / f"snapshot-{seq:08d}.json",
                      json.dumps(doc, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        self._seq = seq
        if self._restored is not None:  # a later resume in place is checked again
            (self.local / RESTORE_RECORD).unlink(missing_ok=True)
            self._restored = None
        # A manifest of a layout this code cannot read (MirrorError) found by the
        # collection is raised past this whole snapshot: newer code writes the mirror,
        # a second writer, which stops this one as a take-over does.
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
