"""The ALFWorld data the frozen EXP-0001 manifests were built from (issue
#13 Phase B; ADR-0008 D6).

alfworld 0.4.2's own ``alfworld-download`` fetches the game files of
release 0.4.0 (``json_2.1.2_tw-pddl.zip``). Those ``game.tw-pddl`` files
differ from the ones the split and pilot manifests hash, so no game of
either manifest would open. The manifests were built from release 0.2.2.
``fetch_data`` downloads exactly that release's two archives, checks each
against its sha256, and extracts them. ``verify_data`` then hashes every
game of both manifests, before the model download and any episode."""

from __future__ import annotations

import hashlib
import shutil
import urllib.request
import zipfile
from collections.abc import Callable, Mapping
from pathlib import Path

from evaluation.benchmarks.alfworld.splits import _content_sha256
from evaluation.exp0001.backends import GAME_FILE, TRAJ_FILE, BackendError, manifest_content_index

ALFWORLD_RELEASE = "0.2.2"
ARCHIVES = {  # name -> sha256, release 0.2.2 (the data the manifests were built from)
    "json_2.1.1_json.zip": "25171f16e20ad7b048c47275c45b0babf3aa1cbab29cec97387922350a9844bc",
    "json_2.1.1_tw-pddl.zip": "d376c4bbb097c068528a2a9805e358e4cd14bc75c4d68a24e5141aebd234ef67",
}
URL = "https://github.com/alfworld/alfworld/releases/download/{release}/{name}"
DATA_DIR = "json_2.1.1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _download(url: str, path: Path) -> None:
    tmp = path.with_name(path.name + ".part")
    with urllib.request.urlopen(url, timeout=600) as response, tmp.open("wb") as out:
        shutil.copyfileobj(response, out, 1 << 20)
    tmp.replace(path)


def fetch_data(target: Path, *, download: Callable[[str, Path], None] = _download) -> Path:
    """Download (or reuse) the release archives into ``target``, check their
    sha256, extract them, and return ``target/json_2.1.1``."""
    target = Path(target)
    target.mkdir(parents=True, exist_ok=True)
    for name, expected in ARCHIVES.items():
        archive = target / name
        if not archive.exists() or _sha256_file(archive) != expected:
            download(URL.format(release=ALFWORLD_RELEASE, name=name), archive)
        actual = _sha256_file(archive)
        if actual != expected:
            raise BackendError(f"{name} has sha256 {actual}, not the pinned {expected} "
                               f"(ALFWorld release {ALFWORLD_RELEASE})")
        with zipfile.ZipFile(archive) as bundle:
            for member in bundle.namelist():
                parts = Path(member).parts
                if not parts or parts[0] != DATA_DIR or ".." in parts or Path(member).is_absolute():
                    raise BackendError(f"{name} holds an unexpected path {member!r}")
            bundle.extractall(target)
    return target / DATA_DIR


def verify_data(data_root: Path, split_manifest: Mapping, pilot_manifest: Mapping) -> dict:
    """Hash every game of the split and pilot manifests under ``data_root``
    (ALFWorld's ``json_2.1.1``); any missing or different game fails closed."""
    root = Path(data_root)
    index = manifest_content_index(split_manifest)
    index.update({("train", g["game_id"]): g["content_sha256"] for g in pilot_manifest["games"]})
    missing, mismatched = [], []
    for (split, game_id), expected in sorted(index.items()):
        trial = root / split / game_id
        try:
            actual = _content_sha256((trial / TRAJ_FILE).read_bytes(),
                                     (trial / GAME_FILE).read_bytes())
        except OSError:
            missing.append(f"{split}/{game_id}")
            continue
        if actual != expected:
            mismatched.append(f"{split}/{game_id}")
    if missing or mismatched:
        raise BackendError(f"{len(missing)} manifest games are missing and {len(mismatched)} differ "
                           f"from their frozen content hash under {root} (first: "
                           f"{(missing + mismatched)[0]}); fetch ALFWorld release "
                           f"{ALFWORLD_RELEASE} with `cli fetch-data`, not `alfworld-download`")
    return {"games": len(index), "release": ALFWORLD_RELEASE, "archives": dict(ARCHIVES),
            "data_root": str(root)}
