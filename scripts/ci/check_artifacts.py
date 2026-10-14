"""Repository artifact checks for CI (issue #18).

Default mode (offline, gates PRs):
1. every tracked *.schema.json parses, declares JSON Schema draft 2020-12 and
   is itself a valid 2020-12 schema; $id values are unique;
2. the example genomes pass the EAG loader (schema + semantic checks): the
   doc example in architecture/eag-v0-schema.md and the G0 seed genome;
3. every tracked experiment manifest parses as a YAML mapping (style lint is
   yamllint's job, see .yamllint.yaml);
4. every relative Markdown link in a tracked *.md file resolves.

--external-links instead checks the citations in research/**/*.md for link
rot (SPEC section 30): http(s) URLs, `arXiv:<id>` references (checked as
https://arxiv.org/abs/<id>) and bare-domain links such as aclanthology.org/...
It needs network access and is meant for the scheduled workflow: HTTP 404/410,
hosts that do not exist and malformed URLs fail; other HTTP errors, timeouts
and connection failures are reported as unverifiable and do not fail.

Usage: python scripts/ci/check_artifacts.py [--external-links]
"""

from __future__ import annotations

import argparse
import fnmatch
import http.client
import json
import re
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

DRAFT_2020_12 = "https://json-schema.org/draft/2020-12/schema"

# Locked EXP-0001 preregistration invariants (issue #6, SPEC section 33).
PREREGISTERED_STATUS = "preregistered-locked"
# EXP-0001 lifecycle vocabulary: preregistration is validated at EVERY
# stage — running/completed does NOT exempt a manifest from its locked
# prereg (drift must still fail CI). "executed" is not canonical vocabulary
# in this repository and is not allowed.
EXP0001_LIFECYCLE_STATUSES = {PREREGISTERED_STATUS, "running", "completed",
                               "incomplete", "aborted"}
EXP0001_ID = "EXP-0001"
PREREG_GATE_STAGES = ("replay", "ablation", "generalization_combined",
                      "interaction_regression_overhead", "final_promoted")
PREREG_CAP_FIELDS = ("accelerator_nodes_max", "episodes_max",
                     "model_calls_max", "wall_clock_hours_max")
LOCKED_MINING_SEEDS = [11, 23, 41]
# locked PRIMARY support criteria (design section 5 / ADR-0002): pre-registered
# expectations may never replace or alter these decision rules
LOCKED_H2_CRITERION = "C-B >= +5pp on T_eval, CI excl. 0, AND reg(C) < reg(B), CI excl. 0"
LOCKED_H1_CRITERION = "C-A > 0, CI excl. 0"
LOCKED_SEED_CRITERION = "direction holds in >= 2/3 mining seeds"


def _valid_iso_date(value) -> bool:
    from datetime import date
    if not isinstance(value, str):
        return False
    try:
        date.fromisoformat(value)
    except ValueError:
        return False
    return True


def _generic_prereg_checks(prereg: dict, err) -> None:
    """Minimal structural checks that apply to EVERY experiment manifest:
    a preregistration block claims `locked` and, when present, a valid
    ISO-YYYY-MM-DD lock date. EXP-0001-specific values are NOT imposed on
    future experiments."""
    if prereg.get("locked") is not True:
        err("preregistration.locked must be true")
    if "locked_at" in prereg and \
            not _valid_iso_date(prereg["locked_at"]):
        err(f"preregistration.locked_at must be an ISO YYYY-MM-DD date, "
            f"got {prereg['locked_at']!r}")


def _enforce_exp0001_locked_values(manifest: dict, prereg: dict, err) -> None:
    """EXP-0001's locked scientific preregistration: enforced at EVERY
    lifecycle status (preregistered-locked, running, completed, incomplete,
    aborted) — a status change never exempts the locked values."""
    if prereg.get("expected_results_filled") is not True:
        err("preregistration.expected_results_filled must be true (the "
            "preregistration is locked and can never revert to unfilled)")
    results = prereg.get("expected_results")
    if not isinstance(results, dict):
        err("preregistration.expected_results must be a mapping")
        return

    def heldout_block(hypothesis: str):
        block = results.get(hypothesis)
        if not isinstance(block, dict):
            err(f"expected_results.{hypothesis} must be a mapping")
            return None
        if not isinstance(block.get("direction"), str) or not block["direction"]:
            err(f"expected_results.{hypothesis}.direction must be a "
                f"non-empty string")
        heldout = block.get("heldout_success_pp")
        if not isinstance(heldout, dict):
            err(f"expected_results.{hypothesis}.heldout_success_pp must be "
                f"a mapping")
            return None
        central = heldout.get("central")
        spread = heldout.get("range")
        if not isinstance(central, (int, float)) or isinstance(central, bool):
            err(f"expected_results.{hypothesis}.heldout_success_pp.central "
                f"must be numeric, got {central!r}")
            return None
        if not (isinstance(spread, list) and len(spread) == 2 and all(
                isinstance(v, (int, float)) and not isinstance(v, bool)
                for v in spread)):
            err(f"expected_results.{hypothesis}.heldout_success_pp.range "
                f"must be [min, max] numerics, got {spread!r}")
            return None
        if not spread[0] <= central <= spread[1]:
            err(f"expected_results.{hypothesis}: range {spread} must "
                f"satisfy min <= central ({central}) <= max")
        return block

    h2 = heldout_block("h2")
    h1 = heldout_block("h1")
    if isinstance(h2, dict):
        regression = h2.get("regression_difference_pp")
        if not isinstance(regression, dict) or \
                not isinstance(regression.get("direction"), str):
            err("expected_results.h2.regression_difference_pp must be a "
                "mapping with a direction")
        elif not (isinstance(regression.get("expected_reduction_range"), list)
                  and len(regression["expected_reduction_range"]) == 2):
            err("expected_results.h2.regression_difference_pp."
                "expected_reduction_range must be [min, max]")
    if isinstance(h1, dict) and isinstance(h2, dict):
        if h2.get("direction") == h1.get("direction"):
            err("expected_results.h2.direction and expected_results.h1."
                "direction must describe distinct arm contrasts")

    gate = results.get("gate_survival_per_seed")
    if not isinstance(gate, dict):
        err("expected_results.gate_survival_per_seed must be a mapping")
    else:
        frozen = gate.get("frozen_candidates")
        if not isinstance(frozen, int) or isinstance(frozen, bool) or frozen <= 0:
            err("gate_survival_per_seed.frozen_candidates must be a "
                "positive int, got {0!r}".format(frozen))
        else:
            previous = None
            for stage in PREREG_GATE_STAGES:
                value = gate.get(stage)
                if not isinstance(value, int) or isinstance(value, bool):
                    err(f"gate_survival_per_seed.{stage} must be an int, "
                        f"got {value!r}")
                    continue
                if not 0 <= value <= frozen:
                    err(f"gate_survival_per_seed.{stage}={value} outside "
                        f"0..K ({frozen})")
                if previous is not None and value > previous:
                    err(f"gate_survival_per_seed.{stage}={value} exceeds the "
                        f"preceding stage expectation {previous}")
                previous = value

    cap = results.get("compute_cap")
    if not isinstance(cap, dict):
        err("expected_results.compute_cap must be a mapping")
    else:
        for field in PREREG_CAP_FIELDS:
            value = cap.get(field)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                err(f"compute_cap.{field} must be a positive int, got {value!r}")
        if cap.get("mining_seeds") != LOCKED_MINING_SEEDS:
            err(f"compute_cap.mining_seeds must be exactly "
                f"{LOCKED_MINING_SEEDS}, got {cap.get('mining_seeds')!r}")

    # no T_eval threshold-tuning fields inside the preregistration block
    def scan(node, path):
        if isinstance(node, dict):
            for key, value in node.items():
                flattened = str(key).lower().replace("_", "").replace("-", "")
                if "teval" in flattened:
                    err(f"prereg must not introduce T_eval-driven fields: "
                        f"{path}.{key}")
                scan(value, f"{path}.{key}")
        elif isinstance(node, list):
            for index, value in enumerate(node):
                scan(value, f"{path}[{index}]")

    scan(prereg, "preregistration")

    # the PRIMARY support criteria are locked decision rules — expectations
    # never replace them, at any lifecycle status
    statistics = (manifest.get("evaluation") or {}).get("statistics") or {}
    for field, locked in (("h2_supported_if", LOCKED_H2_CRITERION),
                          ("h1_supported_if", LOCKED_H1_CRITERION),
                          ("seed_consistency", LOCKED_SEED_CRITERION)):
        actual = statistics.get(field)
        if actual != locked:
            err(f"locked primary support criterion {field} changed: "
                f"{actual!r} != {locked!r}")


def check_preregistration(manifest: dict, label: str) -> list[str]:
    """Manifest preregistration checks (issue #6, SPEC section 33).

    Lifecycle-safe and scoped: a preregistration block claiming
    ``locked`` gets the generic checks (locked flag, ISO lock date). The
    EXP-0001-specific locked scientific values are enforced ONLY for
    ``experiment.id == EXP-0001`` — but at EVERY lifecycle status:
    ``running`` / ``completed`` / ``incomplete`` / ``aborted`` must keep
    the exact prereg (drift still fails CI), and ``preregistered-locked``
    is the only valid pre-execution status. Future experiment manifests
    (EXP-0002, ...) are never constrained by EXP-0001's seeds, criteria
    strings, gate shape or caps."""
    issues: list[str] = []

    def err(message: str) -> None:
        issues.append(f"{label}: {message}")

    experiment = manifest.get("experiment")
    if not isinstance(experiment, dict):
        return issues  # not an experiment manifest body
    prereg = experiment.get("preregistration")
    if prereg is None:
        return issues  # pre-preregistration manifests are allowed
    if not isinstance(prereg, dict):
        err(f"preregistration must be a mapping, got {type(prereg).__name__}")
        return issues

    _generic_prereg_checks(prereg, err)

    if experiment.get("id") != EXP0001_ID:
        return issues  # future experiments: generic checks only

    status = experiment.get("status")
    if status not in EXP0001_LIFECYCLE_STATUSES:
        err(f"invalid EXP-0001 lifecycle status {status!r}; allowed: "
            f"{sorted(EXP0001_LIFECYCLE_STATUSES)}")
    # locked values are enforced REGARDLESS of the (valid or invalid)
    # status — a status change never exempts the preregistration
    _enforce_exp0001_locked_values(manifest, prereg, err)
    return issues

FENCED_BLOCK = re.compile(r"^ {0,3}(`{3,}|~{3,}).*?^ {0,3}\1", re.S | re.M)
INLINE_CODE = re.compile(r"`[^`\n]*`")
MD_LINK = re.compile(r"!?\[[^\]]*\]\(\s*<?([^)\s>]+)>?(?:\s+[\"'(][^)]*)?\s*\)")
REF_DEFINITION = re.compile(r"^ {0,3}\[(?!\^)[^\]]+\]:\s*<?([^\s>]+)", re.M)  # [^n]: is a footnote
NON_FILE_TARGET = re.compile(r"^(#|//|[a-zA-Z][a-zA-Z0-9+.-]*:)")
URL_CHARS = r"[^\s()\[\]<>\"'`*]"  # '[' too: a footnote marker glued to a URL is not part of it
URL = re.compile(rf"https?://{URL_CHARS}+")
ARXIV_ID = re.compile(r"arXiv:\s?(\d{4}\.\d{4,5}(?:v\d+)?)", re.I)
# Bare-domain citations (aclanthology.org/..., proceedings.mlr.press/...). A
# list of real TLDs, not "any letters": code names such as llama.cpp/vLLM or
# os.path/pathlib must not become hosts that the weekly job reports dead.
CITATION_TLDS = ("com|org|net|edu|gov|int|io|ai|dev|app|co|me|info|press|science|"
                 "tech|cloud|page|site|uk|us|eu|de|fr|nl|ch|cn|jp|kr|ca|au|it|es|"
                 "se|no|fi|dk|pl|ru|br|il|sg|hk|tw|at|be|cz|pt|ie|nz|in")
BARE_DOMAIN = re.compile(
    rf"(?<![A-Za-z0-9./@:-])((?:[a-z0-9-]+\.)+(?:{CITATION_TLDS})/{URL_CHARS}*)", re.I)
TRAILING_PUNCT = ".,;:_"
POLITE_DELAY_S = {"arxiv.org": 1.0}
DEAD_DNS_ERRORS = {getattr(socket, name) for name in ("EAI_NONAME", "EAI_NODATA") if hasattr(socket, name)}


def tracked(pattern: str) -> list[Path]:
    """Files matching a git pathspec glob (``*`` crosses directories).

    Only a tree with no git metadata at the root (e.g. a ``git archive``
    export) falls back to a filesystem walk that skips what .gitignore
    excludes. In a git work tree a failing ``git ls-files`` is an error, not
    a reason to scan untracked files."""
    if not (REPO_ROOT / ".git").exists():
        return sorted(p for p in REPO_ROOT.rglob("*")
                      if p.is_file() and not _untracked_by_convention(p)
                      and fnmatch.fnmatchcase(rel(p), pattern))
    out = subprocess.run(
        ["git", "ls-files", "-z", "--", pattern],
        cwd=REPO_ROOT, check=True, capture_output=True, text=True,
    ).stdout
    return [REPO_ROOT / p for p in out.split("\0") if p]


# Directories .gitignore excludes (build output, virtualenvs, caches): without
# git they would otherwise be scanned, e.g. build/lib copies of every schema.
# Mirrors .gitignore: the first set matches at any depth, the second only at
# the root (/results/, /checkpoints/, /runs/, /data/).
UNTRACKED_DIRS_ANYWHERE = {".git", "build", "dist", ".venv", "venv", "__pycache__",
                           ".pytest_cache", ".mypy_cache", ".ruff_cache"}
UNTRACKED_DIRS_AT_ROOT = {"results", "checkpoints", "runs", "data"}


def _untracked_by_convention(path: Path) -> bool:
    dirs = path.relative_to(REPO_ROOT).parts[:-1]
    if dirs and dirs[0] in UNTRACKED_DIRS_AT_ROOT:
        return True
    return any(part in UNTRACKED_DIRS_ANYWHERE or part.endswith(".egg-info") for part in dirs)


def rel(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def check_schemas() -> list[str]:
    from jsonschema import Draft202012Validator
    from jsonschema.exceptions import SchemaError

    errors, ids = [], {}
    paths = tracked("*.schema.json")
    if not paths:
        return ["no *.schema.json files found"]
    for path in paths:
        try:
            schema = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            errors.append(f"{rel(path)}: invalid JSON: {exc}")
            continue
        if schema.get("$schema") != DRAFT_2020_12:
            errors.append(f"{rel(path)}: $schema is {schema.get('$schema')!r}, expected {DRAFT_2020_12!r}")
        try:
            Draft202012Validator.check_schema(schema)
        except SchemaError as exc:
            errors.append(f"{rel(path)}: not a valid 2020-12 schema: {exc.message}")
        if "$id" in schema:
            if schema["$id"] in ids:
                errors.append(f"{rel(path)}: $id {schema['$id']!r} also used by {ids[schema['$id']]}")
            ids[schema["$id"]] = rel(path)
    print(f"schemas: {len(paths)} checked")
    return errors


def check_example_genomes() -> list[str]:
    from genome.validation.errors import GenomeValidationError
    from genome.validation.loader import load_genome
    from genome.validation.registry import TraitRegistry
    from runtime.seed_g0 import build_g0_seed_genome

    errors = []
    doc = REPO_ROOT / "architecture" / "eag-v0-schema.md"
    examples = []
    for block in re.findall(r"```json\n(.*?)```", doc.read_text(encoding="utf-8"), re.S):
        data = json.loads(block)
        if "genome_id" in data and "genes" in data:
            examples.append(data)
    if not examples:
        errors.append(f"{rel(doc)}: no example genome found")
    for data in examples:
        try:
            load_genome(data)
        except GenomeValidationError as exc:
            errors.append(f"{rel(doc)}: example genome {data['genome_id']} rejected: {exc}")
    with tempfile.TemporaryDirectory() as tmp:
        registry = TraitRegistry(Path(tmp) / "registry")
        g0 = build_g0_seed_genome(registry)
        try:
            load_genome(g0, registry)
        except GenomeValidationError as exc:
            errors.append(f"runtime/seed_g0.py: G0 seed genome rejected: {exc}")
    print(f"example genomes: {len(examples) + 1} checked")
    return errors


def check_manifests() -> list[str]:
    import yaml

    errors = []
    paths = tracked("experiments/manifests/*.yaml") + tracked("experiments/manifests/*.yml")
    if not paths:
        return ["no experiment manifests found"]
    for path in paths:
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            errors.append(f"{rel(path)}: invalid YAML: {exc}")
            continue
        if not isinstance(data, dict):
            errors.append(f"{rel(path)}: top level is {type(data).__name__}, expected a mapping")
            continue
        errors.extend(check_preregistration(data, rel(path)))
    print(f"manifests: {len(paths)} checked")
    return errors


def markdown_text(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    return INLINE_CODE.sub("", FENCED_BLOCK.sub("", text))


def check_relative_links() -> list[str]:
    errors, count = [], 0
    for path in tracked("*.md"):
        text = markdown_text(path)
        for target in MD_LINK.findall(text) + REF_DEFINITION.findall(text):
            if NON_FILE_TARGET.match(target):
                continue
            count += 1
            clean = urllib.parse.unquote(target.split("#", 1)[0].split("?", 1)[0])
            base = REPO_ROOT if clean.startswith("/") else path.parent
            resolved = (base / clean.lstrip("/")).resolve()
            if not resolved.is_relative_to(REPO_ROOT) or not resolved.exists():
                errors.append(f"{rel(path)}: broken relative link {target!r}")
    print(f"relative links: {count} checked")
    return errors


def citation_urls(text: str) -> list[str]:
    """http(s) URLs, arXiv ids and bare-domain links, normalised to https URLs."""
    found = [url.rstrip(TRAILING_PUNCT) for url in URL.findall(text)]
    found += [f"https://arxiv.org/abs/{arxiv_id}" for arxiv_id in ARXIV_ID.findall(text)]
    found += [f"https://{bare.rstrip(TRAILING_PUNCT)}" for bare in BARE_DOMAIN.findall(text)]
    return found


def check_external_links() -> list[str]:
    urls = {}
    for path in tracked("research/*.md"):
        for url in citation_urls(path.read_text(encoding="utf-8")):
            urls.setdefault(url, rel(path))
    errors, unverifiable = [], []
    for url, source in sorted(urls.items()):
        try:
            host = urllib.parse.urlsplit(url).hostname or ""
        except ValueError as exc:
            errors.append(f"{source}: dead link {url} (malformed URL: {exc})")
            continue
        time.sleep(POLITE_DELAY_S.get(host, 0.0))
        verdict, detail = probe(url)
        if verdict == "dead":
            errors.append(f"{source}: dead link {url} ({detail})")
        elif verdict == "unverifiable":
            unverifiable.append(f"{source}: unverifiable {url} ({detail})")
    for line in unverifiable:
        print(f"warning: {line}")
    print(f"external links: {len(urls)} checked, {len(unverifiable)} unverifiable")
    if not urls:
        errors.append("no citations found in research/; the extraction is broken")
    elif len(unverifiable) == len(urls):
        print("warning: no link could be verified; check the runner's network access")
    return errors


def probe(url: str) -> tuple[str, str]:
    """Return ("ok" | "dead" | "unverifiable", detail).

    Only evidence of rot counts as dead: HTTP 404/410 on GET, a host that does
    not exist (EAI_NONAME / EAI_NODATA, not the transient EAI_AGAIN) or a
    malformed URL. Bot blocking, rate limits, server errors, timeouts and
    other connection failures are unverifiable. HEAD is only a shortcut: any
    HEAD failure other than a dead host or a malformed URL falls through to
    GET, which decides."""
    headers = {"User-Agent": "heritagent-link-check/0.1 (+https://github.com/sametyilmaztemel/heritagent)"}
    try:
        _request(url, "HEAD", headers)
        return "ok", "HEAD"
    except urllib.error.HTTPError as exc:
        if exc.code in (429, 503):
            time.sleep(1.0)
    except Exception as exc:  # noqa: BLE001 - classified below, then GET decides
        verdict = _classify_failure(exc)
        if verdict[0] == "dead":
            return verdict
    try:
        _request(url, "GET", headers)
        return "ok", "GET"
    except urllib.error.HTTPError as exc:
        return ("dead" if exc.code in (404, 410) else "unverifiable"), f"HTTP {exc.code}"
    except Exception as exc:  # noqa: BLE001
        return _classify_failure(exc)


def _request(url: str, method: str, headers: dict) -> None:
    with urllib.request.urlopen(urllib.request.Request(url, method=method, headers=headers),
                                timeout=20):
        pass


def _classify_failure(exc: Exception) -> tuple[str, str]:
    if isinstance(exc, urllib.error.URLError):
        if isinstance(exc.reason, socket.gaierror) and exc.reason.errno in DEAD_DNS_ERRORS:
            return "dead", f"host does not exist: {exc.reason}"
        return "unverifiable", f"connection failed: {exc.reason}"
    if isinstance(exc, UnicodeError):
        return "unverifiable", f"non-ASCII URL not requested: {exc}"
    if isinstance(exc, (ValueError, http.client.InvalidURL)):  # InvalidURL is an HTTPException
        return "dead", f"malformed URL: {exc}"
    if isinstance(exc, (TimeoutError, OSError, http.client.HTTPException)):
        return "unverifiable", f"connection failed: {exc!r}"
    raise exc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--external-links", action="store_true",
                        help="check http(s) URLs in research/ (network)")
    args = parser.parse_args()
    if args.external_links:
        checks = [check_external_links]
    else:
        checks = [check_schemas, check_example_genomes, check_manifests,
                  check_relative_links]
    errors = [error for check in checks for error in check()]
    for error in errors:
        print(f"FAIL: {error}")
    print("OK" if not errors else f"{len(errors)} problem(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
