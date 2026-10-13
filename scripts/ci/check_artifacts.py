"""Repository artifact checks for CI (issue #18).

Default mode (offline, gates PRs):
1. every tracked *.schema.json parses, declares JSON Schema draft 2020-12 and
   is itself a valid 2020-12 schema; $id values are unique;
2. the example genomes pass the EAG loader (schema + semantic checks): the
   doc example in architecture/eag-v0-schema.md and the G0 seed genome;
3. every tracked experiment manifest parses as a YAML mapping (style lint is
   yamllint's job, see .yamllint.yaml);
4. every relative Markdown link in a tracked *.md file resolves;
5. scripts/ci/requirements-ci.txt carries every pyproject dependency.

--external-links instead checks the citations in research/**/*.md for link
rot (SPEC section 30): http(s) URLs, `arXiv:<id>` references (checked as
https://arxiv.org/abs/<id>) and bare-domain links such as aclanthology.org/...
It needs network access and is meant for the scheduled workflow: HTTP 404/410,
hosts that do not exist and malformed URLs fail; other HTTP errors, timeouts
and connection failures are reported as unverifiable and do not fail.

Usage: python scripts/ci/check_artifacts.py [--external-links]
Requires Python >= 3.11 (tomllib).
"""

from __future__ import annotations

import argparse
import http.client
import json
import re
import socket
import subprocess
import sys
import tempfile
import time
import tomllib
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

DRAFT_2020_12 = "https://json-schema.org/draft/2020-12/schema"
REQUIREMENTS_CI = REPO_ROOT / "scripts" / "ci" / "requirements-ci.txt"

FENCED_BLOCK = re.compile(r"^ {0,3}(`{3,}|~{3,}).*?^ {0,3}\1", re.S | re.M)
INLINE_CODE = re.compile(r"`[^`\n]*`")
MD_LINK = re.compile(r"!?\[[^\]]*\]\(\s*<?([^)\s>]+)>?(?:\s+[\"'(][^)]*)?\s*\)")
REF_DEFINITION = re.compile(r"^ {0,3}\[[^\]]+\]:\s*<?([^\s>]+)", re.M)
NON_FILE_TARGET = re.compile(r"^(#|//|[a-zA-Z][a-zA-Z0-9+.-]*:)")
URL = re.compile(r"https?://[^\s)\]>\"'`*]+")
ARXIV_ID = re.compile(r"arXiv:\s?(\d{4}\.\d{4,5}(?:v\d+)?)")
BARE_DOMAIN = re.compile(r"(?<![\w./@:-])((?:[a-z0-9-]+\.)+(?:org|com|ai|io|net|dev|edu)/[^\s)\]>\"'`*]*)")
TRAILING_PUNCT = ".,;:_"
POLITE_DELAY_S = {"arxiv.org": 1.0}
DEAD_DNS_ERRORS = {getattr(socket, name) for name in ("EAI_NONAME", "EAI_NODATA") if hasattr(socket, name)}


def tracked(pattern: str) -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files", "-z", "--", pattern],
        cwd=REPO_ROOT, check=True, capture_output=True, text=True,
    ).stdout
    return [REPO_ROOT / p for p in out.split("\0") if p]


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


def check_requirements_sync() -> list[str]:
    project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    wanted = list(project.get("dependencies", []))
    wanted += project.get("optional-dependencies", {}).get("dev", [])
    have = {
        line.split("#", 1)[0].replace(" ", "").lower()
        for line in REQUIREMENTS_CI.read_text(encoding="utf-8").splitlines()
    }
    missing = [spec for spec in wanted if spec.replace(" ", "").lower() not in have]
    return [f"{rel(REQUIREMENTS_CI)}: missing pyproject dependency {spec!r}" for spec in missing]


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
        time.sleep(POLITE_DELAY_S.get(urllib.parse.urlsplit(url).hostname or "", 0.0))
        verdict, detail = probe(url)
        if verdict == "dead":
            errors.append(f"{source}: dead link {url} ({detail})")
        elif verdict == "unverifiable":
            unverifiable.append(f"{source}: unverifiable {url} ({detail})")
    for line in unverifiable:
        print(f"warning: {line}")
    print(f"external links: {len(urls)} checked, {len(unverifiable)} unverifiable")
    if urls and len(unverifiable) == len(urls):
        print("warning: no link could be verified; check the runner's network access")
    return errors


def probe(url: str) -> tuple[str, str]:
    """Return ("ok" | "dead" | "unverifiable", detail).

    Only evidence of rot counts as dead: HTTP 404/410, a host that does not
    exist (EAI_NONAME / EAI_NODATA, not the transient EAI_AGAIN) or a
    malformed URL. Bot blocking, rate limits, server errors, timeouts and
    other connection failures are unverifiable.
    """
    headers = {"User-Agent": "heritagent-link-check/0.1 (+https://github.com/sametyilmaztemel/heritagent)"}
    for method in ("HEAD", "GET"):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, method=method, headers=headers), timeout=20):
                return "ok", method
        except urllib.error.HTTPError as exc:
            if method == "HEAD" and exc.code in (403, 405, 501):
                continue
            return ("dead" if exc.code in (404, 410) else "unverifiable"), f"HTTP {exc.code}"
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, socket.gaierror) and exc.reason.errno in DEAD_DNS_ERRORS:
                return "dead", f"host does not exist: {exc.reason}"
            return "unverifiable", f"connection failed: {exc.reason}"
        except http.client.InvalidURL as exc:
            return "dead", f"malformed URL: {exc}"
        except UnicodeError as exc:
            return "unverifiable", f"non-ASCII URL not requested: {exc}"
        except ValueError as exc:
            return "dead", f"malformed URL: {exc}"
        except (TimeoutError, OSError, http.client.HTTPException) as exc:
            return "unverifiable", f"connection failed: {exc!r}"
    return "unverifiable", "HEAD and GET refused"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--external-links", action="store_true",
                        help="check http(s) URLs in research/ (network)")
    args = parser.parse_args()
    if args.external_links:
        checks = [check_external_links]
    else:
        checks = [check_schemas, check_example_genomes, check_manifests,
                  check_relative_links, check_requirements_sync]
    errors = [error for check in checks for error in check()]
    for error in errors:
        print(f"FAIL: {error}")
    print("OK" if not errors else f"{len(errors)} problem(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
