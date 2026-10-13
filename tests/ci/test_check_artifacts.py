"""Regression tests for scripts/ci/check_artifacts.py (issue #18 CI checks):
link extraction and the external-link classification, without network."""

import http.client
import importlib.util
import io
import socket
import urllib.error
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "ci" / "check_artifacts.py"


@pytest.fixture(scope="module")
def ca():
    spec = importlib.util.spec_from_file_location("check_artifacts", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("text,expected", [
    ("Sources: arxiv.org/abs/2010.14592 · proceedings.mlr.press/v130/wang21b.html",
     ["https://arxiv.org/abs/2010.14592", "https://proceedings.mlr.press/v130/wang21b.html"]),
    ("see arXiv:2502.00510 and ARXIV:2010.14592v2", ["https://arxiv.org/abs/2502.00510",
                                                     "https://arxiv.org/abs/2010.14592v2"]),
    ("https://arxiv.org/abs/2010.14592[^1] in prose", ["https://arxiv.org/abs/2010.14592"]),
    ("**https://x.org/a**, then aclanthology.org/2026.findings-acl.359/.",
     ["https://x.org/a", "https://aclanthology.org/2026.findings-acl.359/"]),
    ("_arxiv.org/abs/2010.14592_", ["https://arxiv.org/abs/2010.14592"]),
    ("GitHub.com/jennyzzt/dgm", ["https://GitHub.com/jennyzzt/dgm"]),
    # code and file names are not hosts
    ("llama.cpp/vLLM, os.path/pathlib, context.tags/metadata, tf.keras/torch", []),
    ("genome/validation/loader.py/x and research/novelty-ledger.md/ and a@b.com/x", []),
])
def test_citation_extraction(ca, text, expected):
    assert ca.citation_urls(text) == expected


def test_footnote_definitions_are_not_links(ca):
    text = "[^1]: Some explanatory text.\n[ref]: ok.md\n"
    assert ca.REF_DEFINITION.findall(text) == ["ok.md"]


def raising(exc):
    def urlopen(*args, **kwargs):
        raise exc
    return urlopen


def http_error(code):
    return urllib.error.HTTPError("u", code, "x", {}, io.BytesIO())


@pytest.mark.parametrize("exc,verdict", [
    (urllib.error.URLError(socket.gaierror(socket.EAI_NONAME, "unknown")), "dead"),
    (urllib.error.URLError(socket.gaierror(socket.EAI_AGAIN, "temporary")), "unverifiable"),
    (http_error(404), "dead"),
    (http_error(410), "dead"),
    (http_error(403), "unverifiable"),
    (http_error(429), "unverifiable"),
    (http_error(500), "unverifiable"),
    (TimeoutError("timed out"), "unverifiable"),
    (urllib.error.URLError(ConnectionRefusedError(111, "refused")), "unverifiable"),
    (http.client.BadStatusLine("x"), "unverifiable"),
    (http.client.InvalidURL("bad"), "dead"),
    (ValueError("Invalid IPv6 URL"), "dead"),
    (UnicodeEncodeError("ascii", "é", 0, 1, "x"), "unverifiable"),
])
def test_probe_classification(ca, monkeypatch, exc, verdict):
    monkeypatch.setattr(ca.urllib.request, "urlopen", raising(exc))
    monkeypatch.setattr(ca.time, "sleep", lambda s: None)
    assert ca.probe("https://example.org/")[0] == verdict


def test_head_failure_falls_through_to_get(ca, monkeypatch):
    calls = []

    class Ok:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def urlopen(request, timeout):
        calls.append(request.get_method())
        if request.get_method() == "HEAD":
            raise urllib.error.URLError(ConnectionResetError(104, "reset"))
        return Ok()

    monkeypatch.setattr(ca.urllib.request, "urlopen", urlopen)
    assert ca.probe("https://example.org/") == ("ok", "GET")
    assert calls == ["HEAD", "GET"]


def test_head_404_is_not_final(ca, monkeypatch):
    class Ok:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def urlopen(request, timeout):
        if request.get_method() == "HEAD":
            raise http_error(404)
        return Ok()

    monkeypatch.setattr(ca.urllib.request, "urlopen", urlopen)
    assert ca.probe("https://example.org/") == ("ok", "GET")


def test_bracketed_placeholder_is_not_extracted(ca):
    assert ca.citation_urls("Placeholder: https://[HOST]/path") == []


def test_malformed_url_is_reported_not_raised(ca, monkeypatch, tmp_path):
    monkeypatch.setattr(ca, "REPO_ROOT", tmp_path)
    (tmp_path / "research").mkdir()
    (tmp_path / "research" / "note.md").write_text("see the note\n")
    monkeypatch.setattr(ca, "citation_urls", lambda text: ["https://[::1/x"])
    errors = ca.check_external_links()
    assert len(errors) == 1 and "malformed URL" in errors[0]


def test_no_citations_fails_closed(ca, monkeypatch, tmp_path):
    monkeypatch.setattr(ca, "REPO_ROOT", tmp_path)
    (tmp_path / "research").mkdir()
    (tmp_path / "research" / "note.md").write_text("no links here\n")
    assert ca.check_external_links() == ["no citations found in research/; the extraction is broken"]


def test_fallback_walk_only_without_git(ca, monkeypatch, tmp_path):
    monkeypatch.setattr(ca, "REPO_ROOT", tmp_path)
    (tmp_path / "a.md").write_text("x")
    (tmp_path / "build" / "lib").mkdir(parents=True)
    (tmp_path / "build" / "lib" / "b.md").write_text("x")
    (tmp_path / "research" / "results").mkdir(parents=True)
    (tmp_path / "research" / "results" / "c.md").write_text("x")
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "d.md").write_text("x")
    assert [ca.rel(p) for p in ca.tracked("*.md")] == ["a.md", "research/results/c.md"]
    (tmp_path / ".git").mkdir()  # a broken work tree: git must decide, not the walk
    with pytest.raises(Exception):
        ca.tracked("*.md")
