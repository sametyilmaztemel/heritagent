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


# -- preregistration manifest invariants (issue #6, SPEC section 33) ---------


# the LOCKED primary support criteria, mirrored here so fixture and checker
# cannot silently drift (cross-checked against the script constants below)
LOCKED_CRITERIA = {
    "h2_supported_if": "C-B >= +5pp on T_eval, CI excl. 0, AND reg(C) < reg(B), CI excl. 0",
    "h1_supported_if": "C-A > 0, CI excl. 0",
    "seed_consistency": "direction holds in >= 2/3 mining seeds",
}


def valid_prereg_manifest():
    """A minimal manifest satisfying every preregistration invariant
    (mirrors the locked EXP-0001 prereg)."""
    return {
        "experiment": {
            "id": "EXP-0001",
            "status": "preregistered-locked",
            "preregistration": {
                "locked": True,
                "expected_results_filled": True,
                "expected_results": {
                    "h2": {
                        "direction": "C>B",
                        "heldout_success_pp": {"central": 7, "range": [5, 10]},
                        "regression_difference_pp": {
                            "direction": "C<B",
                            "expected_reduction_range": [3, 8],
                        },
                    },
                    "h1": {
                        "direction": "C>A",
                        "heldout_success_pp": {"central": 10, "range": [5, 15]},
                    },
                    "gate_survival_per_seed": {
                        "frozen_candidates": 10,
                        "replay": 6,
                        "ablation": 4,
                        "generalization_combined": 4,
                        "interaction_regression_overhead": 3,
                        "final_promoted": 3,
                    },
                    "compute_cap": {
                        "accelerator_nodes_max": 1,
                        "episodes_max": 7000,
                        "model_calls_max": 225000,
                        "wall_clock_hours_max": 96,
                        "mining_seeds": [11, 23, 41],
                    },
                },
            },
        },
        "evaluation": {
            "statistics": dict(LOCKED_CRITERIA),
        },
    }


def _load_module():
    spec = importlib.util.spec_from_file_location("check_artifacts_prereg", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prereg_errors(mutate):
    manifest = valid_prereg_manifest()
    mutate(manifest)
    return _load_module().check_preregistration(manifest, "EXP-0001-test")


def test_valid_prereg_manifest_passes():
    assert _load_module().check_preregistration(
        valid_prereg_manifest(), "EXP-0001-test") == []


def test_manifest_without_preregistration_is_allowed():
    manifest = {"experiment": {"id": "EXP-XXXX", "status": "design-revised"}}
    assert _load_module().check_preregistration(manifest, "m") == []


def test_missing_or_false_prereg_flags_rejected():
    assert any("expected_results_filled must be true" in e
                for e in prereg_errors(lambda m: m["experiment"]["preregistration"]
                                        .update(expected_results_filled=False)))
    assert any("locked must be true" in e
                for e in prereg_errors(lambda m: m["experiment"]["preregistration"]
                                        .update(locked=False)))


def test_filled_prereg_requires_locked_status_and_forbids_post_run_statuses():
    assert any("preregistered-locked" in e
                for e in prereg_errors(lambda m: m["experiment"].update(status="design-revised")))
    for status in ("running", "completed", "executed"):
        assert any("post-run status" in e
                    for e in prereg_errors(lambda m, s=status: m["experiment"].update(status=s)))


def test_expectation_range_must_bracket_central():
    def mutate(manifest):
        heldout = manifest["experiment"]["preregistration"]["expected_results"]["h2"]["heldout_success_pp"]
        heldout.update(central=12)  # outside [5, 10]
    assert any("min <= central" in e for e in prereg_errors(mutate))


def test_gate_survival_bounds_and_monotonicity():
    def below_zero(manifest):
        gate = manifest["experiment"]["preregistration"]["expected_results"]["gate_survival_per_seed"]
        gate.update(replay=-1)
    def above_k(manifest):
        gate = manifest["experiment"]["preregistration"]["expected_results"]["gate_survival_per_seed"]
        gate.update(final_promoted=11)
    def exceeds_preceding(manifest):
        gate = manifest["experiment"]["preregistration"]["expected_results"]["gate_survival_per_seed"]
        gate.update(final_promoted=5)  # > interaction_regression_overhead (3)
    assert any("outside 0..K" in e for e in prereg_errors(below_zero))
    assert any("outside 0..K" in e for e in prereg_errors(above_k))
    assert any("exceeds the preceding stage" in e for e in prereg_errors(exceeds_preceding))


def test_hard_caps_positive_and_seeds_locked():
    def zero_cap(manifest):
        cap = manifest["experiment"]["preregistration"]["expected_results"]["compute_cap"]
        cap.update(episodes_max=0)
    def wrong_seeds(manifest):
        cap = manifest["experiment"]["preregistration"]["expected_results"]["compute_cap"]
        cap.update(mining_seeds=[11, 23])
    assert any("must be a positive int" in e for e in prereg_errors(zero_cap))
    assert any("must be exactly [11, 23, 41]" in e for e in prereg_errors(wrong_seeds))


def test_locked_criteria_constants_match_checker(ca):
    """The mirrored fixture constants and the script's locked constants
    must stay identical."""
    assert LOCKED_CRITERIA["h2_supported_if"] == ca.LOCKED_H2_CRITERION
    assert LOCKED_CRITERIA["h1_supported_if"] == ca.LOCKED_H1_CRITERION
    assert LOCKED_CRITERIA["seed_consistency"] == ca.LOCKED_SEED_CRITERION


def test_primary_support_criteria_must_stay_unchanged():
    def move_h2(manifest):
        manifest["evaluation"]["statistics"]["h2_supported_if"] = "C-B >= +7pp on T_eval, CI excl. 0"
    def move_h1(manifest):
        manifest["evaluation"]["statistics"]["h1_supported_if"] = "C-A >= +10pp, CI excl. 0"
    assert any("h2_supported_if" in e and "changed" in e
                for e in prereg_errors(move_h2))
    assert any("h1_supported_if" in e and "changed" in e
                for e in prereg_errors(move_h1))


def test_no_t_eval_tuning_fields_in_prereg():
    def add_teval_field(manifest):
        manifest["experiment"]["preregistration"]["expected_results"]["t_eval_tuning"] = {
            "threshold_source": "T_eval"}
    assert any("T_eval" in e for e in prereg_errors(add_teval_field))


def test_required_prereg_sections_must_exist():
    def drop_results(manifest):
        manifest["experiment"]["preregistration"]["expected_results"] = "todo"
    def drop_gate(manifest):
        del manifest["experiment"]["preregistration"]["expected_results"]["gate_survival_per_seed"]
    def drop_cap(manifest):
        del manifest["experiment"]["preregistration"]["expected_results"]["compute_cap"]
    assert any("must be a mapping" in e for e in prereg_errors(drop_results))
    assert any("gate_survival_per_seed must be a mapping" in e for e in prereg_errors(drop_gate))
    assert any("compute_cap must be a mapping" in e for e in prereg_errors(drop_cap))


def test_real_exp0001_manifest_satisfies_prereg_invariants(ca):
    """End-to-end: parse the REAL tracked manifest and run the invariant
    checker (requires pyyaml; skipped in envs without it — the CI
    artifacts job always has it)."""
    yaml = pytest.importorskip("yaml")
    path = ca.REPO_ROOT / "experiments" / "manifests" / "EXP-0001-minimal-cig.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert ca.check_preregistration(data, "EXP-0001-minimal-cig.yaml") == []
