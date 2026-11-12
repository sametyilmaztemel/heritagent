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


def test_exp0001_lifecycle_statuses_all_pass_with_same_prereg():
    """preregistration is validated at EVERY lifecycle status: moving to
    running/completed/incomplete/aborted never exempts the locked prereg
    (the identical prereg block passes at each stage)."""
    for status in ("preregistered-locked", "running", "completed",
                    "incomplete", "aborted"):
        errors = prereg_errors(lambda m, s=status: m["experiment"].update(status=s))
        assert errors == [], (status, errors)


def test_invalid_exp0001_lifecycle_statuses_rejected():
    for status in ("executed", "design-revised", "design", None, 42):
        errors = prereg_errors(
            lambda m, s=status: m["experiment"].update(status=s))
        assert any("invalid EXP-0001 lifecycle status" in e
                    for e in errors), status


def test_drift_still_rejected_during_and_after_execution():
    """A status change never exempts the locked values: h2/seed/prereg
    drift fails at running AND completed."""
    def h2_drift(status):
        def mutate(manifest):
            manifest["experiment"]["status"] = status
            manifest["evaluation"]["statistics"]["h2_supported_if"] = \
                "C-B >= +7pp on T_eval, CI excl. 0"
        return mutate
    assert any("h2_supported_if" in e and "changed" in e
                for e in prereg_errors(h2_drift("running")))
    assert any("h2_supported_if" in e and "changed" in e
                for e in prereg_errors(h2_drift("completed")))

    def seed_drift(status):
        def mutate(manifest):
            manifest["experiment"]["status"] = status
            manifest["experiment"]["preregistration"]["expected_results"]\
                ["compute_cap"]["mining_seeds"] = [11, 23]
        return mutate
    assert any("must be exactly [11, 23, 41]" in e
                for e in prereg_errors(seed_drift("running")))
    assert any("must be exactly [11, 23, 41]" in e
                for e in prereg_errors(seed_drift("completed")))

    def results_missing(manifest):
        manifest["experiment"]["status"] = "running"
        del manifest["experiment"]["preregistration"]["expected_results"]
    assert any("expected_results must be a mapping" in e
                for e in prereg_errors(results_missing))

    def filled_reverted(manifest):
        manifest["experiment"]["status"] = "running"
        manifest["experiment"]["preregistration"]["expected_results_filled"] = False
    assert any("can never revert to unfilled" in e
                for e in prereg_errors(filled_reverted))


def test_future_experiment_isolation():
    """A future manifest passes with its own lifecycle vocabulary and is
    NOT constrained by EXP-0001's seeds, criteria strings, gate shape or
    caps."""
    manifest = {
        "experiment": {
            "id": "EXP-0002",
            "status": "design-revised",
            "preregistration": {"locked": True,
                                 "expected_results_filled": False},
        },
        "evaluation": {"statistics": {"h2_supported_if": "TBD",
                                       "h1_supported_if": "TBD"}},
    }
    assert _load_module().check_preregistration(manifest, "EXP-0002") == []
    # EXP-0001-specific values are not imposed: other seeds / criteria / caps
    manifest["experiment"]["preregistration"]["expected_results_filled"] = True
    manifest["experiment"]["preregistration"]["expected_results"] = {
        "compute_cap": {"episodes_max": 50, "mining_seeds": [7]},
    }
    assert _load_module().check_preregistration(manifest, "EXP-0002") == []


def test_locked_at_must_be_iso_date():
    def bad_date(manifest):
        manifest["experiment"]["preregistration"]["locked_at"] = "10/14/2026"
    assert any("ISO YYYY-MM-DD" in e for e in prereg_errors(bad_date))
    def bad_type(manifest):
        manifest["experiment"]["preregistration"]["locked_at"] = 20261014
    assert any("ISO YYYY-MM-DD" in e for e in prereg_errors(bad_type))
    def not_a_date(manifest):
        manifest["experiment"]["preregistration"]["locked_at"] = "2026-13-99"
    assert any("ISO YYYY-MM-DD" in e for e in prereg_errors(not_a_date))


def test_real_manifest_locked_at_is_corrected_date():
    """The prereg lock date is the actual first-recording commit date
    (2026-10-14), not a backdated one."""
    manifest = _load_manifest()
    prereg = manifest["experiment"]["preregistration"]
    assert prereg["locked_at"] == "2026-10-14"
    from datetime import date
    date.fromisoformat(prereg["locked_at"])  # valid ISO calendar date


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


def _load_manifest():
    """Parse the REAL tracked EXP-0001 manifest (requires pyyaml, which the
    dev extra installs for the CI tests jobs; skipped where it is missing)."""
    yaml = pytest.importorskip("yaml")
    root = Path(__file__).resolve().parents[2]
    path = root / "experiments" / "manifests" / "EXP-0001-minimal-cig.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_real_exp0001_manifest_satisfies_prereg_invariants(ca):
    """End-to-end: the REAL tracked manifest passes the lifecycle-aware
    invariant checker."""
    data = _load_manifest()
    assert ca.check_preregistration(data, "EXP-0001-minimal-cig.yaml") == []


def test_the_real_manifest_names_the_values_the_run_executes(ca):
    """SPEC section 33: each locked value equals its code constant."""
    assert ca.check_exp0001_code_binding(_load_manifest(), "EXP-0001-minimal-cig.yaml") == []


def _drifted(value):
    if isinstance(value, list):
        return value[:-1] + [value[-1] + 1]
    if isinstance(value, str):
        return value + " (edited)"
    return value + (1 if isinstance(value, int) else 0.25)


# every locked value with a code constant; a binding row removed fails here
@pytest.mark.parametrize("path", sorted({
    "evaluation.mining_seeds", "evaluation.statistics.primary",
    "environment.splits.T_eval.size", "environment.splits.T_reg.size",
    "environment.splits.T_mine.per_family_min",
    "environment.splits.T_gate.benefit_bank.per_family", "environment.episode_max_steps",
    "model.eval_temperature", "model.mining_temperature", "genome.candidate_set.cap",
    "inheritance.expression_overhead_cap_tokens",
    "inheritance.stages.interaction.harmful_pair_threshold",
    "run_protocol.lifetime.attempts_per_task", "run_protocol.lifetime.pairing",
    "run_protocol.lifetime.temperature", "run_protocol.t_reg.size",
    "run_protocol.t_reg.seed", "run_protocol.benefit_bank.per_family"}))
def test_a_locked_value_edited_in_its_own_block_fails_the_binding(ca, path):
    """The same list or number elsewhere in the manifest (compute_cap's
    mining_seeds, the budgets' 512) never stands in for the edited one."""
    manifest = _load_manifest()
    *parents, key = path.split(".")
    node = manifest
    for parent in parents:
        node = node[parent]
    node[key] = _drifted(node[key])
    issues = ca.check_exp0001_code_binding(manifest, "m")
    assert issues and all(issue.startswith(f"m: {path} is ") for issue in issues)


def test_a_value_of_another_type_fails_the_binding(ca):
    manifest = _load_manifest()
    manifest["model"]["eval_temperature"] = 0  # YAML 0, not 0.0
    manifest["environment"]["splits"]["T_eval"]["size"] = 134.0
    manifest["genome"]["candidate_set"].pop("cap")
    issues = ca.check_exp0001_code_binding(manifest, "m")
    assert sorted({issue.split(" is ")[0] for issue in issues}) == [
        "m: environment.splits.T_eval.size", "m: genome.candidate_set.cap",
        "m: model.eval_temperature"]


def test_the_locked_primary_statistic_is_the_final_bootstrap(ca):
    """ADR-0006 D2: the primary line names the final analysis's confidence
    and resample count."""
    from evaluation.metrics.final_statistics import (
        FINAL_BOOTSTRAP_CONFIDENCE,
        FINAL_BOOTSTRAP_RESAMPLES,
    )
    assert ca.LOCKED_PRIMARY_STATISTIC == (
        f"cluster bootstrap {FINAL_BOOTSTRAP_CONFIDENCE:.0%} CI over tasks, "
        f"{FINAL_BOOTSTRAP_RESAMPLES // 1000}k resamples")
    assert FINAL_BOOTSTRAP_RESAMPLES % 1000 == 0


def test_only_the_exp0001_manifest_is_bound_to_the_code(ca, monkeypatch, tmp_path):
    yaml = pytest.importorskip("yaml")
    manifest = _load_manifest()
    manifest["evaluation"]["mining_seeds"] = [11, 23, 42]
    other = {"experiment": {"id": "EXP-0002"}, "evaluation": {"mining_seeds": [1]}}
    folder = tmp_path / "experiments" / "manifests"
    folder.mkdir(parents=True)
    (folder / "EXP-0001-minimal-cig.yaml").write_text(yaml.safe_dump(manifest), encoding="utf-8")
    (folder / "EXP-0002.yaml").write_text(yaml.safe_dump(other), encoding="utf-8")
    monkeypatch.setattr(ca, "REPO_ROOT", tmp_path)
    errors = ca.check_manifests()
    assert len(errors) == 2 and all(
        e.startswith("experiments/manifests/EXP-0001-minimal-cig.yaml: evaluation.mining_seeds")
        for e in errors)


def _check_manifest_file(ca, monkeypatch, tmp_path, manifest, name="EXP-0001-minimal-cig.yaml"):
    yaml = pytest.importorskip("yaml")
    folder = tmp_path / "experiments" / "manifests"
    folder.mkdir(parents=True)
    (folder / name).write_text(yaml.safe_dump(manifest), encoding="utf-8")
    monkeypatch.setattr(ca, "REPO_ROOT", tmp_path)
    return ca.check_manifests()


@pytest.mark.parametrize("edit", ["id changed", "experiment removed", "experiment a list"])
def test_the_exp0001_manifest_is_bound_by_its_path_not_by_its_own_id(ca, monkeypatch,
                                                                      tmp_path, edit):
    """Changing the manifest's own id must not lift the binding of its values."""
    manifest = _load_manifest()
    manifest["evaluation"]["mining_seeds"] = [11, 23, 42]
    if edit == "id changed":
        manifest["experiment"]["id"] = "EXP-0001-R1"
    elif edit == "experiment removed":
        del manifest["experiment"]
    else:
        manifest["experiment"] = [1]
    errors = _check_manifest_file(ca, monkeypatch, tmp_path, manifest)
    prefix = "experiments/manifests/EXP-0001-minimal-cig.yaml: "
    assert any(e.startswith(prefix + "is the EXP-0001 manifest, but its experiment.id is")
               for e in errors), errors
    assert any(e.startswith(prefix + "evaluation.mining_seeds is [11, 23, 42]") for e in errors)


def test_a_missing_exp0001_manifest_fails(ca, monkeypatch, tmp_path):
    errors = _check_manifest_file(ca, monkeypatch, tmp_path, {"experiment": {"id": "EXP-0002"}},
                                  name="EXP-0002.yaml")
    assert errors == ["experiments/manifests/EXP-0001-minimal-cig.yaml is missing"]


@pytest.mark.parametrize("value", [5, [1], {"statistics": 5}])
def test_an_evaluation_block_that_is_not_a_mapping_is_a_fail_line(ca, value):
    manifest = _load_manifest()
    manifest["evaluation"] = value
    issues = ca.check_preregistration(manifest, "m")
    assert any(i.startswith("m: locked primary support criterion h2_supported_if changed: None")
               for i in issues), issues


@pytest.mark.parametrize("edit", ["block renamed", "block null", "block renamed, completed"])
def test_the_exp0001_manifest_without_its_preregistration_block_is_still_checked(
        ca, monkeypatch, tmp_path, edit):
    """At its own path the manifest's decision rules and status are checked
    whatever its preregistration block says."""
    manifest = _load_manifest()
    experiment = manifest["experiment"]
    block = experiment.pop("preregistration")
    if edit == "block null":
        experiment["preregistration"] = None
    else:
        experiment["prereg"] = block
    if edit.endswith("completed"):
        experiment["status"] = "done"
        manifest["evaluation"]["statistics"]["seed_consistency"] = "any seed"
        changed = "seed_consistency"
    else:
        manifest["evaluation"]["statistics"]["h2_supported_if"] = "always"
        changed = "h2_supported_if"
    errors = _check_manifest_file(ca, monkeypatch, tmp_path, manifest)
    prefix = "experiments/manifests/EXP-0001-minimal-cig.yaml: "
    assert prefix + "is the EXP-0001 manifest, but has no preregistration block (got " \
        "NoneType); restore it" in errors, errors
    assert any(e.startswith(prefix + f"locked primary support criterion {changed} changed")
               for e in errors), errors
    if edit.endswith("completed"):
        assert any(e.startswith(prefix + "invalid EXP-0001 lifecycle status 'done'")
                   for e in errors), errors


@pytest.mark.parametrize("status", [[1], {"a": 1}])
def test_a_status_that_is_a_list_or_a_mapping_is_a_fail_line(ca, status):
    manifest = _load_manifest()
    manifest["experiment"]["status"] = status
    issues = ca.check_preregistration(manifest, "m")
    assert any(i.startswith(f"m: invalid EXP-0001 lifecycle status {status!r}")
               for i in issues), issues
