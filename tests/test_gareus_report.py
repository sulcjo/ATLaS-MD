"""Tests for gareus_report: human-facing health verdict + warning triage.

These lock the *presentation* contract of the analysis result health block:
- a composite PASS/CAUTION/FAIL verdict derived from numbers MBAR/ESS/overlap/
  boost already produce (no new physics),
- warning severity classification + deduplication,
- render helpers that never crash on partial/old-format summaries.
"""
import json
from pathlib import Path

import pytest

import gareus_report as gr


# ----------------------------------------------------------------------------
# fixtures / builders
# ----------------------------------------------------------------------------
def _good_summary():
    """A summary dict that should score PASS on every applicable check."""
    return {
        "n_samples": 1_000_000,
        "n_windows": 10,
        "selected_unbiased_method": "gamd_cumulant2",
        "pmf_span_kcal_mol": 8.0,
        "pmf_minimum_cv_A": 5.0,
        "primary_cv_units": "A",
        "mbar": {
            "converged": True,
            "max_delta": 2.0e-10,
            "base_ess": 300_000.0,       # 30% -> pass
            "backend": "numba-anderson",
            "n_k": [100_000] * 10,
        },
        "boost": {
            "available": True,
            "std_kcal_mol": 1.8,
            "anharmonicity_score": 0.5,  # <=1.0 -> pass
            "boost_reweight_ess_fraction": 1.6e-05,  # tiny, expected
        },
        "neighbor_overlap": [0.5] * 9,   # all >= 0.30 -> pass
        "convergence": {"enabled": True, "summary": {"converged_bool": 1}},
        "warnings": [],
    }


# ----------------------------------------------------------------------------
# build_health_verdict — overall + per-check
# ----------------------------------------------------------------------------
def test_verdict_all_good_is_pass():
    v = gr.build_health_verdict(_good_summary(), min_neighbor_overlap=0.30)
    assert v["overall"] == "PASS"
    assert all(c["status"] in ("pass", "na") for c in v["checks"])
    # every check we expect is present
    names = {c["name"] for c in v["checks"]}
    assert "MBAR convergence" in names
    assert "Window overlap" in names


def test_verdict_mbar_not_converged_is_fail():
    s = _good_summary()
    s["mbar"]["converged"] = False
    s["mbar"]["max_delta"] = 3.1e-3
    v = gr.build_health_verdict(s, 0.30)
    assert v["overall"] == "FAIL"
    mbar = next(c for c in v["checks"] if c["name"] == "MBAR convergence")
    assert mbar["status"] == "fail"


def test_verdict_zero_sample_window_is_fail():
    s = _good_summary()
    s["mbar"]["n_k"] = [100_000] * 9 + [0]
    v = gr.build_health_verdict(s, 0.30)
    assert v["overall"] == "FAIL"


def test_verdict_bad_overlap_fails():
    s = _good_summary()
    s["neighbor_overlap"] = [0.5, 0.5, 0.074, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    v = gr.build_health_verdict(s, 0.30)
    ov = next(c for c in v["checks"] if c["name"] == "Window overlap")
    assert ov["status"] == "fail"          # 0.074 < 0.5*0.30
    assert v["overall"] == "FAIL"
    assert "0.074" in ov["detail"] or "0.07" in ov["detail"]


def test_verdict_marginal_overlap_cautions():
    s = _good_summary()
    s["neighbor_overlap"] = [0.5, 0.21, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    v = gr.build_health_verdict(s, 0.30)
    ov = next(c for c in v["checks"] if c["name"] == "Window overlap")
    assert ov["status"] == "caution"       # 0.15 <= 0.21 < 0.30
    assert v["overall"] == "CAUTION"


def test_verdict_low_base_ess_fraction_fails():
    s = _good_summary()
    s["mbar"]["base_ess"] = 5_000.0        # 0.5% -> below fail line
    v = gr.build_health_verdict(s, 0.30)
    ess = next(c for c in v["checks"] if "ESS" in c["name"])
    assert ess["status"] == "fail"


def test_verdict_high_anharmonicity_fails_gamd():
    s = _good_summary()
    s["boost"]["anharmonicity_score"] = 2.5
    v = gr.build_health_verdict(s, 0.30)
    g = next(c for c in v["checks"] if "cumulant" in c["name"].lower() or "GaMD" in c["name"])
    assert g["status"] == "fail"


def test_verdict_low_exp_ess_not_penalized_for_cumulant():
    """KEY NUANCE: exponential-reweight ESS ~0 is expected when cumulant2 is the
    selected estimator; it must NOT drag the GaMD check to fail/caution."""
    s = _good_summary()
    s["boost"]["boost_reweight_ess_fraction"] = 1.0e-7   # catastrophically low
    s["boost"]["anharmonicity_score"] = 0.5              # but cumulant still valid
    v = gr.build_health_verdict(s, 0.30)
    g = next(c for c in v["checks"] if "cumulant" in c["name"].lower() or "GaMD" in c["name"])
    assert g["status"] == "pass"


def test_verdict_umbrella_only_marks_gamd_na():
    s = _good_summary()
    s["selected_unbiased_method"] = "umbrella_only"
    s["boost"] = {"available": False}
    v = gr.build_health_verdict(s, 0.30)
    g = next((c for c in v["checks"] if "GaMD" in c["name"] or "cumulant" in c["name"].lower()), None)
    assert g is not None and g["status"] == "na"


def test_verdict_pmf_convergence_disabled_is_na():
    s = _good_summary()
    s["convergence"] = {"enabled": False}
    v = gr.build_health_verdict(s, 0.30)
    pc = next(c for c in v["checks"] if "convergence" in c["name"].lower() and "MBAR" not in c["name"])
    assert pc["status"] == "na"


def test_verdict_pmf_not_converged_cautions():
    s = _good_summary()
    s["convergence"] = {"enabled": True, "summary": {"converged_bool": 0}}
    v = gr.build_health_verdict(s, 0.30)
    assert v["overall"] == "CAUTION"


def test_verdict_robust_to_empty_summary():
    v = gr.build_health_verdict({}, 0.30)
    assert isinstance(v, dict)
    assert "overall" in v
    assert isinstance(v["checks"], list)


# ----------------------------------------------------------------------------
# classify_warnings — severity + dedup
# ----------------------------------------------------------------------------
def test_classify_warnings_severity_mapping():
    warns = [
        "MBAR solver did not fully converge: max_delta=3.1e-03",
        "Weak neighbor CV overlap below 0.30 for pairs: 11-12 (0.07)",
        "GaMD boost anharmonicity score is high (2.30)",
        "Applied analysis stride 3 with offset 0: kept 1686832/5060468 samples.",
    ]
    groups = gr.classify_warnings(warns)
    by_text = {g["representative"]: g["severity"] for g in groups}
    assert by_text["MBAR solver did not fully converge: max_delta=3.1e-03"] == "CRITICAL"
    assert by_text["Weak neighbor CV overlap below 0.30 for pairs: 11-12 (0.07)"] == "HIGH"
    assert by_text["GaMD boost anharmonicity score is high (2.30)"] == "HIGH"
    assert by_text["Applied analysis stride 3 with offset 0: kept 1686832/5060468 samples."] == "INFO"


def test_classify_warnings_exp_ess_downgraded_to_info():
    """Low GaMD *exponential* ESS is expected (cumulant2 is used instead)."""
    groups = gr.classify_warnings(
        ["GaMD exponential reweighting ESS is very low: 2.2/1063879"]
    )
    assert groups[0]["severity"] == "INFO"


def test_classify_warnings_exp_ess_not_downgraded_when_exponential_selected():
    """If exponential IS the selected estimator, low exp-ESS is a real problem."""
    w = ["GaMD exponential reweighting ESS is very low: 2.2/1063879"]
    assert gr.classify_warnings(w, selected="gamd_exponential")[0]["severity"] == "HIGH"
    # cumulant selection (or unspecified) keeps the expected-low downgrade
    assert gr.classify_warnings(w, selected="gamd_cumulant2")[0]["severity"] == "INFO"


def test_classify_warnings_dedupes_near_identical():
    warns = [f"Extra observable PMF pass failed for replica 0: xyz must be shape (Any, {n}, 3)."
             for n in (19008, 19008, 19008, 19008, 19008)]
    groups = gr.classify_warnings(warns)
    assert len(groups) == 1
    assert groups[0]["count"] == 5


def test_classify_warnings_sorted_severe_first():
    warns = [
        "Applied analysis stride 3.",                       # INFO
        "MBAR solver did not fully converge: max_delta=1e-2",  # CRITICAL
        "Weak neighbor CV overlap below 0.30 for pairs: 1-2 (0.1)",  # HIGH
    ]
    groups = gr.classify_warnings(warns)
    order = [g["severity"] for g in groups]
    rank = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "INFO": 3}
    assert order == sorted(order, key=lambda x: rank[x])


def test_classify_warnings_empty():
    assert gr.classify_warnings([]) == []


# ----------------------------------------------------------------------------
# render helpers — never crash, contain the essentials
# ----------------------------------------------------------------------------
def test_render_terminal_contains_verdict_and_is_plain_safe():
    v = gr.build_health_verdict(_good_summary(), 0.30)
    groups = gr.classify_warnings(["Weak neighbor CV overlap below 0.30 for pairs: 1-2 (0.1)"])
    txt = gr.render_verdict_terminal(v, groups, use_color=False)
    assert "RESULT HEALTH" in txt
    assert "PASS" in txt
    assert "\x1b[" not in txt          # no ANSI when use_color=False


def test_render_terminal_color_has_ansi():
    v = gr.build_health_verdict(_good_summary(), 0.30)
    txt = gr.render_verdict_terminal(v, [], use_color=True)
    assert "\x1b[" in txt


def test_render_md_returns_lines_with_section():
    v = gr.build_health_verdict(_good_summary(), 0.30)
    groups = gr.classify_warnings(["GaMD exponential reweighting ESS is very low: 2/1e6"])
    lines = gr.render_verdict_md(v, groups)
    assert isinstance(lines, list)
    blob = "\n".join(lines)
    assert "## Result health" in blob
    assert "MBAR convergence" in blob


def test_render_handles_none_and_empty():
    # Must not raise on degraded inputs.
    assert isinstance(gr.render_verdict_terminal(None, None, use_color=False), str)
    assert isinstance(gr.render_verdict_md(None, None), list)
    assert isinstance(gr.render_verdict_terminal({"overall": "UNKNOWN", "checks": []}, [], use_color=False), str)


# ----------------------------------------------------------------------------
# integration — real summaries on disk must not crash the verdict path
# ----------------------------------------------------------------------------
_REAL = sorted(Path(__file__).resolve().parents[1].glob("RUNS/**/pmf_summary.json"))


@pytest.mark.skipif(not _REAL, reason="no real pmf_summary.json outputs present")
@pytest.mark.parametrize("path", _REAL[:6], ids=lambda p: p.parent.name)
def test_real_summaries_render_without_error(path):
    s = json.loads(path.read_text())
    v = gr.build_health_verdict(s, 0.30)
    groups = gr.classify_warnings(s.get("warnings", []))
    assert v["overall"] in ("PASS", "CAUTION", "FAIL", "UNKNOWN")
    # both renderers must succeed
    assert gr.render_verdict_terminal(v, groups, use_color=False)
    assert gr.render_verdict_md(v, groups)
