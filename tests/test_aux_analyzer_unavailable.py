# tests/test_aux_analyzer_unavailable.py
"""Final fix wave I2 (spec Section 15): analyzer diagnostics not yet audited for auxiliary states report
'unavailable', never PASS -- gated strictly on the aux_models flag load_parquet's aux branch sets.
Legacy (no flag) rows and summaries are pinned unchanged."""
import copy
import json
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import analyze_gareus_mbar as agm  # noqa: E402
import gareus_report as gr  # noqa: E402
from gareus.mbar_analysis.pmf import AUX_DIAGNOSTICS_UNAVAILABLE  # noqa: E402

R = "unavailable: aux states (Stage D audit)"
GUARDED = ("Window overlap", "Overlap connectivity", "λ-ladder cross-check")


def _summary():
    from test_gareus_report import _good_summary
    s = _good_summary()
    s["ladder_crosscheck"] = {"status": "pass", "max_abs_diff_kcal": 0.05, "tolerance_kcal": 0.3,
                              "n_bins_compared": 20}
    s["overlap_connectivity"] = {"marginal": {"space": "cv1_marginal", "available": True, "reason": "",
                                              "threshold": 0.3, "n_states_graded": 10, "n_components": 1,
                                              "components": [list(range(10))], "component_samples": [1],
                                              "excluded_unsampled_states": []}}
    return s


def _rows(v):
    return {c["name"]: c for c in v["checks"]}


def test_reason_constant():
    assert AUX_DIAGNOSTICS_UNAVAILABLE == R


def test_report_rows_are_unavailable_on_aux_summaries():
    s = _summary()
    s["aux_states"] = {"reason": R, "models": ["a" * 64]}
    rows = _rows(gr.build_health_verdict(s, 0.30))
    for name in GUARDED:
        assert rows[name]["status"] == gr.NA and rows[name]["detail"] == R


def test_report_rows_unchanged_without_the_aux_flag():
    s = _summary()
    legacy = gr.build_health_verdict(copy.deepcopy(s), 0.30)
    rows = _rows(legacy)
    assert rows["Window overlap"]["status"] == gr.PASS and rows["λ-ladder cross-check"]["status"] == gr.PASS
    assert all(R not in c["detail"] for c in legacy["checks"])
    s["aux_states"] = None                       # falsy flag: still the legacy grading
    assert gr.build_health_verdict(s, 0.30) == legacy


def _data(tmp_path, *, aux, ladder=False):
    from test_per_regime_full_analysis import _make_data
    d = _make_data(tmp_path, ["torsion-pca"], n_per=(220, 220), seed=11)
    if aux:
        d.meta["aux_models"] = ["a" * 64]
    if ladder:
        d.meta["gamd_ladder"] = True
        d.state_lambdas = np.array([0.0, 0.0])
    return d


def _analyze(d):
    from test_per_regime_full_analysis import _args
    args = _args()
    args.no_extra_pmfs = True
    args.no_convergence = True
    args.no_basin_tracking = True
    agm.analyze(d, args)
    return json.loads((Path(d.out_dir) / "pmf_summary.json").read_text())


def test_aux_analysis_marks_overlap_diagnostics_unavailable(tmp_path):
    s = _analyze(_data(tmp_path, aux=True))
    assert s["aux_states"]["reason"] == R
    rows = _rows(s["health"])
    for name in ("Window overlap", "Overlap connectivity"):
        assert rows[name]["status"] == "na" and rows[name]["detail"] == R
    assert not any("Weak CV-space nearest-neighbour overlap" in w for w in s["warnings"])
    assert any(R in w for w in s["warnings"])


def test_aux_ladder_analysis_marks_ladder_diagnostics_unavailable(tmp_path):
    s = _analyze(_data(tmp_path, aux=True, ladder=True))
    assert s["ladder_crosscheck"]["status"] == "skipped" and s["ladder_crosscheck"]["reason"] == R
    assert s["ladder_overlap"] == {"available": False, "reason": R}
    rows = _rows(s["health"])
    assert rows["λ-ladder cross-check"]["status"] == "na" and rows["λ-ladder cross-check"]["detail"] == R
    assert rows["Ladder state overlap"]["status"] == "na" and rows["Ladder state overlap"]["detail"] == R
    assert not any(c["name"].startswith(("Overlap along", "Connectivity along")) for c in s["health"]["checks"])


def test_legacy_analysis_has_no_aux_marker(tmp_path):
    s = _analyze(_data(tmp_path, aux=False))
    assert "aux_states" not in s
    assert all(R not in c["detail"] for c in s["health"]["checks"])
    assert all(R not in w for w in s["warnings"])


# Follow-up: the overall banner is never PASS on aux data (capped at CAUTION; FAIL stays FAIL) --------

CAP = "auxiliary states: diagnostics audit pending (Stage D)"


def test_aux_summary_with_all_pass_rows_is_capped_at_caution():
    s = _summary()
    s["aux_states"] = {"reason": R, "models": ["a" * 64]}
    v = gr.build_health_verdict(s, 0.30)
    assert all(c["status"] in (gr.PASS, gr.NA, gr.CAUTION) for c in v["checks"])
    assert v["overall"] == "CAUTION"
    cap = [c for c in v["checks"] if c["detail"] == CAP]
    assert len(cap) == 1 and cap[0]["status"] == gr.CAUTION
    assert gr.overall_from_checks(v["checks"]) == "CAUTION"      # analyzer recomputes keep the cap


def test_aux_summary_with_a_fail_row_stays_fail():
    s = _summary()
    s["aux_states"] = {"reason": R, "models": ["a" * 64]}
    s["mbar"]["converged"] = False
    assert gr.build_health_verdict(s, 0.30)["overall"] == "FAIL"


def test_legacy_summary_verdict_and_bytes_unchanged():
    s = _summary()
    v = gr.build_health_verdict(copy.deepcopy(s), 0.30)
    assert v["overall"] == "PASS" and all(c["detail"] != CAP for c in v["checks"])
    # Same verdict bytes with and without an (absent/falsy) aux key, and no cap row.
    s2 = copy.deepcopy(s)
    s2["aux_states"] = None
    assert json.dumps(gr.build_health_verdict(s2, 0.30), sort_keys=True) == json.dumps(v, sort_keys=True)
