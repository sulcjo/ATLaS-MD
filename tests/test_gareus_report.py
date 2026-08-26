"""Tests for gareus_report: human-facing health verdict + warning triage.

These lock the *presentation* contract of the analysis result health block:
- a composite PASS/CAUTION/FAIL verdict derived from numbers MBAR/ESS/overlap/
  boost already produce (no new physics),
- warning severity classification + deduplication,
- render helpers that never crash on partial/old-format summaries.
"""
import json
import sys
from pathlib import Path

import pytest

import gareus_report as gr

_TESTS_DIR = Path(__file__).resolve().parent


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


# ----------------------------------------------------------------------------
# overlap connectivity (round-3 review, deeper half of FINDING 1)
#
# MBAR fixes free energies only up to one additive constant PER CONNECTED
# COMPONENT of the overlap graph. So a window set can have every graded pair
# above threshold and still be split, with the offsets between the blocks fixed
# by nothing in the data -- which is chignolin_6's actual failure (94% of its
# posterior on one state). This is not gradable from any worst-pair number, so
# it is its own check, fed by gareus.mbar_analysis.pmf.overlap_components via
# s['overlap_connectivity'].
# ----------------------------------------------------------------------------
def _conn(marg_n=1, joint_n=1, joint_available=True):
    return {
        "marginal": {"space": "cv1_marginal", "available": True, "reason": "",
                     "threshold": 0.30, "n_components": marg_n,
                     "components": [[0, 1, 2]] if marg_n == 1 else [[0, 1], [2]],
                     "component_samples": [300] if marg_n == 1 else [200, 100]},
        "joint": ({"space": "cv1_cv2_joint", "available": True, "reason": "",
                   "threshold": 0.09, "n_components": joint_n,
                   "components": [[0, 1, 2]] if joint_n == 1 else [[0, 1], [2]],
                   "component_samples": [300] if joint_n == 1 else [200, 100]}
                  if joint_available else
                  {"space": "cv1_cv2_joint", "available": False,
                   "reason": "disabled by --no-joint-overlap", "threshold": 0.09,
                   "n_components": 0, "components": [], "component_samples": []}),
    }


def test_verdict_connected_overlap_graph_is_pass():
    s = _good_summary()
    s["overlap_connectivity"] = _conn()
    v = gr.build_health_verdict(s, 0.30)
    c = next(c for c in v["checks"] if c["name"] == "Overlap connectivity")
    assert c["status"] == "pass"
    assert v["overall"] == "PASS"


def test_verdict_split_overlap_graph_is_fail_and_names_the_components():
    """The failure the whole check exists for: MBAR has no information linking
    the blocks, so the offset between them is arbitrary."""
    s = _good_summary()
    s["overlap_connectivity"] = _conn(joint_n=2)
    v = gr.build_health_verdict(s, 0.30)
    c = next(c for c in v["checks"] if c["name"] == "Overlap connectivity")
    assert c["status"] == "fail"
    assert c["metric"] == 2
    assert "[0,1]" in c["detail"] and "[2]" in c["detail"]
    # The per-component sample counts are the number that separates small-N
    # histogram deflation from a real physical gap, so they must be shown.
    assert "200" in c["detail"] and "100" in c["detail"]
    assert v["overall"] == "FAIL"


def test_verdict_split_in_the_marginal_space_alone_is_still_a_fail():
    """Neither space's connectivity implies the other's: joint overlap is
    bounded above by the marginal while the joint THRESHOLD is below the
    marginal one, so worse-of-both is the only safe combination."""
    s = _good_summary()
    s["overlap_connectivity"] = _conn(marg_n=3, joint_n=1)
    v = gr.build_health_verdict(s, 0.30)
    c = next(c for c in v["checks"] if c["name"] == "Overlap connectivity")
    assert c["status"] == "fail"
    assert "SPLIT into 3" in c["detail"]


def test_verdict_unmeasured_joint_connectivity_says_so_rather_than_passing_quietly():
    s = _good_summary()
    s["overlap_connectivity"] = _conn(joint_available=False)
    v = gr.build_health_verdict(s, 0.30)
    c = next(c for c in v["checks"] if c["name"] == "Overlap connectivity")
    assert "not measured" in c["detail"]
    assert "no-joint-overlap" in c["detail"]
    assert c["status"] == "pass"          # the marginal WAS measured and is connected


def test_verdict_overlap_connectivity_is_na_for_a_summary_written_before_it_existed():
    """This module's standing contract: never raise, never invent a status, for
    an old or partial pmf_summary.json."""
    for block in (None, {}, "nonsense", {"marginal": "junk"},
                  {"marginal": {"available": True, "n_components": 0}}):
        s = _good_summary()
        if block is not None:
            s["overlap_connectivity"] = block
        v = gr.build_health_verdict(s, 0.30)
        c = next(c for c in v["checks"] if c["name"] == "Overlap connectivity")
        assert c["status"] == "na", (block, c)
        assert v["overall"] == "PASS"      # an na check never moves the verdict


def test_disconnected_overlap_graph_warning_is_triaged_high():
    groups = gr.classify_warnings([
        "Umbrella overlap graph is DISCONNECTED in the joint (CV1, CV2) space: 3 "
        "components at overlap >= 0.090 -- [18] (3,410 samples), [20] (6,030 samples)."])
    assert groups[0]["severity"] == "HIGH"


def test_components_text_truncates_a_shattered_geometry():
    """A K=364 oracle geometry must not render a multi-kilobyte terminal line;
    the full lists stay in pmf_summary.json."""
    block = {"available": True, "threshold": 0.09,
             "n_components": 20,
             "components": [list(range(i * 20, i * 20 + 20)) for i in range(20)],
             "component_samples": [1000] * 20}
    txt = gr._components_text(block)
    assert len(txt) < 200, txt
    assert "+16 more" in txt      # 20 components, 4 shown
    assert "+14" in txt           # 20 states per component, 6 shown


def test_render_includes_the_connectivity_row():
    s = _good_summary()
    s["overlap_connectivity"] = _conn(joint_n=2)
    v = gr.build_health_verdict(s, 0.30)
    assert "Overlap connectivity" in gr.render_verdict_terminal(v, [], use_color=False)
    assert "Overlap connectivity" in "\n".join(gr.render_verdict_md(v, []))


# ----------------------------------------------------------------------------
# FINDING 1 (first half) -- the index-adjacent number is graded again
# ----------------------------------------------------------------------------
def test_verdict_index_adjacent_overlap_is_graded_even_when_cv_space_passes():
    """A6 made the CV-space pairing REPLACE the index one, which turned the
    index branch into dead code for every new analysis (the summary always
    carries cv_space_neighbor_overlap). A broken ladder could then read PASS on
    unchanged data."""
    s = _good_summary()
    s["cv_space_neighbor_overlap"] = [{"window": 0, "neighbor": 1, "overlap": 0.9},
                                      {"window": 1, "neighbor": 0, "overlap": 0.9}]
    s["neighbor_overlap"] = [0.5] * 4 + [0.01] + [0.5] * 4
    v = gr.build_health_verdict(s, 0.30)
    ov = next(c for c in v["checks"] if c["name"] == "Window overlap")
    assert ov["status"] == "fail", ov
    assert "index-adjacent" in ov["detail"]
    assert "4-5" in ov["detail"]
    assert v["overall"] == "FAIL"


def test_verdict_both_pairings_appear_in_the_overlap_detail():
    s = _good_summary()
    s["cv_space_neighbor_overlap"] = [{"window": 0, "neighbor": 7, "overlap": 0.44}]
    s["neighbor_overlap"] = [0.66] * 9
    v = gr.build_health_verdict(s, 0.30)
    ov = next(c for c in v["checks"] if c["name"] == "Window overlap")
    assert ov["status"] == "pass"
    assert "0.440" in ov["detail"] and "0.660" in ov["detail"]
    assert "CV-space nearest" in ov["detail"] and "index-adjacent" in ov["detail"]


def test_cv_space_worst_is_defined_exactly_once():
    """FINDING 2: the A6 refactor left a first, dead definition of
    _cv_space_worst above the one that delegates to _worst_of -- behaviourally
    identical, so nothing broke, but the next reader of the very function
    FINDING 1 asks to be changed would have been reading the dead copy."""
    src = Path(gr.__file__).read_text()
    assert src.count("def _cv_space_worst(") == 1
    assert src.count("def _worst_of(") == 1
    assert src.count("def _check_overlap(") == 1


# ----------------------------------------------------------------------------
# how BADLY split: a bare component count conflates two different failures
# ----------------------------------------------------------------------------
def _split_conn(best):
    block = {"space": "cv1_cv2_joint", "available": True, "reason": "",
             "threshold": 0.09, "n_components": 2,
             "components": [[0, 1], [2]], "component_samples": [200, 100]}
    if best is not None:
        block["worst_component_best_cross_overlap"] = best
        block["most_isolated_component"] = [2]
    return {"marginal": {"available": True, "threshold": 0.30, "n_components": 1,
                         "components": [[0, 1, 2]], "component_samples": [300]},
            "joint": block}


def test_a_split_with_essentially_no_cross_overlap_is_a_fail():
    """chignolin_6's isolated state 20: 0.0048 to anything outside its own
    component. That block's free energy relative to the rest genuinely is
    undetermined."""
    s = _good_summary()
    s["overlap_connectivity"] = _split_conn(0.0008)
    v = gr.build_health_verdict(s, 0.30)
    c = next(c for c in v["checks"] if c["name"] == "Overlap connectivity")
    assert c["status"] == "fail", c
    assert "0.0008" in c["detail"]
    assert "[2]" in c["detail"]


def test_a_split_bridged_only_by_a_weak_but_real_pair_is_a_caution():
    """chignolin_6's state 18: 0.065 against a 0.09 target -- below target, so
    already a CAUTION on the pairwise check, but NOT a pair that carries no
    information. Grading a bare component count FAIL would flip every run with
    one slightly-under-target link into a FAIL on unchanged data, while claiming
    in the detail text that its free energies are undetermined."""
    s = _good_summary()
    s["overlap_connectivity"] = _split_conn(0.065)
    v = gr.build_health_verdict(s, 0.30)
    c = next(c for c in v["checks"] if c["name"] == "Overlap connectivity")
    assert c["status"] == "caution", c
    assert "0.0650" in c["detail"]
    assert v["overall"] == "CAUTION"


def test_a_split_can_never_grade_pass_however_strong_the_reported_link():
    """A split is a split: no summary value may talk this check up to pass."""
    s = _good_summary()
    s["overlap_connectivity"] = _split_conn(0.95)
    v = gr.build_health_verdict(s, 0.30)
    c = next(c for c in v["checks"] if c["name"] == "Overlap connectivity")
    assert c["status"] == "caution", c


def test_a_split_of_unknown_strength_fails():
    """An older summary, or an all-non-finite cut: a split whose severity cannot
    be measured is not evidence of a mild one."""
    s = _good_summary()
    s["overlap_connectivity"] = _split_conn(None)
    v = gr.build_health_verdict(s, 0.30)
    c = next(c for c in v["checks"] if c["name"] == "Overlap connectivity")
    assert c["status"] == "fail", c
    assert "isolation depth unknown" in c["detail"]


# ---------------------------------------------------------------------------
# Triage of the equal-count window-map membership note
# ---------------------------------------------------------------------------
# These feed _severity_of the note text the LOADER really produces, not a
# hand-written copy of it.  The note and the rule that grades it live in
# different files (gareus/mbar_analysis/loaders_adaptive.py and this one) with
# nothing but a regex holding them together, and that pair silently drifting
# apart -- a reworded note falling through to the MEDIUM default -- is exactly
# how the bug this branch exists for stayed invisible for a whole campaign.  A
# fixture string could not catch it; only the real one can.
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

from test_equal_count_map_fingerprint import (            # noqa: E402
    _check_notes, _ladder, _phase, _samples, _write_window_table)


def _real_membership_note(tmp_path) -> str:
    """The real '[window map check]' note for a map that lists a genuinely
    DIFFERENT window set of the same size.

    Two attempts at one phase dropped equally many but different windows:
    attempt 1 (whose map survived) dropped ladder entry 2, attempt 2 (which
    actually ran) dropped entry 5. Six rows either way -- which is what hides
    this from every count-based check -- but entry 5 really ran and has no row
    in the map at all, so its state_id is recorded nowhere and nothing can
    re-derive the mapping. That is what makes this the UNREPAIRABLE fault the
    three tests below grade, and it is the same fixture shape as
    `test_equal_count_map_fingerprint`'s own
    `test_an_equal_count_map_describing_a_shifted_window_set_is_flagged`.

    Deliberately NOT an adjacent-window exchange, which is what this helper
    used to build: an exchange is a PERMUTATION -- same window set, wrong order
    -- and the loader now repairs one in memory and returns a `[stale window
    map]` repair note instead of the `[window map check]` fault note. Staging
    the defect with a permutation therefore stopped staging a membership fault
    at all, which is exactly the drift these tests exist to catch, one file
    over.
    """
    ladder = _ladder(7)
    real = ladder[:5] + ladder[6:]           # attempt 2 dropped ladder entry 5
    phase, rows = _phase(tmp_path, ladder[:2] + ladder[3:], real)   # attempt 1 dropped entry 2
    ids, cv2 = _samples(real)
    notes = _check_notes(_run_loader(phase, rows, ids, cv2))
    assert len(notes) == 1, notes
    return notes[0]


def _real_could_not_check_note(tmp_path) -> str:
    """The real sibling note: the check could not run at all."""
    centers = _ladder(5)
    phase, rows = _phase(tmp_path, centers)
    _write_window_table(phase / "umbrella_explicit_windows.csv", centers[:4])
    ids, cv2 = _samples(centers)
    notes = _check_notes(_run_loader(phase, rows, ids, cv2))
    assert len(notes) == 1, notes
    return notes[0]


def _run_loader(phase, rows, ids, cv2):
    from gareus.mbar_analysis.loaders_adaptive import _validate_and_repair_epoch_window_map
    return _validate_and_repair_epoch_window_map(phase, rows, window_ids=ids, cv2=cv2)[1]


def test_the_real_membership_note_is_triaged_high(tmp_path):
    """The headline failure mode of this branch: samples attributed to the wrong
    umbrella state.  It must not fall through to the MEDIUM default, which is
    where an unmatched warning goes and where nobody looks."""
    note = _real_membership_note(tmp_path)

    assert gr._severity_of(note) == "HIGH", note


def test_the_real_could_not_cross_check_note_is_graded_below_a_detected_fault(tmp_path):
    """Its deliberate twin.  Same tag, weaker fact: the check did not run, which
    is not the same as the check finding something -- so it must NOT be graded
    with the note above, and the rule that says so must be the one matching it
    (first match wins, and the broad tag rule would otherwise swallow it)."""
    note = _real_could_not_check_note(tmp_path)

    assert gr._severity_of(note) == "MEDIUM", note
    assert gr._severity_of(note) != gr._severity_of(_real_membership_note(tmp_path))


def test_the_real_membership_note_fails_the_mapping_check(tmp_path):
    """A HIGH warning alone can still sit inside a PASS verdict -- the gap this
    check row exists to close.  Self-bias is deliberately healthy here: two
    attempts that drop equally many but different windows leave every sample on
    a restraint one ladder step from its own, so every per-state energy stays
    plausible and the mapping defect never reaches the self-bias numbers.  The
    warning has to carry the verdict on its own."""
    s = _good_summary()
    s["warnings"] = [_real_membership_note(tmp_path)]
    s["self_bias"] = {"median_kT": [1.0, 1.1, 0.9], "p90_kT": [2.0, 2.1, 1.9]}

    v = gr.build_health_verdict(s, 0.30)

    c = next(c for c in v["checks"] if c["name"] == "Sample-to-state mapping")
    assert c["status"] == "fail", c
    assert "DIFFERENT window set of the same size" in c["detail"]
    assert v["overall"] == "FAIL"


def test_the_real_could_not_cross_check_note_does_not_fail_the_mapping_check(tmp_path):
    """Non-firing twin of the test above, and the whole point of separating the
    two notes: an absent check must not be reported as a detected fault."""
    s = _good_summary()
    s["warnings"] = [_real_could_not_check_note(tmp_path)]
    s["self_bias"] = {"median_kT": [1.0, 1.1, 0.9], "p90_kT": [2.0, 2.1, 1.9]}

    v = gr.build_health_verdict(s, 0.30)

    c = next(c for c in v["checks"] if c["name"] == "Sample-to-state mapping")
    assert c["status"] == "pass", c
    assert "DIFFERENT window set" not in c["detail"]


def test_a_clean_phase_contributes_no_note_to_triage_at_all(tmp_path):
    """The base rate that makes the two tests above worth anything: the note
    only exists when the two artifacts really disagree.  Measured on real data
    -- 117 of 117 adaptive-production phases under RUNS/ whose row count agrees
    also agree row for row, and none of them emits this note."""
    centers = _ladder(5)
    phase, rows = _phase(tmp_path, centers)
    ids, cv2 = _samples(centers)

    assert _run_loader(phase, rows, ids, cv2) == []
