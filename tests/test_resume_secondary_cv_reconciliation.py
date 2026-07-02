"""Tests for resume secondary-CV metadata reconciliation.

Scientific audit finding: the fast-resume path in gareus/production.py reads
``secondary_cv_metadata`` and ``secondary_cv_centers`` independently from the
checkpoint manifest. A resumed 2D run can have ``secondary_cv_centers``
present (non-empty) while ``secondary_cv_metadata`` is missing or has
``enabled`` false — silently dropping CV2 from seed scoring/biasing on
resume, even though production windows are still 2D.

``reconcile_resume_secondary_cv_metadata`` is a pure helper (no I/O, no
OpenMM) that takes the two independently-read values and reconciles them:
if centers are present but metadata is not enabled, it warns loudly and
returns metadata with ``enabled`` forced True, preserving any other fields
that were already present.

The unit tests below exercise that pure helper in isolation. That is not
sufficient on its own: the real caller, ``load_resume_run_definition``, reads
``secondary_meta``/``secondary_centers``/``secondary_k`` from the manifest and
then, in its own ``if enabled: ... else: secondary_centers = None;
secondary_k = None`` branch, nulls the centers/k whenever metadata is not
enabled. If reconciliation happens *after* that branch (or is never wired
into the loader at all), the desync is already destroyed by the time
reconciliation would run — the precondition the helper checks
(``centers present but metadata disabled``) can never be observed, and the
helper becomes dead code. The integration test below calls through
``load_resume_run_definition`` itself with a realistic desynced checkpoint
manifest to catch exactly that class of regression.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from gareus.production import load_resume_run_definition, reconcile_resume_secondary_cv_metadata


def test_centers_present_metadata_missing_is_reconciled_and_warns(capsys):
    reconciled = reconcile_resume_secondary_cv_metadata(None, [0.1, 0.2, 0.3])
    assert reconciled["enabled"] is True
    captured = capsys.readouterr()
    assert "WARNING" in captured.out
    assert "secondary_cv_centers" in captured.out


def test_centers_present_metadata_disabled_is_reconciled_and_warns(capsys):
    metadata = {"enabled": False, "mode": "rama-map", "phi0_deg": -60.0}
    reconciled = reconcile_resume_secondary_cv_metadata(metadata, [0.1, 0.2])
    assert reconciled["enabled"] is True
    # Other fields already present in the manifest must be preserved, not lost.
    assert reconciled["mode"] == "rama-map"
    assert reconciled["phi0_deg"] == -60.0
    captured = capsys.readouterr()
    assert "WARNING" in captured.out


def test_healthy_resume_already_enabled_does_not_warn(capsys):
    metadata = {"enabled": True, "mode": "rama-map"}
    reconciled = reconcile_resume_secondary_cv_metadata(metadata, [0.1, 0.2])
    assert reconciled["enabled"] is True
    assert reconciled["mode"] == "rama-map"
    captured = capsys.readouterr()
    assert "WARNING" not in captured.out


def test_legitimate_1d_resume_no_centers_no_metadata_is_left_alone(capsys):
    reconciled = reconcile_resume_secondary_cv_metadata({"enabled": False}, None)
    assert reconciled["enabled"] is False
    captured = capsys.readouterr()
    assert "WARNING" not in captured.out


def test_legitimate_1d_resume_empty_centers_list_is_left_alone(capsys):
    reconciled = reconcile_resume_secondary_cv_metadata(None, [])
    assert reconciled.get("enabled") is not True
    captured = capsys.readouterr()
    assert "WARNING" not in captured.out


def test_does_not_mutate_input_metadata_dict():
    original = {"enabled": False, "mode": "rama-map"}
    reconcile_resume_secondary_cv_metadata(original, [0.1])
    assert original["enabled"] is False


# --- Integration tests: through load_resume_run_definition -----------------
#
# These build a realistic checkpoint manifest (the dict shape produced by the
# production checkpoint writer / gareus_metadata.json) and call the real
# resume loader, rather than the isolated helper. This is the level at which
# the original bug was invisible: the loader's own enabled/disabled branch
# nulls secondary_cv_centers/secondary_cv_k_kcal_mol whenever secondary_cv
# metadata is not enabled, so reconciliation must happen against the raw
# values inside the loader, before that branch runs.

def _build_desynced_manifest() -> dict:
    return {
        "windows_A": [0.10, 0.20, 0.30],
        "window_k_kcal_mol_A2": [5.0, 5.0, 5.0],
        "cv_atom1_index": 3,
        "cv_atom2_index": 47,
        "cv_label": "nonlocal contact fraction",
        "shared_gamd_calibration_steps": 200000,
        # Desynced: enabled is False, but the real CV2 definition fields
        # (mode, phi0_deg, psi0_deg, sigma_deg, ...) are still present, and
        # so are non-empty secondary_cv_centers / secondary_cv_k_kcal_mol —
        # this is what a checkpoint corrupted mid-write or hand-edited
        # between windows would look like, not a genuine 1D run.
        "secondary_cv": {
            "enabled": False,
            "mode": "rama-map",
            "label": "explicit Ramachandran basin map (beta/PPII/right-alpha/left-alpha)",
            "phi0_deg": -60.0,
            "psi0_deg": -45.0,
            "sigma_deg": 35.0,
            "range_min": -1.0,
            "range_max": 1.0,
        },
        "secondary_cv_centers": [-1.0, -1.0 / 3.0, 1.0 / 3.0],
        "secondary_cv_k_kcal_mol": [3.0, 3.0, 3.0],
    }


def test_load_resume_run_definition_reconciles_desynced_secondary_cv(tmp_path: Path, capsys):
    """A checkpoint with real secondary_cv_centers/k but enabled=False metadata
    must resume as 2D, loudly — not silently collapse to a 1D resume."""
    out_dir = tmp_path / "run"
    out_dir.mkdir()
    manifest = _build_desynced_manifest()

    result = load_resume_run_definition(out_dir, topology=None, args=SimpleNamespace(), manifest=manifest)

    captured = capsys.readouterr()
    assert "WARNING" in captured.out
    assert "secondary_cv_centers" in captured.out

    meta = result["secondary_cv_metadata"]
    assert meta["enabled"] is True
    # The original CV2 definition must survive, not degrade to a bare
    # {"enabled": True} or generic "custom" defaults.
    assert meta["mode"] == "rama-map"
    assert meta["phi0_deg"] == -60.0
    assert meta["psi0_deg"] == -45.0
    assert meta["sigma_deg"] == 35.0

    centers = result["secondary_cv_centers"]
    assert centers is not None
    assert list(centers) == manifest["secondary_cv_centers"]

    k_list = result["secondary_cv_k_kcal_list"]
    assert k_list is not None
    assert list(k_list) == manifest["secondary_cv_k_kcal_mol"]


def test_load_resume_run_definition_leaves_genuine_1d_resume_alone(tmp_path: Path, capsys):
    """A resume with no secondary CV at all (no centers, no metadata) must not
    be reconciled into a fabricated 2D run."""
    out_dir = tmp_path / "run"
    out_dir.mkdir()
    manifest = _build_desynced_manifest()
    del manifest["secondary_cv"]
    del manifest["secondary_cv_centers"]
    del manifest["secondary_cv_k_kcal_mol"]

    result = load_resume_run_definition(out_dir, topology=None, args=SimpleNamespace(), manifest=manifest)

    captured = capsys.readouterr()
    assert "WARNING" not in captured.out
    assert result["secondary_cv_metadata"] == {"enabled": False}
    assert result["secondary_cv_centers"] is None
    assert result["secondary_cv_k_kcal_list"] is None


def test_load_resume_run_definition_healthy_2d_resume_is_unaffected(tmp_path: Path, capsys):
    """A checkpoint where metadata is already correctly enabled must resume
    as 2D without emitting a reconciliation warning."""
    out_dir = tmp_path / "run"
    out_dir.mkdir()
    manifest = _build_desynced_manifest()
    manifest["secondary_cv"]["enabled"] = True

    result = load_resume_run_definition(out_dir, topology=None, args=SimpleNamespace(), manifest=manifest)

    captured = capsys.readouterr()
    assert "WARNING" not in captured.out

    meta = result["secondary_cv_metadata"]
    assert meta["enabled"] is True
    assert meta["mode"] == "rama-map"
    assert list(result["secondary_cv_centers"]) == manifest["secondary_cv_centers"]
    assert list(result["secondary_cv_k_kcal_list"]) == manifest["secondary_cv_k_kcal_mol"]
