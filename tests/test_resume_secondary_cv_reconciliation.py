"""Unit tests for reconcile_resume_secondary_cv_metadata — pure logic.

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
"""

from __future__ import annotations

from gareus.production import reconcile_resume_secondary_cv_metadata


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
