"""Unit tests for _write_box_audit in gareus.system_setup.

These tests use only numpy and types.SimpleNamespace — no OpenMM, PeptideBuilder,
or gamd-openmm is imported.
"""

import json
import types

import numpy as np
import pytest

from gareus.system_setup import _write_box_audit


def _make_args(padding_nm: float = 1.0, seq: str = "GYDPETGTWG") -> types.SimpleNamespace:
    return types.SimpleNamespace(padding_nm=padding_nm, seq=seq)


def _make_pos_nm(extent_xyz=(2.0, 1.5, 1.0)):
    """Return a (2, 3) position array whose bounding box matches *extent_xyz*."""
    lo = np.array([0.0, 0.0, 0.0])
    hi = np.array(extent_xyz, dtype=float)
    return np.vstack([lo, hi])


def _make_ca_pos_nm(n: int = 10, spacing: float = 0.38):
    """Return n Cα positions in a straight line spaced *spacing* nm apart."""
    return np.column_stack([
        np.arange(n, dtype=float) * spacing,
        np.zeros(n),
        np.zeros(n),
    ])


class TestAuditFields:
    """Verify all required JSON fields are present and correctly computed."""

    def test_fields_present(self, tmp_path):
        pos_nm = _make_pos_nm((2.0, 1.5, 1.0))
        ca_pos_nm = _make_ca_pos_nm(10)
        args = _make_args(padding_nm=1.0, seq="GYDPETGTWG")
        box_nm = 6.0
        contour_nm = 4.0

        result = _write_box_audit(tmp_path, pos_nm, ca_pos_nm, contour_nm, box_nm, args)

        required_keys = {
            "actual_extent_nm",
            "sequence_contour_estimate_nm",
            "backbone_path_length_nm",
            "box_size_nm",
            "padding_nm",
            "minimum_margin_nm",
            "pbc_self_contact_warning",
            "warning_message",
        }
        assert required_keys == set(result.keys())

    def test_json_written(self, tmp_path):
        pos_nm = _make_pos_nm((2.0, 1.5, 1.0))
        ca_pos_nm = _make_ca_pos_nm(10)
        args = _make_args(padding_nm=1.0, seq="GYDPETGTWG")
        _write_box_audit(tmp_path, pos_nm, ca_pos_nm, 4.0, 6.0, args)

        audit_path = tmp_path / "box_audit.json"
        assert audit_path.exists()
        with audit_path.open() as fh:
            data = json.load(fh)
        assert isinstance(data, dict)
        assert "actual_extent_nm" in data

    def test_actual_extent_nm(self, tmp_path):
        # Known extent: [3.0, 2.0, 1.0]
        pos_nm = _make_pos_nm((3.0, 2.0, 1.0))
        ca_pos_nm = _make_ca_pos_nm(10)
        args = _make_args(padding_nm=1.0)
        result = _write_box_audit(tmp_path, pos_nm, ca_pos_nm, 4.0, 8.0, args)

        assert result["actual_extent_nm"] == pytest.approx([3.0, 2.0, 1.0])

    def test_sequence_contour_estimate_10_residues(self, tmp_path):
        # 10 residues → (10-1)*0.38 + 0.40 = 3.42 + 0.40 = 3.82
        pos_nm = _make_pos_nm()
        ca_pos_nm = _make_ca_pos_nm(10)
        args = _make_args(seq="GYDPETGTWG")  # 10 residues
        result = _write_box_audit(tmp_path, pos_nm, ca_pos_nm, 4.0, 8.0, args)

        assert result["sequence_contour_estimate_nm"] == pytest.approx(3.82)

    def test_sequence_contour_uses_ca_count_not_seq_length(self, tmp_path):
        # Give a 5-residue seq but 10 CA atoms — the topology count should win.
        pos_nm = _make_pos_nm()
        ca_pos_nm = _make_ca_pos_nm(10)
        args = _make_args(seq="AACDF")  # only 5 residues in seq
        result = _write_box_audit(tmp_path, pos_nm, ca_pos_nm, 4.0, 8.0, args)
        # 10 CA atoms → (10-1)*0.38 + 0.40 = 3.82
        assert result["sequence_contour_estimate_nm"] == pytest.approx(3.82)

    def test_backbone_path_length(self, tmp_path):
        # 10 CA atoms spaced 0.38 nm → total = 9 * 0.38 = 3.42
        pos_nm = _make_pos_nm()
        ca_pos_nm = _make_ca_pos_nm(10, spacing=0.38)
        args = _make_args()
        result = _write_box_audit(tmp_path, pos_nm, ca_pos_nm, 4.0, 8.0, args)

        assert result["backbone_path_length_nm"] == pytest.approx(9 * 0.38, abs=1e-9)

    def test_backbone_path_length_single_ca_is_zero(self, tmp_path):
        pos_nm = _make_pos_nm()
        ca_pos_nm = _make_ca_pos_nm(1)
        args = _make_args()
        result = _write_box_audit(tmp_path, pos_nm, ca_pos_nm, 4.0, 8.0, args)
        assert result["backbone_path_length_nm"] == pytest.approx(0.0)

    def test_backbone_path_length_empty_ca_is_zero(self, tmp_path):
        pos_nm = _make_pos_nm()
        ca_pos_nm = np.empty((0, 3))
        args = _make_args()
        result = _write_box_audit(tmp_path, pos_nm, ca_pos_nm, 4.0, 8.0, args)
        assert result["backbone_path_length_nm"] == pytest.approx(0.0)

    def test_box_size_nm(self, tmp_path):
        pos_nm = _make_pos_nm()
        ca_pos_nm = _make_ca_pos_nm(10)
        args = _make_args()
        result = _write_box_audit(tmp_path, pos_nm, ca_pos_nm, 4.0, 7.5, args)
        assert result["box_size_nm"] == pytest.approx(7.5)

    def test_padding_nm(self, tmp_path):
        pos_nm = _make_pos_nm()
        ca_pos_nm = _make_ca_pos_nm(10)
        args = _make_args(padding_nm=1.25)
        result = _write_box_audit(tmp_path, pos_nm, ca_pos_nm, 4.0, 8.0, args)
        assert result["padding_nm"] == pytest.approx(1.25)

    def test_minimum_margin_nm(self, tmp_path):
        # extent [2.0, 1.5, 1.0] → max_extent = 2.0
        # box_nm = 8.0 → margin = 8.0/2 - 2.0/2 = 4.0 - 1.0 = 3.0
        pos_nm = _make_pos_nm((2.0, 1.5, 1.0))
        ca_pos_nm = _make_ca_pos_nm(10)
        args = _make_args()
        result = _write_box_audit(tmp_path, pos_nm, ca_pos_nm, 4.0, 8.0, args)
        assert result["minimum_margin_nm"] == pytest.approx(3.0)

    def test_json_values_are_plain_python_types(self, tmp_path):
        """All values must JSON-serialize without error (no numpy scalars)."""
        pos_nm = _make_pos_nm()
        ca_pos_nm = _make_ca_pos_nm(10)
        args = _make_args()
        result = _write_box_audit(tmp_path, pos_nm, ca_pos_nm, 4.0, 8.0, args)
        # json.dumps raises TypeError for numpy scalars — if this passes, we're clean
        json.dumps(result)


class TestPbcSelfContactWarning:
    """Test the PBC self-contact warning logic."""

    def test_no_warning_large_box(self, tmp_path):
        """Box easily large enough: no warning expected.

        seq = "GYDPETGTWG" (10 res) → contour = 3.82 nm
        box_nm = 10.0, padding = 1.0
        threshold = 10.0/2 - 1.0 = 4.0 nm
        3.82 < 4.0 → no warning
        """
        pos_nm = _make_pos_nm()
        ca_pos_nm = _make_ca_pos_nm(10)
        args = _make_args(padding_nm=1.0, seq="GYDPETGTWG")
        result = _write_box_audit(tmp_path, pos_nm, ca_pos_nm, 4.0, 10.0, args)

        assert result["pbc_self_contact_warning"] is False
        assert result["warning_message"] is None

    def test_warning_box_too_small(self, tmp_path):
        """Box too small for the sequence: warning expected.

        seq = "GYDPETGTWG" (10 res) → contour = 3.82 nm
        box_nm = 6.0, padding = 1.0
        threshold = 6.0/2 - 1.0 = 2.0 nm
        3.82 > 2.0 → warning
        """
        pos_nm = _make_pos_nm()
        ca_pos_nm = _make_ca_pos_nm(10)
        args = _make_args(padding_nm=1.0, seq="GYDPETGTWG")
        result = _write_box_audit(tmp_path, pos_nm, ca_pos_nm, 4.0, 6.0, args)

        assert result["pbc_self_contact_warning"] is True
        assert result["warning_message"] is not None
        assert "PBC warning" in result["warning_message"]

    def test_warning_message_contains_numeric_context(self, tmp_path):
        """Warning message must include both contour and threshold values."""
        pos_nm = _make_pos_nm()
        ca_pos_nm = _make_ca_pos_nm(10)
        args = _make_args(padding_nm=1.0, seq="GYDPETGTWG")
        result = _write_box_audit(tmp_path, pos_nm, ca_pos_nm, 4.0, 6.0, args)

        msg = result["warning_message"]
        assert "3.82" in msg   # contour estimate
        assert "2.00" in msg   # threshold = 6.0/2 - 1.0

    def test_warning_boundary_exactly_equal_is_false(self, tmp_path):
        """When contour == threshold exactly, no warning (strict >)."""
        # contour = 3.82, threshold must equal 3.82 → box/2 - padding = 3.82
        # → box = 2*(3.82 + 1.0) = 9.64
        pos_nm = _make_pos_nm()
        ca_pos_nm = _make_ca_pos_nm(10)
        args = _make_args(padding_nm=1.0, seq="GYDPETGTWG")
        box_nm = 2.0 * (3.82 + 1.0)  # = 9.64
        result = _write_box_audit(tmp_path, pos_nm, ca_pos_nm, 4.0, box_nm, args)

        assert result["pbc_self_contact_warning"] is False

    def test_warning_triggers_for_collapsed_input_pdb_scenario(self, tmp_path):
        """Simulates an input PDB of a collapsed 10-residue peptide in a small box.

        A compact structure might have an actual extent of only ~1.5 nm,
        so the *box-sizing* contour is small.  But the sequence contour
        estimate (3.82 nm) reveals PBC risk — the audit must catch this.
        """
        # Compact structure: all atoms clustered in 1.5 nm
        n_atoms = 80
        rng = np.random.default_rng(42)
        pos_nm = rng.uniform(0.0, 1.5, size=(n_atoms, 3))
        ca_pos_nm = _make_ca_pos_nm(10)

        # Box sized from compact extent: ~1.5 + 0.40 + 2*1.0 = 3.9 nm
        box_nm = 3.9
        args = _make_args(padding_nm=1.0, seq="GYDPETGTWG")

        result = _write_box_audit(tmp_path, pos_nm, ca_pos_nm, 1.9, box_nm, args)

        # threshold = 3.9/2 - 1.0 = 0.95; contour 3.82 >> 0.95 → warning
        assert result["pbc_self_contact_warning"] is True
        assert result["sequence_contour_estimate_nm"] == pytest.approx(3.82)
