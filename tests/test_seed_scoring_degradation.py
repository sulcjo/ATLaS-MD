"""Unit tests for silent GENPEPT seed-scoring collapse detection.

Scientific audit finding: when 2D sampling is active (CV1 = contacts,
CV2 = rama-map) GENPEPT seeds must be scored in the same active CV space as
production. Two failure modes silently collapse that scoring to a
lower-dimensional proxy with no warning:

  (a) primary CV is contacts, but no production contact pair maps into the
      peptide-only atom range, so the loader substitutes terminal distance.
  (b) secondary CV is intended active for seed selection, but the relative
      secondary-CV metadata resolves to None (missing/disabled/unmapped),
      so seed scoring silently becomes CV1-only.

``detect_seed_scoring_degradations`` is a pure helper that flags both cases
from already-resolved flags/definitions, without touching OpenMM or the
filesystem.
"""

from gareus.seeding import detect_seed_scoring_degradations


def test_contacts_primary_with_distance_fallback_is_flagged():
    rel_primary = {"mode": "distance", "label": "terminal distance", "units": "A"}
    degradations = detect_seed_scoring_degradations(
        primary_is_contacts=True,
        rel_primary=rel_primary,
        secondary_available=False,
        seed_secondary_weight=1.0,
        seed_selection_mode="active-cv",
        rel_secondary=None,
    )
    assert len(degradations) == 1
    assert degradations[0]["kind"] == "primary_cv_collapsed_to_distance"


def test_active_secondary_dropped_is_flagged():
    rel_primary = {"mode": "nonlocal-contacts", "contact_pairs": [(0, 1, 1.0)]}
    degradations = detect_seed_scoring_degradations(
        primary_is_contacts=True,
        rel_primary=rel_primary,
        secondary_available=True,
        seed_secondary_weight=1.0,
        seed_selection_mode="active-cv",
        rel_secondary=None,
    )
    assert len(degradations) == 1
    assert degradations[0]["kind"] == "secondary_cv_dropped"


def test_healthy_2d_scoring_has_no_degradations():
    rel_primary = {"mode": "nonlocal-contacts", "contact_pairs": [(0, 1, 1.0)]}
    rel_secondary = {"enabled": True, "phi_torsions": [(0, 1, 2, 3)], "psi_torsions": [(1, 2, 3, 4)]}
    degradations = detect_seed_scoring_degradations(
        primary_is_contacts=True,
        rel_primary=rel_primary,
        secondary_available=True,
        seed_secondary_weight=1.0,
        seed_selection_mode="active-cv",
        rel_secondary=rel_secondary,
    )
    assert degradations == []


def test_distance_primary_run_never_flags_distance_fallback():
    # A run that legitimately chose distance as the primary CV must not be
    # flagged just because rel_primary is distance-mode.
    rel_primary = {"mode": "distance", "label": "terminal distance", "units": "A"}
    degradations = detect_seed_scoring_degradations(
        primary_is_contacts=False,
        rel_primary=rel_primary,
        secondary_available=False,
        seed_secondary_weight=1.0,
        seed_selection_mode="active-cv",
        rel_secondary=None,
    )
    assert degradations == []


def test_secondary_not_intended_active_never_flags_dropped_cv2():
    # seed_selection_mode == "distance" (explicit legacy mode) or weight == 0
    # means CV2 is deliberately excluded from seed scoring; that is not a
    # degradation.
    rel_primary = {"mode": "nonlocal-contacts", "contact_pairs": [(0, 1, 1.0)]}
    degradations_mode = detect_seed_scoring_degradations(
        primary_is_contacts=True,
        rel_primary=rel_primary,
        secondary_available=True,
        seed_secondary_weight=1.0,
        seed_selection_mode="distance",
        rel_secondary=None,
    )
    assert degradations_mode == []

    degradations_weight = detect_seed_scoring_degradations(
        primary_is_contacts=True,
        rel_primary=rel_primary,
        secondary_available=True,
        seed_secondary_weight=0.0,
        seed_selection_mode="active-cv",
        rel_secondary=None,
    )
    assert degradations_weight == []


def test_both_degradations_can_fire_together():
    rel_primary = {"mode": "distance", "label": "terminal distance", "units": "A"}
    degradations = detect_seed_scoring_degradations(
        primary_is_contacts=True,
        rel_primary=rel_primary,
        secondary_available=True,
        seed_secondary_weight=1.0,
        seed_selection_mode="active-cv",
        rel_secondary=None,
    )
    kinds = {d["kind"] for d in degradations}
    assert kinds == {"primary_cv_collapsed_to_distance", "secondary_cv_dropped"}


def test_loader_exposes_resolved_cv_defs_for_degradation_check(tmp_path):
    """The loader must expose rel_primary/rel_secondary via an out-param so
    callers can run detect_seed_scoring_degradations without recomputing the
    private relative-CV mapping logic themselves."""
    from gareus.seeding import load_genpept_conformer_library

    class Atom:
        def __init__(self, index, name):
            self.index = index
            self.name = name

    class Residue:
        def __init__(self, index, name, atoms):
            self.index = index
            self.name = name
            self._atoms = atoms

        def atoms(self):
            return iter(self._atoms)

    class Topology:
        def __init__(self):
            self._res = [
                Residue(0, "GLY", [Atom(0, "N"), Atom(1, "CA"), Atom(2, "C")]),
                Residue(1, "ALA", [Atom(3, "N"), Atom(4, "CA"), Atom(5, "C")]),
            ]

        def residues(self):
            return iter(self._res)

    seed_dir = tmp_path / "seeds"
    seed_dir.mkdir()
    # No conformer rows needed; we only care about the resolved CV defs.
    (seed_dir / "final_survivor_seeds.csv").write_text("survivor_pdb_path\n")

    # Primary CV is nominally contacts, but no contact pair maps into the
    # 6-atom peptide-only range below -> loader must fall back to distance.
    primary = {
        "mode": "nonlocal-contacts",
        "contact_pairs": [(100, 101, 1.0)],
    }
    resolved: dict = {}
    from types import SimpleNamespace

    load_genpept_conformer_library(
        seed_dir,
        cv_atom1=1,
        cv_atom2=4,
        primary_cv_def=primary,
        args=SimpleNamespace(),
        topology=Topology(),
        secondary_cv_metadata=None,
        resolved_cv_defs_out=resolved,
    )
    assert resolved["rel_primary"]["mode"] == "distance"
    assert resolved["rel_secondary"] is None
