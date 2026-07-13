import argparse
from pathlib import Path

import numpy as np
import pytest

from gareus.cv import (
    secondary_cv_enabled,
    secondary_cv_is_transition,
    secondary_cv_mode,
    secondary_cv_range,
)
from gareus.tica import (
    TICAResult,
    compute_bootstrap_torsion_pca,
    project_tica1,
)


def _args(**kw) -> argparse.Namespace:
    ns = argparse.Namespace()
    for key, value in kw.items():
        setattr(ns, key, value)
    return ns


def test_secondary_centers_are_data_derived_only_for_torsion_pca():
    from gareus.adaptive_feedback import secondary_centers_are_data_derived

    assert secondary_centers_are_data_derived(_args(secondary_cv="torsion-pca")) is True
    assert secondary_centers_are_data_derived(_args(secondary_cv="alpha-coil-beta")) is False
    assert secondary_centers_are_data_derived(_args(secondary_cv="rama-map")) is False
    assert secondary_centers_are_data_derived(_args(secondary_cv="rama-regions")) is False
    assert secondary_centers_are_data_derived(_args(secondary_cv="none")) is False


def test_fallback_secondary_centers_skips_torsion_pca(monkeypatch):
    """Regression test: run_adaptive_feedback_auto_loop used to pre-set
    secondary_cv_centers to the generic [0.25, 0.55, 0.85] placeholder before
    the seed-PCA bootstrap ran, permanently blocking the data-derived ladder
    (and --cv2-n-centers) via the bootstrap's own "only fill in when None" guard.
    """
    import gareus.adaptive_feedback as af_mod

    args = _args(
        secondary_cv="torsion-pca",
        secondary_cv_centers=None,
        adaptive_secondary_cv="auto",
    )

    def _boom(*_a, **_kw):
        raise AssertionError("adaptive_secondary_default_centers must not run for torsion-pca")

    monkeypatch.setattr(af_mod, "adaptive_secondary_default_centers", _boom)

    # Left None: the round-1 setup's seed-PCA bootstrap fills this in next.
    assert af_mod._fallback_secondary_centers_if_needed(args, n_rounds=1) is None


def test_fallback_secondary_centers_still_applies_for_fixed_ladder_modes():
    from gareus.adaptive_feedback import _fallback_secondary_centers_if_needed

    args = _args(
        secondary_cv="alpha-coil-beta",
        secondary_cv_centers=None,
        adaptive_secondary_cv="auto",
    )
    assert _fallback_secondary_centers_if_needed(args, n_rounds=1) == [-0.80, 0.0, 0.80]


def test_fallback_secondary_centers_noop_when_already_set_or_disabled():
    from gareus.adaptive_feedback import _fallback_secondary_centers_if_needed

    already_set = _args(secondary_cv="alpha-coil-beta", secondary_cv_centers=[-1.0, 0.0, 1.0], adaptive_secondary_cv="auto")
    assert _fallback_secondary_centers_if_needed(already_set, n_rounds=1) is None

    zero_rounds = _args(secondary_cv="alpha-coil-beta", secondary_cv_centers=None, adaptive_secondary_cv="auto")
    assert _fallback_secondary_centers_if_needed(zero_rounds, n_rounds=0) is None

    fixed_mode = _args(secondary_cv="alpha-coil-beta", secondary_cv_centers=None, adaptive_secondary_cv="fixed")
    assert _fallback_secondary_centers_if_needed(fixed_mode, n_rounds=1) is None


def test_torsion_pca_mode_aliases_and_range():
    assert secondary_cv_mode("torsion-pca") == "torsion-pca"
    assert secondary_cv_mode("bootstrap-torsion") == "torsion-pca"
    assert secondary_cv_mode("bootstrap_linear") == "torsion-pca"
    assert secondary_cv_mode("torsion-linear") == "torsion-pca"
    assert secondary_cv_mode(_args(secondary_cv="bootstrap-torsion")) == "torsion-pca"
    assert secondary_cv_mode({"secondary_cv": "bootstrap-torsion"}) == "torsion-pca"
    assert secondary_cv_enabled(_args(secondary_cv="torsion-pca")) is True
    assert secondary_cv_is_transition("torsion-pca") is True
    assert secondary_cv_range("torsion-pca") == pytest.approx((-6.0, 6.0))


def test_bootstrap_torsion_cli_keys_are_known_config_dests():
    from gareus.cli import build_gareus_parser
    from gareus.config import _build_known_config_dests

    parser = build_gareus_parser()
    dests = _build_known_config_dests(parser)
    for key in (
        "bootstrap_torsion_source",
        "bootstrap_torsion_residualize_against_cv1",
        "bootstrap_torsion_component",
        "bootstrap_torsion_min_seed_count",
        "bootstrap_torsion_state_file",
    ):
        assert key in dests


def test_torsion_pca_no_bad_generic_auto_centers():
    from gareus.cli import parse_args

    args = parse_args(["--seq", "GYDPETGTWG", "--cv1", "contacts", "--cv2", "torsion-pca"])
    assert args.cv2 == "torsion-pca"
    assert args.cv2_centers is None
    assert args.secondary_cv_centers is None
    assert args._cv2_auto_centers is True


def test_cv2_n_centers_cli_flag_defaults_and_overrides():
    from gareus.cli import parse_args

    args = parse_args(["--seq", "GYDPETGTWG", "--cv1", "contacts", "--cv2", "torsion-pca"])
    assert args.cv2_n_centers == 3

    args = parse_args([
        "--seq", "GYDPETGTWG", "--cv1", "contacts", "--cv2", "torsion-pca",
        "--cv2-n-centers", "6",
    ])
    assert args.cv2_n_centers == 6


def test_ensure_bootstrap_torsion_cv_ready_builds_seed_state_and_auto_centers(tmp_path, monkeypatch):
    import gareus.cv as cv_mod
    import gareus.seeding as seeding_mod
    from gareus.production import _ensure_bootstrap_torsion_cv_ready

    out_dir = tmp_path / "out"
    seed_dir = tmp_path / "seeds"
    seed_dir.mkdir()
    phi = [(0, 1, 2, 3)]
    psi = [(1, 2, 3, 4)]

    def _positions(theta: float) -> np.ndarray:
        return np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [1.0 + np.cos(theta), 1.0, np.sin(theta)],
                [2.0 + np.cos(0.5 * theta), 1.5, np.sin(0.5 * theta)],
            ],
            dtype=float,
        )

    library = [
        {"positions_nm": _positions(theta), "primary_cv_value": float(theta)}
        for theta in np.linspace(0.2, 2.6, 8)
    ]
    args = _args(
        secondary_cv="torsion-pca",
        bootstrap_torsion_source="seeds",
        bootstrap_torsion_residualize_against_cv1=True,
        bootstrap_torsion_component=1,
        bootstrap_torsion_min_seed_count=3,
        bootstrap_torsion_state_file="",
        seed_conformers_dir=str(seed_dir),
        _cv2_auto_centers=True,
        secondary_cv_centers=None,
        cv2_centers=None,
    )

    monkeypatch.setattr(cv_mod, "secondary_structure_torsions", lambda topology: (phi, psi))
    monkeypatch.setattr(
        seeding_mod,
        "load_genpept_conformer_library",
        lambda *args, **kwargs: list(library),
    )

    summary = _ensure_bootstrap_torsion_cv_ready(args, out_dir, topology=object(), primary_cv_def={"mode": "distance"})

    state_path = out_dir / "tica" / "bootstrap_torsion_cv.json"
    assert Path(args.bootstrap_torsion_state_file) == state_path
    assert state_path.exists()
    assert summary["state_file"] == str(state_path)
    assert summary["method"] == "pca"
    assert summary["n_samples"] == len(library)
    assert len(args.cv2_centers) == 3
    assert args.secondary_cv_centers == args.cv2_centers
    assert summary["seed_projection_centers"] == args.cv2_centers
    assert all(-6.0 <= value <= 6.0 for value in args.cv2_centers)


def test_ensure_bootstrap_torsion_cv_ready_honours_cv2_n_centers(tmp_path, monkeypatch):
    import gareus.cv as cv_mod
    import gareus.seeding as seeding_mod
    from gareus.production import _ensure_bootstrap_torsion_cv_ready

    out_dir = tmp_path / "out"
    seed_dir = tmp_path / "seeds"
    seed_dir.mkdir()
    phi = [(0, 1, 2, 3)]
    psi = [(1, 2, 3, 4)]

    def _positions(theta: float) -> np.ndarray:
        return np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [1.0 + np.cos(theta), 1.0, np.sin(theta)],
                [2.0 + np.cos(0.5 * theta), 1.5, np.sin(0.5 * theta)],
            ],
            dtype=float,
        )

    library = [
        {"positions_nm": _positions(theta), "primary_cv_value": float(theta)}
        for theta in np.linspace(0.2, 2.6, 12)
    ]
    args = _args(
        secondary_cv="torsion-pca",
        bootstrap_torsion_source="seeds",
        bootstrap_torsion_residualize_against_cv1=True,
        bootstrap_torsion_component=1,
        bootstrap_torsion_min_seed_count=3,
        bootstrap_torsion_state_file="",
        seed_conformers_dir=str(seed_dir),
        _cv2_auto_centers=True,
        secondary_cv_centers=None,
        cv2_centers=None,
        cv2_n_centers=6,
    )

    monkeypatch.setattr(cv_mod, "secondary_structure_torsions", lambda topology: (phi, psi))
    monkeypatch.setattr(
        seeding_mod,
        "load_genpept_conformer_library",
        lambda *args, **kwargs: list(library),
    )

    summary = _ensure_bootstrap_torsion_cv_ready(args, out_dir, topology=object(), primary_cv_def={"mode": "distance"})

    assert len(args.cv2_centers) == 6
    assert args.secondary_cv_centers == args.cv2_centers
    assert summary["seed_projection_centers"] == args.cv2_centers
    assert all(-6.0 <= value <= 6.0 for value in args.cv2_centers)
    assert sorted(args.cv2_centers) == args.cv2_centers


def test_seed_projection_centers_respects_n_centers_param():
    from gareus.production import _seed_projection_centers

    rng = np.random.default_rng(7)
    values = rng.normal(size=500)

    centers3 = _seed_projection_centers(values, n_centers=3)
    centers6 = _seed_projection_centers(values, n_centers=6)

    assert len(centers3) == 3
    assert len(centers6) == 6
    assert len(set(round(c, 6) for c in centers6)) == 6
    assert centers6 == sorted(centers6)
    # Both use the same 2%/98% outer quantiles regardless of count, so they
    # should span (near-)the same full distribution.
    assert centers6[0] == pytest.approx(centers3[0], abs=1.0e-9)
    assert centers6[-1] == pytest.approx(centers3[-1], abs=1.0e-9)


def test_ensure_bootstrap_torsion_cv_ready_rejects_non_pca_state(tmp_path, monkeypatch):
    import gareus.cv as cv_mod
    import gareus.seeding as seeding_mod
    from gareus.production import _ensure_bootstrap_torsion_cv_ready

    out_dir = tmp_path / "out"
    seed_dir = tmp_path / "seeds"
    seed_dir.mkdir()
    phi = [(0, 1, 2, 3)]
    psi = [(1, 2, 3, 4)]

    def _positions(theta: float) -> np.ndarray:
        return np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [1.0 + np.cos(theta), 1.0, np.sin(theta)],
                [2.0 + np.cos(0.5 * theta), 1.5, np.sin(0.5 * theta)],
            ],
            dtype=float,
        )

    library = [
        {"positions_nm": _positions(theta), "primary_cv_value": float(theta)}
        for theta in np.linspace(0.2, 2.6, 8)
    ]
    state_path = out_dir / "tica" / "bootstrap_torsion_cv.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    TICAResult(
        weights=np.ones(4, dtype=float),
        eigenvalue=0.5,
        mean=np.zeros(4, dtype=float),
        offset=0.0,
        lag=1,
        phi_torsion_indices=phi,
        psi_torsion_indices=psi,
        n_samples=len(library),
        method="tica",
    ).save(state_path)

    args = _args(
        secondary_cv="torsion-pca",
        bootstrap_torsion_source="seeds",
        bootstrap_torsion_residualize_against_cv1=True,
        bootstrap_torsion_component=1,
        bootstrap_torsion_min_seed_count=3,
        bootstrap_torsion_state_file=str(state_path),
        seed_conformers_dir=str(seed_dir),
        _cv2_auto_centers=True,
        secondary_cv_centers=None,
        cv2_centers=None,
    )

    monkeypatch.setattr(cv_mod, "secondary_structure_torsions", lambda topology: (phi, psi))
    monkeypatch.setattr(
        seeding_mod,
        "load_genpept_conformer_library",
        lambda *args, **kwargs: list(library),
    )

    with pytest.raises((ValueError, RuntimeError), match="method.*pca|pca.*method"):
        _ensure_bootstrap_torsion_cv_ready(args, out_dir, topology=object(), primary_cv_def={"mode": "distance"})


def test_bootstrap_torsion_pca_uses_atom_name_mapping_for_genpept_seeds(tmp_path, monkeypatch):
    import gareus.cv as cv_mod
    from gareus.production import _ensure_bootstrap_torsion_cv_ready

    class Atom:
        def __init__(self, index, name):
            self.index = index
            self.name = name

    class Residue:
        def __init__(self, index, name, atoms):
            self.index = index
            self.name = name
            self._atoms = [Atom(index + i, atom_name) for i, atom_name in enumerate(atoms)]

        def atoms(self):
            return iter(self._atoms)

    class Topology:
        def __init__(self):
            self._residues = [
                Residue(0, "GLY", ["N", "CA", "C", "O"]),
                Residue(4, "TYR", ["N", "CA", "C", "O"] + [f"SC{i}" for i in range(44)]),
                Residue(52, "ASP", ["N", "CA", "C", "O"]),
            ]

        def residues(self):
            return iter(self._residues)

    def write_seed(path: Path, bend: float) -> None:
        rows = [
            ("N", 1, 0.0, 0.0, 0.0),
            ("CA", 1, 1.0, 0.0, 0.0),
            ("C", 1, 2.0, 0.0, 0.0),
            ("O", 1, 2.4, -0.6, 0.0),
            ("N", 2, 2.8, 0.2, 0.2 * np.sin(bend)),
            ("CA", 2, 3.4, 1.0, 0.3 * np.cos(bend)),
            ("C", 2, 4.6, 0.8, 0.7 * np.sin(bend)),
            ("O", 2, 5.0, 0.0, 0.4),
            ("N", 3, 5.3, 1.8, 0.5 * np.cos(bend)),
            ("CA", 3, 6.5, 1.7, 0.2),
            ("C", 3, 7.2, 2.6, -0.2),
            ("O", 3, 6.8, 3.6, -0.6),
        ]
        text = "".join(
            f"ATOM  {i:5d} {name:^4s} GLY A{resid:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           {name[0]:>2s}\n"
            for i, (name, resid, x, y, z) in enumerate(rows, start=1)
        )
        path.write_text(text, encoding="utf-8")

    seed_dir = tmp_path / "seeds"
    seed_dir.mkdir()
    csv_rows = ["survivor_pdb_path\n"]
    for idx, bend in enumerate(np.linspace(0.2, 2.8, 6)):
        pdb = seed_dir / f"seed_{idx:03d}.pdb"
        write_seed(pdb, float(bend))
        csv_rows.append(f"{pdb.name}\n")
    (seed_dir / "final_survivor_seeds.csv").write_text("".join(csv_rows), encoding="utf-8")

    phi = [(2, 4, 5, 6)]
    psi = [(4, 5, 6, 52)]
    monkeypatch.setattr(cv_mod, "secondary_structure_torsions", lambda topology: (phi, psi))
    args = _args(
        secondary_cv="torsion-pca",
        bootstrap_torsion_source="seeds",
        bootstrap_torsion_residualize_against_cv1=False,
        bootstrap_torsion_component=1,
        bootstrap_torsion_min_seed_count=3,
        bootstrap_torsion_state_file="",
        seed_conformers_dir=str(seed_dir),
        _cv2_auto_centers=False,
        secondary_cv_centers=None,
        cv2_centers=None,
    )
    primary = {
        "mode": "nonlocal-contacts",
        "contact_pairs": [(20, 21, 1.0)],
        "contact_normalize": True,
        "_np_r0_nm": 0.36,
        "_np_beta_nm_inv": 25.0,
    }

    summary = _ensure_bootstrap_torsion_cv_ready(args, tmp_path / "out", Topology(), primary)

    assert summary["n_samples"] == 6
    state = TICAResult.load(tmp_path / "out" / "tica" / "bootstrap_torsion_cv.json")
    assert state.phi_torsion_indices == phi
    assert state.psi_torsion_indices == psi


def test_linear_torsion_force_stays_under_customcv_variable_limit(tmp_path):
    from gareus.production import _add_linear_torsion_cv_force

    class FakeCustomTorsionForce:
        def __init__(self, expr):
            self.expr = expr
            self.torsions = []
            self.globals = []
            self.per_torsion = []

        def addGlobalParameter(self, name, value):
            self.globals.append((name, value))

        def addPerTorsionParameter(self, name):
            self.per_torsion.append(name)

        def addTorsion(self, a, b, c, d, params):
            self.torsions.append((a, b, c, d, list(params)))

    class FakeCustomCVForce:
        def __init__(self, expr):
            self.expr = expr
            self.variables = []
            self.globals = []
            self.group = None

        def addCollectiveVariable(self, name, variable):
            if len(self.variables) >= 32:
                raise RuntimeError("CustomCVForce cannot have more than 32 collective variables")
            self.variables.append((name, variable))

        def addGlobalParameter(self, name, value):
            self.globals.append((name, value))

        def setEnergyFunction(self, expr):
            self.expr = expr

        def setForceGroup(self, group):
            self.group = group

    class FakeSystem:
        def __init__(self):
            self.forces = []

        def addForce(self, force):
            self.forces.append(force)

    class FakeOpenMM:
        CustomCVForce = FakeCustomCVForce
        CustomTorsionForce = FakeCustomTorsionForce

    phi = [(i, i + 1, i + 2, i + 3) for i in range(9)]
    psi = [(100 + i, 101 + i, 102 + i, 103 + i) for i in range(8)]
    weights = np.linspace(-1.0, 1.0, 2 * len(phi) + 2 * len(psi))
    result = TICAResult(
        weights=weights,
        eigenvalue=1.0,
        mean=np.zeros_like(weights),
        offset=0.25,
        lag=0,
        phi_torsion_indices=phi,
        psi_torsion_indices=psi,
        n_samples=64,
        method="pca",
        explained_variance_ratio=0.4,
    )
    system = FakeSystem()

    info = _add_linear_torsion_cv_force(
        FakeOpenMM(),
        system,
        phi,
        psi,
        result,
        mode="torsion-pca",
        state_path=tmp_path / "bootstrap_torsion_cv.json",
        force_group=29,
    )

    assert info["enabled"] is True
    assert len(system.forces) == 1
    assert len(system.forces[0].variables) <= 4


def test_linear_torsion_state_loader_rejects_wrong_method(tmp_path):
    from gareus.production import _linear_torsion_state_for_mode

    tica_path = tmp_path / "tica.json"
    pca_path = tmp_path / "pca.json"
    common = dict(
        weights=np.ones(4, dtype=float),
        eigenvalue=0.5,
        mean=np.zeros(4, dtype=float),
        offset=0.0,
        lag=1,
        phi_torsion_indices=[(0, 1, 2, 3)],
        psi_torsion_indices=[(1, 2, 3, 4)],
        n_samples=8,
    )
    TICAResult(method="tica", **common).save(tica_path)
    TICAResult(method="pca", explained_variance_ratio=0.5, **common).save(pca_path)

    with pytest.raises(RuntimeError, match="torsion-pca.*method='pca'"):
        _linear_torsion_state_for_mode(_args(bootstrap_torsion_state_file=str(tica_path)), "torsion-pca")
    with pytest.raises(RuntimeError, match="tica-linear.*method='tica'"):
        _linear_torsion_state_for_mode(_args(tica_state_file=str(pca_path)), "tica-linear")


def test_resume_restores_linear_torsion_state_path_from_metadata(tmp_path):
    from gareus.production import _restore_secondary_cv_args_from_metadata

    tica_path = tmp_path / "tica.json"
    pca_path = tmp_path / "pca.json"
    tica_path.write_text("{}", encoding="utf-8")
    pca_path.write_text("{}", encoding="utf-8")

    args = _args(secondary_cv="none", tica_state_file="", bootstrap_torsion_state_file="")
    _restore_secondary_cv_args_from_metadata(
        args,
        {
            "enabled": True,
            "mode": "tica-linear",
            "tica_state_path": str(tica_path),
        },
    )
    assert args.secondary_cv == "tica-linear"
    assert args.tica_state_file == str(tica_path)

    args = _args(secondary_cv="none", tica_state_file="", bootstrap_torsion_state_file="")
    _restore_secondary_cv_args_from_metadata(
        args,
        {
            "enabled": True,
            "mode": "torsion-pca",
            "tica_state_path": str(pca_path),
        },
    )
    assert args.secondary_cv == "torsion-pca"
    assert args.bootstrap_torsion_state_file == str(pca_path)


def test_resume_rejects_missing_linear_torsion_state_path(tmp_path):
    from gareus.production import _restore_secondary_cv_args_from_metadata

    args = _args(secondary_cv="none", tica_state_file="", bootstrap_torsion_state_file="")
    with pytest.raises(RuntimeError, match="missing.*tica_state_path"):
        _restore_secondary_cv_args_from_metadata(
            args,
            {
                "enabled": True,
                "mode": "tica-linear",
                "tica_state_path": str(tmp_path / "missing.json"),
            },
        )


def test_bootstrap_pca_returns_ticaresult_compatible_state():
    rng = np.random.default_rng(123)
    latent = np.linspace(-2.0, 2.0, 80)
    X = np.column_stack(
        [
            np.sin(latent),
            np.cos(latent),
            0.5 * latent,
            rng.normal(0.0, 0.02, size=latent.shape),
        ]
    )
    result = compute_bootstrap_torsion_pca(
        X,
        residualize=False,
        component=1,
        phi_torsion_indices=[(0, 1, 2, 3)],
        psi_torsion_indices=[(1, 2, 3, 4)],
    )
    assert isinstance(result, TICAResult)
    assert result.method == "pca"
    assert result.lag == 0
    assert result.weights.shape == (4,)
    assert result.mean.shape == (4,)
    assert result.n_samples == 80
    assert result.eigenvalue > 0.0
    projected = project_tica1(X, result)
    assert projected.shape == (80,)
    assert np.isfinite(projected).all()


def test_bootstrap_pca_residualizes_linear_cv1_signal():
    cv1 = np.linspace(-1.0, 1.0, 120)
    orthogonal = np.sin(np.linspace(0.0, 4.0 * np.pi, 120))
    X = np.column_stack(
        [
            3.0 * cv1 + 0.05 * orthogonal,
            -2.0 * cv1 + 0.10 * orthogonal,
            orthogonal,
            np.cos(np.linspace(0.0, 4.0 * np.pi, 120)),
        ]
    )
    result = compute_bootstrap_torsion_pca(X, cv1=cv1, residualize=True, component=1)
    projected = project_tica1(X, result)
    corr = np.corrcoef(projected, cv1)[0, 1]
    assert abs(corr) < 0.20


def test_bootstrap_pca_raw_projection_remains_decorrelated_from_cv1():
    cv1 = np.linspace(-1.0, 1.0, 200)
    orthogonal = np.sin(np.linspace(0.0, 8.0 * np.pi, 200))
    nuisance = np.cos(np.linspace(0.0, 6.0 * np.pi, 200))
    X = np.column_stack(
        [
            50.0 * cv1 + 5.0 * orthogonal,
            -40.0 * cv1 + 5.0 * orthogonal,
            0.20 * nuisance,
            -0.10 * nuisance,
        ]
    )
    result = compute_bootstrap_torsion_pca(X, cv1=cv1, residualize=True, component=1)

    projected = project_tica1(X, result)
    corr = np.corrcoef(projected, cv1)[0, 1]
    slope = np.polyfit(cv1, projected, deg=1)[0]

    assert abs(corr) < 0.05
    assert abs(slope) < 0.05


def test_bootstrap_pca_fail_closed_on_zero_variance():
    X = np.ones((30, 4), dtype=float)
    with pytest.raises(ValueError, match="zero bootstrap torsion PCA variance"):
        compute_bootstrap_torsion_pca(X, residualize=False)


def test_bootstrap_pca_fail_closed_on_zero_component_variance():
    X = np.column_stack([np.linspace(-1.0, 1.0, 12), np.zeros(12), np.zeros(12), np.zeros(12)])
    with pytest.raises(ValueError, match="zero bootstrap torsion PCA variance"):
        compute_bootstrap_torsion_pca(X, residualize=False, component=2)


def test_ticaresult_preserves_method_roundtrip(tmp_path):
    result = compute_bootstrap_torsion_pca(np.eye(6, dtype=float), residualize=False)
    path = tmp_path / "bootstrap.json"
    result.save(path)
    loaded = TICAResult.load(path)
    assert loaded.method == "pca"
    assert loaded.lag == 0
    assert loaded.weights.shape == result.weights.shape
    assert loaded.explained_variance_ratio == result.explained_variance_ratio


def test_positions_score_uses_linear_torsion_weights():
    from gareus.cv import secondary_structure_score_from_positions_nm
    from gareus.tica import backbone_dihedral_features

    positions = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [1.0, 1.0, 1.0],
    ], dtype=float)
    metadata = {
        "enabled": True,
        "mode": "torsion-pca",
        "weights": [1.0, 0.0],
        "tica_offset": 0.25,
        "phi_torsions": [[0, 1, 2, 3]],
        "psi_torsions": [],
    }
    value = secondary_structure_score_from_positions_nm(positions, metadata)
    expected = (
        backbone_dihedral_features(positions, [(0, 1, 2, 3)], [])
        @ np.asarray(metadata["weights"], dtype=float)
        + float(metadata["tica_offset"])
    )
    assert np.isfinite(value)
    assert value == pytest.approx(expected, abs=1.0e-6)
