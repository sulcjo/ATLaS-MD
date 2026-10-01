"""Contact-map CV1 in production (spec 2026-10-01-contact-map-cv1.md sections 4-5, step 3)."""
from __future__ import annotations

import json
import math
from types import SimpleNamespace

import numpy as np
import pytest

openmm = pytest.importorskip("openmm")
from openmm import app, unit  # noqa: E402

from gareus import contact_map_force as F  # noqa: E402
from gareus import cv as CV  # noqa: E402
from gareus.cv_selection import anchor_spec as AS  # noqa: E402
from gareus.cv_selection import contact_map_cv1 as CM1  # noqa: E402
from gareus.swarm import contact_map as CMAP  # noqa: E402


def _topology(n_res=8):
    top = app.Topology()
    chain = top.addChain()
    C, N, H = (app.Element.getBySymbol(x) for x in ("C", "N", "H"))
    for k in range(n_res):
        res = top.addResidue("ALA", chain, id=str(k + 1))
        top.addAtom("N", N, res)
        top.addAtom("CA", C, res)
        top.addAtom("C", C, res)
        top.addAtom("H", H, res)
    w = top.addResidue("HOH", top.addChain())
    top.addAtom("O", app.Element.getBySymbol("O"), w)
    return top


def _model(top, seed=0, atoms=CMAP.ATOMS_CA):
    d = CMAP.contact_map_definition(top, atoms=atoms)
    rng = np.random.default_rng(seed)
    cfg = CM1.CV1FitConfig()
    return CM1.build_model(d, cfg, rng.normal(size=len(d["pairs"])), 0.37, {"mode": 1, "family": CM1.FAMILY},
                           {"n_rows": 0})


def _positions(top, seed=1, scale=0.6):
    return np.random.default_rng(seed).normal(scale=scale, size=(top.getNumAtoms(), 3))


def _write_model(tmp_path, model):
    path = tmp_path / "cv1_model.json"
    CM1.write_model(path, model)
    return path


def _context(system, platform="Reference"):
    return openmm.Context(system, openmm.VerletIntegrator(0.001), openmm.Platform.getPlatformByName(platform))


# --------------------------------------------------------------------------- the force (G2)

def test_force_energy_matches_both_evaluators_and_forces_match_finite_differences():
    top = _topology()
    model = _model(top)
    x = _positions(top)
    system = openmm.System()
    for _ in range(top.getNumAtoms()):
        system.addParticle(1.0)
    F.add_contact_map_umbrella_force(openmm, system, model)
    ctx = _context(system)
    ctx.setParameter("k", 30.0)
    ctx.setParameter("r0", -0.2)
    ctx.setPositions(x)
    s = ctx.getState(getEnergy=True, getForces=True)
    cv = F.cv_from_positions_nm(model, x)
    fit_cv = float(CM1.evaluate_model(model, CMAP.ContactMapEvaluator(model["contact_map"])(x)))
    assert cv == pytest.approx(fit_cv, abs=1e-12)
    assert s.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole) == pytest.approx(0.5 * 30.0 * (cv + 0.2) ** 2, rel=1e-10)
    f = np.asarray(s.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer))
    h = 1e-6
    for a in range(0, top.getNumAtoms() - 1, 3):
        for d in range(3):
            xp, xm = x.copy(), x.copy()
            xp[a, d] += h
            xm[a, d] -= h
            fd = -(0.5 * 30 * (F.cv_from_positions_nm(model, xp) + 0.2) ** 2
                   - 0.5 * 30 * (F.cv_from_positions_nm(model, xm) + 0.2) ** 2) / (2 * h)
            assert f[a, d] == pytest.approx(fd, abs=1e-5)


@pytest.mark.parametrize("platform", ["Reference", "CPU"])
def test_far_apart_residues_stay_finite(platform):
    top = _topology()
    model = _model(top)
    x = _positions(top, scale=0.3)
    x[:4] += np.array([5.0, 0.0, 0.0])               # residue 1 5 nm away: all its sums underflow
    system = openmm.System()
    for _ in range(top.getNumAtoms()):
        system.addParticle(1.0)
    F.add_contact_map_umbrella_force(openmm, system, model)
    ctx = _context(system, platform)
    ctx.setParameter("k", 10.0)
    ctx.setPositions(x)
    s = ctx.getState(getEnergy=True, getForces=True)
    assert math.isfinite(s.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole))
    assert np.isfinite(np.asarray(s.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer))).all()
    assert math.isfinite(F.cv_from_positions_nm(model, x))


def test_atom_map_reads_a_renumbered_structure():
    top = _topology()
    model = _model(top)
    x = _positions(top)
    perm = np.random.default_rng(5).permutation(top.getNumAtoms())
    y = x[perm]                                       # y[k] = x[perm[k]]
    atom_map = {int(a): int(np.flatnonzero(perm == a)[0]) for a in range(top.getNumAtoms())}
    assert F.cv_from_positions_nm(model, y, atom_map=atom_map) == pytest.approx(F.cv_from_positions_nm(model, x), abs=1e-12)


# --------------------------------------------------------------------------- CV1 mode

def test_mode_helpers_keep_contacts_strict():
    assert CV.primary_cv_mode("contact-map") == "contact-map"
    assert CV.primary_cv_is_dimensionless("contact-map") and not CV.primary_cv_is_contacts("contact-map")
    assert CV.primary_cv_units("contact-map") == "dimensionless"
    assert CV.primary_k_units("contact-map") == "kcal/mol/CV^2"
    assert CV.primary_center_to_openmm_value(1.25, "contact-map") == 1.25
    assert CV.primary_cv_mode({"mode": "contact-map"}) == "contact-map"
    assert CV.swarm_member_cv_args(SimpleNamespace(primary_cv="contact-map", cv1="contact-map")).primary_cv == "nonlocal-contacts"
    plain = SimpleNamespace(primary_cv="distance")
    assert CV.swarm_member_cv_args(plain) is plain


def test_definition_binds_the_model_to_the_topology_and_resume_sha(tmp_path):
    top = _topology()
    path = _write_model(tmp_path, _model(top))
    args = SimpleNamespace(primary_cv="contact-map", cv1_model=str(path))
    d = CV.prepare_primary_cv_definition(top, args, cv_atom1=1, cv_atom2=29)
    assert d["mode"] == "contact-map" and d["cv1_model_sha256"] == _model(top)["sha256"] and d["contact_pairs"] == []
    x = _positions(top)
    assert CV.primary_cv_value_from_positions_nm(x, d, args) == pytest.approx(F.cv_from_positions_nm(_model(top), x))
    with pytest.raises(RuntimeError, match="does not match this topology"):
        CV.prepare_primary_cv_definition(_topology(n_res=9), args, cv_atom1=1, cv_atom2=29)
    # resume: metadata restores path + sha; a different model on that path is refused
    resumed = SimpleNamespace(primary_cv="distance", cv1_model=None)
    CV.apply_primary_cv_metadata_to_args(resumed, {"primary_cv": {k: v for k, v in d.items() if not k.startswith("_")}})
    assert resumed.primary_cv == "contact-map" and resumed.cv1_model == str(path)
    CM1.write_model(path, _model(top, seed=9))
    with pytest.raises(RuntimeError, match="resume must use the model"):
        CV.prepare_primary_cv_definition(top, resumed, cv_atom1=1, cv_atom2=29)
    with pytest.raises(ValueError, match="--cv1-model"):
        CV.prepare_primary_cv_definition(top, SimpleNamespace(primary_cv="contact-map", cv1_model=None))


def test_cli_accepts_contact_map_and_refuses_what_cannot_run(tmp_path):
    from gareus.cli import parse_args
    top = _topology()
    path = _write_model(tmp_path, _model(top))
    base = ["--seq", "GYDPETGTWG", "--out", str(tmp_path / "o")]
    a = parse_args(base + ["--cv1", "contact-map", "--cv1-model", str(path), "--window-mode", "manual"])
    assert a.primary_cv == "contact-map" and a.cv1_model == str(path)
    for bad in (["--cv1", "contact-map", "--window-mode", "adaptive"],
                ["--cv1", "contact-map", "--cv1-model", str(tmp_path / "missing.json"), "--window-mode", "manual"],
                ["--cv1", "contacts", "--cv1-model", str(path)],
                ["--swarm-seed-frame-interval-ps", "5", "--swarm-stage", "run"]):   # 2.5 trace rows at 2 ps
        with pytest.raises(SystemExit):
            parse_args(base + bad)
    cfg = tmp_path / "c.yaml"
    cfg.write_text("cvs:\n  cv1: contactz\n")
    with pytest.raises(SystemExit):
        parse_args(["--config", str(cfg)] + base)                      # YAML typo never runs as distance
    ok = parse_args(base + ["--swarm-output-interval-ps", "2.5", "--swarm-seed-frame-interval-ps", "5"])
    assert ok.swarm_seed_frame_interval_ps == 5.0


# --------------------------------------------------------------------------- the CV2 anchor kind

def test_anchor_kind_contract():
    top = _topology()
    model = _model(top)
    d = AS.cmap_definition(model)
    assert "cv1_model_path" not in d                                  # content identity only
    assert AS.units(AS.KIND_CMAP) == "dimensionless" and AS.value_bounds(AS.KIND_CMAP) == (None, None)
    assert AS.k_bounds(AS.KIND_CMAP, SimpleNamespace()) == (0.5, 200.0)
    assert AS.k_bounds(AS.KIND_CMAP, SimpleNamespace(cv1_contact_map_k_min=1.0, cv1_contact_map_k_max=50.0)) == (1.0, 50.0)
    assert AS.anchor_norm(AS.KIND_CMAP, d, [], None) == 1.0 and AS.anchor_pairs(AS.KIND_CMAP, {}) == []
    assert AS.binding_digest(AS.KIND_CMAP, d) != AS.binding_digest(AS.KIND_CMAP, AS.cmap_definition(_model(top, 3)))
    assert AS.anchor_kind_for_args(SimpleNamespace(primary_cv="contact-map")) == AS.KIND_CMAP
    with pytest.raises(ValueError, match="no contact-map CV1"):
        AS.trace_values(AS.KIND_CMAP, {"cv1": np.zeros(2)})
    with pytest.raises(ValueError, match="composite"):
        AS.build_anchor_subcv(openmm, AS.KIND_CMAP, [], None)


def test_composite_anchor_in_a_cv_force_equals_the_evaluator(tmp_path):
    top = _topology()
    model = _model(top)
    path = _write_model(tmp_path, model)
    d = AS.cmap_definition(model)
    AS._MODEL_CACHE.clear()
    system = openmm.System()
    for _ in range(top.getNumAtoms()):
        system.addParticle(1.0)
    force = openmm.CustomCVForce("0")
    name, defs = AS.add_anchor_to_cv_force(openmm, force, AS.KIND_CMAP, [], SimpleNamespace(cv1_model=str(path)), d)
    force.setEnergyFunction(f"({name})*2 + 1; " + defs)
    system.addForce(force)
    ctx = _context(system)
    x = _positions(top)
    ctx.setPositions(x)
    e = ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    assert e == pytest.approx(2 * F.cv_from_positions_nm(model, x) + 1, rel=1e-10)
    rt = SimpleNamespace(anchor_definition=d)                         # evaluator finds it by sha
    assert AS.value_from_positions(AS.KIND_CMAP, x, [], rt) == pytest.approx(F.cv_from_positions_nm(model, x))
    CM1.write_model(tmp_path / "other.json", _model(top, 4))
    with pytest.raises(RuntimeError, match="model mismatch"):
        AS.check_run_matches(AS.KIND_CMAP, d, [], args=SimpleNamespace(cv1_model=str(tmp_path / "other.json")))
    AS.check_run_matches(AS.KIND_CMAP, d, [], args=SimpleNamespace(cv1_model=str(path)))


def test_fast_path_reads_cv1_from_the_cached_sub_cv():
    """CV1 = sub-CV 0 - cmap_offset: the sample/exchange fast path, no positions copy."""
    from gareus.production import observe_fast_path
    top = _topology()
    model = _model(top)
    system = openmm.System()
    for _ in range(top.getNumAtoms()):
        system.addParticle(1.0)
    force = F.add_contact_map_umbrella_force(openmm, system, model)
    ctx = _context(system)
    x = _positions(top)
    ctx.setPositions(x)
    ctx.getState(getEnergy=True)
    cv, ss = observe_fast_path(ctx, force, None, SimpleNamespace(contact_normalize=True), {"enabled": False})
    assert cv == pytest.approx(F.cv_from_positions_nm(model, x), abs=1e-12) and math.isnan(ss)


def test_residual_scalar_subtracts_the_contact_map_offset():
    from gareus.cv_selection.residual_runtime import CompiledResidualComponent
    comp = CompiledResidualComponent(component_index=1, feature_weights=(1.0, 0.0), k0=0.1, k1=0.5, k2=0.0, sigma_j=2.0,
                                     anchor_mean=0.2, anchor_std=1.5, norm=1.0, transform="identity", clamp_lo=-6.0,
                                     clamp_hi=6.0, degree=1)
    roles = [{"name": "t", "role": "torsion_sum"}, {"name": "res_cms0", "role": "contact_sum", "offset": 0.7}]
    z = comp.evaluate_subcvs({"t": 0.9, "res_cms0": 1.7}, roles)       # c = 1.7 - 0.7 = 1.0
    assert z == pytest.approx((0.9 - 0.1 - 0.5 * (1.0 - 0.2) / 1.5) / 2.0)
    plain = comp.evaluate_subcvs({"t": 0.9, "res_cms0": 1.0}, [roles[0], {"name": "res_cms0", "role": "contact_sum"}])
    assert plain == pytest.approx(z)                                   # contacts: no offset, unchanged


# --------------------------------------------------------------------------- window table + handoff

def _csv(path, mode, k=True):
    cols = ["window", "primary_cv_mode", "primary_cv_center"] + (["primary_cv_k_kcal"] if k else []) + ["gamd_lambda"]
    rows = [[0, mode, -0.5] + ([5.0] if k else []) + [0.0], [1, mode, 0.5] + ([5.0] if k else []) + [0.0]]
    path.write_text("\n".join(",".join(map(str, r)) for r in [cols] + rows) + "\n")
    return path


def test_window_table_kind_must_match_a_contact_map_run(tmp_path):
    from gareus.cli import parse_args
    from gareus.windows import load_explicit_2d_window_csv
    top = _topology()
    path = _write_model(tmp_path, _model(top))
    cmap_args = parse_args(["--seq", "GYDPETGTWG", "--out", str(tmp_path / "o"), "--cv1", "contact-map",
                            "--cv1-model", str(path), "--window-mode", "manual"])
    contact_args = parse_args(["--seq", "GYDPETGTWG", "--out", str(tmp_path / "o2"), "--cv1", "contacts"])
    centers, ks, *_ = load_explicit_2d_window_csv(cmap_args, _csv(tmp_path / "a.csv", "contact-map"))
    assert list(centers) == [-0.5, 0.5] and list(ks) == [5.0, 5.0]
    with pytest.raises(ValueError, match="belongs to another CV1"):
        load_explicit_2d_window_csv(contact_args, _csv(tmp_path / "b.csv", "contact-map"))
    with pytest.raises(ValueError, match="belongs to another CV1"):
        load_explicit_2d_window_csv(cmap_args, _csv(tmp_path / "c.csv", "contacts"))
    with pytest.raises(ValueError, match="force constant"):
        load_explicit_2d_window_csv(cmap_args, _csv(tmp_path / "d.csv", "contact-map", k=False))
    load_explicit_2d_window_csv(contact_args, _csv(tmp_path / "e.csv", "distance"))    # historical: unchecked


def test_epoch0_sidecar_hands_over_the_cv1_and_its_fallback(tmp_path):
    from gareus.swarm import epoch0
    an = epoch0.analysis_dir(tmp_path)
    an.mkdir(parents=True)
    (an / epoch0.SIDECAR_NAME).write_text(json.dumps({"cvs": {"cv1": "contact-map", "cv2": "none"},
                                                      "cv1_model": "/x/cv1_model.json"}))
    args = SimpleNamespace(primary_cv="contact-map", cv1="contact-map", secondary_cv="auto")
    applied = epoch0.apply_epoch0_sidecar(args, tmp_path)
    assert args.primary_cv == "contact-map" and args.cv1_model == "/x/cv1_model.json" and applied["cv1_model"]
    (tmp_path / epoch0.CV1_BINDING_NAME).unlink()                     # a separate campaign below
    (an / epoch0.SIDECAR_NAME).write_text(json.dumps({"cvs": {"cv1": "contacts", "cv2": "none"}}))
    args = SimpleNamespace(primary_cv="contact-map", cv1="contact-map", secondary_cv="auto")
    epoch0.apply_epoch0_sidecar(args, tmp_path)
    assert args.primary_cv == "nonlocal-contacts"                     # spec section 7 fallback
    args = SimpleNamespace(primary_cv="distance", cv1="distance", secondary_cv="auto")
    epoch0.apply_epoch0_sidecar(args, tmp_path)
    assert args.primary_cv == "distance"                              # other campaigns untouched


def test_swarm_falls_back_to_contacts_without_a_recorded_map(tmp_path):
    """A round whose members did not all record the map cannot design a contact-map ladder."""
    from test_contact_map_cv1 import _swarm_on_disk
    from gareus.swarm.analyze import _prepare_contact_map_anchor
    ds, tr, fc, rd, topo, _ = _swarm_on_disk(tmp_path / "s", record=(True, False, True))
    warnings = []
    out = _prepare_contact_map_anchor(tmp_path, rd, tmp_path / "nowhere", [], tr, {}, fc, 0,
                                      SimpleNamespace(), warnings, round_index=0)
    assert out is None and any("falls back to contacts" in w for w in warnings)
    with pytest.raises(RuntimeError, match="round 0"):                # later rounds never refit or guess
        _prepare_contact_map_anchor(tmp_path, rd, tmp_path, [], tr, {}, fc, 0, SimpleNamespace(), [], round_index=1)


def test_seed_conformer_definition_maps_atoms_without_touching_the_model(tmp_path):
    from gareus.seeding import _relative_primary_cv_def_for_conformer
    top = _topology()
    path = _write_model(tmp_path, _model(top))
    d = CV.prepare_primary_cv_definition(top, SimpleNamespace(primary_cv="contact-map", cv1_model=str(path)))
    n = top.getNumAtoms() - 1                                         # peptide atoms (water last)
    rel = _relative_primary_cv_def_for_conformer(d, top, {i: i for i in range(n)})
    assert rel is not None and rel["_cv1_model"]["sha256"] == d["cv1_model_sha256"]
    x = _positions(top)[:n]
    assert CV.primary_cv_value_from_positions_nm(x, rel, None) == pytest.approx(
        F.cv_from_positions_nm(_model(top), _positions(top)))
    assert _relative_primary_cv_def_for_conformer(d, top, {i: i for i in range(4)}) is None   # unmapped atoms


# --------------------------------------------------------------------------- 2026-10-01 review fixes

def test_multi_atom_soft_min_models_are_refused_in_production(tmp_path):
    top = _topology()
    heavy = _model(top, atoms=CMAP.ATOMS_HEAVY)
    with pytest.raises(ValueError, match="multi-atom"):
        F.add_contact_map_umbrella_force(openmm, openmm.System(), heavy)
    path = _write_model(tmp_path, heavy)
    with pytest.raises(ValueError, match="multi-atom"):
        CV.prepare_primary_cv_definition(top, SimpleNamespace(primary_cv="contact-map", cv1_model=str(path)))


def _has_platform(name):
    return any(openmm.Platform.getPlatform(i).getName() == name for i in range(openmm.Platform.getNumPlatforms()))


@pytest.mark.skipif(not _has_platform("OpenCL"), reason="no OpenCL platform")
@pytest.mark.parametrize("precision", ["single", "mixed"])
def test_gpu_forces_match_reference(precision):
    """The first force (exp-sum sub-CVs) lost the CV1 force on OpenCL (fixed-point saturation)."""
    top = _topology()
    model = _model(top)
    x = _positions(top, scale=0.4)
    system = openmm.System()
    for _ in range(top.getNumAtoms()):
        system.addParticle(1.0)
    F.add_contact_map_umbrella_force(openmm, system, model)

    def forces(platform, props=None):
        ctx = openmm.Context(system, openmm.VerletIntegrator(0.001), openmm.Platform.getPlatformByName(platform),
                             props or {})
        ctx.setParameter("k", 200.0)
        ctx.setParameter("r0", 1.0)
        ctx.setPositions(x)
        s = ctx.getState(getEnergy=True, getForces=True)
        return (s.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole),
                np.asarray(s.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer)))

    e_ref, f_ref = forces("Reference")
    e_gpu, f_gpu = forces("OpenCL", {"Precision": precision})
    assert np.abs(f_ref).max() > 1.0
    assert e_gpu == pytest.approx(e_ref, rel=1e-5)
    assert np.abs(f_gpu - f_ref).max() <= 1e-5 * np.abs(f_ref).max() + 1e-4


def test_ca_map_skips_residues_without_a_ca():
    top = _topology()
    cap = top.addResidue("ACE", top.addChain(), id="0")
    top.addAtom("CH3", app.Element.getBySymbol("C"), cap)
    d = CMAP.contact_map_definition(top, atoms=CMAP.ATOMS_CA)
    assert len(d["residues"]) == 8 and all(len(p["atoms_i"]) == len(p["atoms_j"]) == 1 for p in d["pairs"])
    heavy = CMAP.contact_map_definition(_topology())                  # heavy keeps its historical bytes
    assert "atom_selection" not in heavy and heavy["quantity"] == CMAP.QUANTITY


def test_later_jobs_reapply_the_model_whatever_cv2_is(tmp_path):
    from gareus.adaptive_production import _apply_epoch0_cv_decision_on_resume
    from gareus.swarm import epoch0
    top = _topology()
    path = _write_model(tmp_path, _model(top))
    an = epoch0.analysis_dir(tmp_path)
    an.mkdir(parents=True)
    (an / epoch0.SIDECAR_NAME).write_text(json.dumps({"cvs": {"cv1": "contact-map"}, "cv1_model": str(path),
                                                      "cv1_model_sha256": _model(top)["sha256"]}))
    args = SimpleNamespace(primary_cv="contact-map", cv1="contact-map", secondary_cv="none")
    _apply_epoch0_cv_decision_on_resume(args, tmp_path)
    assert args.cv1_model == str(path) and args.cv1_model_sha256_expected == _model(top)["sha256"]
    # a re-analysis that rewrote the model is refused at the force build
    CM1.write_model(path, _model(top, seed=7))
    with pytest.raises(RuntimeError, match="resume must use the model"):
        CV.prepare_primary_cv_definition(top, args)
    plain = SimpleNamespace(primary_cv="nonlocal-contacts", cv1="contacts", secondary_cv="none")
    assert _apply_epoch0_cv_decision_on_resume(plain, tmp_path) == {}


def test_later_swarm_rounds_reuse_round0s_model_or_refuse(tmp_path):
    from gareus.swarm.analyze import _frozen_contact_map_anchor
    from test_contact_map_cv1 import _swarm_on_disk
    ds, tr, fc, rd, topo, _ = _swarm_on_disk(tmp_path / "s")
    top = app.PDBFile(str(topo)).topology
    an = tmp_path / "an"
    an.mkdir()
    with pytest.raises(RuntimeError, match="round 0"):
        _frozen_contact_map_anchor(an, rd, tr, [])
    (an / "ladder_run_args.yaml").write_text(json.dumps({"cvs": {"cv1": "contacts"}}))
    assert _frozen_contact_map_anchor(an, rd, tr, []) is None          # round 0 fell back
    model = _model(top)
    CM1.write_model(an / CM1.MODEL_NAME, model)
    (an / "ladder_run_args.yaml").write_text(json.dumps({"cvs": {"cv1": "contact-map"},
                                                         "cv1_model_sha256": model["sha256"]}))
    out = _frozen_contact_map_anchor(an, rd, tr, [])                    # round 0's model on this round's CA maps
    assert out["model"]["sha256"] == model["sha256"] and out["summary"]["status"] == "frozen"
    X, _d = CMAP.load_member_contact_map(rd / "member_0000", CMAP.ATOMS_CA)
    assert np.allclose(tr[0][AS.CMAP_TRACE_KEY], CM1.evaluate_model(model, X))
    (an / "ladder_run_args.yaml").write_text(json.dumps({"cvs": {"cv1": "contact-map"}, "cv1_model_sha256": "other"}))
    with pytest.raises(RuntimeError, match="differs"):
        _frozen_contact_map_anchor(an, rd, tr, [])


def test_anchor_model_found_next_to_the_pair_model_in_a_fresh_process(tmp_path):
    top = _topology()
    model = _model(top, seed=11)
    CM1.write_model(tmp_path / "cv1_model.json", model)
    AS._MODEL_CACHE.clear()
    rt = SimpleNamespace(anchor_definition=AS.cmap_definition(model), artifact_dir=str(tmp_path))
    x = _positions(top)
    assert AS.value_from_positions(AS.KIND_CMAP, x, [], rt) == pytest.approx(F.cv_from_positions_nm(model, x))
    AS._MODEL_CACHE.clear()
    with pytest.raises(RuntimeError, match="next to the pair model"):
        AS.value_from_positions(AS.KIND_CMAP, x, [], SimpleNamespace(anchor_definition=AS.cmap_definition(model),
                                                                     artifact_dir=str(tmp_path / "none")))


def test_campaign_freezes_the_epoch0_cv1_decision(tmp_path):
    """A re-analysis mid-campaign (new model, or a fallback to contacts) is refused unless overridden."""
    from gareus.swarm import epoch0
    top = _topology()
    an = epoch0.analysis_dir(tmp_path)
    an.mkdir(parents=True)
    a, b = _model(top, seed=1), _model(top, seed=2)

    def sidecar(cvs, model=None):
        CM1.write_model(an / "cv1_model.json", model) if model else None
        side = {"cvs": cvs}
        if model:
            side.update(cv1_model=str(an / "cv1_model.json"), cv1_model_sha256=model["sha256"])
        (an / epoch0.SIDECAR_NAME).write_text(json.dumps(side))

    def job(**kw):
        args = SimpleNamespace(primary_cv="contact-map", cv1="contact-map", secondary_cv="none", cv1_model=None, **kw)
        epoch0.apply_epoch0_sidecar(args, tmp_path)
        return args

    sidecar({"cv1": "contact-map"}, a)
    assert job().cv1_model_sha256_expected == a["sha256"]
    assert json.loads((tmp_path / epoch0.CV1_BINDING_NAME).read_text())["cv1_model_sha256"] == a["sha256"]
    sidecar({"cv1": "contact-map"}, b)                                 # re-analysis wrote model B + its sidecar
    with pytest.raises(RuntimeError, match="re-run mid-campaign"):
        job()
    sidecar({"cv1": "contacts"})                                       # re-analysis fell back to contacts
    with pytest.raises(RuntimeError, match="re-run mid-campaign"):
        job()
    assert job(cv1_binding_override=True).primary_cv == "nonlocal-contacts"   # adopted knowingly
    sidecar({"cv1": "contact-map"}, b)
    user = SimpleNamespace(primary_cv="contact-map", cv1="contact-map", secondary_cv="none",
                           cv1_model="/moved/cv1_model.json", cv1_binding_override=True)
    epoch0.apply_epoch0_sidecar(user, tmp_path)
    assert user.cv1_model == "/moved/cv1_model.json"                   # a user-given path survives the sidecar


def test_contact_map_k_bounds_are_separate_from_the_contact_ones(tmp_path):
    from gareus.cli import parse_args
    base = ["--seq", "GYDPETGTWG", "--out", str(tmp_path / "o"), "--cv1-k-max", "1200"]
    a = parse_args(base)
    assert AS.k_bounds(AS.KIND_CMAP, a) == (0.5, 200.0)                # cv1_k_max stays contact-scaled
    b = parse_args(base + ["--cv1-contact-map-k-max", "50"])
    assert AS.k_bounds(AS.KIND_CMAP, b) == (0.5, 50.0) and AS.k_bounds(AS.KIND_CONTACT, b)[1] == 1200.0


def test_heavy_fit_cannot_be_a_contact_map_campaign_and_frame_check_scope(tmp_path):
    from gareus.cli import parse_args
    base = ["--seq", "GYDPETGTWG", "--out", str(tmp_path / "o")]
    with pytest.raises(SystemExit):
        parse_args(base + ["--cv1", "contact-map", "--window-mode", "manual", "--swarm-cv1-contact-map-atoms", "heavy"])
    parse_args(base + ["--swarm-cv1-contact-map-atoms", "heavy"])     # diagnostic fit on a contacts run: fine
    parse_args(base + ["--swarm-seed-frame-interval-ps", "5"])        # no swarm runs: not checked
    with pytest.raises(SystemExit):
        parse_args(base + ["--swarm-seed-frame-interval-ps", "5", "--swarm-stage", "run"])


def test_resume_keeps_a_user_given_model_path():
    args = SimpleNamespace(primary_cv="contact-map", cv1_model="/moved/cv1_model.json")
    CV.apply_primary_cv_metadata_to_args(args, {"primary_cv": {"mode": "contact-map", "cv1_model_path": "/old/x.json",
                                                               "cv1_model_sha256": "ab"}})
    assert args.cv1_model == "/moved/cv1_model.json" and args.cv1_model_sha256_expected == "ab"


def test_cv2_residual_degree_defaults_to_best_of_both(tmp_path):
    from gareus.cli import parse_args
    from gareus.swarm.analyze import _residual_degrees
    assert _residual_degrees(parse_args(["--seq", "GYDPETGTWG", "--out", str(tmp_path / "o")])) == (1, 2)
    assert _residual_degrees(parse_args(["--seq", "GYDPETGTWG", "--out", str(tmp_path / "o2"),
                                         "--cv-selection-residual-degree", "1"])) == (1,)
    assert _residual_degrees(SimpleNamespace(cv_selection_residual_degree=2)) == (2,)   # YAML int


def test_pooled_pick_takes_the_tie_set_then_the_slowest_across_degrees():
    from gareus.cv_selection.select_pair import SelectionConfig, _pooled_pick
    cfg = SelectionConfig()
    pooled = {(1, 7): {"gain_nats_mean": 0.06, "gain_nats_sd": 0.03, "slowness_rho": 0.95, "gain_nats": 0.06},
              (2, 7): {"gain_nats_mean": 0.05, "gain_nats_sd": 0.03, "slowness_rho": 0.96, "gain_nats": 0.05},
              (2, 9): {"gain_nats_mean": -0.20, "gain_nats_sd": 0.01, "slowness_rho": 0.99, "gain_nats": -0.2}}
    assert _pooled_pick(pooled, cfg, "slowness") == (2, 7)            # (2, 9) is outside the tie-set
    pooled[(2, 7)]["slowness_rho"] = 0.95
    assert _pooled_pick(pooled, cfg, "slowness") == (1, 7)            # exact tie: linear
    assert _pooled_pick(pooled, cfg, "gain") == (1, 7)
    assert _pooled_pick({}, cfg, "slowness") is None
