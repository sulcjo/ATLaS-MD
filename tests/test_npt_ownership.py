"""Barostat ownership resolution and preflight (NPT package B).

``resolve_barostat_ownership`` is the single decision point that must run
BEFORE any Context is created from a System.  The actual backend dispatch
lives in the frozen ``gareus.npt.resolve_npt_backend`` contract (package 1),
so these tests monkeypatch it with a fake implementing the documented
semantics -- exactly like production will consume it.

``preflight_barostat_ownership`` runs on the fully-assembled System before
Context creation and asserts exactly one volume controller, no unsupported
barostat family, and the physical/molecule prerequisites for the biased-MC
backend.
"""

from __future__ import annotations

import types

import pytest

import gareus.npt as npt
import gareus.system_setup as system_setup


def _args(**over):
    base = dict(
        npt_barostat_backend="auto",
        production_ensemble="npt",
        run_mode="gamd",
        gamd_boost_type="pep-gamd-lower-dual",
        pressure_bar=1.0,
        temperature_k=300.0,
        barostat_frequency=100,
        production_barostat_frequency=0,
        barostat_volume_step_fraction=0.01,
        seed=2026,
    )
    base.update(over)
    return types.SimpleNamespace(**base)


def _fake_resolver(mapping=None, default="biased_mc"):
    """Fake honouring the frozen resolve_npt_backend contract semantics."""
    def resolve(*, ensemble, requested, run_mode, boost_type):
        if requested == "native" and run_mode in {"gamd", "hmr-gamd"}:
            raise ValueError(
                "explicit native barostat with boosted dynamics must fail "
                "rather than preserve the known mismatch"
            )
        if mapping is not None:
            return mapping.get((run_mode, boost_type), default)
        return default

    return resolve


# ----------------------------------------------------------- resolution


def test_nvt_never_calls_the_resolver_and_has_no_controller(monkeypatch):
    calls = []

    def resolve(**kw):
        calls.append(kw)
        raise AssertionError("resolver must not be consulted for NVT")

    monkeypatch.setattr(npt, "resolve_npt_backend", resolve)
    ownership = system_setup.resolve_barostat_ownership(
        _args(production_ensemble="nvt"), ensemble="nvt"
    )
    assert ownership.backend == "none"
    assert ownership.include_native_barostat is False
    assert not calls


def test_nvt_with_explicit_backend_is_rejected(monkeypatch):
    monkeypatch.setattr(npt, "resolve_npt_backend", _fake_resolver())
    with pytest.raises(ValueError, match="nvt"):
        system_setup.resolve_barostat_ownership(
            _args(production_ensemble="nvt", npt_barostat_backend="native"),
            ensemble="nvt",
        )


def test_boosted_npt_resolves_to_biased_mc_and_drops_the_native_barostat(monkeypatch):
    monkeypatch.setattr(npt, "resolve_npt_backend", _fake_resolver(default="biased_mc"))
    ownership = system_setup.resolve_barostat_ownership(_args(), ensemble="npt")
    assert ownership.backend == "biased_mc"
    assert ownership.include_native_barostat is False
    assert ownership.barostat_frequency == 100
    assert ownership.volume_step_fraction == 0.01


def test_conventional_npt_resolves_to_native_and_keeps_the_barostat(monkeypatch):
    monkeypatch.setattr(npt, "resolve_npt_backend", _fake_resolver(default="native"))
    ownership = system_setup.resolve_barostat_ownership(
        _args(run_mode="cmd"), ensemble="npt", run_mode="cmd"
    )
    assert ownership.backend == "native"
    assert ownership.include_native_barostat is True


def test_explicit_native_with_boosted_dynamics_fails(monkeypatch):
    monkeypatch.setattr(npt, "resolve_npt_backend", _fake_resolver())
    with pytest.raises(ValueError, match="native"):
        system_setup.resolve_barostat_ownership(
            _args(npt_barostat_backend="native"), ensemble="npt"
        )


def test_unsupported_boosted_mode_fails_rather_than_downgrading_to_nvt(monkeypatch):
    def resolve(**kw):
        raise ValueError(
            "no validated target adapter for boost type 'upper-total' under NPT"
        )

    monkeypatch.setattr(npt, "resolve_npt_backend", resolve)
    with pytest.raises(ValueError, match="upper-total"):
        system_setup.resolve_barostat_ownership(
            _args(gamd_boost_type="upper-total"), ensemble="npt"
        )


def test_production_frequency_override_semantics_preserved(monkeypatch):
    monkeypatch.setattr(npt, "resolve_npt_backend", _fake_resolver())
    ownership = system_setup.resolve_barostat_ownership(
        _args(production_barostat_frequency=37), ensemble="npt"
    )
    assert ownership.barostat_frequency == 37
    ownership = system_setup.resolve_barostat_ownership(
        _args(barostat_frequency=55), ensemble="npt"
    )
    assert ownership.barostat_frequency == 55


def test_unknown_backend_choice_is_rejected_before_the_resolver(monkeypatch):
    monkeypatch.setattr(npt, "resolve_npt_backend", _fake_resolver())
    with pytest.raises(ValueError, match="auto, native or biased_mc"):
        system_setup.resolve_barostat_ownership(
            _args(npt_barostat_backend="fast"), ensemble="npt"
        )


# ------------------------------------------------------------- preflight


def _tiny_system(monkeypatch, *, with_barostat=False):
    import openmm
    from openmm import unit

    system = openmm.System()
    system.addParticle(12.0 * unit.dalton)
    nb = openmm.NonbondedForce()
    nb.addParticle(0.0, 0.2, 0.0)
    nb.setNonbondedMethod(openmm.NonbondedForce.CutoffPeriodic)
    nb.setCutoffDistance(0.9 * unit.nanometer)
    system.addForce(nb)
    system.setDefaultPeriodicBoxVectors(
        openmm.Vec3(2.5, 0, 0), openmm.Vec3(0, 2.5, 0), openmm.Vec3(0, 0, 2.5)
    )
    if with_barostat:
        system.addForce(openmm.MonteCarloBarostat(1 * unit.bar, 300 * unit.kelvin, 100))
    return system


def _ownership(backend, run_mode="cmd"):
    return system_setup.BarostatOwnership(
        backend,
        requested="auto",
        ensemble="npt",
        run_mode=run_mode,
        boost_type="pep-gamd-lower-dual",
        barostat_frequency=100,
        pressure_bar=1.0,
        temperature_k=300.0,
        volume_step_fraction=0.01,
    )


def test_native_preflight_requires_exactly_one_barostat(monkeypatch):
    monkeypatch.setattr(npt, "count_native_barostats", lambda system: 1)
    report = system_setup.preflight_barostat_ownership(
        _tiny_system(monkeypatch, with_barostat=True), _ownership("native")
    )
    assert report["native_barostats_in_system"] == 1

    monkeypatch.setattr(npt, "count_native_barostats", lambda system: 0)
    with pytest.raises(RuntimeError, match="exactly one"):
        system_setup.preflight_barostat_ownership(
            _tiny_system(monkeypatch), _ownership("native")
        )


def test_biased_mc_preflight_requires_zero_native_barostats(monkeypatch):
    monkeypatch.setattr(npt, "count_native_barostats", lambda system: 0)
    report = system_setup.preflight_barostat_ownership(
        _tiny_system(monkeypatch), _ownership("biased_mc", run_mode="gamd")
    )
    assert report["native_barostats_in_system"] == 0

    monkeypatch.setattr(npt, "count_native_barostats", lambda system: 1)
    with pytest.raises(RuntimeError, match="removed from application-controlled Systems"):
        system_setup.preflight_barostat_ownership(
            _tiny_system(monkeypatch), _ownership("biased_mc", run_mode="gamd")
        )


def test_native_with_boosted_dynamics_is_refused_by_the_preflight_too(monkeypatch):
    monkeypatch.setattr(npt, "count_native_barostats", lambda system: 1)
    with pytest.raises(RuntimeError, match="boosted dynamics"):
        system_setup.preflight_barostat_ownership(
            _tiny_system(monkeypatch, with_barostat=True),
            _ownership("native", run_mode="gamd"),
        )


def test_unsupported_barostat_families_are_rejected(monkeypatch):
    monkeypatch.setattr(npt, "count_native_barostats", lambda system: 1)
    import openmm
    from openmm import unit

    unsupported = [
        openmm.MonteCarloAnisotropicBarostat(
            openmm.Vec3(1, 1, 1), 300 * unit.kelvin, True, True, True, 100
        ),
        openmm.MonteCarloMembraneBarostat(
            1 * unit.bar, 0 * unit.bar * unit.nanometer, 300 * unit.kelvin, 100,
            openmm.MonteCarloMembraneBarostat.XYIsotropic,
        ),
        openmm.MonteCarloFlexibleBarostat(300 * unit.kelvin, 1 * unit.bar, 100),
    ]
    for extra in unsupported:
        system = _tiny_system(monkeypatch, with_barostat=True)
        system.addForce(extra)
        with pytest.raises(ValueError, match="not supported"):
            system_setup.preflight_barostat_ownership(system, _ownership("native"))


def test_biased_mc_rejects_immobile_non_virtual_particles(monkeypatch):
    monkeypatch.setattr(npt, "count_native_barostats", lambda system: 0)
    system = _tiny_system(monkeypatch)
    system.addParticle(0.0)  # massless, not a virtual site
    with pytest.raises(RuntimeError, match="immobile"):
        system_setup.preflight_barostat_ownership(
            system, _ownership("biased_mc", run_mode="gamd")
        )


def test_biased_mc_accepts_massless_virtual_sites(monkeypatch):
    monkeypatch.setattr(npt, "count_native_barostats", lambda system: 0)
    import openmm

    system = _tiny_system(monkeypatch)
    system.addParticle(0.0)
    system.setVirtualSite(1, openmm.TwoParticleAverageSite(0, 0, 0.5, 0.5))
    report = system_setup.preflight_barostat_ownership(
        system, _ownership("biased_mc", run_mode="gamd")
    )
    assert report["backend"] == "biased_mc"
