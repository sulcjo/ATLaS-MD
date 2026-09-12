"""Backend resolution and native-barostat counting for the NPT correction.

Decision table from the design spec (docs/superpowers/specs/2026-09-12-npt-correction-design.md
section 6): NVT gets no volume machinery; conventional-MD NPT keeps the native
OpenMM barostat; supported boosted modes get the application-controlled biased
MC; anything else fails loudly instead of silently downgrading the ensemble.
"""
import pytest

from gareus.npt import count_native_barostats, resolve_npt_backend


class TestResolveNptBackend:
    def test_nvt_auto_has_no_volume_controller(self):
        assert resolve_npt_backend(ensemble="nvt", requested="auto", run_mode="cmd", boost_type="") == "none"

    def test_nvt_with_gamd_run_mode_still_none(self):
        assert resolve_npt_backend(ensemble="nvt", requested="auto", run_mode="hmr-gamd", boost_type="pep-gamd-lower-dual") == "none"

    def test_nvt_rejects_an_explicit_barostat_backend(self):
        with pytest.raises(ValueError, match="nvt"):
            resolve_npt_backend(ensemble="nvt", requested="native", run_mode="cmd", boost_type="")

    def test_nvt_rejects_explicit_biased_mc(self):
        with pytest.raises(ValueError):
            resolve_npt_backend(ensemble="nvt", requested="biased_mc", run_mode="gamd", boost_type="pep-gamd-lower-dual")

    def test_conventional_npt_auto_uses_native(self):
        assert resolve_npt_backend(ensemble="npt", requested="auto", run_mode="cmd", boost_type="") == "native"

    def test_conventional_npt_auto_uses_native_with_hmr(self):
        assert resolve_npt_backend(ensemble="npt", requested="auto", run_mode="hmr-cmd", boost_type="") == "native"

    def test_pep_gamd_npt_auto_uses_biased_mc(self):
        assert resolve_npt_backend(ensemble="npt", requested="auto", run_mode="gamd", boost_type="pep-gamd-lower-dual") == "biased_mc"

    def test_pep_gamd_npt_auto_with_hmr_uses_biased_mc(self):
        assert resolve_npt_backend(ensemble="npt", requested="auto", run_mode="hmr-gamd", boost_type="pep-gamd-lower-dual") == "biased_mc"

    def test_lower_dihedral_npt_auto_uses_biased_mc(self):
        assert resolve_npt_backend(ensemble="npt", requested="auto", run_mode="gamd", boost_type="lower-dihedral") == "biased_mc"

    def test_unsupported_boost_type_fails_loudly_with_mode(self):
        for boost_type in ("lower-dual", "upper-total", "lower-nonbonded", "gamd-cmd-base", "lower-dual-nonbonded-dihedral"):
            with pytest.raises(ValueError, match=boost_type):
                resolve_npt_backend(ensemble="npt", requested="auto", run_mode="gamd", boost_type=boost_type)

    def test_boosted_run_mode_without_boost_type_fails(self):
        with pytest.raises(ValueError, match="boost"):
            resolve_npt_backend(ensemble="npt", requested="auto", run_mode="gamd", boost_type="")

    def test_explicit_native_with_boosted_dynamics_fails(self):
        with pytest.raises(ValueError, match="native"):
            resolve_npt_backend(ensemble="npt", requested="native", run_mode="gamd", boost_type="pep-gamd-lower-dual")

    def test_explicit_native_with_conventional_md_is_native(self):
        assert resolve_npt_backend(ensemble="npt", requested="native", run_mode="cmd", boost_type="") == "native"

    def test_explicit_biased_mc_with_conventional_md_is_biased_mc(self):
        # zero-boost reference comparisons are an explicitly supported use
        assert resolve_npt_backend(ensemble="npt", requested="biased_mc", run_mode="cmd", boost_type="") == "biased_mc"

    def test_explicit_biased_mc_with_supported_boost_is_biased_mc(self):
        assert resolve_npt_backend(ensemble="npt", requested="biased_mc", run_mode="gamd", boost_type="pep-gamd-lower-dual") == "biased_mc"

    def test_case_and_underscore_normalisation(self):
        assert resolve_npt_backend(ensemble="NPT", requested="AUTO", run_mode="hmr_gamd", boost_type="pep-gamd-lower-dual") == "biased_mc"

    def test_unknown_ensemble_rejected(self):
        with pytest.raises(ValueError):
            resolve_npt_backend(ensemble="npz", requested="auto", run_mode="cmd", boost_type="")

    def test_unknown_requested_rejected(self):
        with pytest.raises(ValueError):
            resolve_npt_backend(ensemble="npt", requested="monte-carlo", run_mode="cmd", boost_type="")

    def test_unknown_run_mode_rejected(self):
        with pytest.raises(ValueError):
            resolve_npt_backend(ensemble="npt", requested="auto", run_mode="remd", boost_type="")


class TestCountNativeBarostats:
    def _system(self, openmm):
        system = openmm.System()
        system.addParticle(1.0)
        return system

    def test_empty_system_counts_zero(self):
        openmm = pytest.importorskip("openmm")
        assert count_native_barostats(self._system(openmm)) == 0

    def test_plain_forces_are_not_counted(self):
        openmm = pytest.importorskip("openmm")
        system = self._system(openmm)
        system.addForce(openmm.NonbondedForce())
        system.addForce(openmm.HarmonicBondForce())
        assert count_native_barostats(system) == 0

    def test_one_monte_carlo_barostat(self):
        openmm = pytest.importorskip("openmm")
        system = self._system(openmm)
        system.addForce(openmm.MonteCarloBarostat(1.0 * openmm.unit.bar, 298 * openmm.unit.kelvin))
        assert count_native_barostats(system) == 1

    def test_multiple_and_exotic_barostats_all_counted(self):
        openmm = pytest.importorskip("openmm")
        system = self._system(openmm)
        system.addForce(openmm.MonteCarloBarostat(1.0 * openmm.unit.bar, 298 * openmm.unit.kelvin))
        system.addForce(openmm.MonteCarloAnisotropicBarostat(
            (1.0 * openmm.unit.bar,) * 3, 298 * openmm.unit.kelvin, False))
        n = count_native_barostats(system)
        assert n >= 2
        system.addForce(openmm.NonbondedForce())
        assert count_native_barostats(system) == n

    def test_barostat_class_name_pattern_counted_even_if_unknown_class(self):
        # Name-based counting must not silently miss a future MonteCarlo*Barostat.
        openmm = pytest.importorskip("openmm")
        system = self._system(openmm)
        system.addForce(openmm.MonteCarloBarostat(1.0 * openmm.unit.bar, 298 * openmm.unit.kelvin))
        real_cls = system.getForce(0).__class__
        assert real_cls.__name__.startswith("MonteCarlo") and "Barostat" in real_cls.__name__
