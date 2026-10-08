"""Spec 17: direct cross-energy check through the production assembly (set_window(aux_state=),
umbrella_bias_matrix_kcal, assemble_bias_matrices), exact Gibbs permutations with ordinary/aux/
duplicate states, global candidates. The exchange kernel is unchanged; these pin it against
auxiliary matrices."""
import itertools
import types

import numpy as np
import pytest

from aux_cv_fixture import dipeptide, model_payload
from gareus.auxiliary_cv.model import AuxModel
from gareus.auxiliary_cv.runtime import add_aux_cv_force, aux_bias_matrix_kcal, observe_aux_z
from gareus.auxiliary_cv.state_table import AuxStateTable
from gareus.production import _gibbs_window_proposal_distribution, assemble_bias_matrices, umbrella_bias_matrix_kcal
from gareus.windows import set_window
from test_exchange_kernel_exact import BETA, _assert_kernel_ok, _gibbs_kernel, _pair_kernel, _pi, _states

ARGS = types.SimpleNamespace(umbrella_force_group=31, secondary_cv_force_group=29)
KJ = 4.184


def _umbrella_forces(system):
    """CV1 = x of atom 0 (group 31, globals r0/k), CV2 = y of atom 1 (group 29, globals ss0/ss_k), nm units."""
    import openmm as mm
    f1 = mm.CustomExternalForce("0.5*k*(x-r0)^2")
    f1.addGlobalParameter("k", 0.0)
    f1.addGlobalParameter("r0", 0.0)
    f1.addParticle(0, [])
    f1.setForceGroup(31)
    system.addForce(f1)
    f2 = mm.CustomExternalForce("0.5*ss_k*(y-ss0)^2")
    f2.addGlobalParameter("ss_k", 0.0)
    f2.addGlobalParameter("ss0", 0.0)
    f2.addParticle(1, [])
    f2.setForceGroup(29)
    system.addForce(f2)


def _carriers(table, n_carriers=3):
    """Distinct real configurations: carrier r is the fixture after 20*r Langevin steps (k = 0 everywhere)."""
    import openmm as mm
    from openmm import unit
    from pep_gamd_fixture import _fresh_system
    d = dipeptide()
    ctxs, rt = [], None
    for r in range(n_carriers):
        system = _fresh_system()
        _umbrella_forces(system)
        rt = add_aux_cv_force(mm, system, table, ARGS)
        integ = mm.LangevinMiddleIntegrator(300 * unit.kelvin, 1 / unit.picosecond, 0.002 * unit.picoseconds)
        integ.setRandomNumberSeed(11 + r)
        ctx = mm.Context(system, integ, mm.Platform.getPlatformByName("Reference"))
        ctx.setPositions(d["positions_nm"])
        ctx.setVelocitiesToTemperature(300 * unit.kelvin, 11 + r)
        if r:
            integ.step(20 * r)
        ctxs.append(ctx)
    return ctxs, rt, unit


def _group_energy_kj(ctx, groups, unit):
    return ctx.getState(getEnergy=True, groups=set(groups)).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)


def test_assembled_matrix_equals_direct_context_energies_and_all_four_swap_terms():
    d = dipeptide()
    blocks = [lab.split("-")[0] for lab in d["labels"]]
    m = AuxModel.from_mapping(model_payload(d["quads"], [1.2] + [0.6] * (2 * len(d["quads"]) - 1),
                                            offset=0.0, blocks=blocks))
    probe_table = AuxStateTable(m, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (None, None, None))
    ctxs, rt, unit = _carriers(probe_table)
    pos = [c.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.nanometer) for c in ctxs]
    z = np.array([observe_aux_z(c, rt, force=c.getSystem().getForce(rt.force_index)) for c in ctxs])
    assert np.ptp(z) > 1e-3, "carriers must differ in z for a meaningful cross-energy check"
    x = np.array([p[0, 0] for p in pos])          # CV1 values (nm)
    y = np.array([p[1, 1] for p in pos])          # CV2 values (nm)
    # Three states: ordinary, auxiliary, and a sham duplicate of the ordinary state.
    c1, k1 = np.array([x[0] + 0.01, x[1] - 0.02, x[0] + 0.01]), np.array([800.0, 600.0, 800.0])   # kcal/mol/nm^2
    c2, k2 = np.array([y[1], y[2] + 0.03, y[1]]), np.array([400.0, 500.0, 400.0])
    table = AuxStateTable(m, (0.0, float(z.mean()) + 0.3, 0.0), (0.0, 2.5, 0.0), (None, None, None))
    # Contexts must carry the SAME table the matrix uses: rebuild the runtime view with it.
    rt_states = type(rt)(table, rt.info, rt.force_index, rt.topology_sha256)
    # Production assembly path.
    d_kcal, s_kcal = umbrella_bias_matrix_kcal(x, y, c1, k1, c2, k2)
    _, matrix_kj = assemble_bias_matrices(d_kcal, s_kcal, np.zeros_like(d_kcal),
                                          aux_bias_kcal=aux_bias_matrix_kcal(z, table))
    # Direct: apply each state's complete target with set_window and read groups {31, 29, aux}.
    groups = (31, 29, rt.info.force_group)
    direct = np.zeros_like(matrix_kj)
    for s in range(3):
        for r, ctx in enumerate(ctxs):
            set_window(ctx, c1, KJ * k1, s, c2, KJ * k2, aux_state=rt_states)
            direct[s, r] = _group_energy_kj(ctx, groups, unit)
    np.testing.assert_allclose(matrix_kj, direct, rtol=1e-9, atol=1e-8)
    np.testing.assert_array_equal(matrix_kj[0], matrix_kj[2])       # sham == parent row
    for (a, b), (i, j) in itertools.product(itertools.permutations(range(3), 2),
                                            itertools.combinations(range(3), 2)):
        d_matrix = matrix_kj[b, i] + matrix_kj[a, j] - matrix_kj[a, i] - matrix_kj[b, j]
        d_direct = direct[b, i] + direct[a, j] - direct[a, i] - direct[b, j]
        assert d_matrix == pytest.approx(d_direct, abs=1e-7)


def _aux_bias(n, with_duplicate):
    """[state, replica] kJ: random CV1-like umbrella part + an auxiliary row; state 3 duplicates state 1."""
    rng = np.random.default_rng(20261007)
    base = rng.uniform(0.0, 5.0, size=(n, n))
    z = rng.normal(0.0, 1.0, size=n)
    k = np.zeros(n)
    c = np.zeros(n)
    k[2], c[2] = 3.0, 0.5                     # state 2 is auxiliary
    table = AuxStateTable(AuxModel.from_mapping(model_payload([(0, 1, 2, 3)], [1.0, 0.0])),
                          tuple(c), tuple(k), (None,) * n)
    bias = base + KJ * aux_bias_matrix_kcal(z, table)
    if with_duplicate:
        bias[3] = bias[1]                     # sham: identical Hamiltonian to its parent, separate slot
    return bias


@pytest.mark.parametrize("n, dup", [(3, False), (4, False), (4, True)])
def test_gibbs_and_pair_kernels_are_detailed_balanced_with_aux_and_sham_states(n, dup):
    states = _states(n)
    bias = _aux_bias(n, dup and n >= 4)
    pi = _pi(states, bias)
    for rep in range(n):
        _assert_kernel_ok(_gibbs_kernel(states, bias, rep), pi, f"gibbs rep={rep} n={n} dup={dup}")
    for wi, wj in itertools.combinations(range(n), 2):
        _assert_kernel_ok(_pair_kernel(states, bias, wi, wj), pi, f"pair {wi},{wj} n={n} dup={dup}")


def test_composed_gibbs_sweep_is_stationary_with_a_duplicate_state():
    n = 4
    states, bias = _states(n), _aux_bias(n, True)
    pi = _pi(states, bias)
    sweep = np.eye(len(states))
    for rep in range(n):
        sweep = sweep @ _gibbs_kernel(states, bias, rep)
    assert np.abs(pi @ sweep - pi).max() < 1e-12


def test_auxiliary_carrier_sees_every_occupied_state_as_a_candidate():
    """No parent-only or neighbour mask: a carrier in the aux state may propose any ordinary state."""
    n = 4
    bias = _aux_bias(n, True)
    replica_of_window = np.arange(n, dtype=np.int64)
    prop = _gibbs_window_proposal_distribution(BETA, bias, replica_index=2, current_window=2,
                                               replica_of_window=replica_of_window)
    assert sorted(prop["windows"].tolist()) == list(range(n))
    assert np.all(prop["probabilities"] > 0.0)


def test_production_gibbs_proposal_offers_every_state_with_an_auxiliary_row():
    """Drive production gibbs_propose_one_replica: no mask, every state is a candidate with p > 0."""
    from gareus.production import gibbs_propose_one_replica
    n = 4
    bias = _aux_bias(n, True)
    seen = []

    def choose(k, probs):
        seen.append((int(k), np.asarray(probs, dtype=float).copy()))
        return 0

    gibbs_propose_one_replica(bias, BETA, np.arange(n), np.arange(n), 2, choose)
    assert len(seen) == 1
    k, probs = seen[0]
    assert k == n
    assert probs.shape == (n,) and np.all(probs > 0.0)
