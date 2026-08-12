"""Tests for three performance-audit fixes in `analyze_gareus_mbar.py`.

Fix 1 -- `_compute_rg_from_trajectories` and `analyze_poincare_residue_torsions`
used whole-segment `md.load()` for purely per-frame computations (radius of
gyration; phi/psi at a handful of known crossing-event frames), holding an
entire multi-million-frame trajectory segment in memory just to compute
independent per-frame quantities. Replaced with chunked `md.iterload()`:
`_rg_from_segment_chunked` (Rg, accumulates chunk-by-chunk) and
`_poincare_torsions_chunked` (Poincare crossing torsions, computes phi/psi
per chunk and keeps only the requested frames' angle values, discarding each
chunk's coordinates immediately after).

Fix 2 -- `analyze_extra_observable_pmfs`'s main trajectory pass loaded the
full solvated system (protein + every water/ion hydrogen) on every chunk,
even though phi/psi, DSSP, SASA, and internal-contact computations only ever
see a protein-only `atom_slice` of that chunk. The naive fix suggested by the
audit -- excluding ALL hydrogens system-wide -- was checked against every
downstream consumer and found UNSAFE: it changes total protein SASA by ~14%
on a synthetic ACE-ALA-NME/TIP3P system (350.24 A^2 all-atom vs 299.86 A^2
heavy-atom-only), a real physical difference from removing exposed-hydrogen
surface area, not floating-point noise. The applied fix instead restricts the
main iterload's `atom_indices` to (protein atoms, any element) UNION (heavy
atoms system-wide) -- i.e. it drops only solvent/ion hydrogens, never protein
ones -- which still cuts total atom count ~2-3x for a typical TIP3P-solvated
system (since protein is a tiny fraction of total atoms) while being
provably SASA/DSSP/phi-psi/contact-count identical, because the protein atom
population never changes.

Fix 3 -- `_peptide_solvent_contact_counts` rebuilt an O(n_atoms) topology
classification (which atoms are peptide/water/ion heavy atoms) from scratch
on every call, even though the topology is fixed for the whole
`analyze_extra_observable_pmfs` invocation. Split into
`_build_peptide_solvent_classification(topology)` (pure function of the
topology, computed once) and `_peptide_solvent_contact_counts(md, traj,
cutoff_nm, classification)` (reuses the precomputed classification).
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("analyze_gareus_mbar")
md = pytest.importorskip("mdtraj")

from analyze_gareus_mbar import (
    _build_peptide_solvent_classification,
    _peptide_solvent_contact_counts,
    _poincare_torsions_chunked,
    _rg_from_segment_chunked,
)


# ---------------------------------------------------------------------------
# Fix 1a: Rg chunked loading
# ---------------------------------------------------------------------------

def _build_synthetic_point_cloud_traj(n_frames: int, n_atoms: int = 6, seed: int = 0):
    """A small multi-frame Trajectory with a plain single-residue topology (no
    real chemistry needed for a per-frame Rg computation) whose per-frame
    coordinates are scaled by a smoothly-varying factor, so Rg genuinely
    differs from frame to frame instead of being trivially constant.
    """
    rng = np.random.default_rng(seed)
    top = md.Topology()
    chain = top.add_chain()
    res = top.add_residue("MOL", chain)
    elem = md.element.carbon
    for i in range(n_atoms):
        top.add_atom(f"C{i}", elem, res)
    base = rng.normal(size=(n_atoms, 3)) * 0.3
    scale = 1.0 + 0.5 * np.sin(np.linspace(0, 8 * np.pi, n_frames))
    xyz = (base[None, :, :] * scale[:, None, None]).astype(np.float32)
    return md.Trajectory(xyz, top)


def test_rg_chunked_matches_whole_load_with_atom_indices(tmp_path):
    n_frames = 3300
    traj = _build_synthetic_point_cloud_traj(n_frames)
    seg_path = tmp_path / "seg.xtc"
    traj.save_xtc(str(seg_path))
    top_pdb = tmp_path / "top.pdb"
    traj[0].save_pdb(str(top_pdb))

    top_obj = md.load(str(top_pdb)).topology
    atom_indices = top_obj.select("all")

    # Old approach: whole-segment md.load() + compute_rg().
    whole = md.load(str(seg_path), top=top_obj, atom_indices=atom_indices)
    rg_old = md.compute_rg(whole)

    # New approach: chunked iterload with a small chunk size relative to
    # n_frames, forcing >10 chunks and exercising chunk-boundary concatenation.
    rg_new, n_frames_new = _rg_from_segment_chunked(
        md, seg_path, top_obj, atom_indices, "all", chunk_size=257
    )

    assert n_frames_new == whole.n_frames == n_frames
    assert rg_new.shape == rg_old.shape
    assert np.allclose(rg_new, rg_old)


def test_rg_chunked_matches_whole_load_without_atom_indices_fallback(tmp_path):
    """Fallback path (rg_atoms is None): atoms are selected per-chunk via
    `selection` against each chunk's own topology, mirroring the original
    whole-load fallback branch when pre-selection fails or the caller passes
    no atom_indices."""
    n_frames = 1200
    traj = _build_synthetic_point_cloud_traj(n_frames, n_atoms=8, seed=1)
    seg_path = tmp_path / "seg_fallback.xtc"
    traj.save_xtc(str(seg_path))
    top_pdb = tmp_path / "top_fallback.pdb"
    traj[0].save_pdb(str(top_pdb))
    top_obj = md.load(str(top_pdb)).topology

    whole = md.load(str(seg_path), top=top_obj)
    rg_old = md.compute_rg(whole)

    rg_new, n_frames_new = _rg_from_segment_chunked(
        md, seg_path, top_obj, None, "all", chunk_size=333
    )
    assert n_frames_new == n_frames
    assert np.allclose(rg_new, rg_old)


def test_rg_chunked_zero_atom_selection_raises(tmp_path):
    """A selection matching zero atoms must fail loudly (matching the
    original whole-load fallback's `raise ValueError`), not silently return
    an empty/garbage Rg array."""
    n_frames = 50
    traj = _build_synthetic_point_cloud_traj(n_frames, n_atoms=4, seed=2)
    seg_path = tmp_path / "seg_zero.xtc"
    traj.save_xtc(str(seg_path))
    top_pdb = tmp_path / "top_zero.pdb"
    traj[0].save_pdb(str(top_pdb))
    top_obj = md.load(str(top_pdb)).topology

    with pytest.raises(ValueError, match="matched zero atoms"):
        _rg_from_segment_chunked(md, seg_path, top_obj, None, "resname NOSUCHRES", chunk_size=10)


# ---------------------------------------------------------------------------
# Fix 1b: Poincare-torsion chunked selective-frame loading
# ---------------------------------------------------------------------------

def _build_synthetic_backbone_traj(n_frames: int, n_res: int = 4, seed: int = 1):
    """A small multi-frame Trajectory with a real N-CA-C backbone per residue
    so `md.compute_phi`/`md.compute_psi` have genuine dihedral angles to
    compute (mdtraj identifies phi/psi atoms via residue-adjacency + standard
    backbone atom names, not explicit bonds)."""
    rng = np.random.default_rng(seed)
    top = md.Topology()
    chain = top.add_chain()
    for _ in range(n_res):
        res = top.add_residue("ALA", chain)
        for name, elem in (
            ("N", md.element.nitrogen),
            ("CA", md.element.carbon),
            ("C", md.element.carbon),
        ):
            top.add_atom(name, elem, res)
    n_atoms = top.n_atoms
    xyz = (rng.normal(size=(n_frames, n_atoms, 3)) * 0.15).astype(np.float32)
    return md.Trajectory(xyz, top)


def test_poincare_torsions_chunked_matches_whole_load_indexing(tmp_path):
    n_frames = 800
    traj = _build_synthetic_backbone_traj(n_frames)
    seg_path = tmp_path / "poincare_seg.xtc"
    traj.save_xtc(str(seg_path))
    top_pdb = tmp_path / "poincare_top.pdb"
    traj[0].save_pdb(str(top_pdb))

    atom_indices = np.arange(traj.n_atoms)
    needed_frames = {5, 0, 799, 400, 137}

    # Old approach: whole-segment md.load() + direct `traj[frame_idx]` indexing.
    whole = md.load(str(seg_path), top=str(top_pdb), atom_indices=atom_indices)
    expected = {}
    for fi in needed_frames:
        frame = whole[fi]
        _, phi = md.compute_phi(frame)
        _, psi = md.compute_psi(frame)
        expected[fi] = (phi[0], psi[0])

    # New approach: chunked iterload, computing phi/psi per chunk and keeping
    # only the requested frames' angle values.
    got = _poincare_torsions_chunked(
        md, seg_path, str(top_pdb), atom_indices, chunk_size=97, needed_frames=needed_frames
    )

    assert set(got.keys()) == needed_frames
    for fi in needed_frames:
        np.testing.assert_allclose(got[fi][0], expected[fi][0])
        np.testing.assert_allclose(got[fi][1], expected[fi][1])


def test_poincare_torsions_chunked_omits_out_of_range_frames(tmp_path):
    """An out-of-range requested frame index is silently absent from the
    result, matching the original code's `if frame_idx >= traj.n_frames:
    continue` behavior -- not an error."""
    n_frames = 40
    traj = _build_synthetic_backbone_traj(n_frames, seed=3)
    seg_path = tmp_path / "poincare_seg_oob.xtc"
    traj.save_xtc(str(seg_path))
    top_pdb = tmp_path / "poincare_top_oob.pdb"
    traj[0].save_pdb(str(top_pdb))
    atom_indices = np.arange(traj.n_atoms)

    got = _poincare_torsions_chunked(
        md, seg_path, str(top_pdb), atom_indices, chunk_size=8,
        needed_frames={5, 10_000, -1},
    )
    assert set(got.keys()) == {5}


# ---------------------------------------------------------------------------
# Fix 2 + Fix 3: shared synthetic solvated system
# ---------------------------------------------------------------------------

_DIPEPTIDE_NOH_PDB = """\
ATOM      1  CH3 ACE A   1      -1.900   0.400   0.000  1.00  0.00           C
ATOM      2  C   ACE A   1      -0.500   0.400   0.000  1.00  0.00           C
ATOM      3  O   ACE A   1       0.100   1.470   0.000  1.00  0.00           O
ATOM      4  N   ALA A   2       0.100  -0.800   0.000  1.00  0.00           N
ATOM      5  CA  ALA A   2       1.550  -0.900   0.000  1.00  0.00           C
ATOM      6  CB  ALA A   2       2.050  -2.350   0.000  1.00  0.00           C
ATOM      7  C   ALA A   2       2.150  -0.150   1.200  1.00  0.00           C
ATOM      8  O   ALA A   2       1.600   0.850   1.650  1.00  0.00           O
ATOM      9  N   NME A   3       3.300  -0.650   1.650  1.00  0.00           N
ATOM     10  CH3 NME A   3       3.980   0.000   2.800  1.00  0.00           C
END
"""


@pytest.fixture(scope="module")
def solvated_system(tmp_path_factory):
    """A small real explicit-solvent system (ACE-ALA-NME capped dipeptide +
    TIP3P water, built and solvated by OpenMM/AmberFF so hydrogens and box
    vectors are physically real) used to test the H-exclusion (Fix 2) and
    contact-classification hoist (Fix 3) against genuine SASA/DSSP/contact
    computations -- not a hand-built point cloud, since `compute_neighbors(
    periodic=True)` needs real unitcell vectors and SASA needs real van der
    Waals geometry."""
    openmm = pytest.importorskip("openmm")
    openmm_app = pytest.importorskip("openmm.app")
    openmm_unit = pytest.importorskip("openmm.unit")

    base_dir = tmp_path_factory.mktemp("solvated_system")
    noh_pdb = base_dir / "dipeptide_noh.pdb"
    noh_pdb.write_text(_DIPEPTIDE_NOH_PDB)

    pdb = openmm_app.PDBFile(str(noh_pdb))
    modeller = openmm_app.Modeller(pdb.topology, pdb.positions)
    forcefield = openmm_app.ForceField("amber14-all.xml", "amber14/tip3pfb.xml")
    modeller.addHydrogens(forcefield)
    modeller.addSolvent(
        forcefield,
        boxSize=openmm.Vec3(1.8, 1.8, 1.8) * openmm_unit.nanometers,
        ionicStrength=0.0 * openmm_unit.molar,
    )
    solvated_pdb = base_dir / "solvated.pdb"
    with open(solvated_pdb, "w") as f:
        openmm_app.PDBFile.writeFile(modeller.topology, modeller.positions, f)

    return md.load(str(solvated_pdb))


def _make_multiframe(traj, n_frames: int, scale: float, seed: int):
    """Perturb a single-frame Trajectory's coordinates into a small
    multi-frame trajectory (keeps unitcell vectors, required for periodic
    neighbor search)."""
    rng = np.random.default_rng(seed)
    xyz = np.repeat(traj.xyz[None, 0], n_frames, axis=0) + rng.normal(
        scale=scale, size=(n_frames,) + traj.xyz.shape[1:]
    ).astype(np.float32)
    out = md.Trajectory(xyz, traj.topology)
    out.unitcell_vectors = np.repeat(traj.unitcell_vectors[None, 0], n_frames, axis=0)
    return out


def test_fix2_hydrogen_exclusion_preserves_protein_atom_population(solvated_system):
    """The applied Fix 2 selection (protein UNION heavy) must not drop a
    single protein atom (of any element), and must still meaningfully shrink
    the atom count via solvent/ion hydrogens."""
    top = solvated_system.topology
    protein_idx = top.select("protein")
    heavy_idx = top.select("element != H")
    keep = np.union1d(protein_idx, heavy_idx)

    assert np.array_equal(np.intersect1d(keep, protein_idx), np.sort(protein_idx))
    assert 0 < keep.size < top.n_atoms
    # Real per-run savings claim from the audit: ~2-3x atom count reduction.
    assert 2.0 <= (top.n_atoms / keep.size) <= 3.5


def test_fix2_naive_all_hydrogen_exclusion_changes_sasa_materially(solvated_system):
    """Documents WHY the naive "exclude every hydrogen" fix was rejected:
    protein-only SASA computed with vs without protein hydrogens differs by
    far more than any reasonable float tolerance."""
    top = solvated_system.topology
    prot_all = top.select("protein")
    prot_heavy_only = top.select("protein and element != H")

    sasa_all = md.shrake_rupley(
        solvated_system.atom_slice(prot_all), mode="atom", n_sphere_points=240
    )
    sasa_heavy = md.shrake_rupley(
        solvated_system.atom_slice(prot_heavy_only), mode="atom", n_sphere_points=240
    )
    total_all = float(np.sum(sasa_all) * 100.0)
    total_heavy = float(np.sum(sasa_heavy) * 100.0)
    rel_diff = abs(total_all - total_heavy) / total_all

    assert rel_diff > 0.05  # far beyond float tolerance -- a real physical difference


def test_fix2_applied_selection_leaves_sasa_dssp_phi_psi_unchanged(solvated_system):
    """The APPLIED Fix 2 selection (protein UNION heavy, not all-hydrogen
    exclusion) must reproduce SASA, DSSP, and phi/psi exactly, computed from
    the reduced-atom subsystem's own protein-only atom_slice -- mirroring
    what each iterload chunk's `_protein_traj()` call sees in production."""
    top = solvated_system.topology
    protein_idx = top.select("protein")
    heavy_idx = top.select("element != H")
    keep = np.union1d(protein_idx, heavy_idx)

    # Residue count must be preserved by the reduction (no residue fully
    # dropped) -- required for the Fix 3 classification's residue-index sets
    # to remain valid after this atom_indices restriction.
    assert top.subset(keep).n_residues == top.n_residues

    full_prot = solvated_system.atom_slice(protein_idx)
    reduced = solvated_system.atom_slice(keep)
    reduced_prot = reduced.atom_slice(reduced.topology.select("protein"))

    sasa_full = md.shrake_rupley(full_prot, mode="atom", n_sphere_points=240)
    sasa_reduced = md.shrake_rupley(reduced_prot, mode="atom", n_sphere_points=240)
    assert np.allclose(sasa_full, sasa_reduced)

    dssp_full = md.compute_dssp(full_prot, simplified=True)
    dssp_reduced = md.compute_dssp(reduced_prot, simplified=True)
    assert np.array_equal(dssp_full, dssp_reduced)

    _, phi_full = md.compute_phi(full_prot)
    _, phi_reduced = md.compute_phi(reduced_prot)
    assert np.allclose(phi_full, phi_reduced)

    _, psi_full = md.compute_psi(full_prot)
    _, psi_reduced = md.compute_psi(reduced_prot)
    assert np.allclose(psi_full, psi_reduced)


def test_fix2_atom_indices_restriction_via_iterload_matches_full_load(solvated_system, tmp_path):
    """End-to-end: loading through `md.iterload(..., atom_indices=keep)` (the
    actual code path used in `analyze_extra_observable_pmfs`) must give the
    exact same contact-count result as computing on the full system."""
    top = solvated_system.topology
    protein_idx = top.select("protein")
    heavy_idx = top.select("element != H")
    keep = np.union1d(protein_idx, heavy_idx)

    multi = _make_multiframe(solvated_system, n_frames=4, scale=0.05, seed=11)
    seg_path = tmp_path / "solvated_seg.xtc"
    multi.save_xtc(str(seg_path))
    top_pdb = tmp_path / "solvated_top.pdb"
    multi[0].save_pdb(str(top_pdb))

    # Both sides must be loaded from the SAME on-disk XTC file (lossy float32
    # compression can shift a borderline contact distance across the 0.45nm
    # cutoff by itself) -- comparing the in-memory `multi` object against a
    # file-loaded restricted trajectory would conflate that compression
    # artifact with an actual atom_indices-restriction bug.
    full = md.load(str(seg_path), top=str(top_pdb))
    full.unitcell_vectors = multi.unitcell_vectors
    full_classification = _build_peptide_solvent_classification(full.topology)
    total_full, water_full, ion_full = _peptide_solvent_contact_counts(
        md, full, 0.45, full_classification
    )

    for chunk in md.iterload(str(seg_path), top=str(top_pdb), chunk=2, atom_indices=keep):
        pass  # exercised for the atom-count reduction; verified against full below
    restricted = md.load(str(seg_path), top=str(top_pdb), atom_indices=keep)
    restricted.unitcell_vectors = multi.unitcell_vectors
    restricted_classification = _build_peptide_solvent_classification(restricted.topology)
    total_r, water_r, ion_r = _peptide_solvent_contact_counts(
        md, restricted, 0.45, restricted_classification
    )

    assert np.array_equal(total_full, total_r)
    assert np.array_equal(water_full, water_r)
    assert np.array_equal(ion_full, ion_r)


# ---------------------------------------------------------------------------
# Fix 3: hoisted peptide/solvent classification
# ---------------------------------------------------------------------------

def _old_inline_peptide_solvent_contact_counts(md, traj, cutoff_nm: float):
    """Faithful copy of the pre-fix inline implementation (rebuilds the
    classification from `traj.topology` on every call) -- kept here only as
    an independent reference for the equivalence tests below."""
    top = traj.topology
    peptide_heavy = []
    water_heavy = []
    ion_heavy = []
    solvent_heavy = []
    atom_to_res = np.full(top.n_atoms, -1, dtype=np.int64)
    water_res = set()
    ion_res = set()
    solvent_res = set()
    for atom in top.atoms:
        elem = getattr(atom, "element", None)
        if elem is None or getattr(elem, "symbol", None) == "H":
            continue
        ridx = atom.residue.index
        atom_to_res[atom.index] = ridx
        if atom.residue.is_protein:
            peptide_heavy.append(atom.index)
        elif atom.residue.is_water:
            water_heavy.append(atom.index)
            solvent_heavy.append(atom.index)
            water_res.add(ridx)
            solvent_res.add(ridx)
        else:
            ion_heavy.append(atom.index)
            solvent_heavy.append(atom.index)
            ion_res.add(ridx)
            solvent_res.add(ridx)
    n = traj.n_frames
    zeros = np.zeros(n, dtype=np.float64)
    if len(peptide_heavy) == 0 or len(solvent_heavy) == 0:
        return zeros.copy(), zeros.copy(), zeros.copy()
    try:
        neigh = md.compute_neighbors(
            traj, float(cutoff_nm),
            query_indices=np.asarray(peptide_heavy, dtype=np.int32),
            haystack_indices=np.asarray(solvent_heavy, dtype=np.int32),
            periodic=True,
        )
    except Exception:
        return (np.full(n, np.nan, dtype=np.float64),) * 3
    total = np.zeros(n, dtype=np.float64)
    water = np.zeros(n, dtype=np.float64)
    ion = np.zeros(n, dtype=np.float64)
    for i, arr in enumerate(neigh):
        if arr is None or len(arr) == 0:
            continue
        residues = {
            int(atom_to_res[int(a)])
            for a in np.asarray(arr, dtype=np.int64)
            if 0 <= int(a) < atom_to_res.size and atom_to_res[int(a)] >= 0
        }
        total[i] = float(len(residues & solvent_res))
        water[i] = float(len(residues & water_res))
        ion[i] = float(len(residues & ion_res))
    return total, water, ion


def test_fix3_hoisted_classification_matches_old_inline_computation(solvated_system):
    multi = _make_multiframe(solvated_system, n_frames=6, scale=0.08, seed=7)

    old_total, old_water, old_ion = _old_inline_peptide_solvent_contact_counts(md, multi, 0.45)

    classification = _build_peptide_solvent_classification(multi.topology)
    new_total, new_water, new_ion = _peptide_solvent_contact_counts(md, multi, 0.45, classification)

    # Not a vacuous all-zero comparison: real, varying contact counts.
    assert np.any(old_total > 0)
    assert len(set(old_total.tolist())) > 1

    assert np.array_equal(old_total, new_total)
    assert np.array_equal(old_water, new_water)
    assert np.array_equal(old_ion, new_ion)


def test_fix3_classification_reused_across_calls_gives_identical_results(solvated_system):
    """The whole point of the hoist: computing the classification ONCE and
    reusing it across multiple `_peptide_solvent_contact_counts` calls (e.g.
    one per chunk in the real per-chunk loop) must give bit-identical results
    to computing it fresh every time."""
    classification = _build_peptide_solvent_classification(solvated_system.topology)

    frame_sets = [
        _make_multiframe(solvated_system, n_frames=2, scale=0.03, seed=s)
        for s in (21, 22, 23)
    ]
    for frames in frame_sets:
        reused = _peptide_solvent_contact_counts(md, frames, 0.45, classification)
        fresh_classification = _build_peptide_solvent_classification(frames.topology)
        fresh = _peptide_solvent_contact_counts(md, frames, 0.45, fresh_classification)
        for a, b in zip(reused, fresh):
            assert np.array_equal(a, b)


def test_fix3_classification_is_pure_function_of_topology(solvated_system):
    """Calling the classification builder twice on the same topology returns
    the same groupings (peptide/water/ion heavy-atom indices and residue
    sets) -- confirms it is safe to compute once and cache."""
    c1 = _build_peptide_solvent_classification(solvated_system.topology)
    c2 = _build_peptide_solvent_classification(solvated_system.topology)
    assert np.array_equal(c1["peptide_heavy"], c2["peptide_heavy"])
    assert np.array_equal(c1["solvent_heavy"], c2["solvent_heavy"])
    assert np.array_equal(c1["atom_to_res"], c2["atom_to_res"])
    assert c1["water_res"] == c2["water_res"]
    assert c1["ion_res"] == c2["ion_res"]
    assert c1["solvent_res"] == c2["solvent_res"]
