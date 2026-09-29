"""X8 discovery census: basin strings, leader clustering, first-seen counting, phase order, I/O."""
import json
import os
from pathlib import Path

import numpy as np
import pytest

from gareus.adaptive import discovery_census as dc


# --------------------------------------------------------------------------- basins

@pytest.mark.parametrize("phi,psi,label", [
    (-60.0, -45.0, "A"),      # alpha-R
    (-60.0, -120.0, "A"),     # lower psi edge is inside alpha-R (half-open [-120, 50))
    (-60.0, 50.0, "P"),       # upper psi edge leaves alpha-R; phi >= -90 -> PPII
    (-60.0, 49.999, "A"),
    (-120.0, 130.0, "B"),     # beta
    (-90.0, 130.0, "P"),      # phi = -90 is PPII (beta is phi < -90)
    (-90.001, 130.0, "B"),
    (-75.0, 145.0, "P"),      # PPII
    (-100.0, -150.0, "B"),    # psi wraps: below -120 is the extended half
    (60.0, 45.0, "L"),        # alpha-L
    (0.0, -90.0, "L"),        # phi = 0 belongs to phi >= 0; psi = -90 inside [-90, 90)
    (0.0, 90.0, "O"),         # psi = 90 leaves alpha-L
    (80.0, -170.0, "O"),      # other
    (-0.001, 0.0, "A"),
    (180.0, 180.0, "B"),      # +180 wraps to -180 on both angles: phi < -90, psi extended
    (-180.0, -180.0, "B"),
    (540.0, -405.0, "A"),     # wrap: phi 540 -> -180, psi -405 -> -45 (alpha-R needs only phi < 0)
])
def test_basin_labels_boundaries(phi, psi, label):
    code = dc.basin_codes(np.array([phi]), np.array([psi]))
    assert dc.BASIN_LETTERS[int(code[0])] == label


def test_basin_partition_is_complete():
    g = np.linspace(-180, 180, 181)
    phi, psi = np.meshgrid(g, g)
    codes = dc.basin_codes(phi.ravel(), psi.ravel())
    assert codes.min() >= 0 and codes.max() < len(dc.BASIN_LETTERS)
    assert set(np.unique(codes)) == set(range(len(dc.BASIN_LETTERS)))


def test_encode_decode_round_trip():
    codes = np.array([[0, 1, 2, 3, 4], [4, 4, 4, 4, 4], [0, 0, 0, 0, 0]], dtype=np.uint8)
    ids = dc.encode_basin_strings(codes)
    assert len(set(ids.tolist())) == 3
    assert [dc.decode_basin_string(int(i), 5) for i in ids] == ["ABPLO", "OOOOO", "AAAAA"]


# --------------------------------------------------------------------------- clustering

def _kabsch_rmsd(a, b):
    a = a - a.mean(0)
    b = b - b.mean(0)
    u, s, vt = np.linalg.svd(a.T @ b)
    d = np.sign(np.linalg.det(u @ vt))
    s[-1] *= d
    val = (np.sum(a * a) + np.sum(b * b) - 2 * s.sum()) / len(a)
    return np.sqrt(max(val, 0.0))


def _naive_leader(xyz, cutoff):
    leaders, assign = [], np.empty(len(xyz), dtype=int)
    for i, x in enumerate(xyz):
        for li, l in enumerate(leaders):
            if _kabsch_rmsd(x, xyz[l]) <= cutoff:
                assign[i] = li
                break
        else:
            leaders.append(i)
            assign[i] = len(leaders) - 1
    return np.array(leaders), assign


def _rotation(seed):
    q = np.random.default_rng(seed).normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def test_leader_cluster_matches_naive_sequential():
    rng = np.random.default_rng(3)
    base = rng.normal(scale=0.5, size=(10, 3))
    xyz = base[None] + rng.normal(scale=0.15, size=(120, 10, 3))
    for cutoff in (0.1, 0.2, 0.3):
        leaders, assign = dc.leader_cluster(xyz, cutoff_nm=cutoff)
        ref_l, ref_a = _naive_leader(xyz, cutoff)
        assert leaders.tolist() == ref_l.tolist()
        assert assign.tolist() == ref_a.tolist()


def test_leader_cluster_known_count_and_rigid_invariance():
    rng = np.random.default_rng(7)
    shapes = [rng.normal(scale=0.6, size=(10, 3)) for _ in range(3)]
    frames = []
    for k in range(60):
        s = shapes[k % 3] + rng.normal(scale=0.01, size=(10, 3))
        frames.append(s @ _rotation(k).T + rng.normal(scale=2.0, size=3))   # rigid motion
    xyz = np.array(frames)
    leaders, assign = dc.leader_cluster(xyz, cutoff_nm=0.2)
    assert leaders.tolist() == [0, 1, 2]
    assert assign.tolist() == [k % 3 for k in range(60)]
    again = dc.leader_cluster(xyz, cutoff_nm=0.2)
    assert again[0].tolist() == leaders.tolist() and again[1].tolist() == assign.tolist()


def test_leader_cluster_empty():
    leaders, assign = dc.leader_cluster(np.zeros((0, 10, 3)), cutoff_nm=0.2)
    assert leaders.size == 0 and assign.size == 0


# --------------------------------------------------------------------------- census

def test_first_seen_census_counts_raw_and_populated():
    ids = [np.array([1, 1, 2]), np.array([2, 3, 1, 3]), np.array([3, 4, 4, 4, 5])]
    ns = [10.0, 20.0, 50.0]
    rows = dc.first_seen_census(["p0", "p1", "p2"], ids, ns, min_frames=2)
    assert [r["new_raw"] for r in rows] == [2, 1, 2]           # {1,2}, {3}, {4,5}
    assert [r["cumulative_raw"] for r in rows] == [2, 3, 5]
    # populated (>= 2 frames cumulative): 1 in p0; 2 and 3 in p1; 4 in p2; 5 never
    assert [r["new_populated"] for r in rows] == [1, 2, 1]
    assert [r["cumulative_populated"] for r in rows] == [1, 3, 4]
    assert [r["distinct_in_phase"] for r in rows] == [2, 3, 3]
    assert rows[2]["new_populated_per_100ns"] == pytest.approx(2.0)
    assert rows[0]["frames"] == 3


def test_saturation_verdicts():
    rows = [{"phase": "a", "ns": 100.0, "new_populated": 100, "cumulative_populated": 100},
            {"phase": "b", "ns": 100.0, "new_populated": 50, "cumulative_populated": 150},
            {"phase": "c", "ns": 100.0, "new_populated": 30, "cumulative_populated": 180}]
    s = dc.saturation(rows, key="populated")
    assert s["last_rate_per_100ns"] == pytest.approx(30.0)
    assert s["last_vs_previous"] == pytest.approx(0.6)
    assert s["last_vs_first"] == pytest.approx(0.3)
    assert s["last_vs_campaign_mean"] == pytest.approx(30.0 / 60.0)
    assert s["verdict"] == "discovering"
    rows[-1]["new_populated"] = 1
    assert dc.saturation(rows, key="populated")["verdict"] == "saturated"   # 1 / (151/300*100) < 0.1
    rows[-1]["new_populated"] = 0
    assert dc.saturation(rows, key="populated")["verdict"] == "saturated"
    assert dc.saturation(rows[:1], key="populated")["verdict"] == "insufficient_phases"


# --------------------------------------------------------------------------- phase order

def _mk_phase(d: Path, mtime=None):
    (d / "samples").mkdir(parents=True)
    (d / "segments.json").write_text("{}")
    (d / "epoch_window_map.csv").write_text("local_window_index,state_id\n")
    (d / "replica_trajectories").mkdir()
    if mtime is not None:
        os.utime(d, (mtime, mtime))


def test_ordered_phases_topups_by_ordinal_then_mtime(tmp_path):
    ad = tmp_path / "adaptive_production"
    _mk_phase(ad / "epoch_000")
    _mk_phase(ad / "epoch_001" / "baseline", 100)
    _mk_phase(ad / "epoch_001" / "topup_001_900", 300)   # later by mtime although larger suffix
    _mk_phase(ad / "epoch_001" / "topup_001_100", 400)
    _mk_phase(ad / "epoch_001" / "topup_002_500", 200)
    _mk_phase(ad / "final" / "baseline")
    _mk_phase(ad / "final_extension_001")
    (ad / "epoch_003").mkdir()                            # no data: reported skipped
    (ad / "epoch_010").mkdir()
    _mk_phase(ad / "epoch_010" / "baseline")
    phases, skipped = dc.ordered_phases(ad)
    assert [p[0] for p in phases] == [
        "epoch_000", "epoch_001/baseline", "epoch_001/topup_001_900", "epoch_001/topup_001_100",
        "epoch_001/topup_002_500", "epoch_010/baseline", "final/baseline", "final_extension_001"]
    assert [s["phase"] for s in skipped] == ["epoch_003"]


# --------------------------------------------------------------------------- end to end (XTC)

def _place(a, b, c, bond, angle_deg, torsion_deg):
    """NeRF: position of d given a, b, c, |cd|, angle bcd, torsion abcd."""
    ang, tor = np.radians(angle_deg), np.radians(torsion_deg)
    bc = c - b
    bc /= np.linalg.norm(bc)
    n = np.cross(b - a, bc)
    n /= np.linalg.norm(n)
    m = np.cross(n, bc)
    d2 = np.array([-bond * np.cos(ang), bond * np.sin(ang) * np.cos(tor), bond * np.sin(ang) * np.sin(tor)])
    return c + d2[0] * bc + d2[1] * m + d2[2] * n


def _backbone(phis, psis):
    """N/CA/C coordinates (nm) for len(phis) residues; phi[0] and psi[-1] are unused."""
    n_res = len(phis)
    xyz = [np.array([0.0, 0.0, 0.0]), np.array([0.146, 0.0, 0.0]), np.array([0.2, 0.14, 0.0])]
    for i in range(1, n_res):
        xyz.append(_place(xyz[-3], xyz[-2], xyz[-1], 0.133, 116.0, psis[i - 1]))   # N_i
        xyz.append(_place(xyz[-3], xyz[-2], xyz[-1], 0.146, 122.0, 180.0))         # CA_i (omega)
        xyz.append(_place(xyz[-3], xyz[-2], xyz[-1], 0.152, 111.0, phis[i]))       # C_i
    return np.array(xyz)


def _topology(n_res):
    import mdtraj as md
    top = md.Topology()
    ch = top.add_chain()
    for i in range(n_res):
        r = top.add_residue("ALA", ch, resSeq=i + 1)
        for name, el in (("N", "nitrogen"), ("CA", "carbon"), ("C", "carbon")):
            top.add_atom(name, getattr(md.element, el), r)
    return top


BASIN_ANGLES = {"A": (-63.0, -42.0), "B": (-130.0, 135.0), "P": (-70.0, 145.0), "L": (60.0, 40.0)}


def _frames_for(strings, rng):
    out = []
    for s in strings:
        phis = [-60.0] + [BASIN_ANGLES[ch][0] + rng.normal(scale=3) for ch in s] + [-60.0]
        psis = [140.0] + [BASIN_ANGLES[ch][1] + rng.normal(scale=3) for ch in s] + [140.0]
        out.append(_backbone(phis, psis))
    return np.array(out, dtype=np.float32)


def _write_phase(ad, rel, per_replica, top, t0=0.0, dt=1.0, manifest=True):
    import mdtraj as md
    d = ad / rel
    _mk_phase(d)
    md.Trajectory(per_replica[0][0][:1], top).save_pdb(str(d / "solute_only.pdb"))
    if manifest:
        (d / "run_manifest.json").write_text(json.dumps(
            {"resolved_args": {"traj_interval": 500, "timestep_fs": 2.0}}))   # 1 ps per frame
    for rep, segs in enumerate(per_replica):
        t = t0
        for k, xyz in enumerate(segs):
            name = f"replica_{rep:03d}.xtc" if k == 0 else f"replica_{rep:03d}_resume_from_{k * 1000:09d}.xtc"
            times = t + dt * (1 + np.arange(len(xyz)))
            md.Trajectory(xyz, top, time=times).save_xtc(str(d / "replica_trajectories" / name))
            t = times[-1]


def test_run_census_end_to_end_on_xtc(tmp_path):
    rng = np.random.default_rng(0)
    top = _topology(4)                                  # core = residues 2 and 3
    ad = tmp_path / "adaptive_production"
    p0 = [[_frames_for(["AA"] * 6 + ["BB"] * 4, rng)], [_frames_for(["AA"] * 10, rng)]]
    # phase 1: replica 0 has a resume segment; its base file overlaps the resume by 2 frames
    base = _frames_for(["AA"] * 5 + ["LL"] * 2, rng)    # the 2 "LL" frames are superseded
    resume = _frames_for(["PB"] * 3, rng)
    _write_phase(ad, "epoch_000", p0, top)
    _write_phase(ad, "final/baseline", [[base]], top, t0=100.0)
    import mdtraj as md
    # overwrite the resume so it starts at frame 6 (time 106): base frames at 106, 107 superseded
    md.Trajectory(resume, top, time=106.0 + np.arange(3)).save_xtc(
        str(ad / "final/baseline/replica_trajectories/replica_000_resume_from_000001000.xtc"))
    (ad / "final/baseline/replica_trajectories/replica_001.xtc").write_bytes(b"")   # 0-byte: skipped

    res = dc.run_census(ad, basin_stride=1, cluster_stride=1, min_frames=1, workers=1)
    basins = {r["phase"]: r for r in res["basin_strings"]["phases"]}
    assert basins["epoch_000"]["new_raw"] == 2                        # AA, BB
    assert basins["final/baseline"]["new_raw"] == 1                   # PB only; LL superseded
    assert basins["final/baseline"]["frames"] == 5 + 3
    assert basins["final/baseline"]["ns"] == pytest.approx(8 * 1.0 / 1000)
    assert res["definitions"]["basin_strings"]["core_residues"] == ["ALA2", "ALA3"]
    assert sorted(res["basin_strings"]["states_first_seen"]["epoch_000"]) == ["AA", "BB"]
    assert res["files"]["skipped"][0]["reason"] == "empty"
    cl = {r["phase"]: r for r in res["ca_clusters"]["phases"]}
    assert cl["epoch_000"]["new_raw"] >= 1
    assert cl["epoch_000"]["frames"] == 20 and cl["final/baseline"]["frames"] == 8
    fine = dc.run_census(ad, basin_stride=1, cluster_stride=1, min_frames=1, workers=1, cutoff_nm=0.02)
    fine_cl = {r["phase"]: r for r in fine["ca_clusters"]["phases"]}
    assert fine_cl["final/baseline"]["new_raw"] >= 1     # PB geometry differs from AA/BB at 0.2 A
    again = dc.run_census(ad, basin_stride=1, cluster_stride=1, min_frames=1, workers=2)
    assert again["ca_clusters"]["phases"] == res["ca_clusters"]["phases"]
    assert again["basin_strings"]["phases"] == res["basin_strings"]["phases"]

    out = tmp_path / "out"
    paths = dc.write_outputs(res, out, plot=False)
    assert (out / "discovery_census.json").exists() and (out / "discovery_census.csv").exists()
    rows = (out / "discovery_census.csv").read_text().splitlines()
    assert rows[0].startswith("definition,level,phase")
    assert "json" in paths


def test_run_census_strides_are_recorded_and_applied(tmp_path):
    rng = np.random.default_rng(1)
    top = _topology(4)
    ad = tmp_path / "adaptive_production"
    _write_phase(ad, "epoch_000", [[_frames_for(["AA"] * 10 + ["BB"] * 10, rng)]], top)
    res = dc.run_census(ad, basin_stride=4, cluster_stride=10, min_frames=1, workers=1)
    row = res["basin_strings"]["phases"][0]
    assert row["frames"] == 5                             # frames 0,4,8,12,16
    assert row["ns"] == pytest.approx(20 / 1000)          # time basis = all frames, before stride
    assert res["ca_clusters"]["phases"][0]["frames"] == 2
    assert res["definitions"]["basin_strings"]["frame_stride"] == 4
    assert res["definitions"]["ca_clusters"]["frame_stride"] == 10
