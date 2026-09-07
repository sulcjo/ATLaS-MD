import math, types
import numpy as np


def test_rg_and_e2e_from_ca_positions():
    from gareus.swarm.stratify import rg_and_e2e_nm
    ca = np.array([[0.0, 0, 0], [0.38, 0, 0], [0.76, 0, 0], [1.14, 0, 0]])
    rg, e2e = rg_and_e2e_nm(ca)
    assert math.isclose(e2e, 1.14, rel_tol=1e-9)
    assert math.isclose(rg, math.sqrt(((ca[:, 0] - ca[:, 0].mean()) ** 2).mean()), rel_tol=1e-9)


def test_quantile_edges_are_unique_and_cover_range():
    from gareus.swarm.stratify import quantile_edges
    vals = np.concatenate([np.zeros(50), np.linspace(0, 1, 50)])   # heavy tie at 0
    edges = quantile_edges(vals, 4)
    assert edges[0] <= vals.min() and edges[-1] >= vals.max()
    assert np.all(np.diff(edges) > 0)
    assert 2 <= len(edges) <= 5


def _seeds(n=60, seed=0):
    from gareus.swarm.stratify import SeedDescriptor
    rng = np.random.default_rng(seed)
    return [SeedDescriptor(f"s{i}", f"/x/s{i}.pdb", float(rng.uniform(0, 1)), float(rng.uniform(0.4, 0.9)),
                           float(rng.uniform(0.3, 2.5))) for i in range(n)]


def test_stratify_cells_assigns_every_seed_exactly_once():
    from gareus.swarm.stratify import stratify_cells
    seeds = _seeds()
    cells = stratify_cells(seeds, (3, 2, 2))
    assert sum(len(v) for v in cells.values()) == len(seeds)
    assert all(len(v) > 0 for v in cells.values())            # only occupied cells are returned
    assert all(len(k) == 3 for k in cells)


def test_plan_members_equal_quota_distinct_seeds_first_then_velocity_replicates():
    from gareus.swarm.stratify import stratify_cells, plan_members
    seeds = _seeds(n=12)
    cells = stratify_cells(seeds, (2, 1, 1))                  # 2 cells, ~6 seeds each
    rows, meta = plan_members(cells, replicates_per_cell=8, seed_ns=1.0, budget_ns=None, base_seed=7)
    assert meta["n_members"] == 2 * 8 and math.isclose(meta["budget_ns"], 16.0)
    for cell_id in {r["cell_id"] for r in rows}:
        cell_rows = [r for r in rows if r["cell_id"] == cell_id]
        assert len(cell_rows) == 8
        seeds_used = [r["seed_id"] for r in cell_rows]
        n_distinct_available = len(cells[tuple(int(x) for x in cell_id.split("_"))])
        # first n_distinct members use distinct seeds; the remainder are velocity replicates
        assert len(set(seeds_used[:n_distinct_available])) == min(8, n_distinct_available)
        assert all(r["replicate"] == 0 for r in cell_rows[:n_distinct_available])
    assert len({r["velocity_seed"] for r in rows}) == len(rows)


def test_plan_members_derives_replicates_from_budget():
    from gareus.swarm.stratify import stratify_cells, plan_members
    cells = stratify_cells(_seeds(n=40), (2, 2, 2))
    rows, meta = plan_members(cells, replicates_per_cell=1, seed_ns=1.0, budget_ns=3.0 * len(cells), base_seed=1)
    assert meta["replicates_per_cell"] == 3 and meta["n_members"] == 3 * len(cells)


def test_plan_members_refuses_budget_below_one_replicate():
    from gareus.swarm.stratify import stratify_cells, plan_members
    cells = stratify_cells(_seeds(n=40), (2, 2, 2))
    try:
        plan_members(cells, replicates_per_cell=1, seed_ns=1.0, budget_ns=0.5 * len(cells), base_seed=1)
    except ValueError as e:
        assert "budget" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_describe_seeds_uses_contact_cv_and_ca_geometry():
    from gareus.swarm.stratify import describe_seeds
    # 4 "CA" atoms in a line; contact pair (0,3) far apart -> cv1 ~ 0
    pos = np.array([[0.0, 0, 0], [0.38, 0, 0], [0.76, 0, 0], [1.14, 0, 0]])
    lib = [{"pdb_path": "/x/a.pdb", "positions_nm": pos, "primary_cv_value": float("nan")}]
    args = types.SimpleNamespace(contact_r0_a=4.5, contact_beta_a_inv=6.0, contact_normalize=True)
    d = describe_seeds(lib, [0, 1, 2, 3], [(0, 3, 1.0)], args)
    assert len(d) == 1 and d[0].cv1 < 0.05 and math.isclose(d[0].e2e_nm, 1.14, rel_tol=1e-9)


def test_stratify_uses_library_quantiles_so_a_narrow_cv1_range_fills_every_bin():
    """r7 measured: heavy-CV1 spans ~0-0.069. A fixed [0,1] grid would leave the upper CV1 bins empty."""
    from gareus.swarm.stratify import SeedDescriptor, stratify_cells, quantile_edges
    rng = np.random.default_rng(5)
    seeds = [SeedDescriptor(f"s{i}", f"/x/{i}.pdb", float(rng.uniform(0.0, 0.069)), float(rng.uniform(0.43, 0.957)),
                            float(rng.uniform(0.40, 3.03))) for i in range(400)]
    cells = stratify_cells(seeds, (4, 3, 3))
    cv1_bins_used = {k[0] for k in cells}
    assert cv1_bins_used == {0, 1, 2, 3}                                  # all four CV1 bins occupied
    edges = quantile_edges(np.array([s.cv1 for s in seeds]), 4)
    assert edges[-1] <= 0.069 + 1e-9 and edges[0] >= 0.0                # edges live inside the library's own range
    assert min(len(v) for v in cells.values()) >= 1


def test_describe_seeds_crosschecks_library_rg_e2e_columns_and_recomputed_wins():
    from gareus.swarm.stratify import describe_seeds
    pos = np.array([[0.0, 0, 0], [0.38, 0, 0], [0.76, 0, 0], [1.14, 0, 0]])
    lib = [{"pdb_path": "/x/a.pdb", "positions_nm": pos, "primary_cv_value": float("nan"),
            "source_row": {"rg_nm": "0.9", "end_to_end_nm": "1.14", "contact_count": "3"}}]     # rg_nm disagrees by > 0.05 nm
    args = types.SimpleNamespace(contact_r0_a=4.5, contact_beta_a_inv=6.0, contact_normalize=True)
    mism = []
    d = describe_seeds(lib, [0, 1, 2, 3], [(0, 3, 1.0)], args, mismatches=mism)
    assert len(mism) == 1 and mism[0]["column"] == "rg_nm"
    assert not math.isclose(d[0].rg_nm, 0.9, abs_tol=0.05)          # recomputed value, not the CSV's


def test_describe_seeds_reports_dropped_seeds_with_reason_and_keeps_survivors():
    from gareus.swarm.stratify import describe_seeds
    bad_pos = np.array([[0.0, 0, 0], [0.38, 0, 0], [0.76, 0, 0], [1.14, 0, 0]])          # only 4 atoms
    good_pos = np.array([[0.0, 0, 0], [0.38, 0, 0], [0.76, 0, 0], [1.14, 0, 0],
                          [1.52, 0, 0], [1.90, 0, 0]])                                    # 6 atoms
    lib = [
        {"pdb_path": "/x/bad.pdb", "positions_nm": bad_pos, "primary_cv_value": float("nan")},
        {"pdb_path": "/x/good.pdb", "positions_nm": good_pos, "primary_cv_value": float("nan")},
    ]
    args = types.SimpleNamespace(contact_r0_a=4.5, contact_beta_a_inv=6.0, contact_normalize=True)
    # contact pair references atom index 5: out of bounds for bad_pos (4 atoms), valid for good_pos (6 atoms)
    dropped = []
    d = describe_seeds(lib, [0, 1, 2, 3], [(0, 5, 1.0)], args, dropped_out=dropped)
    assert len(d) == 1 and d[0].pdb_path == "/x/good.pdb"           # surviving seed unaffected
    assert len(dropped) == 1
    seed_id, reason = dropped[0]
    assert seed_id == "seed_00000"
    assert "Error" in reason
