"""Shared synthetic-swarm builders for the automatic CV2 selection tests.

`tests/test_swarm_analyze.py` keeps its own `_fake_round`/`_args` helpers untouched;
these fixtures extend the same on-disk shape with torsion features per member
(Task 1) and the configuration attributes the selector reads (Tasks 8-9). No MD,
no OpenMM: every member directory is written from literals.
"""
from __future__ import annotations

import csv
import json
import types
from pathlib import Path

import numpy as np
import pytest

from gareus.swarm.members import TRACE_COLUMNS

PHI_TORSIONS = [[0, 1, 2, 3], [4, 5, 6, 7]]
PSI_TORSIONS = [[8, 9, 10, 11], [12, 13, 14, 15]]


def make_fake_round(out: Path, *, n_members: int = 12, n_frames: int = 300, wide_anchor: bool = True,
                    with_features: bool = True, crash_one_member: bool = False, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    rd = out / "swarm" / "round_000"
    rd.mkdir(parents=True)
    with (rd / "plan.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["member_id", "cell_id", "cell_cv1", "cell_rg", "cell_e2e",
                                          "seed_id", "seed_pdb", "replicate", "velocity_seed"])
        w.writeheader()
        for m in range(n_members):
            w.writerow({"member_id": m, "cell_id": f"{m % 2}_0_0", "cell_cv1": m % 2, "cell_rg": 0,
                        "cell_e2e": 0, "seed_id": f"s{m // 2}", "seed_pdb": "/x", "replicate": m % 2,
                        "velocity_seed": m})
    json.dump({"n_cells": 2, "replicates_per_cell": n_members // 2, "n_members": n_members,
               "budget_ns": float(n_members), "seed_ns": 1.0,
               "edges": {"cv1": [0, 0.5, 1], "rg": [0, 2], "e2e": [0, 4]}},
              (rd / "plan_meta.json").open("w"))
    scale = 1.0 if wide_anchor else 0.069
    d = 2 * (len(PHI_TORSIONS) + len(PSI_TORSIONS))
    for m in range(n_members):
        md = rd / f"member_{m:04d}"
        (md / "frames").mkdir(parents=True)
        crashed = crash_one_member and m == n_members - 1
        rows = n_frames // 3 if crashed else n_frames
        cv1 = scale * rng.beta(2, 4, rows)
        hidden = rng.normal(size=(rows, d)) @ np.diag(np.linspace(2.0, 0.3, d))
        feats = np.tanh(hidden + 0.3 * ((cv1 - cv1.mean()) / max(cv1.std(), 1e-9))[:, None])
        with (md / "trace.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=TRACE_COLUMNS)
            w.writeheader()
            for i in range(rows):
                w.writerow({"frame": i, "t_ps": 2.0 * (i + 1), "cv1": float(cv1[i]),
                            "rg_nm": float(0.6 + 0.3 * cv1[i] + 0.02 * rng.normal()),
                            "e2e_nm": float(1.0 + 0.2 * hidden[i, 1]),
                            "v_pep_kj": float(rng.normal(-50, 5)), "v_dih_kj": float(rng.normal(20, 2)),
                            "potential_kj": -1e4})
                if (i + 1) % 10 == 0:
                    (md / "frames" / f"frame_{i:05d}.pdb").write_text(
                        "ATOM      1  CA  GLY A   1       0.000   0.000   0.000  1.00  0.00           C\nEND\n")
        if with_features and not crashed:
            np.save(md / "torsion_features.npy", feats)
            json.dump({"phi_torsions": PHI_TORSIONS, "psi_torsions": PSI_TORSIONS},
                      (md / "torsion_index.json").open("w"))
        done = {"member_id": m, "n_frames": rows, "graft_fallback": False, "wall_s": 10.0,
                "ns_per_day": 400.0, "status": "md_failed" if crashed else "ok"}
        json.dump(done, (md / "done.json").open("w"))
    return rd


def make_swarm_args(**overrides) -> types.SimpleNamespace:
    base = dict(
        temperature_k=300.0, sigma0p_kcal_mol=6.0, sigma0d_kcal_mol=6.0, swarm_n_windows=8,
        swarm_overlap_sigma=1.5, swarm_target_beta_sigma=1.0, swarm_min_rungs=3, swarm_max_rungs=12,
        swarm_ess_floor=50, swarm_seeds_per_window=2, swarm_discard_block_frames=20,
        swarm_min_discard_ps=0.0, swarm_output_interval_ps=2.0, contact_adaptive_max_k_kcal=1200.0,
        contact_adaptive_min_k_kcal=5.0, swarm_round=0,
        # CV configuration the selector reads
        primary_cv="nonlocal-contacts", secondary_cv="none", contact_r0_a=12.0, contact_beta_a_inv=3.0,
        contact_min_sequence_separation=4, contact_atom_selection="heavy", contact_scheme="atom-pairs",
        contact_normalize=True, diversity_bank_preset="broad", max_replicas=128,
        swarm_n_windows_cv2=4, cv2_k_max=1000.0, cv2_k_min=0.0,
        cv_selection_residual_degree=1, cv_selection_max_nonlinear_r2=0.20,
        cv_selection_max_coupling_fraction=0.25, cv_selection_k2_reference_kcal=1.0,
        cv_selection_min_windows_cv1=4, cv_selection_min_gain_nats=0.02,
        cv_selection_fallback="cv1_only",
    )
    base.update(overrides)
    return types.SimpleNamespace(**base)


@pytest.fixture
def synthetic_swarm(tmp_path):
    """Factory: build a fake swarm round under a fresh directory; returns the run's out dir."""
    counter = {"n": 0}

    def _build(**kw):
        counter["n"] += 1
        out = tmp_path / f"run{counter['n']}"
        out.mkdir()
        make_fake_round(out, **kw)
        return out

    return _build


@pytest.fixture
def swarm_args():
    return make_swarm_args
