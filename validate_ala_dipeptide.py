#!/usr/bin/env python3
"""Validate GAREUS phi/psi reweighting on Ace-Ala-Nme.

Compares an INDEPENDENT unbiased reference phi/psi histogram against the GAREUS
MBAR/GaMD-reweighted Ramachandran 2D FES (extra_observable_pmfs/ramachandran_2d_fes).
Two independent code paths: a bug in either shows up as disagreement.

Pass criteria are calibrated for explicit ff14SB/TIP3P solution.
"""
from __future__ import annotations
import argparse
import glob
import json
from pathlib import Path

import numpy as np

KCAL_PER_K = 0.0019872041  # kcal/mol/K

# Explicit-solvent basins: name -> (phi_deg, psi_deg, radius_deg)
BASINS = {
    "ppii":    (-70.0, 140.0, 35.0),
    "beta":    (-150.0, 150.0, 35.0),
    "alpha_r": (-70.0, -30.0, 35.0),
    "alpha_l": (60.0,  40.0,  35.0),
    "c7ax":    (60.0, -70.0,  35.0),
}

# Which reweighting estimator GAREUS auto-selects for GaMD production = chignolin's.
CHIGNOLIN_PRODUCTION_ESTIMATOR = "gamd_cumulant2"


def periodic_delta_deg(a, b):
    """Signed minimal angular difference a-b in degrees, wrapped to (-180,180]."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    return ((d + 180.0) % 360.0) - 180.0


def basin_metrics(phi_centers, psi_centers, pmf, counts, basins, kbt_kcal,
                  min_counts=20):
    """Per-basin minimum PMF and sampled-counts within a periodic disk mask."""
    phi_centers = np.asarray(phi_centers, dtype=float)
    psi_centers = np.asarray(psi_centers, dtype=float)
    PHI, PSI = np.meshgrid(phi_centers, psi_centers, indexing="ij")
    out = {}
    for name, (p0, q0, r) in basins.items():
        mask = (periodic_delta_deg(PHI, p0) ** 2 + periodic_delta_deg(PSI, q0) ** 2) <= r * r
        basin_counts = float(np.nansum(np.where(mask, counts, 0.0)))
        finite = mask & np.isfinite(pmf)
        min_pmf = float(np.nanmin(np.where(finite, pmf, np.inf))) if np.any(finite) else float("nan")
        out[name] = {
            "min_pmf_kcal": min_pmf,
            "counts": basin_counts,
            "sampled": bool(basin_counts >= min_counts),
        }
    return out


def load_rama_npz(path):
    z = np.load(path)
    return {
        "phi_deg": z["phi_deg"],
        "psi_deg": z["psi_deg"],
        "pmf_kcal_mol": z["pmf_kcal_mol"],
        "counts": z["counts"],
        "probability": z["probability"],
    }


def reference_fes_from_traj(xtc, top, bins, kbt_kcal, selection="protein"):
    """Independent unbiased phi/psi 2D FES from a reference trajectory (mdtraj)."""
    import mdtraj as md
    t = md.load(str(xtc), top=str(top))
    sel = t.topology.select(selection)
    if sel.size == 0:
        sel = t.topology.select("not water and not resname HOH NA CL")
    prot = t.atom_slice(sel)
    _, phi = md.compute_phi(prot)
    _, psi = md.compute_psi(prot)
    phi_deg = np.degrees(phi[:, 0])
    psi_deg = np.degrees(psi[:, 0])
    edges = np.linspace(-180.0, 180.0, bins + 1)
    counts, _, _ = np.histogram2d(phi_deg, psi_deg, bins=[edges, edges])
    prob = counts / max(1.0, counts.sum())
    with np.errstate(divide="ignore"):
        pmf = -kbt_kcal * np.log(prob)
    pmf[~np.isfinite(pmf)] = np.nan
    if np.any(np.isfinite(pmf)):
        pmf = pmf - np.nanmin(pmf)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return {"phi_deg": centers, "psi_deg": centers, "pmf_kcal_mol": pmf,
            "counts": counts, "probability": prob}


def compare(ref, estimators, basins, kbt_kcal):
    """Compare reference vs each estimator on well-sampled basins."""
    ref_m = basin_metrics(ref["phi_deg"], ref["psi_deg"], ref["pmf_kcal_mol"],
                          ref["counts"], basins, kbt_kcal)
    report = {"reference_basins": ref_m, "estimators": {},
              "chignolin_production_estimator": CHIGNOLIN_PRODUCTION_ESTIMATOR}
    for name, fes in estimators.items():
        est_m = basin_metrics(fes["phi_deg"], fes["psi_deg"], fes["pmf_kcal_mol"],
                              fes["counts"], basins, kbt_kcal)
        diffs = {}
        for b in basins:
            both = ref_m[b]["sampled"] and est_m[b]["sampled"]
            ddg = (est_m[b]["min_pmf_kcal"] - ref_m[b]["min_pmf_kcal"]) if both else float("nan")
            diffs[b] = {"delta_kcal": ddg, "both_sampled": both,
                        "ref_sampled": ref_m[b]["sampled"], "est_sampled": est_m[b]["sampled"]}
        well = [d["delta_kcal"] for d in diffs.values() if d["both_sampled"] and np.isfinite(d["delta_kcal"])]
        max_abs = float(np.nanmax(np.abs(well))) if well else float("nan")
        report["estimators"][name] = {
            "basins": est_m,
            "delta_vs_reference": diffs,
            "max_abs_delta_well_sampled_kcal": max_abs,
            "n_well_sampled_basins": len(well),
            "passes_1kcal": bool(np.isfinite(max_abs) and max_abs <= 1.0),
        }
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description="Validate Ace-Ala-Nme phi/psi reweighting")
    ap.add_argument("--reference-run", required=True, help="Reference run dir (unbiased)")
    ap.add_argument("--rama-npz", nargs="+", required=True,
                    help="One or more rama_*_2d_fes.npz estimator surfaces")
    ap.add_argument("--estimator-names", nargs="+", default=None,
                    help="Labels parallel to --rama-npz (default: file stems)")
    ap.add_argument("--bins", type=int, default=36)
    ap.add_argument("--temperature-k", type=float, default=300.0)
    ap.add_argument("--selection", default="protein")
    ap.add_argument("--out", default="RUNS/ala_dipeptide_validation/VALIDATION_REPORT")
    args = ap.parse_args(argv)

    kbt = KCAL_PER_K * args.temperature_k
    top = sorted(glob.glob(str(Path(args.reference_run) / "*.pdb")))
    top = [t for t in top if "npt_equilibrated" in t] or top
    xtc = sorted(glob.glob(str(Path(args.reference_run) / "replica_trajectories" / "*.xtc")))
    if not top or not xtc:
        raise SystemExit(f"reference run missing topology/xtc under {args.reference_run}")
    ref = reference_fes_from_traj(xtc[0], top[0], args.bins, kbt, args.selection)

    names = args.estimator_names or [Path(p).stem for p in args.rama_npz]
    estimators = {n: load_rama_npz(p) for n, p in zip(names, args.rama_npz)}
    report = compare(ref, estimators, BASINS, kbt)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "validation_report.json").write_text(json.dumps(report, indent=2, default=float))
    print(json.dumps({k: {"passes_1kcal": v["passes_1kcal"],
                          "max_abs_delta_kcal": v["max_abs_delta_well_sampled_kcal"],
                          "n_well_sampled": v["n_well_sampled_basins"]}
                      for k, v in report["estimators"].items()}, indent=2))
    print(f"chignolin production estimator: {CHIGNOLIN_PRODUCTION_ESTIMATOR}")
    print(f"report -> {out/'validation_report.json'}")
    return report


if __name__ == "__main__":
    main()
