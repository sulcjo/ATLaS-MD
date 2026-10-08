"""Measured stored-vs-offline aux z parity per segment (GPU validation, spec Section 17)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def parity_report(run_dir) -> dict[str, Any]:
    from ..correctness._io import IntegrityError
    from ..correctness.export import R_KJ_MOL_K
    from ..correctness.state_identity import canonical_state_definition
    from ..query import load_samples, load_windows_metadata
    from .offline import (_float_column, parity_context, parity_violation, registry_model,
                          segment_aux_runtime, segment_aux_schemas)
    from .evaluate import z_from_dihedrals
    from .sample_schema import PARITY_TOLERANCE
    run_dir = Path(run_dir)
    schemas, runtimes = segment_aux_schemas(run_dir), segment_aux_runtime(run_dir)
    out: dict[str, Any] = {"segments": {}}
    for seg, schema in schemas.items():
        if schema is None:
            continue
        # Each segment is graded against its own frozen state definition (board condition 1).
        meta = load_windows_metadata(run_dir, seg) or {}
        if not meta.get("state_definition"):
            raise IntegrityError(f"segment {seg} has auxiliary samples but no frozen state_definition snapshot")
        state = canonical_state_definition(meta["state_definition"])
        beta = 1.0 / (R_KJ_MOL_K * float(state["temperature_k"]))
        ctx = parity_context(state["windows"], beta)
        s = load_samples(run_dir, segment_ids=[seg])
        n = len(np.asarray(s["cv1"]))
        theta = np.stack([_float_column(s, c, n) for c in schema.torsion_columns], axis=1)
        sha = schema.model_shas[0]
        model = registry_model(state, sha, where=f"parity_report ({seg})")
        z_off = z_from_dihedrals(theta[:, schema.model_basis_index(model)], model)
        z_run = _float_column(s, schema.z_columns[0], n)
        fin_run, fin_off = np.isfinite(z_run), np.isfinite(z_off)
        both = fin_run & fin_off
        n_mismatch = int((fin_run ^ fin_off).sum())
        prec = (runtimes.get(seg) or {}).get("precision")
        if prec is None:
            raise IntegrityError(f"segment {seg}: no precision recorded in the aux sample runtime block")
        tol = PARITY_TOLERANCE.get(prec, 0.0)
        # Strict k_max lookup: absent only if every state of this model is a sham (aux_k == 0).
        if sha in ctx["k_max_kcal"]:
            k_max = ctx["k_max_kcal"][sha]
        elif any(w.get("aux_model_sha256") == sha and float(w.get("aux_k", 0.0)) > 0 for w in state["windows"]):
            raise IntegrityError(f"segment {seg}: model {sha} has an active state but no k_max in the parity context")
        else:
            k_max = 0.0
        du = parity_violation(z_run[both], z_off[both], beta=beta, k_max_kcal=k_max,
                              centers=ctx["centers"].get(sha, []))
        n_cmp = int(both.sum())
        max_red = float(du.max()) if du.size else 0.0
        vacuous = k_max == 0.0
        ok = n_cmp > 0 and n_mismatch == 0 and (vacuous or max_red <= tol)
        row = {
            "platform": (runtimes.get(seg) or {}).get("platform"), "precision": prec, "rows": int(n),
            "n_compared": n_cmp, "n_nonfinite_stored": int((~fin_run).sum()),
            "n_nonfinite_offline": int((~fin_off).sum()), "n_finite_mismatch": n_mismatch,
            "k_max_kcal": float(k_max),
            "max_abs_dz": float(np.max(np.abs(z_run[both] - z_off[both]))) if n_cmp else 0.0,
            "max_reduced": max_red, "tolerance": tol, "ok": bool(ok)}
        if vacuous:
            row["bound_vacuous"] = True
        if n == 0:
            row["reason"] = "no rows"
        out["segments"][seg] = row
    if not out["segments"]:
        out["ok"], out["reason"] = False, "no aux segments"
        return out
    out["ok"] = all(r["ok"] for r in out["segments"].values())
    return out


if __name__ == "__main__":  # python -m gareus.auxiliary_cv.parity_report RUN_DIR [RUN_DIR ...]
    import json
    import sys
    print(json.dumps({d: parity_report(d) for d in sys.argv[1:]}, indent=2, sort_keys=True))
