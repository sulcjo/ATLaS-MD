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
        both = np.isfinite(z_off) & np.isfinite(z_run)
        du = parity_violation(z_run[both], z_off[both], beta=beta, k_max_kcal=ctx["k_max_kcal"].get(sha, 0.0),
                              centers=ctx["centers"].get(sha, []))
        prec = (runtimes.get(seg) or {}).get("precision")
        tol = PARITY_TOLERANCE.get(prec, 0.0)
        out["segments"][seg] = {
            "platform": (runtimes.get(seg) or {}).get("platform"), "precision": prec, "rows": int(n),
            "max_abs_dz": float(np.max(np.abs(z_run[both] - z_off[both]))) if both.any() else 0.0,
            "max_reduced": float(du.max()) if du.size else 0.0, "tolerance": tol,
            "ok": bool(du.size == 0 or du.max() <= tol)}
    out["ok"] = all(r["ok"] for r in out["segments"].values())
    return out


if __name__ == "__main__":  # python -m gareus.auxiliary_cv.parity_report RUN_DIR [RUN_DIR ...]
    import json
    import sys
    print(json.dumps({d: parity_report(d) for d in sys.argv[1:]}, indent=2, sort_keys=True))
