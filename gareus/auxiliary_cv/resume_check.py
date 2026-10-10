"""Compare an interrupted+resumed auxiliary run with its uninterrupted control (spec Section 16 Stage C)."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np


def ledger_completeness(events, *, selected_per_step: Optional[int]) -> dict[str, Any]:
    step = np.asarray(np.ma.getdata(events["step"])).astype(np.int64)
    steps, counts = np.unique(step, return_counts=True)
    bad = {int(s): int(c) for s, c in zip(steps, counts)
           if selected_per_step is not None and int(c) != int(selected_per_step)}
    return {"ok": not bad, "steps": int(steps.size), "bad_steps": bad}


def _keys(samples) -> list[tuple]:
    return list(zip(np.asarray(samples["step"]).tolist(), np.asarray(samples["replica"]).tolist(),
                    np.asarray(samples["window_id"]).tolist()))


def _key_rows(samples) -> dict[tuple, int]:
    return {k: i for i, k in enumerate(_keys(samples))}


def compare_runs(control_dir, resumed_dir, *, z_atol: float = 1e-9, torsion_atol: float = 1e-9) -> dict[str, Any]:
    from ..query import load_exchanges, load_samples
    out: dict[str, Any] = {}
    ec, er = load_exchanges(Path(control_dir)), load_exchanges(Path(resumed_dir))

    def _events(ev):
        if not ev or "step" not in ev or len(ev["step"]) == 0:
            return []                                   # empty ledger: no KeyError, graded below
        return list(zip(np.asarray(np.ma.getdata(ev["step"])).tolist(),
                        np.asarray(np.ma.getdata(ev["attempt_seq"])).tolist(),
                        np.asarray(ev["assignment_sha256_after"]).tolist()))

    pc, pr = _events(ec), _events(er)
    out["ledger_duplicates"] = len(pr) - len({(s, q) for s, q, _ in pr})
    out["ledger_equal"] = sorted(pc) == sorted(pr)
    sc, sr = load_samples(Path(control_dir)), load_samples(Path(resumed_dir))
    if sc and sr:
        keys_c, keys_r = _keys(sc), _keys(sr)
        # Board condition 5: a dict/set comparison cannot see phantom duplicate rows (a reseal regression
        # with a clean ledger). Compare row counts and duplicate keys explicitly.
        out["sample_rows"] = {"control": len(keys_c), "resumed": len(keys_r)}
        out["sample_duplicate_keys"] = {"control": len(keys_c) - len(set(keys_c)),
                                        "resumed": len(keys_r) - len(set(keys_r))}
        kc, kr = _key_rows(sc), _key_rows(sr)
        out["sample_keys_equal"] = (set(kc) == set(kr) and len(keys_c) == len(keys_r)
                                    and out["sample_duplicate_keys"]["resumed"] == 0
                                    and out["sample_duplicate_keys"]["control"] == 0)
        common = sorted(set(kc) & set(kr))
        ic, ir = [kc[k] for k in common], [kr[k] for k in common]
        # Fix round 2 (minor 9): an absent value column is recorded as None (not compared), never as 0.
        out["max_abs_dz"] = None
        out["max_abs_dtorsion"] = None
        if "aux_z_00" in sc and "aux_z_00" in sr and common:
            zc = np.asarray(sc["aux_z_00"], dtype=float)[ic]
            zr = np.asarray(sr["aux_z_00"], dtype=float)[ir]
            out["max_abs_dz"] = float(np.nanmax(np.abs(zc - zr)))
        tor = sorted(c for c in sc if c.startswith("tor_") and c in sr)
        if tor and common:
            d = [np.abs(np.angle(np.exp(1j * (np.asarray(sc[c], dtype=float)[ic] - np.asarray(sr[c], dtype=float)[ir]))))
                 for c in tor]
            out["max_abs_dtorsion"] = float(np.nanmax(np.stack(d)))
    else:
        out["sample_keys_equal"] = True
    values_ok = True
    if sc and sr:
        values_ok = (out["max_abs_dz"] is not None and out["max_abs_dz"] <= z_atol
                     and out["max_abs_dtorsion"] is not None and out["max_abs_dtorsion"] <= torsion_atol)
    # Final fix wave I3: a comparison needs data on BOTH sides; an empty side (or two empty sides) is a
    # vacuous comparison, never a pass, and says why.
    reasons = []
    for label, a, b in (("exchange ledger", pc, pr), ("samples", bool(sc), bool(sr))):
        if not a and not b:
            reasons.append(f"{label} empty on both sides (vacuous comparison)")
        elif not a or not b:
            reasons.append(f"{label} empty on the {'control' if not a else 'resumed'} side")
    out["ok"] = bool(not reasons and out["ledger_duplicates"] == 0 and out["ledger_equal"]
                     and out["sample_keys_equal"] and values_ok)
    if reasons:
        out["reason"] = "; ".join(reasons)
    return out
