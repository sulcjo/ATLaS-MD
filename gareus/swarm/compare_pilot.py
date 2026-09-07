"""Swarm-vs-S3-pilot envelope comparison -- the campaign-freeze criterion.

Both files are ``shared_gamd_setup_globals.json`` documents: ``PepGamdEnvelope.from_json``
gives Vmax/Vmin/threshold/k0max per channel, and sigma_V (not part of that frozen
dataclass) is read straight from the JSON, preferring ``all_globals["sigmaV_<Group>"]``
and falling back to the swarm writer's ``joint_envelope[<Group>]["sigmaV_kj_mol"]``
layout (see ``gareus/swarm/envelope.py:write_envelope_setup_dir``).

The pilot's envelope is measured under one boosted umbrella window; the swarm's is
unbiased and unboosted. They sample different ensembles on purpose, so this check is
a plausibility gate, not an identity test -- the swarm envelope stays authoritative
for the campaign (spec Sec 2 S1, Sec 3.6). A ``fail`` means: do not freeze the
campaign config, inspect which channel disagrees, and extend the swarm; never hand-edit
the envelope.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from gareus.pep_gamd import PepGamdEnvelope

_GROUPS = ("Total", "Dihedral")


def _sigma_v(doc: dict, group: str, path) -> float:
    all_globals = doc.get("all_globals")
    if isinstance(all_globals, dict) and f"sigmaV_{group}" in all_globals:
        return float(all_globals[f"sigmaV_{group}"])
    joint = doc.get("joint_envelope")
    if isinstance(joint, dict) and isinstance(joint.get(group), dict) and "sigmaV_kj_mol" in joint[group]:
        return float(joint[group]["sigmaV_kj_mol"])
    raise KeyError(f"{path}: no sigmaV_{group} in all_globals and no joint_envelope[{group!r}]['sigmaV_kj_mol']")


def _channel(env: PepGamdEnvelope, group: str):
    if group == "Total":
        return env.vmax_total, env.vmin_total, env.k0max_total
    return env.vmax_dih, env.vmin_dih, env.k0max_dih


def compare_envelopes(swarm_json, pilot_json, *, sigma_rel_tol: float = 0.25,
                       extrema_sigma_tol: float = 2.0,
                       k0_ratio_bounds=(0.7, 1.4)) -> dict:
    """Compare the swarm's frozen envelope against the S3 pilot's per channel.

    Per group: sigma_rel_diff = |sigma_swarm - sigma_pilot| / sigma_pilot must be
    <= sigma_rel_tol; vmax/vmin_diff_in_pilot_sigma = |swarm - pilot| / sigma_pilot
    must be <= extrema_sigma_tol; k0_ratio = k0max_swarm / k0max_pilot must fall
    inside k0_ratio_bounds. Returns a dict with a "groups" sub-dict (one entry per
    channel), overall "status" ("pass"/"fail"), "freeze_allowed" and "reasons".
    """
    swarm_doc = json.loads(Path(swarm_json).read_text())
    pilot_doc = json.loads(Path(pilot_json).read_text())
    swarm_env = PepGamdEnvelope.from_json(swarm_json)
    pilot_env = PepGamdEnvelope.from_json(pilot_json)

    groups: dict = {}
    reasons: list = []
    for group in _GROUPS:
        sigma_swarm = _sigma_v(swarm_doc, group, swarm_json)
        sigma_pilot = _sigma_v(pilot_doc, group, pilot_json)
        vmax_swarm, vmin_swarm, k0_swarm = _channel(swarm_env, group)
        vmax_pilot, vmin_pilot, k0_pilot = _channel(pilot_env, group)

        sigma_rel_diff = abs(sigma_swarm - sigma_pilot) / sigma_pilot
        vmax_diff_in_pilot_sigma = abs(vmax_swarm - vmax_pilot) / sigma_pilot
        vmin_diff_in_pilot_sigma = abs(vmin_swarm - vmin_pilot) / sigma_pilot
        k0_ratio = k0_swarm / k0_pilot

        checks = {
            "sigma": sigma_rel_diff <= sigma_rel_tol,
            "vmax": vmax_diff_in_pilot_sigma <= extrema_sigma_tol,
            "vmin": vmin_diff_in_pilot_sigma <= extrema_sigma_tol,
            "k0": k0_ratio_bounds[0] <= k0_ratio <= k0_ratio_bounds[1],
        }
        ok = all(checks.values())
        groups[group] = {
            "sigma_rel_diff": sigma_rel_diff,
            "vmax_diff_in_pilot_sigma": vmax_diff_in_pilot_sigma,
            "vmin_diff_in_pilot_sigma": vmin_diff_in_pilot_sigma,
            "k0_ratio": k0_ratio,
            "ok": ok,
        }
        if not ok:
            failed = [name for name, passed in checks.items() if not passed]
            reasons.append(f"{group}: {', '.join(failed)} out of tolerance")

    status = "pass" if all(g["ok"] for g in groups.values()) else "fail"
    return {
        "groups": groups,
        "status": status,
        "freeze_allowed": status == "pass",
        "reasons": reasons,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("swarm_json", help="Swarm's shared_gamd_setup_globals.json (authoritative envelope).")
    parser.add_argument("pilot_json", help="S3 pilot's shared_gamd_setup_globals.json.")
    parser.add_argument("--out", default=None, help="Write the comparison dict as JSON to this path (e.g. pilot_comparison.json).")
    parser.add_argument("--sigma-rel-tol", type=float, default=0.25)
    parser.add_argument("--extrema-sigma-tol", type=float, default=2.0)
    parser.add_argument("--k0-ratio-min", type=float, default=0.7)
    parser.add_argument("--k0-ratio-max", type=float, default=1.4)
    args = parser.parse_args(list(argv) if argv is not None else None)

    result = compare_envelopes(
        args.swarm_json, args.pilot_json,
        sigma_rel_tol=args.sigma_rel_tol,
        extrema_sigma_tol=args.extrema_sigma_tol,
        k0_ratio_bounds=(args.k0_ratio_min, args.k0_ratio_max),
    )
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.out:
        Path(args.out).write_text(text)
        print(f"Wrote {args.out}: status={result['status']} freeze_allowed={result['freeze_allowed']}")
    else:
        print(text)
    return 0 if result["freeze_allowed"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
