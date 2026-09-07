"""S1 envelope: equilibration discard read off the V-trace, Welford pooling over members,
and a shared_gamd_setup_globals.json that both PepGamdEnvelope.from_json and
production.load_reusable_shared_gamd_setup read. Fitted once; frozen (spec §3.6, §5)."""
from __future__ import annotations
import math
from pathlib import Path
from typing import Dict, List
import numpy as np
from gareus.gamd_calibration import (
    WelfordAccumulator,
    WindowEnergyStats,
    PooledEnvelope,
    pool_window_stats,
    lower_bound_threshold_and_k0,
)
from gareus.io import write_json

GROUPS = ("Total", "Dihedral")
_TRACE_KEY = {"Total": "v_pep_kj", "Dihedral": "v_dih_kj"}


def discard_frames_from_trace(v: np.ndarray, block: int = 25, tol_sigma: float = 1.0) -> int:
    """First frame from which the running tail-of-block-means agrees with the second-half
    reference mean to within ``tol_sigma`` standard errors of that tail mean.

    The trace is split into ``block``-sized chunks; the reference (``ref``) and its spread
    (``sig``, the std of block means) come from the trace's second half, on the assumption
    that any equilibration transient does not extend past the midpoint. Scanning forward from
    the first block, the earliest block index ``b`` is kept whose tail mean (mean of all block
    means from ``b`` to the end) is statistically indistinguishable from ``ref`` -- comparing
    against the *tail mean's* standard error (``sig / sqrt(len(tail))``), not against a single
    block's own spread, keeps the detector from being derailed by one noisy block inside an
    otherwise-equilibrated tail (a single-block check at ``tol_sigma = 1`` would misfire on
    roughly a third of purely stationary blocks by chance). Returns 0 when the trace is already
    stationary; never more than half the trace.
    """
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    n = v.size
    if n < 4 * block:
        return 0
    nb = n // block
    means = v[: nb * block].reshape(nb, block).mean(axis=1)
    half = nb // 2
    tail0 = means[half:]
    ref = tail0.mean()
    sig = tail0.std(ddof=1) if tail0.size > 1 else 0.0
    first_ok = half
    for b in range(nb):
        tail = means[b:]
        se = sig / math.sqrt(tail.size) if tail.size > 0 else math.inf
        tol = max(tol_sigma * se, 1e-9)
        if abs(tail.mean() - ref) <= tol:
            first_ok = b
            break
    return int(min(first_ok * block, n // 2))


def pooled_discard(per_member: List[int], quantile: float = 0.95, floor_frames: int = 0) -> int:
    if not per_member:
        return int(floor_frames)
    q = float(np.quantile(np.asarray(per_member, dtype=float), quantile))
    return int(max(math.ceil(q), int(floor_frames)))


def pool_member_envelopes(traces: Dict[int, Dict[str, np.ndarray]], discard: int) -> Dict[str, PooledEnvelope]:
    out: Dict[str, PooledEnvelope] = {}
    for grp in GROUPS:
        stats: List[WindowEnergyStats] = []
        for member, tr in sorted(traces.items()):
            v = np.asarray(tr[_TRACE_KEY[grp]], dtype=float)[int(discard):]
            v = v[np.isfinite(v)]
            if v.size == 0:
                continue
            acc = WelfordAccumulator()
            for x in v:
                acc.update(float(x))
            stats.append(acc.to_stats(grp, int(member)))
        if not stats:
            raise ValueError(f"no finite {grp} energies after discarding {discard} frames")
        out[grp] = pool_window_stats(stats)
    return out


def write_envelope_setup_dir(setup_dir: Path, envelopes: Dict[str, PooledEnvelope], *, sigma0_kj: Dict[str, float],
                              temperature_k: float, meta: dict) -> Path:
    setup_dir = Path(setup_dir)
    setup_dir.mkdir(parents=True, exist_ok=True)
    all_globals: Dict[str, float] = {}
    report = {}
    for grp, env in envelopes.items():
        thr = lower_bound_threshold_and_k0(env, float(sigma0_kj[grp]))
        all_globals.update({
            f"Vmax_{grp}": env.vmax, f"Vmin_{grp}": env.vmin, f"Vavg_{grp}": env.vavg, f"sigmaV_{grp}": env.sigmav,
            f"k0_{grp}": thr.k0, f"k_{grp}": thr.k, f"threshold_energy_{grp}": thr.threshold_energy,
            f"sigma0_{grp}": float(sigma0_kj[grp]),
        })
        report[grp] = {
            "vmax_kj_mol": env.vmax, "vmin_kj_mol": env.vmin, "vavg_kj_mol": env.vavg, "sigmaV_kj_mol": env.sigmav,
            "sigma0_kj_mol": float(sigma0_kj[grp]), "k0": thr.k0, "k": thr.k, "threshold_energy_kj_mol": thr.threshold_energy,
            "boosted": thr.boosted, "n_windows_pooled": env.n_windows, "n_samples_pooled": env.n_total,
        }
    path = setup_dir / "shared_gamd_setup_globals.json"
    write_json(path, {
        "mode": "swarm_unbiased_envelope",
        "description": (
            "Vmax/Vmin/Vavg/sigmaV per channel pooled over the equilibrated tail of every unbiased swarm member "
            "(Pep-GaMD partition present, boost off). k0/threshold from gamd-openmm's lower-bound formula. "
            "Frozen for the whole campaign; production consumes it via --shared-gamd-setup-dir."
        ),
        "gamd_boost_type": "pep-gamd-lower-dual",
        "temperature_K": float(temperature_k),
        "joint_envelope": report,
        "interesting_globals": dict(all_globals),
        "all_globals": all_globals,
        **{k: v for k, v in meta.items()},
    })
    return path
