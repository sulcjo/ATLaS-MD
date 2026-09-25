"""Model core of the synthetic top-up study (spec 6.1): ladder, layout graphs, mixing, draws, union NPZ.

See ``topup_study`` for the full model description; this module holds the pieces that
``topup_study``, ``sigma_rule_study`` and ``union_solve_bench`` share.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from .ess import tau_int
from .landscapes import Landscape
from .oracle import LOW_F_THRESHOLD_KBT, reference_pmf
from .sampler import BIAS, Window, boost_dv, sample_window_exact

RUNGS = (0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0)
K_WINDOW = 50.0
SPACING_SIGMA = 2.0
EDGE_MARGIN = 0.05
N_EPOCHS = 3
REPORT_INTERVAL = 5000
TIMESTEP_FS = 4.0
N_GPUS = 4
TEMPERATURE_K = 300.0
KT_KCAL = 0.0019872041 * TEMPERATURE_K
BLOCK = 256
PMF_BINS = 60


def _axis(lo: float, hi: float) -> np.ndarray:
    a, b = lo + EDGE_MARGIN * (hi - lo), hi - EDGE_MARGIN * (hi - lo)
    n = int(round((b - a) / (SPACING_SIGMA / math.sqrt(K_WINDOW)))) + 1
    return np.linspace(a, b, max(2, n))


def _ladder(landscape: Landscape, remove_bridge: bool = False) -> List[Window]:
    """Centres x rungs, ordered centre-major. ``remove_bridge`` drops the gated pass.

    The bridge is the CV2 row nearest gated-barrier's low-F pass (cv2 = 0.2), removed in
    the two CV1 columns flanking the ridge (cv1 = 0.5) on every rung: the cross-ridge
    edge then has weak but nonzero overlap (the ridge's own CV2 flanks still leak).
    """
    c1s, c2s = _axis(*landscape.cv1_bounds), _axis(*landscape.cv2_bounds)
    drop = set()
    if remove_bridge:
        row = float(c2s[np.argmin(np.abs(c2s - 0.2))])
        cols = sorted(c1s, key=lambda c: abs(c - 0.5))[:2]
        drop = {(float(c), row) for c in cols}
    return [Window(float(c1), K_WINDOW, float(c2), K_WINDOW, lam=float(lam))
            for c1 in c1s for c2 in c2s if (float(c1), float(c2)) not in drop for lam in RUNGS]


def _layout(windows: Sequence[Window]):
    """Edges, same-rung neighbours and rung partners, built exactly as the driver builds them."""
    from ..adaptive_production import AdaptiveDecisionPolicy, WindowStateRegistry, build_geometry_edges
    from ..layout_neighbours import other_rung_same_centre, same_rung_neighbours, spatial_neighbour_pairs
    registry = WindowStateRegistry()
    for w in windows:   # the driver's k is kcal/mol/CV^2
        registry.add_state(primary_center=w.center1, primary_k=w.k1 * KT_KCAL, secondary_center=w.center2,
                           secondary_k=w.k2 * KT_KCAL, gamd_lambda=w.lam, source="synthetic")
    policy = AdaptiveDecisionPolicy(topups_enabled=True)
    edges = [(a, b) for a, b, _t, _d in build_geometry_edges(registry, policy)]
    c1 = [w.center1 for w in windows]; c2 = [w.center2 for w in windows]; lam = [w.lam for w in windows]
    pairs = spatial_neighbour_pairs(c1, c2, lam, [w.k1 * KT_KCAL for w in windows],
                                    [w.k2 * KT_KCAL for w in windows], TEMPERATURE_K)
    nb, rp = same_rung_neighbours(pairs, lam), other_rung_same_centre(c1, c2, lam)
    n = len(windows)
    return edges, {w: nb.get(w, []) for w in range(n)}, {w: rp.get(w, []) for w in range(n)}, policy


class _Streams:
    """Per-window i.i.d. draw streams, blocked so sample i never depends on how draws were chunked."""

    def __init__(self, landscape: Landscape, windows: Sequence[Window], seed: int):
        self._ls, self._w, self._seed, self._blocks = landscape, windows, int(seed), {}

    def take(self, w: int, start: int, n: int) -> np.ndarray:
        first, last = start // BLOCK, (start + n - 1) // BLOCK
        for b in range(first, last + 1):
            if (w, b) not in self._blocks:
                rng = np.random.default_rng([self._seed, w, b])
                self._blocks[(w, b)] = sample_window_exact(self._ls, self._w[w], BLOCK, rng=rng)
        pooled = np.concatenate([self._blocks[(w, b)] for b in range(first, last + 1)])
        off = start - first * BLOCK
        return pooled[off:off + n]


class _Campaign:
    def __init__(self, n_states: int):
        self.steps = np.zeros(n_states)
        self.eff = np.zeros(n_states)
        self.kept = np.zeros(n_states, dtype=np.int64)
        self.rows_w: List[np.ndarray] = []
        self.rows_x: List[np.ndarray] = []
        self.hours = 0.0
        self.topup_hours = 0.0


def _partners_in(members, neighbours, rung_partners) -> Dict[int, int]:
    mset = set(members)
    return {w: len((set(neighbours.get(w, [])) | set(rung_partners.get(w, []))) & mset) for w in members}


def _sample_segment(landscape, windows, members, length, streams, n_partners, camp, table) -> float:
    """One lockstep segment: every member advances ``length`` steps; returns its wall hours."""
    from ..adaptive.throughput import wall_hours
    for w in members:
        g = 1.0 + 2.0 * tau_int(landscape, windows[w], n_partners=n_partners[w])
        camp.steps[w] += length
        camp.eff[w] += length / (REPORT_INTERVAL * g)
        new = int(math.floor(camp.eff[w] + 1e-9)) - int(camp.kept[w])
        if new > 0:
            camp.rows_x.append(streams.take(w, int(camp.kept[w]), new))
            camp.rows_w.append(np.full(new, w, dtype=np.int64))
            camp.kept[w] += new
    hours = wall_hours(int(length), len(members), TIMESTEP_FS, N_GPUS, table)
    camp.hours += hours
    return hours


def _reduced_potentials(landscape, windows, x: np.ndarray) -> np.ndarray:
    u = np.empty((x.shape[0], len(windows)))
    for k, win in enumerate(windows):
        u[:, k] = BIAS(win, x[:, 0], x[:, 1]) + boost_dv(landscape, win, x[:, 0], x[:, 1])
    return u


def _write_union_npz(path: Path, landscape, camp: _Campaign, windows) -> dict:
    """The union NPZ format read by ``union_diagnostics_from_npz``; returns subsample counts."""
    x = np.concatenate(camp.rows_x) if camp.rows_x else np.empty((0, 2))
    w = np.concatenate(camp.rows_w) if camp.rows_w else np.empty(0, dtype=np.int64)
    np.savez(path, umbrella_reduced_bias_nk=_reduced_potentials(landscape, windows, x),
             state_ids=np.arange(len(windows)), sampled_state_ids=w, cv1=x[:, 0], cv2=x[:, 1])
    counts = {}
    for k in range(len(windows)):
        raw = int(camp.steps[k] // REPORT_INTERVAL)
        g = raw / camp.eff[k] if camp.eff[k] > 0 else 1.0
        counts[str(k)] = {"raw": raw, "t0": 0, "kept": int(camp.kept[k]), "g": float(g), "status": "subsampled"}
    return counts


def _diagnose(tmp: Path, landscape, camp, windows, layout, f_init):
    """The driver's ``_phase_union_diagnostics`` call: sigma against same-rung neighbours, rung fallback.

    Frozen with the study (ruling 31): the driver has since also dropped tried-out
    edges from the sigma neighbour sets and forces the sigma-setting neighbour into
    the patch (ruling 33, I4/I5); this harness does not mirror either.
    """
    from ..adaptive.union_diagnostics import union_diagnostics_from_npz
    edges, neighbours, rung_partners, policy = layout
    path = Path(tmp) / "union.npz"
    counts = _write_union_npz(path, landscape, camp, windows)
    sigma_nb = {w: neighbours.get(w) or rung_partners.get(w, []) for w in range(len(windows))}
    return union_diagnostics_from_npz(path, edges, kt_kcal=KT_KCAL, subsample_counts=counts,
                                      min_effect_kcal=float(policy.topup_min_effect), f_init=f_init or None,
                                      sigma_neighbours=sigma_nb)


def _pmf_rmse(landscape, windows, camp, f_kT: Dict[int, float]) -> float:
    """Low-F-weighted RMSE (kBT) of the union-MBAR CV1 PMF vs the reference PMF."""
    from scipy.special import logsumexp
    x = np.concatenate(camp.rows_x)
    n_k = np.bincount(np.concatenate(camp.rows_w), minlength=len(windows))
    act = [k for k in range(len(windows)) if n_k[k] > 0 and k in f_kT]
    u = _reduced_potentials(landscape, [windows[k] for k in act], x)
    f = np.array([f_kT[k] for k in act])
    log_w = -logsumexp(f[None, :] - u, b=n_k[act][None, :], axis=1)
    bins = np.linspace(*landscape.cv1_bounds, PMF_BINS + 1)
    dens, _ = np.histogram(x[:, 0], bins=bins, weights=np.exp(log_w - log_w.max()))
    raw, _ = np.histogram(x[:, 0], bins=bins)
    xr, pr = reference_pmf(landscape, axis="cv1")
    ref = np.interp(0.5 * (bins[:-1] + bins[1:]), xr, pr)
    use = (raw > 0) & (dens > 0) & (ref <= LOW_F_THRESHOLD_KBT)
    diff = -np.log(dens[use]) - ref[use]
    wgt = np.exp(-ref[use]); wgt /= wgt.sum()
    diff -= np.sum(wgt * diff)
    return float(np.sqrt(np.sum(wgt * diff ** 2)))
