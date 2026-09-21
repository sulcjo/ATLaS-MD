"""Frozen discovery-region inventory (repair spec F05; findings I02, I03).

The swarm's discovery frames define a *partition*, not equilibrium populations: which
stretches of the primary CV are supported, and which structural cells the frames fall in.
Every retained region must receive a supported design representative or be reported as
unresolved. Rare regions are kept -- the old endpoint tuner lowered the CV1 upper bound
until every uniformly spaced centre had a seed, which deleted a separated compact cluster
the discovery had found (review I02). Numerical-invalid rows are failures, counted
separately; exact duplicate frames do not create independent support.

NumPy only. Python 3.9 compatible.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

REGION_ROLE = "discovery_partition"
INVENTORY_VERSION = "region_inventory_v1"


@dataclass(frozen=True)
class RegionPolicy:
    """Frozen detection parameters. A gap wider than ``gap_sigma`` window widths at the
    stiffest allowed umbrella separates two supported CV1 intervals."""

    gap_sigma: float = 3.0
    min_members: int = 5               # fewer rows than this is a diagnostic singleton, still kept
    representative_quantile: float = 0.5
    version: str = INVENTORY_VERSION


@dataclass
class Region:
    region_id: str
    axis: str                          # "cv1" | "structural"
    lo: float
    hi: float
    representative: float              # CV1 value of the representative
    representative_row: int
    n_members: int
    n_families: int
    n_distinct_frames: int
    eligible_seed_rows: List[int]
    radius: float                      # half-width of the interval
    is_rare: bool
    role: str = REGION_ROLE

    def as_record(self) -> dict:
        return {"region_id": self.region_id, "axis": self.axis, "lo": self.lo, "hi": self.hi,
                "representative": self.representative, "representative_row": int(self.representative_row),
                "n_members": int(self.n_members), "n_families": int(self.n_families),
                "n_distinct_frames": int(self.n_distinct_frames), "n_eligible_seeds": len(self.eligible_seed_rows),
                "radius": self.radius, "is_rare": bool(self.is_rare), "role": self.role}


@dataclass
class RegionInventory:
    policy: RegionPolicy
    regions: List[Region]
    n_rows: int
    n_invalid_rows: int
    n_failed_seed_preparations: int
    gap_threshold: float
    unresolved: List[str] = field(default_factory=list)
    seed_values: Optional[np.ndarray] = None       # CV1 of every exportable frame (what centres snap to)

    @property
    def cv1_regions(self) -> List[Region]:
        return [r for r in self.regions if r.axis == "cv1"]

    def as_record(self) -> dict:
        return {"version": self.policy.version, "gap_sigma": self.policy.gap_sigma, "gap_threshold": self.gap_threshold,
                "n_rows": int(self.n_rows), "n_invalid_rows": int(self.n_invalid_rows),
                "n_failed_seed_preparations": int(self.n_failed_seed_preparations),
                "regions": [r.as_record() for r in self.regions], "unresolved": list(self.unresolved),
                "role": REGION_ROLE}


def cv1_support_intervals(values: np.ndarray, gap_threshold: float) -> List[tuple]:
    """Disconnected supported intervals of a 1-D sample: split where consecutive sorted
    distinct values are further apart than ``gap_threshold``. Returns [(lo, hi), ...]."""
    v = np.unique(np.asarray(values, dtype=np.float64))
    if v.size == 0:
        return []
    breaks = np.where(np.diff(v) > gap_threshold)[0]
    starts = np.r_[0, breaks + 1]
    ends = np.r_[breaks, v.size - 1]
    return [(float(v[s]), float(v[e])) for s, e in zip(starts, ends)]


def build_region_inventory(cv1, *, groups=None, seed_values=None, temperature_k: float, k_max_kcal: float,
                           policy: Optional[RegionPolicy] = None, n_failed_seed_preparations: int = 0) -> RegionInventory:
    """Inventory the supported CV1 intervals of a discovery sample.

    ``cv1`` are per-frame values (non-finite rows are counted as invalid, never binned);
    ``groups`` the seed family per row; ``seed_values`` the CV1 values of the frames that
    have an exported structure (eligible as production seeds; defaults to ``cv1`` itself).
    The representative of a region is the eligible seed nearest the region's median CV1,
    else the median row; ``eligible_seed_rows`` index ``seed_values``.
    """
    from gareus.swarm.ladder_design import window_sigma_cv

    policy = policy or RegionPolicy()
    x = np.asarray(cv1, dtype=np.float64).reshape(-1)
    n = x.size
    finite = np.isfinite(x)
    n_invalid = int((~finite).sum())
    idx = np.where(finite)[0]
    xf = x[idx]
    fam = np.asarray(groups if groups is not None else np.arange(n)).reshape(-1)[idx]
    seeds = np.asarray(seed_values if seed_values is not None else xf, dtype=np.float64).reshape(-1)
    seeds_ok = np.isfinite(seeds)
    gap = float(policy.gap_sigma) * float(window_sigma_cv(float(k_max_kcal), float(temperature_k)))
    regions: List[Region] = []
    for k, (lo, hi) in enumerate(cv1_support_intervals(xf, gap)):
        member_mask = (xf >= lo) & (xf <= hi)
        members = idx[member_mask]
        values = xf[member_mask]
        target = float(np.quantile(values, policy.representative_quantile))
        eligible = [int(i) for i in np.where(seeds_ok & (seeds >= lo) & (seeds <= hi))[0]]
        if eligible:
            rep_i = eligible[int(np.argmin(np.abs(seeds[eligible] - target)))]
            representative, rep_row = float(seeds[rep_i]), int(rep_i)
        else:
            rep_row = int(members[np.argmin(np.abs(values - target))])
            representative = float(x[rep_row])
        regions.append(Region(
            region_id=f"cv1_region_{k:02d}", axis="cv1", lo=lo, hi=hi, representative=representative,
            representative_row=rep_row, n_members=int(members.size), n_families=int(np.unique(fam[member_mask]).size),
            n_distinct_frames=int(np.unique(values).size), eligible_seed_rows=eligible, radius=0.5 * (hi - lo),
            is_rare=bool(members.size < policy.min_members)))
    unresolved = [f"{r.region_id}: no eligible seed structure" for r in regions if not r.eligible_seed_rows]
    return RegionInventory(policy=policy, regions=regions, n_rows=n, n_invalid_rows=n_invalid,
                           n_failed_seed_preparations=int(n_failed_seed_preparations), gap_threshold=gap,
                           unresolved=unresolved, seed_values=seeds)


def region_centres(inventory: RegionInventory, n_windows: int, *, temperature_k: float, k_max_kcal: float,
                   overlap_sigma: float = 1.5) -> Dict[str, object]:
    """CV1 centres that cover every supported interval: each region gets its representative
    plus as many additional uniformly spaced centres as its width supports (proportional
    share of ``n_windows`` by width, at least the representative). Gaps between regions get
    nothing -- bridging them is the unrestrained anchor stack's job, not a seedless window's.
    Returns {"centres", "region_of_centre", "n_requested", "n_placed"}."""
    from gareus.swarm.ladder_design import window_sigma_cv

    regs = inventory.cv1_regions
    if not regs:
        return {"centres": [], "region_of_centre": [], "n_requested": int(n_windows), "n_placed": 0}
    widths = np.array([max(r.hi - r.lo, 0.0) for r in regs])
    total = float(widths.sum())
    sigma_w = float(window_sigma_cv(float(k_max_kcal), float(temperature_k)))
    centres: List[float] = []
    owners: List[str] = []
    remaining = int(n_windows)
    for r, w in zip(regs, widths):
        share = int(round(int(n_windows) * (w / total))) if total > 0 else 0
        resolvable = int(np.floor(w / (overlap_sigma * sigma_w))) if sigma_w > 0 else 0
        k = max(1, min(share, resolvable + 1, remaining if remaining > 0 else 1))
        if k == 1:
            pts = [r.representative]
        else:
            pts = list(np.linspace(r.lo, r.hi, k))
            # keep the representative exactly: replace the nearest grid point
            j = int(np.argmin(np.abs(np.asarray(pts) - r.representative)))
            pts[j] = r.representative
        # Every centre sits on an exported seed structure: snap each grid point to the nearest
        # eligible seed of its region (a window that no seed can start is a pull, not a design).
        if r.eligible_seed_rows and inventory.seed_values is not None:
            pool = np.asarray(inventory.seed_values, dtype=float)[np.asarray(r.eligible_seed_rows, dtype=int)]
            pts = [float(pool[int(np.argmin(np.abs(pool - float(p))))]) for p in pts]
        pts = sorted(set(float(p) for p in pts))
        centres.extend(pts)
        owners.extend([r.region_id] * len(pts))
        remaining -= len(pts)
    order = np.argsort(centres)
    return {"centres": [centres[i] for i in order], "region_of_centre": [owners[i] for i in order],
            "n_requested": int(n_windows), "n_placed": len(centres)}
