"""X3: slow-mode-aware reseeding at epoch boundaries (adaptive-CV2 spec, Section 12).

A walker stuck on one side of a slow motion the deployed pair (CV1, CV2) does not bias stays
there for the whole segment. At an epoch boundary this module finds that motion without any
reference structure -- the leading conditional tICA mode of the torsion residual after BOTH
deployed CVs are removed -- measures, per window, how one-sided the window's samples were
along it, and for the most one-sided windows picks a pooled end configuration that lies
inside the window's restraint AND on the under-sampled side. The next epoch grafts and pulls
that configuration instead of the default seed.

Only starting points change: every numbered epoch already re-grafts and re-pulls every
window from a seed-bank structure, so no Hamiltonian, sample or MBAR input is touched.

Hidden mode, precisely. With the deployed fit's residual R = X - m(a) - mean (the same
regression on CV1 the deployed CV2 uses) and the deployed direction v (unit), the residual
after both CVs is R_perp = R - (R.v) v. Because R is affine in X, passing
X_perp = X - (R.v) v to :func:`fit_conditional_tica` reproduces R_perp as that function's
own residual, so the existing estimator (lagged pairs within one trajectory, pair-centred,
symmetrised) is reused unchanged; C0 is singular along v and the whitening drops it.

Side split. The two sides are split at the density minimum between the two largest peaks of
the (smoothed) pooled histogram of the hidden-mode score when that minimum is real (each side
>= ``SPLIT_MIN_SIDE_MASS`` of the frames, dip <= ``SPLIT_MAX_DIP_RATIO`` of the lower peak);
otherwise at the median. The report records which rule was used.

This module holds the pure math, the planner and the seed-bank overrides; loading frames and
end states from phase directories is in :mod:`gareus.adaptive.slow_mode_reseed_io`.
"""
from __future__ import annotations

import csv
import os
import math
import shutil
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np

#: Sub-directory of a seed bank (and of a filtered seed bank) holding the override rows.
DIR_NAME = "slow_mode_reseed"
#: Per-epoch report, written into ``epoch_NNN/``.
REPORT_NAME = "slow_mode_reseed.json"
#: ``filter_source`` marker carried by every override row.
FILTER_SOURCE = "slow_mode_reseed"
#: Lag of the hidden-mode tICA (the CV selection's slowness lag).
DEFAULT_LAG_PS = 200.0
#: A candidate is inside a window's restraint if every restrained axis is within this many
#: sigma_w = sqrt(RT/k) of the centre. An unrestrained axis (k <= 0) imposes nothing (P6).
ADMISSIBLE_SIGMA = 2.0
#: A window is one-sided when its minority side holds at most this fraction of its frames.
ONE_SIDED_MINORITY_MAX = 0.2
#: Fewest frames a window needs before its side occupancy is trusted.
MIN_FRAMES_PER_STATE = 50
SPLIT_BINS = 80
SPLIT_MIN_SIDE_MASS = 0.05
SPLIT_MAX_DIP_RATIO = 0.8
#: Gas constant in kcal/mol/K (restraint constants are kcal/mol per CV unit squared).
R_KCAL = 0.0019872043


# ---------------------------------------------------------------------------------------
# Hidden mode
# ---------------------------------------------------------------------------------------

def _stripped(fit):
    """The fit's regression and mean with no components: the base for one appended mode."""
    d = int(fit.width)
    return replace(fit, singular_values=np.zeros(0), right_vectors=np.zeros((0, d)),
                   projection_mean=np.zeros(0), projection_std=np.zeros(0),
                   families=(), tica_lag_frames=(), tica_eigenvalues=())


def residual_after_pair(fit, j: int, X, anchor) -> np.ndarray:
    """R_perp: the torsion residual after CV1 (the fit's regression) and CV2 (direction j)."""
    from gareus.cv_selection.slowness import _residual

    R = _residual(fit, X, anchor)
    v = np.asarray(fit.right_vectors[int(j) - 1], dtype=np.float64)
    return R - np.outer(R @ v, v)


@dataclass(frozen=True)
class HiddenMode:
    """The leading conditional tICA mode of the residual after the deployed pair."""

    pair_fit: Any                 # the deployed ResidualFit (from the frozen candidate set)
    deployed_j: int
    direction: np.ndarray         # unit vector in feature space, orthogonal to the CV2 direction
    mean: float                   # score standardisation on the training frames
    std: float
    eigenvalue: float
    lag: int                      # in the frame_index units the fit was given
    split: float                  # side boundary on the standardised score
    split_method: str

    def scores(self, X, anchor) -> np.ndarray:
        R_perp = residual_after_pair(self.pair_fit, self.deployed_j, X, anchor)
        return (R_perp @ self.direction - self.mean) / self.std

    def side(self, z) -> np.ndarray:
        """1 above the split, 0 at or below it."""
        return (np.asarray(z, dtype=np.float64) > self.split).astype(np.int64)


def fit_hidden_mode(pair_fit, deployed_j: int, X, anchor, member_ids, frame_index, *, lag: int,
                    weights=None) -> HiddenMode:
    """Fit the hidden mode (see the module docstring) and its side split."""
    from gareus.cv_selection.slowness import _residual, fit_conditional_tica

    X = np.asarray(X, dtype=np.float64)
    v = np.asarray(pair_fit.right_vectors[int(deployed_j) - 1], dtype=np.float64)
    along_v = np.outer(_residual(pair_fit, X, anchor) @ v, v)
    # residual(X_perp) = R - (R.v) v exactly, because the residual is affine in X
    hf = fit_conditional_tica(_stripped(pair_fit), X - along_v, anchor, member_ids, frame_index,
                              lag=int(lag), n_modes=1, weights=weights)
    direction = np.asarray(hf.right_vectors[0], dtype=np.float64)
    R_perp = residual_after_pair(pair_fit, deployed_j, X, anchor)
    z = (R_perp @ direction - float(hf.projection_mean[0])) / float(hf.projection_std[0])
    split, method = split_point(z)
    return HiddenMode(pair_fit, int(deployed_j), direction, float(hf.projection_mean[0]),
                      float(hf.projection_std[0]), float(hf.tica_eigenvalues[0]), int(lag),
                      float(split), method)


def split_point(z) -> tuple:
    """(boundary, method): the density minimum between the two main peaks, else the median."""
    z = np.asarray(z, dtype=np.float64)
    z = z[np.isfinite(z)]
    median = float(np.median(z))
    lo, hi = np.quantile(z, [0.005, 0.995])
    if not hi > lo:
        return median, "median"
    h, edges = np.histogram(z, bins=SPLIT_BINS, range=(float(lo), float(hi)))
    kernel = np.array([1.0, 2.0, 3.0, 2.0, 1.0]) / 9.0
    s = np.convolve(h.astype(np.float64), kernel, mode="same")
    peaks = [i for i in range(1, s.size - 1) if s[i] >= s[i - 1] and s[i] > s[i + 1]]
    if len(peaks) < 2:
        return median, "median"
    p1, p2 = sorted(sorted(peaks, key=lambda i: -s[i])[:2])
    i_min = p1 + int(np.argmin(s[p1:p2 + 1]))
    boundary = float(0.5 * (edges[i_min] + edges[i_min + 1]))
    left = float(np.mean(z <= boundary))
    if (min(left, 1.0 - left) >= SPLIT_MIN_SIDE_MASS
            and s[i_min] <= SPLIT_MAX_DIP_RATIO * min(s[p1], s[p2])):
        return boundary, "density_minimum"
    return median, "median"


def torsion_loadings(direction, torsion_labels: Sequence[str]) -> List[Dict[str, Any]]:
    """Per torsion: sqrt(w_sin^2 + w_cos^2), largest first (the scree convention)."""
    w = np.asarray(direction, dtype=np.float64).reshape(-1, 2)
    mags = np.sqrt(np.sum(w ** 2, axis=1))
    order = np.argsort(-mags, kind="stable")
    return [{"torsion": str(torsion_labels[i]), "loading": float(mags[i]),
             "w_sin": float(w[i, 0]), "w_cos": float(w[i, 1])} for i in order]


# ---------------------------------------------------------------------------------------
# Occupancy, admissibility, selection
# ---------------------------------------------------------------------------------------

def side_occupancy(sides, state_ids) -> Dict[int, Dict[str, Any]]:
    """state_id -> frame counts per side and the minority side/fraction."""
    sides = np.asarray(sides, dtype=np.int64)
    state_ids = np.asarray(state_ids, dtype=np.int64)
    out: Dict[int, Dict[str, Any]] = {}
    for sid in np.unique(state_ids):
        m = state_ids == sid
        n = int(m.sum())
        n_high = int(sides[m].sum())
        frac_high = n_high / n if n else float("nan")
        out[int(sid)] = {"n_frames": n, "n_low": n - n_high, "n_high": n_high,
                         "frac_high": float(frac_high),
                         "minority_side": 1 if frac_high < 0.5 else 0,
                         "minority_fraction": float(min(frac_high, 1.0 - frac_high))}
    return out


def _axis_terms(cv1: float, cv2: Optional[float], target: Mapping[str, Any], rt_kcal: float):
    """(axis, |d| / sigma_w) for each restrained axis; None when a restrained CV is missing."""
    terms = []
    for axis, value, c_key, k_key in (("cv1", cv1, "primary_center", "primary_k"),
                                      ("cv2", cv2, "secondary_center", "secondary_k")):
        k = target.get(k_key)
        c = target.get(c_key)
        if k is None or c is None or not float(k) > 0.0:
            continue
        if value is None or not math.isfinite(float(value)):
            return None
        sigma = math.sqrt(rt_kcal / float(k))
        terms.append((axis, abs(float(value) - float(c)) / sigma))
    return terms


def restraint_distance(cv1: float, cv2: Optional[float], target: Mapping[str, Any],
                       rt_kcal: float) -> float:
    """Largest |d|/sigma_w over the target's restrained axes (0 if none is restrained)."""
    terms = _axis_terms(cv1, cv2, target, rt_kcal)
    if terms is None:
        return float("inf")
    return max((t for _a, t in terms), default=0.0)


def admissible(cv1: float, cv2: Optional[float], target: Mapping[str, Any], rt_kcal: float,
               n_sigma: float = ADMISSIBLE_SIGMA) -> bool:
    return restraint_distance(cv1, cv2, target, rt_kcal) <= float(n_sigma)


def _n_target(fraction: float, n_windows: int) -> int:
    if fraction <= 0.0 or n_windows <= 0:
        return 0
    return max(1, int(math.floor(float(fraction) * int(n_windows) + 1e-9)))


def plan_reseed(targets: Sequence[Mapping[str, Any]], occupancy: Mapping[int, Mapping[str, Any]],
                candidates: Sequence[Mapping[str, Any]], *, fraction: float, rt_kcal: float,
                n_sigma: float = ADMISSIBLE_SIGMA,
                one_sided_max: float = ONE_SIDED_MINORITY_MAX,
                min_frames: int = MIN_FRAMES_PER_STATE,
                default_sides: Optional[Mapping[int, int]] = None) -> Dict[str, Any]:
    """Choose which windows to reseed and from which end state. Deterministic.

    ``targets``: the next epoch's windows (state_id, primary/secondary centre and k).
    ``candidates``: pooled end states (path, cv1, cv2, side, hidden_z, source_state_id).
    A window is eligible when it has >= ``min_frames`` frames and its minority side holds
    <= ``one_sided_max`` of them; eligible windows are ranked most one-sided first
    (minority fraction, then more frames, then state_id), and up to
    floor(fraction x n_windows) of them are reseeded, each from the admissible
    minority-side candidate nearest its centre (in sigma_w), preferring candidates not yet
    used. A window with no such candidate, or whose default seed (``default_sides``: state ->
    side of the seed today's assignment gives it) is already on its minority side, is
    skipped and the next one takes its slot.
    """
    n_target = _n_target(fraction, len(targets))
    rows: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    eligible = []
    for t in targets:
        sid = int(t["state_id"])
        occ = occupancy.get(sid)
        if occ is None or int(occ["n_frames"]) < int(min_frames):
            skipped.append({"state_id": sid, "reason": "too_few_frames",
                            "n_frames": 0 if occ is None else int(occ["n_frames"])})
            continue
        if float(occ["minority_fraction"]) > float(one_sided_max):
            continue
        eligible.append(t)
    eligible.sort(key=lambda t: (float(occupancy[int(t["state_id"])]["minority_fraction"]),
                                 -int(occupancy[int(t["state_id"])]["n_frames"]), int(t["state_id"])))
    used: Dict[str, int] = {}
    for t in eligible:
        if len(rows) >= n_target:
            break
        sid = int(t["state_id"])
        want = int(occupancy[sid]["minority_side"])
        if default_sides is not None and default_sides.get(sid) == want:
            skipped.append({"state_id": sid, "reason": "default_seed_already_on_minority_side",
                            "minority_side": want})
            continue
        ranked = []
        for c in candidates:
            if int(c["side"]) != want:
                continue
            dist = restraint_distance(float(c["cv1"]), c.get("cv2"), t, rt_kcal)
            if dist <= float(n_sigma):
                ranked.append((used.get(str(c["path"]), 0), dist, str(c["path"]), c))
        if not ranked:
            skipped.append({"state_id": sid, "reason": "no_admissible_minority_side_candidate",
                            "minority_side": want,
                            "minority_fraction": float(occupancy[sid]["minority_fraction"])})
            continue
        ranked.sort(key=lambda r: (r[0], r[1], r[2]))
        _u, dist, path, c = ranked[0]
        used[path] = used.get(path, 0) + 1
        rows.append({
            "target_state_id": sid,
            "target_primary_center": t.get("primary_center"), "target_primary_k": t.get("primary_k"),
            "target_secondary_center": t.get("secondary_center"), "target_secondary_k": t.get("secondary_k"),
            "target_gamd_lambda": t.get("gamd_lambda"),
            "minority_side": want,
            "minority_fraction_before": float(occupancy[sid]["minority_fraction"]),
            "frac_high_before": float(occupancy[sid]["frac_high"]),
            "n_frames": int(occupancy[sid]["n_frames"]),
            "seed_path": path, "seed_source_state_id": c.get("source_state_id"),
            "seed_source_phase": c.get("source_phase"),
            "seed_cv1": float(c["cv1"]), "seed_cv2": None if c.get("cv2") is None else float(c["cv2"]),
            "seed_hidden_z": float(c["hidden_z"]), "seed_side": int(c["side"]),
            "restraint_distance_sigma": float(dist),
            "seed_from_other_state": c.get("source_state_id") is not None and int(c["source_state_id"]) != sid,
            "seed_reuse_index": int(_u),
        })
    n_one_sided = sum(1 for t in targets
                      if int(t["state_id"]) in occupancy
                      and int(occupancy[int(t["state_id"])]["n_frames"]) >= int(min_frames)
                      and float(occupancy[int(t["state_id"])]["minority_fraction"]) <= float(one_sided_max))
    return {"n_windows": len(targets), "n_target": n_target, "n_one_sided": int(n_one_sided),
            "n_eligible": len(eligible), "n_reseeded": len(rows), "reseeded": rows, "skipped": skipped,
            "rules": {"fraction": float(fraction), "admissible_sigma": float(n_sigma),
                      "one_sided_minority_max": float(one_sided_max), "min_frames_per_state": int(min_frames)}}


# ---------------------------------------------------------------------------------------
# Seed-bank overrides
# ---------------------------------------------------------------------------------------

_OVERRIDE_FIELDS = [
    "seed_name", "survivor_pdb_path", "source_pdb_path", "target_state_id",
    "target_primary_center", "target_primary_k", "target_secondary_center", "target_secondary_k",
    "source_state_id", "primary_cv_value", "secondary_cv_value", "hidden_mode_value", "hidden_mode_side",
    "filter_source",
]


def clear_overrides(seed_bank_dir) -> None:
    """Remove a bank's override directory (so seeding proceeds exactly as without X3)."""
    d = Path(seed_bank_dir) / DIR_NAME
    if d.is_dir():
        shutil.rmtree(d)


def _write_rows(out: Path, rows: List[Dict[str, Any]]) -> None:
    with (out / "final_survivor_seeds.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_OVERRIDE_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_overrides(seed_bank_dir, reseeded: Sequence[Mapping[str, Any]]) -> Path:
    """Copy each chosen end state into ``<bank>/slow_mode_reseed/pdbs/`` under a new name
    and write the override table (a GENPEPT-style ``final_survivor_seeds.csv``)."""
    out = Path(seed_bank_dir) / DIR_NAME
    clear_overrides(seed_bank_dir)
    (out / "pdbs").mkdir(parents=True)
    rows = []
    for k, r in enumerate(reseeded):
        name = f"x3_state_{int(r['target_state_id']):04d}_{k:03d}"
        dst = out / "pdbs" / f"{name}.pdb"
        shutil.copy2(r["seed_path"], dst)          # never a link: source dirs may be rewritten
        rows.append({
            "seed_name": name, "survivor_pdb_path": str(Path("pdbs") / dst.name),
            "source_pdb_path": str(r["seed_path"]), "target_state_id": int(r["target_state_id"]),
            "target_primary_center": r.get("target_primary_center"), "target_primary_k": r.get("target_primary_k"),
            "target_secondary_center": "" if r.get("target_secondary_center") is None else r["target_secondary_center"],
            "target_secondary_k": "" if r.get("target_secondary_k") is None else r["target_secondary_k"],
            "source_state_id": "" if r.get("seed_source_state_id") is None else r["seed_source_state_id"],
            "primary_cv_value": r.get("seed_cv1"),
            "secondary_cv_value": "" if r.get("seed_cv2") is None else r["seed_cv2"],
            "hidden_mode_value": r.get("seed_hidden_z"), "hidden_mode_side": r.get("seed_side"),
            "filter_source": FILTER_SOURCE,
        })
    _write_rows(out, rows)
    return out


def _read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def filter_overrides(seed_bank_dir, target_state_ids: Iterable[int], output_dir) -> int:
    """Carry the bank's override rows for ``target_state_ids`` into a filtered bank.

    Returns the number of rows written; 0 (and nothing written) when the bank has none.
    PDBs are hard-linked (the bank's copies are never modified after they are written).
    """
    src = Path(seed_bank_dir) / DIR_NAME
    table = src / "final_survivor_seeds.csv"
    if not table.exists():
        return 0
    wanted = {int(s) for s in target_state_ids}
    rows = [r for r in _read_rows(table) if _int(r.get("target_state_id")) in wanted]
    out = Path(output_dir) / DIR_NAME
    if out.is_dir():
        shutil.rmtree(out)
    if not rows:
        return 0
    (out / "pdbs").mkdir(parents=True)
    for r in rows:
        name = Path(r["survivor_pdb_path"]).name
        try:
            os.link(src / "pdbs" / name, out / "pdbs" / name)
        except OSError:
            shutil.copy2(src / "pdbs" / name, out / "pdbs" / name)
    _write_rows(out, rows)
    return len(rows)


def _int(v) -> Optional[int]:
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def _float(v) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def reseed_conformers_by_window(seed_dir, out_dir, primary_centers, secondary_centers,
                                load_library: Callable[[Path], List[dict]]) -> Dict[int, dict]:
    """window -> conformer entry for the windows X3 reseeds; ``{}`` when the bank has none.

    Windows are resolved to states through ``out_dir``'s own ``epoch_window_map.csv``
    (already compacted for a pre-pull drop at this point of run_gareus). An override whose
    recorded restraint centre does not match its window's is ignored with a warning: the
    bookkeeping disagrees, so the window keeps today's seed.
    """
    table = Path(seed_dir) / DIR_NAME / "final_survivor_seeds.csv"
    if not table.exists():
        return {}
    from gareus.topup_seeding import state_id_of_window_from_epoch_map

    state_of_window = state_id_of_window_from_epoch_map(out_dir)
    if not state_of_window:
        print(f"WARNING [slow-mode reseed]: {out_dir} has no epoch_window_map.csv; overrides ignored")
        return {}
    entries = load_library(Path(seed_dir) / DIR_NAME)
    by_state: Dict[int, dict] = {}
    for e in sorted(entries, key=lambda e: str(e.get("pdb_path", ""))):
        sid = _int((e.get("source_row") or {}).get("target_state_id"))
        if sid is not None and sid not in by_state:
            by_state[sid] = e
    out: Dict[int, dict] = {}
    for w, sid in sorted(state_of_window.items()):
        e = by_state.get(int(sid))
        if e is None or w >= len(primary_centers):
            continue
        row = e.get("source_row") or {}
        pc, sc = _float(row.get("target_primary_center")), _float(row.get("target_secondary_center"))
        ok = pc is not None and abs(pc - float(primary_centers[w])) <= 1e-6 * max(1.0, abs(pc))
        if ok and sc is not None and secondary_centers is not None:
            ok = abs(sc - float(secondary_centers[w])) <= 1e-6 * max(1.0, abs(sc))
        if not ok:
            print(f"WARNING [slow-mode reseed]: override for state {sid} does not match window {w}'s "
                  "restraint centre; that window keeps its default seed")
            continue
        out[int(w)] = e
    if out:
        print(f"    Slow-mode reseed: {len(out)} window(s) start from an X3-chosen end state")
    return out
