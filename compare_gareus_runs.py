#!/usr/bin/env python3
"""
Multi-run ATLaS-MD PMF comparison.

Loads N ATLaS-MD run directories, runs MBAR on each independently, then
optionally computes a combined "united MBAR" PMF by pooling all samples
against the full global state space.  Produces a multi-panel comparison
figure and per-run PMF CSVs.

Usage
-----
    python compare_gareus_runs.py RUN1 RUN2 ... [options]
    python compare_gareus_runs.py RUN1 RUN2 --labels "run A" "run B" --out figs/
    python compare_gareus_runs.py RUN1 RUN2 --no-united   # skip combined MBAR
    python compare_gareus_runs.py RUN1 RUN2 --2d          # include 2D FES grid

Compatibility guard
-------------------
United MBAR is valid only when all runs share the same CV definition,
CV units, and temperature.  If any mismatch is detected the script prints
a warning, skips the united PMF, and still produces the per-run overlay.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import sys
from pathlib import Path
from typing import Optional

import numpy as np

# The stale-`epoch_window_map.csv` guard, imported from the package rather
# than re-implemented here. `load_npz_adaptive_union` below is a second,
# parallel union-MBAR implementation, so it inherits the same auto-drop
# mis-attribution bug as the Parquet loader and must fail closed the same way
# (docs/chignolin_6_low_ess_root_cause.md). Imported via
# gareus.mbar_analysis.loaders -- the module that owns the shared entry points
# for every non-Parquet consumer of that file -- so this script has one import
# surface for the whole guard family.
from gareus.mbar_analysis.loaders import (_validate_and_repair_epoch_window_map,
                                          check_union_npz_window_map_provenance)

# ---------------------------------------------------------------------------
# Import key pieces from analyze_gareus_mbar.py (same repo root)
# ---------------------------------------------------------------------------

def _load_agm():
    here = Path(__file__).resolve().parent
    candidates = [here / 'analyze_gareus_mbar.py']
    for c in candidates:
        if c.exists():
            spec = importlib.util.spec_from_file_location('_agm', c)
            mod = importlib.util.module_from_spec(spec)
            sys.modules['_agm'] = mod  # must be registered before exec so @dataclass works
            spec.loader.exec_module(mod)   # type: ignore[union-attr]
            return mod
    raise ImportError('analyze_gareus_mbar.py not found next to this script')

agm = _load_agm()

K_B_KJ = agm.K_B_KJ_PER_MOL_K
KJ_PER_KCAL = agm.KJ_PER_KCAL

# ---------------------------------------------------------------------------
# Secondary CV parameter extraction
# ---------------------------------------------------------------------------

def _sec_params(d: agm.Data):
    """Return (sec_centers, sec_ks) arrays aligned with d.centers."""
    K = d.centers.size
    sec_c = np.full(K, np.nan)
    sec_k = np.zeros(K)
    rows = d.meta.get('umbrella_window_rows', [])
    for i, row in enumerate(rows[:K]):
        for ck in ('secondary_center', 'secondary_cv_center', 'secondary'):
            v = row.get(ck, '')
            if v not in ('', 'None', 'nan', None):
                try:
                    sec_c[i] = float(v)
                    break
                except Exception:
                    pass
        for kk in ('secondary_k', 'secondary_cv_k_kcal_mol', 'secondary_k_kcal_mol'):
            v = row.get(kk, '')
            if v not in ('', 'None', 'nan', None):
                try:
                    val = float(v)
                    if val > 0:
                        sec_k[i] = val
                        break
                except Exception:
                    pass
    return sec_c, sec_k


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _find_npz_epoch_dirs(adaptive_dir: Path) -> list[tuple[Path, Path]]:
    """Return (npz_dir, wmap_path) for each epoch's NPZ analysis data.

    Handles both flat (epoch_NNN/analysis_arrays.npz) and nested
    (epoch_NNN/topup_NNN/analysis_arrays.npz) layouts.
    """
    result = []
    for epoch_dir in sorted(adaptive_dir.iterdir()):
        if not epoch_dir.is_dir() or not epoch_dir.name.startswith('epoch_'):
            continue
        # Flat layout: epoch_NNN/analysis_arrays.npz
        if (epoch_dir / 'analysis_arrays.npz').exists() or \
                (epoch_dir / 'analysis_chunks_manifest.json').exists():
            wmap = epoch_dir / 'epoch_window_map.csv'
            if wmap.exists():
                result.append((epoch_dir, wmap))
            continue
        # Nested layout: epoch_NNN/<subdir>/analysis_arrays.npz
        for sub in sorted(epoch_dir.iterdir()):
            if sub.is_dir() and ((sub / 'analysis_arrays.npz').exists() or
                                  (sub / 'analysis_chunks_manifest.json').exists()):
                wmap = sub / 'epoch_window_map.csv'
                if not wmap.exists():
                    wmap = epoch_dir / 'epoch_window_map.csv'
                if wmap.exists():
                    result.append((sub, wmap))
    return result


def load_npz_adaptive_union(adaptive_dir: Path) -> agm.Data:
    """Pool all epoch NPZ data from an NPZ-based adaptive run.

    Mirrors load_parquet_adaptive_union but reads analysis_arrays.npz
    per epoch instead of Parquet files.  Requires final_registry_used_for_mbar.csv
    and per-epoch epoch_window_map.csv for window remapping.
    """
    registry_csv = adaptive_dir / 'final_registry_used_for_mbar.csv'
    if not registry_csv.exists():
        raise FileNotFoundError(f'No final_registry_used_for_mbar.csv in {adaptive_dir}')

    with registry_csv.open(newline='') as f:
        reg_rows = [r for r in csv.DictReader(f)
                    if str(r.get('usable_for_mbar', '')).strip().lower() in ('true', '1', 'yes')]
    if not reg_rows:
        raise ValueError(f'No usable states in {registry_csv}')
    reg_rows.sort(key=lambda r: int(r['state_id']))

    state_ids = [int(r['state_id']) for r in reg_rows]
    state_id_to_k = {sid: k for k, sid in enumerate(state_ids)}
    K = len(state_ids)

    primary_centers = np.array([float(r['primary_center']) for r in reg_rows])
    primary_ks = np.array([float(r['primary_k']) for r in reg_rows])
    sec_centers = np.array([
        float(r['secondary_center']) if r.get('secondary_center', '') not in ('', 'None', 'nan') else np.nan
        for r in reg_rows])
    sec_ks = np.array([
        float(r['secondary_k']) if r.get('secondary_k', '') not in ('', 'None', 'nan') else 0.0
        for r in reg_rows])
    has_secondary = np.any(np.isfinite(sec_centers))

    epoch_dirs = _find_npz_epoch_dirs(adaptive_dir)
    if not epoch_dirs:
        raise FileNotFoundError(f'No epoch NPZ data found in {adaptive_dir}')

    all_cv = []; all_cv2 = []; all_window = []; all_step = []
    all_replica = []; all_boost = []; all_potential = []
    beta = float('nan')
    load_notes: list[str] = []
    meta: dict = agm.rjson(adaptive_dir.parent / 'run_manifest.json', {})

    for npz_dir, wmap_path in epoch_dirs:
        arr, _ = agm._load_merged_arrays(npz_dir)
        if not arr or 'cv_A' not in arr.files:
            continue

        def _get_first(*keys):
            for key in keys:
                v = arr.get(key)
                if v is not None:
                    return v
            return None

        ep_meta = agm.rjson(npz_dir / 'umbrella_pymbar_metadata.json', {})
        if not math.isfinite(beta):
            b = float(ep_meta.get('beta_1_over_kJ_mol') or 0.0)
            beta = b if b > 0 else agm.infer_temp_beta(npz_dir, ep_meta)[1]

        raw_w = np.asarray(arr['window'], dtype=np.int32)
        cv2_raw = _get_first('cv2_A', 'secondary_cv')
        cv2_full = np.asarray(cv2_raw, dtype=np.float64) if cv2_raw is not None else None

        # The map used to be trusted verbatim. It can be STALE: an
        # `--us-auto-drop-bad-windows` phase renumbers its surviving windows
        # 0..N-1 without rewriting the identity map the registry already wrote
        # over all *active* states, so every local index at or after the first
        # dropped one resolves to the wrong state. Validate/repair before the
        # lookup is built, and fail closed when neither is possible -- this
        # loader feeds comparison PMFs, so a silently shifted mapping here is
        # exactly as damaging as it is on the Parquet path.
        #
        # `npz_dir`, not `wmap_path.parent`: _find_npz_epoch_dirs falls back to
        # the parent epoch's map for a nested sub-run, but the metadata saying
        # what actually ran (and the drop record) belongs to the sub-run whose
        # samples these are. That inherited-map shape is precisely the case the
        # guard was built for (chignolin_6's epoch_001/topup_004: 3 map rows,
        # 2 real windows, drop record inherited from a sibling baseline).
        with wmap_path.open(newline='') as f:
            wmap_rows = list(csv.DictReader(f))
        wmap_rows, wmap_notes = _validate_and_repair_epoch_window_map(
            npz_dir, wmap_rows, window_ids=raw_w, cv2=cv2_full)
        for _n in wmap_notes:
            print(f'    {_n}')
        load_notes.extend(wmap_notes)
        wmap = {int(r['epoch_window']): int(r['state_id']) for r in wmap_rows}

        remapped = np.array([wmap.get(int(w), -1) for w in raw_w], dtype=np.int32)
        valid = remapped >= 0
        if not np.any(valid):
            continue

        cv = np.asarray(arr['cv_A'], dtype=np.float64)[valid]
        all_cv.append(cv)

        all_cv2.append(cv2_full[valid]
                       if cv2_full is not None else np.full(int(valid.sum()), np.nan))

        all_window.append(np.array([state_id_to_k[int(s)] for s in remapped[valid]], dtype=np.int32))
        step_arr = arr.get('step')
        all_step.append(np.asarray(step_arr, dtype=np.int64)[valid]
                        if step_arr is not None else np.arange(int(valid.sum()), dtype=np.int64))
        rep_arr = arr.get('replica')
        all_replica.append(np.asarray(rep_arr, dtype=np.int32)[valid]
                           if rep_arr is not None else np.zeros(int(valid.sum()), dtype=np.int32))
        boost_arr = _get_first('gamd_boost_total_kj_mol', 'gamd_boost_kj_mol', 'boost_kj_mol')
        all_boost.append(np.asarray(boost_arr, dtype=np.float64)[valid]
                         if boost_arr is not None else np.full(int(valid.sum()), np.nan))
        pot_arr = _get_first('potential_kj_mol', 'potential_energy_kj_mol')
        all_potential.append(np.asarray(pot_arr, dtype=np.float64)[valid]
                             if pot_arr is not None else np.full(int(valid.sum()), np.nan))

    if not all_cv:
        raise ValueError(f'No valid samples after window remapping in {adaptive_dir}')

    cv = np.concatenate(all_cv)
    cv2 = np.concatenate(all_cv2)
    window = np.concatenate(all_window)
    step = np.concatenate(all_step)
    replica = np.concatenate(all_replica)
    boost = np.concatenate(all_boost)
    pot_all = np.concatenate(all_potential)
    potential = pot_all if np.any(np.isfinite(pot_all)) else None

    if not math.isfinite(beta):
        _, beta = agm.infer_temp_beta(adaptive_dir, meta)
    temp = 1.0 / (K_B_KJ * beta)

    N = len(cv)
    u_nk = np.zeros((N, K), dtype=np.float64)
    scale = beta * KJ_PER_KCAL
    for k in range(K):
        d1 = cv - primary_centers[k]
        u_nk[:, k] = scale * 0.5 * primary_ks[k] * d1 * d1
        if has_secondary and math.isfinite(sec_centers[k]) and sec_ks[k] > 0:
            d2 = np.where(np.isfinite(cv2), cv2 - sec_centers[k], 0.0)
            u_nk[:, k] += scale * 0.5 * sec_ks[k] * d2 * d2

    meta_out = dict(meta)
    meta_out.update({
        'temperature_K': temp, 'beta_1_over_kJ_mol': beta,
        'adaptive_union_states': K,
        'primary_cv': ep_meta.get('primary_cv', ''),
        'primary_cv_label': ep_meta.get('primary_cv_label', ''),
        'primary_cv_units': ep_meta.get('primary_cv_units', ''),
        'umbrella_window_rows': list(reg_rows),
    })
    if load_notes:
        meta_out['load_notes'] = list(meta.get('load_notes') or []) + load_notes
    return agm.clean(agm.Data(
        prod_dir=adaptive_dir, out_dir=adaptive_dir / 'pmf_analysis',
        cv=cv, cv2=cv2, rg_A=np.full(cv.shape, np.nan),
        window=window, replica=replica, step=step,
        u_nk=u_nk, centers=primary_centers, k_kcal=primary_ks,
        beta=beta, temp=temp, boost_kj=boost, potential_kj=potential,
        source=str(registry_csv), meta=meta_out,
    ))


def load_run(path: str) -> agm.Data:
    p = Path(path).resolve()
    # Try standard prod_dir_of first
    try:
        prod = agm.prod_dir_of(p)
    except FileNotFoundError:
        prod = None

    if prod is not None:
        if (prod / 'final_registry_used_for_mbar.csv').exists():
            return agm.load_parquet_adaptive_union(prod)
        if (prod / 'adaptive_union_mbar.npz').exists():
            # Same provenance check load_data does on this branch: the npz's
            # state attribution was baked in by the driver from the phase
            # window maps and cannot be repaired from the npz, so the maps it
            # was built from are checked instead (refuses on a stale one,
            # warns when provenance cannot be established, silent otherwise).
            for _n in check_union_npz_window_map_provenance(prod):
                print(f'    {_n}')
            return agm.load_union_npz(prod)
        if (prod / 'analysis_arrays.npz').exists() or (prod / 'analysis_chunks_manifest.json').exists():
            return agm.load_npz(prod)
        return agm.load_csv(prod)

    # Fallback: NPZ-based adaptive run (epoch dirs with epoch_window_map.csv)
    ap = p / 'adaptive_production'
    if (ap / 'final_registry_used_for_mbar.csv').exists():
        return load_npz_adaptive_union(ap)

    raise FileNotFoundError(
        f'Cannot determine data source for {p}. '
        f'Expected adaptive_production/final_registry_used_for_mbar.csv or '
        f'final_production/analysis_arrays.npz or samples.csv.'
    )


def _cv_key(d: agm.Data) -> str:
    """String describing CV type+units for compatibility comparison."""
    meta = d.meta or {}
    cv = str(meta.get('primary_cv', '') or meta.get('cv_label', '') or 'unknown').strip().lower()
    units = str(meta.get('primary_cv_units', '') or '').strip().lower()
    return f'{cv}|{units}'


def _check_compatibility(datasets: list[agm.Data], labels: list[str]) -> tuple[bool, str]:
    """
    Return (compatible, reason_if_not).
    Checks: same CV type, same temperature (beta within 0.5%).
    """
    ref_cv = _cv_key(datasets[0])
    ref_beta = datasets[0].beta
    mismatches = []
    for i, d in enumerate(datasets[1:], 1):
        cv_i = _cv_key(d)
        if cv_i != ref_cv and 'unknown' not in (cv_i, ref_cv):
            mismatches.append(
                f'{labels[i]} CV={cv_i!r} differs from {labels[0]} CV={ref_cv!r}')
        rel = abs(d.beta - ref_beta) / (ref_beta + 1e-30)
        if rel > 0.005:
            T0 = 1.0 / (K_B_KJ * ref_beta)
            Ti = 1.0 / (K_B_KJ * d.beta)
            mismatches.append(
                f'{labels[i]} T={Ti:.1f} K differs from {labels[0]} T={T0:.1f} K (>0.5%)')
    if mismatches:
        return False, '; '.join(mismatches)
    return True, ''


# ---------------------------------------------------------------------------
# MBAR weights → PMF
# ---------------------------------------------------------------------------

def _run_mbar(d: agm.Data, tol: float = 1e-10, backend: str = 'sambar') -> dict:
    """Run MBAR for a single run; return result dict from solve_mbar."""
    return agm.solve_mbar(d.u_nk, d.window, tol=tol, backend=backend)


def _pmf_1d(cv: np.ndarray, logw: np.ndarray, bins: np.ndarray, beta: float) -> dict:
    kbt = (1.0 / beta) / KJ_PER_KCAL
    w = np.exp(logw - logw.max())
    return agm.pmf_from_weights(cv, w, bins, kbt)


def _pmf_2d(cv: np.ndarray, cv2: np.ndarray, logw: np.ndarray,
            bins1: np.ndarray, bins2: np.ndarray, beta: float) -> dict:
    kbt = (1.0 / beta) / KJ_PER_KCAL
    w = np.exp(logw - logw.max())
    return agm.pmf2d_from_weights(cv, cv2, w, bins1, bins2, kbt)


# ---------------------------------------------------------------------------
# United MBAR: pool all runs with global u_nk
# ---------------------------------------------------------------------------

def united_mbar(datasets: list[agm.Data], tol: float = 1e-10,
                backend: str = 'sambar') -> tuple[np.ndarray, np.ndarray, np.ndarray, float, dict]:
    """
    Pool samples from all runs, rebuild full N×K_global bias matrix, solve MBAR.

    Returns (cv_all, cv2_all, logw, beta, mbar_result).
    """
    K_sizes = [d.centers.size for d in datasets]
    offsets = np.concatenate([[0], np.cumsum(K_sizes[:-1])]).astype(int)
    K_global = sum(K_sizes)

    # Global window parameters
    global_c = np.concatenate([d.centers for d in datasets])
    global_k = np.concatenate([d.k_kcal for d in datasets])
    global_sc = np.concatenate([_sec_params(d)[0] for d in datasets])
    global_sk = np.concatenate([_sec_params(d)[1] for d in datasets])
    has_secondary = np.any(np.isfinite(global_sc) & (global_sk > 0))

    # Pool samples
    cv_all = np.concatenate([d.cv for d in datasets])
    cv2_all = np.concatenate([d.cv2 for d in datasets])
    win_all = np.concatenate([d.window + offsets[i] for i, d in enumerate(datasets)])

    beta = datasets[0].beta
    scale = beta * KJ_PER_KCAL
    N = len(cv_all)

    print(f'  Building global u_nk: {N:,} samples × {K_global} states ...', flush=True)
    u_global = np.empty((N, K_global), dtype=np.float64)
    for k in range(K_global):
        d1 = cv_all - global_c[k]
        u_global[:, k] = scale * 0.5 * global_k[k] * d1 * d1
        if has_secondary and np.isfinite(global_sc[k]) and global_sk[k] > 0:
            d2 = np.where(np.isfinite(cv2_all), cv2_all - global_sc[k], 0.0)
            u_global[:, k] += scale * 0.5 * global_sk[k] * d2 * d2

    print(f'  Solving united MBAR ({backend}) ...', flush=True)
    mb = agm.solve_mbar(u_global, win_all, tol=tol, backend=backend)
    return cv_all, cv2_all, mb['logw'], beta, mb


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _make_palette(n: int):
    """Return n visually distinct hex colors."""
    import matplotlib
    name = 'tab10' if n <= 10 else 'tab20'
    try:
        cmap = matplotlib.colormaps[name]
    except (AttributeError, KeyError):
        import matplotlib.cm as cm
        cmap = cm.get_cmap(name)
    return [cmap(i / max(1, n - 1)) for i in range(n)]


def _smooth(arr: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return arr
    try:
        from scipy.ndimage import gaussian_filter1d
        return gaussian_filter1d(arr, sigma=sigma)
    except Exception:
        return arr


def plot_comparison(
    datasets: list[agm.Data],
    labels: list[str],
    pmfs: list[dict],
    united_pmf: Optional[dict],
    united_cv: Optional[np.ndarray],
    out_dir: Path,
    cv_label: str,
    smooth_sigma: float = 0.0,
    do_2d: bool = False,
    pmfs_2d: Optional[list[dict]] = None,
    united_pmf_2d: Optional[dict] = None,
    cv2_label: str = 'Secondary CV',
) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    N = len(datasets)
    colors = _make_palette(N)
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        'figure.dpi': 150,
        'font.size': 10,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.linewidth': 0.8,
        'legend.framealpha': 0.85,
        'legend.fontsize': 8,
    })

    has_united = united_pmf is not None
    has_2d = do_2d and pmfs_2d is not None

    # ------------------------------------------------------------------
    # Figure 1: 1D PMF comparison
    # ------------------------------------------------------------------
    n_rows = 3 if has_united else 2
    fig = plt.figure(figsize=(9, 3.5 * n_rows))
    gs = gridspec.GridSpec(n_rows, 1, hspace=0.45)

    # --- row 0: per-run PMFs (+ united if available) ---
    ax0 = fig.add_subplot(gs[0])
    for i, (d, pmf, label, col) in enumerate(zip(datasets, pmfs, labels, colors)):
        x = pmf['cv_A']
        y = _smooth(np.where(np.isfinite(pmf['pmf']), pmf['pmf'], np.nan), smooth_sigma)
        n_samples = len(d.cv)
        ax0.plot(x, y, color=col, lw=1.8, label=f'{label} ({n_samples:,} samples)')
    if has_united:
        xu = united_pmf['cv_A']
        yu = _smooth(np.where(np.isfinite(united_pmf['pmf']), united_pmf['pmf'], np.nan), smooth_sigma)
        ax0.plot(xu, yu, 'k-', lw=2.5, label=f'United MBAR ({len(united_cv):,} total)', zorder=5)
    ax0.set_xlabel(cv_label)
    ax0.set_ylabel('PMF (kcal mol⁻¹)')
    ax0.set_title('PMF comparison — per-run MBAR', fontweight='bold')
    ax0.legend(loc='best', ncol=min(N + 1, 3))
    ax0.set_ylim(bottom=0)

    # --- row 1: window coverage ---
    ax1 = fig.add_subplot(gs[1])
    y_ticks = []
    for i, (d, label, col) in enumerate(zip(datasets, labels, colors)):
        yi = i * np.ones_like(d.centers)
        sec_c, _ = _sec_params(d)
        has_sec = np.any(np.isfinite(sec_c))
        ax1.scatter(d.centers, yi, s=40, color=col, alpha=0.85,
                    label=label, zorder=3)
        ax1.plot(d.centers, yi, color=col, lw=0.5, alpha=0.4)
        y_ticks.append((i, f'{label} (K={d.centers.size})'))
        if has_sec:
            # secondary CV coverage shown as colour gradient along y = secondary
            sc_finite = sec_c[np.isfinite(sec_c)]
            if sc_finite.size:
                ax1.scatter(d.centers[np.isfinite(sec_c)], sec_c[np.isfinite(sec_c)],
                            s=18, color=col, alpha=0.35, marker='x')
    ax1.set_xlabel(cv_label)
    ax1.set_ylabel('Run index')
    ax1.set_title('Window center coverage per run', fontweight='bold')
    ax1.set_yticks([v for v, _ in y_ticks])
    ax1.set_yticklabels([l for _, l in y_ticks], fontsize=8)

    # --- row 2 (if has_united): ΔPMF vs united ---
    if has_united:
        ax2 = fig.add_subplot(gs[2])
        xu = united_pmf['cv_A']
        yu = np.where(np.isfinite(united_pmf['pmf']), united_pmf['pmf'], np.nan)
        for i, (pmf, label, col) in enumerate(zip(pmfs, labels, colors)):
            # interpolate per-run PMF onto united bins for fair diff
            y_interp = np.interp(xu, pmf['cv_A'], np.where(np.isfinite(pmf['pmf']), pmf['pmf'], np.nan),
                                 left=np.nan, right=np.nan)
            delta = y_interp - yu
            ax2.plot(xu, _smooth(delta, smooth_sigma), color=col, lw=1.6, label=label)
        ax2.axhline(0, color='k', lw=0.8, ls='--')
        ax2.set_xlabel(cv_label)
        ax2.set_ylabel('ΔPMF vs united (kcal mol⁻¹)')
        ax2.set_title('Per-run PMF deviation from united MBAR', fontweight='bold')
        ax2.legend(loc='best', ncol=min(N, 3))

    out_path = out_dir / 'pmf_comparison_1d.png'
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {out_path}')

    # ------------------------------------------------------------------
    # Figure 2: sample-count bar chart
    # ------------------------------------------------------------------
    fig2, ax = plt.subplots(figsize=(max(5, N * 1.2 + 1.5), 3.5))
    counts = [len(d.cv) for d in datasets]
    bars = ax.bar(range(N), counts, color=colors, edgecolor='white', linewidth=0.5)
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(counts) * 0.01,
                f'{count:,}', ha='center', va='bottom', fontsize=8)
    if has_united:
        ax.axhline(sum(counts), color='k', lw=1.2, ls='--',
                   label=f'Total united: {sum(counts):,}')
        ax.legend()
    ax.set_xticks(range(N))
    ax.set_xticklabels(labels, fontsize=9, rotation=20, ha='right')
    ax.set_ylabel('Number of samples')
    ax.set_title('Sample counts per run', fontweight='bold')
    out2 = out_dir / 'sample_counts.png'
    fig2.savefig(out2, bbox_inches='tight')
    plt.close(fig2)
    print(f'  Saved {out2}')

    # ------------------------------------------------------------------
    # Figure 3 (optional): 2D FES grid
    # ------------------------------------------------------------------
    if has_2d and pmfs_2d:
        n_panels = N + (1 if united_pmf_2d is not None else 0)
        ncols = min(3, n_panels)
        nrows = math.ceil(n_panels / ncols)
        fig3, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows))
        axes = np.array(axes).flatten()
        all_pmfs_2d = list(pmfs_2d) + ([united_pmf_2d] if united_pmf_2d else [])
        all_labels_2d = list(labels) + (['United MBAR'] if united_pmf_2d else [])
        vmax = 8.0
        for ax3, fes, lbl in zip(axes, all_pmfs_2d, all_labels_2d):
            if fes is None:
                ax3.set_visible(False)
                continue
            F = np.where(np.isfinite(fes['pmf']), fes['pmf'], np.nan)
            im = ax3.pcolormesh(fes['cv_edges_A'], fes['rg_edges_A'], F.T,
                                cmap='viridis', vmin=0, vmax=vmax, shading='flat')
            plt.colorbar(im, ax=ax3, label='kcal mol⁻¹')
            ax3.set_xlabel(cv_label, fontsize=8)
            ax3.set_ylabel(cv2_label, fontsize=8)
            ax3.set_title(lbl, fontweight='bold', fontsize=9)
        for ax3 in axes[len(all_pmfs_2d):]:
            ax3.set_visible(False)
        out3 = out_dir / 'pmf_comparison_2d.png'
        fig3.savefig(out3, bbox_inches='tight')
        plt.close(fig3)
        print(f'  Saved {out3}')


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def _save_pmf_csv(pmf: dict, path: Path, cv_label: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow([cv_label, 'pmf_kcal_mol', 'probability', 'counts'])
        for x, F, p, c in zip(pmf['cv_A'], pmf['pmf'], pmf['prob'], pmf['counts']):
            w.writerow([
                f'{x:.8g}',
                f'{F:.8g}' if np.isfinite(F) else '',
                f'{p:.8g}',
                int(c),
            ])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('runs', nargs='+', help='Run directories')
    ap.add_argument('--labels', nargs='*', default=None,
                    help='Display labels for each run (default: directory names)')
    ap.add_argument('--out', default='pmf_comparison', help='Output directory')
    ap.add_argument('--bins', type=int, default=60, help='PMF histogram bins')
    ap.add_argument('--cv-min', type=float, default=None)
    ap.add_argument('--cv-max', type=float, default=None)
    ap.add_argument('--smooth-sigma', type=float, default=0.0,
                    help='Gaussian smoothing sigma (bins) for PMF curves in plots')
    ap.add_argument('--no-united', action='store_true',
                    help='Skip combined united MBAR (only per-run comparison)')
    ap.add_argument('--2d', dest='do_2d', action='store_true',
                    help='Include 2D FES panel (primary CV × secondary CV)')
    ap.add_argument('--mbar-backend', default='sambar',
                    choices=['sambar', 'numba', 'numba-anderson', 'lbfgs', 'numpy', 'anderson'],
                    help='MBAR solver backend')
    ap.add_argument('--mbar-tol', type=float, default=1e-10)
    args = ap.parse_args()

    run_paths = args.runs
    labels = args.labels
    if labels is None:
        labels = [Path(p).resolve().name for p in run_paths]
    if len(labels) != len(run_paths):
        ap.error(f'--labels count ({len(labels)}) must match run count ({len(run_paths)})')

    out_dir = Path(args.out)

    # --- Load ---
    print('Loading runs ...')
    datasets: list[agm.Data] = []
    for label, path in zip(labels, run_paths):
        print(f'  {label}: {path}')
        try:
            d = load_run(path)
            datasets.append(d)
            print(f'    {len(d.cv):,} samples, {d.centers.size} windows, '
                  f'T={1.0/(K_B_KJ*d.beta):.1f} K, '
                  f'CV={_cv_key(d)!r}')
        except Exception as exc:
            print(f'  ERROR loading {path}: {exc}', file=sys.stderr)
            sys.exit(1)

    cv_label = agm._primary_cv_axis_label(datasets[0].meta)
    cv2_label = agm._secondary_cv_label(datasets[0].meta)

    # --- Compatibility check for united MBAR ---
    do_united = not args.no_united
    if do_united and len(datasets) > 1:
        ok, reason = _check_compatibility(datasets, labels)
        if not ok:
            print(f'\n  WARNING: runs are not compatible for united MBAR: {reason}')
            print('  Skipping united MBAR; producing per-run comparison only.\n')
            do_united = False

    # --- Shared CV bins ---
    all_cv = np.concatenate([d.cv for d in datasets])
    cv_lo = args.cv_min if args.cv_min is not None else float(np.nanpercentile(all_cv, 0.5))
    cv_hi = args.cv_max if args.cv_max is not None else float(np.nanpercentile(all_cv, 99.5))
    bins = np.linspace(cv_lo, cv_hi, args.bins + 1)

    # 2D bins (secondary CV, only when --2d)
    do_2d = args.do_2d
    all_cv2 = np.concatenate([d.cv2 for d in datasets])
    has_cv2 = np.any(np.isfinite(all_cv2))
    if do_2d and not has_cv2:
        print('  WARNING: --2d requested but no secondary CV data found; skipping 2D.')
        do_2d = False
    bins2: Optional[np.ndarray] = None
    if do_2d:
        cv2_lo = float(np.nanpercentile(all_cv2[np.isfinite(all_cv2)], 0.5))
        cv2_hi = float(np.nanpercentile(all_cv2[np.isfinite(all_cv2)], 99.5))
        bins2 = np.linspace(cv2_lo, cv2_hi, args.bins + 1)

    # --- Per-run MBAR ---
    print('\nPer-run MBAR ...')
    pmfs: list[dict] = []
    pmfs_2d: list[Optional[dict]] = []
    for label, d in zip(labels, datasets):
        print(f'  {label} ...', flush=True)
        mb = _run_mbar(d, tol=args.mbar_tol, backend=args.mbar_backend)
        pmf = _pmf_1d(d.cv, mb['logw'], bins, d.beta)
        pmfs.append(pmf)
        _save_pmf_csv(pmf, out_dir / f'pmf_{_safe_name(label)}.csv', cv_label)
        if do_2d and bins2 is not None:
            pmf2 = _pmf_2d(d.cv, d.cv2, mb['logw'], bins, bins2, d.beta)
            pmfs_2d.append(pmf2)
        else:
            pmfs_2d.append(None)
        span = float(np.nanmax(pmf['pmf'][np.isfinite(pmf['pmf'])]))
        print(f'    PMF span: {span:.3f} kcal/mol, {mb.get("iterations", "?")} iters')

    # --- United MBAR ---
    united_pmf: Optional[dict] = None
    united_pmf_2d: Optional[dict] = None
    united_cv: Optional[np.ndarray] = None
    if do_united:
        print('\nUnited MBAR ...')
        cv_all, cv2_all, logw_u, beta_u, mb_u = united_mbar(
            datasets, tol=args.mbar_tol, backend=args.mbar_backend)
        united_cv = cv_all
        united_pmf = _pmf_1d(cv_all, logw_u, bins, beta_u)
        _save_pmf_csv(united_pmf, out_dir / 'pmf_united.csv', cv_label)
        span_u = float(np.nanmax(united_pmf['pmf'][np.isfinite(united_pmf['pmf'])]))
        print(f'  United PMF span: {span_u:.3f} kcal/mol, {mb_u.get("iterations", "?")} iters')
        if do_2d and bins2 is not None:
            united_pmf_2d = _pmf_2d(cv_all, cv2_all, logw_u, bins, bins2, beta_u)
    elif not args.no_united:
        print('\n  (United MBAR skipped — only one run)')

    # --- Plots ---
    print('\nGenerating plots ...')
    plot_comparison(
        datasets=datasets,
        labels=labels,
        pmfs=pmfs,
        united_pmf=united_pmf,
        united_cv=united_cv,
        out_dir=out_dir,
        cv_label=cv_label,
        smooth_sigma=args.smooth_sigma,
        do_2d=do_2d,
        pmfs_2d=pmfs_2d if do_2d else None,
        united_pmf_2d=united_pmf_2d,
        cv2_label=cv2_label,
    )

    print(f'\nDone. Results in {out_dir}/')


def _safe_name(s: str) -> str:
    return ''.join(c if c.isalnum() or c in '-_' else '_' for c in s).strip('_')


if __name__ == '__main__':
    main()
