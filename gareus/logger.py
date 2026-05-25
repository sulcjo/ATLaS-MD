"""
Distance logging and live dashboard rendering.

This module contains the updated DistanceLogger implementation extracted from
``gareus_peptide.py``.  It owns CSV/JSONL distance logging, live terminal
rendering, resume-history reconstruction, sparse/rectangular 2D topology maps,
GaMD boost diagnostics, replica-diffusion diagnostics, and health/action
recommendations for the GaREUS dashboard.
"""

from __future__ import annotations

import collections
import concurrent.futures
import csv
import math
import shutil
import time
from pathlib import Path
from typing import Optional

import numpy as np

from .colors import color_text, style_text
from .io import BufferedCsvDictWriter, BufferedJsonlWriter
from .progress import GuiProgressSink
from .tui import (
    _ansi_pad,
    _ansi_truncate,
    _ascii_position,
    _dashboard_density,
    _dashboard_full_width_panel,
    _dashboard_weighted_row,
    _join_columns,
    _panel_lines,
    format_duration,
    make_progress_bar,
    render_distance_ascii,
    replica_fg256,
    replica_marker,
    strip_ansi,
    strip_ansi_len,
    write_tui_frame,
)
from .cv import (
    format_primary_cv_value,
    format_primary_delta_value,
    primary_cv_is_contacts,
    primary_cv_label,
    primary_cv_mode,
    primary_cv_units,
    primary_k_units,
    secondary_cv_mode,
)
from .windows import build_explicit_2d_neighbor_edges
from .math_helpers import _hist_overlap

def _sparkline(values: list[float], width: int = 18) -> str:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    if not vals:
        return "·" * max(1, width)
    vals = vals[-max(1, width):]
    lo, hi = min(vals), max(vals)
    chars = "▁▂▃▄▅▆▇█"
    if hi <= lo:
        return color_text("▄" * len(vals), "dim")
    return "".join(chars[max(0, min(len(chars) - 1, int((v - lo) / (hi - lo) * (len(chars) - 1))))] for v in vals)


def _mini_bar(frac: float, width: int = 10) -> str:
    frac = max(0.0, min(1.0, float(frac) if math.isfinite(float(frac)) else 0.0))
    n = int(round(frac * width))
    return color_text("█" * n, "green") + color_text("░" * (width - n), "dim")


def _coverage_bar(values: list[float], lo: float, hi: float, width: int = 48) -> str:
    if hi <= lo or not values:
        return " " * width
    counts = [0] * width
    for v in values:
        if not math.isfinite(float(v)):
            continue
        idx = _ascii_position(float(v), lo, hi, width)
        counts[idx] += 1
    mx = max(counts) if counts else 0
    chars = " ░▒▓█"
    out = []
    for c in counts:
        if mx <= 0 or c <= 0:
            out.append(color_text("·", "dim"))
        else:
            lvl = max(1, min(4, int(math.ceil(c / mx * 4))))
            out.append(chars[lvl])
    return "".join(out)


def boost_anharmonicity(values: list[float]) -> dict:
    """Rolling Gaussianity/anharmonicity score for GaMD boost samples.

    GaMD cumulant reweighting assumes the boost distribution is close to
    Gaussian. This cheap live score combines standardized skewness and excess
    kurtosis: 0 is ideal Gaussian-like, >~0.5 is suspicious, >~1 is ugly.
    """
    arr = np.asarray([float(v) for v in values if math.isfinite(float(v))], dtype=float)
    n = int(arr.size)
    if n < 8:
        return {"n": n, "mean": float("nan"), "sd": float("nan"), "skew": float("nan"), "excess_kurtosis": float("nan"), "score": float("nan")}
    mean = float(np.mean(arr))
    sd = float(np.std(arr))
    if not math.isfinite(sd) or sd <= 1.0e-12:
        # A constant/flat boost trace is not a non-Gaussian distribution; it is
        # simply not informative for cumulant diagnostics.  Do not report the
        # old hard-coded 1.5 BAD-looking score for zero-variance or unavailable
        # boost data.
        return {"n": n, "mean": mean, "sd": sd, "skew": float("nan"), "excess_kurtosis": float("nan"), "score": float("nan")}
    z = (arr - mean) / sd
    skew = float(np.mean(z ** 3))
    excess = float(np.mean(z ** 4) - 3.0)
    score = float(math.sqrt(skew * skew + 0.25 * excess * excess))
    return {"n": n, "mean": mean, "sd": sd, "skew": skew, "excess_kurtosis": excess, "score": score}


def anharmonicity_label(score: float) -> tuple[str, str]:
    if not math.isfinite(float(score)):
        return "n/a", "dim"
    if score < 0.5:
        return "OK", "green"
    if score < 1.0:
        return "WARN", "yellow"
    return "BAD", "red"


def is_gamd_production_phase(phase: str) -> bool:
    """Return True only for phases where live GaMD boost-shape diagnostics are meaningful.

    The calibration/prep stages can have transient/non-stationary boost behavior.
    Keep anharmonicity and skew/kurtosis scores restricted to production,
    otherwise the dashboard screams about deliberately non-equilibrated data.
    """
    return str(phase or "").lower() in {"gareus_production", "gamd_production", "production"}


def _theme_header(title: str, subtitle: str = "") -> str:
    body = f"== {title} =="
    if subtitle:
        body += f"  {subtitle}"
    return color_text(body, "magenta", bold=True)


def _short_status(ok: bool, label: str) -> str:
    return color_text(label, "green" if ok else "red", bold=not ok)


class DistanceLogger:
    def __init__(self, out_dir: Path, args, progress: Optional[GuiProgressSink] = None):
        self.out_dir = Path(out_dir)
        self.args = args
        self.progress = progress
        self.mode = str(getattr(args, "distance_output_mode", "both") or "both").lower()
        self.ascii_mode = str(getattr(args, "distance_ascii_mode", "hist3d") or "hist3d").lower()
        self.ascii_width = int(getattr(args, "distance_ascii_width", 54) or 54)
        self.ascii_max_replicas = int(getattr(args, "distance_ascii_max_replicas", 32) or 32)
        self.history_limit = int(getattr(args, "distance_history_limit", 4000) or 4000)
        self.no_gui = bool(getattr(args, "no_distance_gui_events", False))
        self.dashboard_render_interval_sec = max(0.0, float(getattr(args, "dashboard_render_interval_sec", 0.0) or 0.0))
        self.dashboard_panels = str(getattr(args, "dashboard_panels", "normal") or "normal").lower()
        self.dashboard_heavy_panels_every = max(1, int(getattr(args, "dashboard_heavy_panels_every", 1) or 1))
        self._last_dashboard_render_wall = 0.0
        self._dashboard_render_count = 0
        self.start_wall = time.time()
        self.last_rows: list[dict] = []
        self.last_summary: dict = {}
        self.history_by_replica: dict[int, collections.deque] = {}
        self.history_by_window: dict[int, collections.deque] = {}
        self.secondary_history_by_replica: dict[int, collections.deque] = {}
        self.secondary_history_by_window: dict[int, collections.deque] = {}
        self.cv_secondary_history_by_window: dict[int, collections.deque] = {}
        self.cv_secondary_history_all: list[tuple[float, float]] = []
        self.potential_history_by_replica: dict[int, collections.deque] = {}
        self.boost_history_by_replica: dict[int, collections.deque] = {}
        self.boost_history_all: list[float] = []
        self.window_trace_by_replica: dict[int, list[int]] = {}
        self.roundtrip_state: dict[int, dict] = {}
        self._production_cv_history_reset_done = False
        self._render_executor: Optional[concurrent.futures.ThreadPoolExecutor] = None
        self._pending_render: Optional[concurrent.futures.Future] = None
        self.csv_handle = None
        self.csv_writer = None
        self.jsonl_handle = None
        if self.mode in {"csv", "both"}:
            p = self.out_dir / str(getattr(args, "distance_csv", "distances.csv"))
            append_csv = bool(getattr(args, "resume", False)) and p.exists() and p.stat().st_size > 0
            self.csv_writer = BufferedCsvDictWriter(p, fieldnames=[
                "step", "phase", "replica", "window", "center_A", "k_kcal_mol_A2",
                "cv_A", "primary_cv", "primary_cv_label", "primary_cv_units",
                "primary_cv_value", "primary_cv_center", "primary_cv_k", "primary_cv_k_units",
                "primary_umbrella_bias_kcal_mol",
                "secondary_cv", "secondary_cv_center", "secondary_cv_k_kcal_mol",
                "distance_umbrella_bias_kcal_mol", "secondary_cv_bias_kcal_mol",
                "umbrella_bias_kcal_mol", "umbrella_pull_kcal_mol_A",
                "potential_kj_mol", "gamd_boost_total_kj_mol", "gamd_boost_total_kcal_mol",
                "cv_min_A", "cv_mean_A", "cv_max_A",
                "k_min_kcal_mol_A2", "k_mean_kcal_mol_A2", "k_max_kcal_mol_A2",
                "umbrella_bias_mean_kcal_mol", "umbrella_bias_max_kcal_mol",
                "gamd_boost_mean_kcal_mol", "gamd_boost_sd_kcal_mol", "gamd_boost_max_kcal_mol",
                "gamd_boost_anharmonicity_score", "gamd_boost_skew", "gamd_boost_excess_kurtosis", "gamd_boost_anharmonicity_n",
            ], append=append_csv, flush_rows=int(getattr(args, "csv_flush_rows", 1000) or 1000), extrasaction="ignore")
            self.csv_handle = self.csv_writer
        if self.mode in {"jsonl", "both"}:
            p = self.out_dir / str(getattr(args, "distance_jsonl", "distances.jsonl"))
            self.jsonl_handle = BufferedJsonlWriter(p, append=True, flush_rows=int(getattr(args, "jsonl_flush_rows", 500) or 500))

    def close(self):
        if self._render_executor is not None:
            try:
                self._render_executor.shutdown(wait=True)
            except Exception:
                pass
        for h in (self.csv_handle, self.jsonl_handle):
            if h is not None:
                try:
                    h.close()
                except Exception:
                    pass


    def _restore_candidate_paths(self) -> list[Path]:
        """Return previous scalar sample logs suitable for rebuilding live TUI history."""
        paths = []
        # samples.csv is production-only and usually has the richest scalar rows.
        paths.append(self.out_dir / "samples.csv")
        # distances.csv is the DistanceLogger companion and is a useful fallback
        # for older/partial runs or modes where samples.csv was not opened yet.
        paths.append(self.out_dir / str(getattr(self.args, "distance_csv", "distances.csv")))
        out = []
        seen = set()
        for path in paths:
            path = Path(path)
            key = str(path.resolve()) if path.exists() else str(path)
            if key not in seen:
                seen.add(key)
                out.append(path)
        return out

    @staticmethod
    def _tail_csv_rows(path: Path, limit: int, max_step: Optional[int] = None) -> list[dict]:
        """Read at most the last *limit* CSV rows, optionally capped by step.

        This intentionally streams through the CSV rather than materializing all
        rows.  Resume dashboards only need recent history for histograms,
        sparklines, stuck-replica checks, boost anharmonicity, and 2D topology
        occupancy.  The canonical long-term analysis data remain in samples.csv
        and analysis_chunks/*.npz.
        """
        from collections import deque
        path = Path(path)
        if not path.exists() or path.stat().st_size <= 0:
            return []
        limit = max(1, int(limit or 1))
        rows = deque(maxlen=limit)
        try:
            with path.open(newline="") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    if max_step is not None:
                        try:
                            if int(float(row.get("step", "nan"))) > int(max_step):
                                continue
                        except Exception:
                            continue
                    rows.append(dict(row))
        except Exception:
            return []
        return list(rows)

    @staticmethod
    def _row_float(row: dict, key: str, default: float = float("nan")) -> float:
        try:
            value = row.get(key, default)
            if value in (None, "", "None", "nan"):
                return default
            out = float(value)
            return out if math.isfinite(out) else default
        except Exception:
            return default

    @staticmethod
    def _row_int(row: dict, key: str, default: int = 0) -> int:
        try:
            value = row.get(key, default)
            if value in (None, "", "None"):
                return default
            return int(float(value))
        except Exception:
            return default

    def _clean_resume_rows(self, rows: list[dict]) -> list[dict]:
        clean_rows: list[dict] = []
        for row in rows:
            phase = str(row.get("phase", "gareus_production") or "gareus_production")
            # Only resume production/dashboard statistics from production-like
            # rows.  Setup/equilibration samples should not pollute coverage and
            # GaMD boost diagnostics after a production checkpoint resume.
            if phase and not is_gamd_production_phase(phase):
                continue
            cv = self._row_float(row, "cv_A")
            if not math.isfinite(cv):
                continue
            replica = self._row_int(row, "replica", 0)
            window = self._row_int(row, "window", 0)
            center = self._row_float(row, "center_A", cv)
            k = self._row_float(row, "k_kcal_mol_A2", 0.0)
            bias = self._row_float(row, "umbrella_bias_kcal_mol")
            if not math.isfinite(bias):
                db = self._row_float(row, "distance_umbrella_bias_kcal_mol", 0.0)
                sb = self._row_float(row, "secondary_cv_bias_kcal_mol", 0.0)
                bias = float(db if math.isfinite(db) else 0.0) + float(sb if math.isfinite(sb) else 0.0)
            pull = self._row_float(row, "umbrella_pull_kcal_mol_A")
            if not math.isfinite(pull):
                pull = float(k) * (float(center) - float(cv))
            clean = {
                "replica": int(replica),
                "window": int(window),
                "center_A": float(center),
                "k_kcal_mol_A2": float(k),
                "cv_A": float(cv),
                "umbrella_bias_kcal_mol": float(bias),
                "umbrella_pull_kcal_mol_A": float(pull),
            }
            for key in (
                "secondary_cv", "secondary_cv_center", "secondary_cv_k_kcal_mol",
                "distance_umbrella_bias_kcal_mol", "secondary_cv_bias_kcal_mol",
                "potential_kj_mol", "gamd_boost_total_kj_mol", "gamd_boost_total_kcal_mol",
            ):
                value = self._row_float(row, key)
                if math.isfinite(value):
                    clean[key] = float(value)
            clean_rows.append(clean)
        return clean_rows

    def restore_from_previous_outputs(self, n_windows: Optional[int] = None, max_step: Optional[int] = None) -> dict:
        """Rebuild live dashboard histories from existing run output on resume.

        OpenMM checkpoint resume restores physical/integrator state.  This method
        restores the *visual/statistical* state used by the dashboard: CV coverage
        histories, 2D density, boost histories, potential-energy histories,
        window-walk traces, round-trip counters, and the last per-replica rows.
        Without this, a resumed run is physically continuous but the TUI appears
        to have forgotten everything it already sampled.  Very rude, very CSV.
        """
        nwin = int(n_windows or 0)
        max_rows = int(self.history_limit) * max(1, nwin or 1)
        chosen_path = None
        raw_rows: list[dict] = []
        for path in self._restore_candidate_paths():
            rows = self._tail_csv_rows(path, limit=max_rows, max_step=max_step)
            cleaned = self._clean_resume_rows(rows)
            if cleaned:
                raw_rows = cleaned
                chosen_path = path
                break
        if not raw_rows:
            return {"restored": False, "reason": "no previous production sample rows found"}

        # Replace any empty/new-session histories with the resumed histories.
        self.history_by_replica.clear()
        self.history_by_window.clear()
        self.secondary_history_by_replica.clear()
        self.secondary_history_by_window.clear()
        self.cv_secondary_history_by_window.clear()
        self.cv_secondary_history_all.clear()
        self.potential_history_by_replica.clear()
        self.boost_history_by_replica.clear()
        self.boost_history_all.clear()
        self.window_trace_by_replica.clear()
        self.roundtrip_state.clear()

        self._update_history(raw_rows, n_windows=nwin, phase="gareus_production")
        # Prevent the first post-resume production log from clearing these restored
        # production histories in _maybe_reset_cv_history_for_phase().
        self._production_cv_history_reset_done = True

        # Last dashboard rows: use the highest restored sample step if available,
        # otherwise the last occurrence per replica in the tail.
        by_rep: dict[int, dict] = {}
        for row in raw_rows:
            by_rep[int(row.get("replica", 0))] = row
        self.last_rows = [by_rep[k] for k in sorted(by_rep)]
        self.last_summary = self.summarize(self.last_rows)
        self.last_summary.update(self._boost_anharmonicity_summary("gareus_production"))
        return {
            "restored": True,
            "source": str(chosen_path),
            "rows": int(len(raw_rows)),
            "n_windows": int(nwin),
            "max_step": int(max_step) if max_step is not None else None,
            "replica_histories": int(len(self.history_by_replica)),
            "window_histories": int(len(self.history_by_window)),
        }

    @staticmethod
    def summarize(rows: list[dict]) -> dict:
        vals = np.asarray([float(r["cv_A"]) for r in rows if "cv_A" in r], dtype=float)
        if vals.size == 0:
            return {}
        ks = np.asarray([float(r["k_kcal_mol_A2"]) for r in rows if "k_kcal_mol_A2" in r], dtype=float)
        biases = np.asarray([float(r["umbrella_bias_kcal_mol"]) for r in rows if "umbrella_bias_kcal_mol" in r], dtype=float)
        boosts = []
        for r in rows:
            try:
                b = float(r.get("gamd_boost_total_kcal_mol", "nan"))
                if math.isfinite(b):
                    boosts.append(b)
            except Exception:
                pass
        out = {
            "cv_min_A": float(np.nanmin(vals)),
            "cv_mean_A": float(np.nanmean(vals)),
            "cv_max_A": float(np.nanmax(vals)),
        }
        if ks.size:
            out.update({
                "k_min_kcal_mol_A2": float(np.nanmin(ks)),
                "k_mean_kcal_mol_A2": float(np.nanmean(ks)),
                "k_max_kcal_mol_A2": float(np.nanmax(ks)),
            })
        if biases.size:
            out.update({
                "umbrella_bias_mean_kcal_mol": float(np.nanmean(biases)),
                "umbrella_bias_max_kcal_mol": float(np.nanmax(biases)),
            })
        ss_vals = []
        for r in rows:
            try:
                ss = float(r.get("secondary_cv", "nan"))
                if math.isfinite(ss):
                    ss_vals.append(ss)
            except Exception:
                pass
        if ss_vals:
            ss_arr = np.asarray(ss_vals, dtype=float)
            out.update({
                "secondary_cv_min": float(np.nanmin(ss_arr)),
                "secondary_cv_mean": float(np.nanmean(ss_arr)),
                "secondary_cv_max": float(np.nanmax(ss_arr)),
            })
        if boosts:
            arr = np.asarray(boosts, dtype=float)
            out.update({
                "gamd_boost_mean_kcal_mol": float(np.nanmean(arr)),
                "gamd_boost_sd_kcal_mol": float(np.nanstd(arr)),
                "gamd_boost_max_kcal_mol": float(np.nanmax(arr)),
            })
        return out

    def _maybe_reset_cv_history_for_phase(self, phase: str) -> None:
        """Start production CV coverage from a clean slate.

        Calibration/equilibration CV samples are useful while they are happening,
        but once GaMD/GaREUS production starts they make the live CV histogram,
        coverage bar, and raw-PMF preview look as if production has already
        explored regions that only belonged to the pre-production stage.  Reset
        only the CV histories; keep exchange/diffusion traces and output files
        intact.
        """
        if not is_gamd_production_phase(phase):
            return
        if self._production_cv_history_reset_done:
            return
        self.history_by_replica.clear()
        self.history_by_window.clear()
        self.secondary_history_by_replica.clear()
        self.secondary_history_by_window.clear()
        self.cv_secondary_history_by_window.clear()
        self.cv_secondary_history_all.clear()
        self._production_cv_history_reset_done = True

    def _update_history(self, clean_rows: list[dict], n_windows: Optional[int] = None, phase: str = "") -> None:
        if n_windows is None:
            n_windows = 0
        low_win = 0
        high_win = max(0, int(n_windows) - 1)
        for clean in clean_rows:
            r = int(clean["replica"])
            w = int(clean["window"])
            cv = float(clean["cv_A"])
            hist = self.history_by_replica.setdefault(r, collections.deque(maxlen=self.history_limit))
            hist.append(cv)
            wh = self.history_by_window.setdefault(w, collections.deque(maxlen=self.history_limit))
            wh.append(cv)
            try:
                ss = float(clean.get("secondary_cv", "nan"))
                if math.isfinite(ss):
                    sr = self.secondary_history_by_replica.setdefault(r, collections.deque(maxlen=self.history_limit))
                    sr.append(ss)
                    sw = self.secondary_history_by_window.setdefault(w, collections.deque(maxlen=self.history_limit))
                    sw.append(ss)
                    pair_hist = self.cv_secondary_history_by_window.setdefault(w, collections.deque(maxlen=self.history_limit))
                    pair_hist.append((cv, ss))
                    self.cv_secondary_history_all.append((cv, ss))
                    max_all_2d = max(self.history_limit, self.history_limit * max(1, len(self.history_by_window)))
                    if len(self.cv_secondary_history_all) > max_all_2d:
                        del self.cv_secondary_history_all[: len(self.cv_secondary_history_all) - max_all_2d]
            except Exception:
                pass
            try:
                pe = float(clean.get("potential_kj_mol", "nan"))
                if math.isfinite(pe):
                    ph = self.potential_history_by_replica.setdefault(r, collections.deque(maxlen=self.history_limit))
                    ph.append(pe)
            except Exception:
                pass
            # Only collect boost-shape history during production. Calibration/prep
            # boost distributions are intentionally non-stationary and should not
            # contribute to live cumulant-reweighting quality diagnostics.
            if is_gamd_production_phase(phase):
                try:
                    boost = float(clean.get("gamd_boost_total_kcal_mol", "nan"))
                    if math.isfinite(boost):
                        bh = self.boost_history_by_replica.setdefault(r, collections.deque(maxlen=self.history_limit))
                        bh.append(boost)
                        self.boost_history_all.append(boost)
                        max_all = max(self.history_limit, self.history_limit * max(1, len(self.history_by_replica)))
                        if len(self.boost_history_all) > max_all:
                            del self.boost_history_all[: len(self.boost_history_all) - max_all]
                except Exception:
                    pass
            trace = self.window_trace_by_replica.setdefault(r, [])
            if not trace or trace[-1] != w:
                trace.append(w)
                if len(trace) > 200:
                    del trace[: len(trace) - 200]
            rt = self.roundtrip_state.setdefault(r, {"side": None, "roundtrips": 0, "min": w, "max": w})
            rt["min"] = min(int(rt.get("min", w)), w)
            rt["max"] = max(int(rt.get("max", w)), w)
            if n_windows and high_win > 0:
                side = rt.get("side")
                if w == low_win:
                    if side == "high":
                        rt["roundtrips"] = int(rt.get("roundtrips", 0)) + 1
                    rt["side"] = "low"
                elif w == high_win:
                    if side == "low":
                        rt["roundtrips"] = int(rt.get("roundtrips", 0)) + 1
                    rt["side"] = "high"

    def _output_status_lines(self) -> list[str]:
        names = [
            str(getattr(self.args, "distance_csv", "distances.csv")),
            str(getattr(self.args, "distance_jsonl", "distances.jsonl")),
            str(getattr(self.args, "progress_jsonl", "progress.jsonl")),
            "samples.csv",
            "exchanges.csv",
        ]
        lines = []
        now = time.time()
        for name in names:
            p = self.out_dir / name
            if not p.exists():
                lines.append(f"  {name:<18} missing")
                continue
            age = max(0.0, now - p.stat().st_mtime)
            size_kb = p.stat().st_size / 1024.0
            lines.append(f"  {name:<18} {size_kb:8.1f} kB  last {age:4.0f}s")
        return lines

    def _render_potential_energy_map(self, rows: list[dict], width: Optional[int] = None) -> list[str]:
        """Render per-replica potential-energy histograms beside the CV map."""
        lines = [color_text("potential energy histograms", "cyan", bold=True)]
        width = max(10, int(width or 24))
        all_vals = []
        for vals in self.potential_history_by_replica.values():
            all_vals.extend([float(v) for v in vals if math.isfinite(float(v))])
        if not all_vals:
            lines.append(color_text("  PE unavailable/not sampled yet", "dim"))
            return lines
        lo = float(np.nanpercentile(np.asarray(all_vals, dtype=float), 2.0))
        hi = float(np.nanpercentile(np.asarray(all_vals, dtype=float), 98.0))
        if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
            lo = float(np.nanmin(all_vals))
            hi = float(np.nanmax(all_vals))
        if hi <= lo:
            hi = lo + 1.0
        lines.append(color_text(f"  PE range {lo:.0f} to {hi:.0f} kJ/mol; shaded history, ● current", "dim"))
        max_reps = max(1, min(len(rows), int(getattr(self.args, "distance_ascii_max_replicas", 32) or 32)))
        for r in rows[:max_reps]:
            rep = int(r.get("replica", 0))
            try:
                pe = float(r.get("potential_kj_mol", "nan"))
            except Exception:
                pe = float("nan")
            hist = self.potential_history_by_replica.get(rep, [])
            if hist:
                counts, _edges = np.histogram(np.asarray(hist, dtype=float), bins=width, range=(lo, hi))
                mx = max(1, int(np.nanmax(counts)))
                chars = [" ", "·", "░", "▒", "▓", "█"]
                bar = "".join(chars[min(len(chars)-1, int(round(c / mx * (len(chars)-1))))] for c in counts)
            else:
                bar = "·" * width
            if math.isfinite(pe):
                pos = int(round((pe - lo) / (hi - lo) * (width - 1)))
                pos = max(0, min(width - 1, pos))
                marker = replica_marker("●", rep, bg256=236, bold=True)
                bar = bar[:pos] + marker + bar[pos+1:]
                pe_txt = f"{pe:9.0f}"
            else:
                pe_txt = "      n/a"
            lines.append(f"  {replica_marker(f'r{rep:02d}', rep)} {pe_txt} kJ |{bar}|")
        if len(rows) > max_reps:
            lines.append(color_text(f"  … {len(rows)-max_reps} more replicas", "dim"))
        return lines

    def _render_exchange_acceptance(self, exchange_stats: dict, n_windows: int) -> list[str]:
        lines = [color_text("exchange / window-walk diagnostics", "cyan", bold=True)]
        if not isinstance(exchange_stats, dict):
            lines.append("  no exchange attempts yet")
            return lines
        mode = str(exchange_stats.get("mode", "neighbor"))
        attempts = int(exchange_stats.get("attempts", 0) or 0)
        accepted = int(exchange_stats.get("accepted", 0) or 0)
        if attempts <= 0 and int(exchange_stats.get("gibbs_choices", 0) or 0) <= 0:
            lines.append(f"  mode {mode}; no exchange attempts yet")
            return lines

        if attempts > 0:
            frac_total = accepted / max(1, attempts)
            col = "green" if frac_total >= 0.20 else ("yellow" if frac_total >= 0.08 else "red")
            lines.append(f"  mode {mode:<14} total {100*frac_total:5.1f}% {_mini_bar(frac_total, 10)} {accepted}/{attempts} {color_text('OK' if frac_total >= 0.08 else 'LOW', col, bold=frac_total < 0.08)}")
        if mode == "gibbs-walk" or int(exchange_stats.get("gibbs_choices", 0) or 0) > 0:
            choices = int(exchange_stats.get("gibbs_choices", 0) or 0)
            moves = int(exchange_stats.get("gibbs_moves", 0) or 0)
            stays = int(exchange_stats.get("gibbs_stays", 0) or 0)
            move_frac = moves / max(1, choices)
            lines.append(f"  Gibbs choices: moves {moves}/{choices} ({100*move_frac:4.1f}%), stays {stays}; p shown in exchanges.csv is heat-bath choice prob")

        jumps = exchange_stats.get("jump_bins", {}) or {}
        if jumps:
            lines.append(color_text("  by window jump distance", "white", bold=True))
            def _jump_sort_key(item):
                key, _ = item
                try:
                    return int(str(key).replace("dw", ""))
                except Exception:
                    return 999
            for key, st in sorted(jumps.items(), key=_jump_sort_key)[:6]:
                att = int(st.get("attempts", 0) or 0)
                acc = int(st.get("accepted", 0) or 0)
                if att <= 0:
                    continue
                frac = acc / max(1, att)
                col = "green" if frac >= 0.20 else ("yellow" if frac >= 0.08 else "red")
                label = str(key).replace("dw", "Δw=")
                lines.append(f"  {label:<6} {100*frac:5.1f}% {_mini_bar(frac, 10)} {acc}/{att} {color_text('LOW' if frac < 0.08 else 'OK', col, bold=frac < 0.08)}")

        pairs = exchange_stats.get("pairs", {}) or {}
        if pairs:
            if mode == "neighbor":
                # In neighbor mode, show the adjacent window ladder explicitly.
                for a in range(max(0, n_windows - 1)):
                    key = f"{a}-{a+1}"
                    st = pairs.get(key, {"attempts": 0, "accepted": 0})
                    att = int(st.get("attempts", 0) or 0)
                    acc = int(st.get("accepted", 0) or 0)
                    if att == 0:
                        lines.append(f"  w{a:02d}-w{a+1:02d}   --   {color_text('not tried', 'dim')}")
                    else:
                        frac = acc / max(1, att)
                        col = "green" if frac >= 0.20 else ("yellow" if frac >= 0.08 else "red")
                        lines.append(f"  w{a:02d}-w{a+1:02d} {100*frac:5.1f}% {_mini_bar(frac, 10)} {acc}/{att} {color_text('BAD' if frac < 0.08 else 'OK', col, bold=frac < 0.08)}")
            else:
                # For long-jump modes there may be many pairs. Show the busiest pairs only.
                lines.append(color_text("  busiest specific window pairs", "white", bold=True))
                items = sorted(pairs.items(), key=lambda kv: int(kv[1].get("attempts", 0) or 0), reverse=True)
                for key, st in items[:6]:
                    att = int(st.get("attempts", 0) or 0)
                    acc = int(st.get("accepted", 0) or 0)
                    if att <= 0:
                        continue
                    frac = acc / max(1, att)
                    col = "green" if frac >= 0.20 else ("yellow" if frac >= 0.08 else "red")
                    lines.append(f"  w{key:<5} {100*frac:5.1f}% {_mini_bar(frac, 10)} {acc}/{att} {color_text('LOW' if frac < 0.08 else 'OK', col, bold=frac < 0.08)}")
        return lines

    def _render_overlap(self, centers_a: list[float]) -> list[str]:
        lines = [color_text("window histogram overlap", "cyan", bold=True)]
        if not centers_a:
            lines.append("  no centers")
            return lines
        lo = min(centers_a)
        hi = max(centers_a)
        if hi <= lo:
            hi = lo + 1.0
        any_line = False
        for a in range(len(centers_a) - 1):
            ov = _hist_overlap(self.history_by_window.get(a, []), self.history_by_window.get(a + 1, []), lo, hi)
            if math.isfinite(ov):
                col = "green" if ov >= 0.35 else ("yellow" if ov >= 0.12 else "red")
                lines.append(f"  w{a:02d}-w{a+1:02d} {ov:5.2f} {_mini_bar(ov, 10)} {color_text('disconnected' if ov < 0.12 else 'ok', col, bold=ov < 0.12)}")
                any_line = True
        if not any_line:
            lines.append("  insufficient samples")
        return lines

    def _render_pull_map(self, rows: list[dict], n_windows: int) -> list[str]:
        """Render umbrella restoring force as fixed-height visual gauges.

        Positive pull = umbrella wants to extend the CV; negative pull = compact.
        Keep every window on exactly one short line so the dashboard panel never
        wraps or changes height because one pull entry became too long. ANSI
        truncation is then handled only by the panel formatter.
        """
        lines = [color_text("umbrella pull field", "cyan", bold=True)]
        by_w: dict[int, list[float]] = {}
        for r in rows:
            try:
                by_w.setdefault(int(r["window"]), []).append(float(r.get("umbrella_pull_kcal_mol_A", 0.0)))
            except Exception:
                continue

        finite_vals = [abs(float(v)) for vals in by_w.values() for v in vals if math.isfinite(float(v))]
        # Use a robust scale so one angry replica does not turn the whole map
        # into a row of saturated arrows.  The unit is derivative of the primary
        # umbrella with respect to the selected primary CV.
        contact_mode = primary_cv_is_contacts(self.args)
        pull_units = "kcal/mol/CV" if contact_mode else "kcal/mol/A"
        left_label = "lower" if contact_mode else "compact"
        right_label = "higher" if contact_mode else "extend"
        scale = max(0.5, float(np.nanpercentile(finite_vals, 90)) if finite_vals else 1.0)
        half = 7
        empty_l = " " * half
        empty_r = " " * half
        header = f"  {'win':>3} {left_label:>{half}}│{right_label:<{half}}  pull {pull_units}"
        lines.append(color_text(header, "white", bold=True))

        for w in range(int(n_windows)):
            vals = by_w.get(w, [])
            vals = [float(v) for v in vals if math.isfinite(float(v))]
            if not vals:
                lines.append(color_text(f"  w{w:02d} {empty_l}│{empty_r}     no sample", "dim"))
                continue

            p = float(np.nanmean(vals))
            frac = min(1.0, abs(p) / scale) if scale > 0 else 0.0
            n = max(1 if abs(p) > 0.02 else 0, int(round(frac * half)))
            if abs(p) < 0.02:
                left = " " * half
                right = " " * half
                col = "green"
                label = "neutral"
            elif p < 0.0:
                # Negative pull means the umbrella wants a smaller primary CV.
                left_raw = ("<" * n).rjust(half)
                left = color_text(left_raw, "yellow" if abs(p) < 2.0 else "red", bold=abs(p) >= 2.0)
                right = " " * half
                col = "yellow" if abs(p) < 2.0 else "red"
                label = left_label
            else:
                left = " " * half
                right_raw = (">" * n).ljust(half)
                right = color_text(right_raw, "yellow" if abs(p) < 2.0 else "red", bold=abs(p) >= 2.0)
                col = "yellow" if abs(p) < 2.0 else "red"
                label = right_label

            if abs(p) < 0.5:
                col = "green"
            elif abs(p) < 2.0:
                col = "yellow"
            else:
                col = "red"
            lines.append(f"  w{w:02d} {left}│{right} {p:+7.3f} {color_text(label, col, bold=abs(p) >= 2.0)}")

        lines.append(color_text(f"  scale: full bar ≈ {scale:.2f} {pull_units}; sign from k(center-CV)", "dim"))
        return lines

    def _boost_anharmonicity_summary(self, phase: str) -> dict:
        if not is_gamd_production_phase(phase):
            return {
                "gamd_boost_anharmonicity_score": float("nan"),
                "gamd_boost_skew": float("nan"),
                "gamd_boost_excess_kurtosis": float("nan"),
                "gamd_boost_anharmonicity_n": 0,
            }
        stats = boost_anharmonicity(self.boost_history_all)
        return {
            "gamd_boost_anharmonicity_score": stats["score"],
            "gamd_boost_skew": stats["skew"],
            "gamd_boost_excess_kurtosis": stats["excess_kurtosis"],
            "gamd_boost_anharmonicity_n": int(stats["n"]),
        }

    def _render_gamd_boost(self, rows: list[dict], summary: dict, phase: str) -> list[str]:
        lines = [color_text("GaMD boost diagnostics", "cyan", bold=True)]
        if "gamd_boost_mean_kcal_mol" not in summary:
            lines.append("  boost values unavailable/not yet active")
            return lines
        mean = summary.get("gamd_boost_mean_kcal_mol", float("nan"))
        sd = summary.get("gamd_boost_sd_kcal_mol", float("nan"))
        mx = summary.get("gamd_boost_max_kcal_mol", float("nan"))
        col = "green" if sd < 4.0 else ("yellow" if sd < 6.0 else "red")
        lines.append(f"  all reps mean {mean:7.2f}  sd {color_text(f'{sd:6.2f}', col, bold=sd>=6.0)}  max {mx:7.2f} kcal/mol")
        if is_gamd_production_phase(phase):
            an = boost_anharmonicity(self.boost_history_all)
            lab, lab_col = anharmonicity_label(an["score"])
            if math.isfinite(float(an["score"])):
                lines.append(
                    "  anharm "
                    + color_text(f"{an['score']:5.2f}", lab_col, bold=lab_col in {'yellow','red'})
                    + f" {color_text(lab, lab_col, bold=lab_col!='green')}"
                    + f"  skew {an['skew']:6.2f}  excess kurt {an['excess_kurtosis']:6.2f}  n={int(an['n'])}"
                )
            else:
                lines.append(color_text(f"  anharm n/a; need variable finite production boosts (n={int(an['n'])})", "dim"))
        else:
            lines.append(color_text("  anharm score paused until GaMD/GaREUS production", "dim"))
        for r in rows:
            b = r.get("gamd_boost_total_kcal_mol", "")
            try:
                rep = int(r['replica'])
                bf = float(b)
                if is_gamd_production_phase(phase):
                    pr = boost_anharmonicity(self.boost_history_by_replica.get(rep, []))
                    plab, pcol = anharmonicity_label(pr["score"])
                    if math.isfinite(float(pr["score"])):
                        score_txt = color_text(f"{pr['score']:4.2f}", pcol, bold=pcol != 'green')
                    else:
                        score_txt = color_text(" n/a", "dim")
                    lines.append(f"  {replica_marker(f'r{rep:02d}', rep)} w{int(r['window']):02d} boost {bf:7.2f} kcal/mol  anh {score_txt} {color_text(plab, pcol)}")
                else:
                    lines.append(f"  {replica_marker(f'r{rep:02d}', rep)} w{int(r['window']):02d} boost {bf:7.2f} kcal/mol")
            except Exception:
                pass
        return lines

    def _render_pmf_preview(self, centers_a: list[float], width: Optional[int] = None) -> list[str]:
        """Render a compact horizontal rough raw-PMF preview."""
        lines = [color_text("live raw PMF preview", "cyan", bold=True)]
        vals = []
        for h in self.history_by_replica.values():
            vals.extend(h)
        vals = [v for v in vals if math.isfinite(float(v))]
        if len(vals) < 20:
            lines.append("  insufficient samples")
            return lines
        lo = min(centers_a) if centers_a else min(vals)
        hi = max(centers_a) if centers_a else max(vals)
        target_bins = int(width) if width is not None else int(getattr(self.args, "distance_ascii_width", 54) or 54)
        bins = min(max(12, target_bins), 96)
        if centers_a:
            bins = max(bins, min(96, len(centers_a) * 4))
        hist, edges = np.histogram(np.asarray(vals), bins=bins, range=(lo, hi))
        if hist[hist > 0].size == 0:
            lines.append("  no occupied bins")
            return lines
        pmf = np.full_like(hist, np.nan, dtype=float)
        p = hist.astype(float) / max(1, hist.sum())
        mask = p > 0
        pmf[mask] = -0.0019872041 * float(getattr(self.args, "temperature_k", 300.0)) * np.log(p[mask])
        pmf[mask] -= np.nanmin(pmf[mask])
        finite = pmf[np.isfinite(pmf)]
        maxp = max(1e-12, float(np.nanmax(finite))) if finite.size else 1.0
        chars = []
        for f in pmf:
            if not math.isfinite(float(f)):
                chars.append(color_text("·", "dim"))
                continue
            # Low free energy is visually heavy; high free energy is light.
            inv = max(0.0, min(1.0, 1.0 - float(f) / maxp))
            if inv > 0.80:
                ch, col = "█", "green"
            elif inv > 0.60:
                ch, col = "▓", "green"
            elif inv > 0.40:
                ch, col = "▒", "yellow"
            elif inv > 0.20:
                ch, col = "░", "yellow"
            else:
                ch, col = "·", "dim"
            chars.append(color_text(ch, col, bold=(ch == "█")))
        min_i = int(np.nanargmin(pmf)) if finite.size else -1
        if 0 <= min_i < len(chars):
            chars[min_i] = color_text("▼", "magenta", bold=True)
        bar = "".join(chars)
        lo_lbl = format_primary_cv_value(lo, self.args, precision=2)
        hi_lbl = format_primary_cv_value(hi, self.args, precision=2)
        tick = " " * 2 + lo_lbl + " " * max(1, strip_ansi_len(bar) - len(lo_lbl) - len(hi_lbl)) + hi_lbl
        lines.append("  " + bar)
        lines.append(tick)
        if finite.size:
            min_c = 0.5 * (edges[min_i] + edges[min_i+1])
            lines.append(f"  min ~{format_primary_cv_value(min_c, self.args, precision=3)}; raw unweighted preview, not final PMF")
        return lines

    def _render_replica_diffusion(self, n_windows: int) -> list[str]:
        lines = [color_text("replica diffusion / round trips", "cyan", bold=True)]
        if not self.window_trace_by_replica:
            lines.append("  no traces yet")
            return lines
        for r in sorted(self.window_trace_by_replica):
            tr = self.window_trace_by_replica.get(r, [])[-10:]
            rt = self.roundtrip_state.get(r, {})
            span = int(rt.get("max", 0)) - int(rt.get("min", 0))
            stuck = len(set(tr[-5:])) <= 1 and len(tr) >= 5
            trtxt = "→".join(f"w{x:02d}" for x in tr[-6:])
            lines.append(f"  r{r:02d} {trtxt:<34} span {span:2d}/{max(0,n_windows-1):2d}  trips {int(rt.get('roundtrips',0)):2d} {color_text('STUCK' if stuck else 'OK', 'red' if stuck else 'green', bold=stuck)}")
        return lines

    def _render_replica_table(self, rows: list[dict], phase: str, n_windows: int, ncols: int = 1) -> list[str]:
        """All-replica table combining health, boost and diffusion data, optionally multi-column."""
        if not rows:
            return [color_text("no replica data", "dim")]
        max_span = max(0, n_windows - 1)
        rep_lines: list[str] = []
        for r in sorted(rows, key=lambda x: int(x.get("replica", 0))):
            rep = int(r.get("replica", 0))
            win = int(r.get("window", 0))
            cv = float(r.get("cv_A", float("nan")))
            center = float(r.get("center_A", float("nan")))
            delta = cv - center if math.isfinite(cv) and math.isfinite(center) else float("nan")
            pe = r.get("potential_kj_mol", float("nan"))
            try:
                pe = float(pe)
            except Exception:
                pe = float("nan")
            b_kcal = r.get("gamd_boost_total_kcal_mol", float("nan"))
            try:
                b_kcal = float(b_kcal)
            except Exception:
                b_kcal = float("nan")

            h = self.history_by_replica.get(rep, [])
            spark = _sparkline(list(h)[-24:], 12) if h else " " * 12
            recent10 = list(h)[-10:]
            recent_span = (max(recent10) - min(recent10)) if len(recent10) >= 2 else float("nan")
            cv_stuck = len(recent10) >= 10 and math.isfinite(recent_span) and recent_span < 0.05

            tr = self.window_trace_by_replica.get(rep, [])
            rt = self.roundtrip_state.get(rep, {})
            span_val = int(rt.get("max", 0)) - int(rt.get("min", 0))
            trips = int(rt.get("roundtrips", 0))
            win_stuck = len(set(tr[-5:])) <= 1 and len(tr) >= 5
            trail = "→".join(f"w{x:02d}" for x in tr[-5:])

            if not math.isfinite(pe) and not math.isfinite(cv):
                status, scol = "BAD", "red"
            elif win_stuck or cv_stuck:
                status, scol = "STUCK", "yellow"
            else:
                status, scol = "OK", "green"

            cv_str = (format_primary_cv_value(cv, self.args, precision=3).rjust(9) if math.isfinite(cv) else "      n/a")
            delta_str = ("Δ" + format_primary_delta_value(delta, self.args, precision=3).rjust(8)) if math.isfinite(delta) else "Δ    n/a"
            pe_str = f"{pe:8.1f}kJ" if math.isfinite(pe) else "     n/akJ"
            b_str = f"b{b_kcal:6.2f}kc" if math.isfinite(b_kcal) else "b   n/akc"
            status_txt = color_text(status, scol, bold=scol != "green")
            rep_txt = replica_marker(f"r{rep:02d}", rep)

            line = (
                f"  {rep_txt} w{win:02d}  {cv_str}  {delta_str}  {pe_str}  {b_str}"
                f"  {spark}  {trail:<17}  {span_val:2d}/{max_span:2d}  {trips:2d}t  {status_txt}"
            )
            rep_lines.append(line)

        if ncols <= 1 or len(rep_lines) <= 1:
            return rep_lines

        per_col = math.ceil(len(rep_lines) / ncols)
        columns: list[list[str]] = []
        for c in range(ncols):
            columns.append(rep_lines[c * per_col : (c + 1) * per_col])
        return _join_columns(columns, gap=3)

    def _render_sparklines(self) -> list[str]:
        lines = [color_text("CV sparklines", "cyan", bold=True)]
        if not self.history_by_replica:
            lines.append("  no history")
            return lines
        for r in sorted(self.history_by_replica):
            h = self.history_by_replica[r]
            if h:
                lines.append(f"  r{r:02d} {_sparkline(h, 24):<24} {format_primary_cv_value(h[-1], self.args, precision=3):>10}")
        return lines

    def _render_health(self, rows: list[dict]) -> list[str]:
        lines = [color_text("replica health", "cyan", bold=True)]
        if not rows:
            lines.append("  no rows")
            return lines
        for r in rows:
            rep = int(r["replica"])
            pe = r.get("potential_kj_mol", float("nan"))
            cv = float(r.get("cv_A", float("nan")))
            h = self.history_by_replica.get(rep, [])
            recent = list(h)[-10:]
            recent_span = (max(recent) - min(recent)) if len(recent) >= 2 else float("nan")
            finite = math.isfinite(cv)
            try:
                finite = finite and math.isfinite(float(pe))
            except Exception:
                pass
            stuck = len(recent) >= 10 and recent_span < 0.05
            status = "OK"
            col = "green"
            if not finite:
                status, col = "BAD", "red"
            elif stuck:
                status, col = "STUCK", "yellow"
            pe_txt = f"{float(pe):10.1f}" if isinstance(pe, (float, int)) and math.isfinite(float(pe)) else "       n/a"
            lines.append(f"  r{rep:02d} w{int(r['window']):02d} cv {format_primary_cv_value(cv, self.args, precision=3):>10}  PE {pe_txt} kJ/mol  {color_text(status, col, bold=col!='green')}")
        return lines

    def _render_recommendations(self, exchange_stats: dict, centers_a: list[float], summary: dict) -> list[str]:
        recs = []
        pairs = exchange_stats.get("pairs", {}) if isinstance(exchange_stats, dict) else {}
        bad_pairs = []
        for key, st in pairs.items():
            att = int(st.get("attempts", 0))
            if att >= 5:
                frac = int(st.get("accepted", 0)) / att
                if frac < 0.08:
                    bad_pairs.append(key)
        if bad_pairs:
            recs.append(f"low exchange for {', '.join(bad_pairs[:4])}: add windows, narrow range, or tune k")
        if "gamd_boost_sd_kcal_mol" in summary and summary["gamd_boost_sd_kcal_mol"] >= 6.0:
            recs.append("GaMD boost SD is large: cumulant reweighting may be noisy; consider longer calibration or lower sigma0")
        if math.isfinite(float(summary.get("gamd_boost_anharmonicity_score", float("nan")))) and float(summary.get("gamd_boost_anharmonicity_score")) >= 1.0:
            recs.append("GaMD boost distribution is anharmonic/non-Gaussian: cumulant reweighting is suspect")
        vals = []
        for h in self.history_by_replica.values():
            vals.extend(h)
        if centers_a and vals:
            lo, hi = min(centers_a), max(centers_a)
            span = max(vals) - min(vals)
            target = hi - lo
            if target > 0 and span / target < 0.35 and len(vals) > 100:
                recs.append("CV coverage is narrow: umbrellas/exchanges are not moving through the full range yet")
        if not recs:
            return []
        return [color_text("recommendations", "cyan", bold=True)] + ["  - " + r for r in recs[:5]]


    def _unique_axis_values(self, values: list[float], ndigits: int = 4) -> list[float]:
        clean = []
        seen = set()
        for value in values:
            try:
                v = float(value)
            except Exception:
                continue
            if not math.isfinite(v):
                continue
            key = round(v, int(ndigits))
            if key in seen:
                continue
            seen.add(key)
            clean.append(v)
        return sorted(clean)

    def _nearest_index(self, value: float, axis: list[float]) -> Optional[int]:
        try:
            v = float(value)
        except Exception:
            return None
        if not axis or not math.isfinite(v):
            return None
        return int(min(range(len(axis)), key=lambda i: abs(float(axis[i]) - v)))

    def _primary_axis_bounds_for_2d_map(self, info: dict, cv1_arr=None, centers=None) -> tuple[float, float, str]:
        """Return stable x-axis bounds for the 2D live density map.

        The old renderer mixed sampled values with an Angstrom-style +/-0.5
        padding around target centers.  That made contact-fraction maps either
        zoom into the tiny early sampled smear or waste half the axis past the
        requested contact range.  For dimensionless contact CVs, prefer the
        configured/effective adaptive maximum and all target centers so the map
        always shows the intended 0..max exploration range.
        """
        info = info or {}
        primary_mode = str(info.get("primary_cv", getattr(self.args, "primary_cv", "distance")) or "distance").lower()
        primary_units = str(info.get("primary_cv_units", "A") or "A").lower()
        clean_centers = []
        for c in centers if centers is not None else info.get("centers_a", []):
            try:
                cf = float(c)
                if math.isfinite(cf):
                    clean_centers.append(cf)
            except Exception:
                pass
        clean_samples = []
        if cv1_arr is not None:
            try:
                arr = np.asarray(cv1_arr, dtype=float)
                arr = arr[np.isfinite(arr)]
                clean_samples = [float(x) for x in arr]
            except Exception:
                clean_samples = []

        axis_min = None
        axis_max = None
        source = "sampled"

        for key in ("primary_cv_axis_min", "primary_axis_min", "x_axis_min"):
            try:
                val = info.get(key, None)
                if val is not None and math.isfinite(float(val)):
                    axis_min = float(val)
                    source = "configured"
                    break
            except Exception:
                pass
        for key in ("primary_cv_axis_max", "primary_axis_max", "x_axis_max"):
            try:
                val = info.get(key, None)
                if val is not None and math.isfinite(float(val)):
                    axis_max = float(val)
                    source = "configured"
                    break
            except Exception:
                pass

        win_meta = info.get("window_metadata", {}) or {}
        contact_meta = win_meta.get("contact_adaptive", {}) if isinstance(win_meta, dict) else {}
        if primary_mode == "nonlocal-contacts" or primary_units in {"dimensionless", ""}:
            # Contact maps should display the intended contact-fraction range,
            # not an Angstrom-style padded range.  effective_max is what
            # autocalibration selected; max is the user-requested ceiling.
            if axis_min is None:
                axis_min = 0.0
            if axis_max is None:
                candidates = []
                for key in ("effective_max", "max", "requested_max"):
                    try:
                        val = contact_meta.get(key, None) if isinstance(contact_meta, dict) else None
                        if val is not None and math.isfinite(float(val)):
                            candidates.append(float(val))
                    except Exception:
                        pass
                candidates.extend(clean_centers)
                candidates.extend(clean_samples)
                if candidates:
                    axis_max = max(candidates)
                    source = "contact-target"
                else:
                    axis_max = 1.0
            if bool(getattr(self.args, "contact_normalize", True)):
                axis_min = max(0.0, min(1.0, float(axis_min)))
                axis_max = max(axis_min + 1.0e-6, min(1.0, float(axis_max)))
            # Add only a tiny visual margin; do not add 0.5 to a dimensionless
            # contact fraction.  Include observed outliers if they exceed the
            # configured axis, because hiding data is how dashboards become lies.
            if clean_samples:
                axis_min = min(axis_min, min(clean_samples))
                axis_max = max(axis_max, max(clean_samples))
            if clean_centers:
                axis_min = min(axis_min, min(clean_centers))
                axis_max = max(axis_max, max(clean_centers))
            span = max(1.0e-6, axis_max - axis_min)
            pad = min(0.02, 0.03 * span)
            axis_min = max(0.0, axis_min - pad)
            axis_max = axis_max + pad
            if bool(getattr(self.args, "contact_normalize", True)):
                axis_max = min(1.0, axis_max)
            if axis_max <= axis_min:
                axis_max = min(1.0, axis_min + 0.05) if bool(getattr(self.args, "contact_normalize", True)) else axis_min + 1.0
            return float(axis_min), float(axis_max), source

        # Distance/default behavior: include both target centers and samples, but
        # use distance-scale padding rather than contact-scale padding.
        vals = clean_centers + clean_samples
        if not vals:
            return 0.0, 1.0, source
        lo = min(vals)
        hi = max(vals)
        span = max(0.5, hi - lo)
        lo -= max(0.25, 0.08 * span)
        hi += max(0.25, 0.08 * span)
        return float(max(0.0, lo)), float(max(hi, lo + 0.5)), "distance-target"

    def _primary_axis_tick_label(self, value: float, info: dict) -> str:
        """Compact x-axis label for the 2D map, with contact-aware precision."""
        primary_units = str((info or {}).get("primary_cv_units", "A") or "A").lower()
        primary_mode = str((info or {}).get("primary_cv", getattr(self.args, "primary_cv", "distance")) or "distance").lower()
        try:
            v = float(value)
        except Exception:
            return str(value)[:5]
        if primary_mode == "nonlocal-contacts" or primary_units in {"dimensionless", ""}:
            if abs(v) < 0.995:
                return f"{v:.2f}"
            return f"{v:.2f}"
        return f"{v:.1f}"

    @staticmethod
    def _acb_y_label(val: float, mode_name: str, label_w: int) -> str:
        """Return a y-axis label for the given secondary-CV value."""
        mode = secondary_cv_mode(str(mode_name))
        if mode == "alpha-coil-beta":
            if abs(val - 1.0) < 0.15:
                return "alpha".rjust(label_w)
            elif abs(val) < 0.15:
                return " coil".rjust(label_w)
            elif abs(val + 1.0) < 0.15:
                return " beta".rjust(label_w)
        if mode == "rama-regions":
            labels = [
                (-1.00, "beta"),
                (-0.50, "PPII"),
                (0.00, "turn"),
                (0.50, "alpha"),
                (1.00, "L-alpha"),
            ]
            nearest, label = min(labels, key=lambda item: abs(float(val) - item[0]))
            if abs(float(val) - nearest) < 0.16:
                return label.rjust(label_w)
        raw = f"{val:+.2f}" if val < 0 else f" {val:.2f}"
        return raw.rjust(label_w)

    def _render_2d_diffusion_map(self, rows: list[dict], info: dict, width: Optional[int] = None) -> list[str]:
        """2D sampling density heat-map: x = distance CV (Å), y = secondary structure CV.

        Cell density is shaded with log-scale block chars so both sparse tails and
        dense cores are visible.  Current replica positions are overlaid as colored
        ● dots; window-center ticks appear as ▲ along the bottom x-axis.
        """
        pairs = self.cv_secondary_history_all
        if not pairs:
            return []
        cv1_arr = np.asarray([p[0] for p in pairs], dtype=float)
        cv2_arr = np.asarray([p[1] for p in pairs], dtype=float)
        mask = np.isfinite(cv1_arr) & np.isfinite(cv2_arr)
        cv1_arr = cv1_arr[mask]
        cv2_arr = cv2_arr[mask]
        if cv1_arr.size < 2:
            return []

        width = max(50, int(width or 80))
        map_w = max(30, width - 16)
        map_h = 16

        centers_a = [float(x) for x in info.get("centers_a", []) if str(x) not in {"", "None", "nan"}]
        x_lo, x_hi, x_axis_source = self._primary_axis_bounds_for_2d_map(info, cv1_arr=cv1_arr, centers=centers_a)

        sec_meta = info.get("secondary_cv", {}) or {}
        y_lo = float(sec_meta.get("range_min", -1.0))
        y_hi = float(sec_meta.get("range_max", 1.0))

        try:
            H, xedges, yedges = np.histogram2d(
                cv1_arr, cv2_arr,
                bins=[map_w, map_h],
                range=[[x_lo, x_hi], [y_lo, y_hi]],
            )
        except Exception:
            return []
        H = H.T  # H[y_idx, x_idx]
        log_H = np.log1p(H)
        max_log = float(np.max(log_H))
        if max_log <= 0.0:
            max_log = 1.0

        density_chars = " ░▒▓█"
        density_fg256 = [None, 238, 244, 250, 231]

        rep_cells: dict[tuple[int, int], list[int]] = {}
        for r in rows:
            try:
                rep = int(r["replica"])
                cv1 = float(r.get("cv_A", "nan"))
                cv2 = float(r.get("secondary_cv", "nan"))
                if not (math.isfinite(cv1) and math.isfinite(cv2)):
                    continue
                xi = max(0, min(map_w - 1, int((cv1 - x_lo) / (x_hi - x_lo) * map_w)))
                yi = max(0, min(map_h - 1, int((cv2 - y_lo) / (y_hi - y_lo) * map_h)))
                rep_cells.setdefault((yi, xi), []).append(rep)
            except Exception:
                pass

        y_label_w = 7
        mode_name = str(sec_meta.get("mode", getattr(self.args, "secondary_cv", "secondary")))
        primary_label = str(info.get('primary_cv_label', 'primary CV'))
        primary_units = str(info.get('primary_cv_units', '') or '')
        unit_suffix = f" ({primary_units})" if primary_units and primary_units != "dimensionless" else ""
        lines = [color_text(f"  2D sampling density  x = {primary_label}{unit_suffix}  y = {mode_name}", "white", bold=True)]
        lines.append(color_text(f"  x-axis shown range {self._primary_axis_tick_label(x_lo, info)}..{self._primary_axis_tick_label(x_hi, info)} from {x_axis_source}; includes target centers even before sampling reaches them", "dim"))

        for yi in range(map_h - 1, -1, -1):
            y_mid = 0.5 * (float(yedges[yi]) + float(yedges[yi + 1]))
            label = self._acb_y_label(y_mid, mode_name, y_label_w)
            row_str = f"{label} │"
            for xi in range(map_w):
                reps = rep_cells.get((yi, xi), [])
                if reps:
                    row_str += style_text("●", fg256=replica_fg256(reps[0]), bold=True)
                else:
                    level = min(4, int(5 * float(log_H[yi, xi]) / max_log))
                    ch = density_chars[level]
                    fg = density_fg256[level]
                    row_str += style_text(ch, fg256=fg) if fg is not None else ch
            row_str += "│"
            lines.append(row_str)

        lines.append(" " * y_label_w + " └" + "─" * map_w + "┘")

        tick_row = [" "] * map_w
        for c in centers_a:
            if x_lo <= c <= x_hi:
                xi = max(0, min(map_w - 1, int((c - x_lo) / (x_hi - x_lo) * map_w)))
                tick_row[xi] = "▲"
        lines.append(" " * (y_label_w + 2) + "".join(tick_row))

        n_labels = min(8, max(2, map_w // 8))
        x_indices = np.linspace(0, map_w - 1, n_labels, dtype=int)
        x_label_row = list(" " * map_w)
        for xi in x_indices:
            x_val = x_lo + (float(xi) + 0.5) / map_w * (x_hi - x_lo)
            lbl = self._primary_axis_tick_label(x_val, info)
            start = max(0, xi - len(lbl) // 2)
            for k, ch in enumerate(lbl):
                if start + k < map_w:
                    x_label_row[start + k] = ch
        lines.append(" " * (y_label_w + 2) + "".join(x_label_row))

        lines.append(color_text(
            f"  n={len(pairs)} 2D samples  log-density  ● current replica  ▲ window center  x={self._primary_axis_tick_label(x_lo, info)}..{self._primary_axis_tick_label(x_hi, info)}",
            "dim",
        ))
        return lines

    def _render_2d_replica_map(self, rows: list[dict], info: dict, width: Optional[int] = None) -> list[str]:
        """Render a compact distance x secondary-CV map for 2D umbrella runs.

        Two grids are shown when possible:
        - assigned thermodynamic state: which replica currently carries each 2D window
        - current nearest 2D bin: where each replica's instantaneous (distance, secondary-CV) value lies
        """
        if not rows:
            return []
        ss_targets = [float(x) for x in (info or {}).get("secondary_cv_centers", []) if str(x) not in {"", "None", "nan"}]
        if not ss_targets:
            ss_targets = []
            for r in rows:
                try:
                    ss_targets.append(float(r.get("secondary_cv_center", "nan")))
                except Exception:
                    pass
        if not ss_targets:
            return []
        has_ss_value = False
        for r in rows:
            try:
                if math.isfinite(float(r.get("secondary_cv", "nan"))):
                    has_ss_value = True
                    break
            except Exception:
                pass
        if not has_ss_value:
            return []

        centers = [float(x) for x in (info or {}).get("centers_a", [])] or [float(r.get("center_A", float("nan"))) for r in rows]
        dist_axis = self._unique_axis_values(centers, ndigits=3)
        sec_axis = self._unique_axis_values(ss_targets, ndigits=4)
        if len(dist_axis) < 1 or len(sec_axis) < 1:
            return []

        width = max(44, int(width or 100))
        y_label_w = 8
        max_cell_w = 5
        cell_w = max(3, min(max_cell_w, (width - y_label_w - 6) // max(1, len(dist_axis))))
        if cell_w < 3:
            cell_w = 3

        def _cell_text(items: list[int], bg: int = 236) -> str:
            if not items:
                return color_text(".".center(cell_w), "dim")
            items = sorted(int(x) for x in items)
            if len(items) == 1:
                rep = items[0]
                label = f"r{rep:02d}" if cell_w >= 3 else str(rep % 10)
                return replica_marker(label[-cell_w:].center(cell_w), rep, bg256=bg, bold=True)
            label = f"+{len(items)}" if cell_w >= 3 else "*"
            return style_text(label.center(cell_w), fg256=231, bg256=52, bold=True)

        window_to_replica = {}
        for r in rows:
            try:
                window_to_replica[int(r["window"])] = int(r["replica"])
            except Exception:
                pass

        assigned_grid: dict[tuple[int, int], list[int]] = {}
        for w, (dc, sc) in enumerate(zip(centers, ss_targets)):
            di = self._nearest_index(float(dc), dist_axis)
            si = self._nearest_index(float(sc), sec_axis)
            rep = window_to_replica.get(int(w))
            if rep is None or di is None or si is None:
                continue
            assigned_grid.setdefault((si, di), []).append(rep)

        current_grid: dict[tuple[int, int], list[int]] = {}
        for r in rows:
            try:
                rep = int(r["replica"])
                di = self._nearest_index(float(r.get("cv_A", "nan")), dist_axis)
                si = self._nearest_index(float(r.get("secondary_cv", "nan")), sec_axis)
            except Exception:
                continue
            if di is None or si is None:
                continue
            current_grid.setdefault((si, di), []).append(rep)

        lines = [color_text(f"2D {str((info or {}).get('primary_cv_label', 'primary CV'))} x secondary-CV replica map", "cyan", bold=True)]
        mode = (info or {}).get("secondary_cv", {}) or {}
        mode_name = str(mode.get("mode", getattr(self.args, "secondary_cv", "secondary")))
        lines.append(color_text(f"  x = {str((info or {}).get('primary_cv_label', 'primary CV'))} target, y = {mode_name}; cells show replica ids", "dim"))

        # Column labels are intentionally terse so the map survives normal terminal widths.
        header_cells = []
        for d in dist_axis:
            if cell_w <= 3:
                txt = f"{d:.0f}" if abs(d) >= 10 else f"{d:.1f}"[:cell_w]
            else:
                txt = f"{d:.1f}"[:cell_w]
            header_cells.append(txt.center(cell_w))
        dist_header = " " * y_label_w + " " + "".join(header_cells)

        # Build both grid bodies as parallel row lists for side-by-side display.
        assigned_rows: list[str] = []
        current_rows: list[str] = []
        for si, sec in list(enumerate(sec_axis))[::-1]:
            y_lbl = self._acb_y_label(sec, mode_name, y_label_w)
            a_row = f"{y_lbl} "
            c_row = f"{y_lbl} "
            for di in range(len(dist_axis)):
                a_row += _cell_text(assigned_grid.get((si, di), []), bg=236)
                c_row += _cell_text(current_grid.get((si, di), []), bg=24)
            assigned_rows.append(a_row.rstrip())
            current_rows.append(c_row.rstrip())

        # Place the two grids side by side with a small divider gap.
        a_title = color_text("  assigned states", "white", bold=True)
        c_title = color_text("  instantaneous positions", "white", bold=True)
        grid_w = max(strip_ansi_len(dist_header), max((strip_ansi_len(r) for r in assigned_rows), default=0))
        gap_str = "   "
        lines.append(a_title + " " * max(1, grid_w - strip_ansi_len(a_title) + len(gap_str)) + c_title)
        lines.append(dist_header + gap_str + dist_header)
        for a_row, c_row in zip(assigned_rows, current_rows):
            lines.append(_ansi_pad(a_row, grid_w) + gap_str + c_row)

        # A compact per-replica offset table helps distinguish assigned-window exchange from actual CV motion.
        offset_lines = []
        for r in sorted(rows, key=lambda x: int(x.get("replica", 0)))[: min(8, len(rows))]:
            try:
                rep = int(r["replica"])
                cv = float(r.get("cv_A", "nan"))
                dc = float(r.get("center_A", "nan"))
                ss = float(r.get("secondary_cv", "nan"))
                sc = float(r.get("secondary_cv_center", "nan"))
                if not all(math.isfinite(x) for x in (cv, dc, ss, sc)):
                    continue
                offset_lines.append(
                    f"  {replica_marker(f'r{rep:02d}', rep)} w{int(r['window']):02d} "
                    f"target({format_primary_cv_value(dc, self.args, precision=3)},{sc:+5.2f}) "
                    f"now({format_primary_cv_value(cv, self.args, precision=3)},{ss:+5.2f}) "
                    f"Δ({format_primary_delta_value(cv-dc, self.args, precision=3)},{ss-sc:+5.2f})"
                )
            except Exception:
                continue
        if offset_lines:
            lines.append(color_text("  current offsets", "white", bold=True))
            lines.extend(offset_lines)
            if len(rows) > len(offset_lines):
                lines.append(color_text(f"  … {len(rows) - len(offset_lines)} more replicas", "dim"))
        return lines



    def _dashboard_2d_window_type_lookup(self, info: dict, n_windows: int) -> dict[int, str]:
        """Return per-window display type for the live 2D topology panel."""
        out: dict[int, str] = {i: "grid" for i in range(max(0, int(n_windows)))}
        sec_meta = (info or {}).get("secondary_cv", {}) or {}
        win_meta = (info or {}).get("window_metadata", {}) or {}
        rows = []
        for source in (sec_meta, win_meta):
            candidate = source.get("normalized_rows", []) if isinstance(source, dict) else []
            if isinstance(candidate, list):
                rows.extend(candidate)
        for row in rows:
            try:
                w = int(row.get("window", len(out)))
            except Exception:
                continue
            wtype = str(row.get("window_type") or row.get("lifecycle") or row.get("source") or "explicit").strip().lower()
            if w >= 0:
                out[w] = wtype or "explicit"
        return out

    def _dashboard_2d_window_type_symbol(self, wtype: str) -> str:
        """One-letter topology symbol for the fixed 2D TUI map."""
        t = str(wtype or "").strip().lower()
        if any(x in t for x in ("retained", "keep")):
            return "r"
        if any(x in t for x in ("patch", "local")):
            return "p"
        if any(x in t for x in ("grid", "rect", "base", "cross")):
            return "o"
        if any(x in t for x in ("manual", "explicit", "csv")):
            return "e"
        return "?"

    def _dashboard_edge_quality(self, wi: int, wj: int, edge_type: str, exchange_stats: dict, centers_a: list[float], secondary_centers: list[float]) -> dict:
        """Classify one 2D graph edge for the fixed topology panel."""
        key = f"{min(int(wi), int(wj))}-{max(int(wi), int(wj))}"
        pairs = exchange_stats.get("pairs", {}) if isinstance(exchange_stats, dict) else {}
        st = pairs.get(key, {}) if isinstance(pairs, dict) else {}
        attempts = int(st.get("attempts", 0) or 0) if isinstance(st, dict) else 0
        accepted = int(st.get("accepted", 0) or 0) if isinstance(st, dict) else 0
        acc_frac = (accepted / max(1, attempts)) if attempts > 0 else float("nan")

        level = "untested"
        score = 3
        reason = "not enough exchange attempts"
        if attempts >= 5:
            if acc_frac < 0.04:
                level, score, reason = "bad", 0, f"exchange {100.0*acc_frac:.1f}%"
            elif acc_frac < 0.08:
                level, score, reason = "warn", 1, f"exchange {100.0*acc_frac:.1f}%"
            else:
                level, score, reason = "ok", 2, f"exchange {100.0*acc_frac:.1f}%"

        # Add an overlap fallback. Distance-axis edges use distance histories;
        # secondary-axis edges use secondary-CV histories when available.
        overlap = float("nan")
        try:
            if "secondary" in str(edge_type).lower() and self.secondary_history_by_window:
                vals = [float(x) for x in secondary_centers if math.isfinite(float(x))]
                lo = min(vals) if vals else -1.0
                hi = max(vals) if vals else 1.0
                if hi <= lo:
                    hi = lo + 1.0
                overlap = _hist_overlap(
                    self.secondary_history_by_window.get(int(wi), []),
                    self.secondary_history_by_window.get(int(wj), []),
                    lo,
                    hi,
                )
            else:
                vals = [float(x) for x in centers_a if math.isfinite(float(x))]
                lo = min(vals) if vals else 0.0
                hi = max(vals) if vals else 1.0
                if hi <= lo:
                    hi = lo + 1.0
                overlap = _hist_overlap(
                    self.history_by_window.get(int(wi), []),
                    self.history_by_window.get(int(wj), []),
                    lo,
                    hi,
                )
        except Exception:
            overlap = float("nan")
        if math.isfinite(float(overlap)):
            if overlap < 0.05 and score > 0:
                level, score, reason = "bad", 0, f"overlap {overlap:.2f}"
            elif overlap < 0.12 and score > 1:
                level, score, reason = "warn", 1, f"overlap {overlap:.2f}"
            elif score == 3:
                level, score, reason = "ok", 2, f"overlap {overlap:.2f}"

        return {
            "key": key,
            "wi": int(wi),
            "wj": int(wj),
            "edge_type": str(edge_type or "graph"),
            "attempts": int(attempts),
            "accepted": int(accepted),
            "acceptance_fraction": float(acc_frac),
            "overlap": float(overlap),
            "level": level,
            "score": int(score),
            "reason": reason,
        }

    def _dashboard_edge_connector(self, quality: dict, horizontal: bool = True) -> str:
        level = str((quality or {}).get("level", "untested")).lower()
        if horizontal:
            if level == "bad":
                return color_text("xxx", "red", bold=True)
            if level == "warn":
                return color_text("!!!", "yellow", bold=True)
            if level == "ok":
                return color_text("---", "green")
            return color_text("...", "dim")
        if level == "bad":
            return color_text(" x ", "red", bold=True)
        if level == "warn":
            return color_text(" ! ", "yellow", bold=True)
        if level == "ok":
            return color_text(" | ", "green")
        return color_text(" . ", "dim")

    def _render_2d_topology_edge_table(self, edge_qualities: list[dict], centers_a: list[float], secondary_centers: list[float], max_rows: int = 10) -> list[str]:
        """Compact fallback for terminals too narrow for a useful graph map."""
        if not edge_qualities:
            return [color_text("  no 2D graph edges available yet", "dim")]
        lines = [color_text("  edge      status  acc      overlap  type                 centers", "white", bold=True)]
        ordered = sorted(edge_qualities, key=lambda q: (int(q.get("score", 3)), -int(q.get("attempts", 0)), str(q.get("key", ""))))
        for q in ordered[:max(1, int(max_rows))]:
            wi, wj = int(q.get("wi", -1)), int(q.get("wj", -1))
            level = str(q.get("level", "untested")).upper()
            col = "red" if level == "BAD" else "yellow" if level == "WARN" else "green" if level == "OK" else "dim"
            attempts = int(q.get("attempts", 0) or 0)
            acc = q.get("acceptance_fraction", float("nan"))
            acc_txt = f"{100.0*float(acc):5.1f}%" if attempts > 0 and math.isfinite(float(acc)) else "   -- "
            ov = q.get("overlap", float("nan"))
            ov_txt = f"{float(ov):6.2f}" if math.isfinite(float(ov)) else "    --"
            try:
                ci = f"({float(centers_a[wi]):.2f},{float(secondary_centers[wi]):+.2f})"
                cj = f"({float(centers_a[wj]):.2f},{float(secondary_centers[wj]):+.2f})"
                centers_txt = f"{ci}->{cj}"
            except Exception:
                centers_txt = ""
            etype = str(q.get("edge_type", "graph"))[:20]
            lines.append(
                f"  {str(q.get('key','')):<9} {color_text(f'{level:<6}', col, bold=level in {'BAD','WARN'})} "
                f"{acc_txt} {ov_txt}  {etype:<20} {centers_txt}"
            )
        if len(edge_qualities) > max_rows:
            lines.append(color_text(f"  ... {len(edge_qualities) - max_rows} more edges", "dim"))
        return lines

    def _render_2d_overlap_summary_lines(self, edge_qualities: list[dict], centers_a: list[float], secondary_centers: list[float], max_rows: int = 4) -> list[str]:
        """Summarize 2D graph-edge histogram overlap directly inside the topology panel.

        The old dashboard had a separate 1D adjacent-window overlap panel.  For
        2D/sparse runs that is misleading because useful neighbors are graph
        edges, not flattened window indices.  This helper reports overlap on the
        same 2D graph used by exchange/topology rendering.
        """
        finite = [q for q in edge_qualities if math.isfinite(float(q.get("overlap", float("nan"))))]
        if not edge_qualities:
            return [color_text("  2D overlap: no graph edges available", "dim")]
        if not finite:
            return [color_text("  2D overlap: insufficient window histories for graph-edge overlap yet", "dim")]
        bad = [q for q in finite if float(q.get("overlap", float("nan"))) < 0.05]
        warn = [q for q in finite if 0.05 <= float(q.get("overlap", float("nan"))) < 0.12]
        ok = [q for q in finite if float(q.get("overlap", float("nan"))) >= 0.12]
        min_q = min(finite, key=lambda q: float(q.get("overlap", float("nan"))))
        min_ov = float(min_q.get("overlap", float("nan")))
        min_col = "red" if min_ov < 0.05 else "yellow" if min_ov < 0.12 else "green"
        lines = [
            color_text("  2D graph-overlap:", "white", bold=True)
            + f" {len(finite)}/{len(edge_qualities)} edges measured; "
            + color_text(f"BAD {len(bad)}", "red", bold=bool(bad))
            + " / "
            + color_text(f"WARN {len(warn)}", "yellow", bold=bool(warn))
            + " / "
            + color_text(f"OK {len(ok)}", "green")
            + f"; min {color_text(f'{min_ov:.2f}', min_col, bold=min_col != 'green')} on {min_q.get('key')}"
        ]
        worst = sorted(finite, key=lambda q: float(q.get("overlap", float("nan"))))[:max(0, int(max_rows))]
        for q in worst:
            try:
                wi, wj = int(q.get("wi", -1)), int(q.get("wj", -1))
                where = f"({float(centers_a[wi]):.2f},{float(secondary_centers[wi]):+.2f})->({float(centers_a[wj]):.2f},{float(secondary_centers[wj]):+.2f})"
            except Exception:
                where = str(q.get("key", ""))
            ov = float(q.get("overlap", float("nan")))
            col = "red" if ov < 0.05 else "yellow" if ov < 0.12 else "green"
            edge_type = str(q.get("edge_type", "graph"))[:18]
            lines.append(
                "  overlap: "
                + color_text(f"{ov:5.2f}", col, bold=col != "green")
                + f" {str(q.get('key', '')):<7} {edge_type:<18} {where}"
            )
        return lines

    def _render_2d_sparse_topology_map(self, rows: list[dict], info: dict, exchange_stats: dict, width: Optional[int] = None) -> list[str]:
        """Render the fixed 2D window topology/weak-edge overlay panel.

        This is intentionally always-on when 2D window data exist. It has no
        user-facing mode switch: the dashboard should show where exchange/overlap
        problems live in the 2D thermodynamic-state graph.
        """
        info = info or {}
        centers_a = [float(x) for x in info.get("centers_a", []) if str(x) not in {"", "None", "nan"}]
        sec = [float(x) for x in info.get("secondary_cv_centers", []) if str(x) not in {"", "None", "nan"}]
        nwin = int(info.get("n_windows", len(centers_a) or len(sec) or 0) or 0)
        if not centers_a or not sec or len(centers_a) != len(sec) or nwin < 2:
            return []

        sec_meta = info.get("secondary_cv", {}) or {}
        explicit_2d = bool(sec_meta.get("explicit_2d_windows", info.get("explicit_2d", False)))
        rectangular_2d = bool(sec_meta.get("grid", info.get("rectangular_2d", False)))
        unique_sec = self._unique_axis_values(sec, ndigits=4)
        if not explicit_2d and len(unique_sec) <= 1:
            return []

        # Reconstruct the graph from the current window table. For explicit sparse
        # tables this is the same geometry-graph helper used by exchange; for
        # rectangular 2D it remains readable and still catches weak row/column edges.
        try:
            graph_edges = build_explicit_2d_neighbor_edges(centers_a, sec, args=self.args)
        except Exception:
            graph_edges = []
        if not graph_edges and nwin > 1:
            graph_edges = [{"wi": i, "wj": i + 1, "edge_type": "linear_neighbor", "normalized_distance": 1.0} for i in range(nwin - 1)]

        edge_quality_by_pair: dict[tuple[int, int], dict] = {}
        edge_qualities: list[dict] = []
        for edge in graph_edges:
            try:
                wi, wj = int(edge.get("wi")), int(edge.get("wj"))
            except Exception:
                continue
            q = self._dashboard_edge_quality(wi, wj, str(edge.get("edge_type", "graph")), exchange_stats, centers_a, sec)
            edge_quality_by_pair[(min(wi, wj), max(wi, wj))] = q
            edge_qualities.append(q)

        dist_axis = self._unique_axis_values(centers_a, ndigits=3)
        sec_axis = self._unique_axis_values(sec, ndigits=4)
        width = int(width or shutil.get_terminal_size((160, 40)).columns or 160)
        y_label_w = 8
        node_w = 3
        map_w = y_label_w + 1 + len(dist_axis) * node_w + max(0, len(dist_axis) - 1) * 3
        too_dense = len(dist_axis) > 18 or len(sec_axis) > 10 or map_w > max(60, width - 4)

        title = "2D sparse/exchange topology" if explicit_2d and not rectangular_2d else "2D exchange topology"
        lines = [color_text(title, "cyan", bold=True)]
        topo_label = "sparse explicit 2D" if explicit_2d and not rectangular_2d else "rectangular 2D" if rectangular_2d or len(unique_sec) > 1 else "2D"
        bad_n = sum(1 for q in edge_qualities if q.get("level") == "bad")
        warn_n = sum(1 for q in edge_qualities if q.get("level") == "warn")
        lines.append(color_text(f"  {topo_label}; {nwin} windows, {len(edge_qualities)} graph edges, weak/overlap edges BAD={bad_n} WARN={warn_n}", "dim"))
        lines.extend(self._render_2d_overlap_summary_lines(edge_qualities, centers_a, sec, max_rows=3))

        if too_dense:
            lines.extend(self._render_2d_topology_edge_table(edge_qualities, centers_a, sec, max_rows=10))
            return lines

        type_lookup = self._dashboard_2d_window_type_lookup(info, nwin)
        assigned_rep_by_window = {}
        current_nearest_by_cell: dict[tuple[int, int], list[int]] = {}
        for r in rows or []:
            try:
                w = int(r.get("window"))
                rep = int(r.get("replica"))
                assigned_rep_by_window[w] = rep
                cv = float(r.get("cv_A", float("nan")))
                ss = float(r.get("secondary_cv", float("nan")))
                di = self._nearest_index(cv, dist_axis)
                si = self._nearest_index(ss, sec_axis)
                if di is not None and si is not None:
                    current_nearest_by_cell.setdefault((si, di), []).append(rep)
            except Exception:
                continue

        win_at_cell: dict[tuple[int, int], int] = {}
        for w, (dc, sc) in enumerate(zip(centers_a, sec)):
            di = self._nearest_index(dc, dist_axis)
            si = self._nearest_index(sc, sec_axis)
            if di is None or si is None:
                continue
            # Keep the first window in a duplicated cell visible; duplicates are rare
            # and reported in metadata rather than hidden in the topology map.
            win_at_cell.setdefault((si, di), int(w))

        def node_cell(si: int, di: int) -> str:
            w = win_at_cell.get((si, di))
            if w is None:
                reps = current_nearest_by_cell.get((si, di), [])
                if reps:
                    return replica_marker(" * ", int(reps[0]), bg256=236, bold=True)
                return color_text(" . ", "dim")
            sym = self._dashboard_2d_window_type_symbol(type_lookup.get(w, "grid"))
            rep = assigned_rep_by_window.get(w)
            if rep is not None:
                label = (sym.upper() + str(int(rep) % 10))[-2:].center(3)
                return replica_marker(label, int(rep), bg256=236, bold=True)
            return color_text(f" {sym} ", "white" if sym in {"o", "e"} else "cyan" if sym == "p" else "magenta" if sym == "r" else "yellow", bold=sym in {"p", "r"})

        def hconn(si: int, di: int) -> str:
            wa = win_at_cell.get((si, di))
            wb = win_at_cell.get((si, di + 1))
            if wa is None or wb is None:
                return "   "
            q = edge_quality_by_pair.get((min(wa, wb), max(wa, wb)))
            return self._dashboard_edge_connector(q or {"level": "untested"}, horizontal=True)

        def vconn(si_hi: int, di: int) -> str:
            # si_hi is the upper row index; lower row is si_hi - 1.
            wa = win_at_cell.get((si_hi, di))
            wb = win_at_cell.get((si_hi - 1, di))
            if wa is None or wb is None:
                return "   "
            q = edge_quality_by_pair.get((min(wa, wb), max(wa, wb)))
            return self._dashboard_edge_connector(q or {"level": "untested"}, horizontal=False)

        mode_name = str((sec_meta or {}).get("mode", getattr(self.args, "secondary_cv", "secondary")))
        lines.append(color_text(f"  y = secondary CV, x = {str((info or {}).get('primary_cv_label', 'primary CV'))} target; O/P/R+digit = occupied window by replica", "dim"))
        for si in range(len(sec_axis) - 1, -1, -1):
            label = self._acb_y_label(float(sec_axis[si]), mode_name, y_label_w)
            row = f"{label} "
            for di in range(len(dist_axis)):
                row += node_cell(si, di)
                if di + 1 < len(dist_axis):
                    row += hconn(si, di)
            lines.append(row.rstrip())
            if si > 0:
                vrow = " " * y_label_w + " "
                for di in range(len(dist_axis)):
                    vrow += vconn(si, di)
                    if di + 1 < len(dist_axis):
                        vrow += "   "
                lines.append(vrow.rstrip())

        # Compact x-axis labels. They are approximate because columns are symbolic.
        xrow = " " * y_label_w + " "
        for di, d in enumerate(dist_axis):
            raw = f"{float(d):.1f}"
            xrow += raw[:node_w].center(node_w)
            if di + 1 < len(dist_axis):
                xrow += "   "
        lines.append(color_text(xrow.rstrip(), "dim"))
        lines.append(color_text("  legend: o/O=base grid, p/P=local patch, r/R=retained patch, e/E=explicit/manual; edge --- OK, !!! weak, xxx bad, ... untested", "dim"))
        worst = sorted([q for q in edge_qualities if q.get("level") in {"bad", "warn"}], key=lambda q: (int(q.get("score", 3)), str(q.get("key", ""))))[:3]
        for q in worst:
            wi, wj = int(q.get("wi", -1)), int(q.get("wj", -1))
            try:
                where = f"({centers_a[wi]:.2f},{sec[wi]:+.2f})->({centers_a[wj]:.2f},{sec[wj]:+.2f})"
            except Exception:
                where = str(q.get("key", ""))
            level = str(q.get("level", "warn")).upper()
            col = "red" if level == "BAD" else "yellow"
            lines.append("  weak: " + color_text(f"{level} {q.get('key')} {q.get('reason')}", col, bold=True) + " " + where)
        return lines

    def _dashboard_decision_state(self, rows: list[dict], exchange_stats: dict, centers_a: list[float], summary: dict, phase: str, dashboard_info: Optional[dict]) -> dict:
        """Return fixed top-level TUI health and action guidance.

        This deliberately has no CLI switch: the dashboard should always answer
        the same operational question first -- is this run healthy enough to keep
        spending wall time on it, and what is the next useful action?
        """
        info = dashboard_info or {}
        issues: list[dict] = []

        def add_issue(level: str, label: str, detail: str, action: str) -> None:
            level = str(level or "watch").upper()
            if level not in {"OK", "WATCH", "BAD"}:
                level = "WATCH"
            issues.append({"level": level, "label": str(label), "detail": str(detail), "action": str(action)})

        # Basic per-replica finite/stuck checks.
        bad_finite = 0
        stuck_reps = []
        for r in rows or []:
            rep = int(r.get("replica", 0))
            try:
                cv = float(r.get("cv_A", float("nan")))
            except Exception:
                cv = float("nan")
            try:
                pe = float(r.get("potential_kj_mol", float("nan")))
            except Exception:
                pe = float("nan")
            if not (math.isfinite(cv) and math.isfinite(pe)):
                bad_finite += 1
            h = self.history_by_replica.get(rep, [])
            recent = list(h)[-10:]
            if len(recent) >= 10 and (max(recent) - min(recent)) < 0.05:
                stuck_reps.append(rep)
        if bad_finite:
            add_issue("BAD", "replica health", f"{bad_finite} replica rows have non-finite CV or potential energy", "inspect coordinates/checkpoint before continuing")
        if stuck_reps:
            shown = ",".join(f"r{x:02d}" for x in stuck_reps[:5])
            add_issue("WATCH", "replica motion", f"possible stuck CV trace for {shown}", "continue if early; otherwise inspect restraints/exchange")

        # Exchange acceptance checks. Use concrete pair stats when available.
        pairs = exchange_stats.get("pairs", {}) if isinstance(exchange_stats, dict) else {}
        weak_pairs = []
        bad_pairs = []
        for key, st in pairs.items():
            try:
                att = int(st.get("attempts", 0) or 0)
                acc = int(st.get("accepted", 0) or 0)
            except Exception:
                continue
            if att < 5:
                continue
            frac = acc / max(1, att)
            if frac < 0.04:
                bad_pairs.append((str(key), frac, acc, att))
            elif frac < 0.08:
                weak_pairs.append((str(key), frac, acc, att))
        if bad_pairs:
            key, frac, acc, att = sorted(bad_pairs, key=lambda x: x[1])[0]
            add_issue("BAD", "exchange", f"very low exchange on {key}: {100.0*frac:.1f}% ({acc}/{att})", "add/refine windows or lower local stiffness before trusting diffusion")
        elif weak_pairs:
            key, frac, acc, att = sorted(weak_pairs, key=lambda x: x[1])[0]
            add_issue("WATCH", "exchange", f"weak exchange on {key}: {100.0*frac:.1f}% ({acc}/{att})", "continue pilot or let adaptive feedback repair the weak edge")

        if isinstance(exchange_stats, dict) and str(exchange_stats.get("mode", "")).lower() == "gibbs-walk":
            choices = int(exchange_stats.get("gibbs_choices", 0) or 0)
            moves = int(exchange_stats.get("gibbs_moves", 0) or 0)
            if choices >= 10:
                move_frac = moves / max(1, choices)
                if move_frac < 0.05:
                    add_issue("WATCH", "gibbs-walk", f"low Gibbs movement {100.0*move_frac:.1f}% ({moves}/{choices})", "check overlap; heat-bath choices may be mostly staying put")

        # Histogram overlap along adjacent distance centers. This is still useful
        # for both 1D and rectangular/sparse 2D because it catches broken distance
        # coverage visible in the live histories.
        if centers_a and len(centers_a) > 1:
            lo, hi = min(centers_a), max(centers_a)
            if hi <= lo:
                hi = lo + 1.0
            poor_overlap = []
            for a in range(len(centers_a) - 1):
                ov = _hist_overlap(self.history_by_window.get(a, []), self.history_by_window.get(a + 1, []), lo, hi)
                if math.isfinite(ov) and ov < 0.12:
                    poor_overlap.append((a, a + 1, ov))
            if poor_overlap:
                a, b, ov = sorted(poor_overlap, key=lambda x: x[2])[0]
                level = "BAD" if ov < 0.05 else "WATCH"
                add_issue(level, "window overlap", f"low histogram overlap w{a:02d}-w{b:02d}: {ov:.2f}", "adaptive feedback should add/shift windows here")

        # CV coverage check.
        vals = []
        for h in self.history_by_replica.values():
            vals.extend(h)
        if centers_a and len(vals) > 100:
            target = max(1.0e-12, max(centers_a) - min(centers_a))
            span = max(vals) - min(vals)
            ratio = span / target
            if ratio < 0.35:
                add_issue("WATCH", "CV coverage", f"sampled span covers only {100.0*ratio:.0f}% of target range", "continue pilot or inspect overly stiff/end windows")

        # GaMD boost diagnostics.
        boost_sd = summary.get("gamd_boost_sd_kcal_mol", float("nan"))
        try:
            if math.isfinite(float(boost_sd)) and float(boost_sd) >= 6.0:
                add_issue("WATCH", "GaMD boost", f"boost SD {float(boost_sd):.2f} kcal/mol", "consider longer calibration or smaller sigma0 if this persists")
        except Exception:
            pass
        anh = summary.get("gamd_boost_anharmonicity_score", float("nan"))
        try:
            if math.isfinite(float(anh)):
                if float(anh) >= 1.0:
                    add_issue("BAD", "GaMD reweighting", f"boost anharmonicity {float(anh):.2f}", "do not trust cumulant reweighting until boost distribution improves")
                elif float(anh) >= 0.5:
                    add_issue("WATCH", "GaMD reweighting", f"boost anharmonicity {float(anh):.2f}", "watch reweighting quality; more sampling may help")
        except Exception:
            pass

        # Adaptive/sparse context. These are not necessarily errors, but they are
        # operationally important at the top of the dashboard.
        adaptive = info.get("adaptive_phase") or {}
        if adaptive.get("is_pilot"):
            add_issue("WATCH", "adaptive pilot", "pilot data are diagnostic only and will be discarded", "wait for final clean production before MBAR/PMF")
        elif adaptive.get("is_final"):
            # Keep final-production context visible without degrading OK health.
            pass

        # Sort by severity, preserving useful determinism inside each level.
        sev_rank = {"BAD": 0, "WATCH": 1, "OK": 2}
        issues_sorted = sorted(issues, key=lambda x: (sev_rank.get(x.get("level", "WATCH"), 1), x.get("label", "")))
        health = "OK"
        if any(x.get("level") == "BAD" for x in issues_sorted):
            health = "BAD"
        elif any(x.get("level") == "WATCH" for x in issues_sorted):
            health = "WATCH"

        if health == "OK":
            actions = ["continue; no major live-dashboard issues detected"]
            reasons = ["major live diagnostics look coherent"]
        else:
            actions = []
            reasons = []
            seen_actions = set()
            for issue in issues_sorted:
                reasons.append(f"{issue['label']}: {issue['detail']}")
                act = issue.get("action", "")
                if act and act not in seen_actions:
                    seen_actions.add(act)
                    actions.append(act)
                if len(actions) >= 4:
                    break
        return {
            "health": health,
            "issues": issues_sorted,
            "reasons": reasons[:5],
            "actions": actions[:4],
        }

    def _render_dashboard_decision_panel(self, decision: dict, term_w: int) -> list[str]:
        """Render an always-on health/action panel for the dashboard."""
        health = str((decision or {}).get("health", "OK")).upper()
        color = "green" if health == "OK" else "yellow" if health == "WATCH" else "red"
        issues = (decision or {}).get("issues", []) or []
        reasons = (decision or {}).get("reasons", []) or []
        actions = (decision or {}).get("actions", []) or []
        issue_count = len(issues)
        icon = "OK" if health == "OK" else "!!" if health == "WATCH" else "XX"
        body = [
            color_text(f"{icon} RUN HEALTH: {health}", color, bold=True) + f"  active issues: {issue_count}",
            color_text("meaning:", "dim") + " OK=keep sampling, WATCH=monitor/likely adaptive repair, BAD=inspect before trusting output",
        ]
        if reasons:
            for r in reasons[:3]:
                body.append("reason: " + str(r))
        else:
            body.append("reason: major live diagnostics look coherent")
        if actions:
            for a in actions[:3]:
                body.append("action: " + str(a))
        else:
            body.append("action: continue monitoring")
        return _dashboard_full_width_panel(
            "Run decision",
            "health + recommended action",
            body,
            term_w=term_w,
            max_body_lines=8,
        )

    def _dashboard_phase_explanation_lines(self, phase: str, adaptive: dict, display_step: int, display_total: Optional[int]) -> list[str]:
        """Short plain-language explanation of the current dashboard phase."""
        phase_l = str(phase or "").lower()
        lines: list[str] = []
        if adaptive and adaptive.get("is_pilot"):
            round_txt = f"round {adaptive.get('round', '?')}/{adaptive.get('rounds', '?')}"
            lines.append(
                color_text("NOW ", "cyan", bold=True)
                + color_text("adaptive pilot", "yellow", bold=True)
                + f" {round_txt}: short diagnostic sampling; these frames tune windows and are discarded before final PMF."
            )
            lines.append(
                color_text("NEXT ", "cyan", bold=True)
                + "adaptive feedback will look for weak overlap/exchange and add, shift, or remove windows."
            )
        elif adaptive and adaptive.get("is_final"):
            lines.append(
                color_text("NOW ", "cyan", bold=True)
                + color_text("final clean production", "green", bold=True)
                + ": these are the samples intended for MBAR/PMF after pilot repair."
            )
        elif "calibration" in phase_l:
            lines.append(
                color_text("NOW ", "cyan", bold=True)
                + color_text("GaMD calibration", "yellow", bold=True)
                + ": collecting potential-energy statistics so the boost can be set safely."
            )
        elif "production" in phase_l or "gareus" in phase_l:
            lines.append(
                color_text("NOW ", "cyan", bold=True)
                + color_text("GaREUS production", "green", bold=True)
                + ": replicas sample biased windows; exchanges test whether neighboring thermodynamic states overlap."
            )
        else:
            lines.append(
                color_text("NOW ", "cyan", bold=True)
                + f"phase {color_text(str(phase), 'white', bold=True)}: running current workflow stage."
            )
        if display_total:
            lines.append(
                color_text("PROGRESS ", "cyan", bold=True)
                + f"{int(display_step)}/{int(display_total)} steps in this segment; ETA is segment-local, not a promise from the gods."
            )
        return lines

    def _render_dashboard_context_panel(
        self,
        rows: list[dict],
        phase: str,
        display_step: int,
        display_total: Optional[int],
        summary: dict,
        info: dict,
        decision: dict,
        centers_a: list[float],
        exchange_stats: dict,
        term_w: int,
    ) -> list[str]:
        """Render a compact explanation panel for humans staring at the TUI."""
        adaptive = info.get("adaptive_phase") or {}
        body: list[str] = []
        body.extend(self._dashboard_phase_explanation_lines(phase, adaptive, display_step, display_total))

        primary_ctx = {
            "primary_cv": info.get("primary_cv", primary_cv_mode(self.args)),
            "primary_cv_label": info.get("primary_cv_label", primary_cv_label(self.args)),
            "primary_cv_units": info.get("primary_cv_units", primary_cv_units(self.args)),
            "contact_normalize": getattr(self.args, "contact_normalize", True),
        }
        plabel = str(info.get("primary_cv_label", primary_cv_label(primary_ctx)))
        mean_cv = float(summary.get("cv_mean_A", float("nan")))
        min_cv = float(summary.get("cv_min_A", float("nan")))
        max_cv = float(summary.get("cv_max_A", float("nan")))
        center_lo = min(centers_a) if centers_a else float("nan")
        center_hi = max(centers_a) if centers_a else float("nan")
        cv_triplet = (
            f"{format_primary_cv_value(min_cv, primary_ctx, 3)}.."
            f"{format_primary_cv_value(max_cv, primary_ctx, 3)}; "
            f"mean {format_primary_cv_value(mean_cv, primary_ctx, 3)}"
        )
        center_txt = (
            f"targets {format_primary_cv_value(center_lo, primary_ctx, 3)}.."
            f"{format_primary_cv_value(center_hi, primary_ctx, 3)}"
            if centers_a else "targets n/a"
        )
        if primary_cv_is_contacts(primary_ctx):
            scheme = str(getattr(self.args, "contact_scheme", "atom-pairs"))
            atom_sel = str(getattr(self.args, "contact_atom_selection", "heavy"))
            body.append(
                color_text("CV1 ", "magenta", bold=True)
                + color_text(plabel, "cyan", bold=True)
                + f" ({scheme}, {atom_sel}): low=few nonlocal residue contacts/extended; high=collapsed. {cv_triplet}; {center_txt}."
            )
        else:
            body.append(
                color_text("CV1 ", "magenta", bold=True)
                + color_text(plabel, "cyan", bold=True)
                + f": end-to-end extension/compaction. {cv_triplet}; {center_txt}."
            )

        sec_centers = [float(x) for x in info.get("secondary_cv_centers", []) if str(x) not in {"", "None", "nan"}]
        if sec_centers:
            mode_name = str((info.get("secondary_cv", {}) or {}).get("mode", getattr(self.args, "secondary_cv", "secondary")))
            if secondary_cv_mode(mode_name) == "rama-regions":
                body.append(
                    color_text("CV2 ", "magenta", bold=True)
                    + color_text("rama-regions", "cyan", bold=True)
                    + ": -1 beta/extended, -0.5 PPII, 0 turn/coil, +0.5 right-alpha, +1 left-alpha."
                )
            else:
                body.append(
                    color_text("CV2 ", "magenta", bold=True)
                    + color_text(mode_name, "cyan", bold=True)
                    + f": secondary-CV targets {min(sec_centers):+.2f}..{max(sec_centers):+.2f}."
                )

        vals = []
        for hist in self.history_by_replica.values():
            vals.extend(hist)
        if centers_a and len(vals) >= 5:
            target_span = max(1.0e-12, float(max(centers_a) - min(centers_a)))
            sample_span = max(vals) - min(vals)
            ratio = max(0.0, min(1.5, sample_span / target_span))
            cov_col = "green" if ratio >= 0.65 else "yellow" if ratio >= 0.35 else "red"
            body.append(
                color_text("COVERAGE ", "magenta", bold=True)
                + color_text(f"sampled {100.0*ratio:.0f}%", cov_col, bold=cov_col != "green")
                + f" of target CV1 span; narrow early pilots are normal, persistent red means windows/restraints need repair."
            )

        if isinstance(exchange_stats, dict):
            mode = str(exchange_stats.get("mode", "neighbor"))
            attempts = int(exchange_stats.get("attempts", 0) or 0)
            accepted = int(exchange_stats.get("accepted", 0) or 0)
            if attempts > 0:
                frac = accepted / max(1, attempts)
                ex_col = "green" if frac >= 0.20 else "yellow" if frac >= 0.08 else "red"
                body.append(
                    color_text("EXCHANGE ", "magenta", bold=True)
                    + f"mode {mode}; "
                    + color_text(f"{100.0*frac:.1f}%", ex_col, bold=ex_col != "green")
                    + f" accepted ({accepted}/{attempts}). Low exchange means poor overlap; adaptive feedback should patch it."
                )
            else:
                body.append(
                    color_text("EXCHANGE ", "magenta", bold=True)
                    + f"mode {mode}; no attempts yet. Wait for the first exchange interval before judging overlap."
                )

        health = str((decision or {}).get("health", "OK")).upper()
        hcol = "green" if health == "OK" else "yellow" if health == "WATCH" else "red"
        body.append(
            color_text("LEGEND ", "magenta", bold=True)
            + color_text("green", "green", bold=True) + "=OK, "
            + color_text("yellow", "yellow", bold=True) + "=watch, "
            + color_text("red", "red", bold=True) + "=fix/inspect; shaded bars=history, colored dots=current replicas; health="
            + color_text(health, hcol, bold=True)
            + "."
        )
        return _dashboard_full_width_panel(
            "What is happening",
            "short interpretation guide",
            body,
            term_w=term_w,
            max_body_lines=9,
        )

    def _render_dashboard(self, rows: list[dict], phase: str, step: int, total_steps: Optional[int], summary: dict, dashboard_info: Optional[dict]) -> str:
        info = dashboard_info or {}
        panel_mode = str(info.get("dashboard_panels", self.dashboard_panels) or self.dashboard_panels).lower()
        heavy_panels = bool(info.get("render_heavy_panels", True))
        centers_a = [float(x) for x in info.get("centers_a", [])]
        n_windows = int(info.get("n_windows", len(centers_a) or (max([int(r['window']) for r in rows]) + 1 if rows else 0)))
        exchange_stats = info.get("exchange_stats", {})

        # The MD sample step is absolute for analysis, but progress/ETA should be
        # based on the current production segment.  In article-A shared-GaMD runs
        # the calibration/setup has already happened before the dashboard appears;
        # including those setup steps in the denominator made pilot/adaptive ETA
        # look much better than reality.
        display_step = int(info.get("display_step", step) or 0)
        display_total = info.get("display_total_steps", total_steps)
        display_total = int(display_total) if display_total is not None else None
        frac = (float(display_step) / float(display_total)) if display_total else 0.0
        frac = max(0.0, min(1.0, frac))

        eta_start_wall = float(info.get("eta_start_wall", self.start_wall) or self.start_wall)
        elapsed = max(0.0, time.time() - eta_start_wall)
        eta = (elapsed * (1.0 - frac) / frac) if frac > 0.0 and display_total else None
        timestep_fs = float(getattr(self.args, "timestep_fs", 0.0) or 0.0)
        nrep = max(1, len(rows))
        sim_time_ns = (float(display_step) * timestep_fs / 1.0e6) if timestep_fs > 0 else float("nan")
        ns_day = (sim_time_ns / elapsed * 86400.0) if elapsed > 0 and math.isfinite(sim_time_ns) else float("nan")
        agg_ns_day = ns_day * nrep if math.isfinite(ns_day) else float("nan")
        progress_bar = make_progress_bar(frac, min(42, max(20, int(getattr(self.args, "progress_bar_width", 36) or 36))))
        perf_txt = f"  sim {sim_time_ns:.3g} ns/rep  perf {ns_day:.2g} ns/day/rep  {agg_ns_day:.2g} aggregate" if math.isfinite(ns_day) else ""
        term_size = shutil.get_terminal_size((160, 40))
        term_w = int(term_size.columns or 160)
        term_h = int(term_size.lines or 40)
        density = _dashboard_density(self.args, term_w=term_w, term_h=term_h)
        compact_dashboard = density == "compact"
        full_dashboard = density == "full"
        usable_w = max(40, term_w - 2)

        primary_label = str(info.get("primary_cv_label", primary_cv_label(self.args)))
        primary_units = str(info.get("primary_cv_units", primary_cv_units(self.args)))
        primary_k_units_text = str(info.get("primary_k_units", primary_k_units(self.args)))
        unit_suffix = "" if primary_units in {"", "dimensionless"} else f" {primary_units}"
        primary_mode = str(info.get("primary_cv", primary_cv_mode(self.args)))

        adaptive = info.get("adaptive_phase") or {}
        header = []
        if adaptive.get("is_pilot"):
            banner_text = "  !!!  ADAPTIVE PHASE / PILOT ONLY - DATA WILL BE DISCARDED AFTER WINDOW DIAGNOSTICS  !!!  "
            banner_w = max(40, min(term_w - 2, len(strip_ansi(banner_text)) + 4))
            banner = banner_text.center(banner_w, "!")
            header.append(style_text(banner, color="red", bold=True, bg256=52))
            header.append(color_text(
                f"adaptive pilot round {adaptive.get('round', '?')}/{adaptive.get('rounds', '?')}  "
                f"target overlap {float(adaptive.get('target_overlap', 0.30)):.2f}  "
                f"pilot {display_step}/{display_total or '?'} steps  "
                f"final target {adaptive.get('final_steps', '?')} steps",
                "red", bold=True,
            ))
            try:
                wf_start = float(adaptive.get("workflow_start_wall", eta_start_wall))
                wf_done = float(adaptive.get("workflow_done_before", 0.0)) + float(display_step)
                wf_total = float(adaptive.get("workflow_total_steps", 0.0))
                wf_frac = max(0.0, min(1.0, wf_done / wf_total)) if wf_total > 0 else 0.0
                wf_elapsed = max(0.0, time.time() - wf_start)
                wf_eta = (wf_elapsed * (1.0 - wf_frac) / wf_frac) if wf_frac > 0 else None
                header.append(color_text(
                    f"adaptive workflow rough ETA {format_duration(wf_eta)}  "
                    f"({wf_done:.0f}/{wf_total:.0f} production-step equivalents including final run)",
                    "yellow", bold=True,
                ))
            except Exception:
                pass
        elif adaptive.get("is_final"):
            header.append(color_text(
                "FINAL CLEAN PRODUCTION after adaptive-feedback pilots - use this directory for MBAR/PMF",
                "green", bold=True,
            ))

        header.extend([
            color_text("GaREUS dashboard", "magenta", bold=True) + f"  phase {phase}  [{progress_bar}]  {display_step}/{display_total or '?'}  {100*frac:5.1f}%  wall {format_duration(elapsed)}  eta {format_duration(eta)}" + perf_txt,
            f"{primary_label} {summary.get('cv_min_A', float('nan')):6.2f}–{summary.get('cv_max_A', float('nan')):6.2f}{unit_suffix}  mean {summary.get('cv_mean_A', float('nan')):6.2f}{unit_suffix}  k {summary.get('k_min_kcal_mol_A2', float('nan')):.3g}–{summary.get('k_max_kcal_mol_A2', float('nan')):.3g} {primary_k_units_text}",
        ])
        if "secondary_cv_mean" in summary:
            header.append(
                f"secondary CV {summary.get('secondary_cv_min', float('nan')):+6.2f}–{summary.get('secondary_cv_max', float('nan')):+6.2f}  "
                f"mean {summary.get('secondary_cv_mean', float('nan')):+6.2f}"
            )
        sec_meta = info.get("secondary_cv", {}) or {}
        n_secondary_targets = len([x for x in info.get("secondary_cv_centers", []) if str(x) not in {"", "None", "nan"}])
        explicit_2d = bool(sec_meta.get("explicit_2d_windows", info.get("explicit_2d", False)))
        rectangular_2d = bool(sec_meta.get("grid", info.get("rectangular_2d", False))) if explicit_2d else bool(n_secondary_targets > 1 and n_windows == max(1, len(set(round(float(x), 4) for x in centers_a))) * max(1, n_secondary_targets))
        sparse_2d = bool(explicit_2d and not rectangular_2d)
        topology_label = "sparse explicit 2D" if sparse_2d else "rectangular 2D" if n_secondary_targets > 1 else "1D/secondary-fixed"
        header.append(
            color_text("layout", "dim") + f" {density} {term_w}x{term_h}  "
            + color_text("windows", "dim") + f" {n_windows}  "
            + color_text("topology", "dim") + f" {topology_label}  "
            + color_text("secondary targets", "dim") + f" {n_secondary_targets}"
        )
        lo = min(centers_a) if centers_a else summary.get("cv_min_A", 0.0)
        hi = max(centers_a) if centers_a else summary.get("cv_max_A", 1.0)
        vals = []
        for h in self.history_by_replica.values():
            vals.extend(h)
        coverage_w = max(12, min(64, max(12, usable_w - len(f"coverage {lo:.1f} A || {hi:.1f} A") - 2)))
        header.append(f"coverage {lo:.2f}{unit_suffix} |{_coverage_bar(vals, lo, hi, coverage_w)}| {hi:.2f}{unit_suffix}")
        header = [_ansi_truncate(line, usable_w) for line in header]
        decision = self._dashboard_decision_state(rows, exchange_stats, centers_a, summary, phase, info)

        # Row 1 uses asymmetric side-by-side panels: a wide CV map plus a
        # narrower potential-energy map.  Keeping PE beside CV makes it easy to
        # notice a replica whose coordinate looks fine while its energy is doing
        # modern art.
        usable_w = max(40, term_w - 2)
        row_gap = 2 if term_w < 120 else 3
        min_panel_w = max(24, int(getattr(self.args, "dashboard_min_panel_width", 30) or 30))
        wide_threshold = int(getattr(self.args, "dashboard_wide_threshold", 132) or 132)
        pe_panel_w = usable_w if usable_w < 105 else max(34, min(58, usable_w // 3))
        cv_panel_w = usable_w if usable_w < 105 else max(50, usable_w - pe_panel_w - row_gap)
        cv_bar_width = max(24, min(140 if full_dashboard else 110, cv_panel_w - 62))
        pe_bar_width = max(10, min(36 if full_dashboard else 28, pe_panel_w - 24))
        row1_max = (10 if compact_dashboard else 18 if density == "normal" else 30)
        row23_max = (8 if compact_dashboard else 12 if density == "normal" else 18)
        row4_max = (8 if compact_dashboard else 14 if density == "normal" else 18)

        hist_lines = []
        if self.ascii_mode != "none":
            hist_lines = render_distance_ascii(
                rows, phase=phase, step=step, total_steps=total_steps,
                width=cv_bar_width, max_replicas=self.ascii_max_replicas,
                mode=self.ascii_mode,
                history_by_replica=self.history_by_replica,
                history_by_window=self.history_by_window,
                histogram_source="window",
                primary_label=str(info.get("primary_cv_label", "terminal distance")),
                primary_units=str(info.get("primary_cv_units", "A")),
                primary_k_unit_label=str(info.get("primary_k_units", "kcal/mol/A^2")),
            ).splitlines()
            # The histogram block already contains its own big title; remove the
            # very top border/title lines when embedded as a panel to save space.
            if len(hist_lines) > 4:
                hist_lines = hist_lines[3:]

        sections = []
        sections.extend(header)
        sections.append("")
        sections.extend(self._render_dashboard_context_panel(
            rows, phase, display_step, display_total, summary, info, decision, centers_a, exchange_stats, term_w=term_w,
        ))
        sections.append("")
        sections.extend(self._render_dashboard_decision_panel(decision, term_w=term_w))
        sections.append("")
        if panel_mode == "minimal":
            # Keep the live TUI cheap: header + explanation + decision + compact CV map.
            if self.ascii_mode != "none":
                sections.extend(_dashboard_full_width_panel(
                    "CV map",
                    "minimal live dashboard",
                    hist_lines or [color_text("disabled", "dim")],
                    term_w=term_w,
                    max_body_lines=row1_max,
                ))
            return "\n".join(sections)

        topology_lines = self._render_2d_sparse_topology_map(rows, info, exchange_stats, width=usable_w) if heavy_panels else []
        if topology_lines:
            sections.extend(_dashboard_full_width_panel(
                "2D topology",
                "windows + 2D graph-edge overlap/exchange",
                topology_lines[1:],
                term_w=term_w,
                max_body_lines=(10 if compact_dashboard else 18 if density == "normal" else 26),
            ))
            sections.append("")

        sections.extend(_dashboard_weighted_row(
            "1/4 CV + potential-energy maps",
            [
                ("actual per-window CV distributions", hist_lines or [color_text("disabled", "dim")], 2.4),
                ("potential energy histograms", self._render_potential_energy_map(rows, width=pe_bar_width)[1:], 0.9),
            ],
            term_w=term_w,
            gap=row_gap,
            max_panel_body_lines=max(row1_max, min(32, int(getattr(self.args, "distance_ascii_max_replicas", 32) or 32) + 4)),
            min_panel_width=min_panel_w,
        ))
        sections.append("")

        # Wide terminals (≥140 cols): the two 2D panels share a row side-by-side,
        # cutting height by ~18 lines.  Narrow terminals fall back to stacked layout.
        wide_terminal = (term_w >= wide_threshold) and not compact_dashboard
        diff2d_w = max(50, (usable_w - row_gap) * 56 // 100) if wide_terminal else usable_w
        map2d_w = max(44, usable_w - diff2d_w - row_gap) if wide_terminal else usable_w

        diff2d_lines = self._render_2d_diffusion_map(rows, info, width=diff2d_w) if heavy_panels else []
        map2d_lines = self._render_2d_replica_map(rows, info, width=map2d_w) if heavy_panels else []

        n_sec = len(self._unique_axis_values([float(x) for x in info.get("secondary_cv_centers", []) if str(x) not in {"", "None", "nan"}], ndigits=4))
        map2d_max_body = max(10, min(24, n_sec + 8))

        if wide_terminal and diff2d_lines and map2d_lines:
            diff_panel = _panel_lines(
                f"log-density: {primary_label} × secondary-CV  (░▒▓█ = increasing density)",
                diff2d_lines[1:],
                diff2d_w,
                max_body_lines=(14 if compact_dashboard else 22 if density == "normal" else 28),
            )
            map_panel = _panel_lines(
                f"{primary_label} × secondary-CV replica grid",
                map2d_lines[1:],
                map2d_w,
                max_body_lines=max(map2d_max_body, 14 if compact_dashboard else 22),
            )
            sections.append(color_text("2D landscape", "magenta", bold=True))
            sections.extend(_join_columns([diff_panel, map_panel], gap=row_gap))
            sections.append("")
        else:
            if diff2d_lines:
                sections.extend(_dashboard_full_width_panel(
                    "2D sampling diffusion",
                    f"log-density: {primary_label} × secondary-CV  (░▒▓█ = increasing sample density)",
                    diff2d_lines[1:],
                    term_w=term_w,
                    max_body_lines=(14 if compact_dashboard else 22 if density == "normal" else 28),
                ))
                sections.append("")
            if map2d_lines:
                sections.extend(_dashboard_full_width_panel(
                    "2D replica mapping",
                    f"{str(info.get('primary_cv_label', 'primary CV'))} × secondary-CV grid",
                    map2d_lines[1:],
                    term_w=term_w,
                    max_body_lines=min(map2d_max_body, 14) if compact_dashboard else map2d_max_body,
                ))
                sections.append("")

        # Row 2: immediate production diagnostics. Replica-health and raw-PMF
        # panels are intentionally no longer shown as standalone panels; the
        # always-on Run decision panel handles health, and PMF previews belong in
        # post-processing rather than the live TUI.
        sections.extend(_dashboard_weighted_row(
            "2/4 GaMD boost diagnostics",
            [
                ("GaMD boost", self._render_gamd_boost(rows, summary, phase)[1:], 1.0),
            ],
            term_w=term_w,
            gap=row_gap,
            max_panel_body_lines=row23_max,
            min_panel_width=min_panel_w,
        ))
        sections.append("")

        # Row 3: window coupling diagnostics. In 2D mode, graph-edge histogram
        # overlap is integrated into the 2D topology panel above; the legacy
        # flattened 1D overlap panel is shown only for non-2D runs.
        two_d_dashboard_mode = bool(topology_lines)
        coupling_panels = [
            ("exchange acceptance", self._render_exchange_acceptance(exchange_stats, n_windows)[1:], 1.15),
            ("umbrella pull", self._render_pull_map(rows, n_windows)[1:], 1.25),
        ]
        if not two_d_dashboard_mode:
            coupling_panels.insert(1, ("histogram overlap", self._render_overlap(centers_a)[1:], 0.95))
        sections.extend(_dashboard_weighted_row(
            "3/4 window coupling" + (" + 2D graph overlap" if two_d_dashboard_mode else ""),
            coupling_panels,
            term_w=term_w,
            gap=row_gap,
            max_panel_body_lines=row23_max,
            min_panel_width=min_panel_w,
        ))
        sections.append("")

        # Row 4: all-replica combined table (health + boost + diffusion).
        # Measure natural line width at ncols=1, then compute how many columns
        # fit in the available terminal width before re-rendering.
        single_table = self._render_replica_table(rows, phase, n_windows, ncols=1)
        line_w = max((strip_ansi_len(l) for l in single_table), default=78) if single_table else 78
        # panel border consumes 2 chars; each additional column needs line_w + 3 gap
        inner_avail = usable_w - 2
        ncols_rep = max(1, (inner_avail + 3) // (line_w + 3))
        rep_table = (
            single_table if ncols_rep <= 1
            else self._render_replica_table(rows, phase, n_windows, ncols=ncols_rep)
        )
        # Only expand row height beyond row4_max in non-compact mode so compact
        # dashboards don't grow past the terminal.
        if compact_dashboard:
            row4_max_body = row4_max
        else:
            row4_max_body = max(row4_max, math.ceil(len(rows) / max(1, ncols_rep)) + 1)
        row4_panels = [("replica motion + health", rep_table)]
        rec_lines = self._render_recommendations(exchange_stats, centers_a, summary)
        if rec_lines:
            row4_panels.append(("recommendations", rec_lines[1:]))
        weighted_row4 = [(name, body, 1.5 if name == "replica motion + health" else 1.0) for name, body in row4_panels]
        sections.extend(_dashboard_weighted_row(
            "4/4 motion + recommendations",
            weighted_row4,
            term_w=term_w,
            gap=row_gap,
            max_panel_body_lines=row4_max_body,
            min_panel_width=min_panel_w,
        ))

        return "\n".join(sections)

    def log(self, rows: list[dict], phase: str, step: int, total_steps: Optional[int] = None, dashboard_info: Optional[dict] = None) -> dict:
        if not rows:
            return {}
        n_windows = None
        if dashboard_info:
            n_windows = int(dashboard_info.get("n_windows", len(dashboard_info.get("centers_a", [])) or 0) or 0)
        clean_rows = []
        for row in rows:
            try:
                clean = {
                    "replica": int(row["replica"]),
                    "window": int(row["window"]),
                    "center_A": float(row["center_A"]),
                    "k_kcal_mol_A2": float(row["k_kcal_mol_A2"]),
                    "cv_A": float(row["cv_A"]),
                }
                delta_a = clean["cv_A"] - clean["center_A"]
                clean["umbrella_bias_kcal_mol"] = float(row.get("umbrella_bias_kcal_mol", 0.5 * clean["k_kcal_mol_A2"] * delta_a * delta_a))
                clean["umbrella_pull_kcal_mol_A"] = float(row.get("umbrella_pull_kcal_mol_A", clean["k_kcal_mol_A2"] * (clean["center_A"] - clean["cv_A"])))
                for key in (
                    "potential_kj_mol", "gamd_boost_total_kj_mol", "gamd_boost_total_kcal_mol",
                    "primary_cv", "primary_cv_label", "primary_cv_units",
                    "primary_cv_value", "primary_cv_center", "primary_cv_k", "primary_cv_k_units",
                    "primary_umbrella_bias_kcal_mol",
                    "secondary_cv", "secondary_cv_center", "secondary_cv_k_kcal_mol",
                    "distance_umbrella_bias_kcal_mol", "secondary_cv_bias_kcal_mol",
                ):
                    if key in row and row[key] not in (None, ""):
                        try:
                            clean[key] = float(row[key])
                        except Exception:
                            clean[key] = row[key]
                clean_rows.append(clean)
            except Exception:
                continue
        if not clean_rows:
            return {}
        self._maybe_reset_cv_history_for_phase(phase)
        self._update_history(clean_rows, n_windows=n_windows, phase=phase)
        self.last_rows = clean_rows
        summary = self.summarize(clean_rows)
        summary.update(self._boost_anharmonicity_summary(phase))
        self.last_summary = summary
        event = {
            "event": "distances",
            "phase": phase,
            "step": int(step),
            "total_steps": int(total_steps) if total_steps is not None else None,
            **summary,
            "distances": clean_rows,
        }
        if dashboard_info:
            event["dashboard"] = {
                "exchange_stats": dashboard_info.get("exchange_stats", {}),
                "n_windows": n_windows,
                "secondary_cv": dashboard_info.get("secondary_cv", {}),
                "secondary_cv_centers": dashboard_info.get("secondary_cv_centers", []),
            }
        if self.csv_writer is not None:
            for r in clean_rows:
                row = {"step": step, "phase": phase, **r, **summary}
                self.csv_writer.writerow(row)
            if bool(getattr(self.args, "flush_every_log", True)):
                self.csv_handle.flush()
        if self.jsonl_handle is not None:
            self.jsonl_handle.write_json(event)
        if self.progress is not None and not self.no_gui:
            self.progress.emit(event)
        if str(getattr(self.args, "tui_mode", "dashboard")) != "none":
            tui_mode = str(getattr(self.args, "tui_mode", "dashboard") or "dashboard").lower()
            if tui_mode in {"dashboard", "interactive"}:
                now = time.time()
                final_frame = bool(total_steps is not None and int(step) >= int(total_steps))
                render_due = (
                    final_frame
                    or self.dashboard_render_interval_sec <= 0.0
                    or (now - self._last_dashboard_render_wall) >= self.dashboard_render_interval_sec
                )
                if not render_due:
                    return summary
                self._last_dashboard_render_wall = now
                self._dashboard_render_count += 1
                if dashboard_info is None:
                    dashboard_info = {}
                else:
                    dashboard_info = dict(dashboard_info)
                dashboard_info["dashboard_panels"] = self.dashboard_panels
                dashboard_info["render_heavy_panels"] = bool(self._dashboard_render_count % self.dashboard_heavy_panels_every == 0)
                # Offload CPU-bound ASCII rendering to a single background thread so
                # the production loop never waits for TUI string work.  Drop the frame
                # if the previous render hasn't finished yet — one missed frame is
                # invisible to the user and keeps render/simulation decoupled.
                if self._pending_render is None or self._pending_render.done():
                    if self._render_executor is None:
                        self._render_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
                    _snap = (
                        list(clean_rows), phase, step, total_steps,
                        dict(summary), dict(dashboard_info) if dashboard_info else None,
                    )
                    def _do_render(snap):
                        rows_s, phase_s, step_s, total_s, summary_s, info_s = snap
                        block = self._render_dashboard(rows_s, phase_s, step_s, total_s, summary_s, info_s)
                        if block:
                            write_tui_frame(block, self.args)
                    self._pending_render = self._render_executor.submit(_do_render, _snap)
            elif self.ascii_mode != "none":
                block = render_distance_ascii(
                    clean_rows, phase=phase, step=step, total_steps=total_steps,
                    width=self.ascii_width, max_replicas=self.ascii_max_replicas,
                    mode=self.ascii_mode,
                    history_by_replica=self.history_by_replica,
                    history_by_window=self.history_by_window,
                    histogram_source="window",
                    primary_label=str((dashboard_info or {}).get("primary_cv_label", "terminal distance")),
                    primary_units=str((dashboard_info or {}).get("primary_cv_units", "A")),
                    primary_k_unit_label=str((dashboard_info or {}).get("primary_k_units", "kcal/mol/A^2")),
                )
                if block:
                    write_tui_frame(block, self.args)
        return summary

__all__ = [
    "DistanceLogger",
    "boost_anharmonicity",
    "anharmonicity_label",
    "is_gamd_production_phase",
]
