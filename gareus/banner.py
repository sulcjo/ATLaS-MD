from __future__ import annotations

import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from . import __version__ as PACKAGE_VERSION
from .branding import PRODUCT_NAME
from .colors import ROLE_BAD, role_text, style_text
from .provenance import _dist_version, _optional_module_version

WIDTH = 66
WATER_ROW = 8
ROWS = WATER_ROW + 1
NEAR_BASE = 7
NEAR = [3, 4, 2, 6, 3, 4, 3, 3, 3, 2]
REPO_URL = "github.com/sulcjo/ATLaS-MD"

# 5x7 pixel font, one full block per pixel: solid strokes and a shared
# baseline read far better in a log tail than thin figlet strokes (whose
# lowercase "a" dropped a row below the capitals). Add a letter here to use it.
_PIXEL_FONT = {
    "A": [".###.", "#...#", "#...#", "#####", "#...#", "#...#", "#...#"],
    "T": ["#####", "..#..", "..#..", "..#..", "..#..", "..#..", "..#.."],
    "L": ["#....", "#....", "#....", "#....", "#....", "#....", "#####"],
    "a": [".....", ".....", ".###.", "....#", ".####", "#...#", ".####"],
    "S": [".####", "#....", "#....", ".###.", "....#", "....#", "####."],
    "-": [".....", ".....", ".....", "####.", ".....", ".....", "....."],
    "M": ["#...#", "##.##", "#.#.#", "#.#.#", "#...#", "#...#", "#...#"],
    "D": ["####.", "#...#", "#...#", "#...#", "#...#", "#...#", "####."],
}
_FONT_ROWS = 7
# Left-to-right 256-colour gradient (cyan -> blue) for the wordmark when colour is on.
_WORDMARK_GRADIENT = (51, 45, 39, 33, 27)


def _num(value: Any) -> str:
    v = float(value)
    s = f"{v:.1f}"
    return s if float(s) == v else f"{v:.3g}"


def _sci(value: float) -> str:
    mant, exp = f"{float(value):.2e}".split("e")
    return f"{mant}e{int(exp)}"


def _dur(seconds: float) -> str:
    total_min = int(max(0.0, seconds)) // 60
    d, rem = divmod(total_min, 1440)
    h, m = divmod(rem, 60)
    if d:
        return f"{d} d {h} h"
    if h:
        return f"{h} h {m} m"
    return f"{m} m"


def _wordmark(name: str, gap: int = 1) -> list[str]:
    rows = [""] * _FONT_ROWS
    for k, ch in enumerate(name):
        glyph = _PIXEL_FONT[ch]
        for r in range(_FONT_ROWS):
            rows[r] += ("." * gap if k else "") + glyph[r]
    return ["".join("█" if px == "#" else " " for px in row).rstrip() for row in rows]


def _colour_wordmark(line: str) -> str:
    """Tint each block by its column, so the word reads as one gradient."""
    out = []
    for col, ch in enumerate(line):
        if ch == " ":
            out.append(ch)
            continue
        shade = _WORDMARK_GRADIENT[min(len(_WORDMARK_GRADIENT) - 1,
                                       col * len(_WORDMARK_GRADIENT) // WIDTH)]
        out.append(style_text(ch, fg256=shade, bold=True))
    return "".join(out)


def _peak_cells(cx: int, h: int, base_row: int):
    for t in range(h):
        row = base_row - h + t
        for col in (cx - 1 - t, cx + t):
            if col >= 0:
                yield row, col


def _scene(trace_end: int) -> list[str]:
    grid = [[" "] * WIDTH for _ in range(ROWS)]
    cx = None
    prev_h = 0
    for h in NEAR:
        cx = h if cx is None else cx + prev_h + h
        for r, c in _peak_cells(cx, h, NEAR_BASE):
            if c < WIDTH:
                grid[r][c] = "/" if c < cx else "\\"
        prev_h = h
    top = [ROWS] * WIDTH
    for c in range(WIDTH):
        for r in range(ROWS):
            if grid[r][c] != " ":
                top[c] = r
                break
    for j in range(2, WIDTH, 7):
        grid[WATER_ROW][j] = "~"
    for x in range(trace_end):
        if 0 < top[x] < WATER_ROW and x % 3 == 0 and grid[top[x] - 1][x] == " ":
            grid[top[x] - 1][x] = "-"
    hy = max(top[trace_end] - 1, 0)
    grid[hy][trace_end] = "@"
    return ["".join(r).rstrip() for r in grid]


def _read_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return None


def _parse_iso(value: Any) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        return None


def _git_short() -> Optional[str]:
    root = Path(__file__).resolve().parents[1]
    try:
        out = subprocess.run(["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=3)
        if out.returncode == 0:
            return out.stdout.strip() or None
    except Exception:
        pass
    return None


def _gpu_name() -> Optional[str]:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=3)
        if out.returncode == 0 and out.stdout.strip():
            name = out.stdout.strip().splitlines()[0].strip()
            return re.sub(r"^(NVIDIA|AMD|Intel)\s+(GeForce|Radeon\s+Pro)?\s*", "", name) or name
    except Exception:
        pass
    return None


def _hw_row(args: Any) -> str:
    platform = str(getattr(args, "platform", "") or "CPU")
    threads = getattr(args, "cpu_threads", 0) or 0
    if platform.upper().startswith("CUDA"):
        gpu = _gpu_name()
        if gpu:
            return f"CUDA ok ({gpu}), {threads} threads/replica"
        return f"CUDA requested, {threads} threads/replica"
    return f"{platform}, {threads} threads/replica"


def _physics_row(args: Any) -> str:
    run_mode = str(getattr(args, "run_mode", "") or "")
    hmr = "on" if run_mode.startswith("hmr-") else "off"
    return (f"{getattr(args, 'forcefield', '')} / {getattr(args, 'water_model', '')} / "
            f"dt {_num(getattr(args, 'timestep_fs', 0.0))} fs / HMR {hmr} ({run_mode})")


def _cv1_row(args: Any) -> str:
    cv1 = str(getattr(args, "cv1", "") or "")
    if cv1 != "contacts":
        return f"{cv1} cv"
    beta = _num(getattr(args, "contact_beta_a_inv", 0.0))
    r0 = _num(getattr(args, "contact_r0_a", 0.0))
    lo = _num(getattr(args, "cv1_range_min", 0.0))
    hi = _num(getattr(args, "cv1_range_max", 0.0))
    return f"heavy contacts, b0 {beta} 1/A, midpoint {r0} A, {lo}-{hi}"


def _cv2_row(args: Any) -> str:
    cv2 = str(getattr(args, "cv2", "") or "")
    if cv2 != "rama-map":
        return cv2
    lo = int(getattr(args, "swarm_min_rungs", 3) or 3)
    hi = int(getattr(args, "swarm_max_rungs", 12) or 12)
    return f"{cv2} lambda-ladder, rungs {lo}-{hi}"


def _prov_row() -> str:
    parts = []
    g = _git_short()
    if g:
        parts.append(f"git {g}")
    openmm = _dist_version("openmm")
    if openmm:
        parts.append(f"openmm {openmm}")
    parts.append(f"python {sys.version.split()[0]}")
    return ", ".join(parts)


def _prev_end_row(out_dir: Path) -> Optional[str]:
    m = _read_json(Path(out_dir) / "run_manifest.json")
    if not m:
        return None
    end = _parse_iso(m.get("end_time_utc") or m.get("last_updated_utc"))
    if end is None:
        return None
    text = end.astimezone().strftime("%Y-%m-%d %H:%M")
    start = _parse_iso(m.get("start_time_utc"))
    if start is not None:
        text += f" (run {_dur((end - start).total_seconds())})"
    return text


def _ckpt_row(out_dir: Path) -> Optional[str]:
    out_dir = Path(out_dir)
    frags = []
    ap = out_dir / "adaptive_production"
    epochs = len([d for d in ap.glob("epoch_*") if d.is_dir()]) if ap.is_dir() else 0
    summary = _read_json(ap / "adaptive_production_driver_summary.json")
    if summary:
        try:
            epochs = max(epochs, int(summary.get("epochs_completed") or 0))
        except (TypeError, ValueError):
            pass
    if epochs:
        frags.append(f"epoch {epochs}")
    feedback = _read_json(out_dir / "adaptive_feedback_driver_summary.json")
    if feedback:
        rounds = feedback.get("rounds")
        if isinstance(rounds, list) and rounds:
            frags.append(f"pilot round {len(rounds)}")
    steps = 0
    for seg_path in ap.glob("epoch_*/segments.json"):
        data = _read_json(seg_path)
        for seg in data or []:
            try:
                steps = max(steps, int(seg.get("end_step") or 0))
            except (TypeError, ValueError):
                pass
    if steps:
        frags.append(f"prod step {_sci(steps)}")
    return " / ".join(frags) if frags else None


def _panel_rows(args: Any, out_dir: Path, resume: bool, mbar_version: Optional[str]) -> list[tuple[str, str]]:
    version = PACKAGE_VERSION or _dist_version("gareus-peptide") or "dev"
    rows = [
        ("campaign", f"{Path(out_dir).name}   version  v{version}   date  {datetime.now().strftime('%Y-%m-%d')}"),
        ("hw", _hw_row(args)),
        ("mbar", f"pymbar {mbar_version} ok" if mbar_version else "MISSING: pip install pymbar (PMF/MBAR analysis off)"),
        ("physics", _physics_row(args)),
        ("cv1", _cv1_row(args)),
        ("cv2", _cv2_row(args)),
    ]
    seed_dir = str(getattr(args, "seed_conformers_dir", "") or "").strip()
    if seed_dir:
        rows.append(("seeds", f"{seed_dir} (genpept)"))
    rows.append(("prov", _prov_row()))
    rows.append(("repo", REPO_URL))
    if resume:
        insert = []
        prev_end = _prev_end_row(out_dir)
        if prev_end:
            insert.append(("prev end", prev_end))
        ckpt = _ckpt_row(out_dir)
        if ckpt:
            insert.append(("ckpt", ckpt))
        rows[1:1] = insert
    return rows


def banner_lines(args: Any, out_dir: Path, *, resume: bool = False,
                 mbar_version: Optional[str] = None) -> list[str]:
    mark = _wordmark(PRODUCT_NAME)
    indent = " " * max(0, (WIDTH - max(len(r) for r in mark)) // 2)
    lines = [_colour_wordmark(indent + r) for r in mark]
    lines.append("")
    subtitle = "resuming run" if resume else "starting run"
    lines.append(subtitle.center(WIDTH).rstrip())
    lines.append("")
    lines += _scene(63 if resume else 22)
    lines.append("")
    inner = WIDTH - 2
    lines.append("+" + "-" * inner + "+")
    for label, value in _panel_rows(args, out_dir, resume, mbar_version):
        content = f" {label:<9}{value}".ljust(inner)[:inner]
        if label == "mbar" and not mbar_version:
            content = role_text(content, ROLE_BAD)
        lines.append("|" + content + "|")
    lines.append("+" + "-" * inner + "+")
    return lines


def print_startup_banner(args: Any, out_dir: Path, *, resume: bool = False) -> None:
    if not sys.stdout.isatty():
        return
    try:
        lines = banner_lines(args, out_dir, resume=resume,
                             mbar_version=_optional_module_version("pymbar"))
    except Exception:
        return
    print("\n".join(lines))
