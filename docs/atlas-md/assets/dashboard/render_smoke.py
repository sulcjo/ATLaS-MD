"""Smoke render of the ATLaS-MD dashboard to PNG (throwaway).

Real chignolin_9 window layout (236 windows, 16 CV1 x 5 CV2, 4 lambda rungs,
from epoch_000/umbrella_explicit_windows.csv); live samples and exchange stats
are SYNTHETIC. Colour forced on, ANSI drawn cell-by-cell with PIL.
"""
import argparse
import csv
import pathlib
import random
import re
import sys

REPO = pathlib.Path("/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler")
sys.path.insert(0, str(REPO))
HERE = pathlib.Path(__file__).parent

from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from gareus import colors  # noqa: E402
from gareus.banner import banner_lines  # noqa: E402
from gareus.dashboard.context import build_context  # noqa: E402
from gareus.dashboard.ranking import restraint_sigma  # noqa: E402
from gareus.dashboard.screen import render_screen  # noqa: E402
from gareus.dashboard.sidecar import SidecarSnapshot  # noqa: E402
from gareus.logger import DistanceLogger  # noqa: E402

TERM_W, TERM_H = 150, 52


def _context(view: str):
    rows_csv = list(csv.DictReader(open(HERE / "c9_windows.csv")))
    n = len(rows_csv)
    centers = [float(r["primary_center"]) for r in rows_csv]
    sec = [float(r["secondary_cv_center"]) for r in rows_csv]
    lam = [float(r["gamd_lambda"]) for r in rows_csv]
    k = [float(r["primary_k"]) for r in rows_csv]
    k2 = [float(r["secondary_cv_k_kcal_mol"]) for r in rows_csv]
    rng = random.Random(9)
    run_dir = pathlib.Path("/tmp/chignolin_9")
    logger = DistanceLogger(run_dir, argparse.Namespace(timestep_fs=4.0), no_file_persistence=True)
    unsampled = {37, 150, 233}
    for w in range(n):
        if w in unsampled:
            continue
        # Spread = the window's own restraint width sqrt(kT/k); unrestrained axes wander.
        s1 = restraint_sigma(k[w], 300.0) if k[w] > 0 else 0.25
        s2 = restraint_sigma(k2[w], 300.0) if k2[w] > 0 else 1.0
        logger.history_by_window[w] = [centers[w] + rng.gauss(0.0, 0.8 * s1) for _ in range(60)]
        logger.secondary_history_by_window[w] = [sec[w] + rng.gauss(0.0, 0.8 * s2) for _ in range(60)]
        logger.potential_history_by_replica[w] = [-31500.0 + rng.gauss(0, 40) for _ in range(40)]
    logger.boost_history_all = [2.0 + abs(rng.gauss(0.0, 1.0)) for _ in range(400)]
    rows = [{"replica": w, "window": w, "center_A": centers[w], "k_kcal_mol_A2": k[w],
             "cv_A": centers[w] + rng.gauss(0.0, 0.02), "gamd_lambda": lam[w],
             "umbrella_bias_kcal_mol": 0.2, "umbrella_pull_kcal_mol_A": 0.1,
             "potential_kj_mol": -31500.0, "gamd_boost_total_kcal_mol": 2.1,
             "secondary_cv_k_kcal_mol": k2[w]} for w in range(n)]
    dead = {61, 118, 177}
    pairs = {f"{i}-{i+1}": {"attempts": 600, "accepted": 0 if i in dead else int(600 * rng.uniform(0.18, 0.5))}
             for i in range(n - 1)}
    info = {"centers_a": centers, "n_windows": n, "k_list": k,
            "secondary_cv_centers": sec,
            "secondary_cv": {"explicit_2d_windows": True, "grid": False, "type": "residual-torsion-pc"},
            "exchange_stats": {"attempts": 600 * (n - 1), "accepted": 200 * (n - 1),
                               "mode": "gibbs", "pairs": pairs},
            "primary_cv_label": "nonlocal contacts", "primary_cv_units": "",
            "primary_k_units": "kcal/mol/CV^2",
            "adaptive_phase": {"epoch_index": 0, "epoch_total": 2, "segment_name": "epoch_000"},
            "eta_start_wall": 1_700_000_000.0 - 9 * 3600.0}
    pool = {"total_ns": 6000.0, "used_ns": 1742.0, "remaining_ns": 4258.0, "events": []}
    gamd = {"joint_envelope": {"Dihedral": {"sigma0_kj_mol": 12.552, "sigmaV_kj_mol": 10.4, "k0": 0.71}}}
    return build_context(
        logger=logger, rows=rows, phase="gareus_production", step=459_000, total_steps=570_218,
        summary={}, dashboard_info=info, sidecar=SidecarSnapshot(pool=pool, gamd=gamd),
        term_w=TERM_W, term_h=TERM_H, now=1_700_000_000.0, view=view, glyphs="unicode")


# --- minimal ANSI -> image -------------------------------------------------
_BASE = {30: (40, 42, 54), 31: (255, 85, 85), 32: (80, 250, 123), 33: (241, 250, 140),
         34: (98, 114, 164), 35: (255, 121, 198), 36: (139, 233, 253), 37: (248, 248, 242)}
FG_DEFAULT, BG = (220, 223, 228), (24, 26, 33)
_SGR = re.compile(r"\x1b\[([0-9;]*)m")


def _xterm256(n: int):
    if n < 16:
        return list(_BASE.values())[n % 8]
    if n < 232:
        n -= 16
        steps = [0, 95, 135, 175, 215, 255]
        return steps[n // 36], steps[(n // 6) % 6], steps[n % 6]
    v = 8 + 10 * (n - 232)
    return v, v, v


def _cells(line: str):
    fg, bold, dim = FG_DEFAULT, False, False
    pos = 0
    for m in _SGR.finditer(line):
        for ch in line[pos:m.start()]:
            yield ch, fg, bold, dim
        codes = [int(c) for c in m.group(1).split(";") if c] or [0]
        i = 0
        while i < len(codes):
            c = codes[i]
            if c == 0:
                fg, bold, dim = FG_DEFAULT, False, False
            elif c == 1:
                bold = True
            elif c == 2:
                dim = True
            elif c in _BASE:
                fg = _BASE[c]
            elif c == 38 and i + 2 < len(codes) and codes[i + 1] == 5:
                fg = _xterm256(codes[i + 2])
                i += 2
            i += 1
        pos = m.end()
    for ch in line[pos:]:
        yield ch, fg, bold, dim


def draw(text: str, out: pathlib.Path, title: str):
    text = re.sub(r"\x1b\[(\?25[hl]|[0-9]*[JKH])", "", text)
    lines = text.splitlines()
    reg = ImageFont.truetype("/usr/share/fonts/TTF/MesloLGS-NF-Regular.ttf", 15)
    bold_f = ImageFont.truetype("/usr/share/fonts/TTF/MesloLGS-NF-Bold.ttf", 15)
    cw = int(round(reg.getlength("M")))
    ch_h = 20
    pad, bar = 16, 30
    width = max(TERM_W, max(len(_SGR.sub("", l)) for l in lines)) * cw + 2 * pad
    img = Image.new("RGB", (width, len(lines) * ch_h + 2 * pad + bar), BG)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, width, bar], fill=(40, 42, 54))
    for i, c in enumerate([(255, 95, 86), (255, 189, 46), (39, 201, 63)]):
        d.ellipse([12 + 20 * i, 9, 24 + 20 * i, 21], fill=c)
    d.text((80, 6), title, font=reg, fill=(170, 170, 180))
    for r, line in enumerate(lines):
        x = pad
        y = bar + pad + r * ch_h
        for ch, fg, bold, dim in _cells(line):
            if dim:
                fg = tuple(int(v * 0.55 + BG[j] * 0.45) for j, v in enumerate(fg))
            if ch in "█▀▄":          # draw blocks as rectangles so they tile seamlessly
                top = y if ch != "▄" else y + ch_h // 2
                bot = y + ch_h if ch != "▀" else y + ch_h // 2
                d.rectangle([x, top, x + cw - 1, bot - 1], fill=fg)
            elif ch != " ":
                d.text((x, y), ch, font=bold_f if bold else reg, fill=fg)
            x += cw
    img.save(out)
    print(out, img.size)


if __name__ == "__main__":
    colors.configure_color("always")
    banner = "\n".join(banner_lines(argparse.Namespace(
        platform="CUDA", cpu_threads=2, forcefield="ff14SB", water_model="tip3p", timestep_fs=4.0,
        run_mode="hmr-gamd", cv1="contacts", cv2="residual-torsion-pc"),
        pathlib.Path("/tmp/chignolin_9"), resume=True, mbar_version="4.6.1"))
    draw(banner, HERE / "atlas_md_startup_banner.png", "gareus --resume  (startup banner)")
    for view in ("windows", "progress"):
        frame = render_screen(_context(view))
        (HERE / f"frame_{view}.ansi").write_text(frame)
        draw(frame, HERE / f"atlas_md_tui_{view}.png",
             f"ATLaS-MD dashboard -- {view.upper()} view (real chignolin_9 layout, synthetic live data)")
