"""The invariant the old dashboard violated: a frame must fit its terminal.

Measured before this rework, `_render_dashboard` emitted 81-97 lines for a 44-54
line terminal (see tests/manual_render_dashboard_frame.py). Nothing asserted the
fit, so the loss was invisible.
"""

import argparse
import os
import random
import re

import pytest

from gareus.dashboard.context import build_context
from gareus.dashboard.screen import render_screen
from gareus.dashboard.sidecar import SidecarSnapshot
from gareus.logger import DistanceLogger
from gareus.tui import strip_ansi, strip_ansi_len

# 40 and 55 are deliberate: below ~62 columns a two-panel row can no longer sit
# side by side and takes the vertical stacking path instead. A matrix that starts
# at 70 never exercises stacking at all, which is where both of Task 2's verified
# width/height defects lived.
WIDTHS = (40, 55, 70, 80, 100, 120, 140, 200, 400)
HEIGHTS = (18, 20, 24, 35, 45, 55, 80)
VIEWS = ("progress", "physics", "windows")
N_WINDOWS = 25

# The footer's wall-clock (`gareus/dashboard/screen.py`'s
# `time.strftime("%H:%M:%S", time.localtime(ctx.now_wall))`) is deliberately
# local-time for the real live dashboard -- but that makes it the one cell in
# an otherwise-fully-deterministic frame that depends on the host machine's
# timezone, not on any of this test's own inputs. Confirmed real: the golden
# files generated on this machine (UTC+1) read "23:13:20"; regenerating under
# `TZ=UTC` gives "22:13:20" for the identical `now=1_700_000_000.0` input.
# Normalized out before writing/comparing so the snapshot is portable across
# machines/CI regardless of timezone, without changing the production footer
# (which should stay local-time -- that's a display choice, not a bug).
_CLOCK_RE = re.compile(r"\b\d{2}:\d{2}:\d{2}\b")


def _normalize_clock(text: str) -> str:
    return _CLOCK_RE.sub("00:00:00", text)


POOL = {"total_ns": 15000.0, "used_ns": 7500.0, "remaining_ns": 7500.0,
        "events": [{"label": f"seg_{i}", "kind": "scheduled_epoch",
                    "consumed_ns": 100.0 * (i + 1), "n_states": 29} for i in range(12)]}
GAMD = {"joint_envelope": {"Dihedral": {"sigma0_kj_mol": 12.552,
                                        "sigmaV_kj_mol": 11.039, "k0": 1.0}}}


def _ctx(tmp_path, term_w, term_h, view, *, is_2d=False, n=N_WINDOWS, rich=True):
    # Fresh RNG per call (not a module-level advancing generator): a shared
    # module-level `random.Random` would make the noise stream -- and thus the
    # golden snapshot content -- depend on how many other tests/parametrize
    # cases already called `_ctx()` in this process before it, i.e. on test
    # execution order/selection rather than on this call's own inputs alone.
    # Confirmed real: `pytest -k golden` alone vs. the full file produced two
    # different `cv1` noise realizations for the identical golden case.
    rng = random.Random(17)
    centers = [4.0 + 0.55 * i for i in range(n)]
    logger = DistanceLogger(tmp_path, argparse.Namespace(timestep_fs=4.0),
                            no_file_persistence=True)
    if rich:
        for w in range(n):
            # Spread at roughly the restraint's own width (sigma = 0.488 A for k=2.5 at
            # 300 K, windows 0.55 A apart) so neighbours genuinely overlap. A narrow
            # ramp gives overlap 0.0 everywhere and ranks every window BAD, which would
            # make the golden frames a picture of a broken run by accident rather than
            # by choice. Seeded, so the frames stay reproducible.
            logger.history_by_window[w] = [centers[w] + rng.gauss(0.0, 0.35)
                                          for _ in range(60)]
            logger.potential_history_by_replica[w] = [-31500.0 + i for i in range(40)]
            logger.secondary_history_by_window[w] = [0.1 * (i % 7) - 0.3 for i in range(40)]
        logger.boost_history_all = [2.0 + 0.5 * (i % 7) for i in range(300)]
    rows = [{"replica": w, "window": w, "center_A": centers[w], "k_kcal_mol_A2": 2.5,
             "cv_A": centers[w] + 0.05, "umbrella_bias_kcal_mol": 0.01,
             "umbrella_pull_kcal_mol_A": 0.1, "potential_kj_mol": -31500.0,
             "gamd_boost_total_kcal_mol": 2.4} for w in range(n)]
    # Nested "pairs" shape, as gareus/production.py actually writes it, and with
    # ONE deliberate dead pair so the golden frames show a flagged state too -- a
    # reference frame in which nothing is ever wrong cannot show whether the alarm
    # path renders at all.
    pair_stats = {f"{i}-{i+1}": {"attempts": 40, "accepted": (0 if i == 17 else 12)}
                  for i in range(n - 1)}
    info = {"centers_a": centers, "n_windows": n, "k_list": [2.5] * n,
            "exchange_stats": {"attempts": 40 * (n - 1), "accepted": 12 * (n - 2),
                               "mode": "neighbor", "pairs": pair_stats},
            "primary_cv_label": "nonlocal contacts", "primary_cv_units": "A",
            "primary_k_units": "kcal/mol/A^2",
            "adaptive_phase": {"epoch_index": 1, "epoch_total": 2,
                               "segment_name": "epoch_001/topup_002"},
            "eta_start_wall": 1_700_000_000.0 - 6 * 3600.0}
    if is_2d:
        targets = [-2.0, -1.0, 0.0, 1.0, 2.0]
        info["secondary_cv_centers"] = [targets[i % len(targets)] for i in range(n)]
        info["secondary_cv"] = {"explicit_2d_windows": True, "grid": False,
                                "type": "tica-linear"}
    return build_context(
        logger=logger, rows=rows, phase="gareus_production", step=12_450_000,
        total_steps=37_500_000, summary={}, dashboard_info=info,
        sidecar=SidecarSnapshot(pool=POOL, gamd=GAMD), term_w=term_w, term_h=term_h,
        now=1_700_000_000.0, view=view, glyphs="unicode",
    )


@pytest.mark.parametrize("term_w", WIDTHS)
@pytest.mark.parametrize("term_h", HEIGHTS)
@pytest.mark.parametrize("view", VIEWS)
def test_frame_fits_every_terminal_size_and_view(tmp_path, term_w, term_h, view):
    lines = strip_ansi(render_screen(_ctx(tmp_path, term_w, term_h, view))).splitlines()
    assert len(lines) <= term_h - 1, f"{len(lines)} lines in a {term_h}-row terminal"
    for line in lines:
        assert strip_ansi_len(line) <= term_w - 2, f"line wider than {term_w - 2}: {line!r}"


@pytest.mark.parametrize("view", VIEWS)
def test_frame_fits_for_a_2d_run(tmp_path, view):
    lines = strip_ansi(render_screen(_ctx(tmp_path, 140, 45, view, is_2d=True))).splitlines()
    assert len(lines) <= 44
    assert all(strip_ansi_len(l) <= 138 for l in lines)


@pytest.mark.parametrize("view", VIEWS)
def test_frame_fits_with_364_windows(tmp_path, view):
    lines = strip_ansi(render_screen(_ctx(tmp_path, 140, 45, view, n=364))).splitlines()
    assert len(lines) <= 44
    assert all(strip_ansi_len(l) <= 138 for l in lines)


@pytest.mark.parametrize("view", VIEWS)
def test_frame_renders_before_any_samples_exist(tmp_path, view):
    frame = render_screen(_ctx(tmp_path, 140, 45, view, rich=False))
    assert frame.strip()


def test_every_status_word_survives_ansi_stripping(tmp_path):
    """Colour must never be the only carrier of severity."""
    text = strip_ansi(render_screen(_ctx(tmp_path, 200, 55, "physics")))
    assert "SATURATED" in text          # k0 at its ceiling
    assert "ok" in text or "OK" in text


@pytest.mark.parametrize("view", VIEWS)
def test_golden_frame_matches_the_committed_snapshot(view):
    """Layout drift must be a deliberate, reviewed diff.

    Uses a fixed run directory name, not `tempfile.TemporaryDirectory()` --
    `build_context` renders a run identity verbatim into the spine's identity
    line (`gareus/dashboard/context.py`'s `run_label` ->
    `gareus/dashboard/spine.py`'s identity line). `run_label` prefers
    `sidecar.run_root.name` and falls back to `logger.out_dir.name` only when
    no run root was resolved (I5) -- this fixture passes a hand-built
    `SidecarSnapshot()` with no `run_root`, so it still takes the `out_dir.name`
    fallback path, but a randomly-suffixed tmp dir would still bake a
    non-reproducible token straight into the golden text either way, and the
    snapshot could never match on a second run. Nothing needs to exist on disk
    here (`no_file_persistence=True`, and `sidecar` is passed in directly), so
    a fixed path that is never created is sufficient and avoids that trap.
    """
    import pathlib
    import tempfile
    golden = pathlib.Path(__file__).parent / "golden" / "dashboard" / f"{view}_140x45.txt"
    run_dir = pathlib.Path(tempfile.gettempdir()) / "gareus_dashboard_golden_fixture"
    actual = _normalize_clock(strip_ansi(render_screen(_ctx(run_dir, 140, 45, view))))
    if os.environ.get("GAREUS_UPDATE_GOLDEN") == "1":
        golden.parent.mkdir(parents=True, exist_ok=True)
        golden.write_text(actual)
    assert golden.is_file(), f"missing golden file; regenerate with GAREUS_UPDATE_GOLDEN=1"
    assert actual == golden.read_text()
