"""CLI for the synthetic adaptive-window harness.

Run a synthetic adaptive campaign on an analytic landscape and print metrics::

    python -m gareus.synth --landscape mixture-wells --mode exact \
        --subsystem feedback --rounds 5 --seed 0 [--plot OUTDIR]
"""
from __future__ import annotations

import argparse
import contextlib
import json
import sys

from .drivers import (feedback_campaign, production_campaign, sample_final_layout)
from .landscapes import LANDSCAPES
from .metrics import campaign_metrics


def _default_init(landscape):
    lo1, hi1 = landscape.cv1_bounds
    c1 = [lo1 + (hi1 - lo1) * f for f in (0.2, 0.5, 0.8)]
    return dict(initial_centers1=c1, initial_k1=[40.0] * 3,
                initial_centers2=[-0.5, 0.5], initial_k2=[40.0, 40.0])


def _production_summary(landscape, recs) -> dict:
    total_retired = sum(len(r.retired_this_epoch) for r in recs)
    return {
        "subsystem": "production",
        "epochs": len(recs),
        "final_n_active": recs[-1].n_active if recs else 0,
        "total_retired": int(total_retired),
        "retired_per_epoch": [len(r.retired_this_epoch) for r in recs],
        "active_per_epoch": [r.n_active for r in recs],
    }


def run_cli(argv=None) -> dict:
    p = argparse.ArgumentParser(prog="gareus.synth",
                                description="Synthetic adaptive-window harness")
    p.add_argument("--landscape", default="mixture-wells", choices=sorted(LANDSCAPES))
    p.add_argument("--mode", default="exact", choices=["exact", "langevin"])
    p.add_argument("--subsystem", default="feedback", choices=["feedback", "production"])
    p.add_argument("--rounds", type=int, default=5,
                   help="feedback rounds or production epochs")
    p.add_argument("--samples-per-window", type=int, default=1500)
    p.add_argument("--res", type=int, default=120)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--target-overlap", type=float, default=0.30)
    p.add_argument("--aggressiveness", default="balanced")
    p.add_argument("--plot", default=None, help="directory to write PNG plots")
    a = p.parse_args(argv)
    lp = LANDSCAPES[a.landscape]

    # The real adaptive dispatcher prints diagnostics to stdout; route those to
    # stderr so this command's stdout carries only the metrics JSON (pipe-safe).
    with contextlib.redirect_stdout(sys.stderr):
        if a.subsystem == "production":
            recs = production_campaign(lp, mode=a.mode, n_epochs=a.rounds,
                                       samples_per_window=a.samples_per_window,
                                       res=a.res, seed=a.seed,
                                       target_overlap=a.target_overlap)
            m = _production_summary(lp, recs)
        else:
            recs = feedback_campaign(lp, **_default_init(lp), mode=a.mode,
                                     n_rounds=a.rounds,
                                     samples_per_window=a.samples_per_window,
                                     res=a.res, seed=a.seed,
                                     target_overlap=a.target_overlap,
                                     aggressiveness=a.aggressiveness)
            final_samples, final_windows = sample_final_layout(
                lp, recs[-1], mode=a.mode, samples_per_window=a.samples_per_window,
                res=a.res, seed=a.seed + 999)
            m = campaign_metrics(lp, recs, target=a.target_overlap, res=a.res,
                                 samples_by_window=final_samples, windows=final_windows)
            if a.plot:
                from .report import plot_campaign
                m["plots"] = [str(x) for x in plot_campaign(lp, recs, a.plot)]

    print(json.dumps(m, indent=2, default=float))
    return m


if __name__ == "__main__":
    run_cli(sys.argv[1:])
