"""T2: per-epoch union-MBAR diagnostics memory vs number of states.

Runs ``gareus.synth.union_solve_bench`` in a FRESH subprocess per (states, rows) point
(``ru_maxrss`` is the process-lifetime peak) and compares the measured peak RSS with
``estimate_union_diagnostics_peak_gb`` (0.9 GB fixed + 8 B x 7.7 per cell, the
guard behind ``--ap-topup-diagnostics-max-gb``). States = centres x 4 rungs. Also reports,
per state count, the largest kept row count the 8 GB default guard admits.

``python -m gareus.synth.t2_memory --out DIR``
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

POINTS = ((15, 100_000), (30, 100_000), (59, 100_000), (68, 100_000), (80, 100_000), (100, 100_000),
          (59, 250_000), (100, 250_000))
GUARD_GB = 8.0


def run_point(n_centres: int, n_rows: int) -> dict:
    side = max(8, int(n_centres ** 0.5) + 1)
    cmd = [sys.executable, "-m", "gareus.synth.union_solve_bench", "--n-rows", str(n_rows),
           "--n-centres", str(n_centres), "--grid-side", str(side)]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True)
    line = [ln for ln in out.stdout.splitlines() if ln.startswith("{")][-1]
    return json.loads(line)


def main(argv=None) -> int:
    from gareus.adaptive.union_diagnostics import estimate_union_diagnostics_peak_gb, max_union_rows_under_guard
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    rows = []
    for n_centres, n_rows in POINTS:
        r = run_point(n_centres, n_rows)
        est = estimate_union_diagnostics_peak_gb(r["n_rows"], r["n_states"])
        r.update(estimated_peak_gb=round(est, 2),
                 measured_over_estimate=round(r["peak_rss_gb_after_call"] / est, 2) if est else None,
                 guard_max_rows_at_8gb=max_union_rows_under_guard(GUARD_GB, r["n_states"]))
        rows.append(r)
        print(json.dumps(r), flush=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "memory.json").write_text(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
