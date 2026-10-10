"""Read-only replay of the aux discovery hook on an existing campaign."""
from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path
from typing import Optional

from .settings import AuxDiscoverySettings

DEFAULT_EXCHANGE_INTERVAL = 3000


def _recorded_exchange_interval(run_dir: Path) -> Optional[int]:
    """exchange_interval from the run's recorded args (run_manifest.json resolved_args / method_settings,
    else run_args.json); None when no record carries it."""
    for name in ("run_manifest.json", "run_args.json"):
        path = run_dir / name
        if not path.is_file():
            continue
        try:
            raw = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        for block in (raw.get("resolved_args"), raw.get("method_settings"), raw):
            if isinstance(block, dict) and block.get("exchange_interval"):
                return int(block["exchange_interval"])
    return None


def _replay(*, run_dir: Path, epoch: int, out: Path, max_frames: int, workers: int, k3_max,
            stride_steps: Optional[int] = None):
    from gareus.adaptive_production import WindowStateRegistry
    from gareus.adaptive.aux_discovery.frames import build_frame_table
    from gareus.adaptive.aux_discovery.pipeline import run_discovery
    from openmm import app

    ad = run_dir / "adaptive_production"
    reg = WindowStateRegistry.load(ad)
    s = AuxDiscoverySettings(max_frames=int(max_frames))
    exch = stride_steps or _recorded_exchange_interval(run_dir) or DEFAULT_EXCHANGE_INTERVAL
    ft = build_frame_table(ad, epochs=range(0, epoch + 1),
                           registry_lambda={int(x.state_id): float(x.gamd_lambda) for x in reg.all_states()},
                           stride_steps=int(exch), max_frames=s.max_frames, seed=s.partition_seed, workers=workers)
    top = app.PDBFile(str(run_dir / "01_solvated_start.pdb")).topology
    res = run_discovery(ft, train=ft.epoch < epoch, holdout=ft.epoch == epoch, settings=s, full_topology=top,
                        k3_max=k3_max, epoch=epoch)
    if res.model is not None:
        res.model.write(out / "aux_model.json")
    if res.placement is not None:
        (out / "placement.json").write_text(json.dumps(res.placement, indent=1, default=str))
    return {"status": res.status, **res.report}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m gareus.adaptive.aux_discovery")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("replay", help="run discovery on epochs <= N of an existing run, writing only to --out")
    r.add_argument("run_dir")
    r.add_argument("--epoch", type=int, required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--max-frames", type=int, default=AuxDiscoverySettings().max_frames)
    r.add_argument("--workers", type=int, default=1)
    r.add_argument("--k3-max", type=float, default=None)
    r.add_argument("--stride-steps", type=int, default=None,
                   help="frame stride in steps (default: the run's recorded exchange_interval)")
    a = p.parse_args(argv)
    run_dir, out = Path(a.run_dir).resolve(), Path(a.out).resolve()
    if out == run_dir or run_dir in out.parents:
        p.error("--out must be outside the run directory (replay is read-only)")
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rep = _replay(run_dir=run_dir, epoch=a.epoch, out=out, max_frames=a.max_frames, workers=a.workers,
                  k3_max=a.k3_max, stride_steps=a.stride_steps)
    (out / "aux_discovery_report.json").write_text(json.dumps(rep, indent=1, default=str))
    (out / "timing.json").write_text(json.dumps({
        "wall_s": time.time() - t0,
        "peak_rss_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
