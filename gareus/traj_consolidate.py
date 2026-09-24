"""
Trajectory consolidation for GAREUS adaptive-production epochs.

Each epoch (epoch_001+) contains baseline/ and topup_NNN_XXXXXX/ subdirectories,
each with their own replica_trajectories/replica_NNN.{xtc,dcd}.  This module
concatenates them per-replica into epoch_NNN/replica_trajectories/replica_NNN.xtc
using MDAnalysis streaming (no full-RAM load), then optionally removes originals.

Epoch_000 uses the flat replica_trajectories/ layout and needs no consolidation.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional


# Subdirectory name patterns that contain replica trajectories.
_TRAJ_SUBDIR_RE = re.compile(r"^(baseline|topup_\d+_.+)$")


def _find_traj_dirs(epoch_dir: Path) -> list[Path]:
    """Return baseline + topup_* dirs in temporal order (by topup index)."""
    matches: list[tuple[int, Path]] = []
    for child in epoch_dir.iterdir():
        if not child.is_dir():
            continue
        if child.name == "baseline":
            matches.append((0, child))
        elif child.name.startswith("topup_"):
            parts = child.name.split("_")
            try:
                idx = int(parts[1])
            except (IndexError, ValueError):
                idx = 999
            matches.append((idx, child))
    matches.sort(key=lambda x: x[0])
    return [p for _, p in matches]


def _find_topology(epoch_dir: Path) -> Optional[Path]:
    """Find a PDB topology usable for MDAnalysis."""
    candidates = [
        epoch_dir / "shared_gamd_setup_final.pdb",
        epoch_dir / "setup" / "shared_gamd_setup_final.pdb",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    # Fall back: first pdb in baseline or epoch root
    for pdb in sorted(epoch_dir.rglob("shared_gamd_setup_final.pdb")):
        return pdb
    for pdb in sorted(epoch_dir.glob("*.pdb")):
        return pdb
    return None


def _replica_trajs(traj_dirs: list[Path], replica_id: int) -> list[Path]:
    """Collect trajectory files for one replica across all segment dirs."""
    files: list[Path] = []
    for tdir in traj_dirs:
        rep_dir = tdir / "replica_trajectories"
        if not rep_dir.exists():
            continue
        # Try XTC first, then DCD
        for ext in ("xtc", "dcd"):
            p = rep_dir / f"replica_{replica_id:03d}.{ext}"
            if p.exists():
                files.append(p)
                break
    return files


def consolidate_epoch(
    epoch_dir: Path,
    out_dir: Optional[Path] = None,
    delete_originals: bool = False,
    verbose: bool = True,
) -> Optional[Path]:
    """Consolidate per-segment replica trajectories for one epoch.

    Returns the output directory path, or None if nothing was done
    (epoch_000 flat layout or already consolidated).
    """
    try:
        import MDAnalysis as mda
    except ImportError:
        raise ImportError("MDAnalysis required for trajectory consolidation; pip install MDAnalysis")

    epoch_dir = Path(epoch_dir)

    # epoch_000 uses flat layout: replica_trajectories/ lives directly in epoch_dir
    flat_dir = epoch_dir / "replica_trajectories"
    traj_dirs = _find_traj_dirs(epoch_dir)

    if not traj_dirs:
        # Already flat or empty
        if verbose:
            print(f"  {epoch_dir.name}: no baseline/topup dirs found, skipping")
        return None

    if out_dir is None:
        out_dir = epoch_dir / "replica_trajectories"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    top_pdb = _find_topology(epoch_dir)
    if top_pdb is None:
        raise FileNotFoundError(f"No topology PDB found under {epoch_dir}")

    # Detect replica count from first traj dir
    n_replicas = 0
    for tdir in traj_dirs:
        rep_dir = tdir / "replica_trajectories"
        if rep_dir.exists():
            for f in rep_dir.iterdir():
                m = re.match(r"replica_(\d+)\.(xtc|dcd)$", f.name)
                if m:
                    n_replicas = max(n_replicas, int(m.group(1)) + 1)
            break

    if n_replicas == 0:
        if verbose:
            print(f"  {epoch_dir.name}: no replica trajectories found")
        return None

    if verbose:
        print(f"  {epoch_dir.name}: {len(traj_dirs)} segment(s), {n_replicas} replicas → {out_dir}")

    merged: list[Path] = []
    for r in range(n_replicas):
        files = _replica_trajs(traj_dirs, r)
        if not files:
            continue
        out_xtc = out_dir / f"replica_{r:03d}.xtc"
        if out_xtc.exists():
            if verbose:
                print(f"    replica_{r:03d}: already merged, skipping")
            continue

        if len(files) == 1 and files[0].suffix == ".xtc":
            # Single XTC segment — just move/copy rather than re-encode
            tmp = out_xtc.with_suffix(".xtc.tmp")
            import shutil
            shutil.copy2(files[0], tmp)
            tmp.rename(out_xtc)
        else:
            u = mda.Universe(str(top_pdb), [str(f) for f in files])
            tmp = out_xtc.with_suffix(".xtc.tmp")
            with mda.Writer(str(tmp), n_atoms=u.atoms.n_atoms) as w:
                for _ts in u.trajectory:
                    w.write(u.atoms)
            tmp.rename(out_xtc)

        merged.append(out_xtc)
        if verbose:
            size_mb = out_xtc.stat().st_size / 1e6
            print(f"    replica_{r:03d}: {len(files)} → 1 file ({size_mb:.0f} MB)")

    if delete_originals and merged:
        for tdir in traj_dirs:
            rep_dir = tdir / "replica_trajectories"
            if not rep_dir.exists():
                continue
            for f in rep_dir.iterdir():
                if re.match(r"replica_\d+\.(xtc|dcd)$", f.name):
                    f.unlink()
            try:
                rep_dir.rmdir()
            except OSError:
                pass

    return out_dir


def consolidate_run(
    run_dir: Path,
    delete_originals: bool = False,
    epochs: Optional[list[int]] = None,
    verbose: bool = True,
) -> None:
    """Consolidate all epochs in a run directory.

    Args:
        run_dir: Top-level run directory (contains adaptive_production/).
        delete_originals: Remove per-segment traj files after merging.
        epochs: Limit to specific epoch indices (e.g. [1, 2]). None = all.
        verbose: Print progress.
    """
    ap_dir = Path(run_dir) / "adaptive_production"
    if not ap_dir.exists():
        raise FileNotFoundError(f"No adaptive_production/ under {run_dir}")

    epoch_dirs = sorted(ap_dir.glob("epoch_*"))
    for ep_dir in epoch_dirs:
        m = re.match(r"epoch_(\d+)$", ep_dir.name)
        if not m:
            continue
        ep_idx = int(m.group(1))
        if epochs is not None and ep_idx not in epochs:
            continue
        if ep_idx == 0:
            if verbose:
                print(f"  epoch_000: flat layout, skipping")
            continue
        try:
            consolidate_epoch(ep_dir, delete_originals=delete_originals, verbose=verbose)
        except Exception as exc:
            print(f"  WARNING: epoch_{ep_idx:03d} consolidation failed: {exc}")


def main(argv=None) -> None:
    import argparse
    p = argparse.ArgumentParser(
        description="Consolidate per-segment ATLaS-MD replica trajectories per epoch."
    )
    from .branding import product_label
    p.add_argument("-V", "--version", action="version", version=product_label(), dest=argparse.SUPPRESS,
                   help="Print the ATLaS-MD version and exit.")
    p.add_argument("run_dir", help="Run directory (contains adaptive_production/).")
    p.add_argument("--delete-originals", action="store_true",
                   help="Remove per-segment traj files after successful merge.")
    p.add_argument("--epochs", type=int, nargs="+",
                   help="Only process these epoch indices (default: all).")
    args = p.parse_args(argv)
    consolidate_run(
        Path(args.run_dir),
        delete_originals=args.delete_originals,
        epochs=args.epochs,
        verbose=True,
    )


if __name__ == "__main__":
    main()
