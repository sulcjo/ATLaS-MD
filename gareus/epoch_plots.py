from __future__ import annotations

import json
from pathlib import Path

import numpy as np

try:
    import pandas as pd
except ImportError:
    raise ImportError("pandas required for epoch plots")


def _load_epoch_data(adaptive_dir: Path) -> list[dict]:
    """Load per-epoch arrays. Returns list of dicts with keys: epoch_idx, cv_A, secondary_cv, window, n_samples"""
    def _epoch_sort_key(d: Path) -> int:
        try:
            return int(d.name.split("_")[-1])
        except (ValueError, IndexError):
            return -1

    epoch_dirs = sorted(
        [d for d in adaptive_dir.iterdir() if d.is_dir() and d.name.startswith("epoch_")],
        key=_epoch_sort_key,
    )
    if not epoch_dirs:
        raise ValueError(f"No epoch dirs found in {adaptive_dir}")

    epochs = []
    for epoch_dir in epoch_dirs:
        npz_path = epoch_dir / "analysis_arrays.npz"
        if not npz_path.exists():
            continue
        data = np.load(str(npz_path))
        try:
            epoch_idx = int(epoch_dir.name.split("_")[-1])
        except (ValueError, IndexError):
            continue
        cv_A = data["cv_A"].astype(np.float64)
        secondary_cv = data["secondary_cv"].astype(np.float64)
        window = data["window"].astype(np.int32)
        epochs.append(
            {
                "epoch_idx": epoch_idx,
                "cv_A": cv_A,
                "secondary_cv": secondary_cv,
                "window": window,
                "n_samples": len(cv_A),
            }
        )
    return epochs


def _load_window_data(adaptive_dir: Path) -> dict[int, pd.DataFrame]:
    """Load per-epoch window center DataFrames (epoch 1+). Returns {epoch_idx: df}."""
    result: dict[int, pd.DataFrame] = {}
    for csv_path in sorted(adaptive_dir.glob("windows_epoch_*.csv")):
        stem = csv_path.stem  # e.g. "windows_epoch_001"
        epoch_idx = int(stem.split("_")[-1])
        result[epoch_idx] = pd.read_csv(str(csv_path))
    return result


def _read_cv_labels(run_dir: Path) -> tuple[str, str]:
    """Return (cv1_label, cv2_label) from run_manifest.json or defaults."""
    manifest_path = run_dir / "run_manifest.json"
    cv1, cv2 = None, None
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
        resolved = manifest.get("resolved_args", {})
        cv1 = resolved.get("cv1")
        cv2 = resolved.get("cv2")

    label_map_cv1 = {
        "contacts": "Contact fraction CV1",
        "distance": "Distance (Å) CV1",
    }
    label_map_cv2 = {
        "rama-map": "Ramachandran CV2",
    }
    return (
        label_map_cv1.get(cv1, "CV1") if cv1 else "CV1",
        label_map_cv2.get(cv2, "CV2") if cv2 else "CV2",
    )


def _make_grid_plot(
    epochs: list[dict],
    window_data: dict[int, pd.DataFrame],
    cv1_label: str,
    cv2_label: str,
    is_2d: bool,
    out_path: Path,
    dpi: int,
    show: bool,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib required for epoch plots")

    n_epochs = len(epochs)
    n_rows = 2 if is_2d else 1

    all_cv_A = np.concatenate([e["cv_A"] for e in epochs])
    valid_cv_A = all_cv_A[~np.isnan(all_cv_A)]
    if valid_cv_A.size == 0:
        raise ValueError("All cv_A samples are NaN across all epochs — cannot plot")
    cv_A_min, cv_A_max = float(valid_cv_A.min()), float(valid_cv_A.max())

    if is_2d:
        all_sec = np.concatenate([e["secondary_cv"] for e in epochs])
        valid_sec = all_sec[~np.isnan(all_sec)]
        sec_min = float(np.nanmin(valid_sec)) if len(valid_sec) else -1.0
        sec_max = float(np.nanmax(valid_sec)) if len(valid_sec) else 1.0

    fig, axes = plt.subplots(n_rows, n_epochs, figsize=(4 * n_epochs, 4 * n_rows), squeeze=False)

    for col, epoch in enumerate(epochs):
        epoch_idx = epoch["epoch_idx"]
        cv_A = epoch["cv_A"]
        secondary_cv = epoch["secondary_cv"]
        n = epoch["n_samples"]
        title = f"Epoch {epoch_idx} ({n:,} samples)"

        if is_2d:
            ax = axes[0][col]
            valid_mask = ~np.isnan(secondary_cv)
            ax.hexbin(
                cv_A[valid_mask],
                secondary_cv[valid_mask],
                gridsize=30,
                cmap="hot_r",
                mincnt=1,
                extent=[cv_A_min, cv_A_max, sec_min, sec_max],
            )
            ax.set_xlim(cv_A_min, cv_A_max)
            ax.set_ylim(sec_min, sec_max)
            ax.set_title(title)
            ax.set_xlabel(cv1_label)
            ax.set_ylabel(cv2_label)

            if epoch_idx in window_data:
                wdf = window_data[epoch_idx]
                ax.scatter(
                    wdf["primary_cv_center"],
                    wdf["secondary_cv_center"],
                    color="white",
                    edgecolors="black",
                    linewidths=0.5,
                    s=30,
                    zorder=5,
                )

            ax2 = axes[1][col]
            ax2.hist(cv_A, bins=40, range=(cv_A_min, cv_A_max), color="steelblue", alpha=0.8)
            ax2.set_xlim(cv_A_min, cv_A_max)
            ax2.set_xlabel(cv1_label)
            ax2.set_ylabel("Count")
            ax2.set_title(f"Epoch {epoch_idx} histogram")
        else:
            ax = axes[0][col]
            ax.hist(cv_A, bins=40, range=(cv_A_min, cv_A_max), color="steelblue", alpha=0.8)
            ax.set_xlim(cv_A_min, cv_A_max)
            ax.set_title(title)
            ax.set_xlabel(cv1_label)
            ax.set_ylabel("Count")

    fig.tight_layout()
    if show:
        plt.show()
    fig.savefig(str(out_path), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _make_cumulative_plot(
    epochs: list[dict],
    cv1_label: str,
    cv2_label: str,
    is_2d: bool,
    out_path: Path,
    dpi: int,
    show: bool,
) -> None:
    try:
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
    except ImportError:
        raise ImportError("matplotlib required for epoch plots")

    n_epochs = len(epochs)
    cmap = plt.colormaps["tab10" if n_epochs <= 10 else "viridis"]
    colors = [cmap(i / max(n_epochs - 1, 1)) for i in range(n_epochs)]

    rng = np.random.default_rng(0)

    fig, ax = plt.subplots(figsize=(7, 5))

    for i, epoch in enumerate(epochs):
        epoch_idx = epoch["epoch_idx"]
        cv_A = epoch["cv_A"]
        secondary_cv = epoch["secondary_cv"]
        color = colors[i]
        label = f"Epoch {epoch_idx}"

        if is_2d:
            valid_mask = ~np.isnan(secondary_cv)
            x = cv_A[valid_mask]
            y = secondary_cv[valid_mask]
            n = len(x)
            if n > 5000:
                idx = rng.choice(n, 5000, replace=False)
                x, y = x[idx], y[idx]
            ax.scatter(x, y, c=[color], alpha=0.3, s=1, label=label, rasterized=True)
        else:
            ax.hist(cv_A, bins=50, alpha=0.5, color=color, label=label)

    ax.set_xlabel(cv1_label)
    if is_2d:
        ax.set_ylabel(cv2_label)
    else:
        ax.set_ylabel("Count")
    ax.legend(loc="best", markerscale=5 if is_2d else 1)
    ax.set_title("CV exploration — all epochs")

    fig.tight_layout()
    if show:
        plt.show()
    fig.savefig(str(out_path), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _make_window_evolution_plot(
    window_data: dict[int, pd.DataFrame],
    cv1_label: str,
    cv2_label: str,
    is_2d: bool,
    out_path: Path,
    dpi: int,
    show: bool,
) -> None:
    try:
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
    except ImportError:
        raise ImportError("matplotlib required for epoch plots")

    if not window_data:
        return

    epoch_indices = sorted(window_data.keys())
    n_epochs = len(epoch_indices)
    cmap = plt.colormaps["tab10" if n_epochs <= 10 else "viridis"]
    colors = {eidx: cmap(i / max(n_epochs - 1, 1)) for i, eidx in enumerate(epoch_indices)}

    fig, ax = plt.subplots(figsize=(7, 5))

    if is_2d:
        for eidx in epoch_indices:
            wdf = window_data[eidx]
            ax.scatter(
                wdf["primary_cv_center"],
                wdf["secondary_cv_center"],
                c=[colors[eidx]],
                s=40,
                label=f"Epoch {eidx}",
                edgecolors="black",
                linewidths=0.3,
                zorder=3,
            )
        ax.set_xlabel(cv1_label)
        ax.set_ylabel(cv2_label)
    else:
        for eidx in epoch_indices:
            wdf = window_data[eidx]
            ax.hist(
                wdf["primary_cv_center"],
                bins=20,
                alpha=0.5,
                color=colors[eidx],
                label=f"Epoch {eidx}",
            )
        ax.set_xlabel(cv1_label)
        ax.set_ylabel("Count")

    ax.legend(loc="best")
    ax.set_title("Window center evolution across epochs")

    fig.tight_layout()
    if show:
        plt.show()
    fig.savefig(str(out_path), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_epoch_cv_exploration(
    run_dir: Path | str,
    out_dir: Path | str | None = None,
    dpi: int = 150,
    show: bool = False,
) -> list[Path]:
    """Generate epoch CV exploration plots for an adaptive-production run.

    Returns list of generated PNG paths.
    """
    try:
        import matplotlib.pyplot as plt  # noqa: F401
    except ImportError:
        raise ImportError("matplotlib required for epoch plots")

    run_dir = Path(run_dir)
    adaptive_dir = run_dir / "adaptive_production"

    if out_dir is None:
        out_dir = run_dir / "epoch_plots"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    epochs = _load_epoch_data(adaptive_dir)
    if not epochs:
        raise ValueError(f"No epoch dirs found in {adaptive_dir}")

    window_data = _load_window_data(adaptive_dir)
    cv1_label, cv2_label = _read_cv_labels(run_dir)

    is_2d = not all(np.isnan(e["secondary_cv"]).all() for e in epochs)

    generated: list[Path] = []

    grid_path = out_dir / "epoch_cv_grid.png"
    _make_grid_plot(epochs, window_data, cv1_label, cv2_label, is_2d, grid_path, dpi, show)
    generated.append(grid_path)

    cumulative_path = out_dir / "epoch_cv_cumulative.png"
    _make_cumulative_plot(epochs, cv1_label, cv2_label, is_2d, cumulative_path, dpi, show)
    generated.append(cumulative_path)

    window_evo_path = out_dir / "epoch_window_evolution.png"
    _make_window_evolution_plot(window_data, cv1_label, cv2_label, is_2d, window_evo_path, dpi, show)
    generated.append(window_evo_path)

    return generated


def cli_main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Generate epoch CV exploration plots for a GAREUS run.")
    parser.add_argument("--run-dir", required=True, type=Path, help="Path to the GAREUS run directory")
    parser.add_argument("--out-dir", type=Path, default=None, help="Output directory (default: {run_dir}/epoch_plots/)")
    parser.add_argument("--dpi", type=int, default=150, help="Plot DPI (default: 150)")
    args = parser.parse_args()

    plots = plot_epoch_cv_exploration(args.run_dir, out_dir=args.out_dir, dpi=args.dpi)
    for p in plots:
        print(p)


if __name__ == "__main__":
    cli_main()
