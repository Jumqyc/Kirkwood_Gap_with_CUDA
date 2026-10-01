"""Draws the (a, e) evolution of one run as a GIF.

The figures in analysis.ipynb show four epochs each. The interesting claim about
this data is that the structure grows fast and then stops, which is easier to
believe from an animation than from four panels.

Usage:
    python scripts/ae_evolution.py data/sweep/ecc       docs/ae_ecc.gif
    python scripts/ae_evolution.py data/sweep/ecc_dt2   docs/ae_ecc_dt2.gif

The frames are sampled densely at the start, where the structure forms, and
thinly afterwards, where it does not move.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import LogNorm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

from kirkwood_io import elements, read_epoch, resonance_axes, semimajor_axis

MARS_AU = 1.524
A_EDGES = np.arange(2.0, 3.51, 0.005)
E_EDGES = np.arange(0.0, 0.72, 0.005)
DENSE_FRAMES = 12
STRIDE_AFTER = 3


def epoch_paths(run_dir: Path) -> list[Path]:
    """Every epoch file in `run_dir`, ordered by step."""
    return sorted(run_dir.glob("epoch_*.bin"),
                  key=lambda p: int(p.stem.split("_")[1]))


def draw(ax: plt.Axes, path: Path, resonances: dict[str, float]) -> None:
    """One frame: the (a, e) density of a single stored snapshot."""
    epoch = read_epoch(path)
    a, e = elements(epoch)
    ax.hist2d(a, e, bins=[A_EDGES, E_EDGES],
              norm=LogNorm(vmin=1, vmax=200), cmap="magma")
    for label, axis in resonances.items():
        strong = label == "2:1"
        ax.axvline(axis, color="cyan" if strong else "0.75",
                   lw=1.0 if strong else 0.6, ls=":")
        ax.text(axis, 0.705, label, rotation=90, fontsize=7,
                color="cyan" if strong else "0.85", ha="right", va="top")
    grid = np.linspace(2.0, 3.5, 200)
    ax.plot(grid, 1.0 - MARS_AU / grid, color="lime", lw=1.2)
    ax.set(xlim=(2.0, 3.5), ylim=(0, 0.72),
           xlabel="semimajor axis (AU)", ylabel="eccentricity",
           title=f"{epoch.dt_days:.0f}-day steps   t = {epoch.time_years / 1000:.1f} kyr")
    ax.legend(handles=[
        plt.Line2D([], [], color="lime", lw=1.2, label="Mars crossing"),
        plt.Line2D([], [], color="cyan", lw=1.0, ls=":", label="2:1"),
        plt.Line2D([], [], color="0.75", lw=0.6, ls=":", label="other resonances"),
    ], fontsize=7, frameon=False, loc="upper left")


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__.strip(), file=sys.stderr)
        return 1

    run_dir, out_path = Path(sys.argv[1]), Path(sys.argv[2])
    paths = epoch_paths(run_dir)
    if not paths:
        print(f"no epoch files under {run_dir}", file=sys.stderr)
        return 1

    first = read_epoch(paths[0])
    resonances = resonance_axes(
        semimajor_axis(first.planet_r[0], first.planet_v[0], first.G))

    picks = list(range(min(DENSE_FRAMES, len(paths))))
    picks += list(range(DENSE_FRAMES, len(paths), STRIDE_AFTER))
    print(f"{run_dir}: {len(paths)} epochs -> {len(picks)} frames")

    fig, ax = plt.subplots(figsize=(7.2, 4.4))

    def frame(i: int) -> None:
        ax.clear()
        draw(ax, paths[picks[i]], resonances)

    animation = FuncAnimation(fig, frame, frames=len(picks), interval=140)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    animation.save(out_path, writer=PillowWriter(fps=8), dpi=95)
    plt.close(fig)
    print(f"wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
