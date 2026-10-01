"""Renders every epoch in a directory to one PNG per frame.

This is the second half of the video pipeline. It exists as its own program so
that a long run can be processed in rounds: the simulation writes a batch of
epochs, this renders them, and the driver then deletes all but the newest so the
next round can resume from it. Peak disk use is one batch rather than the whole
run.

Two panels per frame, because an MP4 costs nothing extra and the two say
different things:

  left   the (a, e) density, where a resonance shows as a spike rising in e
  right  the a distribution divided by its t = 0 value, where the same resonance
         shows as a dip and a pile-up next to each other

The right panel needs the t = 0 distribution for every frame, and the chunked
pipeline deletes the t = 0 epoch early, so it is cached to .npy on the first
pass.

Usage:
    python scripts/render_frames.py <dumps_dir> <frames_dir> [--delete]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

from kirkwood_io import elements, read_epoch, resonance_axes, semimajor_axis

MARS_AU = 1.524

# The three segments the video is cut into, as (start_year, end_year). They have
# to be on screen: the run covers a hundredfold range and a viewer has no way to
# tell 5 000 years from 500 000 without being told. The progress bar is driven by
# the global frame number, not by the clock, so each segment fills a third of it.
SEGMENTS = [(0.0, 10_000.0), (10_000.0, 100_000.0), (100_000.0, 1_000_000.0)]
SEGMENT_LABELS = ["0-10 kyr", "10-100 kyr", "100 kyr-1 Myr"]
A_EDGES = np.arange(2.0, 3.51, 0.005)
E_EDGES = np.arange(0.0, 0.72, 0.005)
RATIO_EDGES = np.arange(2.0, 3.51, 0.01)
ORDER = ["2:1", "3:1", "5:2", "7:3", "4:1"]


def epoch_paths(dumps: Path) -> list[Path]:
    """Every epoch file in `dumps`, ordered by step."""
    return sorted(dumps.glob("epoch_*.bin"),
                  key=lambda p: int(p.stem.split("_")[1]))


def load_baseline(cache: Path, dumps: Path) -> np.ndarray:
    """The a distribution at t = 0, computing and caching it on first use."""
    if cache.exists():
        return np.load(cache)
    paths = epoch_paths(dumps)
    if not paths:
        raise SystemExit(f"no epoch files under {dumps}")
    a, _ = elements(read_epoch(paths[0]))
    counts = np.histogram(a, bins=RATIO_EDGES)[0].astype(float)
    np.save(cache, counts)
    return counts


def draw_progress(ax: plt.Axes, index: int, count: int) -> None:
    """A bar along the bottom: where the video is, and what runs underneath.

    Driven by the frame number rather than by the clock. The three segments span
    10 kyr, 90 kyr and 900 kyr, so a bar that told the truth about time would put
    99 % of it in the last third and make the first two invisible.
    """
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    ax.add_patch(plt.Rectangle((0.0, 0.42), 1.0, 0.30,
                               facecolor="0.90", edgecolor="0.75", lw=0.8))
    ax.add_patch(plt.Rectangle((0.0, 0.42), (index + 1) / count, 0.30,
                               facecolor="#d94a4a", edgecolor="none"))
    for n, label in enumerate(SEGMENT_LABELS):
        left = n / len(SEGMENTS)
        if n:
            ax.axvline(left, ymin=0.42, ymax=0.72, color="0.45", lw=1.2)
        ax.text(left + 0.5 / len(SEGMENTS), 0.18, label,
                ha="center", va="top", fontsize=10, color="0.35")
    ax.plot([(index + 1) / count], [0.57], marker="v", color="#7a1010", ms=11)


def render(path: Path, frames: Path, resonances: dict[str, float],
           baseline: np.ndarray, index: int = 0, count: int = 1) -> Path:
    """Draws one epoch and writes it to `frames`. Returns the file written."""
    epoch = read_epoch(path)
    a, e = elements(epoch)

    # 1920 x 1080 at dpi 100. Explicit axes rather than subplots, because the
    # bar has to sit under both panels without stealing height from them.
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
    left = fig.add_axes([0.045, 0.20, 0.425, 0.68])
    right = fig.add_axes([0.535, 0.20, 0.425, 0.68])

    left.hist2d(a, e, bins=[A_EDGES, E_EDGES],
                norm=LogNorm(vmin=1, vmax=200), cmap="magma")
    left.set(xlim=(2.0, 3.5), ylim=(0, 0.72),
             xlabel="semimajor axis (AU)", ylabel="eccentricity")
    left.set_title("density in the (a, e) plane", fontsize=17)

    counts = np.histogram(a, bins=RATIO_EDGES)[0].astype(float)
    ratio = counts / np.maximum(baseline, 1.0)
    centres = 0.5 * (RATIO_EDGES[1:] + RATIO_EDGES[:-1])
    right.axhline(1.0, color="0.75", lw=0.8)
    right.plot(centres, ratio, color="C3", lw=1.0)
    right.set(xlim=(2.0, 3.5), ylim=(0.35, 2.2),
              xlabel="semimajor axis (AU)", ylabel="count / value at t = 0")
    right.set_title("the same thing as a ratio, which shows the pile-up",
                    fontsize=17)

    for ax in (left, right):
        ax.tick_params(labelsize=12)
        ax.xaxis.label.set_size(14)
        ax.yaxis.label.set_size(14)
        for label in ORDER:
            strong = label == "2:1"
            ax.axvline(resonances[label], color="cyan" if strong else "0.7",
                       lw=1.0 if strong else 0.6, ls=":")
            ax.text(resonances[label], ax.get_ylim()[1] * 0.99, label,
                    rotation=90, fontsize=10, ha="right", va="top",
                    color="cyan" if strong else "0.75")
    grid = np.linspace(2.0, 3.5, 200)
    left.plot(grid, 1.0 - MARS_AU / grid, color="lime", lw=1.2)

    fig.suptitle(f"t = {epoch.time_years:,.0f} yr", fontsize=26, y=0.945)
    bar = fig.add_axes([0.25, 0.035, 0.50, 0.085])
    draw_progress(bar, index, count)

    out = frames / f"frame_{epoch.step:09d}.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dumps", type=Path)
    parser.add_argument("frames", type=Path)
    parser.add_argument("--index", type=int, default=0,
                        help="this frame's number in the finished video")
    parser.add_argument("--count", type=int, default=1,
                        help="how many frames the finished video has")
    parser.add_argument("--delete", action="store_true",
                        help="remove the epochs that were rendered, keeping the "
                             "newest so the next round can resume from it")
    args = parser.parse_args()

    args.frames.mkdir(parents=True, exist_ok=True)
    baseline = load_baseline(args.frames / "baseline.npy", args.dumps)

    paths = epoch_paths(args.dumps)
    todo = [p for p in paths
            if not (args.frames / f"frame_{int(p.stem.split('_')[1]):09d}.png").exists()]
    print(f"{len(paths)} epochs, {len(todo)} to render")

    first = read_epoch(todo[0] if todo else paths[0])
    resonances = resonance_axes(
        semimajor_axis(first.planet_r[0], first.planet_v[0], first.G))

    already = len(paths) - len(todo)
    for n, path in enumerate(todo, 1):
        out = render(path, args.frames, resonances, baseline,
                     index=args.index + already + n - 1, count=args.count)
        if n % 25 == 0 or n == len(todo):
            print(f"  {n}/{len(todo)}  {out.name}")

    if args.delete and len(paths) > 1:
        for path in paths[:-1]:
            path.unlink()
        print(f"deleted {len(paths) - 1} epochs, kept {paths[-1].name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
