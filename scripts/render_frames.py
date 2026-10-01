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

# The progress bar is a linear 0 - 1 Myr time axis, and the three segments of the
# video are what happens to lie on it: 0-10 kyr is the first 1 % of the bar,
# 10-100 kyr the next 9 %, and 100 kyr-1 Myr the remaining 90 %. The marker
# therefore crawls through the first segment and races through the last, which is
# the point -- the video spends equal time on each, the bar does not.
BAR_MIN_YEARS = 10.0        # the left edge; 0 cannot be put on a log axis
BAR_MAX_YEARS = 1_000_000.0
SEGMENT_EDGES = (10_000.0, 100_000.0)
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


def draw_progress(ax: plt.Axes, year: float) -> None:
    """A logarithmic 10 yr - 1 Myr time axis with the current position marked.

    Args:
        ax: axes to draw into; its limits are set here.
        year: simulated time of this frame, in years. Values at or below the
            left edge are clamped to it.

    Linear in time would put the whole first segment, 0-10 kyr, inside the first
    1 % of the bar, which is not enough to see move. On a log axis the three
    segments cover 60 %, 20 % and 20 %, so the marker is readable throughout and
    still moves at three different speeds.
    """
    def at(t: float) -> float:
        """Position of `t` years on the bar, as a fraction of its width."""
        return float(np.log10(max(t, BAR_MIN_YEARS) / BAR_MIN_YEARS)
                     / np.log10(BAR_MAX_YEARS / BAR_MIN_YEARS))

    ax.set(xlim=(0.0, 1.0), ylim=(0.0, 1.0))
    ax.axis("off")
    ax.add_patch(plt.Rectangle((0.0, 0.40), 1.0, 0.30,
                               facecolor="0.90", edgecolor="0.75", lw=0.8))
    ax.add_patch(plt.Rectangle((0.0, 0.40), at(year), 0.30,
                               facecolor="#d94a4a", edgecolor="none"))
    decade = 10.0
    while decade <= BAR_MAX_YEARS:
        ax.axvline(at(decade), ymin=0.18, ymax=0.40, color="0.6", lw=0.7)
        label = f"{decade / 1000:.0f} kyr" if decade >= 1000 else f"{decade:.0f} yr"
        ax.text(at(decade), 0.10, label, ha="center", va="top",
                fontsize=9, color="0.4")
        decade *= 10.0
    # Where the step size changes. Without these the bar looks like one run.
    for edge in SEGMENT_EDGES:
        ax.axvline(at(edge), ymin=0.30, ymax=0.78, color="0.35", lw=1.0, ls="--")
    ax.plot([at(year)], [0.55], marker="v", color="#7a1010", ms=11)


def render(path: Path, frames: Path, resonances: dict[str, float],
           baseline: np.ndarray) -> Path:
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
    draw_progress(bar, epoch.time_years)

    out = frames / f"frame_{epoch.step:09d}.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dumps", type=Path)
    parser.add_argument("frames", type=Path)
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

    for n, path in enumerate(todo, 1):
        out = render(path, args.frames, resonances, baseline)
        if n % 25 == 0 or n == len(todo):
            print(f"  {n}/{len(todo)}  {out.name}")

    if args.delete and len(paths) > 1:
        for path in paths[:-1]:
            path.unlink()
        print(f"deleted {len(paths) - 1} epochs, kept {paths[-1].name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
