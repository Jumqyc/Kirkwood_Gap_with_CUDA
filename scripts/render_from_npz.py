"""Redraws the video's frames from the per-frame .npz files, at any dpi.

The frames are stored as the two histograms behind the two panels plus the time,
so a frame can be redrawn without the epoch it came from -- which matters,
because the epoch files are deleted as the build consumes them and re-running
seventeen hours to change a resolution is not a trade anyone wants to make.

The layout, the constants and the progress bar come from render_frames, so the
two draw the same picture. The one difference is forced: the left panel here is
an imshow of the stored histogram rather than a hist2d of the particles, because
the particles are not in the file.

Usage:
    python scripts/render_from_npz.py <npz_dir> <out_dir> [--dpi 200] [--jobs 4]

    # the whole video at 4K
    python scripts/render_from_npz.py data/video/work/s1/frames out1 --dpi 200
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from kirkwood_io import resonance_axes  # noqa: E402
from render_frames import (A_EDGES, E_EDGES, ORDER, RATIO_EDGES,  # noqa: E402
                           draw_progress, MARS_AU)

# Must match ph::A_JUPITER_AU in cpp/physics.hpp. The resonance positions are
# not in the npz, so a redraw has to reconstruct them from the perturber's axis.
# A frame redrawn with the wrong value would put its labels in the wrong place
# and nothing else would look wrong.
JUPITER_A_AU = 5.2028


# The top of the density colour scale, in counts per bin. It is fixed across
# the whole video, not per frame: a scale that followed each frame's own maximum
# would make the density incomparable between frames, which is the one thing the
# video is for. The original build used this value at 6.5e6 particles; the npz
# does not carry it, so it is an argument.
VMAX = 16000.0


def draw(npz: Path, out: Path, dpi: float, vmax: float = VMAX) -> Path:
    """Redraws one frame. Args: npz in, png out, dpi, colour top. Returns: out."""
    z = np.load(npz)
    hist = z["ae_hist"]
    counts = z["count_vs_a"]
    edges = z["a_edges"]
    year = float(z["time_years"])
    step = int(z["step"])

    fig = plt.figure(figsize=(19.2, 10.8), dpi=dpi)
    left = fig.add_axes([0.045, 0.20, 0.425, 0.68])
    right = fig.add_axes([0.535, 0.20, 0.425, 0.68])

    # imshow rather than hist2d: the histogram is already counted. LogNorm over
    # the same bins reproduces what hist2d drew from the raw particles.
    left.imshow(hist.T, origin="lower", aspect="auto", cmap="magma",
                norm=LogNorm(vmin=1, vmax=vmax),
                extent=[A_EDGES[0], A_EDGES[-1], E_EDGES[0], E_EDGES[-1]],
                interpolation="nearest")
    left.set(xlim=(2.0, 3.5), ylim=(0, 0.72),
             xlabel="semimajor axis (AU)", ylabel="eccentricity")
    left.set_title("density in the (a, e) plane", fontsize=17)

    baseline = np.load(out.parent / "baseline.npy") if (out.parent / "baseline.npy").exists() else None
    ratio = counts / np.maximum(baseline, 1.0) if baseline is not None else counts
    centres = 0.5 * (edges[1:] + edges[:-1])
    right.axhline(1.0, color="0.75", lw=0.8)
    right.plot(centres, ratio, color="C3", lw=1.0)
    right.set(xlim=(2.0, 3.5), ylim=(0.35, 2.2),
              xlabel="semimajor axis (AU)", ylabel="count / value at t = 0")
    right.set_title("the same thing as a ratio, which shows the pile-up",
                    fontsize=17)

    resonances = resonance_axes(JUPITER_A_AU)
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

    fig.suptitle(f"t = {year:,.0f} yr", fontsize=26, y=0.945)
    bar = fig.add_axes([0.25, 0.035, 0.50, 0.085])
    draw_progress(bar, year)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


def _one(args: tuple[Path, Path, float, float]) -> str:
    npz, out_dir, dpi, vmax = args
    return draw(npz, out_dir / npz.with_suffix(".png").name, dpi, vmax).name


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("npz_dir", type=Path)
    p.add_argument("out_dir", type=Path)
    p.add_argument("--vmax", type=float, default=VMAX,
                   help="top of the density colour scale, in counts per bin; "
                        "must match the build's so frames stay comparable")
    p.add_argument("--dpi", type=float, default=200.0,
                   help="200 gives 3840x2160 from a 19.2x10.8 inch canvas")
    p.add_argument("--shard", default=None, metavar="I/N",
                   help="render only every Nth frame starting at I, for "
                        "driving several processes from xargs")
    p.add_argument("--baseline", type=Path, default=None,
                   help="baseline.npy for the ratio panel; copied into out_dir")
    args = p.parse_args()

    # The ratio panel divides by t = 0, and the redraw needs it beside the
    # output rather than beside the dumps, which may be long gone.
    if args.baseline is not None:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        np.save(args.out_dir / "baseline.npy", np.load(args.baseline))

    files = sorted(args.npz_dir.glob("frame_*.npz"))
    if not files:
        print(f"no frame_*.npz under {args.npz_dir}", file=sys.stderr)
        return 1
    if args.shard is not None:
        i, n = (int(v) for v in args.shard.split("/"))
        files = files[i::n]

    # Serial on purpose. A process pool would be the obvious way to use the
    # cores, but rendering is CPU-bound Python and matplotlib, so threads buy
    # little; and in this environment ProcessPoolExecutor cannot even start,
    # because creating its semaphore is denied. Parallelism comes from running
    # several of these with xargs, which is separate processes and needs none.
    print(f"{len(files)} frames at dpi {args.dpi:.0f}", flush=True)
    for n, f in enumerate(files, 1):
        _one((f, args.out_dir, args.dpi, args.vmax))
        if n % 50 == 0 or n == len(files):
            print(f"  {n}/{len(files)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
