"""Figures for a Kirkwood-gap run.

Reads the epoch files through ``kirkwood_io`` and draws the three views that
answer "did a gap open": where the particles ended up in the (a, e) plane, how
the semimajor-axis distribution changed, and when each part of it changed.

The module never selects a matplotlib backend and never saves anything on its
own: ``plot_run`` returns a Figure and the caller decides between ``show()``
and ``savefig()``. Picking a backend here would override whatever a notebook
or an interactive session has already set up.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt

from kirkwood_io import elements, read_run, resonance_axes, semimajor_axis


def _smooth_rows(
    counts: npt.NDArray[np.float64], window: int
) -> npt.NDArray[np.float64]:
    """Average each row over `window` adjacent bins, without biasing the ends.

    ``np.convolve(..., 'same')`` implicitly pads with zeros, which drags the
    first and last bins toward zero and fakes a gap at the edge of the disk.
    Padding with the edge value instead keeps a flat disk flat.
    """
    pad = window // 2
    kernel = np.ones(window) / window
    padded = np.pad(counts, ((0, 0), (pad, pad)), mode="edge")
    return np.apply_along_axis(lambda row: np.convolve(row, kernel, "valid"), 1, padded)


def plot_run(dump_dir: str | Path, n_bins: int = 150):
    """Draw the standard three-panel figure for one run.

    Args:
        dump_dir: Directory of ``epoch_<step>.bin`` files.
        n_bins: Histogram bins across the initial disk.
    Returns:
        The ``matplotlib.figure.Figure``. The caller decides whether to show
        it or save it, so no backend is chosen here.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    run = read_run(dump_dir)
    first, last = run[0], run[-1]
    a_0, e_0 = elements(first)
    a_f, e_f = elements(last)

    a_planet = semimajor_axis(first.planet_r[0], first.planet_v[0], first.G)
    resonances = resonance_axes(a_planet)
    # Span the disk, not the resonances: if a line lands off-screen the picture
    # is telling you the perturber sits somewhere unhelpful, which is a result.
    edges = np.linspace(a_0.min(), a_0.max(), n_bins + 1)

    fig, axes = plt.subplots(
        2, 2, figsize=(13.0, 8.0), gridspec_kw={"height_ratios": [1.0, 1.15]}
    )
    ax_ae, ax_hist = axes[0]
    ax_time = axes[1, 0]
    axes[1, 1].axis("off")

    def mark_resonances(ax, label_line: bool = False) -> None:
        """Draw the resonance positions. The names go in the side table, not on
        the axes: five rotated labels collide with the titles."""
        for axis in resonances.values():
            if edges[0] < axis < edges[-1]:
                ax.axvline(
                    axis, color="crimson", lw=0.9, alpha=0.65, zorder=0,
                    label="mean-motion resonance" if label_line else None,
                )
                label_line = False

    # Where the particles actually are, in the plane that defines a gap.
    ax_ae.scatter(a_f, e_f, s=1.0, alpha=0.18, linewidths=0, color="steelblue")
    mark_resonances(ax_ae)
    ax_ae.set(
        xlabel="semimajor axis $a$  [AU]",
        ylabel="eccentricity $e$",
        title=f"final state, $t$ = {last.time_years:,.0f} yr  ({last.n_alive:,} of "
              f"{first.n_alive:,} particles)",
        xlim=edges[0], ylim=(0, max(0.12, float(e_f.max()) * 1.05)),
    )

    # Did the population move at all, and where.
    ax_hist.hist(
        a_0, bins=edges, histtype="step", density=True,
        color="black", lw=1.2, label="initial",
    )
    ax_hist.hist(
        a_f, bins=edges, histtype="stepfilled", density=True,
        color="steelblue", alpha=0.45, lw=0, label="final",
    )
    mark_resonances(ax_hist, label_line=True)
    ax_hist.set(
        xlabel="semimajor axis $a$  [AU]",
        ylabel="probability density",
        title="distribution of $a$",
    )
    ax_hist.legend(loc="upper right", frameon=False)

    # When it moved. Row 0 is the initial disk by construction, so the colour
    # is "surviving fraction relative to the start".
    counts = np.stack(
        [
            np.histogram(elements(epoch)[0], bins=edges)[0].astype(np.float64)
            for epoch in run
        ]
    )
    ratio = _smooth_rows(counts, 5) / np.maximum(_smooth_rows(counts[:1], 5), 1e-9)
    # Always in Myr: matplotlib pushes year-scale ticks into a "1e6" offset
    # annotation that collides with the title, and a shared unit lets the
    # figures for two runs be compared by eye.
    times = np.array([epoch.time_years for epoch in run]) / 1e6
    time_edges = np.append(times, times[-1] + float(np.median(np.diff(times))))

    mesh = ax_time.pcolormesh(
        edges, time_edges, ratio, shading="flat", cmap="RdBu_r",
        norm=TwoSlopeNorm(vcenter=1.0, vmin=0.5, vmax=1.5),
    )
    for axis in resonances.values():
        if edges[0] < axis < edges[-1]:
            ax_time.axvline(axis, color="black", lw=0.9, ls="--", alpha=0.7)
    ax_time.set(
        xlabel="semimajor axis $a$  [AU]",
        ylabel="time  [Myr]",
        title="surviving fraction relative to the initial disk",
    )
    fig.colorbar(mesh, ax=ax_time, label="$N(a,t) / N(a,0)$")

    axes[1, 1].text(
        0.0, 1.0,
        "\n".join(
            [
                f"epochs          {len(run)}",
                f"dt              {first.dt_days:g} d",
                f"perturber $a$    {a_planet:.4f} AU",
                f"resonances      " + ", ".join(
                    f"{k} {v:.3f}" for k, v in sorted(resonances.items(), key=lambda kv: kv[1])
                ),
                "",
                f"$e$ max         {e_0.max():.3f} -> {e_f.max():.3f}",
                f"removed         {first.n_alive - last.n_alive:,}",
            ]
        ),
        transform=axes[1, 1].transAxes, va="top", ha="left",
        family="monospace", fontsize=9,
    )

    fig.tight_layout()
    return fig


if __name__ == "__main__":
    import argparse

    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dump_dir", help="directory of epoch_*.bin files")
    parser.add_argument("-o", "--out", help="write this file instead of showing")
    args = parser.parse_args()

    figure = plot_run(args.dump_dir)
    if args.out:
        figure.savefig(args.out, dpi=130)
        print(f"wrote {args.out}")
    else:
        plt.show()
